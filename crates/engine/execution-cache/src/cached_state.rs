//! Execution cache implementation for block processing.
use crate::{CacheCheckoutReason, TxPoolPrewarmCacheSnapshot};
use alloy_primitives::{
    map::{DefaultHashBuilder, FbBuildHasher},
    Address, StorageKey, StorageValue, B256,
};
use fixed_cache::{AnyRef, CacheConfig, Stats, StatsHandler};
use metrics::{Counter, Gauge, Histogram};
use parking_lot::Once;
use reth_errors::ProviderResult;
use reth_metrics::Metrics;
use reth_primitives_traits::{Account, Bytecode};
use reth_provider::{
    AccountReader, BlockHashReader, BytecodeReader, HashedPostStateProvider, StateProofProvider,
    StateProvider, StateRootProvider, StorageRootProvider,
};
use reth_revm::db::BundleState;
use reth_tracing::readiness::{ReadClass, ReadTimer, ReadTotals, Role};
use reth_trie::{
    updates::TrieUpdates, AccountProof, HashedPostState, HashedStorage, MultiProof,
    MultiProofTargets, StorageMultiProof, StorageProof, TrieInput,
};
use std::{
    cell::Cell,
    collections::HashMap,
    fmt,
    hash::BuildHasher,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug_span, info, instrument, trace, warn, Span};

/// Number of independently locked shards used for prewarm-read correlation.
const READINESS_KEY_SHARDS: usize = 64;
/// Maximum exact keys retained in each shard. Shard-local limits keep the hot path bounded and
/// non-blocking; skew can make the usable capacity lower than the total advertised bound.
const READINESS_KEYS_PER_SHARD: usize = 1_024;
/// Maximum number of exact state keys retained for prewarm-read correlation per block.
const READINESS_KEY_CAPACITY: usize = READINESS_KEY_SHARDS * READINESS_KEYS_PER_SHARD;
/// Maximum concurrent provider reads retained for unfinished-read timing for one exact key.
const READINESS_ACTIVE_READS_PER_KEY_CAP: usize = 64;

const PREWARM_READ_ACTIVE: u8 = 0;
const PREWARM_READ_FINISHING: u8 = 1;
const PREWARM_READ_SUCCESS: u8 = 2;
const PREWARM_READ_FAILED: u8 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ReadinessKey {
    Account(Address),
    Storage(Address, StorageKey),
    Code(B256),
}

struct PrewarmKeyState {
    inflight: AtomicUsize,
    completed: AtomicBool,
    active_reads: parking_lot::Mutex<Vec<Arc<PrewarmReadInstance>>>,
    timing_coverage_lost: AtomicBool,
}

impl Default for PrewarmKeyState {
    fn default() -> Self {
        Self {
            inflight: AtomicUsize::new(0),
            completed: AtomicBool::new(false),
            active_reads: parking_lot::Mutex::new(Vec::new()),
            timing_coverage_lost: AtomicBool::new(false),
        }
    }
}

struct PrewarmReadInstance {
    started: Instant,
    /// Published after `finished_elapsed_ns`; Acquire loads of this value observe the duration.
    outcome: std::sync::atomic::AtomicU8,
    /// Nanoseconds since `started`, saturated to `u64`, when this provider read finished.
    finished_elapsed_ns: AtomicU64,
}

#[derive(Default)]
struct PrewarmKeys {
    entries: HashMap<ReadinessKey, Arc<PrewarmKeyState>>,
}

struct PrewarmKeyShard {
    keys: parking_lot::Mutex<PrewarmKeys>,
    /// A failed begin means this shard can no longer prove that an absent or failed entry was
    /// never concurrently observed. Other shards remain independently classifiable.
    coverage_lost: AtomicBool,
    cap_reached: AtomicBool,
}

#[derive(Debug, Clone, Copy)]
enum MissPrewarmState {
    Inflight,
    /// At least one observed backing-provider read succeeded. This does not assert that its value
    /// was published to, or remains present in, the cache.
    Completed,
    Failed,
    NeverObserved,
    UnknownDueCap,
    UnknownContention,
}

#[derive(Debug, Default)]
struct MissCounters {
    inflight: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    never_observed: AtomicU64,
    unknown_due_cap: AtomicU64,
    unknown_contention: AtomicU64,
}

impl MissCounters {
    fn record(&self, state: MissPrewarmState) {
        let counter = match state {
            MissPrewarmState::Inflight => &self.inflight,
            MissPrewarmState::Completed => &self.completed,
            MissPrewarmState::Failed => &self.failed,
            MissPrewarmState::NeverObserved => &self.never_observed,
            MissPrewarmState::UnknownDueCap => &self.unknown_due_cap,
            MissPrewarmState::UnknownContention => &self.unknown_contention,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Default)]
struct LatencyCounters {
    count: AtomicU64,
    ns: AtomicU64,
    max_ns: AtomicU64,
}

impl LatencyCounters {
    fn record(&self, elapsed: Duration) {
        let ns = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.ns.fetch_add(ns, Ordering::Relaxed);
        self.max_ns.fetch_max(ns, Ordering::Relaxed);
    }
}

fn duration_ns(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u64::MAX as u128) as u64
}

#[derive(Debug, Default)]
struct StorageBackingLatency {
    // Every actual authoritative backing call is counted, including calls that return an error.
    inflight: LatencyCounters,
    completed: LatencyCounters,
    failed: LatencyCounters,
    never_observed: LatencyCounters,
    unknown: LatencyCounters,
}

#[derive(Debug, Default)]
struct StorageUnfinishedCounters {
    /// Age of the oldest captured exact-key prewarm provider read at the miss snapshot.
    age: LatencyCounters,
    /// Time after the miss snapshot until the first captured prewarm or canonical read finished.
    overlap: LatencyCounters,
    single: AtomicU64,
    multiple: AtomicU64,
    unknown: AtomicU64,
    prewarm_success_first: AtomicU64,
    prewarm_failed_first: AtomicU64,
    canonical_success_first: AtomicU64,
    canonical_failed_first: AtomicU64,
    winner_unknown: AtomicU64,
}

enum UnfinishedSnapshot {
    None,
    Unknown,
    Known(Vec<CapturedPrewarmRead>),
}

struct CapturedPrewarmRead {
    instance: Arc<PrewarmReadInstance>,
    age_ns: u64,
}

impl StorageBackingLatency {
    fn counters(&self, state: MissPrewarmState) -> &LatencyCounters {
        match state {
            MissPrewarmState::Inflight => &self.inflight,
            MissPrewarmState::Completed => &self.completed,
            MissPrewarmState::Failed => &self.failed,
            MissPrewarmState::NeverObserved => &self.never_observed,
            MissPrewarmState::UnknownDueCap | MissPrewarmState::UnknownContention => &self.unknown,
        }
    }
}

#[derive(Debug, Default)]
struct PrewarmQueueCounters {
    delay: LatencyCounters,
    delay_lt_10us: AtomicU64,
    delay_lt_100us: AtomicU64,
    delay_lt_1ms: AtomicU64,
    delay_lt_10ms: AtomicU64,
    delay_ge_10ms: AtomicU64,
    start_behind: AtomicU64,
    start_current: AtomicU64,
    start_ahead_1_16: AtomicU64,
    start_ahead_17_64: AtomicU64,
    start_ahead_gt_64: AtomicU64,
    queued: AtomicU64,
    running: AtomicU64,
    outstanding: AtomicU64,
    queued_max: AtomicU64,
    running_max: AtomicU64,
    outstanding_max: AtomicU64,
}

impl PrewarmQueueCounters {
    fn dispatch(&self) {
        let queued = self.queued.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        self.queued_max.fetch_max(queued, Ordering::Relaxed);
        let outstanding = self.outstanding.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        self.outstanding_max.fetch_max(outstanding, Ordering::Relaxed);
    }

    fn start(&self, elapsed: Duration, index: usize, executed_index: usize) {
        self.delay.record(elapsed);
        let ns = elapsed.as_nanos();
        let bucket = if ns < 10_000 {
            &self.delay_lt_10us
        } else if ns < 100_000 {
            &self.delay_lt_100us
        } else if ns < 1_000_000 {
            &self.delay_lt_1ms
        } else if ns < 10_000_000 {
            &self.delay_lt_10ms
        } else {
            &self.delay_ge_10ms
        };
        bucket.fetch_add(1, Ordering::Relaxed);

        let lead = index.saturating_sub(executed_index);
        let lead_counter = if index < executed_index {
            &self.start_behind
        } else if lead == 0 {
            &self.start_current
        } else if lead <= 16 {
            &self.start_ahead_1_16
        } else if lead <= 64 {
            &self.start_ahead_17_64
        } else {
            &self.start_ahead_gt_64
        };
        lead_counter.fetch_add(1, Ordering::Relaxed);

        let running = self.running.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        self.running_max.fetch_max(running, Ordering::Relaxed);
        self.queued.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Block-local, bounded correlation state for execution-cache and prewarm diagnostics.
struct ReadinessDiagnostics {
    shards: [PrewarmKeyShard; READINESS_KEY_SHARDS],
    shard_hasher: DefaultHashBuilder,
    account_misses: MissCounters,
    storage_misses: MissCounters,
    code_misses: MissCounters,
    storage_backing_latency: StorageBackingLatency,
    storage_unfinished: StorageUnfinishedCounters,
    prewarm_queue: PrewarmQueueCounters,
    lock_contention: AtomicU64,
    prewarm_totals: Arc<ReadTotals>,
    emitted: AtomicBool,
    #[cfg(test)]
    emission_count: AtomicUsize,
}

impl fmt::Debug for ReadinessDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadinessDiagnostics")
            .field("key_capacity", &READINESS_KEY_CAPACITY)
            .field("lock_contention", &self.lock_contention.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ReadinessDiagnostics {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            shards: std::array::from_fn(|_| PrewarmKeyShard {
                keys: parking_lot::Mutex::new(PrewarmKeys {
                    entries: HashMap::with_capacity(READINESS_KEYS_PER_SHARD),
                }),
                coverage_lost: AtomicBool::new(false),
                cap_reached: AtomicBool::new(false),
            }),
            shard_hasher: DefaultHashBuilder::default(),
            account_misses: MissCounters::default(),
            storage_misses: MissCounters::default(),
            code_misses: MissCounters::default(),
            storage_backing_latency: StorageBackingLatency::default(),
            storage_unfinished: StorageUnfinishedCounters::default(),
            prewarm_queue: PrewarmQueueCounters::default(),
            lock_contention: AtomicU64::new(0),
            prewarm_totals: ReadTotals::new(),
            emitted: AtomicBool::new(false),
            #[cfg(test)]
            emission_count: AtomicUsize::new(0),
        })
    }

    fn shard_index(&self, key: ReadinessKey) -> usize {
        self.shard_hasher.hash_one(key) as usize % READINESS_KEY_SHARDS
    }

    fn shard(&self, key: ReadinessKey) -> &PrewarmKeyShard {
        &self.shards[self.shard_index(key)]
    }

    fn begin_prewarm(self: &Arc<Self>, key: ReadinessKey) -> PrewarmReadGuard {
        let shard = self.shard(key);
        let Some(mut keys) = shard.keys.try_lock() else {
            self.lock_contention.fetch_add(1, Ordering::Relaxed);
            shard.coverage_lost.store(true, Ordering::Release);
            return PrewarmReadGuard { state: None, instance: None, successful: false }
        };
        let state = if let Some(state) = keys.entries.get(&key) {
            let state = Arc::clone(state);
            let _ = state.inflight.fetch_update(Ordering::AcqRel, Ordering::Acquire, |inflight| {
                Some(inflight.saturating_add(1))
            });
            Some(state)
        } else if keys.entries.len() < READINESS_KEYS_PER_SHARD {
            let state = Arc::new(PrewarmKeyState::default());
            state.inflight.store(1, Ordering::Release);
            keys.entries.insert(key, Arc::clone(&state));
            Some(state)
        } else {
            shard.cap_reached.store(true, Ordering::Release);
            None
        };
        let instance = matches!(key, ReadinessKey::Storage(..))
            .then(|| {
                state.as_ref().and_then(|state| {
                    let Some(mut active_reads) = state.active_reads.try_lock() else {
                        state.timing_coverage_lost.store(true, Ordering::Release);
                        return None
                    };
                    active_reads.retain(|read| {
                        matches!(
                            read.outcome.load(Ordering::Acquire),
                            PREWARM_READ_ACTIVE | PREWARM_READ_FINISHING
                        )
                    });
                    if active_reads.len() >= READINESS_ACTIVE_READS_PER_KEY_CAP {
                        state.timing_coverage_lost.store(true, Ordering::Release);
                        return None
                    }
                    let instance = Arc::new(PrewarmReadInstance {
                        started: Instant::now(),
                        outcome: std::sync::atomic::AtomicU8::new(PREWARM_READ_ACTIVE),
                        finished_elapsed_ns: AtomicU64::new(0),
                    });
                    active_reads.push(Arc::clone(&instance));
                    Some(instance)
                })
            })
            .flatten();
        PrewarmReadGuard { state, instance, successful: false }
    }

    fn record_miss(&self, key: ReadinessKey) -> MissPrewarmState {
        self.observe_miss(key, false).0
    }

    fn observe_miss(
        &self,
        key: ReadinessKey,
        capture_unfinished: bool,
    ) -> (MissPrewarmState, UnfinishedSnapshot, Instant) {
        let shard = self.shard(key);
        let (state, unfinished, observed_at) = if let Some(keys) = shard.keys.try_lock() {
            let coverage_lost = shard.coverage_lost.load(Ordering::Acquire);
            match keys.entries.get(&key) {
                Some(key_state) => {
                    // Sample inflight first. Success publishes `completed` before decrementing
                    // inflight, so observing zero and then loading completed cannot misclassify a
                    // concurrently finishing successful read as failed. Completion still wins
                    // over a later overlapping retry.
                    let inflight = key_state.inflight.load(Ordering::Acquire);
                    let completed = key_state.completed.load(Ordering::Acquire);
                    let state = if completed {
                        MissPrewarmState::Completed
                    } else if inflight != 0 {
                        MissPrewarmState::Inflight
                    } else if coverage_lost {
                        MissPrewarmState::UnknownContention
                    } else {
                        MissPrewarmState::Failed
                    };
                    let (unfinished, observed_at) = if !capture_unfinished {
                        (UnfinishedSnapshot::None, Instant::now())
                    } else if coverage_lost ||
                        key_state.timing_coverage_lost.load(Ordering::Acquire)
                    {
                        (UnfinishedSnapshot::Unknown, Instant::now())
                    } else if let Some(active_reads) = key_state.active_reads.try_lock() {
                        // Registration also holds this lock. Taking the timestamp here ensures
                        // every captured read started no later than the exact cohort boundary.
                        let observed_at = Instant::now();
                        let mut finishing = false;
                        let captured = active_reads
                            .iter()
                            .filter_map(|instance| match instance.outcome.load(Ordering::Acquire) {
                                PREWARM_READ_ACTIVE => Some(CapturedPrewarmRead {
                                    instance: Arc::clone(instance),
                                    age_ns: duration_ns(
                                        observed_at.duration_since(instance.started),
                                    ),
                                }),
                                PREWARM_READ_FINISHING => {
                                    finishing = true;
                                    None
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        let unfinished = if finishing {
                            UnfinishedSnapshot::Unknown
                        } else if captured.is_empty() {
                            UnfinishedSnapshot::None
                        } else {
                            UnfinishedSnapshot::Known(captured)
                        };
                        (unfinished, observed_at)
                    } else {
                        (UnfinishedSnapshot::Unknown, Instant::now())
                    };
                    (state, unfinished, observed_at)
                }
                None if coverage_lost => (
                    MissPrewarmState::UnknownContention,
                    UnfinishedSnapshot::Unknown,
                    Instant::now(),
                ),
                None if shard.cap_reached.load(Ordering::Acquire) => {
                    (MissPrewarmState::UnknownDueCap, UnfinishedSnapshot::Unknown, Instant::now())
                }
                None => (MissPrewarmState::NeverObserved, UnfinishedSnapshot::None, Instant::now()),
            }
        } else {
            self.lock_contention.fetch_add(1, Ordering::Relaxed);
            (MissPrewarmState::UnknownContention, UnfinishedSnapshot::Unknown, Instant::now())
        };
        match key {
            ReadinessKey::Account(_) => self.account_misses.record(state),
            ReadinessKey::Storage(..) => self.storage_misses.record(state),
            ReadinessKey::Code(_) => self.code_misses.record(state),
        }
        (state, unfinished, observed_at)
    }

    fn begin_storage_backing_read(
        self: &Arc<Self>,
        account: Address,
        storage_key: StorageKey,
    ) -> StorageBackingReadGuard {
        let (state, unfinished, start) =
            self.observe_miss(ReadinessKey::Storage(account, storage_key), true);
        match &unfinished {
            UnfinishedSnapshot::None => {}
            UnfinishedSnapshot::Unknown => {
                self.storage_unfinished.unknown.fetch_add(1, Ordering::Relaxed);
            }
            UnfinishedSnapshot::Known(reads) => {
                let multiplicity = if reads.len() == 1 {
                    &self.storage_unfinished.single
                } else {
                    &self.storage_unfinished.multiple
                };
                multiplicity.fetch_add(1, Ordering::Relaxed);
                let oldest_age = reads.iter().map(|read| read.age_ns).max().unwrap_or_default();
                self.storage_unfinished.age.record(Duration::from_nanos(oldest_age));
            }
        }
        StorageBackingReadGuard {
            diagnostics: Arc::clone(self),
            state,
            unfinished,
            overlap_start: start,
            backing_start: Instant::now(),
            successful: false,
            finished_elapsed: None,
        }
    }

    fn prewarm_dispatch(self: &Arc<Self>) -> PrewarmDispatchGuard {
        self.prewarm_queue.dispatch();
        PrewarmDispatchGuard {
            diagnostics: Arc::clone(self),
            dispatched: Instant::now(),
            started: false,
        }
    }

    fn emit_once(&self, parent: &Span, checkout_reason: u64) {
        if self.emitted.swap(true, Ordering::AcqRel) {
            return
        }
        #[cfg(test)]
        self.emission_count.fetch_add(1, Ordering::Relaxed);
        let mut keys_tracked = 0;
        for shard in &self.shards {
            if let Some(keys) = shard.keys.try_lock() {
                keys_tracked += keys.entries.len() as u64;
            } else {
                self.lock_contention.fetch_add(1, Ordering::Relaxed);
            }
        }
        let cap_reached =
            u64::from(self.shards.iter().any(|shard| shard.cap_reached.load(Ordering::Acquire)));
        info!(
            target: "lifecycle",
            parent: parent,
            stage = "execution_cache_readiness",
            cache_checkout_reason = checkout_reason,
            cache_diag_keys_tracked = keys_tracked,
            cache_diag_key_capacity = READINESS_KEY_CAPACITY as u64,
            cache_diag_cap_reached = cap_reached,
            cache_diag_lock_contention = self.lock_contention.load(Ordering::Relaxed),
            account_miss_prewarm_inflight = self.account_misses.inflight.load(Ordering::Relaxed),
            account_miss_prewarm_completed = self.account_misses.completed.load(Ordering::Relaxed),
            account_miss_prewarm_failed = self.account_misses.failed.load(Ordering::Relaxed),
            account_miss_prewarm_never_observed = self.account_misses.never_observed.load(Ordering::Relaxed),
            account_miss_prewarm_unknown_due_cap = self.account_misses.unknown_due_cap.load(Ordering::Relaxed),
            account_miss_prewarm_unknown_contention = self.account_misses.unknown_contention.load(Ordering::Relaxed),
            storage_miss_prewarm_inflight = self.storage_misses.inflight.load(Ordering::Relaxed),
            storage_miss_prewarm_completed = self.storage_misses.completed.load(Ordering::Relaxed),
            storage_miss_prewarm_failed = self.storage_misses.failed.load(Ordering::Relaxed),
            storage_miss_prewarm_never_observed = self.storage_misses.never_observed.load(Ordering::Relaxed),
            storage_miss_prewarm_unknown_due_cap = self.storage_misses.unknown_due_cap.load(Ordering::Relaxed),
            storage_miss_prewarm_unknown_contention = self.storage_misses.unknown_contention.load(Ordering::Relaxed),
            code_miss_prewarm_inflight = self.code_misses.inflight.load(Ordering::Relaxed),
            code_miss_prewarm_completed = self.code_misses.completed.load(Ordering::Relaxed),
            code_miss_prewarm_failed = self.code_misses.failed.load(Ordering::Relaxed),
            code_miss_prewarm_never_observed = self.code_misses.never_observed.load(Ordering::Relaxed),
            code_miss_prewarm_unknown_due_cap = self.code_misses.unknown_due_cap.load(Ordering::Relaxed),
            code_miss_prewarm_unknown_contention = self.code_misses.unknown_contention.load(Ordering::Relaxed),
            storage_backing_inflight_count = self.storage_backing_latency.inflight.count.load(Ordering::Relaxed),
            storage_backing_inflight_ns = self.storage_backing_latency.inflight.ns.load(Ordering::Relaxed),
            storage_backing_inflight_max_ns = self.storage_backing_latency.inflight.max_ns.load(Ordering::Relaxed),
            storage_backing_completed_count = self.storage_backing_latency.completed.count.load(Ordering::Relaxed),
            storage_backing_completed_ns = self.storage_backing_latency.completed.ns.load(Ordering::Relaxed),
            storage_backing_completed_max_ns = self.storage_backing_latency.completed.max_ns.load(Ordering::Relaxed),
            storage_backing_never_count = self.storage_backing_latency.never_observed.count.load(Ordering::Relaxed),
            storage_backing_never_ns = self.storage_backing_latency.never_observed.ns.load(Ordering::Relaxed),
            storage_backing_never_max_ns = self.storage_backing_latency.never_observed.max_ns.load(Ordering::Relaxed),
            storage_backing_unknown_count = self.storage_backing_latency.unknown.count.load(Ordering::Relaxed),
            storage_backing_unknown_ns = self.storage_backing_latency.unknown.ns.load(Ordering::Relaxed),
            storage_backing_unknown_max_ns = self.storage_backing_latency.unknown.max_ns.load(Ordering::Relaxed),
            storage_backing_failed_count = self.storage_backing_latency.failed.count.load(Ordering::Relaxed),
            storage_backing_failed_ns = self.storage_backing_latency.failed.ns.load(Ordering::Relaxed),
            storage_backing_failed_max_ns = self.storage_backing_latency.failed.max_ns.load(Ordering::Relaxed),
            storage_unfinished_at_miss_count = self.storage_unfinished.age.count.load(Ordering::Relaxed),
            storage_unfinished_at_miss_single = self.storage_unfinished.single.load(Ordering::Relaxed),
            storage_unfinished_at_miss_multiple = self.storage_unfinished.multiple.load(Ordering::Relaxed),
            storage_unfinished_at_miss_unknown = self.storage_unfinished.unknown.load(Ordering::Relaxed),
            storage_unfinished_age_ns = self.storage_unfinished.age.ns.load(Ordering::Relaxed),
            storage_unfinished_age_max_ns = self.storage_unfinished.age.max_ns.load(Ordering::Relaxed),
            storage_unfinished_overlap_count = self.storage_unfinished.overlap.count.load(Ordering::Relaxed),
            storage_unfinished_overlap_ns = self.storage_unfinished.overlap.ns.load(Ordering::Relaxed),
            storage_unfinished_overlap_max_ns = self.storage_unfinished.overlap.max_ns.load(Ordering::Relaxed),
            storage_unfinished_prewarm_success_first = self.storage_unfinished.prewarm_success_first.load(Ordering::Relaxed),
            storage_unfinished_prewarm_failed_first = self.storage_unfinished.prewarm_failed_first.load(Ordering::Relaxed),
            storage_unfinished_canonical_success_first = self.storage_unfinished.canonical_success_first.load(Ordering::Relaxed),
            storage_unfinished_canonical_failed_first = self.storage_unfinished.canonical_failed_first.load(Ordering::Relaxed),
            storage_unfinished_winner_unknown = self.storage_unfinished.winner_unknown.load(Ordering::Relaxed),
            prewarm_queue_delay_count = self.prewarm_queue.delay.count.load(Ordering::Relaxed),
            prewarm_queue_delay_ns = self.prewarm_queue.delay.ns.load(Ordering::Relaxed),
            prewarm_queue_delay_max_ns = self.prewarm_queue.delay.max_ns.load(Ordering::Relaxed),
            prewarm_queue_delay_lt_10us = self.prewarm_queue.delay_lt_10us.load(Ordering::Relaxed),
            prewarm_queue_delay_lt_100us = self.prewarm_queue.delay_lt_100us.load(Ordering::Relaxed),
            prewarm_queue_delay_lt_1ms = self.prewarm_queue.delay_lt_1ms.load(Ordering::Relaxed),
            prewarm_queue_delay_lt_10ms = self.prewarm_queue.delay_lt_10ms.load(Ordering::Relaxed),
            prewarm_queue_delay_ge_10ms = self.prewarm_queue.delay_ge_10ms.load(Ordering::Relaxed),
            prewarm_start_behind = self.prewarm_queue.start_behind.load(Ordering::Relaxed),
            prewarm_start_current = self.prewarm_queue.start_current.load(Ordering::Relaxed),
            prewarm_start_ahead_1_16 = self.prewarm_queue.start_ahead_1_16.load(Ordering::Relaxed),
            prewarm_start_ahead_17_64 = self.prewarm_queue.start_ahead_17_64.load(Ordering::Relaxed),
            prewarm_start_ahead_gt_64 = self.prewarm_queue.start_ahead_gt_64.load(Ordering::Relaxed),
            prewarm_queued_max = self.prewarm_queue.queued_max.load(Ordering::Relaxed),
            prewarm_running_max = self.prewarm_queue.running_max.load(Ordering::Relaxed),
            prewarm_outstanding_max = self.prewarm_queue.outstanding_max.load(Ordering::Relaxed),
        );
        self.prewarm_totals.emit(parent, Role::Prewarm);
    }
}

struct PrewarmReadGuard {
    state: Option<Arc<PrewarmKeyState>>,
    instance: Option<Arc<PrewarmReadInstance>>,
    successful: bool,
}

impl PrewarmReadGuard {
    /// Marks the backing provider read as successfully completed.
    fn finish_success(&mut self) {
        self.successful = true;
    }
}

impl Drop for PrewarmReadGuard {
    fn drop(&mut self) {
        let Some(state) = &self.state else { return };
        if let Some(instance) = &self.instance {
            // `FINISHING` makes the small publication interval explicit. A concurrent observer
            // reports unknown rather than inferring an ordering from an unpublished timestamp.
            instance.outcome.store(PREWARM_READ_FINISHING, Ordering::Release);
            instance
                .finished_elapsed_ns
                .store(duration_ns(instance.started.elapsed()), Ordering::Relaxed);
            instance.outcome.store(
                if self.successful { PREWARM_READ_SUCCESS } else { PREWARM_READ_FAILED },
                Ordering::Release,
            );
        }
        if self.successful {
            state.completed.store(true, Ordering::Release);
        }
        let _ = state.inflight.fetch_update(Ordering::AcqRel, Ordering::Acquire, |inflight| {
            Some(inflight.saturating_sub(1))
        });
    }
}

struct StorageBackingReadGuard {
    diagnostics: Arc<ReadinessDiagnostics>,
    state: MissPrewarmState,
    unfinished: UnfinishedSnapshot,
    overlap_start: Instant,
    backing_start: Instant,
    successful: bool,
    finished_elapsed: Option<(Duration, Duration)>,
}

impl StorageBackingReadGuard {
    /// Captures the provider-return boundary and whether the call succeeded.
    fn finish(&mut self, successful: bool) {
        self.successful = successful;
        self.finished_elapsed = Some((self.overlap_start.elapsed(), self.backing_start.elapsed()));
    }
}

impl Drop for StorageBackingReadGuard {
    fn drop(&mut self) {
        let (canonical_elapsed, backing_elapsed) = self
            .finished_elapsed
            .unwrap_or_else(|| (self.overlap_start.elapsed(), self.backing_start.elapsed()));
        self.diagnostics.storage_backing_latency.counters(self.state).record(backing_elapsed);
        let UnfinishedSnapshot::Known(reads) = &self.unfinished else { return };

        let canonical_ns = duration_ns(canonical_elapsed);
        let mut earliest_prewarm: Option<(u64, u8)> = None;
        let mut ambiguous = false;
        for read in reads {
            let outcome = read.instance.outcome.load(Ordering::Acquire);
            if outcome == PREWARM_READ_ACTIVE {
                continue
            }
            if outcome == PREWARM_READ_FINISHING {
                ambiguous = true;
                break
            }
            let finished_elapsed_ns = read.instance.finished_elapsed_ns.load(Ordering::Relaxed);
            let Some(after_miss_ns) = finished_elapsed_ns.checked_sub(read.age_ns) else {
                ambiguous = true;
                break
            };
            if earliest_prewarm.is_none_or(|(earliest, _)| after_miss_ns < earliest) {
                earliest_prewarm = Some((after_miss_ns, outcome));
            }
        }

        let counters = &self.diagnostics.storage_unfinished;
        if ambiguous || earliest_prewarm.is_some_and(|(elapsed, _)| elapsed == canonical_ns) {
            counters.winner_unknown.fetch_add(1, Ordering::Relaxed);
            return
        }

        if let Some((elapsed, outcome)) =
            earliest_prewarm.filter(|(elapsed, _)| *elapsed < canonical_ns)
        {
            counters.overlap.record(Duration::from_nanos(elapsed));
            let winner = if outcome == PREWARM_READ_SUCCESS {
                &counters.prewarm_success_first
            } else if outcome == PREWARM_READ_FAILED {
                &counters.prewarm_failed_first
            } else {
                counters.winner_unknown.fetch_add(1, Ordering::Relaxed);
                return
            };
            winner.fetch_add(1, Ordering::Relaxed);
        } else {
            counters.overlap.record(canonical_elapsed);
            let winner = if self.successful {
                &counters.canonical_success_first
            } else {
                &counters.canonical_failed_first
            };
            winner.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Diagnostics-only lifetime token for a receiver transaction prewarm task.
///
/// It retains only the block-local diagnostics allocation. Calling [`Self::start`] records the
/// scheduler delay and the transaction's lead over the canonical completed-transaction counter.
pub struct PrewarmDispatchGuard {
    diagnostics: Arc<ReadinessDiagnostics>,
    dispatched: Instant,
    started: bool,
}

impl fmt::Debug for PrewarmDispatchGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrewarmDispatchGuard").field("started", &self.started).finish()
    }
}

impl PrewarmDispatchGuard {
    /// Marks the task as started before worker-local EVM initialization.
    pub fn start(&mut self, transaction_index: usize, executed_transaction_index: usize) {
        if self.started {
            return
        }
        self.started = true;
        self.diagnostics.prewarm_queue.start(
            self.dispatched.elapsed(),
            transaction_index,
            executed_transaction_index,
        );
    }
}

impl Drop for PrewarmDispatchGuard {
    fn drop(&mut self) {
        if self.started {
            self.diagnostics.prewarm_queue.running.fetch_sub(1, Ordering::Relaxed);
        } else {
            self.diagnostics.prewarm_queue.queued.fetch_sub(1, Ordering::Relaxed);
        }
        self.diagnostics.prewarm_queue.outstanding.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Alignment in bytes for entries in the fixed-cache.
///
/// Each bucket in `fixed-cache` is aligned to 128 bytes (cache line) due to
/// `#[repr(C, align(128))]` on the internal `Bucket` struct.
const FIXED_CACHE_ALIGNMENT: usize = 128;

/// Overhead per entry in the fixed-cache (the `AtomicUsize` tag field).
const FIXED_CACHE_ENTRY_OVERHEAD: usize = size_of::<usize>();

/// Calculates the actual size of a fixed-cache entry for a given key-value pair.
///
/// The entry size is `overhead + size_of::<K>() + size_of::<V>()`, rounded up to the
/// next multiple of [`FIXED_CACHE_ALIGNMENT`] (128 bytes).
const fn fixed_cache_entry_size<K, V>() -> usize {
    fixed_cache_key_size_with_value::<K>(size_of::<V>())
}

/// Calculates the actual size of a fixed-cache entry for a given key-value pair.
///
/// The entry size is `overhead + size_of::<K>() + size_of::<V>()`, rounded up to the
/// next multiple of [`FIXED_CACHE_ALIGNMENT`] (128 bytes).
const fn fixed_cache_key_size_with_value<K>(value: usize) -> usize {
    let raw_size = FIXED_CACHE_ENTRY_OVERHEAD + size_of::<K>() + value;
    // Round up to next multiple of alignment
    raw_size.div_ceil(FIXED_CACHE_ALIGNMENT) * FIXED_CACHE_ALIGNMENT
}

/// Estimated average bytecode size for cache budget calculation.
///
/// The fixed-cache stores `Option<Bytecode>` inline (pointer-sized), but each cached contract
/// also holds bytecode on the heap. For budget estimation we use 8 KiB, which is close to the
/// observed mainnet average (~7 KiB). Using `MAX_CODE_SIZE` (48 KiB) overestimates by ~7x,
/// yielding only 4096 entries for a 228 MB code-cache budget when 16384 fit comfortably.
const ESTIMATED_AVG_CODE_SIZE: usize = 8 * 1024;

/// Size in bytes of a single code cache entry (inline metadata + estimated heap).
const CODE_CACHE_ENTRY_SIZE: usize =
    fixed_cache_key_size_with_value::<Address>(ESTIMATED_AVG_CODE_SIZE);

/// Size in bytes of a single storage cache entry.
const STORAGE_CACHE_ENTRY_SIZE: usize =
    fixed_cache_entry_size::<(Address, StorageKey), StorageValue>();

/// Size in bytes of a single account cache entry.
const ACCOUNT_CACHE_ENTRY_SIZE: usize = fixed_cache_entry_size::<Address, Option<Account>>();

/// Cache configuration with epoch tracking enabled for O(1) cache invalidation.
struct EpochCacheConfig;
impl CacheConfig for EpochCacheConfig {
    const EPOCHS: bool = true;
}

/// Type alias for the fixed-cache used for accounts and storage.
type FixedCache<K, V, H = DefaultHashBuilder> = fixed_cache::Cache<K, V, H, EpochCacheConfig>;

/// A wrapper of a state provider and a shared cache.
///
/// [`CacheFillMode`] controls whether misses populate the shared cache. This is used by background
/// prewarmers and speculative execution workers that intentionally seed the cache for other
/// readers. Canonical execution usually leaves this disabled because the EVM database `State`
/// already caches reads during the block, and the shared cache is updated after the block from the
/// final [`BundleState`]. See also [`ExecutionCache::insert_state`].
///
/// Execution-cache and txpool-snapshot hit/miss metrics are recorded separately when
/// [`CachedStateMetrics`] is provided. Slow-block [`CacheStats`] are controlled separately by
/// [`Self::new_with_mode`].
#[derive(Debug)]
pub struct CachedStateProvider<S> {
    /// The state provider
    state_provider: S,

    /// The caches used for the provider
    caches: ExecutionCache,

    /// Optional immutable txpool-prewarm snapshot consulted before the regular execution cache.
    txpool_snapshot: Option<TxPoolPrewarmCacheSnapshot>,

    /// Metrics for the cached state provider.
    metrics: Option<CachedStateMetrics>,

    /// Provider-local execution-cache hit/miss counters flushed when the provider is dropped.
    execution_metric_counts: CacheMetricCounts,

    /// Provider-local txpool-cache hit/miss counters flushed when the provider is dropped.
    txpool_metric_counts: CacheMetricCounts,

    /// Whether cache misses should populate the shared execution cache.
    fill_mode: CacheFillMode,

    /// Diagnostic role of reads through this provider.
    readiness_access: ReadinessAccess,

    /// Immutable handle to this checkout's block-local diagnostics.
    readiness: Option<Arc<ReadinessDiagnostics>>,

    /// Optional cache statistics for detailed block logging. Only tracked when slow block
    /// threshold is configured.
    cache_stats: Option<Arc<CacheStats>>,
}

impl<S> CachedStateProvider<S> {
    /// Creates a new [`CachedStateProvider`] from an [`ExecutionCache`], state provider, and
    /// optional [`CachedStateMetrics`].
    pub fn new(
        state_provider: S,
        caches: ExecutionCache,
        metrics: Option<CachedStateMetrics>,
    ) -> Self {
        Self::new_with_mode(state_provider, caches, CacheFillMode::LookupOnly, metrics, None)
    }

    /// Creates a cache-filling [`CachedStateProvider`].
    ///
    /// Doesn't accept metrics because prewarming path does not need to report hit/misses.
    pub fn new_prewarm(state_provider: S, caches: ExecutionCache) -> Self {
        let mut provider =
            Self::new_with_mode(state_provider, caches, CacheFillMode::FillOnMiss, None, None);
        provider.readiness_access = ReadinessAccess::Prewarm;
        provider
    }

    /// Creates a [`CachedStateProvider`] with explicit cache fill behavior and optional
    /// block-local cache stats.
    pub fn new_with_mode(
        state_provider: S,
        caches: ExecutionCache,
        fill_mode: CacheFillMode,
        metrics: Option<CachedStateMetrics>,
        cache_stats: Option<Arc<CacheStats>>,
    ) -> Self {
        let readiness = caches.readiness();
        Self {
            state_provider,
            caches,
            txpool_snapshot: None,
            metrics,
            execution_metric_counts: CacheMetricCounts::new(),
            txpool_metric_counts: CacheMetricCounts::new(),
            fill_mode,
            readiness_access: ReadinessAccess::Authoritative,
            readiness,
            cache_stats,
        }
    }

    /// Adds an immutable txpool-prewarm snapshot as the first cache lookup tier.
    pub fn with_txpool_snapshot(mut self, snapshot: Option<TxPoolPrewarmCacheSnapshot>) -> Self {
        self.txpool_snapshot = snapshot;
        self
    }

    fn record_account_hit(&self) {
        self.record_metric(CacheMetricKind::AccountHit);
        if let Some(stats) = &self.cache_stats {
            stats.record_account_hit();
        }
    }

    fn record_account_miss(&self) {
        self.record_metric(CacheMetricKind::AccountMiss);
        if let Some(stats) = &self.cache_stats {
            stats.record_account_miss();
        }
    }

    fn record_storage_hit(&self) {
        self.record_metric(CacheMetricKind::StorageHit);
        if let Some(stats) = &self.cache_stats {
            stats.record_storage_hit();
        }
    }

    fn record_storage_miss(&self) {
        self.record_metric(CacheMetricKind::StorageMiss);
        if let Some(stats) = &self.cache_stats {
            stats.record_storage_miss();
        }
    }

    fn record_code_hit(&self) {
        self.record_metric(CacheMetricKind::CodeHit);
        if let Some(stats) = &self.cache_stats {
            stats.record_code_hit();
        }
    }

    fn record_code_miss(&self) {
        self.record_metric(CacheMetricKind::CodeMiss);
        if let Some(stats) = &self.cache_stats {
            stats.record_code_miss();
        }
    }

    fn record_txpool_account_hit(&self) {
        self.record_txpool_metric(CacheMetricKind::AccountHit);
        if let Some(stats) = &self.cache_stats {
            stats.record_txpool_snapshot_account_hit();
        }
    }

    fn record_txpool_account_miss(&self) {
        self.record_txpool_metric(CacheMetricKind::AccountMiss);
        if let Some(stats) = &self.cache_stats {
            stats.record_txpool_snapshot_account_miss();
        }
    }

    fn record_txpool_storage_hit(&self) {
        self.record_txpool_metric(CacheMetricKind::StorageHit);
        if let Some(stats) = &self.cache_stats {
            stats.record_txpool_snapshot_storage_hit();
        }
    }

    fn record_txpool_storage_miss(&self) {
        self.record_txpool_metric(CacheMetricKind::StorageMiss);
        if let Some(stats) = &self.cache_stats {
            stats.record_txpool_snapshot_storage_miss();
        }
    }

    fn record_txpool_code_hit(&self) {
        self.record_txpool_metric(CacheMetricKind::CodeHit);
        if let Some(stats) = &self.cache_stats {
            stats.record_txpool_snapshot_code_hit();
        }
    }

    fn record_txpool_code_miss(&self) {
        self.record_txpool_metric(CacheMetricKind::CodeMiss);
        if let Some(stats) = &self.cache_stats {
            stats.record_txpool_snapshot_code_miss();
        }
    }

    #[inline]
    fn record_metric(&self, kind: CacheMetricKind) {
        if self.metrics.is_some() {
            self.execution_metric_counts.record(kind);
        }
    }

    #[inline]
    fn record_txpool_metric(&self, kind: CacheMetricKind) {
        if self.metrics.is_some() {
            self.txpool_metric_counts.record(kind);
        }
    }

    fn flush_buffered_metrics(&self) {
        let execution_counts = self.execution_metric_counts.take();
        let txpool_counts = self.txpool_metric_counts.take();
        if execution_counts.is_empty() && txpool_counts.is_empty() {
            return;
        }

        if let Some(metrics) = &self.metrics {
            metrics.record_access_counts(execution_counts);
            metrics.record_txpool_access_counts(txpool_counts);
        }
    }

    const fn should_fill_on_miss(&self) -> bool {
        matches!(self.fill_mode, CacheFillMode::FillOnMiss)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadinessAccess {
    Authoritative,
    Prewarm,
}

impl<S> Drop for CachedStateProvider<S> {
    fn drop(&mut self) {
        self.flush_buffered_metrics();
    }
}

/// Whether cache misses should populate the shared execution cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheFillMode {
    /// Only read existing cache entries.
    LookupOnly,
    /// Insert values loaded from the underlying provider.
    FillOnMiss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheMetricKind {
    AccountHit,
    AccountMiss,
    StorageHit,
    StorageMiss,
    CodeHit,
    CodeMiss,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CacheMetricSnapshot {
    account_hits: u64,
    account_misses: u64,
    storage_hits: u64,
    storage_misses: u64,
    code_hits: u64,
    code_misses: u64,
}

impl CacheMetricSnapshot {
    const fn is_empty(&self) -> bool {
        self.account_hits == 0 &&
            self.account_misses == 0 &&
            self.storage_hits == 0 &&
            self.storage_misses == 0 &&
            self.code_hits == 0 &&
            self.code_misses == 0
    }
}

#[derive(Debug, Default)]
struct CacheMetricCounts {
    account_hits: Cell<u64>,
    account_misses: Cell<u64>,
    storage_hits: Cell<u64>,
    storage_misses: Cell<u64>,
    code_hits: Cell<u64>,
    code_misses: Cell<u64>,
}

impl CacheMetricCounts {
    const fn new() -> Self {
        Self {
            account_hits: Cell::new(0),
            account_misses: Cell::new(0),
            storage_hits: Cell::new(0),
            storage_misses: Cell::new(0),
            code_hits: Cell::new(0),
            code_misses: Cell::new(0),
        }
    }

    #[inline]
    fn record(&self, kind: CacheMetricKind) {
        let counter = match kind {
            CacheMetricKind::AccountHit => &self.account_hits,
            CacheMetricKind::AccountMiss => &self.account_misses,
            CacheMetricKind::StorageHit => &self.storage_hits,
            CacheMetricKind::StorageMiss => &self.storage_misses,
            CacheMetricKind::CodeHit => &self.code_hits,
            CacheMetricKind::CodeMiss => &self.code_misses,
        };
        counter.set(counter.get() + 1);
    }

    const fn take(&self) -> CacheMetricSnapshot {
        CacheMetricSnapshot {
            account_hits: self.account_hits.replace(0),
            account_misses: self.account_misses.replace(0),
            storage_hits: self.storage_hits.replace(0),
            storage_misses: self.storage_misses.replace(0),
            code_hits: self.code_hits.replace(0),
            code_misses: self.code_misses.replace(0),
        }
    }
}

/// Represents the status of a key in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CachedStatus<T> {
    /// The key is not in the cache (or was invalidated). The value was recalculated.
    NotCached(T),
    /// The key exists in cache and has a specific value.
    Cached(T),
}

/// The source that is using the execution cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachedStateMetricsSource {
    /// Engine (validation).
    Engine,
    /// Payload builder.
    Builder,
    /// Tests.
    #[cfg(any(test, feature = "test-utils"))]
    Test,
}

impl fmt::Display for CachedStateMetricsSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine => f.write_str("engine"),
            Self::Builder => f.write_str("builder"),
            #[cfg(any(test, feature = "test-utils"))]
            Self::Test => f.write_str("test"),
        }
    }
}

/// Metrics for the cached state provider, showing hits and misses for each cache tier.
#[derive(Metrics, Clone)]
#[metrics(scope = "sync.caching")]
pub struct CachedStateMetrics {
    /// Number of times a new execution cache was created
    execution_cache_created_total: Counter,

    /// Duration of execution cache creation in seconds
    execution_cache_creation_duration_seconds: Histogram,

    /// Execution-cache code hits
    code_cache_hits: Gauge,

    /// Execution-cache code misses
    code_cache_misses: Gauge,

    /// Execution-cache storage hits
    storage_cache_hits: Gauge,

    /// Execution-cache storage misses
    storage_cache_misses: Gauge,

    /// Execution-cache account hits
    account_cache_hits: Gauge,

    /// Execution-cache account misses
    account_cache_misses: Gauge,

    /// Txpool-prewarm snapshot code hits
    txpool_snapshot_code_hits: Gauge,

    /// Txpool-prewarm snapshot code misses
    txpool_snapshot_code_misses: Gauge,

    /// Txpool-prewarm snapshot storage hits
    txpool_snapshot_storage_hits: Gauge,

    /// Txpool-prewarm snapshot storage misses
    txpool_snapshot_storage_misses: Gauge,

    /// Txpool-prewarm snapshot account hits
    txpool_snapshot_account_hits: Gauge,

    /// Txpool-prewarm snapshot account misses
    txpool_snapshot_account_misses: Gauge,
}

/// Metrics for shared execution cache state.
#[derive(Metrics, Clone)]
#[metrics(scope = "sync.caching")]
pub struct CachedStateCacheMetrics {
    /// Code cache size (number of entries)
    code_cache_size: Gauge,

    /// Code cache capacity (maximum entries)
    code_cache_capacity: Gauge,

    /// Code cache collisions (hash collisions causing eviction)
    code_cache_collisions: Gauge,

    /// Storage cache size (number of entries)
    storage_cache_size: Gauge,

    /// Storage cache capacity (maximum entries)
    storage_cache_capacity: Gauge,

    /// Storage cache collisions (hash collisions causing eviction)
    storage_cache_collisions: Gauge,

    /// Account cache size (number of entries)
    account_cache_size: Gauge,

    /// Account cache capacity (maximum entries)
    account_cache_capacity: Gauge,

    /// Account cache collisions (hash collisions causing eviction)
    account_cache_collisions: Gauge,
}

impl CachedStateMetrics {
    /// Sets all values to zero, indicating that a new block is being executed.
    pub fn reset(&self) {
        // code cache
        self.code_cache_hits.set(0);
        self.code_cache_misses.set(0);

        // storage cache
        self.storage_cache_hits.set(0);
        self.storage_cache_misses.set(0);

        // account cache
        self.account_cache_hits.set(0);
        self.account_cache_misses.set(0);

        // txpool-prewarm code cache
        self.txpool_snapshot_code_hits.set(0);
        self.txpool_snapshot_code_misses.set(0);

        // txpool-prewarm storage cache
        self.txpool_snapshot_storage_hits.set(0);
        self.txpool_snapshot_storage_misses.set(0);

        // txpool-prewarm account cache
        self.txpool_snapshot_account_hits.set(0);
        self.txpool_snapshot_account_misses.set(0);
    }

    /// Returns a new zeroed-out instance of [`CachedStateMetrics`] with a `source` label
    /// to distinguish between different callers (e.g., engine vs builder).
    pub fn zeroed(source: CachedStateMetricsSource) -> Self {
        let zeroed = Self::new_with_labels(&[("source", source.to_string())]);
        zeroed.reset();
        zeroed
    }

    fn record_access(&self, kind: CacheMetricKind, count: u64) {
        match kind {
            CacheMetricKind::AccountHit => self.account_cache_hits.increment(count as f64),
            CacheMetricKind::AccountMiss => self.account_cache_misses.increment(count as f64),
            CacheMetricKind::StorageHit => self.storage_cache_hits.increment(count as f64),
            CacheMetricKind::StorageMiss => self.storage_cache_misses.increment(count as f64),
            CacheMetricKind::CodeHit => self.code_cache_hits.increment(count as f64),
            CacheMetricKind::CodeMiss => self.code_cache_misses.increment(count as f64),
        }
    }

    fn record_access_counts(&self, counts: CacheMetricSnapshot) {
        if counts.account_hits != 0 {
            self.record_access(CacheMetricKind::AccountHit, counts.account_hits);
        }
        if counts.account_misses != 0 {
            self.record_access(CacheMetricKind::AccountMiss, counts.account_misses);
        }
        if counts.storage_hits != 0 {
            self.record_access(CacheMetricKind::StorageHit, counts.storage_hits);
        }
        if counts.storage_misses != 0 {
            self.record_access(CacheMetricKind::StorageMiss, counts.storage_misses);
        }
        if counts.code_hits != 0 {
            self.record_access(CacheMetricKind::CodeHit, counts.code_hits);
        }
        if counts.code_misses != 0 {
            self.record_access(CacheMetricKind::CodeMiss, counts.code_misses);
        }
    }

    fn record_txpool_access(&self, kind: CacheMetricKind, count: u64) {
        match kind {
            CacheMetricKind::AccountHit => {
                self.txpool_snapshot_account_hits.increment(count as f64)
            }
            CacheMetricKind::AccountMiss => {
                self.txpool_snapshot_account_misses.increment(count as f64)
            }
            CacheMetricKind::StorageHit => {
                self.txpool_snapshot_storage_hits.increment(count as f64)
            }
            CacheMetricKind::StorageMiss => {
                self.txpool_snapshot_storage_misses.increment(count as f64)
            }
            CacheMetricKind::CodeHit => self.txpool_snapshot_code_hits.increment(count as f64),
            CacheMetricKind::CodeMiss => self.txpool_snapshot_code_misses.increment(count as f64),
        }
    }

    fn record_txpool_access_counts(&self, counts: CacheMetricSnapshot) {
        if counts.account_hits != 0 {
            self.record_txpool_access(CacheMetricKind::AccountHit, counts.account_hits);
        }
        if counts.account_misses != 0 {
            self.record_txpool_access(CacheMetricKind::AccountMiss, counts.account_misses);
        }
        if counts.storage_hits != 0 {
            self.record_txpool_access(CacheMetricKind::StorageHit, counts.storage_hits);
        }
        if counts.storage_misses != 0 {
            self.record_txpool_access(CacheMetricKind::StorageMiss, counts.storage_misses);
        }
        if counts.code_hits != 0 {
            self.record_txpool_access(CacheMetricKind::CodeHit, counts.code_hits);
        }
        if counts.code_misses != 0 {
            self.record_txpool_access(CacheMetricKind::CodeMiss, counts.code_misses);
        }
    }

    /// Records a new execution cache creation with its duration.
    pub fn record_cache_creation(&self, duration: Duration) {
        self.execution_cache_created_total.increment(1);
        self.execution_cache_creation_duration_seconds.record(duration.as_secs_f64());
    }
}

/// Cache hit/miss statistics for detailed block logging.
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Execution-cache account hits
    account_hits: AtomicUsize,
    /// Execution-cache account misses
    account_misses: AtomicUsize,
    /// Execution-cache storage hits
    storage_hits: AtomicUsize,
    /// Execution-cache storage misses
    storage_misses: AtomicUsize,
    /// Execution-cache code hits
    code_hits: AtomicUsize,
    /// Execution-cache code misses
    code_misses: AtomicUsize,
    /// Txpool-prewarm snapshot account hits
    txpool_snapshot_account_hits: AtomicUsize,
    /// Txpool-prewarm snapshot account misses
    txpool_snapshot_account_misses: AtomicUsize,
    /// Txpool-prewarm snapshot storage hits
    txpool_snapshot_storage_hits: AtomicUsize,
    /// Txpool-prewarm snapshot storage misses
    txpool_snapshot_storage_misses: AtomicUsize,
    /// Txpool-prewarm snapshot code hits
    txpool_snapshot_code_hits: AtomicUsize,
    /// Txpool-prewarm snapshot code misses
    txpool_snapshot_code_misses: AtomicUsize,
}

impl CacheStats {
    /// Records an account cache hit.
    pub fn record_account_hit(&self) {
        self.account_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an account cache miss.
    pub fn record_account_miss(&self) {
        self.account_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the number of account cache hits.
    pub fn account_hits(&self) -> usize {
        self.account_hits.load(Ordering::Relaxed)
    }

    /// Returns the number of account cache misses.
    pub fn account_misses(&self) -> usize {
        self.account_misses.load(Ordering::Relaxed)
    }

    /// Records a storage cache hit.
    pub fn record_storage_hit(&self) {
        self.storage_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a storage cache miss.
    pub fn record_storage_miss(&self) {
        self.storage_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the number of storage cache hits.
    pub fn storage_hits(&self) -> usize {
        self.storage_hits.load(Ordering::Relaxed)
    }

    /// Returns the number of storage cache misses.
    pub fn storage_misses(&self) -> usize {
        self.storage_misses.load(Ordering::Relaxed)
    }

    /// Records a code cache hit.
    pub fn record_code_hit(&self) {
        self.code_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a code cache miss.
    pub fn record_code_miss(&self) {
        self.code_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the number of code cache hits.
    pub fn code_hits(&self) -> usize {
        self.code_hits.load(Ordering::Relaxed)
    }

    /// Returns the number of code cache misses.
    pub fn code_misses(&self) -> usize {
        self.code_misses.load(Ordering::Relaxed)
    }

    /// Records a txpool-prewarm snapshot account hit.
    pub fn record_txpool_snapshot_account_hit(&self) {
        self.txpool_snapshot_account_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a txpool-prewarm snapshot account miss.
    pub fn record_txpool_snapshot_account_miss(&self) {
        self.txpool_snapshot_account_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the number of txpool-prewarm snapshot account hits.
    pub fn txpool_snapshot_account_hits(&self) -> usize {
        self.txpool_snapshot_account_hits.load(Ordering::Relaxed)
    }

    /// Returns the number of txpool-prewarm snapshot account misses.
    pub fn txpool_snapshot_account_misses(&self) -> usize {
        self.txpool_snapshot_account_misses.load(Ordering::Relaxed)
    }

    /// Records a txpool-prewarm snapshot storage hit.
    pub fn record_txpool_snapshot_storage_hit(&self) {
        self.txpool_snapshot_storage_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a txpool-prewarm snapshot storage miss.
    pub fn record_txpool_snapshot_storage_miss(&self) {
        self.txpool_snapshot_storage_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the number of txpool-prewarm snapshot storage hits.
    pub fn txpool_snapshot_storage_hits(&self) -> usize {
        self.txpool_snapshot_storage_hits.load(Ordering::Relaxed)
    }

    /// Returns the number of txpool-prewarm snapshot storage misses.
    pub fn txpool_snapshot_storage_misses(&self) -> usize {
        self.txpool_snapshot_storage_misses.load(Ordering::Relaxed)
    }

    /// Records a txpool-prewarm snapshot code hit.
    pub fn record_txpool_snapshot_code_hit(&self) {
        self.txpool_snapshot_code_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a txpool-prewarm snapshot code miss.
    pub fn record_txpool_snapshot_code_miss(&self) {
        self.txpool_snapshot_code_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the number of txpool-prewarm snapshot code hits.
    pub fn txpool_snapshot_code_hits(&self) -> usize {
        self.txpool_snapshot_code_hits.load(Ordering::Relaxed)
    }

    /// Returns the number of txpool-prewarm snapshot code misses.
    pub fn txpool_snapshot_code_misses(&self) -> usize {
        self.txpool_snapshot_code_misses.load(Ordering::Relaxed)
    }
}

/// A stats handler for fixed-cache that tracks collisions and size.
///
/// Note: Hits and misses are tracked directly by the [`CachedStateProvider`] via
/// [`CachedStateMetrics`], not here. The stats handler is used for:
/// - Collision detection (hash collisions causing eviction of a different key)
/// - Size tracking
///
/// ## Size Tracking
///
/// Size is tracked via `on_insert` and `on_remove` callbacks:
/// - `on_insert`: increment size only when inserting into an empty bucket (no eviction)
/// - `on_remove`: always decrement size
///
/// Collisions (evicting a different key) don't change size since they replace an existing entry.
#[derive(Debug)]
pub struct CacheStatsHandler {
    collisions: AtomicU64,
    size: AtomicUsize,
    capacity: usize,
}

impl CacheStatsHandler {
    /// Creates a new stats handler with all counters initialized to zero.
    pub const fn new(capacity: usize) -> Self {
        Self { collisions: AtomicU64::new(0), size: AtomicUsize::new(0), capacity }
    }

    /// Returns the number of cache collisions.
    pub fn collisions(&self) -> u64 {
        self.collisions.load(Ordering::Relaxed)
    }

    /// Returns the current size (number of entries).
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Returns the capacity (maximum number of entries).
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Increments the size counter. Called on cache insert.
    pub fn increment_size(&self) {
        let _ = self.size.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements the size counter. Called on cache remove.
    pub fn decrement_size(&self) {
        let _ = self.size.fetch_sub(1, Ordering::Relaxed);
    }

    /// Resets size to zero. Called on cache clear.
    pub fn reset_size(&self) {
        self.size.store(0, Ordering::Relaxed);
    }

    /// Resets collision counter to zero (but not size).
    pub fn reset_stats(&self) {
        self.collisions.store(0, Ordering::Relaxed);
    }
}

impl<K: PartialEq, V> StatsHandler<K, V> for CacheStatsHandler {
    fn on_hit(&self, _key: &K, _value: &V) {}

    fn on_miss(&self, _key: AnyRef<'_>) {}

    fn on_insert(&self, key: &K, _value: &V, evicted: Option<(&K, &V)>) {
        match evicted {
            None => {
                // Inserting into an empty bucket
                self.increment_size();
            }
            Some((evicted_key, _)) if evicted_key != key => {
                // Collision: evicting a different key
                self.collisions.fetch_add(1, Ordering::Relaxed);
            }
            Some(_) => {
                // Updating the same key, size unchanged
            }
        }
    }

    fn on_remove(&self, _key: &K, _value: &V) {
        self.decrement_size();
    }
}

impl<S: AccountReader> AccountReader for CachedStateProvider<S> {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        let _read_timer = ReadTimer::start(ReadClass::Account);
        if let Some(snapshot) = &self.txpool_snapshot {
            if let Some(account) = snapshot.account(address) {
                self.record_txpool_account_hit();
                return Ok(account)
            }
            self.record_txpool_account_miss();
        }

        if self.should_fill_on_miss() {
            match self.caches.get_or_try_insert_account_with(*address, || {
                let readiness = self.readiness.clone();
                let mut prewarm = readiness.as_ref().and_then(|diagnostics| {
                    (self.readiness_access == ReadinessAccess::Prewarm)
                        .then(|| diagnostics.begin_prewarm(ReadinessKey::Account(*address)))
                });
                if self.readiness_access == ReadinessAccess::Authoritative {
                    if let Some(diagnostics) = readiness {
                        diagnostics.record_miss(ReadinessKey::Account(*address));
                    }
                }
                let result = self.state_provider.basic_account(address);
                if result.is_ok() {
                    if let Some(prewarm) = prewarm.as_mut() {
                        prewarm.finish_success();
                    }
                }
                result
            })? {
                CachedStatus::NotCached(value) => {
                    self.record_account_miss();
                    Ok(value)
                }
                CachedStatus::Cached(value) => {
                    self.record_account_hit();
                    Ok(value)
                }
            }
        } else if let Some(account) = self.caches.0.account_cache.get(address) {
            self.record_account_hit();
            Ok(account)
        } else {
            self.record_account_miss();
            if let Some(diagnostics) = &self.readiness {
                diagnostics.record_miss(ReadinessKey::Account(*address));
            }
            self.state_provider.basic_account(address)
        }
    }
}

#[inline]
fn nonzero_storage_value(value: StorageValue) -> Option<StorageValue> {
    if value.is_zero() {
        None
    } else {
        Some(value)
    }
}

impl<S: StateProvider> StateProvider for CachedStateProvider<S> {
    fn storage(
        &self,
        account: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        let _read_timer = ReadTimer::start(ReadClass::Storage);
        if let Some(snapshot) = &self.txpool_snapshot {
            if let Some(value) = snapshot.storage(account, storage_key) {
                self.record_txpool_storage_hit();
                return Ok(nonzero_storage_value(value))
            }
            self.record_txpool_storage_miss();
        }

        if self.should_fill_on_miss() {
            match self.caches.get_or_try_insert_storage_with(account, storage_key, || {
                let readiness = self.readiness.clone();
                let mut prewarm = readiness.as_ref().and_then(|diagnostics| {
                    (self.readiness_access == ReadinessAccess::Prewarm).then(|| {
                        diagnostics.begin_prewarm(ReadinessKey::Storage(account, storage_key))
                    })
                });
                let mut backing_read = (self.readiness_access == ReadinessAccess::Authoritative)
                    .then(|| {
                        readiness.as_ref().map(|diagnostics| {
                            diagnostics.begin_storage_backing_read(account, storage_key)
                        })
                    })
                    .flatten();
                let result = self
                    .state_provider
                    .storage(account, storage_key)
                    .map(Option::unwrap_or_default);
                if let Some(backing_read) = backing_read.as_mut() {
                    backing_read.finish(result.is_ok());
                }
                drop(backing_read);
                if result.is_ok() {
                    if let Some(prewarm) = prewarm.as_mut() {
                        prewarm.finish_success();
                    }
                }
                result
            })? {
                CachedStatus::NotCached(value) => {
                    self.record_storage_miss();
                    Ok(nonzero_storage_value(value))
                }
                CachedStatus::Cached(value) => {
                    self.record_storage_hit();
                    Ok(nonzero_storage_value(value))
                }
            }
        } else if let Some(value) = self.caches.0.storage_cache.get(&(account, storage_key)) {
            self.record_storage_hit();
            Ok(nonzero_storage_value(value))
        } else {
            self.record_storage_miss();
            let mut backing_read = self
                .readiness
                .as_ref()
                .map(|diagnostics| diagnostics.begin_storage_backing_read(account, storage_key));
            let result = self.state_provider.storage(account, storage_key);
            if let Some(backing_read) = backing_read.as_mut() {
                backing_read.finish(result.is_ok());
            }
            drop(backing_read);
            result
        }
    }
}

impl<S: BytecodeReader> BytecodeReader for CachedStateProvider<S> {
    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        let _read_timer = ReadTimer::start(ReadClass::Code);
        if let Some(snapshot) = &self.txpool_snapshot {
            if let Some(code) = snapshot.bytecode(code_hash) {
                self.record_txpool_code_hit();
                return Ok(code)
            }
            self.record_txpool_code_miss();
        }

        if self.should_fill_on_miss() {
            match self.caches.get_or_try_insert_code_with(*code_hash, || {
                let readiness = self.readiness.clone();
                let mut prewarm = readiness.as_ref().and_then(|diagnostics| {
                    (self.readiness_access == ReadinessAccess::Prewarm)
                        .then(|| diagnostics.begin_prewarm(ReadinessKey::Code(*code_hash)))
                });
                if self.readiness_access == ReadinessAccess::Authoritative {
                    if let Some(diagnostics) = readiness {
                        diagnostics.record_miss(ReadinessKey::Code(*code_hash));
                    }
                }
                let result = self.state_provider.bytecode_by_hash(code_hash);
                if result.is_ok() {
                    if let Some(prewarm) = prewarm.as_mut() {
                        prewarm.finish_success();
                    }
                }
                result
            })? {
                CachedStatus::NotCached(code) => {
                    self.record_code_miss();
                    Ok(code)
                }
                CachedStatus::Cached(code) => {
                    self.record_code_hit();
                    Ok(code)
                }
            }
        } else if let Some(code) = self.caches.0.code_cache.get(code_hash) {
            self.record_code_hit();
            Ok(code)
        } else {
            self.record_code_miss();
            if let Some(diagnostics) = &self.readiness {
                diagnostics.record_miss(ReadinessKey::Code(*code_hash));
            }
            self.state_provider.bytecode_by_hash(code_hash)
        }
    }
}

impl<S: StateRootProvider> StateRootProvider for CachedStateProvider<S> {
    fn state_root(&self, hashed_state: HashedPostState) -> ProviderResult<B256> {
        self.state_provider.state_root(hashed_state)
    }

    fn state_root_from_nodes(&self, input: TrieInput) -> ProviderResult<B256> {
        self.state_provider.state_root_from_nodes(input)
    }

    fn state_root_with_updates(
        &self,
        hashed_state: HashedPostState,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        self.state_provider.state_root_with_updates(hashed_state)
    }

    fn state_root_from_nodes_with_updates(
        &self,
        input: TrieInput,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        self.state_provider.state_root_from_nodes_with_updates(input)
    }
}

impl<S: StateProofProvider> StateProofProvider for CachedStateProvider<S> {
    fn proof(
        &self,
        input: TrieInput,
        address: Address,
        slots: &[B256],
    ) -> ProviderResult<AccountProof> {
        self.state_provider.proof(input, address, slots)
    }

    fn multiproof(
        &self,
        input: TrieInput,
        targets: MultiProofTargets,
    ) -> ProviderResult<MultiProof> {
        self.state_provider.multiproof(input, targets)
    }

    fn multiproof_v2(
        &self,
        input: TrieInput,
        targets: reth_trie::MultiProofTargetsV2,
    ) -> ProviderResult<reth_trie::DecodedMultiProofV2> {
        self.state_provider.multiproof_v2(input, targets)
    }

    fn witness(
        &self,
        input: TrieInput,
        target: HashedPostState,
        mode: reth_trie::ExecutionWitnessMode,
    ) -> ProviderResult<Vec<alloy_primitives::Bytes>> {
        self.state_provider.witness(input, target, mode)
    }
}

impl<S: StorageRootProvider> StorageRootProvider for CachedStateProvider<S> {
    fn storage_root(
        &self,
        address: Address,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<B256> {
        self.state_provider.storage_root(address, hashed_storage)
    }

    fn storage_proof(
        &self,
        address: Address,
        slot: B256,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageProof> {
        self.state_provider.storage_proof(address, slot, hashed_storage)
    }

    fn storage_multiproof(
        &self,
        address: Address,
        slots: &[B256],
        hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageMultiProof> {
        self.state_provider.storage_multiproof(address, slots, hashed_storage)
    }
}

impl<S: BlockHashReader> BlockHashReader for CachedStateProvider<S> {
    fn block_hash(&self, number: alloy_primitives::BlockNumber) -> ProviderResult<Option<B256>> {
        self.state_provider.block_hash(number)
    }

    fn canonical_hashes_range(
        &self,
        start: alloy_primitives::BlockNumber,
        end: alloy_primitives::BlockNumber,
    ) -> ProviderResult<Vec<B256>> {
        self.state_provider.canonical_hashes_range(start, end)
    }
}

impl<S: HashedPostStateProvider> HashedPostStateProvider for CachedStateProvider<S> {
    fn hashed_post_state(
        &self,
        bundle_state: &reth_revm::db::BundleState,
    ) -> ProviderResult<HashedPostState> {
        self.state_provider.hashed_post_state(bundle_state)
    }
}

/// Execution cache used during block processing.
///
/// Optimizes state access by maintaining in-memory copies of frequently accessed
/// accounts, storage slots, and bytecode. Works in conjunction with prewarming
/// to reduce database I/O during block execution.
///
/// ## Storage Invalidation
///
/// Since EIP-6780, SELFDESTRUCT only works within the same transaction where the
/// contract was created, so we don't need to handle clearing the storage.
#[derive(Debug, Clone)]
pub struct ExecutionCache(Arc<ExecutionCacheInner>);

/// Inner state of the [`ExecutionCache`], wrapped in a single [`Arc`].
#[derive(Debug)]
struct ExecutionCacheInner {
    /// Cache for contract bytecode, keyed by code hash.
    code_cache: FixedCache<B256, Option<Bytecode>, FbBuildHasher<32>>,

    /// Flat storage cache: maps `(Address, StorageKey)` to storage value.
    storage_cache: FixedCache<(Address, StorageKey), StorageValue>,

    /// Cache for basic account information (nonce, balance, code hash).
    account_cache: FixedCache<Address, Option<Account>, FbBuildHasher<20>>,

    /// Stats handler for the code cache (shared with the cache via [`Stats`]).
    code_stats: Arc<CacheStatsHandler>,

    /// Stats handler for the storage cache (shared with the cache via [`Stats`]).
    storage_stats: Arc<CacheStatsHandler>,

    /// Stats handler for the account cache (shared with the cache via [`Stats`]).
    account_stats: Arc<CacheStatsHandler>,

    /// One-time notification when SELFDESTRUCT is encountered
    selfdestruct_encountered: Once,

    /// Per-checkout readiness diagnostics. Replaced only while the cache is exclusively held.
    readiness: parking_lot::Mutex<Option<Arc<ReadinessDiagnostics>>>,
}

impl ExecutionCache {
    /// Minimum cache size required when epochs are enabled.
    /// With EPOCHS=true, fixed-cache requires 12 bottom bits to be zero (2 needed + 10 epoch).
    const MIN_CACHE_SIZE_WITH_EPOCHS: usize = 1 << 12; // 4096

    /// Converts a byte size to number of cache entries, rounding down to a power of two.
    ///
    /// Fixed-cache requires power-of-two sizes for efficient indexing.
    /// With epochs enabled, the minimum size is 4096 entries.
    pub const fn bytes_to_entries(size_bytes: usize, entry_size: usize) -> usize {
        let entries = size_bytes / entry_size;
        // Round down to nearest power of two
        let rounded = if entries == 0 { 1 } else { (entries + 1).next_power_of_two() >> 1 };
        // Ensure minimum size for epoch tracking
        if rounded < Self::MIN_CACHE_SIZE_WITH_EPOCHS {
            Self::MIN_CACHE_SIZE_WITH_EPOCHS
        } else {
            rounded
        }
    }

    /// Build an [`ExecutionCache`] struct, so that execution caches can be easily cloned.
    pub fn new(total_cache_size: usize) -> Self {
        let code_cache_size = (total_cache_size * 556) / 10000; // 5.56% of total
        let storage_cache_size = (total_cache_size * 8888) / 10000; // 88.88% of total
        let account_cache_size = (total_cache_size * 556) / 10000; // 5.56% of total

        let code_capacity = Self::bytes_to_entries(code_cache_size, CODE_CACHE_ENTRY_SIZE);
        let storage_capacity = Self::bytes_to_entries(storage_cache_size, STORAGE_CACHE_ENTRY_SIZE);
        let account_capacity = Self::bytes_to_entries(account_cache_size, ACCOUNT_CACHE_ENTRY_SIZE);

        let code_stats = Arc::new(CacheStatsHandler::new(code_capacity));
        let storage_stats = Arc::new(CacheStatsHandler::new(storage_capacity));
        let account_stats = Arc::new(CacheStatsHandler::new(account_capacity));

        Self(Arc::new(ExecutionCacheInner {
            code_cache: FixedCache::new(code_capacity, FbBuildHasher::<32>::default())
                .with_stats(Some(Stats::new(code_stats.clone()))),
            storage_cache: FixedCache::new(storage_capacity, DefaultHashBuilder::default())
                .with_stats(Some(Stats::new(storage_stats.clone()))),
            account_cache: FixedCache::new(account_capacity, FbBuildHasher::<20>::default())
                .with_stats(Some(Stats::new(account_stats.clone()))),
            code_stats,
            storage_stats,
            account_stats,
            selfdestruct_encountered: Once::new(),
            readiness: parking_lot::Mutex::new(None),
        }))
    }

    /// Starts a fresh diagnostic generation for a cache checkout.
    pub fn start_readiness(&self) {
        if reth_tracing::readiness::enabled() {
            *self.0.readiness.lock() = Some(ReadinessDiagnostics::new());
        }
    }

    fn readiness(&self) -> Option<Arc<ReadinessDiagnostics>> {
        if !reth_tracing::readiness::enabled() {
            return None
        }
        self.0.readiness.try_lock().and_then(|diagnostics| diagnostics.clone())
    }

    /// Returns the shared prewarm read totals for this checkout.
    pub fn prewarm_read_totals(&self) -> Option<Arc<ReadTotals>> {
        self.readiness().map(|diagnostics| Arc::clone(&diagnostics.prewarm_totals))
    }

    /// Starts diagnostics for a receiver transaction when it is dispatched to the prewarm pool.
    fn prewarm_dispatch(&self) -> Option<PrewarmDispatchGuard> {
        self.readiness().map(|diagnostics| diagnostics.prewarm_dispatch())
    }

    /// Emits the bounded per-checkout cache diagnostics and prewarm read totals.
    pub fn emit_readiness(&self, parent: &Span, checkout_reason: u64) {
        if let Some(diagnostics) = self.readiness() {
            diagnostics.emit_once(parent, checkout_reason);
        }
    }

    fn readiness_reporter(
        &self,
        parent: Span,
        checkout_reason: u64,
    ) -> Option<Arc<ReadinessReporter>> {
        let diagnostics = self.0.readiness.try_lock()?.clone()?;
        Some(Arc::new(ReadinessReporter { diagnostics, checkout_reason, parent }))
    }

    /// Returns the number of active handles to the shared cache.
    fn usage_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Gets code from cache, or inserts using the provided function.
    pub fn get_or_try_insert_code_with<E>(
        &self,
        hash: B256,
        f: impl FnOnce() -> Result<Option<Bytecode>, E>,
    ) -> Result<CachedStatus<Option<Bytecode>>, E> {
        let mut miss = false;
        let result = self.0.code_cache.get_or_try_insert_with(hash, |_| {
            miss = true;
            f()
        })?;

        if miss {
            Ok(CachedStatus::NotCached(result))
        } else {
            Ok(CachedStatus::Cached(result))
        }
    }

    /// Gets storage from cache, or inserts using the provided function.
    pub fn get_or_try_insert_storage_with<E>(
        &self,
        address: Address,
        key: StorageKey,
        f: impl FnOnce() -> Result<StorageValue, E>,
    ) -> Result<CachedStatus<StorageValue>, E> {
        let mut miss = false;
        let result = self.0.storage_cache.get_or_try_insert_with((address, key), |_| {
            miss = true;
            f()
        })?;

        if miss {
            Ok(CachedStatus::NotCached(result))
        } else {
            Ok(CachedStatus::Cached(result))
        }
    }

    /// Gets account from cache, or inserts using the provided function.
    pub fn get_or_try_insert_account_with<E>(
        &self,
        address: Address,
        f: impl FnOnce() -> Result<Option<Account>, E>,
    ) -> Result<CachedStatus<Option<Account>>, E> {
        let mut miss = false;
        let result = self.0.account_cache.get_or_try_insert_with(address, |_| {
            miss = true;
            f()
        })?;

        if miss {
            Ok(CachedStatus::NotCached(result))
        } else {
            Ok(CachedStatus::Cached(result))
        }
    }

    /// Insert storage value into cache.
    pub fn insert_storage(&self, address: Address, key: StorageKey, value: Option<StorageValue>) {
        self.0.storage_cache.insert((address, key), value.unwrap_or_default());
    }

    /// Insert code into cache.
    pub fn insert_code(&self, hash: B256, code: Option<Bytecode>) {
        self.0.code_cache.insert(hash, code);
    }

    /// Insert account into cache.
    pub fn insert_account(&self, address: Address, account: Option<Account>) {
        self.0.account_cache.insert(address, account);
    }

    /// Inserts the post-execution state changes into the cache.
    ///
    /// This method is called after transaction execution to update the cache with
    /// the touched and modified state. The insertion order is critical:
    ///
    /// 1. Bytecodes: Insert contract code first
    /// 2. Storage slots: Update storage values for each account
    /// 3. Accounts: Update account info (nonce, balance, code hash)
    ///
    /// ## Why This Order Matters
    ///
    /// Account information references bytecode via code hash. If we update accounts
    /// before bytecode, we might create cache entries pointing to non-existent code.
    /// The current order ensures cache consistency.
    ///
    /// ## Error Handling
    ///
    /// Returns an error if the state updates are inconsistent and should be discarded.
    #[instrument(level = "debug", target = "engine::caching", skip_all)]
    #[expect(clippy::result_unit_err)]
    pub fn insert_state(&self, state_updates: &BundleState) -> Result<(), ()> {
        let _enter =
            debug_span!(target: "engine::tree", "contracts", len = state_updates.contracts.len())
                .entered();
        // Insert bytecodes
        for (code_hash, bytecode) in &state_updates.contracts {
            self.insert_code(*code_hash, Some(Bytecode(bytecode.clone())));
        }
        drop(_enter);

        let _enter = debug_span!(
            target: "engine::tree",
            "accounts",
            accounts = state_updates.state.len(),
            storages =
                state_updates.state.values().map(|account| account.storage.len()).sum::<usize>()
        )
        .entered();
        for (addr, account) in &state_updates.state {
            // If the account was not modified, as in not changed and not destroyed, then we have
            // nothing to do w.r.t. this particular account and can move on
            if account.status.is_not_modified() {
                continue
            }

            // If the original account had code (was a contract), we must clear the entire cache
            // because we can't efficiently invalidate all storage slots for a single address.
            // This should only happen on pre-Dencun networks.
            //
            // If the original account had no code (was an EOA or a not yet deployed contract), we
            // just remove the account from cache - no storage exists for it.
            if account.was_destroyed() {
                let had_code =
                    account.original_info.as_ref().is_some_and(|info| !info.is_empty_code_hash());
                if had_code {
                    self.0.selfdestruct_encountered.call_once(|| {
                        warn!(
                            target: "engine::caching",
                            address = ?addr,
                            info = ?account.info,
                            original_info = ?account.original_info,
                            "Encountered an inter-transaction SELFDESTRUCT that reset the storage cache. Are you running a pre-Dencun network?"
                        );
                    });
                    self.clear();
                    return Ok(())
                }

                self.0.account_cache.remove(addr);
                continue;
            }

            // If we have an account that was modified, but it has a `None` account info, some wild
            // error has occurred because this state should be unrepresentable. An account with
            // `None` current info, should be destroyed.
            let Some(ref account_info) = account.info else {
                trace!(target: "engine::caching", ?account, "Account with None account info found in state updates");
                return Err(())
            };

            // Now we iterate over all storage and make updates to the cached storage values
            for (key, slot) in &account.storage {
                self.insert_storage(*addr, (*key).into(), Some(slot.present_value));
            }

            // Insert will update if present, so we just use the new account info as the new value
            // for the account cache
            self.insert_account(*addr, Some(Account::from(account_info)));
        }

        Ok(())
    }

    /// Clears storage and account caches, resetting them to empty state.
    ///
    /// We do not clear the bytecodes cache, because its mapping can never change, as it's
    /// `keccak256(bytecode) => bytecode`.
    pub fn clear(&self) {
        self.0.storage_cache.clear();
        self.0.account_cache.clear();

        self.0.storage_stats.reset_size();
        self.0.account_stats.reset_size();
    }

    /// Updates the provided metrics with the current stats from the cache's stats handlers,
    /// and resets the hit/miss/collision counters.
    pub fn update_metrics(&self, metrics: &CachedStateCacheMetrics) {
        metrics.code_cache_size.set(self.0.code_stats.size() as f64);
        metrics.code_cache_capacity.set(self.0.code_stats.capacity() as f64);
        metrics.code_cache_collisions.set(self.0.code_stats.collisions() as f64);
        self.0.code_stats.reset_stats();

        metrics.storage_cache_size.set(self.0.storage_stats.size() as f64);
        metrics.storage_cache_capacity.set(self.0.storage_stats.capacity() as f64);
        metrics.storage_cache_collisions.set(self.0.storage_stats.collisions() as f64);
        self.0.storage_stats.reset_stats();

        metrics.account_cache_size.set(self.0.account_stats.size() as f64);
        metrics.account_cache_capacity.set(self.0.account_stats.capacity() as f64);
        metrics.account_cache_collisions.set(self.0.account_stats.collisions() as f64);
        self.0.account_stats.reset_stats();
    }
}

/// Diagnostics-only checkout lifetime guard.
///
/// Clones retain only block-local diagnostic counters and the reporting span. The execution cache
/// itself is deliberately not retained, so late diagnostic reporting cannot delay cache handoff.
pub struct ReadinessReporter {
    diagnostics: Arc<ReadinessDiagnostics>,
    checkout_reason: u64,
    parent: Span,
}

impl fmt::Debug for ReadinessReporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadinessReporter")
            .field("checkout_reason", &self.checkout_reason)
            .finish_non_exhaustive()
    }
}

impl Drop for ReadinessReporter {
    fn drop(&mut self) {
        self.diagnostics.emit_once(&self.parent, self.checkout_reason);
    }
}

/// A saved cache that has been used for executing a specific block, which has been updated for its
/// execution.
#[derive(Debug, Clone)]
pub struct SavedCache {
    /// The hash of the block these caches were used to execute.
    hash: B256,

    /// The caches used for the provider.
    caches: ExecutionCache,

    /// Checkout reason for this block's diagnostic generation.
    checkout_reason: CacheCheckoutReason,
}

impl SavedCache {
    /// Creates a new instance with the internals
    pub const fn new(hash: B256, caches: ExecutionCache) -> Self {
        Self { hash, caches, checkout_reason: CacheCheckoutReason::FreshAbsent }
    }

    /// Starts a fresh, block-local readiness generation after an exclusive checkout.
    pub fn begin_readiness_checkout(&mut self, reason: CacheCheckoutReason) {
        self.checkout_reason = reason;
        self.caches.start_readiness();
    }

    /// Returns the shared prewarm read totals for this checkout.
    pub fn prewarm_read_totals(&self) -> Option<Arc<ReadTotals>> {
        self.caches.prewarm_read_totals()
    }

    /// Starts a diagnostics-only receiver prewarm dispatch token.
    ///
    /// The returned token retains only block-local counters, not the execution cache handle.
    pub fn readiness_prewarm_dispatch(&self) -> Option<PrewarmDispatchGuard> {
        self.caches.prewarm_dispatch()
    }

    /// Emits this checkout's bounded readiness summary.
    pub fn emit_readiness(&self, parent: &Span) {
        self.caches.emit_readiness(parent, self.checkout_reason.as_u64());
    }

    /// Returns a diagnostics-only finalizer for this checkout.
    ///
    /// The summary is emitted once when the last reporter clone is dropped. The reporter does not
    /// retain the execution cache, so it does not affect cache availability or handoff.
    pub fn readiness_reporter(&self, parent: Span) -> Option<Arc<ReadinessReporter>> {
        self.caches.readiness_reporter(parent, self.checkout_reason.as_u64())
    }

    /// Returns the hash for this cache
    pub const fn executed_block_hash(&self) -> B256 {
        self.hash
    }

    /// Returns true if the cache is available for use (no other tasks are currently using it).
    pub fn is_available(&self) -> bool {
        self.caches.usage_count() == 1
    }

    /// Returns the current number of active handles to the shared cache.
    pub fn usage_count(&self) -> usize {
        self.caches.usage_count()
    }

    /// Returns the [`ExecutionCache`] belonging to the tracked hash.
    pub const fn cache(&self) -> &ExecutionCache {
        &self.caches
    }

    /// Updates the cache metrics (size/capacity/collisions) from the stats handlers.
    pub fn update_metrics(&self, metrics: Option<&CachedStateCacheMetrics>) {
        if let Some(metrics) = metrics {
            self.caches.update_metrics(metrics);
        }
    }

    /// Clears all caches, resetting them to empty state,
    /// and updates the hash of the block this cache belongs to.
    pub fn clear_with_hash(&mut self, hash: B256) {
        self.hash = hash;
        self.caches.clear();
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl SavedCache {
    /// Clones the cache handle that acts as the availability guard.
    pub fn clone_guard_for_test(&self) -> ExecutionCache {
        self.caches.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{map::HashMap, U256};
    use reth_provider::test_utils::{ExtendedAccount, MockEthProvider};
    use reth_revm::db::{AccountStatus, BundleAccount};
    use revm::state::AccountInfo;

    fn storage_keys_for_shard(
        diagnostics: &ReadinessDiagnostics,
        shard: usize,
        count: usize,
    ) -> Vec<ReadinessKey> {
        (0..u64::MAX)
            .map(|slot| ReadinessKey::Storage(Address::ZERO, U256::from(slot).into()))
            .filter(|key| diagnostics.shard_index(*key) == shard)
            .take(count)
            .collect()
    }

    #[test]
    fn readiness_correlates_inflight_and_completed_prewarm_reads() {
        let diagnostics = ReadinessDiagnostics::new();
        let key = ReadinessKey::Account(Address::with_last_byte(1));
        let mut first = diagnostics.begin_prewarm(key);
        let mut second = diagnostics.begin_prewarm(key);

        diagnostics.record_miss(key);
        assert_eq!(diagnostics.account_misses.inflight.load(Ordering::Relaxed), 1);

        first.finish_success();
        drop(first);
        diagnostics.record_miss(key);
        assert_eq!(diagnostics.account_misses.completed.load(Ordering::Relaxed), 1);

        second.finish_success();
        drop(second);
        diagnostics.record_miss(key);
        assert_eq!(diagnostics.account_misses.completed.load(Ordering::Relaxed), 2);

        let failed = ReadinessKey::Account(Address::with_last_byte(2));
        drop(diagnostics.begin_prewarm(failed));
        diagnostics.record_miss(failed);
        assert_eq!(diagnostics.account_misses.failed.load(Ordering::Relaxed), 1);

        diagnostics.record_miss(ReadinessKey::Account(Address::with_last_byte(3)));
        assert_eq!(diagnostics.account_misses.never_observed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn readiness_key_capacity_reports_unknown_without_exporting_keys() {
        let diagnostics = ReadinessDiagnostics::new();
        let private_address = Address::repeat_byte(0xaa);
        let overflow = ReadinessKey::Account(private_address);
        let full_shard = diagnostics.shard_index(overflow);
        for key in storage_keys_for_shard(&diagnostics, full_shard, READINESS_KEYS_PER_SHARD) {
            drop(diagnostics.begin_prewarm(key));
        }

        drop(diagnostics.begin_prewarm(overflow));
        diagnostics.record_miss(overflow);

        assert_eq!(READINESS_KEY_CAPACITY, 65_536);
        assert_eq!(
            diagnostics.shards[full_shard].keys.lock().entries.len(),
            READINESS_KEYS_PER_SHARD
        );
        assert!(diagnostics.shards[full_shard].cap_reached.load(Ordering::Acquire));
        assert_eq!(diagnostics.account_misses.unknown_due_cap.load(Ordering::Relaxed), 1);
        assert!(!format!("{diagnostics:?}").contains(&format!("{private_address:?}")));
    }

    #[test]
    fn readiness_contention_invalidates_only_its_shard() {
        let diagnostics = ReadinessDiagnostics::new();
        let blocked_shard = 0;
        let same_shard = storage_keys_for_shard(&diagnostics, blocked_shard, 2);
        let other_shard = storage_keys_for_shard(&diagnostics, 1, 1)[0];
        let keys = diagnostics.shards[blocked_shard].keys.lock();
        drop(diagnostics.begin_prewarm(same_shard[0]));
        drop(keys);

        diagnostics.record_miss(same_shard[1]);
        diagnostics.record_miss(other_shard);
        assert_eq!(diagnostics.storage_misses.unknown_contention.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostics.storage_misses.never_observed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn readiness_lost_begin_preserves_known_completed_key() {
        let diagnostics = ReadinessDiagnostics::new();
        let same_shard = storage_keys_for_shard(&diagnostics, 0, 3);
        let completed = same_shard[0];
        let mut completed_guard = diagnostics.begin_prewarm(completed);
        completed_guard.finish_success();
        drop(completed_guard);

        let keys = diagnostics.shards[0].keys.lock();
        drop(diagnostics.begin_prewarm(same_shard[1]));
        drop(keys);

        diagnostics.record_miss(completed);
        diagnostics.record_miss(same_shard[2]);
        assert_eq!(diagnostics.storage_misses.completed.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostics.storage_misses.unknown_contention.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn readiness_guard_completes_while_key_map_is_locked() {
        let diagnostics = ReadinessDiagnostics::new();
        let key = ReadinessKey::Account(Address::with_last_byte(1));
        let mut guard = diagnostics.begin_prewarm(key);
        let keys = diagnostics.shard(key).keys.lock();
        let state = Arc::clone(keys.entries.get(&key).expect("tracked key"));

        guard.finish_success();
        drop(guard);

        assert_eq!(state.inflight.load(Ordering::Acquire), 0);
        assert!(state.completed.load(Ordering::Acquire));
        drop(keys);
    }

    #[test]
    fn readiness_tracks_exact_keys_within_a_shard() {
        let diagnostics = ReadinessDiagnostics::new();
        let keys = storage_keys_for_shard(&diagnostics, 0, 2);
        let mut guard = diagnostics.begin_prewarm(keys[0]);
        guard.finish_success();
        drop(guard);

        diagnostics.record_miss(keys[0]);
        diagnostics.record_miss(keys[1]);

        assert_eq!(diagnostics.storage_misses.completed.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostics.storage_misses.never_observed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn readiness_storage_backing_latencies_partition_miss_classifications() {
        let diagnostics = ReadinessDiagnostics::new();
        let keys = storage_keys_for_shard(&diagnostics, 0, 5);
        let ReadinessKey::Storage(account, inflight_key) = keys[0] else { unreachable!() };
        let inflight = diagnostics.begin_prewarm(keys[0]);
        drop(diagnostics.begin_storage_backing_read(account, inflight_key));
        drop(inflight);

        let ReadinessKey::Storage(account, completed_key) = keys[1] else { unreachable!() };
        let mut completed = diagnostics.begin_prewarm(keys[1]);
        completed.finish_success();
        drop(completed);
        drop(diagnostics.begin_storage_backing_read(account, completed_key));

        let ReadinessKey::Storage(account, failed_key) = keys[2] else { unreachable!() };
        drop(diagnostics.begin_prewarm(keys[2]));
        drop(diagnostics.begin_storage_backing_read(account, failed_key));

        let ReadinessKey::Storage(account, never_key) = keys[3] else { unreachable!() };
        drop(diagnostics.begin_storage_backing_read(account, never_key));

        let ReadinessKey::Storage(account, unknown_key) = keys[4] else { unreachable!() };
        let shard_lock = diagnostics.shard(keys[4]).keys.lock();
        drop(diagnostics.begin_storage_backing_read(account, unknown_key));
        drop(shard_lock);

        let cap_key = storage_keys_for_shard(&diagnostics, 1, 1)[0];
        diagnostics.shards[1].cap_reached.store(true, Ordering::Release);
        let ReadinessKey::Storage(account, cap_storage_key) = cap_key else { unreachable!() };
        drop(diagnostics.begin_storage_backing_read(account, cap_storage_key));

        let latency = &diagnostics.storage_backing_latency;
        assert_eq!(latency.inflight.count.load(Ordering::Relaxed), 1);
        assert_eq!(latency.completed.count.load(Ordering::Relaxed), 1);
        assert_eq!(latency.failed.count.load(Ordering::Relaxed), 1);
        assert_eq!(latency.never_observed.count.load(Ordering::Relaxed), 1);
        assert_eq!(latency.unknown.count.load(Ordering::Relaxed), 2);

        let latency_count = [
            &latency.inflight,
            &latency.completed,
            &latency.failed,
            &latency.never_observed,
            &latency.unknown,
        ]
        .into_iter()
        .map(|counter| counter.count.load(Ordering::Relaxed))
        .sum::<u64>();
        let classified_count = diagnostics.storage_misses.inflight.load(Ordering::Relaxed) +
            diagnostics.storage_misses.completed.load(Ordering::Relaxed) +
            diagnostics.storage_misses.failed.load(Ordering::Relaxed) +
            diagnostics.storage_misses.never_observed.load(Ordering::Relaxed) +
            diagnostics.storage_misses.unknown_due_cap.load(Ordering::Relaxed) +
            diagnostics.storage_misses.unknown_contention.load(Ordering::Relaxed);
        assert_eq!(latency_count, classified_count);
    }

    #[test]
    fn readiness_unfinished_storage_read_records_age_overlap_and_first_finisher() {
        let diagnostics = ReadinessDiagnostics::new();
        let key = storage_keys_for_shard(&diagnostics, 0, 1)[0];
        let ReadinessKey::Storage(account, storage_key) = key else { unreachable!() };
        let mut prewarm = diagnostics.begin_prewarm(key);
        let mut canonical = diagnostics.begin_storage_backing_read(account, storage_key);

        prewarm.finish_success();
        drop(prewarm);
        canonical.finish(true);
        drop(canonical);

        let unfinished = &diagnostics.storage_unfinished;
        assert_eq!(unfinished.age.count.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.single.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.multiple.load(Ordering::Relaxed), 0);
        assert_eq!(unfinished.overlap.count.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.prewarm_success_first.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.canonical_success_first.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn readiness_unfinished_storage_read_records_canonical_success_and_failure_first() {
        let diagnostics = ReadinessDiagnostics::new();
        let keys = storage_keys_for_shard(&diagnostics, 0, 2);

        let ReadinessKey::Storage(account, storage_key) = keys[0] else { unreachable!() };
        let prewarm = diagnostics.begin_prewarm(keys[0]);
        let mut canonical = diagnostics.begin_storage_backing_read(account, storage_key);
        canonical.finish(true);
        drop(canonical);
        drop(prewarm);

        let ReadinessKey::Storage(account, storage_key) = keys[1] else { unreachable!() };
        let prewarm = diagnostics.begin_prewarm(keys[1]);
        drop(diagnostics.begin_storage_backing_read(account, storage_key));
        drop(prewarm);

        let unfinished = &diagnostics.storage_unfinished;
        assert_eq!(unfinished.age.count.load(Ordering::Relaxed), 2);
        assert_eq!(unfinished.overlap.count.load(Ordering::Relaxed), 2);
        assert_eq!(unfinished.canonical_success_first.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.canonical_failed_first.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn readiness_unfinished_storage_read_captures_exact_cohort_and_earliest_failure() {
        let diagnostics = ReadinessDiagnostics::new();
        let key = storage_keys_for_shard(&diagnostics, 0, 1)[0];
        let ReadinessKey::Storage(account, storage_key) = key else { unreachable!() };
        let first = diagnostics.begin_prewarm(key);
        let mut second = diagnostics.begin_prewarm(key);
        let mut canonical = diagnostics.begin_storage_backing_read(account, storage_key);

        drop(first);
        second.finish_success();
        drop(second);
        canonical.finish(true);
        drop(canonical);

        let unfinished = &diagnostics.storage_unfinished;
        assert_eq!(unfinished.multiple.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.prewarm_failed_first.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.prewarm_success_first.load(Ordering::Relaxed), 0);

        // A retry started after the miss snapshot is outside its fixed cohort and cannot win it.
        let original = diagnostics.begin_prewarm(key);
        let mut canonical = diagnostics.begin_storage_backing_read(account, storage_key);
        let mut later_retry = diagnostics.begin_prewarm(key);
        later_retry.finish_success();
        drop(later_retry);
        canonical.finish(true);
        drop(canonical);
        drop(original);
        assert_eq!(unfinished.canonical_success_first.load(Ordering::Relaxed), 1);

        let winners = unfinished.prewarm_success_first.load(Ordering::Relaxed) +
            unfinished.prewarm_failed_first.load(Ordering::Relaxed) +
            unfinished.canonical_success_first.load(Ordering::Relaxed) +
            unfinished.canonical_failed_first.load(Ordering::Relaxed) +
            unfinished.winner_unknown.load(Ordering::Relaxed);
        assert_eq!(unfinished.age.count.load(Ordering::Relaxed), 2);
        assert_eq!(unfinished.overlap.count.load(Ordering::Relaxed), winners);
    }

    #[test]
    fn readiness_unfinished_storage_timing_contention_is_unknown() {
        let diagnostics = ReadinessDiagnostics::new();
        let key = storage_keys_for_shard(&diagnostics, 0, 1)[0];
        let ReadinessKey::Storage(account, storage_key) = key else { unreachable!() };
        let prewarm = diagnostics.begin_prewarm(key);
        let keys = diagnostics.shard(key).keys.lock();
        let state = Arc::clone(keys.entries.get(&key).expect("tracked key"));
        let active_reads = state.active_reads.lock();
        drop(keys);

        drop(diagnostics.begin_storage_backing_read(account, storage_key));

        let unfinished = &diagnostics.storage_unfinished;
        assert_eq!(unfinished.unknown.load(Ordering::Relaxed), 1);
        assert_eq!(unfinished.age.count.load(Ordering::Relaxed), 0);
        assert_eq!(unfinished.overlap.count.load(Ordering::Relaxed), 0);
        assert_eq!(unfinished.winner_unknown.load(Ordering::Relaxed), 0);
        drop(active_reads);
        drop(prewarm);
    }

    #[test]
    fn readiness_lost_timing_registration_stays_conservatively_unknown() {
        let diagnostics = ReadinessDiagnostics::new();
        let key = storage_keys_for_shard(&diagnostics, 0, 1)[0];
        let ReadinessKey::Storage(account, storage_key) = key else { unreachable!() };
        let first = diagnostics.begin_prewarm(key);
        let keys = diagnostics.shard(key).keys.lock();
        let state = Arc::clone(keys.entries.get(&key).expect("tracked key"));
        let active_reads = state.active_reads.lock();
        drop(keys);

        let untracked = diagnostics.begin_prewarm(key);
        drop(active_reads);
        drop(diagnostics.begin_storage_backing_read(account, storage_key));

        assert!(state.timing_coverage_lost.load(Ordering::Acquire));
        assert_eq!(diagnostics.storage_unfinished.unknown.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostics.storage_unfinished.age.count.load(Ordering::Relaxed), 0);
        drop(untracked);
        drop(first);
    }

    #[test]
    fn readiness_finishing_publication_races_are_unknown() {
        let diagnostics = ReadinessDiagnostics::new();
        let keys = storage_keys_for_shard(&diagnostics, 0, 2);

        // A read already publishing its finish timestamp at the miss boundary cannot be placed
        // precisely on either side of that boundary.
        let prewarm = diagnostics.begin_prewarm(keys[0]);
        prewarm
            .instance
            .as_ref()
            .expect("timed prewarm read")
            .outcome
            .store(PREWARM_READ_FINISHING, Ordering::Release);
        let ReadinessKey::Storage(account, storage_key) = keys[0] else { unreachable!() };
        drop(diagnostics.begin_storage_backing_read(account, storage_key));
        assert_eq!(diagnostics.storage_unfinished.unknown.load(Ordering::Relaxed), 1);
        drop(prewarm);

        // The same publication interval after a known-active snapshot makes only the winner
        // unknown; the age and multiplicity at the snapshot remain valid.
        let prewarm = diagnostics.begin_prewarm(keys[1]);
        let ReadinessKey::Storage(account, storage_key) = keys[1] else { unreachable!() };
        let canonical = diagnostics.begin_storage_backing_read(account, storage_key);
        prewarm
            .instance
            .as_ref()
            .expect("timed prewarm read")
            .outcome
            .store(PREWARM_READ_FINISHING, Ordering::Release);
        drop(canonical);
        assert_eq!(diagnostics.storage_unfinished.age.count.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostics.storage_unfinished.winner_unknown.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostics.storage_unfinished.overlap.count.load(Ordering::Relaxed), 0);
        drop(prewarm);
    }

    #[test]
    fn readiness_prewarm_queue_counts_delay_lead_and_lifetimes() {
        let diagnostics = ReadinessDiagnostics::new();
        let mut first = diagnostics.prewarm_dispatch();
        let second = diagnostics.prewarm_dispatch();
        assert_eq!(diagnostics.prewarm_queue.queued.load(Ordering::Relaxed), 2);
        assert_eq!(diagnostics.prewarm_queue.queued_max.load(Ordering::Relaxed), 2);
        assert_eq!(diagnostics.prewarm_queue.outstanding_max.load(Ordering::Relaxed), 2);
        first.start(4, 5);
        assert_eq!(diagnostics.prewarm_queue.running.load(Ordering::Relaxed), 1);
        drop(first);
        drop(second);

        for (index, executed) in [(5, 5), (6, 5), (22, 5), (70, 5)] {
            let mut dispatch = diagnostics.prewarm_dispatch();
            dispatch.start(index, executed);
        }

        let queue = &diagnostics.prewarm_queue;
        assert_eq!(queue.delay.count.load(Ordering::Relaxed), 5);
        assert_eq!(queue.start_behind.load(Ordering::Relaxed), 1);
        assert_eq!(queue.start_current.load(Ordering::Relaxed), 1);
        assert_eq!(queue.start_ahead_1_16.load(Ordering::Relaxed), 1);
        assert_eq!(queue.start_ahead_17_64.load(Ordering::Relaxed), 1);
        assert_eq!(queue.start_ahead_gt_64.load(Ordering::Relaxed), 1);
        let histogram_count = queue.delay_lt_10us.load(Ordering::Relaxed) +
            queue.delay_lt_100us.load(Ordering::Relaxed) +
            queue.delay_lt_1ms.load(Ordering::Relaxed) +
            queue.delay_lt_10ms.load(Ordering::Relaxed) +
            queue.delay_ge_10ms.load(Ordering::Relaxed);
        assert_eq!(histogram_count, queue.delay.count.load(Ordering::Relaxed));
        assert_eq!(queue.queued.load(Ordering::Relaxed), 0);
        assert_eq!(queue.running.load(Ordering::Relaxed), 0);
        assert_eq!(queue.outstanding.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn readiness_reporter_emits_once_after_late_worker_without_retaining_cache() {
        let cache = ExecutionCache::new(1_000_000);
        let diagnostics = ReadinessDiagnostics::new();
        *cache.0.readiness.lock() = Some(Arc::clone(&diagnostics));
        let saved = SavedCache::new(B256::ZERO, cache);
        assert_eq!(saved.usage_count(), 1);

        let reporter = saved.readiness_reporter(Span::none()).expect("reporter");
        assert_eq!(saved.usage_count(), 1, "reporter must not retain the execution cache");
        let late_worker = Arc::clone(&reporter);
        drop(reporter);
        assert_eq!(diagnostics.emission_count.load(Ordering::Relaxed), 0);

        diagnostics.record_miss(ReadinessKey::Account(Address::with_last_byte(1)));
        drop(late_worker);
        assert_eq!(diagnostics.emission_count.load(Ordering::Relaxed), 1);
        assert_eq!(diagnostics.account_misses.never_observed.load(Ordering::Relaxed), 1);

        // A second finalizer for the same diagnostic generation remains idempotent.
        drop(saved.readiness_reporter(Span::none()));
        assert_eq!(diagnostics.emission_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_empty_storage_cached_state_provider() {
        let address = Address::random();
        let storage_key = StorageKey::random();
        let account = ExtendedAccount::new(0, U256::ZERO);

        let provider = MockEthProvider::default();
        provider.extend_accounts(vec![(address, account)]);

        let caches = ExecutionCache::new(1000);
        let state_provider = CachedStateProvider::new(
            provider,
            caches,
            Some(CachedStateMetrics::zeroed(CachedStateMetricsSource::Test)),
        );

        let res = state_provider.storage(address, storage_key);
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), None);
    }

    #[test]
    fn test_uncached_storage_cached_state_provider() {
        let address = Address::random();
        let storage_key = StorageKey::random();
        let storage_value = U256::from(1);
        let account =
            ExtendedAccount::new(0, U256::ZERO).extend_storage(vec![(storage_key, storage_value)]);

        let provider = MockEthProvider::default();
        provider.extend_accounts(vec![(address, account)]);

        let caches = ExecutionCache::new(1000);
        let state_provider = CachedStateProvider::new(
            provider,
            caches,
            Some(CachedStateMetrics::zeroed(CachedStateMetricsSource::Test)),
        );

        let res = state_provider.storage(address, storage_key);
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), Some(storage_value));
    }

    #[test]
    fn test_get_storage_populated() {
        let address = Address::random();
        let storage_key = StorageKey::random();
        let storage_value = U256::from(1);

        let caches = ExecutionCache::new(1000);
        caches.insert_storage(address, storage_key, Some(storage_value));

        let result = caches
            .get_or_try_insert_storage_with(address, storage_key, || Ok::<_, ()>(U256::from(999)));
        assert_eq!(result.unwrap(), CachedStatus::Cached(storage_value));
    }

    #[test]
    fn test_get_storage_empty() {
        let address = Address::random();
        let storage_key = StorageKey::random();

        let caches = ExecutionCache::new(1000);
        caches.insert_storage(address, storage_key, None);

        let result = caches
            .get_or_try_insert_storage_with(address, storage_key, || Ok::<_, ()>(U256::from(999)));
        assert_eq!(result.unwrap(), CachedStatus::Cached(U256::ZERO));
    }

    #[test]
    fn test_saved_cache_is_available() {
        let execution_cache = ExecutionCache::new(1000);
        let cache = SavedCache::new(B256::ZERO, execution_cache);

        assert!(cache.is_available(), "Cache should be available initially");

        let _cache = cache.clone_guard_for_test();

        assert!(!cache.is_available(), "Cache should not be available with active handle");
    }

    #[test]
    fn test_saved_cache_multiple_references() {
        let execution_cache = ExecutionCache::new(1000);
        let cache = SavedCache::new(B256::from([2u8; 32]), execution_cache);

        let cache1 = cache.clone_guard_for_test();
        let cache2 = cache.clone_guard_for_test();
        let cache3 = cache1.clone();

        assert!(!cache.is_available());

        drop(cache1);
        assert!(!cache.is_available());

        drop(cache2);
        assert!(!cache.is_available());

        drop(cache3);
        assert!(cache.is_available());
    }

    #[test]
    fn test_insert_state_destroyed_account_with_code_clears_cache() {
        let caches = ExecutionCache::new(1000);

        // Pre-populate caches with some data
        let addr1 = Address::random();
        let addr2 = Address::random();
        let storage_key = StorageKey::random();
        caches.insert_account(addr1, Some(Account::default()));
        caches.insert_account(addr2, Some(Account::default()));
        caches.insert_storage(addr1, storage_key, Some(U256::from(42)));

        // Verify caches are populated
        assert!(caches.0.account_cache.get(&addr1).is_some());
        assert!(caches.0.account_cache.get(&addr2).is_some());
        assert!(caches.0.storage_cache.get(&(addr1, storage_key)).is_some());

        let bundle = BundleState {
            // BundleState with a destroyed contract (had code)
            state: HashMap::from_iter([(
                Address::random(),
                BundleAccount::new(
                    Some(AccountInfo {
                        balance: U256::ZERO,
                        nonce: 1,
                        code_hash: B256::random(), // Non-empty code hash
                        code: None,
                        account_id: None,
                    }),
                    None, // Destroyed, so no current info
                    Default::default(),
                    AccountStatus::Destroyed,
                ),
            )]),
            contracts: Default::default(),
            reverts: Default::default(),
            state_size: 0,
            reverts_size: 0,
        };

        // Insert state should clear all caches because a contract was destroyed
        let result = caches.insert_state(&bundle);
        assert!(result.is_ok());

        // Verify all caches were cleared
        assert!(caches.0.account_cache.get(&addr1).is_none());
        assert!(caches.0.account_cache.get(&addr2).is_none());
        assert!(caches.0.storage_cache.get(&(addr1, storage_key)).is_none());
    }

    #[test]
    fn test_insert_state_destroyed_account_without_code_removes_only_account() {
        let caches = ExecutionCache::new(1000);

        // Pre-populate caches with some data
        let addr1 = Address::random();
        let addr2 = Address::random();
        let storage_key = StorageKey::random();
        caches.insert_account(addr1, Some(Account::default()));
        caches.insert_account(addr2, Some(Account::default()));
        caches.insert_storage(addr1, storage_key, Some(U256::from(42)));

        let bundle = BundleState {
            // BundleState with a destroyed EOA (no code)
            state: HashMap::from_iter([(
                addr1,
                BundleAccount::new(
                    Some(AccountInfo {
                        balance: U256::from(100),
                        nonce: 1,
                        code_hash: alloy_primitives::KECCAK256_EMPTY, // Empty code hash = EOA
                        code: None,
                        account_id: None,
                    }),
                    None, // Destroyed
                    Default::default(),
                    AccountStatus::Destroyed,
                ),
            )]),
            contracts: Default::default(),
            reverts: Default::default(),
            state_size: 0,
            reverts_size: 0,
        };

        // Insert state should only remove the destroyed account
        assert!(caches.insert_state(&bundle).is_ok());

        // Verify only addr1 was removed, other data is still present
        assert!(caches.0.account_cache.get(&addr1).is_none());
        assert!(caches.0.account_cache.get(&addr2).is_some());
        assert!(caches.0.storage_cache.get(&(addr1, storage_key)).is_some());
    }

    #[test]
    fn test_insert_state_destroyed_account_no_original_info_removes_only_account() {
        let caches = ExecutionCache::new(1000);

        // Pre-populate caches
        let addr1 = Address::random();
        let addr2 = Address::random();
        caches.insert_account(addr1, Some(Account::default()));
        caches.insert_account(addr2, Some(Account::default()));

        let bundle = BundleState {
            // BundleState with a destroyed account (has no original info)
            state: HashMap::from_iter([(
                addr1,
                BundleAccount::new(
                    None, // No original info
                    None, // Destroyed
                    Default::default(),
                    AccountStatus::Destroyed,
                ),
            )]),
            contracts: Default::default(),
            reverts: Default::default(),
            state_size: 0,
            reverts_size: 0,
        };

        // Insert state should only remove the destroyed account (no code = no full clear)
        assert!(caches.insert_state(&bundle).is_ok());

        // Verify only addr1 was removed
        assert!(caches.0.account_cache.get(&addr1).is_none());
        assert!(caches.0.account_cache.get(&addr2).is_some());
    }

    #[test]
    fn test_insert_state_destroyed_uncached_account_keeps_size_zero() {
        let caches = ExecutionCache::new(1000);
        assert_eq!(caches.0.account_stats.size(), 0);

        let addr = Address::random();
        let bundle = BundleState {
            state: HashMap::from_iter([(
                addr,
                BundleAccount::new(
                    None, // No original info
                    None, // Destroyed
                    Default::default(),
                    AccountStatus::Destroyed,
                ),
            )]),
            contracts: Default::default(),
            reverts: Default::default(),
            state_size: 0,
            reverts_size: 0,
        };

        assert!(caches.insert_state(&bundle).is_ok());
        assert_eq!(caches.0.account_stats.size(), 0);
        assert!(caches.0.account_cache.get(&addr).is_none());
    }

    #[test]
    fn test_code_cache_capacity_with_default_budget() {
        // Default cross-block cache is 4 GB; code gets 5.56% = ~228 MB.
        let total_cache_size = 4 * 1024 * 1024 * 1024; // 4 GB
        let code_budget = (total_cache_size * 556) / 10000; // 228 MB

        let capacity = ExecutionCache::bytes_to_entries(code_budget, CODE_CACHE_ENTRY_SIZE);

        // With ESTIMATED_AVG_CODE_SIZE (8 KiB) we expect 16384 entries.
        // If someone accidentally reverts to MAX_CODE_SIZE (48 KiB), this would drop to 4096.
        assert_eq!(
            capacity, 16384,
            "code cache should have 16384 entries with default 4 GB budget"
        );
    }
}
