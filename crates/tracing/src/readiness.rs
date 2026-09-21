//! Opt-in, bounded read diagnostics. Logical and database timings are inclusive;
//! nested classes must not be added as exclusive wall time. Keys never enter this API.

use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, Mutex, OnceLock},
};
use tracing::Span;

const CLASSES: usize = 10;
const SAMPLE_CAP: usize = 8;
const SLOW_NS: u64 = 100_000;

/// Whether the explicitly requested lifecycle diagnostic is enabled.
#[inline]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("TEMPO_READ_READINESS").is_ok_and(|v| v == "1") &&
            std::env::var_os("RETH_LIFECYCLE_FILE").is_some() &&
            epoch().is_some()
    })
}

fn epoch() -> Option<u64> {
    static EPOCH: OnceLock<Option<u64>> = OnceLock::new();
    *EPOCH.get_or_init(|| std::env::var("RETH_LIFECYCLE_EPOCH_NS").ok()?.parse().ok())
}

/// Closed numeric role vocabulary; no thread or process names are exported.
#[derive(Clone, Copy, Debug)]
#[repr(u64)]
pub enum Role {
    /// Authoritative serial execution.
    Execution = 1,
    /// Speculative prewarming.
    Prewarm = 2,
    /// Account proof worker.
    AccountProof = 3,
    /// Storage proof worker.
    StorageProof = 4,
    /// Serial payload-builder state reads, including setup and finalization.
    PayloadBuilder = 5,
}

/// Closed read classes. Database classes time MDBX access including encoding/decoding,
/// not exclusively disk I/O. Logical classes also include cache/overlay lookup.
#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub enum ReadClass {
    /// Logical account lookup.
    Account,
    /// Logical storage lookup.
    Storage,
    /// Logical bytecode lookup.
    Code,
    /// Plain or hashed account table.
    DbAccount,
    /// Plain or hashed storage table.
    DbStorage,
    /// Bytecode table.
    DbCode,
    /// Packed or legacy account trie table.
    DbAccountTrie,
    /// Packed or legacy storage trie table.
    DbStorageTrie,
    /// History or changeset table.
    DbHistory,
    /// Other database table.
    DbOther,
}

impl ReadClass {
    /// Classifies a compile-time table name without retaining or exporting that name.
    pub fn database(table: &str) -> Self {
        match table {
            "PlainAccountState" | "HashedAccounts" => Self::DbAccount,
            "PlainStorageState" | "HashedStorages" => Self::DbStorage,
            "Bytecodes" => Self::DbCode,
            "AccountsTrie" | "PackedAccountsTrie" => Self::DbAccountTrie,
            "StoragesTrie" | "PackedStoragesTrie" => Self::DbStorageTrie,
            "AccountsHistory" | "StoragesHistory" | "AccountChangeSets" | "StorageChangeSets" => {
                Self::DbHistory
            }
            _ => Self::DbOther,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    calls: u64,
    ns: u64,
    max_ns: u64,
    buckets: [u64; 5],
}

impl Counts {
    fn record(&mut self, ns: u64) {
        self.calls = self.calls.saturating_add(1);
        self.ns = self.ns.saturating_add(ns);
        self.max_ns = self.max_ns.max(ns);
        let bucket = match ns {
            0..10_000 => 0,
            10_000..100_000 => 1,
            100_000..1_000_000 => 2,
            1_000_000..10_000_000 => 3,
            _ => 4,
        };
        self.buckets[bucket] = self.buckets[bucket].saturating_add(1);
    }
    fn merge(&mut self, other: Self) {
        self.calls = self.calls.saturating_add(other.calls);
        self.ns = self.ns.saturating_add(other.ns);
        self.max_ns = self.max_ns.max(other.max_ns);
        for (a, b) in self.buckets.iter_mut().zip(other.buckets) {
            *a = a.saturating_add(b);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    class: usize,
    start: u64,
    end: u64,
    thread: u64,
}

#[derive(Debug, Default)]
struct Stats {
    counts: [Counts; CLASSES],
    samples: [Option<Sample>; SAMPLE_CAP],
    sample_count: usize,
    omitted: u64,
}

impl Stats {
    // Bounded examples selected in timer/task completion order, not a random or
    // chronological sample. Counts include all reads, including omitted samples.
    fn sample(&mut self, sample: Sample) {
        if self.sample_count < SAMPLE_CAP {
            self.samples[self.sample_count] = Some(sample);
            self.sample_count += 1;
        } else {
            self.omitted = self.omitted.saturating_add(1);
        }
    }
    fn merge(&mut self, other: Self) {
        for (a, b) in self.counts.iter_mut().zip(other.counts) {
            a.merge(b);
        }
        for sample in other.samples.into_iter().flatten() {
            self.sample(sample);
        }
        self.omitted = self.omitted.saturating_add(other.omitted);
    }
    fn emit(&self, parent: &Span, role: Role) {
        for (class, c) in self.counts.iter().enumerate().filter(|(_, c)| c.calls != 0) {
            tracing::info!(target: "lifecycle", parent: parent, stage="read_totals",
                read_role=role as u64, read_class=class as u64, read_calls=c.calls,
                read_ns=c.ns, read_max_ns=c.max_ns,
                read_lt_10us=c.buckets[0], read_lt_100us=c.buckets[1],
                read_lt_1ms=c.buckets[2], read_lt_10ms=c.buckets[3], read_ge_10ms=c.buckets[4]);
        }
        tracing::info!(target: "lifecycle", parent: parent, stage="read_samples",
            read_role=role as u64, read_samples_retained=self.sample_count as u64,
            read_samples_omitted=self.omitted, read_sample_cap=SAMPLE_CAP as u64);
        for sample in self.samples.iter().flatten() {
            tracing::info!(target: "lifecycle", parent: parent, stage="read_sample",
                read_role=role as u64, read_class=sample.class as u64,
                read_begin_ns=sample.start, read_end_ns=sample.end,
                read_thread=sample.thread);
        }
    }
}

/// Shared per-block totals for one role. Merge once per scoped task, never per read.
/// The owner must emit only after all scoped tasks complete.
#[derive(Debug, Default)]
pub struct ReadTotals(Mutex<Stats>);

impl ReadTotals {
    /// Creates a bounded accumulator for one block and one role.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Emits and drains totals once under the explicit block/worker parent.
    pub fn emit(&self, parent: &Span, role: Role) {
        let stats = std::mem::take(&mut *self.0.lock().unwrap_or_else(|p| p.into_inner()));
        stats.emit(parent, role);
    }
}

#[derive(Debug)]
struct Active {
    id: u64,
    stats: Stats,
}
thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
    static NEXT: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

/// A synchronous, thread-bound accounting scope. Nested scopes restore their parent.
#[derive(Debug)]
pub struct Scope {
    state: Option<(Option<Active>, Role, Span, Option<Arc<ReadTotals>>)>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Scope {
    /// Accounts one whole execution/worker call and emits once at its end.
    pub fn enter(role: Role) -> Self {
        Self::new(role, None, enabled())
    }

    /// Accounts a short task into shared block totals without emitting per-task events.
    pub fn enter_shared(role: Role, totals: Arc<ReadTotals>) -> Self {
        Self::new(role, Some(totals), enabled())
    }

    fn new(role: Role, totals: Option<Arc<ReadTotals>>, on: bool) -> Self {
        let state = on.then(|| {
            let id = NEXT.with(|next| {
                let id = next.get();
                next.set(id.wrapping_add(1));
                id
            });
            let previous = ACTIVE.with(|a| a.replace(Some(Active { id, stats: Stats::default() })));
            (previous, role, Span::current(), totals)
        });
        Self { state, _thread_bound: PhantomData }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let Some((previous, role, parent, shared)) = self.state.take() else { return };
        let active = ACTIVE.with(|a| a.replace(previous));
        if let Some(active) = active {
            if let Some(shared) = shared {
                shared.0.lock().unwrap_or_else(|p| p.into_inner()).merge(active.stats);
            } else {
                active.stats.emit(&parent, role);
            }
        }
    }
}

/// Measures a read inside an active scope without emitting an event for that read.
#[derive(Debug)]
pub struct ReadTimer {
    class: ReadClass,
    scope: u64,
    start: u64,
}

impl ReadTimer {
    /// Avoids even table classification on the disabled database hot path.
    #[inline]
    pub fn start_database(table: &str) -> Option<Self> {
        if !enabled() {
            return None
        }
        Self::start(ReadClass::database(table))
    }

    /// Returns `None` on the disabled path or outside a classified synchronous scope.
    #[inline]
    pub fn start(class: ReadClass) -> Option<Self> {
        if !enabled() {
            return None
        }
        let scope = ACTIVE.with(|a| a.borrow().as_ref().map(|a| a.id))?;
        Some(Self { class, scope, start: crate::lifecycle::monotonic_ns() })
    }
}

impl Drop for ReadTimer {
    fn drop(&mut self) {
        let end = crate::lifecycle::monotonic_ns();
        let elapsed = end.saturating_sub(self.start);
        ACTIVE.with(|a| {
            let mut a = a.borrow_mut();
            let Some(a) = a.as_mut().filter(|a| a.id == self.scope) else { return };
            a.stats.counts[self.class as usize].record(elapsed);
            if elapsed >= SLOW_NS {
                if let Some(epoch) = epoch() {
                    a.stats.sample(Sample {
                        class: self.class as usize,
                        start: self.start.saturating_sub(epoch),
                        end: end.saturating_sub(epoch),
                        thread: crate::lifecycle::thread_id(Some(epoch)),
                    });
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_reads_accumulate_and_histogram_boundaries_are_exact() {
        let mut c = Counts::default();
        for n in [0, 9_999, 10_000, 99_999, 100_000, 999_999, 1_000_000, 9_999_999, 10_000_000] {
            c.record(n);
        }
        assert_eq!(c.calls, 9);
        assert_eq!(c.buckets, [2, 2, 2, 2, 1]);
        assert_eq!(c.ns, 22_219_996);
        assert_eq!(c.max_ns, 10_000_000);
    }

    #[test]
    fn merged_samples_are_bounded_and_omissions_reconcile() {
        let mut a = Stats::default();
        let mut b = Stats::default();
        for n in 0..20 {
            b.sample(Sample { class: 1, start: n, end: n + 1, thread: 1 });
        }
        for n in 0..6 {
            a.sample(Sample { class: 1, start: n, end: n + 1, thread: 1 });
        }
        a.merge(b);
        assert_eq!(a.sample_count, 8);
        assert_eq!(a.omitted, 18);
    }

    #[test]
    fn nested_scopes_restore_parent_and_shared_scopes_do_not_mix() {
        let outer = ReadTotals::new();
        let inner = ReadTotals::new();
        {
            let _outer = Scope::new(Role::Execution, Some(outer.clone()), true);
            ACTIVE.with(|a| a.borrow_mut().as_mut().unwrap().stats.counts[0].record(5));
            {
                let _inner = Scope::new(Role::Prewarm, Some(inner.clone()), true);
                ACTIVE.with(|a| a.borrow_mut().as_mut().unwrap().stats.counts[0].record(7));
            }
            ACTIVE.with(|a| a.borrow_mut().as_mut().unwrap().stats.counts[0].record(11));
        }
        assert_eq!(outer.0.lock().unwrap().counts[0].ns, 16);
        assert_eq!(inner.0.lock().unwrap().counts[0].ns, 7);
        assert!(ACTIVE.with(|a| a.borrow().is_none()));
    }

    #[test]
    fn disabled_scope_does_not_replace_active_context() {
        let totals = ReadTotals::new();
        let _outer = Scope::new(Role::Execution, Some(totals), true);
        let id = ACTIVE.with(|a| a.borrow().as_ref().unwrap().id);
        {
            let _disabled = Scope::new(Role::Prewarm, None, false);
        }
        assert_eq!(ACTIVE.with(|a| a.borrow().as_ref().unwrap().id), id);
    }

    #[test]
    fn unknown_table_names_cannot_escape_the_closed_classification() {
        assert_eq!(ReadClass::database("private filename") as usize, ReadClass::DbOther as usize);
        assert_eq!(
            ReadClass::database("PackedStoragesTrie") as usize,
            ReadClass::DbStorageTrie as usize
        );
    }
}
