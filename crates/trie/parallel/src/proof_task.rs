//! Parallel proof computation using worker pools with dedicated database transactions.
//!
//!
//! # Architecture
//!
//! - **Worker Pools**: Pre-spawned workers with dedicated database transactions
//!   - Storage pool: Handles storage proofs
//!   - Account pool: Handles account multiproofs
//! - **Direct Channel Access**: `ProofWorkerHandle` provides type-safe queue methods with direct
//!   access to worker channels, eliminating routing overhead
//! - **Automatic Shutdown**: Workers terminate gracefully when all handles are dropped
//!
//! # Message Flow
//!
//! 1. The `SparseTrieCacheTask` prepares a storage or account job and hands it to
//!    `ProofWorkerHandle`. The job carries a `ProofResultContext` so the worker knows how to send
//!    the result back.
//! 2. A worker receives the job, runs the proof, and sends a `ProofResultMessage` through the
//!    provided `ProofResultSender`.
//! 3. The `SparseTrieCacheTask` receives the message and proceeds with its state-root logic.
//!
//! Each job gets its own direct channel so results go straight back to the `SparseTrieCacheTask`.
//! That keeps ordering decisions in one place and lets workers run independently.
//!
//! ```text
//! SparseTrieCacheTask -> ProofWorkerHandle -> Storage/Account Worker
//!        ^                       |
//!        |                       v
//! ProofResultMessage <-- ProofResultSender
//! ```

#[cfg(feature = "metrics")]
use crate::job_counts::JobKind;
use crate::{
    error::StateRootTaskError,
    job_counts::{JobCounts, JobSize},
    value_encoder::{AsyncAccountValueEncoder, ValueEncoderStats},
};
use alloy_primitives::{
    map::{B256Map, B256Set},
    B256, U256,
};
use crossbeam_channel::{unbounded, Receiver as CrossbeamReceiver, Sender as CrossbeamSender};
use reth_execution_errors::StateProofError;
use reth_primitives_traits::{dashmap::DashMap, FastInstant as Instant};
use reth_provider::{DatabaseProviderROFactory, ProviderError, ProviderResult};
use reth_storage_errors::db::DatabaseError;
use reth_tasks::Runtime;
use reth_trie::{
    hashed_cursor::{HashedCursorFactory, HashedStorageCursor, InstrumentedHashedCursor},
    proof_v2,
    trie_cursor::{InstrumentedTrieCursor, TrieCursorFactory, TrieStorageCursor},
    DecodedMultiProofV2, HashedPostState, MultiProofTargetsV2, ProofTrieNodeV2, ProofV2Target,
};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tracing::{debug, debug_span, error, instrument, trace};

#[cfg(feature = "metrics")]
use crate::proof_task_metrics::{
    ProofTaskCursorMetrics, ProofTaskCursorMetricsCache, ProofTaskTrieMetrics,
};

/// Type alias for the V2 account proof calculator with instrumented cursors.
type V2AccountProofCalculator<'a, Provider> = proof_v2::ProofCalculator<
    InstrumentedTrieCursor<'a, <Provider as TrieCursorFactory>::AccountTrieCursor<'a>>,
    InstrumentedHashedCursor<'a, <Provider as HashedCursorFactory>::AccountCursor<'a>>,
    AsyncAccountValueEncoder<
        InstrumentedTrieCursor<'a, <Provider as TrieCursorFactory>::StorageTrieCursor<'a>>,
        InstrumentedHashedCursor<'a, <Provider as HashedCursorFactory>::StorageCursor<'a>>,
    >,
>;

/// Type alias for the V2 storage proof calculator with instrumented cursors.
type V2StorageProofCalculator<'a, Provider> = proof_v2::StorageProofCalculator<
    InstrumentedTrieCursor<'a, <Provider as TrieCursorFactory>::StorageTrieCursor<'a>>,
    InstrumentedHashedCursor<'a, <Provider as HashedCursorFactory>::StorageCursor<'a>>,
>;

/// Tracks worker availability counts.
///
/// It uses cacheline-aligned flags to avoid core-to-core chatter.
#[derive(Debug)]
struct AvailabilitySheet {
    /// One flag per worker, each on its own cacheline. Workers store `true` when idle,
    /// `false` when busy. Only the owning worker writes; the dispatcher only reads.
    flags: Vec<crossbeam_utils::CachePadded<AtomicBool>>,
}

impl AvailabilitySheet {
    /// Creates a new sheet with `count` workers, all initially marked as busy.
    fn new(count: usize) -> Self {
        let flags =
            (0..count).map(|_| crossbeam_utils::CachePadded::new(AtomicBool::new(false))).collect();
        Self { flags }
    }

    /// Returns `true` if more than one worker is currently idle.
    ///
    /// Note, that this is somewhat racy since a flag that was just saying `idle` and we counted it
    /// as such might turn into `busy` right away.
    fn has_multiple_idle(&self) -> bool {
        let mut idle = 0u32;
        for flag in &self.flags {
            if flag.load(Ordering::Relaxed) {
                idle += 1;
                if idle > 1 {
                    return true;
                }
            }
        }
        false
    }

    /// Marks the given worker as idle.
    fn mark_idle(&self, worker_id: usize) {
        self.flags[worker_id].store(true, Ordering::Relaxed);
    }

    /// Marks the given worker as busy.
    fn mark_busy(&self, worker_id: usize) {
        self.flags[worker_id].store(false, Ordering::Relaxed);
    }
}

/// A handle that provides type-safe access to proof worker pools.
///
/// The handle stores direct senders to both storage and account worker pools,
/// eliminating the need for a routing thread. All handles share reference-counted
/// channels, and workers shut down gracefully when all handles are dropped.
#[derive(Debug, Clone)]
pub struct ProofWorkerHandle {
    /// Direct sender to storage worker pool
    storage_work_tx: CrossbeamSender<StorageWorkerJob>,
    /// Direct sender to account worker pool
    account_work_tx: CrossbeamSender<AccountWorkerJob>,
    /// Per-worker availability flags for storage workers. Used to determine whether to chunk
    /// multiproofs.
    storage_availability: Arc<AvailabilitySheet>,
    /// Per-worker availability flags for account workers. Used to determine whether to chunk
    /// multiproofs.
    account_availability: Arc<AvailabilitySheet>,
    /// Total number of storage workers spawned
    storage_worker_count: usize,
    /// Total number of account workers spawned
    account_worker_count: usize,
}

impl ProofWorkerHandle {
    /// Spawns storage and account worker pools with dedicated database transactions.
    ///
    /// Returns a handle for submitting proof tasks to the worker pools.
    /// Workers run until the last handle is dropped.
    ///
    /// # Parameters
    /// - `runtime`: The centralized runtime used to spawn blocking worker tasks
    /// - `task_ctx`: Shared context with database view and prefix sets
    /// - `halve_workers`: Whether to halve the worker pool size (for small blocks)
    #[instrument(
        name = "ProofWorkerHandle::new",
        level = "debug",
        target = "trie::proof_task",
        skip_all
    )]
    pub fn new<Factory>(
        runtime: &Runtime,
        task_ctx: ProofTaskCtx<Factory>,
        halve_workers: bool,
        proof_result_tx: ProofResultSender,
    ) -> Self
    where
        Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let (storage_work_tx, storage_work_rx) = unbounded::<StorageWorkerJob>();
        let (account_work_tx, account_work_rx) = unbounded::<AccountWorkerJob>();
        let cached_storage_roots = Arc::<DashMap<_, _>>::default();

        let divisor = if halve_workers { 2 } else { 1 };
        let storage_worker_count =
            runtime.proof_storage_worker_pool().current_num_threads() / divisor;
        let account_worker_count =
            runtime.proof_account_worker_pool().current_num_threads() / divisor;

        let storage_availability = Arc::new(AvailabilitySheet::new(storage_worker_count));
        let account_availability = Arc::new(AvailabilitySheet::new(account_worker_count));

        debug!(
            target: "trie::proof_task",
            storage_worker_count,
            account_worker_count,
            halve_workers,
            "Spawning proof worker pools"
        );

        // broadcast blocks until all workers exit (channel close), so run on
        // tokio's blocking pool.
        let storage_rt = runtime.clone();
        let storage_task_ctx = task_ctx.clone();
        let storage_avail = storage_availability.clone();
        let storage_roots = cached_storage_roots.clone();
        let storage_result_tx = proof_result_tx.clone();
        let storage_parent_span = tracing::Span::current();
        runtime.spawn_blocking_named("storage-workers", move || {
            let worker_id = AtomicUsize::new(0);
            storage_rt.proof_storage_worker_pool().broadcast(storage_worker_count, |_| {
                let worker_id = worker_id.fetch_add(1, Ordering::Relaxed);
                let span = debug_span!(target: "trie::proof_task", parent: storage_parent_span.clone(), "storage_worker", ?worker_id);
                let _guard = span.enter();

                #[cfg(feature = "metrics")]
                let metrics = ProofTaskTrieMetrics::default();
                #[cfg(feature = "metrics")]
                let cursor_metrics = ProofTaskCursorMetrics::new();

                let worker = StorageProofWorker::new(
                    storage_task_ctx.clone(),
                    storage_work_rx.clone(),
                    worker_id,
                    storage_avail.clone(),
                    storage_roots.clone(),
                    #[cfg(feature = "metrics")]
                    metrics,
                    #[cfg(feature = "metrics")]
                    cursor_metrics,
                );
                #[cfg(feature = "metrics")]
                let cpu_timer = crate::worker_cpu::WorkerCpuTimer::start();
                #[cfg(feature = "metrics")]
                let mut job_counts = cpu_timer.as_ref().map(|_| JobCounts::new(JobKind::Storage));
                #[cfg(not(feature = "metrics"))]
                let mut job_counts = None;
                let result = worker.run(job_counts.as_mut());
                #[cfg(feature = "metrics")]
                if let Some(timer) = cpu_timer {
                    let parent = if span.is_disabled() { &storage_parent_span } else { &span };
                    timer.record(parent, "proof_storage_worker_totals", result.is_ok(), job_counts.as_ref());
                }
                if let Err(error) = result {
                    error!(
                        target: "trie::proof_task",
                        worker_id,
                        ?error,
                        "Storage worker failed"
                    );
                    let _ = storage_result_tx.send(ProofResultMessage {
                        result: Err(StateRootTaskError::ProofWorker(format!(
                            "storage worker {worker_id}: {error}"
                        ))),
                        elapsed: Duration::ZERO,
                        state: Default::default(),
                    });
                }
            });
        });

        let account_rt = runtime.clone();
        let account_tx = storage_work_tx.clone();
        let account_avail = account_availability.clone();
        let account_result_tx = proof_result_tx;
        let account_parent_span = tracing::Span::current();
        runtime.spawn_blocking_named("account-workers", move || {
            let worker_id = AtomicUsize::new(0);
            account_rt.proof_account_worker_pool().broadcast(account_worker_count, |_| {
                let worker_id = worker_id.fetch_add(1, Ordering::Relaxed);
                let span = debug_span!(target: "trie::proof_task", parent: account_parent_span.clone(), "account_worker", ?worker_id);
                let _guard = span.enter();

                #[cfg(feature = "metrics")]
                let metrics = ProofTaskTrieMetrics::default();
                #[cfg(feature = "metrics")]
                let cursor_metrics = ProofTaskCursorMetrics::new();

                let worker = AccountProofWorker::new(
                    task_ctx.clone(),
                    account_work_rx.clone(),
                    worker_id,
                    account_tx.clone(),
                    storage_worker_count,
                    account_avail.clone(),
                    cached_storage_roots.clone(),
                    #[cfg(feature = "metrics")]
                    metrics,
                    #[cfg(feature = "metrics")]
                    cursor_metrics,
                );
                #[cfg(feature = "metrics")]
                let cpu_timer = crate::worker_cpu::WorkerCpuTimer::start();
                #[cfg(feature = "metrics")]
                let mut job_counts = cpu_timer.as_ref().map(|_| JobCounts::new(JobKind::Account));
                #[cfg(not(feature = "metrics"))]
                let mut job_counts = None;
                let result = worker.run(job_counts.as_mut());
                #[cfg(feature = "metrics")]
                if let Some(timer) = cpu_timer {
                    let parent = if span.is_disabled() { &account_parent_span } else { &span };
                    timer.record(parent, "proof_account_worker_totals", result.is_ok(), job_counts.as_ref());
                }
                if let Err(error) = result {
                    error!(
                        target: "trie::proof_task",
                        worker_id,
                        ?error,
                        "Account worker failed"
                    );
                    let _ = account_result_tx.send(ProofResultMessage {
                        result: Err(StateRootTaskError::ProofWorker(format!(
                            "account worker {worker_id}: {error}"
                        ))),
                        elapsed: Duration::ZERO,
                        state: Default::default(),
                    });
                }
            });
        });

        Self {
            storage_work_tx,
            account_work_tx,
            storage_availability,
            account_availability,
            storage_worker_count,
            account_worker_count,
        }
    }

    /// Returns `true` if more than one storage worker is currently idle.
    pub fn has_multiple_idle_storage_workers(&self) -> bool {
        self.storage_availability.has_multiple_idle()
    }

    /// Returns `true` if more than one account worker is currently idle.
    pub fn has_multiple_idle_account_workers(&self) -> bool {
        self.account_availability.has_multiple_idle()
    }

    /// Returns the number of pending storage tasks in the queue.
    pub fn pending_storage_tasks(&self) -> usize {
        self.storage_work_tx.len()
    }

    /// Returns the number of pending account tasks in the queue.
    pub fn pending_account_tasks(&self) -> usize {
        self.account_work_tx.len()
    }

    /// Returns the total number of storage workers in the pool.
    pub const fn total_storage_workers(&self) -> usize {
        self.storage_worker_count
    }

    /// Returns the total number of account workers in the pool.
    pub const fn total_account_workers(&self) -> usize {
        self.account_worker_count
    }

    /// Dispatch a storage proof computation to storage worker pool
    ///
    /// The result will be sent via the `proof_result_sender` channel.
    pub fn dispatch_storage_proof(
        &self,
        input: StorageProofInput,
        proof_result_sender: CrossbeamSender<StorageProofResultMessage>,
    ) -> Result<(), ProviderError> {
        let hashed_address = input.hashed_address;
        self.storage_work_tx
            .send(StorageWorkerJob::StorageProof {
                input,
                proof_result_sender: StorageResultSender::Nested(proof_result_sender),
                trace: ProofJobTrace::storage(self.storage_work_tx.len()),
            })
            .map_err(|err| {
                let StorageWorkerJob::StorageProof { proof_result_sender, .. } = err.0;
                let _ = proof_result_sender.send(
                    hashed_address,
                    Err(DatabaseError::Other("storage workers unavailable".to_string()).into()),
                );

                ProviderError::other(std::io::Error::other("storage workers unavailable"))
            })
    }

    /// Dispatch an account multiproof computation
    ///
    /// The result will be sent via the `result_sender` channel included in the input.
    pub fn dispatch_account_multiproof(
        &self,
        input: AccountMultiproofInput,
    ) -> Result<(), ProviderError> {
        self.account_work_tx
            .send(AccountWorkerJob::AccountMultiproof {
                input: Box::new(input),
                trace: ProofJobTrace::account(self.account_work_tx.len()),
            })
            .map_err(|err| {
                let error =
                    ProviderError::other(std::io::Error::other("account workers unavailable"));

                let AccountWorkerJob::AccountMultiproof { input, .. } = err.0;
                let ProofResultContext { sender: result_tx, state, start_time: start } =
                    input.into_proof_result_sender();

                let _ = result_tx.send(ProofResultMessage {
                    result: Err(StateRootTaskError::ProofDispatch(error.clone())),
                    elapsed: start.elapsed(),
                    state,
                });

                error
            })
    }
}

/// Data used for initializing cursor factories that is shared across all proof worker instances.
#[derive(Clone, Debug)]
pub struct ProofTaskCtx<Factory> {
    /// The factory for creating state providers.
    factory: Factory,
    /// Maximum random jitter to apply before each proof computation (trie-debug only).
    #[cfg(feature = "trie-debug")]
    proof_jitter: Option<Duration>,
}

impl<Factory> ProofTaskCtx<Factory> {
    /// Creates a new [`ProofTaskCtx`] with the given factory.
    pub const fn new(factory: Factory) -> Self {
        Self {
            factory,
            #[cfg(feature = "trie-debug")]
            proof_jitter: None,
        }
    }

    /// Sets the maximum proof jitter duration (trie-debug only).
    #[cfg(feature = "trie-debug")]
    pub const fn with_proof_jitter(mut self, jitter: Option<Duration>) -> Self {
        self.proof_jitter = jitter;
        self
    }
}

/// This contains all information shared between account proof worker instances.
#[derive(Debug)]
pub struct ProofTaskTx<Provider> {
    /// The provider that implements `TrieCursorFactory` and `HashedCursorFactory`.
    provider: Provider,

    /// Identifier for the worker within the worker pool, used only for tracing.
    id: usize,
}

impl<Provider> ProofTaskTx<Provider> {
    /// Initializes a [`ProofTaskTx`] with the given provider and ID.
    const fn new(provider: Provider, id: usize) -> Self {
        Self { provider, id }
    }
}

impl<Provider> ProofTaskTx<Provider>
where
    Provider: TrieCursorFactory + HashedCursorFactory,
{
    fn compute_v2_storage_proof<TC, HC>(
        &self,
        input: StorageProofInput,
        calculator: &mut proof_v2::StorageProofCalculator<TC, HC>,
    ) -> Result<StorageProofResult, StateProofError>
    where
        TC: TrieStorageCursor,
        HC: HashedStorageCursor<Value = U256>,
    {
        let StorageProofInput { hashed_address, mut targets, needs_root } = input;

        let span = debug_span!(
            target: "trie::proof_task",
            "Storage proof calculation",
            n = %targets.len(),
        );
        let _span_guard = span.enter();

        let proof_start = Instant::now();

        // If targets is empty it means the caller only wants the root node.
        let (proof, root) = if targets.is_empty() {
            let root_node = calculator.storage_root_node(hashed_address)?;
            let root = calculator.compute_root_hash(core::slice::from_ref(&root_node))?;
            (vec![root_node], root)
        } else {
            // A partial proof cannot provide the storage root. Calculate it separately without
            // changing the target's parent context, then reset the storage cursors by starting the
            // targeted proof.
            let root = if needs_root && targets.iter().all(|target| target.parent.is_known()) {
                let root_node = calculator.storage_root_node(hashed_address)?;
                calculator.compute_root_hash(core::slice::from_ref(&root_node))?
            } else {
                None
            };

            let proof = calculator.storage_proof(hashed_address, &mut targets)?;
            let root = if root.is_some() { root } else { calculator.compute_root_hash(&proof)? };
            (proof, root)
        };

        trace!(
            target: "trie::proof_task",
            hashed_address = ?hashed_address,
            proof_time_us = proof_start.elapsed().as_micros(),
            ?root,
            worker_id = self.id,
            "Completed V2 storage proof calculation"
        );

        Ok(StorageProofResult { proof, root })
    }
}

/// Channel used by worker threads to deliver proof results back to
/// `SparseTrieCacheTask`.
///
/// Workers use this sender to deliver proof results or terminal initialization errors directly to
/// `SparseTrieCacheTask`.
pub type ProofResultSender = CrossbeamSender<ProofResultMessage>;

/// Message containing a completed proof result with metadata for direct delivery to
/// `SparseTrieCacheTask`.
///
/// This type enables workers to send proof results directly to the `SparseTrieCacheTask` event
/// loop.
#[derive(Debug)]
pub struct ProofResultMessage {
    /// The proof calculation result
    pub result: Result<DecodedMultiProofV2, StateRootTaskError>,
    /// Time taken for the entire proof calculation (from dispatch to completion)
    pub elapsed: Duration,
    /// Original state update that triggered this proof
    pub state: HashedPostState,
}

/// Context for sending proof calculation results back to `SparseTrieCacheTask`.
///
/// This struct contains all context needed to send and track proof calculation results.
/// Workers use this to deliver completed proofs back to the main event loop.
#[derive(Debug, Clone)]
pub struct ProofResultContext {
    /// Channel sender for result delivery
    pub sender: ProofResultSender,
    /// Original state update that triggered this proof
    pub state: HashedPostState,
    /// Calculation start time for measuring elapsed duration
    pub start_time: Instant,
}

impl ProofResultContext {
    /// Creates a new proof result context.
    pub const fn new(
        sender: ProofResultSender,
        state: HashedPostState,
        start_time: Instant,
    ) -> Self {
        Self { sender, state, start_time }
    }
}

/// The results of a storage proof calculation.
#[derive(Debug)]
pub(crate) struct StorageProofResult {
    /// The calculated V2 proof nodes
    pub proof: Vec<ProofTrieNodeV2>,
    /// The storage root calculated by the V2 proof
    pub root: Option<B256>,
}

impl StorageProofResult {
    /// Returns the calculated root of the trie, if one can be calculated from the proof.
    const fn root(&self) -> Option<B256> {
        self.root
    }
}

/// Message containing a completed storage proof result with metadata.
#[derive(Debug)]
pub struct StorageProofResultMessage {
    /// The hashed address this storage proof belongs to
    #[allow(dead_code)]
    pub(crate) hashed_address: B256,
    /// The storage proof calculation result
    pub(crate) result: Result<StorageProofResult, StateProofError>,
}

/// Private delivery mode. A single storage-only multiproof does not need an
/// intermediate receiver or an account worker blocked collecting it.
#[derive(Debug)]
pub(crate) enum StorageResultSender {
    Nested(CrossbeamSender<StorageProofResultMessage>),
    Multiproof(Box<ForwardedStorageProof>),
}

impl StorageResultSender {
    fn send(
        self,
        hashed_address: B256,
        result: Result<StorageProofResult, StateProofError>,
    ) -> bool {
        match self {
            Self::Nested(sender) => {
                sender.send(StorageProofResultMessage { hashed_address, result }).is_ok()
            }
            Self::Multiproof(context) => context.send(
                result
                    .map(|result| DecodedMultiProofV2 {
                        account_proofs: Vec::new(),
                        storage_proofs: B256Map::from_iter([(hashed_address, result.proof)]),
                    })
                    .map_err(StateRootTaskError::from),
            ),
        }
    }
}

/// Replaces the old nested receiver's channel-closed error when queued storage work
/// is dropped or a worker unwinds before delivering it. Normal completion disarms it.
#[derive(Debug)]
pub(crate) struct ForwardedStorageProof {
    context: Option<ProofResultContext>,
    hashed_address: B256,
}

impl ForwardedStorageProof {
    fn send(mut self, result: Result<DecodedMultiProofV2, StateRootTaskError>) -> bool {
        self.context.take().expect("one completion").send(result)
    }
}

impl Drop for ForwardedStorageProof {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            context.send(Err(StateProofError::Database(DatabaseError::Other(format!(
                "Storage proof channel closed for {:?}",
                self.hashed_address,
            )))
            .into()));
        }
    }
}

impl ProofResultContext {
    fn send(self, result: Result<DecodedMultiProofV2, StateRootTaskError>) -> bool {
        let Self { sender, state, start_time } = self;
        sender.send(ProofResultMessage { result, elapsed: start_time.elapsed(), state }).is_ok()
    }
}

/// Capture-only job metadata. Queue spans have no children, so dropping them at dequeue
/// measures queue residence rather than the lifetime of references held by later work.
#[derive(Debug)]
pub(crate) struct ProofJobTrace {
    queue: tracing::Span,
    parent: tracing::Span,
}

impl ProofJobTrace {
    fn storage(queued_jobs: usize) -> Self {
        Self {
            queue: debug_span!(target: "lifecycle", "proof.storage.queue_wait", queued_jobs),
            parent: tracing::Span::current(),
        }
    }

    fn account(queued_jobs: usize) -> Self {
        Self {
            queue: debug_span!(target: "lifecycle", "proof.account.queue_wait", queued_jobs),
            parent: tracing::Span::current(),
        }
    }

    fn start_storage(self) -> ProofServiceTrace {
        let Self { queue, parent } = self;
        drop(queue);
        ProofServiceTrace {
            // The originating service may already have completed. Keep this final
            // reference alive until the subscriber has created and parented the child.
            span: debug_span!(target: "lifecycle", parent: &parent, "proof.storage.work").entered(),
            completed: false,
        }
    }

    fn start_account(self) -> ProofServiceTrace {
        let Self { queue, parent } = self;
        drop(queue);
        ProofServiceTrace {
            span: debug_span!(target: "lifecycle", parent: &parent, "proof.account.work").entered(),
            completed: false,
        }
    }
}

/// Ends worker service independently of span references retained by dispatched storage work.
struct ProofServiceTrace {
    span: tracing::span::EnteredSpan,
    completed: bool,
}

impl ProofServiceTrace {
    fn complete(mut self) {
        self.completed = true;
        tracing::info!(target: "lifecycle", parent: &self.span, stage = "operation_completed");
    }
}

impl Drop for ProofServiceTrace {
    fn drop(&mut self) {
        if !self.completed {
            tracing::info!(target: "lifecycle", parent: &self.span, stage = "operation_abandoned");
        }
    }
}

/// Internal message for storage workers.
#[derive(Debug)]
pub(crate) enum StorageWorkerJob {
    /// Storage proof computation request
    StorageProof {
        /// Storage proof input parameters
        input: StorageProofInput,
        /// Context for sending the proof result.
        proof_result_sender: StorageResultSender,
        /// Queue interval and originating operation, independent of worker lifetime.
        trace: ProofJobTrace,
    },
}

/// Worker for storage trie operations.
///
/// Each worker maintains a dedicated database transaction and processes
/// storage proof requests.
struct StorageProofWorker<Factory> {
    /// Shared task context with database factory and prefix sets
    task_ctx: ProofTaskCtx<Factory>,
    /// Channel for receiving work
    work_rx: CrossbeamReceiver<StorageWorkerJob>,
    /// Unique identifier for this worker (used for tracing)
    worker_id: usize,
    /// Per-worker availability flags
    availability: Arc<AvailabilitySheet>,
    /// Cached storage roots
    cached_storage_roots: Arc<DashMap<B256, B256>>,
    /// Metrics collector for this worker
    #[cfg(feature = "metrics")]
    metrics: ProofTaskTrieMetrics,
    /// Cursor metrics for this worker
    #[cfg(feature = "metrics")]
    cursor_metrics: ProofTaskCursorMetrics,
}

impl<Factory> StorageProofWorker<Factory>
where
    Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>,
{
    /// Creates a new storage proof worker.
    const fn new(
        task_ctx: ProofTaskCtx<Factory>,
        work_rx: CrossbeamReceiver<StorageWorkerJob>,
        worker_id: usize,
        availability: Arc<AvailabilitySheet>,
        cached_storage_roots: Arc<DashMap<B256, B256>>,
        #[cfg(feature = "metrics")] metrics: ProofTaskTrieMetrics,
        #[cfg(feature = "metrics")] cursor_metrics: ProofTaskCursorMetrics,
    ) -> Self {
        Self {
            task_ctx,
            work_rx,
            worker_id,
            availability,
            cached_storage_roots,
            #[cfg(feature = "metrics")]
            metrics,
            #[cfg(feature = "metrics")]
            cursor_metrics,
        }
    }

    /// Runs the worker loop, processing jobs until the channel closes.
    ///
    /// # Lifecycle
    ///
    /// 1. Initializes database provider and transaction
    /// 2. Advertises availability
    /// 3. Processes jobs in a loop:
    ///    - Receives job from channel
    ///    - Marks worker as busy
    ///    - Processes the job
    ///    - Marks worker as available
    /// 4. Shuts down when channel closes
    ///
    /// # Panic Safety
    ///
    /// If this function panics, the worker thread terminates but other workers
    /// continue operating and the system degrades gracefully.
    fn run(mut self, mut job_counts: Option<&mut JobCounts>) -> ProviderResult<()> {
        // Create provider from factory
        let provider = self.task_ctx.factory.database_provider_ro()?;
        let proof_tx = ProofTaskTx::new(provider, self.worker_id);

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            "Storage worker started"
        );

        let mut storage_proofs_processed = 0u64;
        let mut cursor_metrics_cache = ProofTaskCursorMetricsCache::default();
        let trie_cursor = proof_tx.provider.storage_trie_cursor(B256::ZERO)?;
        let hashed_cursor = proof_tx.provider.hashed_storage_cursor(B256::ZERO)?;
        let instrumented_trie_cursor =
            InstrumentedTrieCursor::new(trie_cursor, &mut cursor_metrics_cache.storage_trie_cursor);
        let instrumented_hashed_cursor = InstrumentedHashedCursor::new(
            hashed_cursor,
            &mut cursor_metrics_cache.storage_hashed_cursor,
        );
        let mut v2_calculator = proof_v2::StorageProofCalculator::new_storage(
            instrumented_trie_cursor,
            instrumented_hashed_cursor,
        );

        // Initially mark this worker as available.
        self.availability.mark_idle(self.worker_id);

        let mut total_idle_time = Duration::ZERO;
        let mut idle_start = Instant::now();

        while let Ok(job) = self.work_rx.recv() {
            total_idle_time += idle_start.elapsed();

            // Mark worker as busy.
            self.availability.mark_busy(self.worker_id);
            let StorageWorkerJob::StorageProof { input, proof_result_sender, trace } = job;
            let work = trace.start_storage();
            JobCounts::observe(job_counts.as_deref_mut(), || JobSize {
                targets: input.targets.len(),
                storage_groups: 0,
                needs_root: input.needs_root,
            });

            #[cfg(feature = "trie-debug")]
            if let Some(max_jitter) = self.task_ctx.proof_jitter {
                let jitter =
                    Duration::from_nanos(rand::random_range(0..=max_jitter.as_nanos() as u64));
                trace!(
                    target: "trie::proof_task",
                    worker_id = self.worker_id,
                    jitter_us = jitter.as_micros(),
                    "Storage worker applying proof jitter"
                );
                std::thread::sleep(jitter);
            }

            self.process_storage_proof(
                &proof_tx,
                &mut v2_calculator,
                input,
                proof_result_sender,
                &mut storage_proofs_processed,
            );
            work.complete();

            // Mark worker as available again.
            self.availability.mark_idle(self.worker_id);

            idle_start = Instant::now();
        }

        // Drop calculator to release mutable borrows on cursor_metrics_cache.
        drop(v2_calculator);

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            storage_proofs_processed,
            total_idle_time_us = total_idle_time.as_micros(),
            "Storage worker shutting down"
        );

        #[cfg(feature = "metrics")]
        {
            self.metrics.record_storage_worker_idle_time(total_idle_time);
            self.cursor_metrics.record(&mut cursor_metrics_cache);
        }

        Ok(())
    }

    /// Processes a storage proof request.
    fn process_storage_proof<Provider, TC, HC>(
        &self,
        proof_tx: &ProofTaskTx<Provider>,
        v2_calculator: &mut proof_v2::StorageProofCalculator<TC, HC>,
        input: StorageProofInput,
        proof_result_sender: StorageResultSender,
        storage_proofs_processed: &mut u64,
    ) where
        Provider: TrieCursorFactory + HashedCursorFactory,
        TC: TrieStorageCursor,
        HC: HashedStorageCursor<Value = U256>,
    {
        let hashed_address = input.hashed_address;
        let proof_start = Instant::now();

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            hashed_address = ?hashed_address,
            targets_len = input.targets.len(),
            "Processing V2 storage proof"
        );

        let result = proof_tx.compute_v2_storage_proof(input, v2_calculator);

        let proof_elapsed = proof_start.elapsed();
        *storage_proofs_processed += 1;

        let root = result.as_ref().ok().and_then(|result| result.root());

        if !proof_result_sender.send(hashed_address, result) {
            trace!(
                target: "trie::proof_task",
                worker_id = self.worker_id,
                hashed_address = ?hashed_address,
                storage_proofs_processed,
                "Proof result receiver dropped, discarding result"
            );
        }

        if let Some(root) = root {
            self.cached_storage_roots.insert(hashed_address, root);
        }

        trace!(
            target: "trie::proof_task",
            worker_id = self.worker_id,
            hashed_address = ?hashed_address,
            proof_time_us = proof_elapsed.as_micros(),
            total_processed = storage_proofs_processed,
            ?root,
            "Storage proof completed"
        );
    }
}

/// Worker for account trie operations.
///
/// Each worker maintains a dedicated database transaction and processes
/// account multiproof requests.
struct AccountProofWorker<Factory> {
    /// Shared task context with database factory and prefix sets
    task_ctx: ProofTaskCtx<Factory>,
    /// Channel for receiving work
    work_rx: CrossbeamReceiver<AccountWorkerJob>,
    /// Unique identifier for this worker (used for tracing)
    worker_id: usize,
    /// Channel for dispatching storage proof work (for pre-dispatched target proofs)
    storage_work_tx: CrossbeamSender<StorageWorkerJob>,
    /// Advisory threshold: one queued job per storage worker before using the nested path.
    storage_worker_count: usize,
    /// Per-worker availability flags
    availability: Arc<AvailabilitySheet>,
    /// Cached storage roots
    cached_storage_roots: Arc<DashMap<B256, B256>>,
    /// Metrics collector for this worker
    #[cfg(feature = "metrics")]
    metrics: ProofTaskTrieMetrics,
    /// Cursor metrics for this worker
    #[cfg(feature = "metrics")]
    cursor_metrics: ProofTaskCursorMetrics,
}

impl<Factory> AccountProofWorker<Factory>
where
    Factory: DatabaseProviderROFactory<Provider: TrieCursorFactory + HashedCursorFactory>,
{
    /// Creates a new account proof worker.
    #[expect(clippy::too_many_arguments)]
    const fn new(
        task_ctx: ProofTaskCtx<Factory>,
        work_rx: CrossbeamReceiver<AccountWorkerJob>,
        worker_id: usize,
        storage_work_tx: CrossbeamSender<StorageWorkerJob>,
        storage_worker_count: usize,
        availability: Arc<AvailabilitySheet>,
        cached_storage_roots: Arc<DashMap<B256, B256>>,
        #[cfg(feature = "metrics")] metrics: ProofTaskTrieMetrics,
        #[cfg(feature = "metrics")] cursor_metrics: ProofTaskCursorMetrics,
    ) -> Self {
        Self {
            task_ctx,
            work_rx,
            worker_id,
            storage_work_tx,
            storage_worker_count,
            availability,
            cached_storage_roots,
            #[cfg(feature = "metrics")]
            metrics,
            #[cfg(feature = "metrics")]
            cursor_metrics,
        }
    }

    /// Runs the worker loop, processing jobs until the channel closes.
    ///
    /// # Lifecycle
    ///
    /// 1. Initializes database provider and transaction
    /// 2. Advertises availability
    /// 3. Processes jobs in a loop:
    ///    - Receives job from channel
    ///    - Marks worker as busy
    ///    - Processes the job
    ///    - Marks worker as available
    /// 4. Shuts down when channel closes
    ///
    /// # Panic Safety
    ///
    /// If this function panics, the worker thread terminates but other workers
    /// continue operating and the system degrades gracefully.
    fn run(mut self, mut job_counts: Option<&mut JobCounts>) -> ProviderResult<()> {
        let provider = self.task_ctx.factory.database_provider_ro()?;

        trace!(
            target: "trie::proof_task",
            worker_id=self.worker_id,
            "Account worker started"
        );

        let mut account_proofs_processed = 0u64;
        let mut cursor_metrics_cache = ProofTaskCursorMetricsCache::default();

        // Create both account and storage calculators for V2 proofs.
        // The storage calculator is wrapped in Rc<RefCell<...>> for sharing with value encoders.
        let account_trie_cursor = provider.account_trie_cursor()?;
        let account_hashed_cursor = provider.hashed_account_cursor()?;

        let storage_trie_cursor = provider.storage_trie_cursor(B256::ZERO)?;
        let storage_hashed_cursor = provider.hashed_storage_cursor(B256::ZERO)?;

        let instrumented_account_trie_cursor = InstrumentedTrieCursor::new(
            account_trie_cursor,
            &mut cursor_metrics_cache.account_trie_cursor,
        );
        let instrumented_account_hashed_cursor = InstrumentedHashedCursor::new(
            account_hashed_cursor,
            &mut cursor_metrics_cache.account_hashed_cursor,
        );
        let instrumented_storage_trie_cursor = InstrumentedTrieCursor::new(
            storage_trie_cursor,
            &mut cursor_metrics_cache.storage_trie_cursor,
        );
        let instrumented_storage_hashed_cursor = InstrumentedHashedCursor::new(
            storage_hashed_cursor,
            &mut cursor_metrics_cache.storage_hashed_cursor,
        );

        let mut v2_account_calculator =
            proof_v2::ProofCalculator::<
                _,
                _,
                AsyncAccountValueEncoder<
                    InstrumentedTrieCursor<
                        '_,
                        <Factory::Provider as TrieCursorFactory>::StorageTrieCursor<'_>,
                    >,
                    InstrumentedHashedCursor<
                        '_,
                        <Factory::Provider as HashedCursorFactory>::StorageCursor<'_>,
                    >,
                >,
            >::new(instrumented_account_trie_cursor, instrumented_account_hashed_cursor);
        let v2_storage_calculator =
            Rc::new(RefCell::new(proof_v2::StorageProofCalculator::new_storage(
                instrumented_storage_trie_cursor,
                instrumented_storage_hashed_cursor,
            )));

        // Count this worker as available only after successful initialization.
        self.availability.mark_idle(self.worker_id);

        let mut total_idle_time = Duration::ZERO;
        let mut idle_start = Instant::now();
        let mut value_encoder_stats_cache = ValueEncoderStats::default();

        while let Ok(job) = self.work_rx.recv() {
            total_idle_time += idle_start.elapsed();

            // Mark worker as busy.
            self.availability.mark_busy(self.worker_id);
            let AccountWorkerJob::AccountMultiproof { input, trace } = job;
            let work = trace.start_account();
            JobCounts::observe(job_counts.as_deref_mut(), || JobSize {
                targets: input.targets.account_targets.len(),
                storage_groups: input.targets.storage_targets.len(),
                needs_root: false,
            });

            #[cfg(feature = "trie-debug")]
            if let Some(max_jitter) = self.task_ctx.proof_jitter {
                let jitter =
                    Duration::from_nanos(rand::random_range(0..=max_jitter.as_nanos() as u64));
                trace!(
                    target: "trie::proof_task",
                    worker_id = self.worker_id,
                    jitter_us = jitter.as_micros(),
                    "Account worker applying proof jitter"
                );
                std::thread::sleep(jitter);
            }

            let value_encoder_stats = self.process_account_multiproof::<Factory::Provider>(
                &mut v2_account_calculator,
                v2_storage_calculator.clone(),
                *input,
                &mut account_proofs_processed,
                job_counts.as_deref_mut(),
            );
            total_idle_time += value_encoder_stats.storage_wait_time;
            value_encoder_stats_cache.extend(&value_encoder_stats);
            work.complete();

            // Mark worker as available again.
            self.availability.mark_idle(self.worker_id);

            idle_start = Instant::now();
        }

        // Drop calculators to release mutable borrows on cursor_metrics_cache.
        drop(v2_account_calculator);
        drop(v2_storage_calculator);

        trace!(
            target: "trie::proof_task",
            worker_id=self.worker_id,
            account_proofs_processed,
            total_idle_time_us = total_idle_time.as_micros(),
            "Account worker shutting down"
        );

        #[cfg(feature = "metrics")]
        {
            self.metrics.record_account_worker_idle_time(total_idle_time);
            self.cursor_metrics.record(&mut cursor_metrics_cache);
            self.metrics.record_value_encoder_stats(&value_encoder_stats_cache);
        }

        Ok(())
    }

    fn compute_v2_account_multiproof<'a, Provider>(
        &self,
        v2_account_calculator: &mut V2AccountProofCalculator<'a, Provider>,
        v2_storage_calculator: Rc<RefCell<V2StorageProofCalculator<'a, Provider>>>,
        targets: MultiProofTargetsV2,
    ) -> Result<(DecodedMultiProofV2, ValueEncoderStats), StateRootTaskError>
    where
        Provider: TrieCursorFactory + HashedCursorFactory + 'a,
    {
        let MultiProofTargetsV2 { mut account_targets, storage_targets } = targets;

        let span = debug_span!(
            target: "trie::proof_task",
            "Account multiproof calculation",
            account_targets = account_targets.len(),
            storage_targets = storage_targets.values().map(|t| t.len()).sum::<usize>(),
        );
        let _span_guard = span.enter();

        trace!(target: "trie::proof_task", "Processing V2 account multiproof");

        let storage_proof_receivers =
            dispatch_v2_storage_proofs(&self.storage_work_tx, &account_targets, storage_targets)?;

        let mut value_encoder = AsyncAccountValueEncoder::new(
            storage_proof_receivers,
            self.cached_storage_roots.clone(),
            v2_storage_calculator,
        );

        let account_proofs = debug_span!(target: "lifecycle", "proof.account.walk")
            .in_scope(|| v2_account_calculator.proof(&mut value_encoder, &mut account_targets))?;

        let (storage_proofs, value_encoder_stats) =
            debug_span!(target: "lifecycle", "proof.account.collect_storage")
                .in_scope(|| value_encoder.finalize())?;

        let proof = DecodedMultiProofV2 { account_proofs, storage_proofs };

        Ok((proof, value_encoder_stats))
    }

    /// Processes an account multiproof request.
    ///
    /// Returns stats from the value encoder used during proof computation.
    fn process_account_multiproof<'a, Provider>(
        &self,
        v2_account_calculator: &mut V2AccountProofCalculator<'a, Provider>,
        v2_storage_calculator: Rc<RefCell<V2StorageProofCalculator<'a, Provider>>>,
        input: AccountMultiproofInput,
        account_proofs_processed: &mut u64,
        job_counts: Option<&mut JobCounts>,
    ) -> ValueEncoderStats
    where
        Provider: TrieCursorFactory + HashedCursorFactory + 'a,
    {
        let proof_start = Instant::now();

        let Some(input) = forward_storage_only_job(
            &self.storage_work_tx,
            input,
            job_counts,
            self.storage_worker_count,
        ) else {
            *account_proofs_processed += 1;
            return ValueEncoderStats::default();
        };
        let AccountMultiproofInput { targets, proof_result_sender } = input;
        let (result, value_encoder_stats) = match self.compute_v2_account_multiproof::<Provider>(
            v2_account_calculator,
            v2_storage_calculator,
            targets,
        ) {
            Ok((proof, stats)) => (Ok(proof), stats),
            Err(e) => (Err(e), ValueEncoderStats::default()),
        };

        let ProofResultContext { sender: result_tx, state, start_time: start } =
            proof_result_sender;

        let proof_elapsed = proof_start.elapsed();
        let total_elapsed = start.elapsed();
        *account_proofs_processed += 1;

        // Send result to SparseTrieCacheTask
        if result_tx.send(ProofResultMessage { result, elapsed: total_elapsed, state }).is_err() {
            trace!(
                target: "trie::proof_task",
                worker_id=self.worker_id,
                account_proofs_processed,
                "Account multiproof receiver dropped, discarding result"
            );
        }

        trace!(
            target: "trie::proof_task",
            proof_time_us = proof_elapsed.as_micros(),
            total_elapsed_us = total_elapsed.as_micros(),
            total_processed = account_proofs_processed,
            "Account multiproof completed"
        );

        value_encoder_stats
    }
}

/// Sends only the eligible single-group storage-only case onward without retaining an
/// account worker to wait for its result. All other inputs keep the existing path.
fn forward_storage_only_job(
    storage_work_tx: &CrossbeamSender<StorageWorkerJob>,
    input: AccountMultiproofInput,
    job_counts: Option<&mut JobCounts>,
    storage_worker_count: usize,
) -> Option<AccountMultiproofInput> {
    let Some((queued_jobs, forward)) =
        forwarding_queue_decision(&input.targets, storage_worker_count, || storage_work_tx.len())
    else {
        return Some(input)
    };
    JobCounts::observe_forward(job_counts, !forward);
    if !forward {
        // Reuse the original calculation/collection path, including its result and error
        // handling. The snapshot races with other producers/consumers: this is an
        // admission heuristic, not a queue capacity or memory bound.
        return Some(input)
    }
    let AccountMultiproofInput { targets, proof_result_sender } = input;
    let (hashed_address, targets) = targets.storage_targets.into_iter().next().expect("one group");
    // An empty account proof never invokes the value encoder, so no storage root is
    // requested for account encoding. Use the same storage worker/provider/calculator.
    let job = StorageWorkerJob::StorageProof {
        input: StorageProofInput::new(hashed_address, targets, false),
        proof_result_sender: StorageResultSender::Multiproof(Box::new(ForwardedStorageProof {
            context: Some(proof_result_sender),
            hashed_address,
        })),
        trace: ProofJobTrace::storage(queued_jobs),
    };
    if let Err(error) = storage_work_tx.send(job) {
        let StorageWorkerJob::StorageProof { proof_result_sender, .. } = error.0;
        let StorageResultSender::Multiproof(context) = proof_result_sender else {
            unreachable!("forwarded job owns its original multiproof context")
        };
        context.send(Err(storage_dispatch_error(hashed_address)));
    }
    None
}

/// Do not inspect the shared queue for ineligible inputs. One snapshot controls the
/// route and is reused by the existing queue trace; there is no reservation or wait.
fn forwarding_queue_decision(
    targets: &MultiProofTargetsV2,
    storage_worker_count: usize,
    queued_jobs: impl FnOnce() -> usize,
) -> Option<(usize, bool)> {
    if !targets.account_targets.is_empty() || targets.storage_targets.len() != 1 {
        return None
    }
    let queued_jobs = queued_jobs();
    Some((queued_jobs, queued_jobs < storage_worker_count))
}

fn storage_dispatch_error(hashed_address: B256) -> StateRootTaskError {
    StateRootTaskError::Other(format!(
        "Failed to queue storage proof for {hashed_address:?}: storage worker pool unavailable",
    ))
}

/// Queues V2 storage proofs for all accounts in the targets and returns receivers.
///
/// This function queues all storage proof tasks to the worker pool but returns immediately
/// with receivers, allowing the account trie walk to proceed in parallel with storage proof
/// computation. This enables interleaved parallelism for better performance.
///
/// Propagates errors up if queuing fails. Receivers must be consumed by the caller.
fn dispatch_v2_storage_proofs(
    storage_work_tx: &CrossbeamSender<StorageWorkerJob>,
    account_targets: &[ProofV2Target],
    storage_targets: B256Map<Vec<ProofV2Target>>,
) -> Result<B256Map<CrossbeamReceiver<StorageProofResultMessage>>, StateRootTaskError> {
    if storage_targets.is_empty() {
        return Ok(B256Map::default())
    }

    let mut storage_proof_receivers =
        B256Map::with_capacity_and_hasher(storage_targets.len(), Default::default());

    // Collect hashed addresses from account targets that need their storage roots computed.
    let account_target_addresses: B256Set = account_targets.iter().map(|t| t.key()).collect();

    // Sort storage targets by address for optimal dispatch order.
    // Since trie walk processes accounts in lexicographical order, dispatching in the same order
    // reduces head-of-line blocking when consuming results.
    let mut sorted_storage_targets: Vec<_> = storage_targets.into_iter().collect();
    sorted_storage_targets.sort_unstable_by_key(|(addr, _)| *addr);

    // Dispatch all proofs for targeted storage slots
    for (hashed_address, targets) in sorted_storage_targets {
        // Create channel for receiving StorageProofResultMessage
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let needs_root = account_target_addresses.contains(&hashed_address);
        let input = StorageProofInput::new(hashed_address, targets, needs_root);

        storage_work_tx
            .send(StorageWorkerJob::StorageProof {
                input,
                proof_result_sender: StorageResultSender::Nested(result_tx),
                trace: ProofJobTrace::storage(storage_work_tx.len()),
            })
            .map_err(|_| storage_dispatch_error(hashed_address))?;

        storage_proof_receivers.insert(hashed_address, result_rx);
    }

    Ok(storage_proof_receivers)
}

/// Input parameters for storage proof computation.
#[derive(Debug)]
pub struct StorageProofInput {
    /// The hashed address for which the proof is calculated.
    pub hashed_address: B256,
    /// The set of proof targets
    pub targets: Vec<ProofV2Target>,
    /// Whether the account proof needs the storage root for leaf encoding.
    pub needs_root: bool,
}

impl StorageProofInput {
    /// Creates a new [`StorageProofInput`] with the given hashed address and target slots.
    pub const fn new(hashed_address: B256, targets: Vec<ProofV2Target>, needs_root: bool) -> Self {
        Self { hashed_address, targets, needs_root }
    }
}

/// Input parameters for account multiproof computation.
#[derive(Debug)]
pub struct AccountMultiproofInput {
    /// The targets for which to compute the multiproof.
    pub targets: MultiProofTargetsV2,
    /// Context for sending the proof result.
    pub proof_result_sender: ProofResultContext,
}

impl AccountMultiproofInput {
    /// Returns the [`ProofResultContext`] for this input, consuming the input.
    fn into_proof_result_sender(self) -> ProofResultContext {
        self.proof_result_sender
    }
}

/// Internal message for account workers.
#[derive(Debug)]
enum AccountWorkerJob {
    /// Account multiproof computation request
    AccountMultiproof {
        /// Account multiproof input parameters
        input: Box<AccountMultiproofInput>,
        /// Queue interval and originating operation, independent of worker lifetime.
        trace: ProofJobTrace,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use reth_chainspec::ChainSpec;
    use reth_provider::test_utils::create_test_provider_factory_with_chain_spec;
    use std::sync::Arc;

    #[test]
    fn queued_service_retains_completed_parent_until_child_creation() {
        use tracing_subscriber::prelude::*;
        for storage in [true, false] {
            let dispatch = tracing::Dispatch::new(
                tracing_subscriber::registry()
                    .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink)),
            );
            let queued = tracing::dispatcher::with_default(&dispatch, || {
                let account = ProofJobTrace::account(0).start_account();
                assert!(!account.span.is_disabled());
                let queued =
                    if storage { ProofJobTrace::storage(0) } else { ProofJobTrace::account(0) };
                // Forwarding completes the originating service before dequeue.
                // The queued context now owns its only remaining references.
                account.complete();
                queued
            });
            std::thread::spawn(move || {
                tracing::dispatcher::with_default(&dispatch, || {
                    if storage {
                        queued.start_storage().complete();
                    } else {
                        queued.start_account().complete();
                    }
                });
            })
            .join()
            .expect("creating the child must retain its completed parent");
        }
    }

    /// The capture subscriber closes spans when their last reference disappears. Queue
    /// spans must therefore close before work starts, and must never parent that work.
    #[test]
    fn proof_queue_and_service_end_independently() {
        use std::sync::Mutex;
        use tracing::{
            span::{Attributes, Id, Record},
            Event, Metadata, Subscriber,
        };

        #[derive(Default)]
        struct Captured {
            events: Vec<(&'static str, &'static str)>,
            spans: Vec<(&'static str, usize)>,
        }

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Captured>>);

        impl Subscriber for Capture {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, attrs: &Attributes<'_>) -> Id {
                let mut capture = self.0.lock().unwrap();
                let name = attrs.metadata().name();
                capture.events.push(("start", name));
                capture.spans.push((name, 1));
                Id::from_u64(capture.spans.len() as u64)
            }
            fn record(&self, _: &Id, _: &Record<'_>) {}
            fn record_follows_from(&self, _: &Id, _: &Id) {}
            fn event(&self, event: &Event<'_>) {
                #[derive(Default)]
                struct Stage(Option<&'static str>);
                impl tracing::field::Visit for Stage {
                    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                        if field.name() == "stage" {
                            self.0 = match value {
                                "operation_completed" => Some("operation_completed"),
                                "operation_abandoned" => Some("operation_abandoned"),
                                _ => None,
                            };
                        }
                    }
                    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {
                    }
                }
                let mut stage = Stage::default();
                event.record(&mut stage);
                if let Some(stage) = stage.0 {
                    self.0.lock().unwrap().events.push(("event", stage));
                }
            }
            fn enter(&self, _: &Id) {}
            fn exit(&self, _: &Id) {}
            fn clone_span(&self, id: &Id) -> Id {
                self.0.lock().unwrap().spans[id.into_u64() as usize - 1].1 += 1;
                id.clone()
            }
            fn try_close(&self, id: Id) -> bool {
                let mut capture = self.0.lock().unwrap();
                let (name, refs) = &mut capture.spans[id.into_u64() as usize - 1];
                *refs -= 1;
                let (name, closed) = (*name, *refs == 0);
                if closed {
                    capture.events.push(("end", name));
                }
                closed
            }
        }

        let capture = Capture::default();
        tracing::subscriber::with_default(capture.clone(), || {
            let storage = ProofJobTrace::storage(2).start_storage();
            let retained = storage.span.clone();
            storage.complete();
            assert_eq!(
                capture.0.lock().unwrap().events.last(),
                Some(&("event", "operation_completed"))
            );
            drop(retained);
            let account = ProofJobTrace::account(3).start_account();
            drop(account);
        });
        assert_eq!(
            capture.0.lock().unwrap().events,
            vec![
                ("start", "proof.storage.queue_wait"),
                ("end", "proof.storage.queue_wait"),
                ("start", "proof.storage.work"),
                ("event", "operation_completed"),
                ("end", "proof.storage.work"),
                ("start", "proof.account.queue_wait"),
                ("end", "proof.account.queue_wait"),
                ("start", "proof.account.work"),
                ("event", "operation_abandoned"),
                ("end", "proof.account.work"),
            ]
        );
    }

    fn storage_only_input(
        address: B256,
        targets: Vec<ProofV2Target>,
        tag: u64,
    ) -> (AccountMultiproofInput, CrossbeamReceiver<ProofResultMessage>) {
        let (sender, receiver) = unbounded();
        let mut state = HashedPostState::default();
        state.accounts.insert(
            B256::ZERO,
            Some(reth_primitives_traits::Account { nonce: tag, ..Default::default() }),
        );
        (
            AccountMultiproofInput {
                targets: MultiProofTargetsV2 {
                    account_targets: Vec::new(),
                    storage_targets: B256Map::from_iter([(address, targets)]),
                },
                proof_result_sender: ProofResultContext::new(sender, state, Instant::now()),
            },
            receiver,
        )
    }

    #[test]
    fn forwarding_requires_empty_accounts_and_exactly_one_storage_group() {
        let (tx, rx) = unbounded();
        for case in 0..3 {
            let (mut input, _) = storage_only_input(B256::ZERO, Vec::new(), case);
            match case {
                0 => input.targets.account_targets.push(ProofV2Target::new(B256::ZERO)),
                1 => input.targets.storage_targets.clear(),
                _ => {
                    input.targets.storage_targets.insert(B256::repeat_byte(1), Vec::new());
                }
            }
            let expected_accounts = input.targets.account_targets.clone();
            let expected_storage = input.targets.storage_targets.clone();
            let input = forward_storage_only_job(&tx, input, None, usize::MAX)
                .expect("must retain ordinary path");
            assert_eq!(
                input
                    .targets
                    .account_targets
                    .iter()
                    .map(|t| (t.key_nibbles, t.parent))
                    .collect::<Vec<_>>(),
                expected_accounts.iter().map(|t| (t.key_nibbles, t.parent)).collect::<Vec<_>>()
            );
            assert_eq!(input.targets.storage_targets.len(), expected_storage.len());
            for (address, targets) in expected_storage {
                assert_eq!(
                    input.targets.storage_targets[&address]
                        .iter()
                        .map(|t| (t.key_nibbles, t.parent))
                        .collect::<Vec<_>>(),
                    targets.iter().map(|t| (t.key_nibbles, t.parent)).collect::<Vec<_>>()
                );
            }
            assert!(rx.try_recv().is_err());
        }
    }

    #[test]
    fn forwarding_pressure_boundary_and_race_are_advisory() {
        let (input, _) = storage_only_input(B256::ZERO, Vec::new(), 1);
        for (queued, workers, expected) in [
            (0, 0, false),
            (0, 1, true),
            (1, 1, false),
            (31, 32, true),
            (32, 32, false),
            (usize::MAX, 32, false),
        ] {
            assert_eq!(
                forwarding_queue_decision(&input.targets, workers, || queued),
                Some((queued, expected))
            );
        }
        let calls = std::cell::Cell::new(0);
        let live_queue = std::cell::Cell::new(0);
        let decision = forwarding_queue_decision(&input.targets, 1, || {
            calls.set(calls.get() + 1);
            let snapshot = live_queue.get();
            live_queue.set(2); // Another producer can enqueue after the snapshot.
            snapshot
        });
        assert_eq!(decision, Some((0, true)));
        assert_eq!(calls.get(), 1);
        assert_eq!(live_queue.get(), 2);
        let mut targets = input.targets;
        targets.account_targets.push(ProofV2Target::new(B256::ZERO));
        assert_eq!(
            forwarding_queue_decision(&targets, 1, || panic!(
                "ineligible inputs must not read queue"
            )),
            None
        );
        targets.account_targets.clear();
        targets.storage_targets.clear();
        assert_eq!(
            forwarding_queue_decision(&targets, 1, || panic!(
                "ineligible inputs must not read queue"
            )),
            None
        );
        targets.storage_targets.insert(B256::ZERO, Vec::new());
        targets.storage_targets.insert(B256::repeat_byte(1), Vec::new());
        assert_eq!(
            forwarding_queue_decision(&targets, 1, || panic!(
                "ineligible inputs must not read queue"
            )),
            None
        );
    }

    #[test]
    fn pressure_fallback_preserves_context_and_resumes_after_drain() {
        let (tx, rx) = unbounded();
        let (first, first_result) = storage_only_input(B256::ZERO, Vec::new(), 1);
        assert!(forward_storage_only_job(&tx, first, None, 1).is_none());
        let mut counts = JobCounts::new(crate::job_counts::JobKind::Account);
        let address = B256::repeat_byte(2);
        let targets = vec![ProofV2Target::new(B256::repeat_byte(3))];
        let (input, result) = storage_only_input(address, targets.clone(), 2);
        let start = input.proof_result_sender.start_time;
        let fallback = forward_storage_only_job(&tx, input, Some(&mut counts), 1).unwrap();
        assert_eq!(rx.len(), 1);
        assert_eq!(counts.pressure_fallback_attempts, 1);
        assert_eq!(counts.forward_attempts, 0);
        assert_eq!(fallback.proof_result_sender.start_time, start);
        assert_eq!(fallback.proof_result_sender.state.accounts[&B256::ZERO].unwrap().nonce, 2);
        assert_eq!(fallback.targets.storage_targets[&address][0].key(), targets[0].key());
        assert!(result.try_recv().is_err());
        drop(rx.recv().unwrap());
        assert!(first_result.recv_timeout(Duration::from_secs(1)).unwrap().result.is_err());
        // The exact original context can now follow the ordinary nested error path.
        drop(rx);
        let old_error = dispatch_v2_storage_proofs(
            &tx,
            &fallback.targets.account_targets,
            fallback.targets.storage_targets,
        )
        .unwrap_err();
        fallback.proof_result_sender.send(Err(old_error));
        let message = result.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(message.state.accounts[&B256::ZERO].unwrap().nonce, 2);
        assert!(message.result.is_err());
        assert!(result.try_recv().is_err());
        let (tx, rx) = unbounded();
        let (input, canceled) = storage_only_input(address, Vec::new(), 3);
        drop(canceled);
        assert!(forward_storage_only_job(&tx, input, Some(&mut counts), 1).is_none());
        drop(rx.recv().unwrap());
        assert_eq!((counts.forward_attempts, counts.pressure_fallback_attempts), (1, 1));
    }

    #[test]
    fn forwarded_decision_is_an_attempt_even_if_dispatch_fails() {
        let mut counts = JobCounts::new(crate::job_counts::JobKind::Account);
        let (tx, rx) = unbounded();
        drop(rx);
        let (input, result) = storage_only_input(B256::ZERO, Vec::new(), 1);
        assert!(forward_storage_only_job(&tx, input, Some(&mut counts), usize::MAX).is_none());
        assert_eq!(counts.forward_attempts, 1);
        assert_eq!(counts.pressure_fallback_attempts, 0);
        assert!(result.recv_timeout(Duration::from_secs(1)).unwrap().result.is_err());
        let (mut input, _) = storage_only_input(B256::ZERO, Vec::new(), 2);
        input.targets.account_targets.push(ProofV2Target::new(B256::ZERO));
        assert!(forward_storage_only_job(&tx, input, Some(&mut counts), usize::MAX).is_some());
        assert_eq!(counts.forward_attempts, 1);
    }

    #[test]
    fn forwarded_closed_dispatch_and_dropped_work_keep_error_completion() {
        let address = B256::repeat_byte(7);
        let (tx, rx) = unbounded();
        drop(rx);
        let old_error =
            dispatch_v2_storage_proofs(&tx, &[], B256Map::from_iter([(address, Vec::new())]))
                .unwrap_err();
        let (input, result) = storage_only_input(address, Vec::new(), 1);
        assert!(forward_storage_only_job(&tx, input, None, usize::MAX).is_none());
        let message = result.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(message.result.unwrap_err().to_string(), old_error.to_string());
        assert_eq!(message.state.accounts[&B256::ZERO].unwrap().nonce, 1);
        assert!(result.try_recv().is_err());

        let (tx, rx) = unbounded();
        let (input, result) = storage_only_input(address, Vec::new(), 2);
        assert!(forward_storage_only_job(&tx, input, None, usize::MAX).is_none());
        drop(rx.recv().unwrap());
        let message = result.recv_timeout(Duration::from_secs(1)).unwrap();
        let expected: StateRootTaskError = StateProofError::Database(DatabaseError::Other(
            format!("Storage proof channel closed for {address:?}",),
        ))
        .into();
        assert_eq!(message.result.unwrap_err().to_string(), expected.to_string());
        assert_eq!(message.state.accounts[&B256::ZERO].unwrap().nonce, 2);
        assert!(result.try_recv().is_err());

        // A canceled consumer must neither block storage shutdown nor panic on the drop error.
        let (input, result) = storage_only_input(address, Vec::new(), 3);
        drop(result);
        assert!(forward_storage_only_job(&tx, input, None, usize::MAX).is_none());
        drop(rx.recv().unwrap());
    }

    #[test]
    fn forwarded_storage_errors_match_nested_conversion_and_complete_once() {
        let (tx, rx) = unbounded();
        for tag in 1..=3 {
            let (input, result) =
                storage_only_input(B256::repeat_byte(tag), Vec::new(), u64::from(tag));
            assert!(forward_storage_only_job(&tx, input, None, usize::MAX).is_none());
            let StorageWorkerJob::StorageProof { input, proof_result_sender, .. } =
                rx.recv().unwrap();
            assert!(!input.needs_root);
            let make_error = || match tag {
                1 => StateProofError::Database(DatabaseError::Other("provider failure".into())),
                2 => StateProofError::Rlp(alloy_rlp::Error::InputTooShort),
                _ => StateProofError::TrieInconsistency("invalid proof".into()),
            };
            let expected = StateRootTaskError::from(make_error()).to_string();
            assert!(proof_result_sender.send(input.hashed_address, Err(make_error())));
            let message = result.recv_timeout(Duration::from_secs(1)).unwrap();
            assert_eq!(message.result.unwrap_err().to_string(), expected);
            assert_eq!(message.state.accounts[&B256::ZERO].unwrap().nonce, u64::from(tag));
            assert!(result.try_recv().is_err());
        }
    }

    #[test]
    fn interleaved_forwarded_jobs_match_real_provider_proofs_and_cache() {
        use reth_provider::StateWriter;
        use reth_trie::{proof_v2::StorageProofCalculator, HashedStorage};
        let factory = create_test_provider_factory_with_chain_spec(Arc::new(ChainSpec::default()));
        let anchor = reth_db_common::init::init_genesis(&factory).unwrap();
        let address = B256::repeat_byte(5);
        let slot_a = B256::repeat_byte(1);
        let slot_b = B256::repeat_byte(2);
        let state = HashedPostState::from_hashed_storage(
            address,
            HashedStorage::from_iter([(slot_a, U256::from(11)), (slot_b, U256::from(22))]),
        )
        .with_accounts([(
            address,
            Some(reth_primitives_traits::Account { nonce: 1, ..Default::default() }),
        )])
        .into_sorted();
        let writer = factory.provider_rw().unwrap();
        writer.write_hashed_state(&state).unwrap();
        writer.commit().unwrap();
        let factory = reth_storage_overlay::OverlayStateProviderFactory::new(factory, reth_storage_overlay::OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default().overlay_builder(anchor));
        let (tx, rx) = unbounded();
        let mut pairs = Vec::new();
        for (tag, targets) in [
            (1, Vec::new()),
            (2, vec![ProofV2Target::new(slot_a)]),
            (3, vec![ProofV2Target::new(B256::repeat_byte(3))]),
        ] {
            let nested = dispatch_v2_storage_proofs(
                &tx,
                &[],
                B256Map::from_iter([(address, targets.clone())]),
            )
            .unwrap();
            let (input, result) = storage_only_input(address, targets, tag);
            // Mix forwarding with pressure-selected original fallback on the same provider.
            let threshold = if tag == 2 { 0 } else { usize::MAX };
            let fallback = forward_storage_only_job(&tx, input, None, threshold).map(|input| {
                let receivers = dispatch_v2_storage_proofs(
                    &tx,
                    &input.targets.account_targets,
                    input.targets.storage_targets,
                )
                .unwrap();
                (receivers, input.proof_result_sender)
            });
            pairs.push((tag, nested, result, fallback));
        }
        // Cancel another result receiver; the proof still follows the normal compute/cache path.
        let (input, canceled) = storage_only_input(address, Vec::new(), 4);
        drop(canceled);
        assert!(forward_storage_only_job(&tx, input, None, usize::MAX).is_none());
        drop(tx);
        let roots = Arc::new(DashMap::<B256, B256>::default());
        let worker = StorageProofWorker::new(
            test_ctx(factory.clone()),
            rx,
            0,
            Arc::new(AvailabilitySheet::new(1)),
            roots.clone(),
            #[cfg(feature = "metrics")]
            ProofTaskTrieMetrics::default(),
            #[cfg(feature = "metrics")]
            ProofTaskCursorMetrics::new(),
        );
        worker.run(None).unwrap();
        let provider = factory.database_provider_ro().unwrap();
        let calculator = Rc::new(RefCell::new(StorageProofCalculator::new_storage(
            provider.storage_trie_cursor(address).unwrap(),
            provider.hashed_storage_cursor(address).unwrap(),
        )));
        let mut account_calculator = proof_v2::ProofCalculator::new(
            provider.account_trie_cursor().unwrap(),
            provider.hashed_account_cursor().unwrap(),
        );
        for (tag, nested, receiver, fallback) in pairs {
            // Reconstruct the exact former empty-account path: no account nodes, then collect.
            let mut encoder =
                AsyncAccountValueEncoder::new(nested, roots.clone(), calculator.clone());
            let account_proofs = account_calculator.proof(&mut encoder, &mut []).unwrap();
            assert!(account_proofs.is_empty());
            let (storage_proofs, _) = encoder.finalize().unwrap();
            if let Some((receivers, context)) = fallback {
                // This is the unchanged nested empty-account walk/finalize sequence.
                let mut encoder =
                    AsyncAccountValueEncoder::new(receivers, roots.clone(), calculator.clone());
                let account_proofs = account_calculator.proof(&mut encoder, &mut []).unwrap();
                let (storage_proofs, _) = encoder.finalize().unwrap();
                context.send(Ok(DecodedMultiProofV2 { account_proofs, storage_proofs }));
            }
            let message = receiver.recv_timeout(Duration::from_secs(1)).unwrap();
            let actual = message.result.unwrap();
            assert_eq!(actual.account_proofs, account_proofs);
            assert_eq!(actual.storage_proofs, storage_proofs);
            assert_eq!(message.state.accounts[&B256::ZERO].unwrap().nonce, tag);
            assert!(receiver.try_recv().is_err());
        }
        // A later nonempty account proof resets its cursors and matches a fresh calculator.
        let mut reused_encoder =
            AsyncAccountValueEncoder::new(B256Map::default(), roots.clone(), calculator.clone());
        let reused = account_calculator
            .proof(&mut reused_encoder, &mut [ProofV2Target::new(address)])
            .unwrap();
        reused_encoder.finalize().unwrap();
        let mut fresh = proof_v2::ProofCalculator::new(
            provider.account_trie_cursor().unwrap(),
            provider.hashed_account_cursor().unwrap(),
        );
        let mut fresh_encoder =
            AsyncAccountValueEncoder::new(B256Map::default(), roots.clone(), calculator.clone());
        let expected = fresh.proof(&mut fresh_encoder, &mut [ProofV2Target::new(address)]).unwrap();
        fresh_encoder.finalize().unwrap();
        assert!(!reused.is_empty());
        assert_eq!(reused, expected);
        // Forward another job without touching the already-used account calculator, then reuse it.
        let (tx, rx) = unbounded();
        let (input, receiver) = storage_only_input(address, Vec::new(), 5);
        assert!(forward_storage_only_job(&tx, input, None, usize::MAX).is_none());
        drop(rx.recv().unwrap());
        assert!(receiver.recv().unwrap().result.is_err());
        let mut encoder =
            AsyncAccountValueEncoder::new(B256Map::default(), roots.clone(), calculator);
        let after_forward =
            account_calculator.proof(&mut encoder, &mut [ProofV2Target::new(address)]).unwrap();
        encoder.finalize().unwrap();
        assert_eq!(after_forward, expected);
        assert!(roots.contains_key(&address), "canceled results must still publish valid roots");
    }

    fn test_ctx<Factory>(factory: Factory) -> ProofTaskCtx<Factory> {
        ProofTaskCtx::new(factory)
    }

    /// Ensures `ProofWorkerHandle::new` spawns workers correctly.
    #[test]
    fn spawn_proof_workers_creates_handle() {
        let chain_spec = Arc::new(ChainSpec::default());
        let anchor_hash = chain_spec.genesis_hash();
        let provider_factory = create_test_provider_factory_with_chain_spec(chain_spec);
        let factory = reth_storage_overlay::OverlayStateProviderFactory::new(
            provider_factory,
            reth_storage_overlay::OverlayManager::<
                reth_ethereum_primitives::EthPrimitives,
            >::default()
            .overlay_builder(anchor_hash),
        );
        let ctx = test_ctx(factory);

        let runtime = reth_tasks::Runtime::test();
        let (proof_result_tx, _) = unbounded();
        let proof_handle = ProofWorkerHandle::new(&runtime, ctx, false, proof_result_tx);

        // Verify handle can be cloned
        let _cloned_handle = proof_handle.clone();

        // Workers shut down automatically when handle is dropped
        drop(proof_handle);
    }
}
