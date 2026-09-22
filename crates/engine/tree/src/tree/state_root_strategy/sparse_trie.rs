//! Sparse Trie task related functionality.

use std::{sync::Arc, time::Duration};

use super::{evm_state_to_hashed_post_state, StateRootComputeOutcome, StateRootMessage};
use alloy_primitives::{
    map::{hash_map::Entry, B256Map, B256Set},
    B256,
};
use alloy_rlp::{Decodable, Encodable};
use crossbeam_channel::{Receiver as CrossbeamReceiver, Sender as CrossbeamSender};
use metrics::{Gauge, Histogram};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use reth_metrics::{
    thread::{ThreadResourceUsage, ThreadResourceUsageDelta},
    Metrics,
};
use reth_primitives_traits::{Account, FastInstant as Instant};
use reth_tasks::Runtime;
use reth_trie::{
    updates::TrieUpdates, DecodedMultiProofV2, HashedPostState, TrieAccount, EMPTY_ROOT_HASH,
    TRIE_ACCOUNT_RLP_MAX_SIZE,
};
use reth_trie_common::{MultiProofTargetsV2, ProofV2Target, ProofV2TargetParent};
use reth_trie_parallel::{
    error::StateRootTaskError,
    proof_task::{
        AccountMultiproofInput, ProofResultContext, ProofResultMessage, ProofResultSender,
        ProofWorkerHandle,
    },
};
use reth_trie_sparse::{
    errors::{SparseStateTrieErrorKind, SparseTrieErrorKind, SparseTrieResult},
    ArenaParallelSparseTrie, DeferredDrops, LeafUpdate, RevealableSparseTrie, SparseStateTrie,
    SparseTrie, TrieNodeEpoch,
};
use tracing::{debug, debug_span, error, instrument, trace_span};

/// Sparse trie task implementation that uses in-memory sparse trie data to schedule proof fetching.
pub(super) struct SparseTrieCacheTask<A = ArenaParallelSparseTrie, S = ArenaParallelSparseTrie> {
    /// Sender for proof results.
    proof_result_tx: ProofResultSender,
    /// Receiver for proof results directly from workers.
    proof_result_rx: CrossbeamReceiver<ProofResultMessage>,
    /// Receives updates from execution and prewarming.
    updates: CrossbeamReceiver<SparseTrieTaskMessage>,
    /// Fires (by disconnecting) when the consumer drops its cancel guard, meaning nobody is
    /// waiting for the result anymore. This is the teardown path for a task whose pending
    /// work never drains, since the updates channel closing is a normal end of stream.
    cancel_rx: CrossbeamReceiver<()>,
    /// Sender half for the channel to send final hashed state to.
    final_hashed_state_tx: Option<std::sync::mpsc::Sender<Arc<HashedPostState>>>,
    /// `SparseStateTrie` used for computing the state root.
    trie: SparseStateTrie<A, S>,
    /// The parent block's state root.
    parent_state_root: B256,
    /// The new epoch assigned to nodes modified by this task.
    new_epoch: TrieNodeEpoch,
    /// Handle to the proof worker pools (storage and account).
    proof_worker_handle: ProofWorkerHandle,

    /// The size of proof targets chunk to spawn in one calculation.
    /// If None, chunking is disabled and all targets are processed in a single proof.
    chunk_size: usize,
    /// If this number is exceeded and chunking is enabled, then this will override whether or not
    /// there are any active workers and force chunking across workers. This is to prevent tasks
    /// which are very long from hitting a single worker.
    max_targets_for_chunking: usize,

    /// Account trie updates.
    account_updates: B256Map<LeafUpdate>,
    /// Storage trie updates. hashed address -> slot -> update.
    storage_updates: B256Map<B256Map<LeafUpdate>>,
    /// Enables retrying only storage tries whose reveal state or inputs changed.
    selective_storage_retries: bool,
    /// Storage tries eligible for another retry under the selective policy.
    storage_retry_ready: B256Set,

    /// Account updates that are buffered but were not yet applied to the trie.
    new_account_updates: B256Map<LeafUpdate>,
    /// Storage updates that are buffered but were not yet applied to the trie.
    new_storage_updates: B256Map<B256Map<LeafUpdate>>,
    /// Account updates that are blocked by storage root calculation or account reveal.
    ///
    /// Those are being moved into `account_updates` once storage roots
    /// are revealed and/or calculated.
    ///
    /// Invariant: for each entry in `pending_account_updates` account must either be already
    /// revealed in the trie or have an entry in `account_updates`.
    ///
    /// Values can be either of:
    ///   - None: account had a storage update and is awaiting storage root calculation and/or
    ///     account node reveal to complete.
    ///   - Some(_): account was changed/destroyed and is awaiting storage root calculation/reveal
    ///     to complete.
    pending_account_updates: B256Map<Option<Option<Account>>>,
    /// Cache of account proof targets that were already fetched/requested from the proof workers.
    /// Account to the broadest requested parent context (an unknown parent sorts before every
    /// known parent).
    fetched_account_targets: B256Map<ProofV2TargetParent>,
    /// Cache of storage proof targets that have already been fetched/requested from the proof
    /// workers. Account to slot to the broadest requested parent context.
    fetched_storage_targets: B256Map<B256Map<ProofV2TargetParent>>,
    /// Reusable buffer for RLP encoding of accounts.
    account_rlp_buf: Vec<u8>,
    /// Whether the last state update has been received.
    finished_state_updates: bool,
    /// Accumulated account leaf update cache hits.
    account_cache_hits: u64,
    /// Accumulated account leaf update cache misses.
    account_cache_misses: u64,
    /// Accumulated storage leaf update cache hits.
    storage_cache_hits: u64,
    /// Accumulated storage leaf update cache misses.
    storage_cache_misses: u64,
    /// Pending proof targets queued for dispatch to proof workers.
    pending_targets: PendingTargets,
    /// Proof batches dispatched to workers and not yet received.
    in_flight_proof_batches: usize,
    /// Capture-only aggregate proof dispatch diagnostics.
    dispatch_diagnostics: Option<ProofDispatchDiagnostics>,
    /// Number of pending execution/prewarming updates received but not yet passed to
    /// `update_leaves`.
    pending_updates: usize,
    /// Whether the first buffered leaf batch has been applied.
    initial_updates_applied: bool,
    /// Combined final hashed state.
    ///
    /// Sparse trie task observes and hashes all state updates, allowing it to cheaply construct a
    /// final [`HashedPostState`] and share it with main engine thread without requiring any extra
    /// hashing work.
    final_hashed_state: HashedPostState,

    /// Metrics for the sparse trie.
    metrics: SparseTrieTaskMetrics,
}

impl<A, S> SparseTrieCacheTask<A, S>
where
    A: SparseTrie + Default,
    S: SparseTrie + Default + Clone,
{
    /// Creates a new sparse trie, pre-populating with an existing [`SparseStateTrie`].
    #[expect(clippy::too_many_arguments)]
    pub(super) fn new_with_trie(
        executor: &Runtime,
        updates: CrossbeamReceiver<StateRootMessage>,
        cancel_rx: CrossbeamReceiver<()>,
        final_hashed_state_tx: std::sync::mpsc::Sender<Arc<HashedPostState>>,
        proof_worker_handle: ProofWorkerHandle,
        proof_result_tx: ProofResultSender,
        proof_result_rx: CrossbeamReceiver<ProofResultMessage>,
        metrics: SparseTrieTaskMetrics,
        trie: SparseStateTrie<A, S>,
        parent_state_root: B256,
        new_epoch: TrieNodeEpoch,
        chunk_size: usize,
    ) -> Self {
        let (hashed_state_tx, hashed_state_rx) = crossbeam_channel::unbounded();
        let selective_storage_retries = selective_storage_retries_enabled();

        let parent_span = tracing::Span::current();
        let hashing_metrics = metrics.clone();
        executor.spawn_blocking_named("trie-hashing", move || {
            let _span = crate::tree::task_span::hashing(&parent_span).entered();
            Self::run_hashing_task(updates, hashed_state_tx, hashing_metrics)
        });

        Self {
            proof_result_tx,
            proof_result_rx,
            updates: hashed_state_rx,
            cancel_rx,
            proof_worker_handle,
            final_hashed_state_tx: Some(final_hashed_state_tx),
            trie,
            parent_state_root,
            new_epoch,
            chunk_size,
            max_targets_for_chunking: DEFAULT_MAX_TARGETS_FOR_CHUNKING,
            account_updates: Default::default(),
            storage_updates: Default::default(),
            selective_storage_retries,
            storage_retry_ready: Default::default(),
            new_account_updates: Default::default(),
            new_storage_updates: Default::default(),
            pending_account_updates: Default::default(),
            fetched_account_targets: Default::default(),
            fetched_storage_targets: Default::default(),
            account_rlp_buf: Vec::with_capacity(TRIE_ACCOUNT_RLP_MAX_SIZE),
            finished_state_updates: Default::default(),
            account_cache_hits: 0,
            account_cache_misses: 0,
            storage_cache_hits: 0,
            storage_cache_misses: 0,
            pending_targets: Default::default(),
            in_flight_proof_batches: 0,
            dispatch_diagnostics: ProofDispatchDiagnostics::new(selective_storage_retries),
            pending_updates: Default::default(),
            initial_updates_applied: false,
            final_hashed_state: Default::default(),
            metrics,
        }
    }

    /// Runs the hashing task that drains updates from the channel and converts them to
    /// `HashedPostState` in parallel.
    fn run_hashing_task(
        updates: CrossbeamReceiver<StateRootMessage>,
        hashed_state_tx: CrossbeamSender<SparseTrieTaskMessage>,
        metrics: SparseTrieTaskMetrics,
    ) {
        let mut total_idle_time = std::time::Duration::ZERO;
        let mut idle_start = Instant::now();

        while let Ok(message) = updates.recv() {
            total_idle_time += idle_start.elapsed();

            let msg = match message {
                StateRootMessage::PrefetchProofs(targets) => {
                    SparseTrieTaskMessage::PrefetchProofs(targets)
                }
                StateRootMessage::StateUpdate(state) => {
                    let _span = trace_span!(target: "engine::tree::payload_processor::sparse_trie", "hashing_state_update", n = state.len()).entered();
                    let hashed = evm_state_to_hashed_post_state(state);
                    SparseTrieTaskMessage::HashedState(hashed)
                }
                StateRootMessage::FinishedStateUpdates => {
                    SparseTrieTaskMessage::FinishedStateUpdates
                }
                StateRootMessage::HashedStateUpdate(state) => {
                    SparseTrieTaskMessage::HashedState(state)
                }
            };
            if hashed_state_tx.send(msg).is_err() {
                break;
            }

            idle_start = Instant::now();
        }

        metrics.hashing_task_idle_time_seconds.record(total_idle_time.as_secs_f64());
    }

    /// Returns the trie for reuse in the next payload built on top of this one.
    ///
    /// Should be called after the state root result has been sent.
    pub(super) fn into_trie_for_reuse(self) -> (SparseStateTrie<A, S>, DeferredDrops) {
        let Self { mut trie, .. } = self;
        let deferred = trie.take_deferred_drops();
        (trie, deferred)
    }

    /// Clears the trie, discarding all state.
    ///
    /// Use this when the payload was invalid or cancelled - we don't want to preserve
    /// potentially invalid trie state, but we keep the allocations for reuse.
    pub(super) fn into_cleared_trie(self) -> (SparseStateTrie<A, S>, DeferredDrops) {
        let Self { mut trie, .. } = self;
        trie.clear();
        let deferred = trie.take_deferred_drops();
        (trie, deferred)
    }

    /// Runs the sparse trie task to completion.
    ///
    /// This waits for new incoming [`SparseTrieTaskMessage`]s, applies updates
    /// to the trie and schedules proof fetching when needed.
    ///
    /// This concludes once the last state update has been received and processed.
    #[instrument(
        name = "SparseTrieCacheTask::run",
        level = "debug",
        target = "engine::tree::payload_processor::sparse_trie",
        skip_all
    )]
    pub(super) fn run(&mut self) -> Result<StateRootComputeOutcome, StateRootTaskError> {
        let now = Instant::now();

        let mut total_idle_time = std::time::Duration::ZERO;
        let mut idle_start = Instant::now();
        let mut done = false;
        let mut finalized_hashed_state = None;

        // Streaming phase: updates are still arriving. Ends when the finish marker is
        // processed. Only producers hold update senders, so the channel closing before the
        // marker means they died without finishing the stream.
        while !self.finished_state_updates {
            let mut t = Instant::now();
            let wait = debug_span!(target: "lifecycle", "proof.trie.wait_stream",
                in_flight_proof_batches = self.in_flight_proof_batches,
                pending_updates = self.pending_updates,
                pending_targets = self.pending_targets.len())
            .entered();
            crossbeam_channel::select_biased! {
                recv(self.updates) -> message => {
                    drop(wait);
                    let wake = Instant::now();
                    total_idle_time += wake.duration_since(idle_start);
                    self.metrics
                        .sparse_trie_channel_wait_duration_histogram
                        .record(wake.duration_since(t));

                    let update = message.map_err(|_| StateRootTaskError::Other(
                        "updates channel disconnected before state root calculation".to_string(),
                    ))?;
                    if let Some(hashed_state) = self.on_message(update) {
                        finalized_hashed_state = Some(hashed_state);
                    }
                    self.pending_updates += 1;
                }
                recv(self.proof_result_rx) -> message => {
                    drop(wait);
                    let wake = Instant::now();
                    total_idle_time += wake.duration_since(idle_start);
                    self.metrics
                        .sparse_trie_channel_wait_duration_histogram
                        .record(wake.duration_since(t));
                    t = wake;

                    let Ok(result) = message else {
                        unreachable!("we own the sender half")
                    };
                    self.on_proof_results(result, &mut t)?;
                },
                recv(self.cancel_rx) -> _ => {
                    drop(wait);
                    return Err(StateRootTaskError::Canceled);
                },
            }

            let progress_start = (self.finished_state_updates &&
                self.dispatch_diagnostics.is_some())
            .then(Instant::now);
            let progress = self.make_progress();
            if let (Some(start), Some(diagnostics)) =
                (progress_start, self.dispatch_diagnostics.as_mut())
            {
                diagnostics.record_root_progress(start.elapsed());
            }
            done = progress?;
            idle_start = Instant::now();
        }

        // Draining phase: the marker is the last message read from the updates channel, so
        // after it only proof results and cancellation can occur. The channel closing when
        // the producers drop their senders is not observed here, and late best-effort hints
        // are ignored: with all updates known, prefetching has nothing left to help.
        while !done {
            let mut t = Instant::now();
            let wait = debug_span!(target: "lifecycle", "proof.trie.wait_drain",
                in_flight_proof_batches = self.in_flight_proof_batches)
            .entered();
            crossbeam_channel::select_biased! {
                recv(self.proof_result_rx) -> message => {
                    drop(wait);
                    let wake = Instant::now();
                    total_idle_time += wake.duration_since(idle_start);
                    self.metrics
                        .sparse_trie_channel_wait_duration_histogram
                        .record(wake.duration_since(t));
                    if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                        diagnostics.record_proof_wait(wake.duration_since(t));
                    }
                    t = wake;

                    let Ok(result) = message else {
                        unreachable!("we own the sender half")
                    };
                    self.on_proof_results(result, &mut t)?;
                },
                recv(self.cancel_rx) -> _ => {
                    drop(wait);
                    return Err(StateRootTaskError::Canceled);
                },
            }

            let progress_start = self.dispatch_diagnostics.as_ref().map(|_| Instant::now());
            let progress = self.make_progress();
            if let (Some(start), Some(diagnostics)) =
                (progress_start, self.dispatch_diagnostics.as_mut())
            {
                diagnostics.record_root_progress(start.elapsed());
            }
            done = progress?;
            idle_start = Instant::now();
        }

        self.metrics.sparse_trie_idle_time_seconds.record(total_idle_time.as_secs_f64());

        debug!(target: "engine::root", "All proofs processed, ending calculation");

        let start = Instant::now();
        let final_root = debug_span!(target: "lifecycle", "proof.trie.final_root")
            .in_scope(|| self.trie.root_with_updates(self.new_epoch));
        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.record_final_root(start.elapsed());
        }
        let (state_root, trie_updates) = match final_root {
            Ok(result) => result,
            Err(err)
                if matches!(
                    err.kind(),
                    SparseStateTrieErrorKind::Sparse(SparseTrieErrorKind::Blind)
                ) =>
            {
                // A still-blind account trie means this block never changed state, so preserve
                // the cached parent root instead of fetching and revealing
                // the unchanged root node.
                (self.parent_state_root, TrieUpdates::default())
            }
            Err(err) => {
                return Err(StateRootTaskError::Other(format!(
                    "could not calculate state root: {err:?}"
                )))
            }
        };

        let end = Instant::now();
        self.metrics.sparse_trie_final_update_duration_histogram.record(end.duration_since(start));
        self.metrics.sparse_trie_total_duration_histogram.record(end.duration_since(now));

        self.metrics.sparse_trie_account_cache_hits.record(self.account_cache_hits as f64);
        self.metrics.sparse_trie_account_cache_misses.record(self.account_cache_misses as f64);
        self.metrics.sparse_trie_storage_cache_hits.record(self.storage_cache_hits as f64);
        self.metrics.sparse_trie_storage_cache_misses.record(self.storage_cache_misses as f64);
        self.account_cache_hits = 0;
        self.account_cache_misses = 0;
        self.storage_cache_hits = 0;
        self.storage_cache_misses = 0;

        Ok(StateRootComputeOutcome {
            state_root,
            trie_updates: Arc::new(trie_updates),
            hashed_state: finalized_hashed_state
                .expect("finished state updates publish the hashed post state"),
        })
    }

    /// Handles a received proof result: coalesces everything already queued, reveals the
    /// proof in the trie, and records timing metrics.
    fn on_proof_results(
        &mut self,
        message: ProofResultMessage,
        t: &mut Instant,
    ) -> Result<(), StateRootTaskError> {
        let coalesce_start =
            (self.finished_state_updates && self.dispatch_diagnostics.is_some()).then(Instant::now);
        let coalesce = debug_span!(target: "lifecycle", "proof.trie.coalesce_results",
            result_count = tracing::field::Empty)
        .entered();
        let mut result_count = 1u64;
        let mut result = self.on_proof_result_message(message)?;
        while let Ok(next) = self.proof_result_rx.try_recv() {
            let res = self.on_proof_result_message(next)?;
            result.extend(res);
            result_count += 1;
        }
        let coalesce_end = coalesce_start.map(|_| Instant::now());

        coalesce.record("result_count", result_count);
        drop(coalesce);
        if let (Some(coalesce_start), Some(coalesce_end)) = (coalesce_start, coalesce_end) {
            let queue_after_drain = self.proof_result_rx.len();
            let in_flight_after_drain = self.in_flight_proof_batches;
            if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                diagnostics.record_result_drain(
                    coalesce_end.duration_since(coalesce_start),
                    result_count,
                    queue_after_drain,
                    in_flight_after_drain,
                    coalesce_end,
                );
            }
        }
        let phase_end = Instant::now();
        self.metrics
            .sparse_trie_proof_coalesce_duration_histogram
            .record(phase_end.duration_since(*t));
        *t = phase_end;

        let reveal_start =
            (self.finished_state_updates && self.dispatch_diagnostics.is_some()).then(Instant::now);
        let reveal = debug_span!(target: "lifecycle", "proof.trie.reveal_results")
            .in_scope(|| self.on_proof_result(result));
        if let (Some(start), Some(diagnostics)) = (reveal_start, self.dispatch_diagnostics.as_mut())
        {
            diagnostics.record_reveal(start.elapsed());
        }
        reveal?;
        self.metrics.sparse_trie_reveal_multiproof_duration_histogram.record(t.elapsed());
        Ok(())
    }

    /// Applies buffered updates to the trie and dispatches proof targets.
    ///
    /// Messages queued after the finish marker are best-effort hints and are not actionable.
    /// Returns `true` once the finish marker was received and all pending trie work is done.
    fn make_progress(&mut self) -> Result<bool, StateRootTaskError> {
        let updates_queued = !self.finished_state_updates && !self.updates.is_empty();

        if !updates_queued && self.proof_result_rx.is_empty() {
            // If we don't have any pending messages, we can spend some time on computing
            // storage roots and promoting account updates.
            self.dispatch_pending_targets()?;
            let t = Instant::now();
            self.process_new_updates()?;
            self.promote_pending_account_updates()?;
            self.metrics.sparse_trie_process_updates_duration_histogram.record(t.elapsed());

            if self.finished_state_updates && !self.has_pending_sparse_trie_updates() {
                return Ok(true);
            }

            self.dispatch_pending_targets()?;
            self.ensure_not_stalled(updates_queued)?;

            // If there's still no pending updates spend some time pre-computing the account
            // trie upper hashes
            if self.proof_result_rx.is_empty() {
                let timer = self
                    .dispatch_diagnostics
                    .as_ref()
                    .and_then(|diagnostics| diagnostics.progress_timer(true));
                debug_span!(target: "lifecycle", "proof.trie.calculate_subtries")
                    .in_scope(|| self.trie.calculate_subtries(self.new_epoch));
                if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                    diagnostics.record_progress(
                        RootProgressPhase::SpeculativeHash,
                        timer,
                        false,
                        1,
                        0,
                    );
                }
            }
        } else if !updates_queued {
            // If we don't have any pending updates, apply them to the trie,
            let t = Instant::now();
            self.process_new_updates()?;
            self.metrics.sparse_trie_process_updates_duration_histogram.record(t.elapsed());
            self.dispatch_pending_targets()?;
        } else if !self.initial_updates_applied && self.pending_updates >= INITIAL_UPDATE_BATCH_SIZE
        {
            // Start proof fetching before a continuously arriving state stream drains. Later
            // batches retain the usual coalescing policy to avoid repeatedly sorting small maps.
            let t = Instant::now();
            self.process_new_updates()?;
            self.metrics.sparse_trie_process_updates_duration_histogram.record(t.elapsed());
            self.dispatch_pending_targets()?;
        } else if self.pending_targets.len() > self.chunk_size {
            // Make sure to dispatch targets if we've accumulated a lot of them.
            self.dispatch_pending_targets()?;
        }
        Ok(false)
    }

    /// Processes a [`SparseTrieTaskMessage`] from the hashing task.
    fn on_message(&mut self, message: SparseTrieTaskMessage) -> Option<Arc<HashedPostState>> {
        match message {
            SparseTrieTaskMessage::PrefetchProofs(targets) => {
                self.on_prewarm_targets(targets);
                None
            }
            SparseTrieTaskMessage::HashedState(hashed_state) => {
                self.on_hashed_state_update(hashed_state);
                None
            }
            SparseTrieTaskMessage::FinishedStateUpdates => {
                if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                    diagnostics.start_root_tail();
                    diagnostics.emit_updates_finished_snapshot(
                        self.in_flight_proof_batches,
                        self.pending_targets.account_len(),
                        self.pending_targets.storage_len(),
                        self.proof_worker_handle.pending_account_tasks(),
                        self.proof_worker_handle.pending_storage_tasks(),
                        self.proof_result_rx.len(),
                    );
                }
                let hashed_state = Arc::new(core::mem::take(&mut self.final_hashed_state));
                let _ = self.final_hashed_state_tx.take().unwrap().send(Arc::clone(&hashed_state));
                self.finished_state_updates = true;
                Some(hashed_state)
            }
        }
    }

    /// Emits the post-execution root-tail breakdown immediately before publishing the result.
    pub(super) fn emit_root_tail(&mut self, success: bool) {
        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.emit_root_tail(true, success);
        }
    }

    #[instrument(
        level = "trace",
        target = "engine::tree::payload_processor::sparse_trie",
        skip_all
    )]
    fn on_prewarm_targets(&mut self, targets: MultiProofTargetsV2) {
        for target in targets.account_targets {
            // Only touch accounts that are not yet present in the updates set.
            self.new_account_updates.entry(target.key()).or_insert(LeafUpdate::Touched);
        }

        for (address, slots) in targets.storage_targets {
            if !slots.is_empty() {
                // Look up outer map once per address instead of once per slot.
                let new_updates = self.new_storage_updates.entry(address).or_default();
                for slot in slots {
                    // Only touch storages that are not yet present in the updates set.
                    new_updates.entry(slot.key()).or_insert(LeafUpdate::Touched);
                }
            }

            // Touch corresponding account leaf to make sure its revealed in accounts trie for
            // storage root update.
            self.new_account_updates.entry(address).or_insert(LeafUpdate::Touched);
        }
    }

    /// Processes a hashed state update and encodes all state changes as trie updates.
    #[instrument(
        level = "trace",
        target = "engine::tree::payload_processor::sparse_trie",
        skip_all
    )]
    fn on_hashed_state_update(&mut self, hashed_state_update: HashedPostState) {
        for (&address, storage) in &hashed_state_update.storages {
            if !storage.storage.is_empty() {
                // Look up outer maps once per address instead of once per slot.
                let new_updates = self.new_storage_updates.entry(address).or_default();
                let mut existing_updates = self.storage_updates.get_mut(&address);

                for (&slot, &value) in &storage.storage {
                    let encoded = if value.is_zero() {
                        Vec::new()
                    } else {
                        alloy_rlp::encode_fixed_size(&value).to_vec()
                    };
                    new_updates.insert(slot, LeafUpdate::Changed(encoded));

                    // Remove an existing storage update if it exists.
                    if let Some(ref mut existing) = existing_updates {
                        existing.remove(&slot);
                    }
                }
            }

            // Make sure account is tracked in `account_updates` so that it is revealed in accounts
            // trie for storage root update.
            self.new_account_updates.entry(address).or_insert(LeafUpdate::Touched);

            // Make sure account is tracked in `pending_account_updates` so that once storage root
            // is computed, it will be updated in the accounts trie.
            self.pending_account_updates.entry(address).or_insert(None);
        }

        for (&address, &account) in &hashed_state_update.accounts {
            // Track account as touched.
            //
            // This might overwrite an existing update, which is fine, because storage root from it
            // is already tracked in the trie and can be easily fetched again.
            self.new_account_updates.insert(address, LeafUpdate::Touched);

            // Track account in `pending_account_updates` so that once storage root is computed,
            // it will be updated in the accounts trie.
            self.pending_account_updates.insert(address, Some(account));
        }

        self.final_hashed_state.extend(hashed_state_update);
    }

    fn on_proof_result(&mut self, result: DecodedMultiProofV2) -> Result<(), StateRootTaskError> {
        let storage_addresses = self
            .selective_storage_retries
            .then(|| result.storage_proofs.keys().copied().collect::<Vec<_>>());
        self.trie.reveal_decoded_multiproof_v2(result).map_err(|e| {
            StateRootTaskError::Other(format!("could not reveal multiproof: {e:?}"))
        })?;

        if let Some(storage_addresses) = storage_addresses {
            let newly_ready = mark_storage_retries_ready(
                &mut self.storage_retry_ready,
                &self.storage_updates,
                storage_addresses,
            );
            if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                diagnostics.selective_storage_retries.ready_from_proof = diagnostics
                    .selective_storage_retries
                    .ready_from_proof
                    .saturating_add(newly_ready as u64);
            }
        }

        Ok(())
    }

    fn on_proof_result_message(
        &mut self,
        message: ProofResultMessage,
    ) -> Result<DecodedMultiProofV2, StateRootTaskError> {
        let result = message.result?;
        debug_assert!(
            self.in_flight_proof_batches > 0,
            "received proof result without an in-flight proof batch"
        );
        self.in_flight_proof_batches = self.in_flight_proof_batches.saturating_sub(1);
        Ok(result)
    }

    fn process_new_updates(&mut self) -> SparseTrieResult<()> {
        if self.pending_updates == 0 {
            return Ok(());
        }

        let work_items = self.pending_updates;
        let timer = self
            .dispatch_diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.progress_timer(true));
        let _span = debug_span!("process_new_updates").entered();
        self.pending_updates = 0;
        self.initial_updates_applied = true;

        let storage_input_addresses = self.selective_storage_retries.then(|| {
            self.new_storage_updates
                .iter()
                .filter_map(|(address, updates)| (!updates.is_empty()).then_some(*address))
                .collect::<Vec<_>>()
        });

        // Firstly apply all new storage and account updates to the tries.
        if let Err(error) = self.process_leaf_updates(true) {
            if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                diagnostics.record_progress(
                    RootProgressPhase::NewUpdates,
                    timer,
                    true,
                    work_items,
                    0,
                );
            }
            return Err(error)
        }

        for (address, mut new) in self.new_storage_updates.drain() {
            match self.storage_updates.entry(address) {
                Entry::Vacant(entry) => {
                    entry.insert(new); // insert the whole map at once, no per-slot loop
                }
                Entry::Occupied(mut entry) => {
                    let updates = entry.get_mut();
                    for (slot, new) in new.drain() {
                        match updates.entry(slot) {
                            Entry::Occupied(mut slot_entry) => {
                                if new.is_changed() {
                                    slot_entry.insert(new);
                                }
                            }
                            Entry::Vacant(slot_entry) => {
                                slot_entry.insert(new);
                            }
                        }
                    }
                }
            }
        }

        for (address, new) in self.new_account_updates.drain() {
            match self.account_updates.entry(address) {
                Entry::Occupied(mut entry) => {
                    if new.is_changed() {
                        entry.insert(new);
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(new);
                }
            }
        }

        if let Some(storage_input_addresses) = storage_input_addresses {
            let newly_ready = mark_storage_retries_ready(
                &mut self.storage_retry_ready,
                &self.storage_updates,
                storage_input_addresses,
            );
            if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                diagnostics.selective_storage_retries.ready_from_input = diagnostics
                    .selective_storage_retries
                    .ready_from_input
                    .saturating_add(newly_ready as u64);
            }
        }

        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.record_progress(
                RootProgressPhase::NewUpdates,
                timer,
                false,
                work_items,
                work_items,
            );
        }

        Ok(())
    }

    /// Applies all account and storage leaf updates to corresponding tries and collects any new
    /// multiproof targets.
    #[instrument(
        level = "trace",
        target = "engine::tree::payload_processor::sparse_trie",
        skip_all
    )]
    fn process_leaf_updates(&mut self, new: bool) -> SparseTrieResult<()> {
        self.process_leaf_updates_inner(new, false)
    }

    fn process_leaf_updates_inner(&mut self, new: bool, fallback: bool) -> SparseTrieResult<()> {
        debug_assert!(!fallback || !new, "fallback only applies to old storage updates");
        let selective = !new && self.selective_storage_retries;
        let retry_counting_enabled = !new && self.dispatch_diagnostics.is_some();
        let retry_enabled = !new &&
            self.dispatch_diagnostics
                .as_ref()
                .is_some_and(|diagnostics| diagnostics.root_tail_start.is_some()) &&
            !self.storage_updates.is_empty();
        let ready_addresses = if selective && !fallback {
            core::mem::take(&mut self.storage_retry_ready)
        } else {
            B256Set::default()
        };
        let storage_updates =
            if new { &mut self.new_storage_updates } else { &mut self.storage_updates };
        let retry_timer = RootProgressTimer::start(retry_enabled);
        let mut retry_work_items = 0usize;
        let mut retry_work_outputs = 0usize;
        let mut retry_error = None;
        let maps_considered = if retry_counting_enabled {
            storage_updates.values().filter(|updates| !updates.is_empty()).count()
        } else {
            0
        };
        let mut maps_attempted = 0usize;
        let mut maps_skipped = 0usize;
        let mut productive_requeues = 0usize;
        let mut productive_ready = Vec::new();

        // Process all storage updates, skipping tries with no pending updates.
        let span = trace_span!("process_storage_leaf_updates").entered();
        'storage: for (address, updates) in storage_updates {
            if updates.is_empty() {
                continue;
            }
            if selective && !fallback && !ready_addresses.contains(address) {
                if retry_counting_enabled {
                    maps_skipped += 1;
                }
                continue;
            }
            let _enter = trace_span!(target: "engine::tree::payload_processor::sparse_trie", parent: &span, "storage_trie_leaf_updates", a=%address).entered();

            let trie = self.trie.get_or_create_storage_trie_mut(*address);
            let fetched = self.fetched_storage_targets.entry(*address).or_default();
            let mut targets = Vec::new();
            let updates_len_before = updates.len();
            if retry_enabled || retry_counting_enabled {
                retry_work_items = retry_work_items.saturating_add(updates_len_before);
            }
            if retry_counting_enabled {
                maps_attempted += 1;
            }
            let result = trie.update_leaves(updates, |path, parent| match fetched.entry(path) {
                Entry::Occupied(mut entry) => {
                    if parent < *entry.get() {
                        entry.insert(parent);
                        targets.push(ProofV2Target::new(path).with_parent(parent));
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(parent);
                    targets.push(ProofV2Target::new(path).with_parent(parent));
                }
            });
            let updates_len_after = updates.len();
            let applied = updates_len_before.saturating_sub(updates_len_after);
            if retry_enabled || retry_counting_enabled {
                retry_work_outputs = retry_work_outputs.saturating_add(applied);
            }
            if let Err(error) = result {
                retry_error = Some(error);
                break 'storage
            }
            self.storage_cache_hits += applied as u64;
            self.storage_cache_misses += updates_len_after as u64;

            if !targets.is_empty() {
                self.pending_targets.extend_storage_targets(address, targets);
            }

            if selective && !fallback && should_rearm_productive_retry(applied, updates_len_after) {
                productive_requeues += 1;
                productive_ready.push(*address);
            }
        }

        drop(span);
        self.storage_retry_ready.extend(productive_ready);
        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.record_progress(
                RootProgressPhase::LeafRetry,
                retry_timer,
                retry_error.is_some(),
                retry_work_items,
                retry_work_outputs,
            );
        }
        if retry_counting_enabled {
            if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                let totals = &mut diagnostics.selective_storage_retries;
                if fallback {
                    totals.fallback_calls = totals.fallback_calls.saturating_add(1);
                    totals.fallback_maps_attempted =
                        totals.fallback_maps_attempted.saturating_add(maps_attempted as u64);
                    totals.fallback_entries_attempted =
                        totals.fallback_entries_attempted.saturating_add(retry_work_items as u64);
                    totals.fallback_entries_applied =
                        totals.fallback_entries_applied.saturating_add(retry_work_outputs as u64);
                    totals.fallback_failures =
                        totals.fallback_failures.saturating_add(u64::from(retry_error.is_some()));
                } else {
                    totals.retry_calls = totals.retry_calls.saturating_add(1);
                    totals.maps_considered =
                        totals.maps_considered.saturating_add(maps_considered as u64);
                    totals.maps_attempted =
                        totals.maps_attempted.saturating_add(maps_attempted as u64);
                    totals.maps_skipped = totals.maps_skipped.saturating_add(maps_skipped as u64);
                    totals.entries_attempted =
                        totals.entries_attempted.saturating_add(retry_work_items as u64);
                    totals.entries_applied =
                        totals.entries_applied.saturating_add(retry_work_outputs as u64);
                    totals.productive_requeues =
                        totals.productive_requeues.saturating_add(productive_requeues as u64);
                }
            }
        }
        if let Some(error) = retry_error {
            return Err(error)
        }

        // Process account trie updates and fill the account targets.
        self.process_account_leaf_updates(new)?;

        Ok(())
    }

    /// Invokes `update_leaves` for the accounts trie and collects any new targets.
    ///
    /// Returns whether any updates were drained (applied to the trie).
    #[instrument(
        level = "trace",
        target = "engine::tree::payload_processor::sparse_trie",
        skip_all
    )]
    fn process_account_leaf_updates(&mut self, new: bool) -> SparseTrieResult<bool> {
        let account_updates =
            if new { &mut self.new_account_updates } else { &mut self.account_updates };

        let updates_len_before = account_updates.len();
        let retry_timer = self
            .dispatch_diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.progress_timer(!new && updates_len_before > 0));

        let result =
            self.trie.trie_mut().update_leaves(account_updates, |target, parent| {
                match self.fetched_account_targets.entry(target) {
                    Entry::Occupied(mut entry) => {
                        if parent < *entry.get() {
                            entry.insert(parent);
                            self.pending_targets.push_account_target(
                                ProofV2Target::new(target).with_parent(parent),
                            );
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(parent);
                        self.pending_targets
                            .push_account_target(ProofV2Target::new(target).with_parent(parent));
                    }
                }
            });

        let updates_len_after = account_updates.len();
        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.record_progress(
                RootProgressPhase::AccountRetry,
                retry_timer,
                result.is_err(),
                updates_len_before,
                updates_len_before.saturating_sub(updates_len_after),
            );
        }
        result?;
        self.account_cache_hits += (updates_len_before - updates_len_after) as u64;
        self.account_cache_misses += updates_len_after as u64;

        Ok(updates_len_after < updates_len_before)
    }

    /// Computes storage roots for accounts whose storage updates are fully drained.
    ///
    /// For each storage trie T that:
    /// 1. was modified in the current block,
    /// 2. all the storage updates are fully drained,
    /// 3. but the storage root hasn't been updated yet,
    ///
    /// we trigger state root computation on a rayon pool.
    fn compute_drained_storage_roots(&mut self) {
        struct SendStorageTriePtr<S>(*mut RevealableSparseTrie<S>);
        // SAFETY: this wrapper only forwards the pointer across rayon; deref invariants are
        // documented at the use site below.
        unsafe impl<S: Send> Send for SendStorageTriePtr<S> {}

        let scan_enabled = self.dispatch_diagnostics.as_ref().is_some_and(|diagnostics| {
            diagnostics.root_tail_start.is_some() && !self.storage_updates.is_empty()
        });
        let scan_timer = RootProgressTimer::start(scan_enabled);
        let mut tries_scanned = 0usize;
        let mut tries_to_compute_roots: Vec<(B256, SendStorageTriePtr<S>)> = Vec::new();
        for (address, updates) in &self.storage_updates {
            if scan_enabled {
                tries_scanned = tries_scanned.saturating_add(1);
            }
            if updates.is_empty() &&
                let Some(trie) = self.trie.storage_tries_mut().get_mut(address) &&
                !trie.is_root_cached()
            {
                tries_to_compute_roots.push((*address, SendStorageTriePtr(trie)));
            }
        }
        let tries_to_compute = tries_to_compute_roots.len();
        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.record_progress(
                RootProgressPhase::StorageRootScan,
                scan_timer,
                false,
                tries_scanned,
                tries_to_compute,
            );
        }

        if tries_to_compute_roots.is_empty() {
            return;
        }

        let parent_span =
            debug_span!("compute_drained_storage_roots", n = tries_to_compute_roots.len());
        let new_epoch = self.new_epoch;
        let compute_timer = self
            .dispatch_diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.progress_timer(true));
        tries_to_compute_roots.into_par_iter().for_each(|(address, SendStorageTriePtr(trie))| {
            let span = if tracing::enabled!(tracing::Level::TRACE) {
                debug_span!(
                    target: "engine::tree::payload_processor::sparse_trie",
                    parent: &parent_span,
                    "storage_root",
                    ?address
                )
            } else {
                debug_span!(
                    target: "engine::tree::payload_processor::sparse_trie",
                    parent: &parent_span,
                    "storage_root",
                )
            };
            let _enter = span.entered();
            // SAFETY:
            // - pointers are created from `storage_tries_mut().get_mut(address)` above;
            // - `storage_updates` is a map, so addresses are unique;
            // - we do not insert/remove entries between pointer collection and use, so pointers
            //   stay valid and map reallocation cannot occur;
            // - each pointer is consumed by at most one rayon task, so no aliasing mutable access.
            unsafe {
                (*trie)
                    .root(new_epoch)
                    .expect("updates are drained, trie should be revealed by now")
            };
        });
        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.record_progress(
                RootProgressPhase::StorageRootCompute,
                compute_timer,
                false,
                tries_to_compute,
                tries_to_compute,
            );
        }
    }

    /// Iterates through all storage tries for which all updates were processed, computes their
    /// storage roots, and promotes corresponding pending account updates into proper leaf updates
    /// for accounts trie.
    #[instrument(
        level = "trace",
        target = "engine::tree::payload_processor::sparse_trie",
        skip_all
    )]
    fn promote_pending_account_updates(&mut self) -> SparseTrieResult<()> {
        self.process_leaf_updates(false)?;

        while self.selective_storage_retries &&
            self.finished_state_updates &&
            self.pending_targets.is_empty() &&
            self.in_flight_proof_batches == 0 &&
            self.proof_result_rx.is_empty() &&
            !self.storage_retry_ready.is_empty()
        {
            // No external event can drive the deferred productive retry. Give only the
            // addresses that made progress another normal selective turn before falling back.
            self.process_leaf_updates(false)?;
        }

        if self.selective_storage_retries &&
            self.finished_state_updates &&
            self.pending_targets.is_empty() &&
            self.in_flight_proof_batches == 0 &&
            self.proof_result_rx.is_empty() &&
            self.storage_updates.values().any(|updates| !updates.is_empty())
        {
            self.process_leaf_updates_inner(false, true)?;
        }

        if self.pending_account_updates.is_empty() {
            return Ok(());
        }

        self.compute_drained_storage_roots();

        loop {
            let pending_accounts = self.pending_account_updates.len();
            let promote_timer = self
                .dispatch_diagnostics
                .as_ref()
                .and_then(|diagnostics| diagnostics.progress_timer(pending_accounts > 0));
            let span = trace_span!("promote_updates", promoted = tracing::field::Empty).entered();
            // Now handle pending account updates that can be upgraded to a proper update.
            let account_rlp_buf = &mut self.account_rlp_buf;
            let mut num_promoted = 0;
            self.pending_account_updates.retain(|addr, account| {
                if let Some(updates) = self.storage_updates.get(addr) {
                    if !updates.is_empty() {
                        // If account has pending storage updates, it is still pending.
                        return true;
                    } else if let Some(account) = account.take() {
                        let storage_root = self.trie.storage_root(addr, self.new_epoch).expect("updates are drained, storage trie should be revealed by now");
                        let encoded = encode_account_leaf_value(account, storage_root, account_rlp_buf);
                        self.account_updates.insert(*addr, LeafUpdate::Changed(encoded));
                        num_promoted += 1;
                        return false;
                    }
                }

                // Get the current account state either from the trie or from latest account update.
                let trie_account = match self.account_updates.get(addr) {
                    Some(LeafUpdate::Changed(encoded)) => {
                        Some(encoded).filter(|encoded| !encoded.is_empty())
                    }
                    // Needs to be revealed first
                    Some(LeafUpdate::Touched) => return true,
                    None => self.trie.get_account_value(addr),
                };

                let trie_account = trie_account.map(|value| TrieAccount::decode(&mut &value[..]).expect("invalid account RLP"));

                let (account, storage_root) = if let Some(account) = account.take() {
                    // If account is Some(_) here it means it didn't have any storage updates
                    // and we can fetch the storage root directly from the account trie.
                    //
                    // If it did have storage updates, we would've had processed it above when iterating over storage tries.
                    let storage_root = trie_account.map(|account| account.storage_root).unwrap_or(EMPTY_ROOT_HASH);

                    (account, storage_root)
                } else {
                    (trie_account.map(Into::into), self.trie.storage_root(addr, self.new_epoch).expect("account had storage updates that were applied to its trie, storage root must be revealed by now"))
                };

                let encoded = encode_account_leaf_value(account, storage_root, account_rlp_buf);
                self.account_updates.insert(*addr, LeafUpdate::Changed(encoded));
                num_promoted += 1;

                false
            });
            span.record("promoted", num_promoted);
            drop(span);
            if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
                diagnostics.record_progress(
                    RootProgressPhase::AccountPromote,
                    promote_timer,
                    false,
                    pending_accounts,
                    num_promoted,
                );
            }

            // Only exit when no new updates are processed.
            //
            // We need to keep iterating if any updates are being drained because that might
            // indicate that more pending account updates can be promoted.
            if num_promoted == 0 || !self.process_account_leaf_updates(false)? {
                break
            }
        }

        Ok(())
    }

    fn dispatch_pending_targets(&mut self) -> Result<(), StateRootTaskError> {
        if self.pending_targets.is_empty() {
            return Ok(())
        }

        let dispatch_work_items = self.pending_targets.len();
        let dispatch_timer = self
            .dispatch_diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.progress_timer(true));
        let _span = trace_span!("dispatch_pending_targets").entered();
        let (targets, chunking_length) = self.pending_targets.take();
        let has_multiple_idle_account_workers =
            self.proof_worker_handle.has_multiple_idle_account_workers();
        let has_multiple_idle_storage_workers =
            self.proof_worker_handle.has_multiple_idle_storage_workers();
        let dispatch_sample = self.dispatch_diagnostics.as_ref().map(|_| {
            (
                classify_dispatch(
                    chunking_length,
                    self.chunk_size,
                    self.max_targets_for_chunking,
                    has_multiple_idle_account_workers,
                    has_multiple_idle_storage_workers,
                ),
                self.proof_worker_handle.pending_account_tasks(),
                self.proof_worker_handle.pending_storage_tasks(),
            )
        });
        let diagnostics_enabled = dispatch_sample.is_some();
        let mut chunks_dispatched = 0usize;
        let mut dispatch_error = None;
        dispatch_with_chunking(
            targets,
            chunking_length,
            self.chunk_size,
            self.max_targets_for_chunking,
            has_multiple_idle_account_workers,
            has_multiple_idle_storage_workers,
            MultiProofTargetsV2::chunks,
            |proof_targets| {
                if dispatch_error.is_some() {
                    return;
                }

                match self.proof_worker_handle.dispatch_account_multiproof(AccountMultiproofInput {
                    targets: proof_targets,
                    proof_result_sender: ProofResultContext::new(
                        self.proof_result_tx.clone(),
                        HashedPostState::default(),
                        Instant::now(),
                    ),
                }) {
                    Ok(()) => {
                        if diagnostics_enabled {
                            chunks_dispatched += 1;
                        }
                        self.in_flight_proof_batches += 1;
                    }
                    Err(e) => {
                        error!("failed to dispatch account multiproof: {e:?}");
                        dispatch_error = Some(StateRootTaskError::ProofDispatch(e));
                    }
                }
            },
        );

        if let Some((decision, account_queue_depth, storage_queue_depth)) = dispatch_sample {
            let account_queue_high_water = self.proof_worker_handle.pending_account_tasks();
            let storage_queue_high_water = self.proof_worker_handle.pending_storage_tasks();
            let diagnostics = self
                .dispatch_diagnostics
                .as_mut()
                .expect("dispatch sample requires enabled diagnostics");
            diagnostics.record_dispatch(
                decision,
                chunking_length,
                chunks_dispatched,
                account_queue_depth,
                storage_queue_depth,
                account_queue_high_water,
                storage_queue_high_water,
                self.in_flight_proof_batches,
            );
        }

        if let Some(diagnostics) = self.dispatch_diagnostics.as_mut() {
            diagnostics.record_progress(
                RootProgressPhase::Dispatch,
                dispatch_timer,
                dispatch_error.is_some(),
                dispatch_work_items,
                chunks_dispatched,
            );
        }
        if let Some(error) = dispatch_error {
            return Err(error)
        }

        Ok(())
    }

    fn has_pending_sparse_trie_updates(&self) -> bool {
        !self.account_updates.is_empty() ||
            self.storage_updates.values().any(|updates| !updates.is_empty()) ||
            !self.pending_account_updates.is_empty()
    }

    /// Errors when pending trie updates remain but nothing can deliver them: no update
    /// messages are queued, no proof targets are queued or in flight, and no proof results
    /// are waiting.
    ///
    /// `updates_queued` is passed in instead of reading `self.updates` directly, because in
    /// the draining phase the updates channel is not read anymore and may hold ignored late
    /// hints that must not mask a stall.
    fn ensure_not_stalled(&self, updates_queued: bool) -> Result<(), StateRootTaskError> {
        if self.finished_state_updates &&
            !updates_queued &&
            self.pending_updates == 0 &&
            self.pending_targets.is_empty() &&
            self.in_flight_proof_batches == 0 &&
            self.proof_result_rx.is_empty() &&
            self.has_pending_sparse_trie_updates()
        {
            const MAX_STALLED_PROOF_TARGETS_TO_LOG: usize = 5;

            let mut account_targets = self
                .account_updates
                .keys()
                .map(|target| (*target, self.fetched_account_targets.get(target).copied()))
                .collect::<Vec<_>>();
            account_targets.sort_unstable();
            let account_targets_truncated =
                account_targets.len().saturating_sub(MAX_STALLED_PROOF_TARGETS_TO_LOG);
            account_targets.truncate(MAX_STALLED_PROOF_TARGETS_TO_LOG);

            let mut storage_targets = self
                .storage_updates
                .iter()
                .flat_map(|(address, updates)| {
                    let fetched_targets = self.fetched_storage_targets.get(address);
                    updates.keys().map(move |target| {
                        (
                            *address,
                            *target,
                            fetched_targets.and_then(|targets| targets.get(target)).copied(),
                        )
                    })
                })
                .collect::<Vec<_>>();
            storage_targets.sort_unstable();
            let storage_targets_truncated =
                storage_targets.len().saturating_sub(MAX_STALLED_PROOF_TARGETS_TO_LOG);
            storage_targets.truncate(MAX_STALLED_PROOF_TARGETS_TO_LOG);

            error!(
                ?account_targets,
                account_targets_truncated,
                ?storage_targets,
                storage_targets_truncated,
                "sparse trie task stalled: pending updates remain but no proof targets are queued or in flight"
            );

            return Err(StateRootTaskError::Stalled)
        }

        Ok(())
    }
}

/// Metrics recorded by sparse trie and hashing tasks.
#[derive(Metrics, Clone)]
#[metrics(scope = "tree.root")]
pub(super) struct SparseTrieTaskMetrics {
    /// Histogram of durations spent revealing multiproof results into the sparse trie.
    pub(super) sparse_trie_reveal_multiproof_duration_histogram: Histogram,
    /// Histogram of durations spent coalescing multiple proof results from the channel.
    pub(super) sparse_trie_proof_coalesce_duration_histogram: Histogram,
    /// Histogram of durations the event loop spent blocked waiting on channels.
    pub(super) sparse_trie_channel_wait_duration_histogram: Histogram,
    /// Histogram of durations spent processing trie updates and promoting pending accounts.
    pub(super) sparse_trie_process_updates_duration_histogram: Histogram,
    /// Histogram of sparse trie final update durations.
    pub(super) sparse_trie_final_update_duration_histogram: Histogram,
    /// Histogram of sparse trie total durations.
    pub(super) sparse_trie_total_duration_histogram: Histogram,
    /// Time spent preparing the sparse trie for reuse after state root computation.
    pub(super) into_trie_for_reuse_duration_histogram: Histogram,
    /// Time spent pruning the sparse trie by node epoch.
    pub(super) sparse_trie_prune_duration_histogram: Histogram,
    /// Time spent waiting for preserved sparse trie cache to become available.
    pub(super) sparse_trie_cache_wait_duration_histogram: Histogram,
    /// Histogram for sparse trie task idle time in seconds (waiting for updates or proof
    /// results). Excludes the final wait after the channel is closed.
    pub(super) sparse_trie_idle_time_seconds: Histogram,
    /// Histogram for hashing task idle time in seconds (waiting for messages from execution).
    /// Excludes the final wait after the channel is closed.
    pub(super) hashing_task_idle_time_seconds: Histogram,

    /// Number of account leaf updates applied without needing a new proof (cache hits).
    pub(super) sparse_trie_account_cache_hits: Histogram,
    /// Number of account leaf updates that required a new proof (cache misses).
    pub(super) sparse_trie_account_cache_misses: Histogram,
    /// Number of storage leaf updates applied without needing a new proof (cache hits).
    pub(super) sparse_trie_storage_cache_hits: Histogram,
    /// Number of storage leaf updates that required a new proof (cache misses).
    pub(super) sparse_trie_storage_cache_misses: Histogram,

    /// Number of storage tries retained in the preserved sparse trie cache.
    pub(super) sparse_trie_retained_storage_tries: Gauge,
}

/// The default max targets, for limiting the number of account and storage proof targets to be
/// fetched by a single worker. If exceeded, chunking is forced regardless of worker availability.
const DEFAULT_MAX_TARGETS_FOR_CHUNKING: usize = 300;
const SELECTIVE_STORAGE_RETRIES_ENV: &str = "RETH_EXPERIMENTAL_SELECTIVE_STORAGE_RETRIES";

/// Start proof fetching while the first state-update batch is still arriving.
const INITIAL_UPDATE_BATCH_SIZE: usize = 64;

fn selective_storage_retries_enabled() -> bool {
    selective_storage_retries_enabled_value(
        std::env::var_os(SELECTIVE_STORAGE_RETRIES_ENV).as_deref(),
    )
}

fn selective_storage_retries_enabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

fn mark_storage_retries_ready(
    ready: &mut B256Set,
    storage_updates: &B256Map<B256Map<LeafUpdate>>,
    addresses: impl IntoIterator<Item = B256>,
) -> usize {
    addresses
        .into_iter()
        .filter(|address| {
            storage_updates.get(address).is_some_and(|updates| !updates.is_empty()) &&
                ready.insert(*address)
        })
        .count()
}

const fn should_rearm_productive_retry(applied: usize, remaining: usize) -> bool {
    applied > 0 && remaining > 0
}

/// Why a pending target set was split, or why it remained a single batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum ProofDispatchReason {
    Unsplit = 0,
    Force = 1,
    AccountIdle = 2,
    StorageIdle = 3,
}

/// Disjoint `make_progress` call intervals after the final update marker. Phase wall time can
/// include Rayon work and join waits; caller CPU only measures the sparse-trie thread.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum RootProgressPhase {
    /// Buffered channel messages consumed / consumed (includes finish and prefetch messages).
    NewUpdates = 1,
    /// Pending storage leaf entries attempted / applied.
    LeafRetry = 2,
    /// Storage tries inspected / selected for root calculation.
    StorageRootScan = 3,
    /// Selected storage tries / roots calculated; caller CPU excludes Rayon workers.
    StorageRootCompute = 4,
    /// Pending accounts inspected / promoted.
    AccountPromote = 5,
    /// Pending account leaf entries attempted / applied.
    AccountRetry = 6,
    /// Speculative account-hash invocations / no enumerable output.
    SpeculativeHash = 7,
    /// Proof targets offered / chunks dispatched.
    Dispatch = 8,
}

impl RootProgressPhase {
    const COUNT: usize = 8;

    const fn index(self) -> usize {
        self as usize - 1
    }
}

struct RootProgressTimer {
    wall: Instant,
    resources: ThreadResourceUsage,
}

impl RootProgressTimer {
    fn start(enabled: bool) -> Option<Self> {
        enabled.then(|| Self { wall: Instant::now(), resources: ThreadResourceUsage::now() })
    }

    fn finish(self) -> RootProgressMeasurement {
        RootProgressMeasurement {
            wall_ns: duration_ns(self.wall.elapsed()),
            resources: self.resources.elapsed(),
        }
    }
}

struct RootProgressMeasurement {
    wall_ns: u64,
    resources: Option<ThreadResourceUsageDelta>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RootProgressTotals {
    wall_ns: u64,
    cpu_measured_wall_ns: u64,
    caller_cpu_ns: u64,
    cpu_measured_calls: u64,
    cpu_missing_calls: u64,
    calls: u64,
    failures: u64,
    work_items: u64,
    work_outputs: u64,
    work_items_max: u64,
    minor_faults: u64,
    major_faults: u64,
    voluntary_context_switches: u64,
    involuntary_context_switches: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SelectiveStorageRetryTotals {
    enabled: u64,
    retry_calls: u64,
    maps_considered: u64,
    maps_attempted: u64,
    maps_skipped: u64,
    entries_attempted: u64,
    entries_applied: u64,
    ready_from_proof: u64,
    ready_from_input: u64,
    productive_requeues: u64,
    fallback_calls: u64,
    fallback_maps_attempted: u64,
    fallback_entries_attempted: u64,
    fallback_entries_applied: u64,
    fallback_failures: u64,
}

impl RootProgressTotals {
    fn record(
        &mut self,
        measurement: RootProgressMeasurement,
        failed: bool,
        work_items: usize,
        work_outputs: usize,
    ) {
        self.wall_ns = self.wall_ns.saturating_add(measurement.wall_ns);
        self.calls = self.calls.saturating_add(1);
        self.failures = self.failures.saturating_add(u64::from(failed));
        let work_items = work_items as u64;
        self.work_items = self.work_items.saturating_add(work_items);
        self.work_outputs = self.work_outputs.saturating_add(work_outputs as u64);
        self.work_items_max = self.work_items_max.max(work_items);
        if let Some(resources) = measurement.resources {
            self.cpu_measured_wall_ns =
                self.cpu_measured_wall_ns.saturating_add(measurement.wall_ns);
            self.caller_cpu_ns = self.caller_cpu_ns.saturating_add(duration_ns(
                resources.user_cpu_time.saturating_add(resources.system_cpu_time),
            ));
            self.cpu_measured_calls = self.cpu_measured_calls.saturating_add(1);
            self.minor_faults = self.minor_faults.saturating_add(resources.minor_page_faults);
            self.major_faults = self.major_faults.saturating_add(resources.major_page_faults);
            self.voluntary_context_switches = self
                .voluntary_context_switches
                .saturating_add(resources.voluntary_context_switches);
            self.involuntary_context_switches = self
                .involuntary_context_switches
                .saturating_add(resources.involuntary_context_switches);
        } else {
            self.cpu_missing_calls = self.cpu_missing_calls.saturating_add(1);
        }
    }
}

fn classify_dispatch(
    chunking_len: usize,
    chunk_size: usize,
    max_targets_for_chunking: usize,
    has_multiple_idle_account_workers: bool,
    has_multiple_idle_storage_workers: bool,
) -> ProofDispatchReason {
    if chunking_len <= chunk_size {
        return ProofDispatchReason::Unsplit
    }
    if chunking_len > max_targets_for_chunking {
        return ProofDispatchReason::Force
    }

    let has_full_chunks = chunking_len >= chunk_size.saturating_mul(2);
    if has_full_chunks && has_multiple_idle_account_workers {
        ProofDispatchReason::AccountIdle
    } else if has_full_chunks && has_multiple_idle_storage_workers {
        ProofDispatchReason::StorageIdle
    } else {
        ProofDispatchReason::Unsplit
    }
}

/// Per-root, identity-free proof admission diagnostics.
struct ProofDispatchDiagnostics {
    parent: tracing::Span,
    dispatches: u64,
    targets: u64,
    chunks: u64,
    reason_counts: [u64; 4],
    queue_samples: u64,
    account_queue_high_water: u64,
    storage_queue_high_water: u64,
    account_queue_depth_bins: [u64; 4],
    storage_queue_depth_bins: [u64; 4],
    split_reason_account_queue_nonempty: [u64; 3],
    split_reason_storage_queue_nonempty: [u64; 3],
    split_when_queue_nonempty: u64,
    split_when_storage_queue_nonempty: u64,
    outstanding_max: u64,
    root_tail_start: Option<Instant>,
    root_tail_emitted: bool,
    proof_wait_ns: u64,
    proof_wait_count: u64,
    proof_wait_max_ns: u64,
    proof_wait_last_ns: u64,
    result_drain_ns: u64,
    result_drain_count: u64,
    result_messages_consumed: u64,
    result_last_drain_count: u64,
    result_queue_after_last_drain: u64,
    in_flight_after_last_drain: u64,
    last_result_consumed_at: Option<Instant>,
    reveal_ns: u64,
    progress_ns: u64,
    final_root_ns: u64,
    progress_totals: [RootProgressTotals; RootProgressPhase::COUNT],
    progress_emitted: bool,
    selective_storage_retries: SelectiveStorageRetryTotals,
}

impl ProofDispatchDiagnostics {
    fn new(selective_storage_retries: bool) -> Option<Self> {
        reth_tracing::readiness::enabled().then(|| {
            let mut diagnostics = Self::empty(tracing::Span::current());
            diagnostics.selective_storage_retries.enabled = u64::from(selective_storage_retries);
            diagnostics
        })
    }

    fn empty(parent: tracing::Span) -> Self {
        Self {
            parent,
            dispatches: 0,
            targets: 0,
            chunks: 0,
            reason_counts: [0; 4],
            queue_samples: 0,
            account_queue_high_water: 0,
            storage_queue_high_water: 0,
            account_queue_depth_bins: [0; 4],
            storage_queue_depth_bins: [0; 4],
            split_reason_account_queue_nonempty: [0; 3],
            split_reason_storage_queue_nonempty: [0; 3],
            split_when_queue_nonempty: 0,
            split_when_storage_queue_nonempty: 0,
            outstanding_max: 0,
            root_tail_start: None,
            root_tail_emitted: false,
            proof_wait_ns: 0,
            proof_wait_count: 0,
            proof_wait_max_ns: 0,
            proof_wait_last_ns: 0,
            result_drain_ns: 0,
            result_drain_count: 0,
            result_messages_consumed: 0,
            result_last_drain_count: 0,
            result_queue_after_last_drain: 0,
            in_flight_after_last_drain: 0,
            last_result_consumed_at: None,
            reveal_ns: 0,
            progress_ns: 0,
            final_root_ns: 0,
            progress_totals: [RootProgressTotals::default(); RootProgressPhase::COUNT],
            progress_emitted: false,
            selective_storage_retries: SelectiveStorageRetryTotals::default(),
        }
    }

    fn start_root_tail(&mut self) {
        self.root_tail_start.get_or_insert_with(Instant::now);
    }

    fn progress_timer(&self, has_work: bool) -> Option<RootProgressTimer> {
        RootProgressTimer::start(has_work && self.root_tail_start.is_some())
    }

    fn record_progress(
        &mut self,
        phase: RootProgressPhase,
        timer: Option<RootProgressTimer>,
        failed: bool,
        work_items: usize,
        work_outputs: usize,
    ) {
        if let Some(timer) = timer {
            self.progress_totals[phase.index()].record(
                timer.finish(),
                failed,
                work_items,
                work_outputs,
            );
        }
    }

    fn emit_progress(&mut self) {
        if self.progress_emitted {
            return
        }
        self.progress_emitted = true;
        for (index, totals) in self.progress_totals.iter().enumerate() {
            if totals.calls == 0 {
                continue
            }
            tracing::info!(
                target: "lifecycle",
                parent: &self.parent,
                stage = "proof_progress_totals",
                phase = (index + 1) as u64,
                wall_ns = totals.wall_ns,
                cpu_measured_wall_ns = totals.cpu_measured_wall_ns,
                caller_cpu_ns = totals.caller_cpu_ns,
                cpu_measured_calls = totals.cpu_measured_calls,
                cpu_missing_calls = totals.cpu_missing_calls,
                calls = totals.calls,
                failures = totals.failures,
                work_items = totals.work_items,
                work_outputs = totals.work_outputs,
                work_items_max = totals.work_items_max,
                minor_faults = totals.minor_faults,
                major_faults = totals.major_faults,
                voluntary_context_switches = totals.voluntary_context_switches,
                involuntary_context_switches = totals.involuntary_context_switches,
            );
        }
    }

    fn record_proof_wait(&mut self, elapsed: Duration) {
        let ns = duration_ns(elapsed);
        self.proof_wait_ns = self.proof_wait_ns.saturating_add(ns);
        self.proof_wait_count = self.proof_wait_count.saturating_add(1);
        self.proof_wait_max_ns = self.proof_wait_max_ns.max(ns);
        self.proof_wait_last_ns = ns;
    }

    fn record_result_drain(
        &mut self,
        elapsed: Duration,
        result_count: u64,
        result_queue_after_drain: usize,
        in_flight_after_drain: usize,
        last_result_consumed_at: Instant,
    ) {
        self.result_drain_ns = self.result_drain_ns.saturating_add(duration_ns(elapsed));
        self.result_drain_count = self.result_drain_count.saturating_add(1);
        self.result_messages_consumed = self.result_messages_consumed.saturating_add(result_count);
        self.result_last_drain_count = result_count;
        self.result_queue_after_last_drain = result_queue_after_drain as u64;
        self.in_flight_after_last_drain = in_flight_after_drain as u64;
        self.last_result_consumed_at = Some(last_result_consumed_at);
    }

    fn record_reveal(&mut self, elapsed: Duration) {
        self.reveal_ns = self.reveal_ns.saturating_add(duration_ns(elapsed));
    }

    fn record_root_progress(&mut self, elapsed: Duration) {
        self.progress_ns = self.progress_ns.saturating_add(duration_ns(elapsed));
    }

    fn record_final_root(&mut self, elapsed: Duration) {
        self.final_root_ns = self.final_root_ns.saturating_add(duration_ns(elapsed));
    }

    fn emit_root_tail(&mut self, result_ready: bool, success: bool) {
        self.emit_progress();
        if self.root_tail_emitted {
            return
        }
        self.root_tail_emitted = true;
        let now = Instant::now();
        let root_tail_ns =
            self.root_tail_start.map_or(0, |start| duration_ns(now.duration_since(start)));
        let phase_accounted_ns = self
            .proof_wait_ns
            .saturating_add(self.result_drain_ns)
            .saturating_add(self.reveal_ns)
            .saturating_add(self.progress_ns)
            .saturating_add(self.final_root_ns);
        let phase_residual_ns = root_tail_ns.saturating_sub(phase_accounted_ns);
        let phase_coverage_ppm = if root_tail_ns == 0 {
            0
        } else {
            phase_accounted_ns.saturating_mul(1_000_000).checked_div(root_tail_ns).unwrap_or(0)
        };
        let last_result_consumed_to_root_ready_ns =
            self.last_result_consumed_at.map_or(0, |last| duration_ns(now.duration_since(last)));
        tracing::info!(
            target: "lifecycle",
            parent: &self.parent,
            stage = "proof_root_tail_totals",
            root_tail_started = u64::from(self.root_tail_start.is_some()),
            root_result_ready = u64::from(result_ready),
            root_success = u64::from(success),
            root_tail_ns,
            proof_wait_ns = self.proof_wait_ns,
            proof_wait_count = self.proof_wait_count,
            proof_wait_max_ns = self.proof_wait_max_ns,
            proof_wait_last_ns = self.proof_wait_last_ns,
            result_drain_ns = self.result_drain_ns,
            result_drain_count = self.result_drain_count,
            result_messages_consumed = self.result_messages_consumed,
            result_last_drain_count = self.result_last_drain_count,
            result_queue_after_last_drain = self.result_queue_after_last_drain,
            in_flight_after_last_drain = self.in_flight_after_last_drain,
            reveal_ns = self.reveal_ns,
            progress_ns = self.progress_ns,
            final_root_ns = self.final_root_ns,
            phase_accounted_ns,
            phase_residual_ns,
            phase_coverage_ppm,
            had_result_after_finish = u64::from(self.last_result_consumed_at.is_some()),
            last_result_consumed_to_root_ready_ns,
        );
    }

    #[expect(clippy::too_many_arguments)]
    fn record_dispatch(
        &mut self,
        reason: ProofDispatchReason,
        targets: usize,
        chunks: usize,
        account_queue_depth: usize,
        storage_queue_depth: usize,
        account_queue_high_water: usize,
        storage_queue_high_water: usize,
        outstanding: usize,
    ) {
        self.dispatches = self.dispatches.saturating_add(1);
        self.targets = self.targets.saturating_add(targets as u64);
        self.chunks = self.chunks.saturating_add(chunks as u64);
        self.reason_counts[reason as usize] = self.reason_counts[reason as usize].saturating_add(1);
        self.queue_samples = self.queue_samples.saturating_add(1);
        self.account_queue_high_water = self
            .account_queue_high_water
            .max(account_queue_depth.max(account_queue_high_water) as u64);
        self.storage_queue_high_water = self
            .storage_queue_high_water
            .max(storage_queue_depth.max(storage_queue_high_water) as u64);
        self.account_queue_depth_bins[queue_depth_bin(account_queue_depth)] =
            self.account_queue_depth_bins[queue_depth_bin(account_queue_depth)].saturating_add(1);
        self.storage_queue_depth_bins[queue_depth_bin(storage_queue_depth)] =
            self.storage_queue_depth_bins[queue_depth_bin(storage_queue_depth)].saturating_add(1);
        if let Some(split_reason) = split_reason_index(reason) {
            if account_queue_depth > 0 {
                self.split_when_queue_nonempty = self.split_when_queue_nonempty.saturating_add(1);
                self.split_reason_account_queue_nonempty[split_reason] =
                    self.split_reason_account_queue_nonempty[split_reason].saturating_add(1);
            }
            if storage_queue_depth > 0 {
                self.split_when_storage_queue_nonempty =
                    self.split_when_storage_queue_nonempty.saturating_add(1);
                self.split_reason_storage_queue_nonempty[split_reason] =
                    self.split_reason_storage_queue_nonempty[split_reason].saturating_add(1);
            }
        }
        self.outstanding_max = self.outstanding_max.max(outstanding as u64);
    }

    fn emit(&self) {
        tracing::info!(
            target: "lifecycle",
            parent: &self.parent,
            stage = "proof_dispatch_totals",
            dispatches = self.dispatches,
            targets = self.targets,
            chunks = self.chunks,
            reason_unsplit = self.reason_counts[ProofDispatchReason::Unsplit as usize],
            reason_force = self.reason_counts[ProofDispatchReason::Force as usize],
            reason_account_idle = self.reason_counts[ProofDispatchReason::AccountIdle as usize],
            reason_storage_idle = self.reason_counts[ProofDispatchReason::StorageIdle as usize],
            queue_samples = self.queue_samples,
            account_queue_high_water = self.account_queue_high_water,
            storage_queue_high_water = self.storage_queue_high_water,
            account_queue_depth_0 = self.account_queue_depth_bins[0],
            account_queue_depth_1_8 = self.account_queue_depth_bins[1],
            account_queue_depth_9_32 = self.account_queue_depth_bins[2],
            account_queue_depth_33_plus = self.account_queue_depth_bins[3],
            storage_queue_depth_0 = self.storage_queue_depth_bins[0],
            storage_queue_depth_1_8 = self.storage_queue_depth_bins[1],
            storage_queue_depth_9_32 = self.storage_queue_depth_bins[2],
            storage_queue_depth_33_plus = self.storage_queue_depth_bins[3],
            split_when_queue_nonempty = self.split_when_queue_nonempty,
            split_when_storage_queue_nonempty = self.split_when_storage_queue_nonempty,
            split_force_account_queue_nonempty = self.split_reason_account_queue_nonempty[0],
            split_account_idle_account_queue_nonempty = self.split_reason_account_queue_nonempty[1],
            split_storage_idle_account_queue_nonempty = self.split_reason_account_queue_nonempty[2],
            split_force_storage_queue_nonempty = self.split_reason_storage_queue_nonempty[0],
            split_account_idle_storage_queue_nonempty = self.split_reason_storage_queue_nonempty[1],
            split_storage_idle_storage_queue_nonempty = self.split_reason_storage_queue_nonempty[2],
            outstanding_max = self.outstanding_max,
        );
        let retries = &self.selective_storage_retries;
        tracing::info!(
            target: "lifecycle",
            parent: &self.parent,
            stage = "selective_storage_retry_totals",
            enabled = retries.enabled,
            retry_calls = retries.retry_calls,
            maps_considered = retries.maps_considered,
            maps_attempted = retries.maps_attempted,
            maps_skipped = retries.maps_skipped,
            entries_attempted = retries.entries_attempted,
            entries_applied = retries.entries_applied,
            ready_from_proof = retries.ready_from_proof,
            ready_from_input = retries.ready_from_input,
            productive_requeues = retries.productive_requeues,
            fallback_calls = retries.fallback_calls,
            fallback_maps_attempted = retries.fallback_maps_attempted,
            fallback_entries_attempted = retries.fallback_entries_attempted,
            fallback_entries_applied = retries.fallback_entries_applied,
            fallback_failures = retries.fallback_failures,
        );
    }

    fn emit_updates_finished_snapshot(
        &self,
        in_flight: usize,
        pending_account_targets: usize,
        pending_storage_targets: usize,
        account_queue_depth: usize,
        storage_queue_depth: usize,
        result_queue_depth: usize,
    ) {
        tracing::info!(
            target: "lifecycle",
            parent: &self.parent,
            stage = "proof_state_at_updates_finished",
            in_flight = in_flight as u64,
            pending_targets = pending_account_targets.saturating_add(pending_storage_targets) as u64,
            pending_account_targets = pending_account_targets as u64,
            pending_storage_targets = pending_storage_targets as u64,
            account_queue_depth = account_queue_depth as u64,
            storage_queue_depth = storage_queue_depth as u64,
            result_queue_depth = result_queue_depth as u64,
        );
    }
}

const fn split_reason_index(reason: ProofDispatchReason) -> Option<usize> {
    match reason {
        ProofDispatchReason::Force => Some(0),
        ProofDispatchReason::AccountIdle => Some(1),
        ProofDispatchReason::StorageIdle => Some(2),
        ProofDispatchReason::Unsplit => None,
    }
}

impl Drop for ProofDispatchDiagnostics {
    fn drop(&mut self) {
        self.emit_root_tail(false, false);
        self.emit();
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

const fn queue_depth_bin(depth: usize) -> usize {
    match depth {
        0 => 0,
        1..=8 => 1,
        9..=32 => 2,
        _ => 3,
    }
}

/// Dispatches work items as a single unit or in chunks based on target size and worker
/// availability.
#[expect(clippy::too_many_arguments)]
fn dispatch_with_chunking<T, I>(
    items: T,
    chunking_len: usize,
    chunk_size: usize,
    max_targets_for_chunking: usize,
    has_multiple_idle_account_workers: bool,
    has_multiple_idle_storage_workers: bool,
    chunker: impl FnOnce(T, usize) -> I,
    mut dispatch: impl FnMut(T),
) where
    I: IntoIterator<Item = T>,
{
    let should_chunk = classify_dispatch(
        chunking_len,
        chunk_size,
        max_targets_for_chunking,
        has_multiple_idle_account_workers,
        has_multiple_idle_storage_workers,
    ) != ProofDispatchReason::Unsplit;

    if should_chunk && chunking_len > chunk_size {
        for chunk in chunker(items, chunk_size) {
            dispatch(chunk);
        }
        return;
    }

    dispatch(items);
}

/// RLP-encodes the account as a [`TrieAccount`] leaf value, or returns empty for deletions.
///
/// `Some(Account::default())` with an empty storage root is encoded as a deletion. This is valid
/// for post-Merge state because EIP-7523 (<https://eips.ethereum.org/EIPS/eip-7523>) prohibits
/// empty accounts. Do not use this encoding rule when replaying historical pre-Merge state, where
/// an empty account and a missing account can have different trie representations.
fn encode_account_leaf_value(
    account: Option<Account>,
    storage_root: B256,
    account_rlp_buf: &mut Vec<u8>,
) -> Vec<u8> {
    if account.is_none_or(|account| account.is_empty()) && storage_root == EMPTY_ROOT_HASH {
        return Vec::new();
    }

    account_rlp_buf.clear();
    account.unwrap_or_default().into_trie_account(storage_root).encode(account_rlp_buf);
    account_rlp_buf.clone()
}

/// Pending proof targets queued for dispatch to proof workers, along with their count.
#[derive(Default)]
struct PendingTargets {
    /// The proof targets.
    targets: MultiProofTargetsV2,
    /// Number of account + storage proof targets currently queued.
    len: usize,
}

impl PendingTargets {
    /// Returns the number of pending targets.
    const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if there are no pending targets.
    const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of pending account targets.
    fn account_len(&self) -> usize {
        self.targets.account_targets.len()
    }

    /// Returns the number of pending storage targets.
    fn storage_len(&self) -> usize {
        self.targets.storage_targets.values().map(Vec::len).sum()
    }

    /// Takes the pending targets, replacing with empty defaults.
    fn take(&mut self) -> (MultiProofTargetsV2, usize) {
        (std::mem::take(&mut self.targets), std::mem::take(&mut self.len))
    }

    /// Adds a target to the account targets.
    fn push_account_target(&mut self, target: ProofV2Target) {
        self.targets.account_targets.push(target);
        self.len += 1;
    }

    /// Extends storage targets for the given address.
    fn extend_storage_targets(&mut self, address: &B256, targets: Vec<ProofV2Target>) {
        self.len += targets.len();
        self.targets.storage_targets.entry(*address).or_default().extend(targets);
    }
}

/// Message type for the sparse trie task.
enum SparseTrieTaskMessage {
    /// A hashed state update ready to be processed.
    HashedState(HashedPostState),
    /// Prefetch proof targets (passed through directly).
    PrefetchProofs(MultiProofTargetsV2),
    /// Signals that all state updates have been received.
    FinishedStateUpdates,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    use alloy_primitives::{keccak256, Address, B256, U256};
    use reth_db::{cursor::DbCursorRO, tables, transaction::DbTx};
    use reth_db_common::init::init_genesis;
    use reth_provider::{test_utils::create_test_provider_factory, TrieWriter};
    use reth_storage_overlay::{OverlayManager, OverlayStateProviderFactory};
    use reth_trie_common::{BranchNodeMasks, Nibbles, ProofTrieNodeV2, TrieNodeV2};
    use reth_trie_parallel::proof_task::ProofTaskCtx;
    use reth_trie_sparse::{
        ArenaParallelSparseTrie, LeafLookup, LeafLookupError, SparseTrieUpdates,
    };

    /// Test trie that makes one update productive per call without requesting another proof.
    #[derive(Clone, Debug, Default)]
    struct OneLeafPerRetryTrie;

    impl SparseTrie for OneLeafPerRetryTrie {
        fn set_root(
            &mut self,
            _root: TrieNodeV2,
            _masks: Option<BranchNodeMasks>,
            _retain_updates: bool,
        ) -> SparseTrieResult<()> {
            Ok(())
        }

        fn set_updates(&mut self, _retain_updates: bool) {}

        fn reveal_nodes(&mut self, _nodes: &mut [ProofTrieNodeV2]) -> SparseTrieResult<()> {
            Ok(())
        }

        fn root(&mut self, _new_epoch: TrieNodeEpoch) -> B256 {
            EMPTY_ROOT_HASH
        }

        fn is_root_cached(&self) -> bool {
            true
        }

        fn root_epoch(&self) -> Option<TrieNodeEpoch> {
            Some(TrieNodeEpoch::UNMODIFIED)
        }

        fn update_subtrie_hashes(&mut self, _new_epoch: TrieNodeEpoch) {}

        fn get_leaf_value(&self, _full_path: &Nibbles) -> Option<&Vec<u8>> {
            None
        }

        fn find_leaf(
            &self,
            _full_path: &Nibbles,
            _expected_value: Option<&Vec<u8>>,
        ) -> Result<LeafLookup, LeafLookupError> {
            Ok(LeafLookup::NonExistent)
        }

        fn updates_ref(&self) -> Cow<'_, SparseTrieUpdates> {
            Cow::Owned(SparseTrieUpdates::default())
        }

        fn take_updates(&mut self) -> SparseTrieUpdates {
            SparseTrieUpdates::default()
        }

        fn clear(&mut self) {}

        fn prune(&mut self, _prune_before: TrieNodeEpoch) -> usize {
            0
        }

        fn update_leaves(
            &mut self,
            updates: &mut B256Map<LeafUpdate>,
            _proof_required_fn: impl FnMut(B256, ProofV2TargetParent),
        ) -> SparseTrieResult<()> {
            if let Some(key) = updates.keys().next().copied() {
                updates.remove(&key);
            }
            Ok(())
        }
    }

    fn drain_sparse_trie_tasks(runtime: &Runtime) {
        for task_name in ["trie-hashing", "storage-workers", "account-workers"] {
            runtime.spawn_blocking_named(task_name, || {}).get();
        }
    }

    #[test]
    fn proof_dispatch_reason_classifies_without_changing_chunk_thresholds() {
        assert_eq!(classify_dispatch(5, 5, 300, true, true), ProofDispatchReason::Unsplit);
        assert_eq!(classify_dispatch(301, 5, 300, false, false), ProofDispatchReason::Force);
        assert_eq!(classify_dispatch(10, 5, 300, true, true), ProofDispatchReason::AccountIdle);
        assert_eq!(classify_dispatch(10, 5, 300, false, true), ProofDispatchReason::StorageIdle);
        assert_eq!(classify_dispatch(9, 5, 300, true, true), ProofDispatchReason::Unsplit);
    }

    #[test]
    fn selective_storage_retries_require_exact_opt_in() {
        use std::ffi::OsStr;

        assert!(!selective_storage_retries_enabled_value(None));
        assert!(selective_storage_retries_enabled_value(Some(OsStr::new("1"))));
        for disabled in ["", "0", "true", "yes", "2"] {
            assert!(!selective_storage_retries_enabled_value(Some(OsStr::new(disabled))));
        }
    }

    #[test]
    fn selective_storage_retry_readiness_is_order_independent_and_productive() {
        let address_a = B256::repeat_byte(0x0a);
        let address_b = B256::repeat_byte(0x0b);
        let empty_address = B256::repeat_byte(0x0c);
        let slot = B256::repeat_byte(0x01);
        let mut updates = B256Map::default();
        updates.insert(address_a, B256Map::from_iter([(slot, LeafUpdate::Touched)]));
        updates.insert(address_b, B256Map::from_iter([(slot, LeafUpdate::Touched)]));
        updates.insert(empty_address, B256Map::default());

        let mut forward = B256Set::default();
        assert_eq!(
            mark_storage_retries_ready(
                &mut forward,
                &updates,
                [address_a, empty_address, address_b, address_a],
            ),
            2
        );
        let mut reverse = B256Set::default();
        assert_eq!(
            mark_storage_retries_ready(
                &mut reverse,
                &updates,
                [address_b, address_a, empty_address],
            ),
            2
        );
        assert_eq!(forward, reverse);
        assert!(should_rearm_productive_retry(1, 1));
        assert!(!should_rearm_productive_retry(0, 1));
        assert!(!should_rearm_productive_retry(1, 0));
    }

    #[test]
    fn terminal_productive_retry_reaches_fixed_point_before_fallback() {
        let runtime = reth_tasks::Runtime::test();
        let provider_factory = create_test_provider_factory();
        let anchor_hash = init_genesis(&provider_factory).expect("initialize genesis");
        let state_provider_factory = OverlayStateProviderFactory::new(
            provider_factory,
            OverlayManager::<reth_chain_state::EthPrimitives>::default()
                .overlay_builder(anchor_hash),
        );
        let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
        let proof_worker_handle = ProofWorkerHandle::new(
            &runtime,
            ProofTaskCtx::new(state_provider_factory),
            false,
            proof_result_tx.clone(),
        );
        let trie = SparseStateTrie::default()
            .with_accounts_trie(RevealableSparseTrie::<ArenaParallelSparseTrie>::revealed_empty())
            .with_default_storage_trie(
                RevealableSparseTrie::<OneLeafPerRetryTrie>::revealed_empty(),
            );
        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (_cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
        let (hashed_state_tx, _hashed_state_rx) = std::sync::mpsc::channel();
        let mut task = SparseTrieCacheTask::new_with_trie(
            &runtime,
            updates_rx,
            cancel_rx,
            hashed_state_tx,
            proof_worker_handle,
            proof_result_tx,
            proof_result_rx,
            SparseTrieTaskMetrics::default(),
            trie,
            EMPTY_ROOT_HASH,
            TrieNodeEpoch::new(1),
            5,
        );
        drop(updates_tx);
        task.selective_storage_retries = true;
        task.finished_state_updates = true;
        task.dispatch_diagnostics = Some(ProofDispatchDiagnostics::empty(tracing::Span::none()));

        let address = B256::repeat_byte(0xa1);
        task.storage_updates.insert(
            address,
            B256Map::from_iter([
                (B256::repeat_byte(0x01), LeafUpdate::Touched),
                (B256::repeat_byte(0x02), LeafUpdate::Touched),
            ]),
        );
        task.storage_retry_ready.insert(address);

        task.promote_pending_account_updates().expect("terminal retry succeeds");

        assert!(task.storage_updates[&address].is_empty());
        assert!(task.storage_retry_ready.is_empty());
        let totals = task.dispatch_diagnostics.take().unwrap().selective_storage_retries;
        assert_eq!(totals.retry_calls, 2);
        assert_eq!(totals.maps_attempted, 2);
        assert_eq!(totals.entries_attempted, 3);
        assert_eq!(totals.entries_applied, 2);
        assert_eq!(totals.productive_requeues, 1);
        assert_eq!(totals.fallback_calls, 0);
        drop(task);
        drain_sparse_trie_tasks(&runtime);
    }

    #[test]
    fn db_backed_selective_retries_match_control_root_updates_and_persisted_tries() {
        let shared_address = keccak256(b"shared-storage-account");
        let mut update = HashedPostState::default();
        update.accounts.insert(
            shared_address,
            Some(Account { nonce: 1, balance: U256::from(1_000), bytecode_hash: None }),
        );
        let mut shared_storage = reth_trie::HashedStorage::default();
        for index in 0..320u64 {
            shared_storage
                .storage
                .insert(keccak256(U256::from(index).to_be_bytes::<32>()), U256::from(index + 1));
        }
        update.storages.insert(shared_address, shared_storage);

        for index in 0..12u64 {
            let address = keccak256(U256::from(index + 10_000).to_be_bytes::<32>());
            update.accounts.insert(
                address,
                Some(Account {
                    nonce: index + 2,
                    balance: U256::from(index + 10),
                    bytecode_hash: None,
                }),
            );
            let mut storage = reth_trie::HashedStorage::default();
            storage.storage.insert(
                keccak256(U256::from(index + 20_000).to_be_bytes::<32>()),
                U256::from(index + 1),
            );
            update.storages.insert(address, storage);
        }
        assert!(
            update.accounts.len() +
                update.storages.values().map(|s| s.storage.len()).sum::<usize>() >
                300
        );

        let run = |selective_storage_retries| {
            let runtime = reth_tasks::Runtime::test();
            let provider_factory = create_test_provider_factory();
            let anchor_hash = init_genesis(&provider_factory).expect("initialize genesis");
            let state_provider_factory = OverlayStateProviderFactory::new(
                provider_factory.clone(),
                OverlayManager::<reth_chain_state::EthPrimitives>::default()
                    .overlay_builder(anchor_hash),
            );
            let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
            let proof_worker_handle = ProofWorkerHandle::new(
                &runtime,
                ProofTaskCtx::new(state_provider_factory),
                false,
                proof_result_tx.clone(),
            );
            let default_trie = RevealableSparseTrie::blind_from(ArenaParallelSparseTrie::default());
            let trie = SparseStateTrie::default()
                .with_accounts_trie(default_trie.clone())
                .with_default_storage_trie(default_trie)
                .with_updates(true);
            let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
            let (_cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
            let (hashed_state_tx, _hashed_state_rx) = std::sync::mpsc::channel();
            let mut task = SparseTrieCacheTask::new_with_trie(
                &runtime,
                updates_rx,
                cancel_rx,
                hashed_state_tx,
                proof_worker_handle,
                proof_result_tx,
                proof_result_rx,
                SparseTrieTaskMetrics::default(),
                trie,
                EMPTY_ROOT_HASH,
                TrieNodeEpoch::new(1),
                5,
            );
            task.selective_storage_retries = selective_storage_retries;
            task.dispatch_diagnostics =
                Some(ProofDispatchDiagnostics::empty(tracing::Span::none()));
            task.dispatch_diagnostics.as_mut().unwrap().selective_storage_retries.enabled =
                u64::from(selective_storage_retries);

            updates_tx.send(StateRootMessage::HashedStateUpdate(update.clone())).unwrap();
            updates_tx.send(StateRootMessage::FinishedStateUpdates).unwrap();
            drop(updates_tx);
            let outcome = task.run().expect("sparse trie task");
            let retry_totals =
                task.dispatch_diagnostics.as_ref().unwrap().selective_storage_retries;
            let (trie, deferred) = task.into_trie_for_reuse();
            drop(deferred);
            drain_sparse_trie_tasks(&runtime);
            (provider_factory, anchor_hash, outcome, trie, retry_totals)
        };

        let (control_factory, control_anchor, control, control_trie, control_parent_retries) =
            run(false);
        let (
            selective_factory,
            selective_anchor,
            selective,
            selective_trie,
            selective_parent_retries,
        ) = run(true);
        assert_eq!(control.state_root, selective.state_root);
        assert_eq!(control.trie_updates, selective.trie_updates);

        let deleted_address = keccak256(U256::from(10_000).to_be_bytes::<32>());
        let mut child_update = HashedPostState::default();
        child_update.accounts.insert(
            shared_address,
            Some(Account { nonce: 2, balance: U256::from(2_000), bytecode_hash: None }),
        );
        child_update.accounts.insert(deleted_address, None);
        let mut child_storage = reth_trie::HashedStorage::default();
        child_storage.storage.insert(keccak256(U256::ZERO.to_be_bytes::<32>()), U256::from(9_999));
        child_storage.storage.insert(keccak256(U256::from(1).to_be_bytes::<32>()), U256::ZERO);
        child_storage
            .storage
            .insert(keccak256(U256::from(321).to_be_bytes::<32>()), U256::from(12_345));
        child_update.storages.insert(shared_address, child_storage);
        let second_storage_address = keccak256(U256::from(10_001).to_be_bytes::<32>());
        let mut second_storage = reth_trie::HashedStorage::default();
        second_storage
            .storage
            .insert(keccak256(U256::from(20_001).to_be_bytes::<32>()), U256::ZERO);
        child_update.storages.insert(second_storage_address, second_storage);

        let run_child =
            |provider_factory: reth_provider::ProviderFactory<
                reth_provider::test_utils::MockNodeTypesWithDB,
            >,
             anchor_hash,
             parent_root,
             trie: SparseStateTrie<ArenaParallelSparseTrie, ArenaParallelSparseTrie>,
             selective_storage_retries| {
                let runtime = reth_tasks::Runtime::test();
                let state_provider_factory = OverlayStateProviderFactory::new(
                    provider_factory.clone(),
                    OverlayManager::<reth_chain_state::EthPrimitives>::default()
                        .overlay_builder(anchor_hash),
                );
                let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
                let proof_worker_handle = ProofWorkerHandle::new(
                    &runtime,
                    ProofTaskCtx::new(state_provider_factory),
                    false,
                    proof_result_tx.clone(),
                );
                let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
                let (_cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
                let (hashed_state_tx, _hashed_state_rx) = std::sync::mpsc::channel();
                let mut task = SparseTrieCacheTask::new_with_trie(
                    &runtime,
                    updates_rx,
                    cancel_rx,
                    hashed_state_tx,
                    proof_worker_handle,
                    proof_result_tx,
                    proof_result_rx,
                    SparseTrieTaskMetrics::default(),
                    trie,
                    parent_root,
                    TrieNodeEpoch::new(2),
                    5,
                );
                task.selective_storage_retries = selective_storage_retries;
                task.dispatch_diagnostics =
                    Some(ProofDispatchDiagnostics::empty(tracing::Span::none()));
                task.dispatch_diagnostics.as_mut().unwrap().selective_storage_retries.enabled =
                    u64::from(selective_storage_retries);
                updates_tx.send(StateRootMessage::HashedStateUpdate(child_update.clone())).unwrap();
                updates_tx.send(StateRootMessage::FinishedStateUpdates).unwrap();
                drop(updates_tx);
                let outcome = task.run().expect("child sparse trie task");
                let retry_totals =
                    task.dispatch_diagnostics.as_ref().unwrap().selective_storage_retries;
                drop(task);
                drain_sparse_trie_tasks(&runtime);
                (provider_factory, outcome, retry_totals)
            };

        let (control_factory, control_child, control_child_retries) =
            run_child(control_factory, control_anchor, control.state_root, control_trie, false);
        let (selective_factory, selective_child, selective_child_retries) = run_child(
            selective_factory,
            selective_anchor,
            selective.state_root,
            selective_trie,
            true,
        );
        assert_eq!(control_child.state_root, selective_child.state_root);
        assert_eq!(control_child.trie_updates, selective_child.trie_updates);
        assert_eq!(control_parent_retries.fallback_calls, 0);
        assert_eq!(selective_parent_retries.fallback_calls, 0);
        assert_eq!(control_child_retries.fallback_calls, 0);
        assert_eq!(selective_child_retries.fallback_calls, 0);
        assert!(selective_parent_retries.maps_skipped > 0);
        assert!(
            selective_parent_retries.entries_attempted < control_parent_retries.entries_attempted
        );

        let persist_and_snapshot = |factory: reth_provider::ProviderFactory<
            reth_provider::test_utils::MockNodeTypesWithDB,
        >,
                                    updates: [&TrieUpdates; 2]| {
            let provider = factory.provider_rw().expect("write provider");
            for update in updates {
                provider.write_trie_updates(update.clone()).expect("persist trie updates");
            }
            provider.commit().expect("commit trie updates");

            let provider = factory.provider_rw().expect("read provider");
            let mut account_cursor = provider
                .tx_ref()
                .cursor_read::<tables::PackedAccountsTrie>()
                .expect("account cursor");
            let accounts = account_cursor
                .walk(None)
                .expect("walk account trie")
                .map(|entry| entry.expect("account trie entry"))
                .collect::<Vec<_>>();
            let mut storage_cursor = provider
                .tx_ref()
                .cursor_dup_read::<tables::PackedStoragesTrie>()
                .expect("storage cursor");
            let storages = storage_cursor
                .walk(None)
                .expect("walk storage trie")
                .map(|entry| entry.expect("storage trie entry"))
                .collect::<Vec<_>>();
            (accounts, storages)
        };

        let control_tables = persist_and_snapshot(
            control_factory,
            [control.trie_updates.as_ref(), control_child.trie_updates.as_ref()],
        );
        let selective_tables = persist_and_snapshot(
            selective_factory,
            [selective.trie_updates.as_ref(), selective_child.trie_updates.as_ref()],
        );
        assert_eq!(control_tables, selective_tables);
        assert!(!control_tables.0.is_empty());
        assert!(!control_tables.1.is_empty());
    }

    #[test]
    fn proof_dispatch_diagnostics_aggregate_queue_depths() {
        let mut diagnostics = ProofDispatchDiagnostics::empty(tracing::Span::none());

        diagnostics.record_dispatch(ProofDispatchReason::Unsplit, 3, 1, 0, 8, 1, 8, 1);
        diagnostics.record_dispatch(ProofDispatchReason::Force, 301, 61, 33, 9, 92, 10, 62);
        diagnostics.record_dispatch(ProofDispatchReason::AccountIdle, 10, 2, 2, 0, 92, 10, 3);
        diagnostics.record_dispatch(ProofDispatchReason::StorageIdle, 10, 2, 0, 4, 92, 10, 3);

        assert_eq!(diagnostics.dispatches, 4);
        assert_eq!(diagnostics.targets, 324);
        assert_eq!(diagnostics.chunks, 66);
        assert_eq!(diagnostics.reason_counts, [1, 1, 1, 1]);
        assert_eq!(diagnostics.account_queue_depth_bins, [2, 1, 0, 1]);
        assert_eq!(diagnostics.storage_queue_depth_bins, [1, 2, 1, 0]);
        assert_eq!(diagnostics.account_queue_high_water, 92);
        assert_eq!(diagnostics.storage_queue_high_water, 10);
        assert_eq!(diagnostics.split_when_queue_nonempty, 2);
        assert_eq!(diagnostics.split_when_storage_queue_nonempty, 2);
        assert_eq!(diagnostics.split_reason_account_queue_nonempty, [1, 1, 0]);
        assert_eq!(diagnostics.split_reason_storage_queue_nonempty, [1, 0, 1]);
        assert_eq!(diagnostics.outstanding_max, 62);
    }

    #[test]
    fn proof_root_tail_diagnostics_track_post_finish_phases() {
        let mut diagnostics = ProofDispatchDiagnostics::empty(tracing::Span::none());
        assert!(diagnostics.progress_timer(true).is_none());
        diagnostics.start_root_tail();
        assert!(diagnostics.progress_timer(false).is_none());
        diagnostics.record_proof_wait(Duration::from_nanos(11));
        diagnostics.record_proof_wait(Duration::from_nanos(17));
        diagnostics.record_result_drain(Duration::from_nanos(7), 3, 2, 5, Instant::now());
        diagnostics.record_result_drain(Duration::from_nanos(9), 4, 0, 1, Instant::now());
        diagnostics.record_reveal(Duration::from_nanos(5));
        diagnostics.record_root_progress(Duration::from_nanos(13));
        diagnostics.record_final_root(Duration::from_nanos(19));

        assert_eq!(diagnostics.proof_wait_ns, 28);
        assert_eq!(diagnostics.proof_wait_count, 2);
        assert_eq!(diagnostics.proof_wait_max_ns, 17);
        assert_eq!(diagnostics.proof_wait_last_ns, 17);
        assert_eq!(diagnostics.result_drain_ns, 16);
        assert_eq!(diagnostics.result_drain_count, 2);
        assert_eq!(diagnostics.result_messages_consumed, 7);
        assert_eq!(diagnostics.result_last_drain_count, 4);
        assert_eq!(diagnostics.result_queue_after_last_drain, 0);
        assert_eq!(diagnostics.in_flight_after_last_drain, 1);
        assert_eq!(diagnostics.reveal_ns, 5);
        assert_eq!(diagnostics.progress_ns, 13);
        assert_eq!(diagnostics.final_root_ns, 19);
        assert!(diagnostics.last_result_consumed_at.is_some());

        let totals = &mut diagnostics.progress_totals[RootProgressPhase::LeafRetry.index()];
        totals.record(
            RootProgressMeasurement {
                wall_ns: 23,
                resources: Some(ThreadResourceUsageDelta {
                    user_cpu_time: Duration::from_nanos(5),
                    system_cpu_time: Duration::from_nanos(7),
                    minor_page_faults: 2,
                    major_page_faults: 3,
                    voluntary_context_switches: 4,
                    involuntary_context_switches: 5,
                    ..Default::default()
                }),
            },
            false,
            11,
            7,
        );
        totals.record(RootProgressMeasurement { wall_ns: 29, resources: None }, true, 13, 3);
        assert_eq!(totals.wall_ns, 52);
        assert_eq!(totals.cpu_measured_wall_ns, 23);
        assert_eq!(totals.caller_cpu_ns, 12);
        assert_eq!(totals.cpu_measured_calls, 1);
        assert_eq!(totals.cpu_missing_calls, 1);
        assert_eq!(totals.calls, 2);
        assert_eq!(totals.failures, 1);
        assert_eq!(totals.work_items, 24);
        assert_eq!(totals.work_outputs, 10);
        assert_eq!(totals.work_items_max, 13);
        assert_eq!(totals.minor_faults, 2);
        assert_eq!(totals.major_faults, 3);
        assert_eq!(totals.voluntary_context_switches, 4);
        assert_eq!(totals.involuntary_context_switches, 5);

        // The explicit-parent test below exercises emission. Avoid emitting this shared callsite
        // outside a subscriber because Rust tests run concurrently and tracing caches callsite
        // interest across dispatchers.
        diagnostics.root_tail_emitted = true;
        diagnostics.progress_emitted = true;
    }

    #[test]
    fn proof_dispatch_summary_has_explicit_root_parent() {
        use std::{collections::BTreeMap, sync::Mutex};
        use tracing::{
            field::{Field, Visit},
            span::{Attributes, Id, Record},
            Event, Metadata, Subscriber,
        };

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<(u64, BTreeMap<String, u64>)>>>);

        struct FieldVisitor<'a>(&'a mut BTreeMap<String, u64>);

        impl Visit for FieldVisitor<'_> {
            fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}

            fn record_u64(&mut self, field: &Field, value: u64) {
                self.0.insert(field.name().to_string(), value);
            }
        }

        impl Subscriber for Capture {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &Attributes<'_>) -> Id {
                Id::from_u64(7)
            }
            fn record(&self, _: &Id, _: &Record<'_>) {}
            fn record_follows_from(&self, _: &Id, _: &Id) {}
            fn event(&self, event: &Event<'_>) {
                if event.metadata().target() == "lifecycle" {
                    let parent = event.parent().expect("summary must have an explicit parent");
                    let mut fields = BTreeMap::new();
                    event.record(&mut FieldVisitor(&mut fields));
                    self.0.lock().unwrap().push((parent.into_u64(), fields));
                }
            }
            fn enter(&self, _: &Id) {}
            fn exit(&self, _: &Id) {}
        }

        let capture = Capture::default();
        tracing::subscriber::with_default(capture.clone(), || {
            let root = tracing::info_span!("root_task");
            let diagnostics = ProofDispatchDiagnostics::empty(root);
            drop(diagnostics);
        });
        let captured = capture.0.lock().unwrap();
        let (parent, fields) = captured
            .iter()
            .find(|(_, fields)| fields.contains_key("root_result_ready"))
            .expect("root-tail summary event");
        assert_eq!(*parent, 7);
        assert_eq!(fields["root_result_ready"], 0);
        assert_eq!(fields["root_success"], 0);
    }

    #[test]
    fn updates_finished_snapshot_has_explicit_parent_and_numeric_state() {
        use std::{collections::BTreeMap, sync::Mutex};
        use tracing::{
            field::{Field, Visit},
            span::{Attributes, Id, Record},
            Event, Metadata, Subscriber,
        };

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Option<(u64, BTreeMap<String, u64>)>>>);

        struct FieldVisitor<'a>(&'a mut BTreeMap<String, u64>);

        impl Visit for FieldVisitor<'_> {
            fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}

            fn record_u64(&mut self, field: &Field, value: u64) {
                self.0.insert(field.name().to_string(), value);
            }
        }

        impl Subscriber for Capture {
            fn enabled(&self, _: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &Attributes<'_>) -> Id {
                Id::from_u64(11)
            }
            fn record(&self, _: &Id, _: &Record<'_>) {}
            fn record_follows_from(&self, _: &Id, _: &Id) {}
            fn event(&self, event: &Event<'_>) {
                if event.metadata().target() != "lifecycle" {
                    return
                }
                let mut fields = BTreeMap::new();
                event.record(&mut FieldVisitor(&mut fields));
                if fields.contains_key("in_flight") {
                    let parent = event.parent().expect("snapshot must have an explicit parent");
                    *self.0.lock().unwrap() = Some((parent.into_u64(), fields));
                }
            }
            fn enter(&self, _: &Id) {}
            fn exit(&self, _: &Id) {}
        }

        let capture = Capture::default();
        tracing::subscriber::with_default(capture.clone(), || {
            let diagnostics = ProofDispatchDiagnostics::empty(tracing::info_span!("root_task"));
            diagnostics.emit_updates_finished_snapshot(13, 3, 5, 7, 11, 2);
        });

        let captured = capture.0.lock().unwrap();
        let (parent, fields) = captured.as_ref().expect("snapshot event");
        assert_eq!(*parent, 11);
        assert_eq!(fields["in_flight"], 13);
        assert_eq!(fields["pending_targets"], 8);
        assert_eq!(fields["pending_account_targets"], 3);
        assert_eq!(fields["pending_storage_targets"], 5);
        assert_eq!(fields["account_queue_depth"], 7);
        assert_eq!(fields["storage_queue_depth"], 11);
        assert_eq!(fields["result_queue_depth"], 2);
    }

    #[test]
    fn test_run_hashing_task_hashed_state_update_forwards() {
        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (hashed_state_tx, hashed_state_rx) = crossbeam_channel::unbounded();

        let address = keccak256(Address::random());
        let slot = keccak256(U256::from(42).to_be_bytes::<32>());
        let value = U256::from(999);

        let mut hashed_state = HashedPostState::default();
        hashed_state.accounts.insert(
            address,
            Some(Account { balance: U256::from(100), nonce: 1, bytecode_hash: None }),
        );
        let mut storage = reth_trie::HashedStorage::default();
        storage.storage.insert(slot, value);
        hashed_state.storages.insert(address, storage);

        let expected_state = hashed_state.clone();

        let handle = std::thread::spawn(move || {
            SparseTrieCacheTask::<ArenaParallelSparseTrie, ArenaParallelSparseTrie>::run_hashing_task(
                updates_rx,
                hashed_state_tx,
                SparseTrieTaskMetrics::default(),
            );
        });

        updates_tx.send(StateRootMessage::HashedStateUpdate(hashed_state)).unwrap();
        updates_tx.send(StateRootMessage::FinishedStateUpdates).unwrap();
        drop(updates_tx);

        let SparseTrieTaskMessage::HashedState(received) = hashed_state_rx.recv().unwrap() else {
            panic!("expected HashedState message");
        };

        let account = received.accounts.get(&address).unwrap().unwrap();
        assert_eq!(account.balance, expected_state.accounts[&address].unwrap().balance);
        assert_eq!(account.nonce, expected_state.accounts[&address].unwrap().nonce);

        let storage = received.storages.get(&address).unwrap();
        assert_eq!(*storage.storage.get(&slot).unwrap(), value);

        let second = hashed_state_rx.recv().unwrap();
        assert!(matches!(second, SparseTrieTaskMessage::FinishedStateUpdates));

        assert!(hashed_state_rx.recv().is_err());
        handle.join().unwrap();
    }

    #[test]
    fn test_encode_account_leaf_value_deletion_and_empty_root_is_empty() {
        let mut account_rlp_buf = vec![0xAB];
        let encoded = encode_account_leaf_value(None, EMPTY_ROOT_HASH, &mut account_rlp_buf);

        assert!(encoded.is_empty());
        // Early return should not touch the caller's buffer.
        assert_eq!(account_rlp_buf, vec![0xAB]);
    }

    #[test]
    fn test_encode_account_leaf_value_empty_account_and_empty_root_is_empty() {
        let mut account_rlp_buf = vec![0xAB];
        let encoded = encode_account_leaf_value(
            Some(Account::default()),
            EMPTY_ROOT_HASH,
            &mut account_rlp_buf,
        );

        assert!(encoded.is_empty());
        // Early return should not touch the caller's buffer.
        assert_eq!(account_rlp_buf, vec![0xAB]);
    }

    #[test]
    fn test_encode_account_leaf_value_non_empty_account_is_rlp() {
        let storage_root = B256::from([0x99; 32]);
        let account = Some(Account {
            nonce: 7,
            balance: U256::from(42),
            bytecode_hash: Some(B256::from([0xAA; 32])),
        });
        let mut account_rlp_buf = vec![0x00, 0x01];

        let encoded = encode_account_leaf_value(account, storage_root, &mut account_rlp_buf);
        let decoded = TrieAccount::decode(&mut &encoded[..]).expect("valid account RLP");

        assert_eq!(decoded.nonce, 7);
        assert_eq!(decoded.balance, U256::from(42));
        assert_eq!(decoded.storage_root, storage_root);
        assert_eq!(account_rlp_buf, encoded);
    }

    #[test]
    fn first_leaf_batch_starts_proofs_before_input_queue_drains() {
        let runtime = reth_tasks::Runtime::test();
        let provider_factory = create_test_provider_factory();
        let anchor_hash = init_genesis(&provider_factory).expect("failed to initialize genesis");
        let state_provider_factory = OverlayStateProviderFactory::new(
            provider_factory,
            OverlayManager::<reth_chain_state::EthPrimitives>::default()
                .overlay_builder(anchor_hash),
        );
        let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
        let proof_worker_handle = ProofWorkerHandle::new(
            &runtime,
            ProofTaskCtx::new(state_provider_factory),
            false,
            proof_result_tx.clone(),
        );

        let default_trie = RevealableSparseTrie::blind_from(ArenaParallelSparseTrie::default());
        let trie = SparseStateTrie::default()
            .with_accounts_trie(default_trie.clone())
            .with_default_storage_trie(default_trie)
            .with_updates(true);

        let parent_state_root = B256::from([0x55; 32]);
        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (_cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
        let mut task = SparseTrieCacheTask::new_with_trie(
            &runtime,
            updates_rx,
            cancel_rx,
            std::sync::mpsc::channel().0,
            proof_worker_handle,
            proof_result_tx,
            proof_result_rx,
            SparseTrieTaskMetrics::default(),
            trie,
            parent_state_root,
            TrieNodeEpoch::UNMODIFIED,
            1,
        );
        task.dispatch_diagnostics = Some(ProofDispatchDiagnostics::empty(tracing::Span::none()));

        // Keep an input queued so progress cannot use its normal queue-empty flush.
        updates_tx.send(StateRootMessage::PrefetchProofs(Default::default())).unwrap();
        let deadline = std::time::Instant::now();
        while task.updates.is_empty() {
            assert!(deadline.elapsed() < std::time::Duration::from_secs(1));
            std::thread::yield_now();
        }
        for index in 0..INITIAL_UPDATE_BATCH_SIZE {
            let mut state = HashedPostState::default();
            state.accounts.insert(
                B256::repeat_byte(index as u8),
                Some(Account { nonce: 1, ..Default::default() }),
            );
            task.on_hashed_state_update(state);
            task.pending_updates += 1;
            assert!(!task.make_progress().unwrap());
            if index + 1 < INITIAL_UPDATE_BATCH_SIZE {
                assert_eq!(task.in_flight_proof_batches, 0);
            }
        }
        assert!(task.in_flight_proof_batches > 0, "proof work must start before the queue drains");
        assert_eq!(task.pending_updates, 0);

        // A second batch remains buffered; the early flush must not become a permanent small
        // batch policy that repeatedly scans and sorts pending leaves.
        for index in INITIAL_UPDATE_BATCH_SIZE..INITIAL_UPDATE_BATCH_SIZE * 2 {
            let mut state = HashedPostState::default();
            state.accounts.insert(
                B256::repeat_byte(index as u8),
                Some(Account { nonce: 1, ..Default::default() }),
            );
            task.on_hashed_state_update(state);
            task.pending_updates += 1;
            assert!(!task.make_progress().unwrap());
        }
        assert_eq!(task.pending_updates, INITIAL_UPDATE_BATCH_SIZE);
        assert_eq!(task.new_account_updates.len(), INITIAL_UPDATE_BATCH_SIZE);
        assert!(
            task.dispatch_diagnostics
                .as_ref()
                .unwrap()
                .progress_totals
                .iter()
                .all(|totals| totals.calls == 0),
            "streaming-phase work must not enter root-tail phase totals"
        );
        drop(updates_tx);
        drop(task);
        drain_sparse_trie_tasks(&runtime);
    }

    #[test]
    fn run_returns_parent_root_without_revealing_blind_trie_when_no_state_updates() {
        let runtime = reth_tasks::Runtime::test();
        let provider_factory = create_test_provider_factory();
        let anchor_hash = init_genesis(&provider_factory).expect("failed to initialize genesis");
        let state_provider_factory = OverlayStateProviderFactory::new(
            provider_factory,
            OverlayManager::<reth_chain_state::EthPrimitives>::default()
                .overlay_builder(anchor_hash),
        );
        let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
        let proof_worker_handle = ProofWorkerHandle::new(
            &runtime,
            ProofTaskCtx::new(state_provider_factory),
            false,
            proof_result_tx.clone(),
        );

        let default_trie = RevealableSparseTrie::blind_from(ArenaParallelSparseTrie::default());
        let trie = SparseStateTrie::default()
            .with_accounts_trie(default_trie.clone())
            .with_default_storage_trie(default_trie)
            .with_updates(true);

        let parent_state_root = B256::from([0x55; 32]);
        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (_cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
        let mut task = SparseTrieCacheTask::new_with_trie(
            &runtime,
            updates_rx,
            cancel_rx,
            std::sync::mpsc::channel().0,
            proof_worker_handle,
            proof_result_tx,
            proof_result_rx,
            SparseTrieTaskMetrics::default(),
            trie,
            parent_state_root,
            TrieNodeEpoch::UNMODIFIED,
            1,
        );

        updates_tx.send(StateRootMessage::FinishedStateUpdates).unwrap();
        drop(updates_tx);

        let outcome = task.run().expect("state root computation should succeed");

        assert_eq!(outcome.state_root, parent_state_root);
        assert!(outcome.trie_updates.is_empty());
        assert!(task.trie.state_trie_ref().is_none(), "blind trie should not be revealed");

        drop(task);
        drain_sparse_trie_tasks(&runtime);
    }

    #[test]
    fn root_tail_diagnostics_measure_delayed_final_proof_in_run_loop() {
        let runtime = reth_tasks::Runtime::test();
        let provider_factory = create_test_provider_factory();
        let anchor_hash = init_genesis(&provider_factory).expect("failed to initialize genesis");
        let state_provider_factory = OverlayStateProviderFactory::new(
            provider_factory,
            OverlayManager::<reth_chain_state::EthPrimitives>::default()
                .overlay_builder(anchor_hash),
        );

        // Route worker results through a test relay so the run loop has a controlled final
        // dependency after it consumes FinishedStateUpdates.
        let (worker_result_tx, worker_result_rx) = crossbeam_channel::unbounded();
        let (task_result_tx, task_result_rx) = crossbeam_channel::unbounded();
        let proof_worker_handle = ProofWorkerHandle::new(
            &runtime,
            ProofTaskCtx::new(state_provider_factory),
            false,
            worker_result_tx.clone(),
        );

        let default_trie = RevealableSparseTrie::blind_from(ArenaParallelSparseTrie::default());
        let trie = SparseStateTrie::default()
            .with_accounts_trie(default_trie.clone())
            .with_default_storage_trie(default_trie)
            .with_updates(true);
        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (_cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
        let mut task = SparseTrieCacheTask::new_with_trie(
            &runtime,
            updates_rx,
            cancel_rx,
            std::sync::mpsc::channel().0,
            proof_worker_handle,
            worker_result_tx,
            task_result_rx,
            SparseTrieTaskMetrics::default(),
            trie,
            B256::from([0x55; 32]),
            TrieNodeEpoch::UNMODIFIED,
            1,
        );
        task.dispatch_diagnostics = Some(ProofDispatchDiagnostics::empty(tracing::Span::none()));

        let relay = std::thread::spawn(move || {
            let result =
                worker_result_rx.recv_timeout(Duration::from_secs(2)).expect("proof worker result");
            std::thread::sleep(Duration::from_millis(10));
            task_result_tx.send(result).expect("sparse trie result receiver");
        });

        let mut hashed_state = HashedPostState::default();
        hashed_state
            .accounts
            .insert(keccak256(Address::random()), Some(Account { nonce: 1, ..Default::default() }));
        updates_tx.send(StateRootMessage::HashedStateUpdate(hashed_state)).unwrap();
        updates_tx.send(StateRootMessage::FinishedStateUpdates).unwrap();
        drop(updates_tx);

        task.run().expect("state root computation should succeed");
        relay.join().expect("proof result relay");

        let diagnostics = task.dispatch_diagnostics.as_ref().expect("diagnostics enabled");
        let root_tail_ns =
            duration_ns(diagnostics.root_tail_start.expect("finish marker consumed").elapsed());
        let phase_accounted_ns = diagnostics
            .proof_wait_ns
            .saturating_add(diagnostics.result_drain_ns)
            .saturating_add(diagnostics.reveal_ns)
            .saturating_add(diagnostics.progress_ns)
            .saturating_add(diagnostics.final_root_ns);
        assert_eq!(diagnostics.proof_wait_count, 1);
        assert!(diagnostics.proof_wait_ns >= duration_ns(Duration::from_millis(1)));
        assert_eq!(diagnostics.result_drain_count, 1);
        assert_eq!(diagnostics.result_messages_consumed, 1);
        assert_eq!(diagnostics.result_queue_after_last_drain, 0);
        assert_eq!(diagnostics.in_flight_after_last_drain, 0);
        assert!(diagnostics.last_result_consumed_at.is_some());
        let progress_phase_wall_ns = diagnostics
            .progress_totals
            .iter()
            .fold(0u64, |sum, totals| sum.saturating_add(totals.wall_ns));
        assert!(
            diagnostics.progress_totals[RootProgressPhase::NewUpdates.index()].calls > 0,
            "the buffered update must be processed after the finish marker"
        );
        assert!(
            diagnostics.progress_totals[RootProgressPhase::Dispatch.index()].calls > 0,
            "the final account proof must be dispatched during root-tail progress"
        );
        assert!(progress_phase_wall_ns <= diagnostics.progress_ns);
        assert!(phase_accounted_ns <= root_tail_ns);

        drop(task);
        drain_sparse_trie_tasks(&runtime);
    }

    #[test]
    fn stall_check_waits_for_in_flight_proofs_then_reports_pending_updates() {
        let runtime = reth_tasks::Runtime::test();
        let provider_factory = create_test_provider_factory();
        let anchor_hash = init_genesis(&provider_factory).expect("failed to initialize genesis");
        let state_provider_factory = OverlayStateProviderFactory::new(
            provider_factory,
            OverlayManager::<reth_chain_state::EthPrimitives>::default()
                .overlay_builder(anchor_hash),
        );
        let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
        let proof_worker_handle = ProofWorkerHandle::new(
            &runtime,
            ProofTaskCtx::new(state_provider_factory),
            false,
            proof_result_tx.clone(),
        );

        let default_trie = RevealableSparseTrie::blind_from(ArenaParallelSparseTrie::default());
        let trie = SparseStateTrie::default()
            .with_accounts_trie(default_trie.clone())
            .with_default_storage_trie(default_trie)
            .with_updates(true);

        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (_cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
        let mut task = SparseTrieCacheTask::new_with_trie(
            &runtime,
            updates_rx,
            cancel_rx,
            std::sync::mpsc::channel().0,
            proof_worker_handle,
            proof_result_tx,
            proof_result_rx,
            SparseTrieTaskMetrics::default(),
            trie,
            B256::from([0x55; 32]),
            TrieNodeEpoch::UNMODIFIED,
            1,
        );

        drop(updates_tx);

        let account = B256::from([0x11; 32]);
        let slot = B256::from([0x22; 32]);
        let account_target = B256::from([0x33; 32]);
        let storage_target = B256::from([0x44; 32]);

        task.finished_state_updates = true;
        task.account_updates.insert(account, LeafUpdate::Touched);
        task.storage_updates.entry(account).or_default().insert(slot, LeafUpdate::Touched);
        task.pending_account_updates.insert(account, None);
        task.fetched_account_targets.insert(account_target, ProofV2TargetParent::NONE);
        task.fetched_storage_targets
            .entry(account)
            .or_default()
            .insert(storage_target, ProofV2TargetParent::new(11));
        task.in_flight_proof_batches = 1;

        assert!(task.ensure_not_stalled(false).is_ok());

        let result = ProofResultMessage {
            result: Ok(DecodedMultiProofV2::default()),
            elapsed: std::time::Duration::ZERO,
            state: HashedPostState::default(),
        };
        task.on_proof_result_message(result).expect("proof result should be ok");

        assert_eq!(task.in_flight_proof_batches, 0);
        let error = task.ensure_not_stalled(false).expect_err("task should be stalled");
        assert!(matches!(error, StateRootTaskError::Stalled));
        let error = error.to_string();

        assert!(error.contains("sparse trie task stalled"));
        assert!(!error.contains("account_targets"));
        assert!(!error.contains("storage_targets"));
        assert!(!error.contains(&format!("{account:?}")));
        assert!(!error.contains(&format!("{account_target:?}")));
        assert!(!error.contains(&format!("{storage_target:?}")));
        assert!(!error.contains("pending_account_leaves"));
        assert!(!error.contains("pending_storage_leaves"));
        assert!(!error.contains("pending_account_updates"));
        assert!(!error.contains(&format!("{slot:?}")));

        drop(task);
        drain_sparse_trie_tasks(&runtime);
    }

    #[test]
    fn run_errors_when_cancel_guard_drops_before_updates_finish() {
        let runtime = reth_tasks::Runtime::test();
        let provider_factory = create_test_provider_factory();
        let anchor_hash = init_genesis(&provider_factory).expect("failed to initialize genesis");
        let state_provider_factory = OverlayStateProviderFactory::new(
            provider_factory,
            OverlayManager::<reth_chain_state::EthPrimitives>::default()
                .overlay_builder(anchor_hash),
        );
        let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
        let proof_worker_handle = ProofWorkerHandle::new(
            &runtime,
            ProofTaskCtx::new(state_provider_factory),
            false,
            proof_result_tx.clone(),
        );

        let default_trie = RevealableSparseTrie::blind_from(ArenaParallelSparseTrie::default());
        let trie = SparseStateTrie::default()
            .with_accounts_trie(default_trie.clone())
            .with_default_storage_trie(default_trie)
            .with_updates(true);

        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
        let mut task = SparseTrieCacheTask::new_with_trie(
            &runtime,
            updates_rx,
            cancel_rx,
            std::sync::mpsc::channel().0,
            proof_worker_handle,
            proof_result_tx,
            proof_result_rx,
            SparseTrieTaskMetrics::default(),
            trie,
            B256::from([0x55; 32]),
            TrieNodeEpoch::UNMODIFIED,
            1,
        );

        // The consumer abandons the computation. The updates channel is still open (no finish
        // marker was sent), so without the cancel signal the task would wait forever.
        drop(cancel_guard);

        let error = task.run().expect_err("canceled task must return an error");
        assert!(matches!(error, StateRootTaskError::Canceled));

        drop(updates_tx);
        drop(task);
        drain_sparse_trie_tasks(&runtime);
    }

    #[test]
    fn run_ignores_hints_queued_after_updates_finish() {
        let runtime = reth_tasks::Runtime::test();
        let provider_factory = create_test_provider_factory();
        let anchor_hash = init_genesis(&provider_factory).expect("failed to initialize genesis");
        let state_provider_factory = OverlayStateProviderFactory::new(
            provider_factory,
            OverlayManager::<reth_chain_state::EthPrimitives>::default()
                .overlay_builder(anchor_hash),
        );
        let (proof_result_tx, proof_result_rx) = crossbeam_channel::unbounded();
        let proof_worker_handle = ProofWorkerHandle::new(
            &runtime,
            ProofTaskCtx::new(state_provider_factory),
            false,
            proof_result_tx.clone(),
        );

        let default_trie = RevealableSparseTrie::blind_from(ArenaParallelSparseTrie::default());
        let trie = SparseStateTrie::default()
            .with_accounts_trie(default_trie.clone())
            .with_default_storage_trie(default_trie)
            .with_updates(true);

        let (updates_tx, updates_rx) = crossbeam_channel::unbounded();
        let (cancel_guard, cancel_rx) = crossbeam_channel::bounded::<()>(0);
        let mut task = SparseTrieCacheTask::new_with_trie(
            &runtime,
            updates_rx,
            cancel_rx,
            std::sync::mpsc::channel().0,
            proof_worker_handle,
            proof_result_tx,
            proof_result_rx,
            SparseTrieTaskMetrics::default(),
            trie,
            B256::from([0x55; 32]),
            TrieNodeEpoch::UNMODIFIED,
            1,
        );

        updates_tx.send(StateRootMessage::FinishedStateUpdates).unwrap();
        updates_tx.send(StateRootMessage::PrefetchProofs(Default::default())).unwrap();

        let wait_start = std::time::Instant::now();
        while task.updates.len() < 2 {
            assert!(
                wait_start.elapsed() < std::time::Duration::from_secs(1),
                "hashing task did not queue the test messages"
            );
            std::thread::yield_now();
        }

        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _ = result_tx.send(task.run());
        });

        let result = result_rx.recv_timeout(std::time::Duration::from_secs(1));
        drop(cancel_guard);
        handle.join().unwrap();

        assert!(result.expect("state root task stalled on a late hint").is_ok());

        drop(updates_tx);
        drain_sparse_trie_tasks(&runtime);
    }
}
