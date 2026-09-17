//! Production-provider fixtures established before changing the routing algorithm.
use super::*;
use reth_chain_state::{test_utils::TestBlockBuilder, ExecutedBlock};
use reth_ethereum_primitives::EthPrimitives;
use reth_primitives_traits::Account;
use reth_provider::{
    test_utils::{create_test_provider_factory, MockNodeTypesWithDB},
    BlockWriter, ProviderFactory,
};
use reth_stages_types::{FinishCheckpoint, StageCheckpoint, StageId};
use reth_storage_api::{StageCheckpointWriter, StateWriter};
use reth_storage_overlay::{OverlayManager, OverlayStateProviderFactory};
use reth_trie::{ComputedTrieData, HashedStorage, ProofV2TargetParent};

type Factory = OverlayStateProviderFactory<ProviderFactory<MockNodeTypesWithDB>>;
type Provider = <Factory as DatabaseProviderROFactory>::Provider;

fn address(i: u8) -> B256 {
    B256::repeat_byte(i)
}
fn slot(i: u8) -> B256 {
    B256::repeat_byte(i)
}
fn state(value: u64) -> HashedPostState {
    HashedPostState::default()
        .with_accounts([
            (address(1), Some(Account::default())),
            (address(2), Some(Account::default())),
        ])
        .with_storages([
            (
                address(1),
                HashedStorage::from_iter([
                    (slot(0x20), U256::from(value)),
                    (slot(0x2f), U256::from(2)),
                    (slot(0x80), U256::from(3)),
                ]),
            ),
            (address(2), HashedStorage::from_iter([(slot(0x30), U256::from(4))])),
        ])
}

/// Both pools use this exact anchored factory. Reuse intentionally suppresses an overlay;
/// it is not an assertion that arbitrary provider snapshots are byte-equivalent.
fn factory(reused: bool) -> Factory {
    let db = create_test_provider_factory();
    let blocks: Vec<_> = TestBlockBuilder::eth().get_executed_blocks(0..3).collect();
    let writer = db.provider_rw().unwrap();
    for block in &blocks[..2] {
        writer.insert_block(block.recovered_block()).unwrap();
    }
    writer.write_hashed_state(&state(1).into_sorted()).unwrap();
    writer
        .save_stage_checkpoint(
            StageId::Finish,
            StageCheckpoint::new(blocks[1].block_number()).with_finish_stage_checkpoint(
                FinishCheckpoint { partial_state_trie: Some(blocks[1].block_number()) },
            ),
        )
        .unwrap();
    writer.commit().unwrap();
    let manager = OverlayManager::<EthPrimitives>::default();
    let overlay = ExecutedBlock::new(
        Arc::clone(&blocks[2].recovered_block),
        Arc::clone(&blocks[2].execution_output),
        ComputedTrieData::new(Arc::new(state(9).into_sorted()), Default::default()),
    );
    manager.insert_block(overlay);
    let builder = manager.overlay_builder(blocks[2].recovered_block().hash());
    let factory = OverlayStateProviderFactory::new(db, builder);
    if reused {
        factory.with_skip_overlay_for_reused_sparse_trie(blocks[1].recovered_block().hash())
    } else {
        factory
    }
}

fn with_worker(
    reused: bool,
    f: impl FnOnce(&AccountProofWorker<Factory>, &Provider),
) -> Arc<DashMap<B256, B256>> {
    let factory = factory(reused);
    let provider = factory.database_provider_ro().unwrap();
    let roots = Arc::new(DashMap::default());
    let (storage_tx, storage_rx) = unbounded();
    let (_, account_rx) = unbounded();
    let account = AccountProofWorker::new(
        ProofTaskCtx::new(factory.clone()),
        account_rx,
        0,
        storage_tx,
        Arc::new(AvailabilitySheet::new(1)),
        roots.clone(),
        #[cfg(feature = "metrics")]
        Default::default(),
        #[cfg(feature = "metrics")]
        Default::default(),
    );
    let storage = StorageProofWorker::new(
        ProofTaskCtx::new(factory),
        storage_rx,
        0,
        Arc::new(AvailabilitySheet::new(1)),
        roots.clone(),
        #[cfg(feature = "metrics")]
        Default::default(),
        #[cfg(feature = "metrics")]
        Default::default(),
    );
    std::thread::scope(|scope| {
        let task = scope.spawn(move || storage.run(None));
        f(&account, &provider);
        drop(account);
        task.join().unwrap().unwrap();
    });
    roots
}

// Use the same cursor instrumentation and Rc ownership as AccountProofWorker::run.
macro_rules! calculators {
    ($provider:expr, $account:ident, $storage:ident, $body:block) => {{
        let provider = $provider;
        let mut metrics = ProofTaskCursorMetricsCache::default();
        let mut $account = proof_v2::ProofCalculator::new(
            InstrumentedTrieCursor::new(
                provider.account_trie_cursor().unwrap(),
                &mut metrics.account_trie_cursor,
            ),
            InstrumentedHashedCursor::new(
                provider.hashed_account_cursor().unwrap(),
                &mut metrics.account_hashed_cursor,
            ),
        );
        let $storage = Rc::new(RefCell::new(proof_v2::StorageProofCalculator::new_storage(
            InstrumentedTrieCursor::new(
                provider.storage_trie_cursor(B256::ZERO).unwrap(),
                &mut metrics.storage_trie_cursor,
            ),
            InstrumentedHashedCursor::new(
                provider.hashed_storage_cursor(B256::ZERO).unwrap(),
                &mut metrics.storage_hashed_cursor,
            ),
        )));
        $body
    }};
}

fn targets(addr: B256, slots: Vec<ProofV2Target>) -> MultiProofTargetsV2 {
    MultiProofTargetsV2 {
        account_targets: vec![],
        storage_targets: [(addr, slots)].into_iter().collect(),
    }
}

#[test]
fn anchored_storage_matches_nested_and_reused_provider() {
    for reused in [false, true] {
        with_worker(reused, |worker, provider| {
            calculators!(provider, account, storage, {
                for (addr, slots) in [
                    (
                        address(1),
                        vec![ProofV2Target::new(slot(0x20)), ProofV2Target::new(slot(0x2f))],
                    ),
                    (address(1), vec![]),
                    (
                        address(1),
                        vec![
                            ProofV2Target::new(slot(0x20)).with_parent(ProofV2TargetParent::new(1))
                        ],
                    ),
                    (address(2), vec![ProofV2Target::new(slot(0x30))]),
                    (address(3), vec![]),
                ] {
                    let nested = worker
                        .compute_v2_account_multiproof::<Provider>(
                            &mut account,
                            storage.clone(),
                            targets(addr, slots.clone()),
                        )
                        .unwrap()
                        .0;
                    let direct = ProofTaskTx::new(provider, 0)
                        .compute_v2_storage_proof(
                            StorageProofInput::new(addr, slots, false),
                            &mut storage.borrow_mut(),
                        )
                        .unwrap();
                    assert!(nested.account_proofs.is_empty());
                    assert_eq!(nested.storage_proofs.len(), 1);
                    assert_eq!(nested.storage_proofs[&addr], direct.proof);
                }
                // Verify the fixture actually exercises the skip boundary, not two empty providers.
                let root = storage.borrow_mut().storage_root_node(address(1)).unwrap();
                let hash = storage.borrow_mut().compute_root_hash(&[root]).unwrap().unwrap();
                assert_eq!(
                    hash,
                    reth_trie::test_utils::storage_root_prehashed([
                        (slot(0x20), U256::from(if reused { 1 } else { 9 })),
                        (slot(0x2f), U256::from(2)),
                        (slot(0x80), U256::from(3)),
                    ])
                );
            });
        });
    }
}

#[test]
fn account_storage_calculator_reuse_preserves_owned_results_and_context() {
    for reused in [false, true] {
        with_worker(reused, |worker, provider| {
            calculators!(provider, account, storage, {
                let mut done = 0;
                let mut results = vec![];
                for with_account in [false, true, false] {
                    let mut input_targets =
                        targets(address(1), vec![ProofV2Target::new(slot(0x20))]);
                    if with_account {
                        input_targets.account_targets.push(ProofV2Target::new(address(1)));
                    }
                    // Independent fresh calculators establish the owned result each time.
                    let expected = calculators!(provider, fresh_account, fresh_storage, {
                        worker
                            .compute_v2_account_multiproof::<Provider>(
                                &mut fresh_account,
                                fresh_storage,
                                MultiProofTargetsV2 {
                                    account_targets: input_targets.account_targets.clone(),
                                    storage_targets: input_targets.storage_targets.clone(),
                                },
                            )
                            .unwrap()
                            .0
                    });
                    let (tx, rx) = unbounded();
                    let original_state = state(77);
                    let start = Instant::now();
                    worker.process_account_multiproof::<Provider>(
                        &mut account,
                        storage.clone(),
                        AccountMultiproofInput {
                            targets: input_targets,
                            proof_result_sender: ProofResultContext::new(
                                tx,
                                original_state.clone(),
                                start,
                            ),
                        },
                        &mut done,
                    );
                    let message = rx.recv().unwrap();
                    assert_eq!(message.state, original_state);
                    assert!(message.elapsed <= start.elapsed());
                    let actual = message.result.unwrap();
                    assert_eq!(actual, expected);
                    results.push((actual, expected));
                    worker.cached_storage_roots.clear();
                }
                assert_eq!(done, 3);
                assert!(results.iter().all(|(actual, expected)| actual == expected));
            });
        });
    }
}

#[test]
fn canceled_account_result_still_populates_root_cache_and_next_job_completes() {
    let roots = with_worker(false, |worker, provider| {
        calculators!(provider, account, storage, {
            let mut done = 0;
            for canceled in [true, false] {
                let (tx, rx) = unbounded();
                let rx = if canceled {
                    drop(rx);
                    None
                } else {
                    Some(rx)
                };
                worker.process_account_multiproof::<Provider>(
                    &mut account,
                    storage.clone(),
                    AccountMultiproofInput {
                        targets: targets(address(1), vec![]),
                        proof_result_sender: ProofResultContext::new(tx, state(66), Instant::now()),
                    },
                    &mut done,
                );
                if let Some(rx) = rx {
                    let output = rx.recv().unwrap();
                    assert_eq!(output.state, state(66));
                    assert_eq!(output.result.unwrap().storage_proofs.len(), 1);
                }
            }
            assert_eq!(done, 2);
        });
    });
    // Joining the nested worker removes its existing send-before-cache insertion race.
    assert_eq!(
        *roots.get(&address(1)).unwrap(),
        reth_trie::test_utils::storage_root_prehashed([
            (slot(0x20), U256::from(9)),
            (slot(0x2f), U256::from(2)),
            (slot(0x80), U256::from(3)),
        ])
    );
}

struct FailOnce<C> {
    cursor: C,
    fail: Rc<std::cell::Cell<bool>>,
}
impl<C> FailOnce<C> {
    fn check(&self) -> Result<(), DatabaseError> {
        if self.fail.replace(false) {
            Err(DatabaseError::Other("injected storage cursor failure".into()))
        } else {
            Ok(())
        }
    }
}
impl<C: reth_trie::hashed_cursor::HashedCursor<Value = U256>> reth_trie::hashed_cursor::HashedCursor
    for FailOnce<C>
{
    type Value = U256;
    fn seek(&mut self, key: B256) -> Result<Option<(B256, U256)>, DatabaseError> {
        self.check()?;
        self.cursor.seek(key)
    }
    fn next(&mut self) -> Result<Option<(B256, U256)>, DatabaseError> {
        self.check()?;
        self.cursor.next()
    }
    fn reset(&mut self) {
        self.cursor.reset();
    }
}
impl<C: HashedStorageCursor<Value = U256>> HashedStorageCursor for FailOnce<C> {
    fn is_storage_empty(&mut self) -> Result<bool, DatabaseError> {
        self.check()?;
        self.cursor.is_storage_empty()
    }
    fn set_hashed_address(&mut self, key: B256) {
        self.cursor.set_hashed_address(key);
    }
}

#[test]
fn storage_error_conversion_and_calculator_reset() {
    let factory = factory(false);
    let provider = factory.database_provider_ro().unwrap();
    let fail = Rc::new(std::cell::Cell::new(false));
    let mut calculator = proof_v2::StorageProofCalculator::new_storage(
        provider.storage_trie_cursor(B256::ZERO).unwrap(),
        FailOnce {
            cursor: provider.hashed_storage_cursor(B256::ZERO).unwrap(),
            fail: fail.clone(),
        },
    );
    let worker = ProofTaskTx::new(&provider, 0);
    for slots in [
        vec![],
        vec![ProofV2Target::new(slot(0x20))],
        vec![ProofV2Target::new(slot(0x20)).with_parent(ProofV2TargetParent::new(1))],
    ] {
        fail.set(true);
        let error = worker
            .compute_v2_storage_proof(
                StorageProofInput::new(address(1), slots.clone(), false),
                &mut calculator,
            )
            .unwrap_err();
        assert!(
            matches!(StateRootTaskError::from(error),StateRootTaskError::Provider(ProviderError::Database(DatabaseError::Other(message))) if message=="injected storage cursor failure")
        );
        let result = worker
            .compute_v2_storage_proof(
                StorageProofInput::new(address(1), slots.clone(), false),
                &mut calculator,
            )
            .unwrap();
        let mut fresh = proof_v2::StorageProofCalculator::new_storage(
            provider.storage_trie_cursor(B256::ZERO).unwrap(),
            provider.hashed_storage_cursor(B256::ZERO).unwrap(),
        );
        let expected = worker
            .compute_v2_storage_proof(StorageProofInput::new(address(1), slots, false), &mut fresh)
            .unwrap();
        assert_eq!(result.proof, expected.proof);
        assert_eq!(result.root, expected.root);
    }
}
