//! Worker-local observations of potentially redundant root-only computation.

use std::cell::Cell;

/// Counts observations before computation; cache presence is a racy snapshot,
/// not proof that a lookup would save time or that any work was skipped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RootWorkCounts {
    pub(crate) storage_partial_roots: u64,
    pub(crate) storage_partial_cached: u64,
    pub(crate) account_sync_roots: u64,
    pub(crate) account_sync_cached: u64,
    pub(crate) account_missing_roots: u64,
    pub(crate) account_missing_cached: u64,
}

#[derive(Clone, Copy)]
pub(crate) enum RootWorkKind {
    StoragePartial,
    AccountSync,
    AccountMissing,
}

/// One allocation per observed worker, shared only with its synchronous deferred
/// encoders. No atomics, timers or per-job events are needed.
#[derive(Default)]
pub(crate) struct RootWorkObserver(Cell<RootWorkCounts>);

impl RootWorkObserver {
    pub(crate) fn observe(&self, kind: RootWorkKind, cached: bool) {
        let mut counts = self.0.get();
        let (attempts, hits) = match kind {
            RootWorkKind::StoragePartial => {
                (&mut counts.storage_partial_roots, &mut counts.storage_partial_cached)
            }
            RootWorkKind::AccountSync => {
                (&mut counts.account_sync_roots, &mut counts.account_sync_cached)
            }
            RootWorkKind::AccountMissing => {
                (&mut counts.account_missing_roots, &mut counts.account_missing_cached)
            }
        };
        *attempts = attempts.saturating_add(1);
        *hits = hits.saturating_add(u64::from(cached));
        self.0.set(counts);
    }

    #[cfg(any(feature = "metrics", test))]
    pub(crate) const fn snapshot(&self) -> RootWorkCounts {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_root_observations_and_saturation() {
        let observer = RootWorkObserver::default();
        observer.observe(RootWorkKind::StoragePartial, false);
        observer.observe(RootWorkKind::StoragePartial, true);
        observer.observe(RootWorkKind::AccountSync, true);
        observer.observe(RootWorkKind::AccountMissing, false);
        assert_eq!(
            observer.snapshot(),
            RootWorkCounts {
                storage_partial_roots: 2,
                storage_partial_cached: 1,
                account_sync_roots: 1,
                account_sync_cached: 1,
                account_missing_roots: 1,
                account_missing_cached: 0,
            }
        );
        observer.0.set(RootWorkCounts {
            storage_partial_roots: u64::MAX,
            storage_partial_cached: u64::MAX,
            ..Default::default()
        });
        observer.observe(RootWorkKind::StoragePartial, true);
        assert_eq!(observer.snapshot().storage_partial_roots, u64::MAX);
        assert_eq!(observer.snapshot().storage_partial_cached, u64::MAX);
        assert_eq!(RootWorkObserver::default().snapshot(), RootWorkCounts::default());
    }
}
