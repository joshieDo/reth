//! Capture-only attempted proof-job cardinalities, with no target scans or per-job records.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum JobKind {
    Account,
    Storage,
}

/// Stack-owned by the worker launcher and borrowed for its synchronous run, including errors.
#[derive(Debug)]
pub(crate) struct JobCounts {
    pub(crate) kind: JobKind,
    pub(crate) jobs: u64,
    pub(crate) targets: u64,
    pub(crate) max_targets: u64,
    pub(crate) bins: [u64; 5],
    pub(crate) storage_groups: u64,
    pub(crate) storage_only_single_group: u64,
    pub(crate) root_requests: u64,
    pub(crate) inline_storage_attempts: u64,
    pub(crate) inline_storage_targets: u64,
    pub(crate) saturated: bool,
}

/// Only O(1) vector/map lengths are read. Account storage-group count is not slot count.
pub(crate) struct JobSize {
    pub(crate) targets: usize,
    pub(crate) storage_groups: usize,
    pub(crate) needs_root: bool,
}

impl JobCounts {
    pub(crate) const fn new(kind: JobKind) -> Self {
        Self {
            kind,
            jobs: 0,
            targets: 0,
            max_targets: 0,
            bins: [0; 5],
            storage_groups: 0,
            storage_only_single_group: 0,
            root_requests: 0,
            inline_storage_attempts: 0,
            inline_storage_targets: 0,
            saturated: false,
        }
    }

    /// Account-worker inline calculations are attempts, never storage-pool dequeues.
    /// No target length is inspected when capture is disabled.
    #[allow(dead_code, reason = "matched observer-only control records zero inline attempts")]
    pub(crate) fn observe_inline(counts: Option<&mut Self>, targets: impl FnOnce() -> usize) {
        let Some(counts) = counts else { return };
        debug_assert_eq!(counts.kind, JobKind::Account);
        let targets = u64::try_from(targets()).unwrap_or_else(|_| {
            counts.saturated = true;
            u64::MAX
        });
        add(&mut counts.inline_storage_attempts, 1, &mut counts.saturated);
        add(&mut counts.inline_storage_targets, targets, &mut counts.saturated);
    }

    /// The closure is not called without a capture observer. Counts precede processing;
    /// failed calculations and results whose receiver disappeared remain attempted jobs.
    pub(crate) fn observe(counts: Option<&mut Self>, size: impl FnOnce() -> JobSize) {
        let Some(counts) = counts else { return };
        let size = size();
        let targets = u64::try_from(size.targets).unwrap_or_else(|_| {
            counts.saturated = true;
            u64::MAX
        });
        let bin = match targets {
            0 => 0,
            1 => 1,
            2..=8 => 2,
            9..=32 => 3,
            _ => 4,
        };
        add(&mut counts.jobs, 1, &mut counts.saturated);
        add(&mut counts.targets, targets, &mut counts.saturated);
        add(&mut counts.bins[bin], 1, &mut counts.saturated);
        counts.max_targets = counts.max_targets.max(targets);
        match counts.kind {
            JobKind::Account => {
                let groups = u64::try_from(size.storage_groups).unwrap_or_else(|_| {
                    counts.saturated = true;
                    u64::MAX
                });
                add(&mut counts.storage_groups, groups, &mut counts.saturated);
                if size.targets == 0 && size.storage_groups == 1 {
                    add(&mut counts.storage_only_single_group, 1, &mut counts.saturated);
                }
            }
            JobKind::Storage => {
                add(&mut counts.root_requests, u64::from(size.needs_root), &mut counts.saturated)
            }
        }
    }
}

const fn add(total: &mut u64, value: u64, saturated: &mut bool) {
    let (sum, overflow) = total.overflowing_add(value);
    *saturated |= overflow;
    *total = if overflow { u64::MAX } else { sum };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_counts_are_optional_saturating_attempts_not_dequeues() {
        JobCounts::observe_inline(None, || panic!("disabled observer"));
        let mut counts = JobCounts::new(JobKind::Account);
        for n in [0, 1, 8] {
            JobCounts::observe_inline(Some(&mut counts), || n);
        }
        assert_eq!((counts.inline_storage_attempts, counts.inline_storage_targets), (3, 9));
        assert_eq!((counts.jobs, counts.targets, counts.root_requests), (0, 0, 0));
        counts.inline_storage_targets = u64::MAX;
        JobCounts::observe_inline(Some(&mut counts), || 1);
        assert!(counts.saturated);
        assert_eq!(counts.inline_storage_targets, u64::MAX);
    }

    #[test]
    fn disabled_observer_does_not_read_job_sizes() {
        JobCounts::observe(None, || panic!("capture disabled must not inspect input"));
    }

    #[test]
    fn own_vector_bins_and_groups_remain_distinct() {
        let mut account = JobCounts::new(JobKind::Account);
        let mut storage = JobCounts::new(JobKind::Storage);
        for targets in [0, 1, 2, 8, 9, 32, 33] {
            for counts in [&mut account, &mut storage] {
                JobCounts::observe(Some(counts), || JobSize {
                    targets,
                    storage_groups: 5,
                    needs_root: true,
                });
            }
        }
        for counts in [&account, &storage] {
            assert_eq!(counts.jobs, 7);
            assert_eq!(counts.targets, 85);
            assert_eq!(counts.max_targets, 33);
            assert_eq!(counts.bins, [1, 1, 2, 2, 1]);
            assert!(!counts.saturated);
        }
        assert_eq!(account.storage_groups, 35);
        assert_eq!(account.root_requests, 0);
        assert_eq!(storage.storage_groups, 0);
        assert_eq!(storage.root_requests, 7);
    }

    #[test]
    fn storage_only_single_group_is_joint_account_only_and_saturating() {
        let mut account = JobCounts::new(JobKind::Account);
        let mut storage = JobCounts::new(JobKind::Storage);
        for (targets, groups) in [(0, 0), (0, 1), (0, 2), (1, 1), (0, 1)] {
            for counts in [&mut account, &mut storage] {
                JobCounts::observe(Some(counts), || JobSize {
                    targets,
                    storage_groups: groups,
                    needs_root: false,
                });
            }
        }
        assert_eq!(account.jobs, 5);
        assert_eq!(account.bins[0], 4);
        assert_eq!(account.storage_only_single_group, 2);
        assert_eq!(storage.storage_only_single_group, 0);
        assert!(!account.saturated);
        account.storage_only_single_group = u64::MAX;
        JobCounts::observe(Some(&mut account), || JobSize {
            targets: 0,
            storage_groups: 1,
            needs_root: false,
        });
        assert_eq!(account.storage_only_single_group, u64::MAX);
        assert!(account.saturated);
        JobCounts::observe(Some(&mut account), || JobSize {
            targets: 1,
            storage_groups: 1,
            needs_root: false,
        });
        assert!(account.saturated);
    }

    #[test]
    fn empty_and_overflow_counters_are_explicit() {
        let mut counts = JobCounts::new(JobKind::Storage);
        assert_eq!(counts.jobs, 0);
        assert_eq!(counts.bins, [0; 5]);
        counts.jobs = u64::MAX;
        counts.targets = u64::MAX;
        counts.root_requests = u64::MAX;
        counts.bins[1] = u64::MAX;
        JobCounts::observe(Some(&mut counts), || JobSize {
            targets: 1,
            storage_groups: 0,
            needs_root: true,
        });
        assert!(counts.saturated);
        assert_eq!(counts.jobs, u64::MAX);
        assert_eq!(counts.targets, u64::MAX);
        assert_eq!(counts.bins[1], u64::MAX);
        assert_eq!(counts.root_requests, u64::MAX);
        let mut group = JobCounts::new(JobKind::Account);
        group.storage_groups = u64::MAX;
        JobCounts::observe(Some(&mut group), || JobSize {
            targets: 0,
            storage_groups: 1,
            needs_root: false,
        });
        assert!(group.saturated);
        assert_eq!(group.storage_groups, u64::MAX);
    }
}
