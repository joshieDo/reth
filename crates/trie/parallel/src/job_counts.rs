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
    pub(crate) root_requests: u64,
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
            root_requests: 0,
            saturated: false,
        }
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
