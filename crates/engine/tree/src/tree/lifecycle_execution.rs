//! Gated, thread-bound accounting for the synchronous transaction execution loop.

use reth_metrics::thread::{ThreadResourceUsage, ThreadResourceUsageDelta};
use std::time::{Duration, Instant};

/// At most two CPU samples per completed loop, never per transaction. The guard is `!Send`,
/// so its start and end samples cannot move to another execution thread.
pub(super) struct ExecutionLoopTimer {
    cpu: ThreadResourceUsage,
    wall: Instant,
}

impl ExecutionLoopTimer {
    pub(super) fn start(enabled: bool) -> Option<Self> {
        enabled.then(|| Self { cpu: ThreadResourceUsage::now(), wall: Instant::now() })
    }

    /// Returns matching loop wall time and optional user+system thread CPU time.
    ///
    /// CPU endpoints bracket the wall endpoints by the sampling overhead. Linux
    /// CPU counters have microsecond units; neither value is an exact off-CPU
    /// attribution. Other worker threads and block initialization/finalization
    /// are outside this measurement.
    pub(super) fn finish(self) -> (u64, Option<u64>) {
        let wall = nanos(self.wall.elapsed());
        (wall, cpu_nanos(self.cpu.elapsed()))
    }
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn cpu_nanos(usage: Option<ThreadResourceUsageDelta>) -> Option<u64> {
    usage.map(|usage| nanos(usage.user_cpu_time.saturating_add(usage.system_cpu_time)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_accounting_does_not_start_timer() {
        assert!(ExecutionLoopTimer::start(false).is_none());
    }

    #[test]
    fn missing_cpu_is_distinct_from_measured_zero() {
        assert_eq!(cpu_nanos(None), None);
        assert_eq!(cpu_nanos(Some(ThreadResourceUsageDelta::default())), Some(0));
    }

    #[test]
    fn cpu_sum_and_nanosecond_conversion_saturate() {
        let usage = |user_cpu_time, system_cpu_time| {
            Some(ThreadResourceUsageDelta { user_cpu_time, system_cpu_time, ..Default::default() })
        };
        assert_eq!(
            cpu_nanos(usage(Duration::from_micros(2), Duration::from_micros(3))),
            Some(5000)
        );
        assert_eq!(cpu_nanos(usage(Duration::MAX, Duration::MAX)), Some(u64::MAX));
        assert_eq!(cpu_nanos(usage(Duration::from_secs(u64::MAX), Duration::ZERO)), Some(u64::MAX));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn sleep_is_elapsed_wall_not_thread_cpu() {
        let timer = ExecutionLoopTimer::start(true).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let (wall, cpu) = timer.finish();
        assert!(wall >= 50_000_000);
        assert!(cpu.unwrap() < wall);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn busy_work_consumes_thread_cpu() {
        let timer = ExecutionLoopTimer::start(true).unwrap();
        let mut value = 1u64;
        for _ in 0..1_000_000 {
            value = std::hint::black_box(value).wrapping_mul(3).wrapping_add(1);
        }
        std::hint::black_box(value);
        let (_, cpu) = timer.finish();
        assert!(cpu.unwrap() > 0);
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn unsupported_cpu_is_unmeasured() {
        let (_, cpu) = ExecutionLoopTimer::start(true).unwrap().finish();
        assert_eq!(cpu, None);
    }
}
