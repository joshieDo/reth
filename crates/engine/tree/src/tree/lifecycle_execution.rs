//! Gated, thread-bound accounting for the synchronous transaction execution loop.

use reth_metrics::thread::{ThreadResourceUsage, ThreadResourceUsageDelta};
use std::time::{Duration, Instant};

/// At most two CPU samples per completed loop, never per transaction. The guard is `!Send`,
/// so its start and end samples cannot move to another execution thread.
pub(super) struct ExecutionLoopTimer {
    cpu: ThreadResourceUsage,
    wall: Instant,
}

/// Matching measurements from the same two resource snapshots.
pub(super) struct ExecutionLoopMeasurement {
    pub(super) wall_ns: u64,
    pub(super) cpu_ns: Option<u64>,
    pub(super) resources: Option<ThreadResourceUsageDelta>,
}

impl ExecutionLoopMeasurement {
    fn new(wall_ns: u64, resources: Option<ThreadResourceUsageDelta>) -> Self {
        Self { wall_ns, cpu_ns: cpu_nanos(resources), resources }
    }
}

impl ExecutionLoopTimer {
    /// Match the event kind used for totals; reduced capture deliberately rejects span/HINT
    /// metadata.
    pub(super) fn accounting_enabled() -> bool {
        tracing::event_enabled!(target: "lifecycle", tracing::Level::INFO)
    }

    pub(super) fn start(enabled: bool) -> Option<Self> {
        enabled.then(|| Self { cpu: ThreadResourceUsage::now(), wall: Instant::now() })
    }

    /// Returns matching loop wall time and optional user+system thread CPU time.
    ///
    /// CPU endpoints bracket the wall endpoints by the sampling overhead. Linux
    /// CPU counters have microsecond units; neither value is an exact off-CPU
    /// attribution. Other worker threads and block initialization/finalization
    /// are outside this measurement.
    pub(super) fn finish(self) -> ExecutionLoopMeasurement {
        let wall = nanos(self.wall.elapsed());
        // Reuse the CPU sample's resource deltas; do not resample per counter.
        ExecutionLoopMeasurement::new(wall, self.cpu.elapsed())
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
    fn production_accounting_gate_accepts_event_only_subscriber() {
        use reth_tracing::tracing_subscriber::{layer::SubscriberExt, Layer};
        let subscriber = reth_tracing::tracing_subscriber::registry().with(
            reth_tracing::tracing_subscriber::fmt::layer().with_writer(std::io::sink).with_filter(
                reth_tracing::tracing_subscriber::filter::filter_fn(|meta| {
                    meta.is_event() && meta.target() == "lifecycle"
                }),
            ),
        );
        tracing::subscriber::with_default(subscriber, || {
            assert!(!tracing::enabled!(target: "lifecycle", tracing::Level::INFO));
            assert!(ExecutionLoopTimer::start(ExecutionLoopTimer::accounting_enabled()).is_some());
        });
        tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
            assert!(ExecutionLoopTimer::start(ExecutionLoopTimer::accounting_enabled()).is_none());
        });
    }

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
    fn resources_retain_counts_and_distinguish_missing_from_zero() {
        let usage = ThreadResourceUsageDelta {
            user_cpu_time: Duration::from_micros(2),
            system_cpu_time: Duration::from_micros(3),
            voluntary_context_switches: 7,
            involuntary_context_switches: 11,
            minor_page_faults: 13,
            major_page_faults: 17,
            block_input_operations: 19,
            block_output_operations: u64::MAX,
        };
        let measured = ExecutionLoopMeasurement::new(23_000, Some(usage));
        assert_eq!(measured.wall_ns, 23_000);
        assert_eq!(measured.cpu_ns, Some(5000));
        assert_eq!(measured.resources, Some(usage));
        let zero = ExecutionLoopMeasurement::new(0, Some(ThreadResourceUsageDelta::default()));
        assert_eq!(zero.cpu_ns, Some(0));
        assert_eq!(zero.resources, Some(ThreadResourceUsageDelta::default()));
        let unavailable = ExecutionLoopMeasurement::new(23_000, None);
        assert_eq!(unavailable.wall_ns, 23_000);
        assert_eq!(unavailable.cpu_ns, None);
        assert_eq!(unavailable.resources, None);
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
        let measured = timer.finish();
        assert!(measured.wall_ns >= 50_000_000);
        assert!(measured.cpu_ns.unwrap() < measured.wall_ns);
        assert!(measured.resources.is_some());
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
        assert!(timer.finish().cpu_ns.unwrap() > 0);
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn unsupported_cpu_is_unmeasured() {
        let measured = ExecutionLoopTimer::start(true).unwrap().finish();
        assert_eq!(measured.cpu_ns, None);
        assert_eq!(measured.resources, None);
    }
}
