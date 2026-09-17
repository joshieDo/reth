//! Optional current-thread accounting around a proof worker's synchronous run.

use reth_metrics::thread::{ThreadResourceUsage, ThreadResourceUsageDelta};
use std::{
    sync::OnceLock,
    time::{Duration, Instant},
};

/// Two samples per worker invocation, never per proof job. `ThreadResourceUsage`
/// makes this timer thread-bound. Construction and result-error forwarding are
/// outside its boundary; the worker's receive waits and teardown are inside.
pub(crate) struct WorkerCpuTimer {
    cpu: ThreadResourceUsage,
    wall: Instant,
}

impl WorkerCpuTimer {
    pub(crate) fn start() -> Option<Self> {
        static CAPTURE: OnceLock<bool> = OnceLock::new();
        // Both modes retain coarse worker identity spans, so the explicit parent
        // remains visible to the lifecycle layer even with other subscribers.
        let enabled = *CAPTURE.get_or_init(|| {
            let detail = std::env::var("TEMPO_LIFECYCLE_DETAIL");
            let detail = match &detail {
                Ok(value) => Some(value.as_str()),
                Err(std::env::VarError::NotPresent) => None,
                Err(std::env::VarError::NotUnicode(_)) => return false,
            };
            capture_requested(std::env::var_os("RETH_LIFECYCLE_FILE").is_some(), detail)
        }) && tracing::enabled!(target: "lifecycle", tracing::Level::INFO);
        Self::start_enabled(enabled)
    }

    fn start_enabled(enabled: bool) -> Option<Self> {
        enabled.then(|| Self { cpu: ThreadResourceUsage::now(), wall: Instant::now() })
    }

    fn finish(self) -> (u64, Option<u64>) {
        let wall = nanos(self.wall.elapsed());
        (wall, cpu_nanos(self.cpu.elapsed()))
    }

    pub(crate) fn record(self, parent: &tracing::Span, stage: &'static str, success: bool) {
        let (worker_run_ns, worker_thread_cpu_ns) = self.finish();
        tracing::info!(target: "lifecycle", parent: parent, stage, worker_run_ns, worker_thread_cpu_ns,
            worker_cpu_measured = u64::from(worker_thread_cpu_ns.is_some()),
            worker_success = u64::from(success));
    }
}

fn capture_requested(file_present: bool, detail: Option<&str>) -> bool {
    file_present && matches!(detail, None | Some("full" | "milestones"))
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
    fn sampling_requires_coarse_or_full_capture() {
        assert!(!capture_requested(false, None));
        assert!(!capture_requested(false, Some("full")));
        assert!(capture_requested(true, Some("milestones")));
        assert!(!capture_requested(false, Some("milestones")));
        assert!(!capture_requested(true, Some("unknown")));
        assert!(capture_requested(true, None));
        assert!(capture_requested(true, Some("full")));
    }

    #[test]
    fn unavailable_is_not_zero_and_conversion_saturates() {
        assert!(WorkerCpuTimer::start_enabled(false).is_none());
        assert_eq!(cpu_nanos(None), None);
        assert_eq!(cpu_nanos(Some(ThreadResourceUsageDelta::default())), Some(0));
        let usage = ThreadResourceUsageDelta {
            user_cpu_time: Duration::MAX,
            system_cpu_time: Duration::MAX,
            ..Default::default()
        };
        assert_eq!(cpu_nanos(Some(usage)), Some(u64::MAX));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn receiver_wait_is_wall_time_not_cpu() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let timer = WorkerCpuTimer::start_enabled(true).unwrap();
        let sender = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
        let (wall, cpu) = timer.finish();
        sender.join().unwrap();
        assert!(wall >= 50_000_000);
        assert!(cpu.unwrap() < wall);
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn unsupported_cpu_is_unavailable() {
        let (_, cpu) = WorkerCpuTimer::start_enabled(true).unwrap().finish();
        assert_eq!(cpu, None);
    }
}
