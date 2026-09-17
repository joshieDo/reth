//! Optional current-thread accounting around a proof worker's synchronous run.

use crate::job_counts::{JobCounts, JobKind};
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
        // Milestone mode intentionally omits worker spans and these measurements.
        // Full mode retains the explicit worker parent even with other subscribers.
        let requested = *CAPTURE.get_or_init(|| {
            let detail = std::env::var("TEMPO_LIFECYCLE_DETAIL");
            let detail = match &detail {
                Ok(value) => Some(value.as_str()),
                Err(std::env::VarError::NotPresent) => None,
                Err(std::env::VarError::NotUnicode(_)) => return false,
            };
            full_capture_requested(std::env::var_os("RETH_LIFECYCLE_FILE").is_some(), detail)
        });
        Self::start_requested(requested)
    }

    fn start_requested(requested: bool) -> Option<Self> {
        // This enables event accounting, not a span or an untyped HINT callsite.
        Self::start_enabled(
            requested && tracing::event_enabled!(target: "lifecycle", tracing::Level::INFO),
        )
    }

    fn start_enabled(enabled: bool) -> Option<Self> {
        enabled.then(|| Self { cpu: ThreadResourceUsage::now(), wall: Instant::now() })
    }

    fn finish(self) -> (u64, Option<u64>) {
        let wall = nanos(self.wall.elapsed());
        (wall, cpu_nanos(self.cpu.elapsed()))
    }

    pub(crate) fn record(
        self,
        parent: &tracing::Span,
        stage: &'static str,
        success: bool,
        jobs: Option<&JobCounts>,
    ) {
        let (worker_run_ns, worker_thread_cpu_ns) = self.finish();
        tracing::info!(target: "lifecycle", parent: parent, stage, worker_run_ns, worker_thread_cpu_ns,
            worker_cpu_measured = u64::from(worker_thread_cpu_ns.is_some()),
            worker_success = u64::from(success),
            worker_job_counts_measured = u64::from(jobs.is_some()),
            worker_jobs = jobs.map(|j| j.jobs),
            worker_account_targets = jobs.filter(|j| j.kind == JobKind::Account).map(|j| j.targets),
            worker_storage_targets = jobs.filter(|j| j.kind == JobKind::Storage).map(|j| j.targets),
            worker_jobs_storage_only_single_group = jobs.filter(|j| j.kind == JobKind::Account).map(|j| j.storage_only_single_group),
            worker_storage_groups = jobs.filter(|j| j.kind == JobKind::Account).map(|j| j.storage_groups),
            worker_root_requests = jobs.filter(|j| j.kind == JobKind::Storage).map(|j| j.root_requests),
            worker_target_max = jobs.map(|j| j.max_targets),
            worker_jobs_targets_0 = jobs.map(|j| j.bins[0]),
            worker_jobs_targets_1 = jobs.map(|j| j.bins[1]),
            worker_jobs_targets_2_8 = jobs.map(|j| j.bins[2]),
            worker_jobs_targets_9_32 = jobs.map(|j| j.bins[3]),
            worker_jobs_targets_33_plus = jobs.map(|j| j.bins[4]),
            worker_job_counts_saturated = jobs.map(|j| u64::from(j.saturated)));
    }
}

fn full_capture_requested(file_present: bool, detail: Option<&str>) -> bool {
    file_present && matches!(detail, None | Some("full"))
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
    fn production_gate_accepts_event_only_subscriber() {
        struct EventsOnly;
        impl tracing::Subscriber for EventsOnly {
            fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
                meta.is_event() && meta.target() == "lifecycle"
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                unreachable!("event-only subscriber cannot create spans")
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, _: &tracing::Event<'_>) {}
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }
        tracing::subscriber::with_default(EventsOnly, || {
            assert!(!tracing::enabled!(target: "lifecycle", tracing::Level::INFO));
            assert!(WorkerCpuTimer::start_requested(true).is_some());
            assert!(WorkerCpuTimer::start_requested(false).is_none());
        });
        tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
            assert!(WorkerCpuTimer::start_requested(true).is_none());
        });
    }

    #[test]
    fn sampling_requires_full_capture() {
        assert!(!full_capture_requested(false, None));
        assert!(!full_capture_requested(false, Some("full")));
        assert!(!full_capture_requested(true, Some("milestones")));
        assert!(!full_capture_requested(false, Some("milestones")));
        assert!(!full_capture_requested(true, Some("unknown")));
        assert!(full_capture_requested(true, None));
        assert!(full_capture_requested(true, Some("full")));
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
