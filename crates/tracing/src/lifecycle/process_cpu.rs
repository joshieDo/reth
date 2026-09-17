//! Optional whole-process accounting. Read brackets are not exact snapshot instants.

use super::{CaptureRecord, Writer};
use std::{
    io::{self, Write},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub(super) const PERIOD_NS: u64 = 250_000_000;
const MAX_SAMPLES: u64 = 100_000;

pub(super) fn configured(value: Option<&str>, supported: bool) -> eyre::Result<bool> {
    let enabled = match value {
        None | Some("disabled") => false,
        Some("rusage_self_v1") => true,
        _ => eyre::bail!("TEMPO_LIFECYCLE_PROCESS_CPU must be disabled or rusage_self_v1"),
    };
    eyre::ensure!(!enabled || supported, "process CPU capture requires Linux");
    Ok(enabled)
}

#[derive(Default)]
pub(super) struct Totals {
    samples: AtomicU64,
    unavailable: AtomicU64,
    missed_deadlines: AtomicU64,
    failures: AtomicU64,
}

impl Totals {
    fn failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn footer(&self, value: &mut serde_json::Value) {
        for (key, counter) in [
            ("process_cpu_samples", &self.samples),
            ("process_cpu_unavailable", &self.unavailable),
            ("process_cpu_missed_deadlines", &self.missed_deadlines),
            ("process_cpu_failures", &self.failures),
        ] {
            value[key] = counter.load(Ordering::Relaxed).into();
        }
    }

    fn record(&self, sample: &Sample) -> bool {
        // Sample count and unavailable count cannot exceed MAX_SAMPLES.
        // Skips are time-derived, so check their independent accumulation.
        if self
            .missed_deadlines
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                old.checked_add(sample.missed_deadlines)
            })
            .is_err()
        {
            self.failure();
            return false
        }
        self.samples.fetch_add(1, Ordering::Relaxed);
        if sample.result.is_err() {
            self.unavailable.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Cpu {
    user: u64,
    system: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum Unavailable {
    Read = 1,
    Invalid = 2,
    Decreased = 3,
}

#[derive(Debug)]
pub(super) struct Sample {
    sequence: u64,
    read_start_ns: u64,
    read_end_ns: u64,
    missed_deadlines: u64,
    result: Result<Cpu, Unavailable>,
}

impl Sample {
    pub(super) fn write_json(&self, out: &mut impl Write) -> io::Result<()> {
        write!(out, "{{\"missed_deadlines\":{},\"read_end_ns\":{},\"read_start_ns\":{},\"sequence\":{},\"status\":{}",
            self.missed_deadlines, self.read_end_ns, self.read_start_ns, self.sequence,
            self.result.as_ref().err().map_or(0, |reason| *reason as u8))?;
        if let Ok(cpu) = self.result {
            write!(out, ",\"system_cpu_us\":{}", cpu.system)?;
        }
        out.write_all(b",\"type\":\"process_cpu\"")?;
        if let Ok(cpu) = self.result {
            write!(out, ",\"user_cpu_us\":{}", cpu.user)?;
        }
        out.write_all(b"}")
    }

    const fn fatal(&self) -> bool {
        matches!(self.result, Err(Unavailable::Invalid | Unavailable::Decreased))
    }
}

#[derive(Default)]
struct State {
    sequence: u64,
    end: u64,
    last_cpu: Option<Cpu>,
}

impl State {
    const fn observe(
        &mut self,
        start: u64,
        end: u64,
        mut result: Result<Cpu, Unavailable>,
        missed_deadlines: u64,
    ) -> Option<Sample> {
        if start < self.end || end < start || self.sequence >= MAX_SAMPLES {
            return None
        }
        if let (Ok(now), Some(last)) = (result, self.last_cpu) {
            if now.user < last.user || now.system < last.system {
                result = Err(Unavailable::Decreased);
            }
        }
        self.sequence += 1;
        self.end = end;
        if let Ok(cpu) = result {
            self.last_cpu = Some(cpu);
        }
        Some(Sample {
            sequence: self.sequence,
            read_start_ns: start,
            read_end_ns: end,
            missed_deadlines,
            result,
        })
    }
}

pub(super) struct Sampler {
    stop: Option<mpsc::Sender<()>>,
    worker: Option<JoinHandle<()>>,
    totals: Arc<Totals>,
}

impl Sampler {
    pub(super) fn start(writer: Arc<Writer>, epoch: u64, totals: Arc<Totals>) -> io::Result<Self> {
        let (stop, receiver) = mpsc::channel();
        let thread_totals = Arc::clone(&totals);
        let worker =
            thread::Builder::new().name("lifecycle-process-cpu".into()).spawn(move || {
                run(receiver, writer, epoch, &thread_totals);
            })?;
        Ok(Self { stop: Some(stop), worker: Some(worker), totals })
    }

    pub(super) fn stop(&mut self) {
        // Disconnect wakes recv_timeout immediately; no full-period shutdown sleep.
        self.stop.take();
        if self.worker.take().is_some_and(|worker| worker.join().is_err()) {
            self.totals.failure();
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop();
    }
}

// Advance directly to the first future deadline. No catch-up sampling burst.
fn advance(deadline: Instant, now: Instant, period: Duration) -> Option<(Instant, u64)> {
    let mut next = deadline.checked_add(period)?;
    if next > now {
        return Some((next, 0))
    }
    let skipped = u64::try_from(now.duration_since(next).as_nanos() / period.as_nanos())
        .ok()?
        .checked_add(1)?;
    let delta = period.as_nanos().checked_mul(u128::from(skipped))?;
    let seconds = u64::try_from(delta / 1_000_000_000).ok()?;
    next = next.checked_add(Duration::new(seconds, (delta % 1_000_000_000) as u32))?;
    Some((next, skipped))
}

fn run(stop: mpsc::Receiver<()>, writer: Arc<Writer>, epoch: u64, totals: &Totals) {
    run_with(stop, writer, totals, Duration::from_nanos(PERIOD_NS), MAX_SAMPLES, || {
        let start = source_clock()?.checked_sub(epoch)?;
        let cpu = process_cpu();
        let end = source_clock()?.checked_sub(epoch)?;
        Some((start, end, cpu))
    });
}

fn run_with(
    stop: mpsc::Receiver<()>,
    writer: Arc<Writer>,
    totals: &Totals,
    period: Duration,
    max_samples: u64,
    mut read: impl FnMut() -> Option<(u64, u64, Result<Cpu, Unavailable>)>,
) {
    let mut deadline = Instant::now();
    let mut pending_missed = 0;
    let mut state = State::default();
    loop {
        match stop.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        let Some((start, end, cpu)) = read() else {
            totals.failure();
            break
        };
        let Some(sample) = state.observe(start, end, cpu, pending_missed) else {
            totals.failure();
            break
        };
        let fatal = sample.fatal();
        if !totals.record(&sample) {
            break
        }
        writer.send(CaptureRecord::ProcessCpu(Box::new(sample)));
        if fatal || state.sequence == max_samples {
            totals.failure();
            break
        }
        let Some((next, missed)) = advance(deadline, Instant::now(), period) else {
            totals.failure();
            break
        };
        deadline = next;
        pending_missed = missed;
    }
}

#[cfg(target_os = "linux")]
fn source_clock() -> Option<u64> {
    let mut value = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid writable timespec and a process-independent monotonic clock.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut value) } != 0 {
        return None
    }
    let sec = u64::try_from(value.tv_sec).ok()?;
    let ns = u64::try_from(value.tv_nsec).ok().filter(|v| *v < 1_000_000_000)?;
    sec.checked_mul(1_000_000_000)?.checked_add(ns)
}

#[cfg(target_os = "linux")]
fn micros(value: libc::timeval) -> Option<u64> {
    let sec = u64::try_from(value.tv_sec).ok()?;
    let us = u64::try_from(value.tv_usec).ok().filter(|v| *v < 1_000_000)?;
    sec.checked_mul(1_000_000)?.checked_add(us)
}

#[cfg(target_os = "linux")]
fn process_cpu() -> Result<Cpu, Unavailable> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes this valid out pointer on success. Only the
    // user/system timeval fields are read; no other process metadata is exported.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(Unavailable::Read)
    }
    // SAFETY: the successful call above initialized usage.
    let usage = unsafe { usage.assume_init() };
    Ok(Cpu {
        user: micros(usage.ru_utime).ok_or(Unavailable::Invalid)?,
        system: micros(usage.ru_stime).ok_or(Unavailable::Invalid)?,
    })
}

#[cfg(not(target_os = "linux"))]
const fn source_clock() -> Option<u64> {
    None
}
#[cfg(not(target_os = "linux"))]
const fn process_cpu() -> Result<Cpu, Unavailable> {
    Err(Unavailable::Read)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn as_json(sample: &Sample) -> Value {
        let mut bytes = Vec::new();
        sample.write_json(&mut bytes).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn writer(capacity: usize) -> (Arc<Writer>, mpsc::Receiver<Option<CaptureRecord>>) {
        let (tx, rx) = mpsc::sync_channel(capacity);
        (Arc::new(Writer { tx, dropped: Arc::default(), prewarm_failures: Arc::default() }), rx)
    }

    #[test]
    fn configuration_is_closed_and_disabled_needs_no_platform() {
        for value in [None, Some("disabled")] {
            assert!(!configured(value, false).unwrap());
        }
        assert!(configured(Some("rusage_self_v1"), true).unwrap());
        assert!(configured(Some("rusage_self_v1"), false).is_err());
        for value in ["", "1", "true", "RUSAGE_SELF", "rusage_self_v2"] {
            assert!(configured(Some(value), true).is_err());
        }
    }

    #[test]
    fn exact_scalars_and_unavailable_never_emit_zero_cpu() {
        let mut state = State::default();
        let sample = state.observe(4, 5, Ok(Cpu { user: u64::MAX, system: u64::MAX }), 2).unwrap();
        assert_eq!(
            as_json(&sample),
            json!({"type":"process_cpu","sequence":1,"read_start_ns":4,"read_end_ns":5,"status":0,"missed_deadlines":2,"user_cpu_us":u64::MAX,"system_cpu_us":u64::MAX})
        );
        for reason in [Unavailable::Read, Unavailable::Invalid, Unavailable::Decreased] {
            let sample = state.observe(5, 5, Err(reason), 0).unwrap();
            let encoded = as_json(&sample);
            assert_eq!(encoded.as_object().unwrap().len(), 6);
            assert_eq!(encoded["status"], reason as u8);
            assert!(encoded.get("user_cpu_us").is_none());
            assert!(encoded.get("system_cpu_us").is_none());
        }
    }

    #[test]
    fn failed_read_does_not_reset_successful_counter_or_bracket_history() {
        let mut state = State::default();
        assert!(!state.observe(1, 2, Ok(Cpu { user: 20, system: 30 }), 0).unwrap().fatal());
        assert!(!state.observe(3, 4, Err(Unavailable::Read), 0).unwrap().fatal());
        assert_eq!(
            state.observe(5, 6, Ok(Cpu { user: 19, system: 40 }), 0).unwrap().result,
            Err(Unavailable::Decreased)
        );
        assert!(state.observe(5, 7, Ok(Cpu { user: 30, system: 40 }), 0).is_none());
        assert!(state.observe(9, 8, Ok(Cpu { user: 30, system: 40 }), 0).is_none());
        state.sequence = MAX_SAMPLES;
        assert!(state.observe(9, 10, Ok(Cpu { user: 30, system: 40 }), 0).is_none());
    }

    #[test]
    fn late_deadline_skips_to_future_without_burst() {
        let start = Instant::now();
        let period = Duration::from_nanos(PERIOD_NS);
        assert_eq!(advance(start, start, period), Some((start + period, 0)));
        assert_eq!(advance(start, start + period, period), Some((start + period * 2, 1)));
        assert_eq!(
            advance(start, start + period * 7 + Duration::from_millis(1), period),
            Some((start + period * 8, 7))
        );
    }

    #[test]
    fn cap_and_late_clock_failure_remain_in_footer_without_fabricated_samples() {
        let (writer, rx) = writer(8);
        let (_stop, receiver) = mpsc::channel();
        let totals = Totals::default();
        let mut calls = 0;
        run_with(receiver, writer, &totals, Duration::from_nanos(1), 10, || {
            calls += 1;
            if calls == 3 {
                None
            } else {
                Some((calls, calls, Ok(Cpu { user: calls, system: 0 })))
            }
        });
        let values = rx.try_iter().flatten().collect::<Vec<_>>();
        assert_eq!(values.len(), 2);
        let mut footer = json!({});
        totals.footer(&mut footer);
        assert_eq!(footer["process_cpu_samples"], 2);
        assert_eq!(footer["process_cpu_failures"], 1);

        let (writer, rx) = self::writer(8);
        let (_stop, receiver) = mpsc::channel();
        let totals = Totals::default();
        run_with(receiver, writer, &totals, Duration::from_nanos(1), 2, || {
            Some((0, 0, Err(Unavailable::Read)))
        });
        assert_eq!(rx.try_iter().count(), 2);
        let mut footer = json!({});
        totals.footer(&mut footer);
        assert_eq!(footer["process_cpu_samples"], 2);
        assert_eq!(footer["process_cpu_unavailable"], 2);
        assert_eq!(footer["process_cpu_failures"], 1);
    }

    #[test]
    fn invalid_accounting_stops_and_full_queue_preserves_loss() {
        let (writer, _rx) = writer(0);
        let (_stop, receiver) = mpsc::channel();
        let totals = Totals::default();
        run_with(receiver, Arc::clone(&writer), &totals, Duration::from_nanos(1), 2, || {
            Some((0, 0, Err(Unavailable::Invalid)))
        });
        assert_eq!(writer.dropped.load(Ordering::Relaxed), 1);
        let mut footer = json!({});
        totals.footer(&mut footer);
        assert_eq!(footer["process_cpu_failures"], 1);
        assert_eq!(footer["process_cpu_unavailable"], 1);
        assert_eq!(footer["process_cpu_samples"], 1);
    }

    #[test]
    fn skip_count_overflow_is_fatal_without_wrapping() {
        let totals = Totals::default();
        totals.missed_deadlines.store(u64::MAX, Ordering::Relaxed);
        let sample = State::default().observe(0, 0, Ok(Cpu { user: 0, system: 0 }), 1).unwrap();
        assert!(!totals.record(&sample));
        assert_eq!(totals.samples.load(Ordering::Relaxed), 0);
        assert_eq!(totals.failures.load(Ordering::Relaxed), 1);
        assert_eq!(totals.missed_deadlines.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn stop_disconnect_interrupts_a_long_deadline() {
        let (writer, rx) = writer(8);
        let (stop, receiver) = mpsc::channel();
        let totals = Arc::new(Totals::default());
        let copy = Arc::clone(&totals);
        let (done, finished) = mpsc::channel();
        let join = thread::spawn(move || {
            run_with(receiver, writer, &copy, Duration::from_secs(60), 5, || {
                Some((0, 0, Ok(Cpu { user: 0, system: 0 })))
            });
            done.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(stop);
        finished.recv_timeout(Duration::from_secs(2)).unwrap();
        join.join().unwrap();
        assert_eq!(totals.samples.load(Ordering::Relaxed), 1);
        assert_eq!(totals.failures.load(Ordering::Relaxed), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn timeval_units_ranges_and_checked_arithmetic() {
        assert_eq!(micros(libc::timeval { tv_sec: 12, tv_usec: 34 }), Some(12_000_034));
        for value in [
            libc::timeval { tv_sec: -1, tv_usec: 0 },
            libc::timeval { tv_sec: 0, tv_usec: -1 },
            libc::timeval { tv_sec: 0, tv_usec: 1_000_000 },
            libc::timeval { tv_sec: i64::MAX, tv_usec: 0 },
        ] {
            assert_eq!(micros(value), None);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn actual_sampler_joins_before_writer_footer_and_disabled_has_no_samples() {
        use super::super::{CaptureDetail, LifecycleLayer};
        for enabled in [false, true] {
            let path = std::env::temp_dir()
                .join(format!("process-cpu-test-{}-{enabled}", std::process::id()));
            let file =
                std::fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
            let epoch = source_clock().unwrap();
            let (layer, guard) = LifecycleLayer::start_observers(
                file,
                [0; 32],
                epoch,
                CaptureDetail::Full,
                false,
                enabled,
            )
            .unwrap();
            assert_eq!(guard.process_cpu.is_some(), enabled);
            if let Some(sampler) = &guard.process_cpu {
                let timeout = Instant::now() + Duration::from_secs(2);
                while sampler.totals.samples.load(Ordering::Relaxed) == 0 {
                    assert!(Instant::now() < timeout);
                    thread::yield_now();
                }
            }
            drop(layer);
            drop(guard);
            let data = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            let rows = data
                .lines()
                .map(|line| serde_json::from_str::<Value>(line).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(rows[0]["process_cpu"], if enabled { "rusage_self_v1" } else { "disabled" });
            let footer = rows.last().unwrap();
            assert_eq!(footer["type"], "footer");
            assert_eq!(footer["dropped"], 0);
            let samples =
                rows.iter().filter(|row| row["type"] == "process_cpu").collect::<Vec<_>>();
            if enabled {
                assert_eq!(footer["process_cpu_samples"], samples.len() as u64);
                assert_eq!(footer["process_cpu_failures"], 0);
                assert!(samples.iter().all(|sample| sample["status"] == 0));
            } else {
                assert!(samples.is_empty());
                assert!(footer.get("process_cpu_samples").is_none());
                assert!(rows[0].get("process_cpu_period_ns").is_none());
            }
        }
    }
    #[cfg(target_os = "linux")]
    #[test]
    fn actual_self_accounting_includes_workers_and_excludes_waited_children() {
        const CASE: &str = "lifecycle::process_cpu::tests::actual_self_accounting_includes_workers_and_excludes_waited_children";
        const MODE: &str = "RETH_TEST_PROCESS_CPU_CASE";
        fn busy_thread(micros: u64) {
            let thread_cpu = || {
                let mut stamp = libc::timespec { tv_sec: 0, tv_nsec: 0 };
                // SAFETY: valid writable timespec, current thread's CPU clock.
                assert_eq!(
                    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut stamp) },
                    0
                );
                u64::try_from(stamp.tv_sec).unwrap() * 1_000_000 +
                    u64::try_from(stamp.tv_nsec).unwrap() / 1_000
            };
            let start = thread_cpu();
            let timeout = Instant::now() + Duration::from_secs(8);
            let mut value = 1_u64;
            while thread_cpu() - start < micros {
                for _ in 0..2_000 {
                    value = std::hint::black_box(value.wrapping_mul(3).wrapping_add(1));
                }
                assert!(Instant::now() < timeout);
            }
            std::hint::black_box(value);
        }
        fn waited_children_cpu() -> u64 {
            let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
            // SAFETY: valid out pointer initialized by a successful getrusage.
            assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_CHILDREN, usage.as_mut_ptr()) }, 0);
            // SAFETY: the checked call initialized all fields.
            let usage = unsafe { usage.assume_init() };
            micros(usage.ru_utime).unwrap() + micros(usage.ru_stime).unwrap()
        }
        let mode = std::env::var(MODE).ok();
        if mode.as_deref() == Some("child") {
            busy_thread(80_000);
            return
        }
        if mode.as_deref() == Some("parent") {
            let before = process_cpu().unwrap();
            thread::spawn(|| busy_thread(20_000)).join().unwrap();
            let after = process_cpu().unwrap();
            assert!(after.user + after.system - before.user - before.system >= 15_000);
            let child_before = waited_children_cpu();
            let before = process_cpu().unwrap();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", CASE, "--test-threads=1"])
                .env(MODE, "child")
                .output()
                .unwrap();
            assert!(status.status.success());
            let after = process_cpu().unwrap();
            let child_delta = waited_children_cpu() - child_before;
            assert!(child_delta >= 70_000);
            assert!(after.user + after.system - before.user - before.system < child_delta / 2);
            return
        }
        // A separate test process prevents concurrent tests' own CPU from
        // contaminating the self-versus-child accounting assertion.
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CASE, "--test-threads=1"])
            .env(MODE, "parent")
            .output()
            .unwrap();
        assert!(result.status.success(), "isolated process accounting probe failed");
    }
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "manual producer fixture; requires a new RETH_PROCESS_CPU_FIXTURE_DIR"]
    fn emit_process_cpu_fixture() {
        use super::super::{CaptureDetail, LifecycleLayer};
        let directory =
            std::path::PathBuf::from(std::env::var_os("RETH_PROCESS_CPU_FIXTURE_DIR").unwrap());
        std::fs::create_dir(&directory).unwrap();
        let epoch = source_clock().unwrap();
        let mut captures = Vec::new();
        for name in ["a.jsonl", "b.jsonl"] {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join(name))
                .unwrap();
            captures.push(
                LifecycleLayer::start_observers(
                    file,
                    [0; 32],
                    epoch,
                    CaptureDetail::Milestones,
                    false,
                    true,
                )
                .unwrap(),
            );
        }
        let limit = Instant::now() + Duration::from_secs(5);
        while captures.iter().any(|(_, guard)| {
            guard.process_cpu.as_ref().unwrap().totals.samples.load(Ordering::Relaxed) < 3
        }) {
            assert!(Instant::now() < limit);
            thread::sleep(Duration::from_millis(1));
        }
        drop(captures);
        let window =
            json!({"start_ns":0,"end_ns":source_clock().unwrap()-epoch,"stop_reason":"completed"});
        std::fs::write(directory.join("window.json"), serde_json::to_vec(&window).unwrap())
            .unwrap();
    }
    #[test]
    fn sampler_panic_is_sticky_and_stop_is_idempotent() {
        let totals = Arc::new(Totals::default());
        let (sender, _receiver) = mpsc::channel();
        let worker = thread::spawn(|| panic!("injected sampler failure"));
        let mut sampler =
            Sampler { stop: Some(sender), worker: Some(worker), totals: Arc::clone(&totals) };
        sampler.stop();
        sampler.stop();
        assert_eq!(totals.failures.load(Ordering::Relaxed), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "manual fixed-duration sampler overhead fixture"]
    fn process_cpu_overhead_fixture() {
        use super::super::{CaptureDetail, LifecycleLayer};
        use std::sync::{atomic::AtomicBool, Barrier};
        let Some(compiled_source) = option_env!("RETH_PROCESS_CPU_SOURCE_SHA") else {
            panic!("fixture must be compiled with its exact source hash")
        };
        let enabled = match std::env::var("RETH_PROCESS_CPU_PROBE_MODE").unwrap().as_str() {
            "0" => false,
            "1" => true,
            _ => panic!("invalid fixture mode"),
        };
        let workers: usize =
            std::env::var("RETH_PROCESS_CPU_PROBE_WORKERS").unwrap().parse().unwrap();
        assert!([0, 2, 32, 64].contains(&workers));
        let directory =
            std::path::PathBuf::from(std::env::var_os("RETH_PROCESS_CPU_PROBE_DIR").unwrap());
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("compiled-source.sha256"), compiled_source).unwrap();
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("source.jsonl"))
            .unwrap();
        let start = Arc::new(Barrier::new(workers + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let handles = (0..workers)
            .map(|_| {
                let start = Arc::clone(&start);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    start.wait();
                    let mut count = 0_u64;
                    let mut value = 1_u64;
                    while !stop.load(Ordering::Relaxed) {
                        for _ in 0..1_000 {
                            value = std::hint::black_box(value.wrapping_mul(3).wrapping_add(1));
                        }
                        count += 1_000;
                    }
                    std::hint::black_box(value);
                    count
                })
            })
            .collect::<Vec<_>>();
        let wall_before = source_clock().unwrap();
        let cpu_before = process_cpu().unwrap();
        let cpu_start_end = source_clock().unwrap();
        let (layer, guard) = LifecycleLayer::start_observers(
            file,
            [0; 32],
            wall_before,
            CaptureDetail::Milestones,
            false,
            enabled,
        )
        .unwrap();
        start.wait();
        let work_start = source_clock().unwrap();
        thread::sleep(Duration::from_secs(2));
        let work_stop = source_clock().unwrap();
        stop.store(true, Ordering::Relaxed);
        let iterations = handles.into_iter().map(|worker| worker.join().unwrap()).sum::<u64>();
        drop(layer);
        drop(guard);
        let cpu_end_start = source_clock().unwrap();
        let cpu_after = process_cpu().unwrap();
        let wall_after = source_clock().unwrap();
        // Validation and publication follow measurement. All source rows and
        // exact brackets remain private fixture evidence; no native IDs emitted.
        let source = std::fs::read_to_string(directory.join("source.jsonl")).unwrap();
        let rows = source
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let footer = rows.last().unwrap();
        assert_eq!(footer["type"], "footer");
        assert_eq!(footer["dropped"], 0);
        assert_eq!(footer["io_error"], false);
        if enabled {
            assert_eq!(footer["process_cpu_failures"], 0);
        }
        let report = json!({"schema":1,"mode":u8::from(enabled),"workers":workers,
            "elapsed_ns":wall_after-wall_before,"work_window_ns":work_stop-work_start,
            "cpu_start_read_end_ns":cpu_start_end-wall_before,"cpu_end_read_start_ns":cpu_end_start-wall_before,
            "user_cpu_us":cpu_after.user-cpu_before.user,"system_cpu_us":cpu_after.system-cpu_before.system,
            "work_iterations":iterations});
        std::fs::write(directory.join("numeric.json"), serde_json::to_vec(&report).unwrap())
            .unwrap();
    }
}
