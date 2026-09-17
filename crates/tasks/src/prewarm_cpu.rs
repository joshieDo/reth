//! Opt-in benchmark accounting for selected synchronous prewarming calls.
//!
//! CPU is inclusive current-thread call residency, not exclusive task CPU.
//! Remote worker CPU is excluded; nested same-thread helping may be included.

use std::{
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};
use tracing::Span;

const MAX_CONTEXTS: usize = 128;
static LIVE_CONTEXTS: AtomicUsize = AtomicUsize::new(0);

/// Selected synchronous call site, never a pool or transaction identifier.
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum Role {
    /// Engine transaction prewarming.
    Engine = 1,
    /// Payload-builder transaction prewarming.
    Builder = 2,
}

/// Declared prewarming selection; only transaction leaves have CPU measurements.
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum Mode {
    /// Transaction prewarming is selected.
    Transactions = 1,
    /// BAL work is outside this observer's selected coverage.
    Bal = 2,
    /// Prewarming was skipped at the observed selection point.
    Skipped = 3,
}

/// Fixed outcome of the selected call, not transaction semantic success.
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum Outcome {
    /// The EVM/provider could not be initialized.
    EvmUnavailable = 1,
    /// A stop flag prevented execution.
    Stopped = 2,
    /// Engine execution already passed this transaction.
    AlreadyExecuted = 3,
    /// The EVM returned an error.
    ExecutionError = 4,
    /// The EVM returned normally, including transaction reverts.
    Executed = 5,
    /// Engine execution finished but a stop flag prevented later hints.
    ExecutedThenStopped = 6,
    /// A builder transaction was not eligible for parallel execution.
    ParallelIneligible = 7,
    /// Builder execution finished without replay actions.
    ReplayUnavailable = 8,
    /// Builder execution returned replay actions.
    WithReplay = 9,
}

/// One context per observed prewarming selection; disabled is allocation-free.
#[derive(Debug, Default)]
pub struct Context(Option<Arc<Inner>>);

#[derive(Debug)]
struct Inner {
    span: Span,
    dispatched: AtomicU64,
    started: AtomicU64,
    completed: AtomicU64,
    // Marked only after the existing scope has joined every dispatched closure.
    finished: bool,
}

fn requested() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("TEMPO_LIFECYCLE_PREWARM_CPU").as_deref() == Ok("leaf_v1") &&
            std::env::var_os("RETH_LIFECYCLE_FILE").is_some()
    }) && tracing::event_enabled!(target: "lifecycle", tracing::Level::INFO)
}

fn failure(parent: &Span, reason: u64) {
    tracing::info!(target: "lifecycle", parent: parent,
        stage="prewarm_coverage_failure", prewarm_failure=reason);
}

fn increment(value: &AtomicU64, parent: &Span) -> Option<u64> {
    value
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map(|n| n + 1)
        .map_err(|_| failure(parent, 2))
        .ok()
}

impl Context {
    /// Starts an observed selection, or returns an inert context when disabled.
    pub fn new(role: Role, mode: Mode) -> Self {
        Self::with_parent(role, mode, &Span::current())
    }

    /// Starts with an identity parent retained before moving work to another thread.
    pub fn with_parent(role: Role, mode: Mode, parent: &Span) -> Self {
        Self::new_with_parent(requested(), role, mode, parent)
    }

    #[cfg(test)]
    fn new_enabled(enabled: bool, role: Role, mode: Mode) -> Self {
        Self::new_with_parent(enabled, role, mode, &Span::current())
    }

    fn new_with_parent(enabled: bool, role: Role, mode: Mode, parent: &Span) -> Self {
        if !enabled {
            return Self::default()
        }
        if LIVE_CONTEXTS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < MAX_CONTEXTS).then_some(n + 1)
            })
            .is_err()
        {
            failure(parent, 1);
            return Self::default()
        }
        let span = tracing::debug_span!(target: "lifecycle", parent: parent, "prewarm.context",
            prewarm_role=role as u64, prewarm_mode=mode as u64);
        tracing::info!(target: "lifecycle", parent: &span, stage="prewarm_context_started",
            prewarm_role=role as u64, prewarm_mode=mode as u64);
        Self(Some(Arc::new(Inner {
            span,
            dispatched: AtomicU64::new(0),
            started: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            finished: false,
        })))
    }

    /// Declares one dispatch. Its ordinal need not start in source-time order.
    pub fn dispatch(&self) -> Job {
        let Some(inner) = &self.0 else { return Job::default() };
        let Some(ordinal) = increment(&inner.dispatched, &inner.span) else {
            return Job::default()
        };
        Job(Some((Arc::clone(inner), ordinal)))
    }

    /// Completes after the existing scope has joined all its leaf closures.
    pub fn finish(mut self) {
        let Some(inner) = self.0.take() else { return };
        match Arc::try_unwrap(inner) {
            Ok(mut inner) => inner.finished = true,
            Err(inner) => failure(&inner.span, 3),
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let dispatched = self.dispatched.load(Ordering::Relaxed);
        let started = self.started.load(Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Relaxed);
        let unwound = std::thread::panicking();
        if (!self.finished && !unwound) ||
            (self.finished && (dispatched != started || started != completed))
        {
            failure(&self.span, 4);
        }
        tracing::info!(target: "lifecycle", parent: &self.span, stage="prewarm_context_completed",
            prewarm_dispatched=dispatched, prewarm_started=started, prewarm_completed=completed,
            prewarm_context_outcome=u64::from(unwound));
        LIVE_CONTEXTS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A dispatch token; moving it into a closure allocates no per-leaf state.
#[derive(Debug, Default)]
pub struct Job(Option<(Arc<Inner>, u64)>);

impl Job {
    /// Starts on the thread that invokes the selected synchronous leaf.
    pub fn start(self) -> Leaf {
        let Some((inner, ordinal)) = self.0 else { return Leaf::default() };
        increment(&inner.started, &inner.span);
        tracing::info!(target: "lifecycle", parent: &inner.span,
            stage="prewarm_leaf_started", prewarm_leaf=ordinal);
        let cpu = current_thread_cpu();
        Leaf { inner: Some((inner, ordinal)), cpu, outcome: None, not_send: PhantomData }
    }
}

/// Thread-bound inclusive call measurement. Missing normal outcome is a failure.
#[derive(Debug, Default)]
pub struct Leaf {
    inner: Option<(Arc<Inner>, u64)>,
    cpu: Option<u64>,
    outcome: Option<Outcome>,
    not_send: PhantomData<Rc<()>>,
}

impl Leaf {
    /// Declares the normal return path before the guard leaves scope.
    pub const fn outcome(&mut self, outcome: Outcome) {
        self.outcome = Some(outcome);
    }
}

impl Drop for Leaf {
    fn drop(&mut self) {
        let Some((inner, ordinal)) = &self.inner else { return };
        let cpu =
            self.cpu.zip(current_thread_cpu()).and_then(|(start, end)| end.checked_sub(start));
        let outcome =
            if std::thread::panicking() { Some(0) } else { self.outcome.map(|o| o as u64) };
        if outcome.is_none() {
            failure(&inner.span, 5);
        }
        tracing::info!(target: "lifecycle", parent: &inner.span,
            stage="prewarm_leaf_completed", prewarm_leaf=*ordinal,
            prewarm_cpu_measured=u64::from(cpu.is_some()), prewarm_thread_cpu_ns=cpu,
            prewarm_outcome=outcome);
        increment(&inner.completed, &inner.span);
    }
}

#[cfg(target_os = "linux")]
fn current_thread_cpu() -> Option<u64> {
    let mut value = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid writable timespec, and this clock samples the calling thread.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut value) } != 0 {
        return None
    }
    let seconds = u64::try_from(value.tv_sec).ok()?;
    let nanos = u64::try_from(value.tv_nsec).ok()?;
    if nanos >= 1_000_000_000 {
        return None
    }
    seconds.checked_mul(1_000_000_000)?.checked_add(nanos)
}

#[cfg(not(target_os = "linux"))]
const fn current_thread_cpu() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[derive(Clone, Default)]
    struct Events(Arc<std::sync::Mutex<Vec<std::collections::BTreeMap<String, String>>>>);

    impl tracing::Subscriber for Events {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            #[derive(Default)]
            struct Fields(std::collections::BTreeMap<String, String>);
            impl tracing::field::Visit for Fields {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    self.0.insert(field.name().into(), format!("{value:?}"));
                }
                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    self.0.insert(field.name().into(), value.into());
                }
            }
            let mut fields = Fields::default();
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[test]
    fn unset_normal_outcome_is_failure_but_panic_is_unwound() {
        let _lock = SERIAL.lock().unwrap();
        let events = Events::default();
        tracing::subscriber::with_default(events.clone(), || {
            let context = Context::new_enabled(true, Role::Engine, Mode::Transactions);
            drop(context.dispatch().start());
            let job = context.dispatch();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _leaf = job.start();
                panic!("synthetic selected-call unwind");
            }));
            assert!(result.is_err());
            context.finish();
        });
        let rows = events.0.lock().unwrap();
        let failures: Vec<_> = rows
            .iter()
            .filter(|r| r.get("stage").map(String::as_str) == Some("prewarm_coverage_failure"))
            .collect();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0]["prewarm_failure"], "5");
        let ends: Vec<_> = rows
            .iter()
            .filter(|r| r.get("stage").map(String::as_str) == Some("prewarm_leaf_completed"))
            .collect();
        assert_eq!(ends.len(), 2);
        assert!(!ends[0].contains_key("prewarm_outcome"));
        assert_eq!(ends[1]["prewarm_outcome"], "0");
        let summary = rows.last().unwrap();
        assert_eq!(summary["prewarm_dispatched"], "2");
        assert_eq!(summary["prewarm_started"], "2");
        assert_eq!(summary["prewarm_completed"], "2");
        assert_eq!(LIVE_CONTEXTS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn premature_finish_is_not_declared_complete_coverage() {
        let _lock = SERIAL.lock().unwrap();
        let events = Events::default();
        tracing::subscriber::with_default(events.clone(), || {
            let context = Context::new_enabled(true, Role::Builder, Mode::Transactions);
            let job = context.dispatch();
            context.finish();
            assert_eq!(LIVE_CONTEXTS.load(Ordering::Relaxed), 1);
            let mut leaf = job.start();
            leaf.outcome(Outcome::WithReplay);
        });
        let rows = events.0.lock().unwrap();
        let failures: Vec<_> = rows
            .iter()
            .filter(|r| r.get("stage").map(String::as_str) == Some("prewarm_coverage_failure"))
            .collect();
        assert_eq!(
            failures.iter().map(|r| r["prewarm_failure"].as_str()).collect::<Vec<_>>(),
            ["3", "4"]
        );
        assert_eq!(rows.last().unwrap()["prewarm_completed"], "1");
        assert_eq!(LIVE_CONTEXTS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn disabled_context_has_no_retained_state() {
        let _lock = SERIAL.lock().unwrap();
        let before = LIVE_CONTEXTS.load(Ordering::Relaxed);
        let context = Context::new_enabled(false, Role::Engine, Mode::Transactions);
        assert!(context.0.is_none());
        let leaf = context.dispatch().start();
        assert!(leaf.inner.is_none() && leaf.cpu.is_none());
        drop(leaf);
        context.finish();
        assert_eq!(LIVE_CONTEXTS.load(Ordering::Relaxed), before);
    }

    #[test]
    fn moved_jobs_complete_on_their_invoking_threads() {
        let _lock = SERIAL.lock().unwrap();
        let context = Context::new_enabled(true, Role::Builder, Mode::Transactions);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let job = context.dispatch();
                scope.spawn(move || {
                    let mut leaf = job.start();
                    leaf.outcome(Outcome::Executed);
                });
            }
        });
        let inner = context.0.as_ref().unwrap();
        assert_eq!(inner.dispatched.load(Ordering::Relaxed), 8);
        assert_eq!(inner.started.load(Ordering::Relaxed), 8);
        assert_eq!(inner.completed.load(Ordering::Relaxed), 8);
        context.finish();
        assert_eq!(LIVE_CONTEXTS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn stale_jobs_keep_context_cap_until_last_drop() {
        let _lock = SERIAL.lock().unwrap();
        let contexts: Vec<_> = (0..MAX_CONTEXTS)
            .map(|_| Context::new_enabled(true, Role::Engine, Mode::Transactions))
            .collect();
        let jobs: Vec<_> = contexts.iter().map(Context::dispatch).collect();
        drop(contexts);
        assert_eq!(LIVE_CONTEXTS.load(Ordering::Relaxed), MAX_CONTEXTS);
        assert!(Context::new_enabled(true, Role::Engine, Mode::Transactions).0.is_none());
        drop(jobs);
        assert_eq!(LIVE_CONTEXTS.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn ordinal_overflow_does_not_wrap() {
        let _lock = SERIAL.lock().unwrap();
        let context = Context::new_enabled(true, Role::Engine, Mode::Transactions);
        let inner = context.0.as_ref().unwrap();
        inner.dispatched.store(u64::MAX, Ordering::Relaxed);
        assert!(context.dispatch().0.is_none());
        assert_eq!(inner.dispatched.load(Ordering::Relaxed), u64::MAX);
        drop(context);
    }
}
