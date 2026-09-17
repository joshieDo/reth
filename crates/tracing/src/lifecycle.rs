//! Opt-in benchmark capture. Only structural metadata and allowlisted fields reach disk.

mod process_cpu;

use serde_json::{json, Value};
use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt,
    fs::{File, OpenOptions},
    io::{self, BufWriter, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
};
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
    Event, Subscriber,
};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

// Proof/update bursts can briefly outrun the disk writer. Keep capture bounded
// without dropping valid measurements during short scheduling or I/O stalls.
const QUEUE_CAPACITY: usize = 262_144;
const WRITE_BUFFER_BYTES: usize = 1024 * 1024;
static NEXT_THREAD: AtomicU64 = AtomicU64::new(1);
thread_local! { static THREAD: Cell<u64> = const { Cell::new(0) }; }

/// Captures a source-timestamped, privacy-filtered benchmark stream.
pub(crate) struct LifecycleLayer {
    detail: CaptureDetail,
    prewarm_cpu: bool,
    writer: Arc<Writer>,
    key: [u8; 32],
    epoch: u64,
    next_span: AtomicU64,
    backpressure_seen: AtomicBool,
    root_aggregates: Aggregates,
}

/// Runtime detail selection leaves cutoff and privacy handling unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum CaptureDetail {
    #[default]
    Full,
    Milestones,
}

impl CaptureDetail {
    fn parse(value: Option<&str>) -> eyre::Result<Self> {
        match value {
            None | Some("full") => Ok(Self::Full),
            Some("milestones") => Ok(Self::Milestones),
            _ => eyre::bail!("TEMPO_LIFECYCLE_DETAIL must be full or milestones"),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Milestones => "milestones",
        }
    }

    pub(crate) fn capture_metadata(self, meta: &tracing::Metadata<'_>) -> bool {
        if self == Self::Full || meta.is_event() {
            return capture_metadata(meta)
        }
        // These scopes carry otherwise implicit attempt/execution identities.
        // Filtering at enablement avoids allocating detailed spans at all.
        (meta.target().starts_with("tempo_consensus") &&
            matches!(meta.name(), "handle_propose" | "handle_verify" | "verify")) ||
            (meta.target() == "engine::tree::payload_validator" &&
                matches!(
                    meta.name(),
                    "validate_block_with_state" | "execute_block" | "execute_block_bal"
                )) ||
            (meta.target() == "trie::proof_task" &&
                matches!(meta.name(), "storage_worker" | "account_worker")) ||
            (meta.target() == "lifecycle" && meta.name() == "prewarm.context")
    }
}

#[derive(Default)]
struct MilestoneStage {
    keep: bool,
}

impl Visit for MilestoneStage {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() != "stage" {
            return
        }
        self.keep = matches!(
            value,
            "proposal_start" |
                "payload_built" |
                "proposal_ready" |
                "digest_released" |
                "body_ready" |
                "verify_start" |
                "verify_done" |
                "replay_start" |
                "replay_done" |
                "notarized" |
                "finalized" |
                "vote_sent" |
                "notarize_vote_sent" |
                "finalize_vote_sent" |
                "finalization_received" |
                "certified" |
                "cancelled" |
                "load_start" |
                "load_end" |
                "execution_totals" |
                "proof_storage_worker_totals" |
                "proof_account_worker_totals" |
                "prewarm_context_started" |
                "prewarm_leaf_started" |
                "prewarm_leaf_completed" |
                "prewarm_context_completed" |
                "prewarm_coverage_failure" |
                "backpressure_start" |
                "proposal_failed"
        );
    }

    fn record_debug(&mut self, _: &Field, _: &dyn fmt::Debug) {}
}

// Keep the frequent fixed-shape records inline. The producer captures all timing
// and identity before enqueueing; only JSON formatting moves to the writer.
#[derive(Debug)]
enum CaptureRecord {
    Json(Value),
    Start(Box<SpanStart>),
    Fields(Box<FieldRecord>),
    ProcessCpu(Box<process_cpu::Sample>),
    Enter { id: u64, ts: u64, thread: u64 },
    Exit { id: u64, ts: u64, thread: u64 },
    End { id: u64, ts: u64, thread: u64 },
}

/// Field names come from static tracing metadata or fixed canonical aliases.
/// An ordered map preserves the existing sorted JSON without owning each key.
/// Values retain the same privacy filtering and ownership as before.
type FieldMap = BTreeMap<&'static str, Value>;

/// Only the variable-size start record is boxed, keeping queue slots unchanged.
/// Metadata strings are static; fields have already passed the privacy filter.
#[derive(Debug)]
struct SpanStart {
    id: u64,
    ts: u64,
    thread: u64,
    name: &'static str,
    category: &'static str,
    parent: Option<u64>,
    fields: FieldMap,
}

impl SpanStart {
    fn write_json(&self, out: &mut impl Write) -> io::Result<()> {
        // Preserve sorted JSON keys and delegate string/field escaping to serde.
        out.write_all(b"{\"category\":")?;
        serde_json::to_writer(&mut *out, self.category).map_err(io::Error::other)?;
        out.write_all(b",\"fields\":")?;
        serde_json::to_writer(&mut *out, &self.fields).map_err(io::Error::other)?;
        write!(out, ",\"id\":{},\"name\":", self.id)?;
        serde_json::to_writer(&mut *out, self.name).map_err(io::Error::other)?;
        out.write_all(b",\"parent\":")?;
        serde_json::to_writer(&mut *out, &self.parent).map_err(io::Error::other)?;
        write!(out, ",\"thread\":{},\"ts\":{},\"type\":\"start\"}}", self.thread, self.ts)
    }
}

/// Privacy-filtered event/update fields; outer JSON keys need no producer allocations.
#[derive(Debug)]
struct FieldRecord {
    kind: FieldRecordKind,
    id: u64,
    ts: u64,
    thread: u64,
    fields: FieldMap,
}

#[derive(Debug, Clone, Copy)]
enum FieldRecordKind {
    Event,
    Fields,
}

impl FieldRecord {
    fn write_json(&self, out: &mut impl Write) -> io::Result<()> {
        out.write_all(b"{\"fields\":")?;
        serde_json::to_writer(&mut *out, &self.fields).map_err(io::Error::other)?;
        let kind = match self.kind {
            FieldRecordKind::Event => "event",
            FieldRecordKind::Fields => "fields",
        };
        write!(
            out,
            ",\"id\":{},\"thread\":{},\"ts\":{},\"type\":\"{kind}\"}}",
            self.id, self.thread, self.ts
        )
    }
}

impl From<Value> for CaptureRecord {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

impl CaptureRecord {
    fn write_json(&self, out: &mut impl Write) -> io::Result<()> {
        let (kind, id, ts, thread) = match self {
            Self::Json(value) => return serde_json::to_writer(out, value).map_err(io::Error::other),
            Self::Start(value) => return value.write_json(out),
            Self::Fields(value) => return value.write_json(out),
            Self::ProcessCpu(value) => return value.write_json(out),
            Self::Enter { id, ts, thread } => ("enter", id, ts, thread),
            Self::Exit { id, ts, thread } => ("exit", id, ts, thread),
            Self::End { id, ts, thread } => ("end", id, ts, thread),
        };
        // Only fixed literals and unsigned integers are interpolated. Match the
        // existing sorted JSON keys without allocating a temporary object.
        write!(out, "{{\"id\":{id},\"thread\":{thread},\"ts\":{ts},\"type\":\"{kind}\"}}")
    }

    fn is_backpressure(&self) -> bool {
        match self {
            Self::Json(value) => value["fields"]["stage"] == "backpressure_start",
            Self::Fields(value) => {
                value.fields.get("stage").and_then(Value::as_str) == Some("backpressure_start")
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy)]
enum StampKind {
    Enter,
    Exit,
    End,
}

struct Writer {
    tx: mpsc::SyncSender<Option<CaptureRecord>>,
    dropped: Arc<AtomicU64>,
    prewarm_failures: Arc<AtomicU64>,
}

/// Drains the capture and records loss status before the process exits.
pub(crate) struct LifecycleGuard {
    process_cpu: Option<process_cpu::Sampler>,
    writer: Arc<Writer>,
    worker: Option<JoinHandle<()>>,
    root_aggregates: Aggregates,
}

type Aggregates = Arc<Mutex<BTreeMap<&'static str, TimingSummary>>>;

#[derive(Default)]
struct TimingSummary {
    count: u64,
    elapsed_ns: u64,
    first: u64,
    last: u64,
}

struct CapturedSpan {
    id: u64,
    aggregates: Aggregates,
    // High-frequency accessors/proofs retain every call's elapsed time, grouped by owner.
    sample: Option<(&'static str, u64)>,
}

fn aggregate_name(name: &str) -> bool {
    matches!(
        name,
        "state.overlay.execution_overlay" |
            "database_provider_ro" |
            "state.overlay.state_trie_overlay" |
            "Storage proof calculation" |
            "Account multiproof calculation"
    )
}

fn flush_aggregates(writer: &Writer, owner: u64, aggregates: &Aggregates) {
    for (name, summary) in std::mem::take(&mut *aggregates.lock().unwrap()) {
        writer.send(json!({"type":"aggregate", "id":owner, "name":name,
            "ts":summary.first, "end":summary.last, "count":summary.count,
            "elapsed_ns":summary.elapsed_ns, "category":if name.contains("proof") {"trie"} else {"state"}}));
    }
}

impl LifecycleLayer {
    pub(crate) fn from_env() -> eyre::Result<Option<(Self, LifecycleGuard)>> {
        let process_value = std::env::var("TEMPO_LIFECYCLE_PROCESS_CPU");
        let process_cpu = process_cpu::configured(
            match &process_value {
                Ok(value) => Some(value.as_str()),
                Err(std::env::VarError::NotPresent) => None,
                Err(_) => eyre::bail!("invalid process CPU capture configuration"),
            },
            cfg!(target_os = "linux"),
        )?;
        let prewarm_cpu = match std::env::var("TEMPO_LIFECYCLE_PREWARM_CPU") {
            Ok(value) if value == "leaf_v1" => true,
            Ok(value) if value == "disabled" => false,
            Err(std::env::VarError::NotPresent) => false,
            _ => eyre::bail!("TEMPO_LIFECYCLE_PREWARM_CPU must be disabled or leaf_v1"),
        };
        let Some(path) = std::env::var_os("RETH_LIFECYCLE_FILE") else {
            eyre::ensure!(!prewarm_cpu, "prewarm CPU capture requires lifecycle capture");
            eyre::ensure!(!process_cpu, "process CPU capture requires lifecycle capture");
            return Ok(None)
        };
        let detail = match std::env::var("TEMPO_LIFECYCLE_DETAIL") {
            Ok(value) => CaptureDetail::parse(Some(&value))?,
            Err(std::env::VarError::NotPresent) => CaptureDetail::Full,
            Err(error) => return Err(error.into()),
        };
        let key_path = std::env::var_os("RETH_LIFECYCLE_KEY_FILE")
            .ok_or_else(|| eyre::eyre!("lifecycle capture requires a key file"))?;
        let key: [u8; 32] = std::fs::read(key_path)?
            .try_into()
            .map_err(|_| eyre::eyre!("lifecycle key must contain exactly 32 bytes"))?;
        let epoch = std::env::var("RETH_LIFECYCLE_EPOCH_NS")?.parse::<u64>()?;
        eyre::ensure!(epoch <= monotonic_ns(), "lifecycle epoch is in the future");
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(path)?;
        Ok(Some(Self::start_observers(file, key, epoch, detail, prewarm_cpu, process_cpu)?))
    }

    pub(crate) const fn detail(&self) -> CaptureDetail {
        self.detail
    }

    #[cfg(test)]
    fn start(
        file: File,
        key: [u8; 32],
        epoch: u64,
        detail: CaptureDetail,
    ) -> eyre::Result<(Self, LifecycleGuard)> {
        Self::start_prewarm(file, key, epoch, detail, false)
    }

    #[cfg(test)]
    fn start_prewarm(
        file: File,
        key: [u8; 32],
        epoch: u64,
        detail: CaptureDetail,
        prewarm_cpu: bool,
    ) -> eyre::Result<(Self, LifecycleGuard)> {
        Self::start_observers(file, key, epoch, detail, prewarm_cpu, false)
    }

    fn start_observers(
        file: File,
        key: [u8; 32],
        epoch: u64,
        detail: CaptureDetail,
        prewarm_cpu: bool,
        process_enabled: bool,
    ) -> eyre::Result<(Self, LifecycleGuard)> {
        eyre::ensure!(
            !process_enabled || cfg!(target_os = "linux"),
            "process CPU capture requires Linux"
        );
        let process_totals = process_enabled.then(|| Arc::new(process_cpu::Totals::default()));
        let footer_totals = process_totals.clone();
        let (tx, rx) = mpsc::sync_channel(QUEUE_CAPACITY);
        let dropped = Arc::new(AtomicU64::new(0));
        let prewarm_failures = Arc::new(AtomicU64::new(0));
        let writer = Arc::new(Writer {
            tx,
            dropped: Arc::clone(&dropped),
            prewarm_failures: Arc::clone(&prewarm_failures),
        });
        let worker = thread::Builder::new().name("lifecycle-writer".into()).spawn(move || {
            let mut out = BufWriter::with_capacity(WRITE_BUFFER_BYTES, file);
            let mut written = 0u64;
            let mut failed = false;
            while let Ok(Some(value)) = rx.recv() {
                if value.write_json(&mut out).is_err() || out.write_all(b"\n").is_err() {
                    failed = true;
                    // Continue draining so shutdown never waits on a full queue after an I/O error.
                    continue
                }
                // Admission must see the header before load; the first stop
                // boundary must be visible even if the stream becomes idle.
                let header = matches!(&value, CaptureRecord::Json(v) if v["type"] == "header");
                if (header || value.is_backpressure()) && out.flush().is_err() {
                    failed = true;
                }
                written += 1;
            }
            let mut footer = json!({"type":"footer", "written":written,
                "dropped":dropped.load(Ordering::Relaxed), "io_error":failed,
                "prewarm_coverage_failures":prewarm_failures.load(Ordering::Relaxed)});
            if let Some(totals) = footer_totals {
                totals.footer(&mut footer);
            }
            let _ = serde_json::to_writer(&mut out, &footer);
            let _ = out.write_all(b"\n");
            let _ = out.flush();
        })?;
        let mut header = json!({"type":"header", "schema":1, "clock":"shared_monotonic_relative_ns", "detail":detail.label(), "prewarm_cpu":if prewarm_cpu { "leaf_v1" } else { "disabled" }, "process_cpu":if process_enabled { "rusage_self_v1" } else { "disabled" }});
        if process_enabled {
            header["process_cpu_period_ns"] = process_cpu::PERIOD_NS.into();
        }
        writer.send(header);
        let root_aggregates = Aggregates::default();
        let mut guard = LifecycleGuard {
            process_cpu: None,
            writer: Arc::clone(&writer),
            worker: Some(worker),
            root_aggregates: Arc::clone(&root_aggregates),
        };
        if let Some(totals) = process_totals {
            guard.process_cpu =
                Some(process_cpu::Sampler::start(Arc::clone(&writer), epoch, totals)?);
        }
        Ok((
            Self {
                detail,
                prewarm_cpu,
                writer,
                key,
                epoch,
                next_span: AtomicU64::new(1),
                backpressure_seen: AtomicBool::new(false),
                root_aggregates,
            },
            guard,
        ))
    }

    fn stamp(&self, kind: &str, id: u64) -> Value {
        json!({"type":kind, "id":id, "ts":monotonic_ns().saturating_sub(self.epoch), "thread":thread_id()})
    }

    const fn fields(&self) -> SafeFields<'_> {
        SafeFields { key: &self.key, values: FieldMap::new() }
    }
}

impl Writer {
    fn send(&self, value: impl Into<CaptureRecord>) {
        if self.tx.try_send(Some(value.into())).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for LifecycleGuard {
    fn drop(&mut self) {
        if let Some(mut sampler) = self.process_cpu.take() {
            sampler.stop();
        }
        flush_aggregates(&self.writer, 0, &self.root_aggregates);
        let _ = self.writer.tx.send(None);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl<S> Layer<S> for LifecycleLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if attrs.metadata().name() == "prewarm.context" && !self.prewarm_cpu {
            return
        }
        let Some(span) = ctx.span(id) else { return };
        if aggregate_name(attrs.metadata().name()) {
            let (owner, aggregates) = span
                .parent()
                .and_then(|p| {
                    p.extensions().get::<CapturedSpan>().map(|s| (s.id, Arc::clone(&s.aggregates)))
                })
                .unwrap_or_else(|| (0, Arc::clone(&self.root_aggregates)));
            span.extensions_mut().insert(CapturedSpan {
                id: owner,
                aggregates,
                sample: Some((attrs.metadata().name(), monotonic_ns().saturating_sub(self.epoch))),
            });
            return
        }
        let capture_id = self.next_span.fetch_add(1, Ordering::Relaxed);
        let parent = span.parent().and_then(|p| p.extensions().get::<CapturedSpan>().map(|s| s.id));
        let mut fields = self.fields();
        attrs.record(&mut fields);
        let value = CaptureRecord::Start(Box::new(SpanStart {
            id: capture_id,
            ts: monotonic_ns().saturating_sub(self.epoch),
            thread: thread_id(),
            name: attrs.metadata().name(),
            category: category(attrs.metadata().target()),
            parent,
            fields: fields.values,
        }));
        span.extensions_mut().insert(CapturedSpan {
            id: capture_id,
            aggregates: Aggregates::default(),
            sample: None,
        });
        self.writer.send(value);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let ext = span.extensions();
        let Some(span) = ext.get::<CapturedSpan>() else { return };
        if span.sample.is_some() {
            return
        }
        let mut fields = self.fields();
        values.record(&mut fields);
        if !fields.values.is_empty() {
            self.writer.send(CaptureRecord::Fields(Box::new(FieldRecord {
                kind: FieldRecordKind::Fields,
                id: span.id,
                ts: monotonic_ns().saturating_sub(self.epoch),
                thread: thread_id(),
                fields: fields.values,
            })));
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        // Free-form log messages are never exported. Lifecycle events use a static span name
        // in `stage`, selected by the producer, and numeric or pseudonymized fields only.
        if event.metadata().target() != "lifecycle" {
            return
        }
        if self.detail == CaptureDetail::Milestones {
            // Inspect only the static stage before timestamps, formatting, JSON or
            // queue allocation. Completion wrappers using Span::current can see
            // a retained ancestor when their own scope is disabled. None of the
            // retained identity scopes has a completion wrapper, so they remain
            // span-lifetime context; only milestones determine phase durations.
            let mut stage = MilestoneStage::default();
            event.record(&mut stage);
            if !stage.keep {
                return
            }
        }
        let parent =
            ctx.event_span(event).and_then(|s| s.extensions().get::<CapturedSpan>().map(|s| s.id));
        let mut fields = self.fields();
        event.record(&mut fields);
        let stage = fields.values.get("stage").and_then(Value::as_str);
        if stage.is_some_and(|stage| stage.starts_with("prewarm_")) {
            if !self.prewarm_cpu {
                return
            }
            if stage == Some("prewarm_coverage_failure") {
                let _ = self.writer.prewarm_failures.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |value| Some(value.saturating_add(1)),
                );
            }
        }
        let value = CaptureRecord::Fields(Box::new(FieldRecord {
            kind: FieldRecordKind::Event,
            id: parent.unwrap_or(0),
            ts: monotonic_ns().saturating_sub(self.epoch),
            thread: thread_id(),
            fields: fields.values,
        }));
        if value.is_backpressure() && !self.backpressure_seen.swap(true, Ordering::Relaxed) {
            // A full queue must not drop the first stop boundary. This can block only
            // after the measured pre-backpressure interval has already ended.
            if self.writer.tx.send(Some(value)).is_err() {
                self.writer.dropped.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            self.writer.send(value);
        }
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if self.detail == CaptureDetail::Full {
            self.span_event(StampKind::Enter, id, ctx);
        }
    }
    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        if self.detail == CaptureDetail::Full {
            self.span_event(StampKind::Exit, id, ctx);
        }
    }
    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            if let Some(captured) = span.extensions().get::<CapturedSpan>() {
                if let Some((name, start)) = captured.sample {
                    let end = monotonic_ns().saturating_sub(self.epoch);
                    let mut summaries = captured.aggregates.lock().unwrap();
                    let summary = summaries.entry(name).or_default();
                    if summary.count == 0 {
                        summary.first = start;
                    }
                    summary.first = summary.first.min(start);
                    summary.last = summary.last.max(end);
                    summary.count += 1;
                    summary.elapsed_ns =
                        summary.elapsed_ns.saturating_add(end.saturating_sub(start));
                    return
                }
                flush_aggregates(&self.writer, captured.id, &captured.aggregates);
            }
        }
        self.span_event(StampKind::End, &id, ctx);
    }
    fn on_follows_from(&self, id: &Id, follows: &Id, ctx: Context<'_, S>) {
        let (Some(span), Some(other)) = (ctx.span(id), ctx.span(follows)) else { return };
        let (a, b) = (span.extensions(), other.extensions());
        if let (Some(a), Some(b)) = (a.get::<CapturedSpan>(), b.get::<CapturedSpan>()) {
            let mut value = self.stamp("link", a.id);
            value["from"] = b.id.into();
            self.writer.send(value);
        }
    }
}

impl LifecycleLayer {
    fn span_event<S>(&self, kind: StampKind, id: &Id, ctx: Context<'_, S>)
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        if let Some(span) = ctx.span(id) {
            if let Some(span) = span.extensions().get::<CapturedSpan>() {
                if span.sample.is_none() {
                    let id = span.id;
                    let ts = monotonic_ns().saturating_sub(self.epoch);
                    let thread = thread_id();
                    let record = match kind {
                        StampKind::Enter => CaptureRecord::Enter { id, ts, thread },
                        StampKind::Exit => CaptureRecord::Exit { id, ts, thread },
                        StampKind::End => CaptureRecord::End { id, ts, thread },
                    };
                    self.writer.send(record);
                }
            }
        }
    }
}

pub(crate) fn capture_metadata(meta: &tracing::Metadata<'_>) -> bool {
    if meta.is_event() {
        return meta.target() == "lifecycle"
    }
    *meta.level() <= tracing::Level::DEBUG && category(meta.target()) != "excluded"
}

fn category(target: &str) -> &'static str {
    if target == "lifecycle" {
        "lifecycle"
    } else if target.starts_with("tempo_consensus") {
        "consensus"
    } else if target.starts_with("tempo_payload") || target.starts_with("payload_builder") {
        "builder"
    } else if target.starts_with("tempo_node::gossip") ||
        target.starts_with("commonware_p2p") ||
        target.starts_with("commonware_stream")
    {
        "network"
    } else if target.starts_with("commonware_consensus") ||
        target.starts_with("commonware_broadcast")
    {
        "consensus"
    } else if target.starts_with("commonware_storage") || target.starts_with("engine::persistence")
    {
        "storage"
    } else if target.starts_with("engine") || target.starts_with("reth_engine") {
        "execution"
    } else if target.starts_with("providers::state") || target.starts_with("reth_storage_overlay") {
        "state"
    } else if target.starts_with("trie") || target.starts_with("reth_trie") {
        "trie"
    } else {
        "excluded"
    }
}

struct SafeFields<'a> {
    key: &'a [u8; 32],
    values: FieldMap,
}

impl SafeFields<'_> {
    fn text(&mut self, name: &'static str, value: &str) {
        let name = canonical_field(name);
        if matches!(
            name,
            "block_hash" |
                "hash" |
                "digest" |
                "proposal" |
                "payload" |
                "parent_hash" |
                "parent_digest" |
                "head_block_hash" |
                "payload_id" |
                "frame_hash"
        ) {
            let hex = value.strip_prefix("0x").unwrap_or(value);
            if ((hex.len() == 64) || (name == "payload_id" && hex.len() == 16)) &&
                hex.bytes().all(|b| b.is_ascii_hexdigit())
            {
                let hash = blake3::keyed_hash(self.key, hex.to_ascii_lowercase().as_bytes());
                self.values.insert(name, hash.to_hex()[..24].to_string().into());
            }
        } else if numeric_field(name) {
            if let Ok(number) = value.parse::<u64>() {
                self.values.insert(name, number.into());
            }
        } else if name == "stage" && STAGES.contains(&value) {
            self.values.insert(name, value.into());
        }
    }
}

// Producer markers form a closed vocabulary: an arbitrary string cannot escape through `stage`.
const STAGES: &[&str] = &[
    "proposal_start",
    "payload_built",
    "proposal_ready",
    "digest_released",
    "body_ready",
    "verify_start",
    "verify_done",
    "replay_start",
    "replay_done",
    "notarized",
    "finalized",
    "vote_sent",
    "notarize_vote_sent",
    "finalize_vote_sent",
    "finalization_received",
    "certified",
    "cancelled",
    "load_start",
    "load_end",
    "decode_done",
    "frame_send",
    "frame_receive",
    "durable",
    "execution_totals",
    "proof_storage_worker_totals",
    "proof_account_worker_totals",
    "prewarm_context_started",
    "prewarm_leaf_started",
    "prewarm_leaf_completed",
    "prewarm_context_completed",
    "prewarm_coverage_failure",
    "backpressure_start",
    "proposal_failed",
    "marshal_enqueued",
    "marshal_dequeued",
    "operation_completed",
    "operation_abandoned",
];

fn canonical_field(name: &str) -> &str {
    match name {
        "proposal.digest" | "block.digest" => "block_hash",
        "block.height" | "proposal.height" => "height",
        "parent.digest" => "parent_digest",
        "parent.hash" => "parent_hash",
        "id" => "payload_id",
        other => other,
    }
}

fn numeric_field(name: &str) -> bool {
    let name = canonical_field(name);
    matches!(
        name,
        "prewarm_role" |
            "prewarm_mode" |
            "prewarm_leaf" |
            "prewarm_cpu_measured" |
            "prewarm_thread_cpu_ns" |
            "prewarm_outcome" |
            "prewarm_dispatched" |
            "prewarm_started" |
            "prewarm_completed" |
            "prewarm_context_outcome" |
            "prewarm_failure" |
            "queued_jobs" |
            "in_flight_proof_batches" |
            "pending_updates" |
            "pending_targets" |
            "result_count" |
            "height" |
            "number" |
            "block_number" |
            "proposal_height" |
            "epoch" |
            "view" |
            "bytes" |
            "transactions" |
            "gas_used" |
            "workers" |
            "channel" |
            "sequence" |
            "execution_ns" |
            "execution_loop_ns" |
            "execution_thread_cpu_ns" |
            "execution_cpu_measured" |
            "worker_run_ns" |
            "worker_thread_cpu_ns" |
            "worker_cpu_measured" |
            "worker_success" |
            "worker_job_counts_measured" |
            "worker_jobs" |
            "worker_account_targets" |
            "worker_storage_targets" |
            "worker_storage_groups" |
            "worker_jobs_storage_only_single_group" |
            "worker_root_requests" |
            "worker_target_max" |
            "worker_jobs_targets_0" |
            "worker_jobs_targets_1" |
            "worker_jobs_targets_2_8" |
            "worker_jobs_targets_9_32" |
            "worker_jobs_targets_33_plus" |
            "worker_job_counts_saturated" |
            "execution_resources_measured" |
            "execution_voluntary_context_switches" |
            "execution_involuntary_context_switches" |
            "execution_minor_page_faults" |
            "execution_major_page_faults" |
            "execution_block_input_operations" |
            "execution_block_output_operations" |
            "receipt_ns" |
            "bookkeeping_ns" |
            "wait_ns" |
            "head_block_height" |
            "block_count" |
            "state_trie_block_count" |
            "first_block_number" |
            "last_block_number" |
            "canonical_height" |
            "persisted_height" |
            "state_trie_height" |
            "backlog"
    )
}

impl Visit for SafeFields<'_> {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if numeric_field(field.name()) {
            self.values.insert(canonical_field(field.name()), value.into());
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        if let Ok(value) = u64::try_from(value) {
            self.record_u64(field, value);
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.text(field.name(), value);
    }
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        // Do not even format unknown fields: arguments/errors can contain credentials or payloads.
        if numeric_field(field.name()) ||
            matches!(
                canonical_field(field.name()),
                "block_hash" |
                    "hash" |
                    "digest" |
                    "proposal" |
                    "payload" |
                    "parent_hash" |
                    "parent_digest" |
                    "head_block_hash" |
                    "payload_id" |
                    "frame_hash"
            )
        {
            self.text(field.name(), &format!("{value:?}"));
        }
    }
}

fn thread_id() -> u64 {
    THREAD.with(|id| {
        if id.get() == 0 {
            id.set(NEXT_THREAD.fetch_add(1, Ordering::Relaxed));
        }
        id.get()
    })
}

fn monotonic_ns() -> u64 {
    #[cfg(unix)]
    {
        let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `time` is a valid writable timespec; CLOCK_MONOTONIC is process-independent.
        let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) };
        assert_eq!(result, 0, "monotonic clock unavailable");
        (time.tv_sec as u64).saturating_mul(1_000_000_000).saturating_add(time.tv_nsec as u64)
    }
    #[cfg(not(unix))]
    {
        // The benchmark collector is intentionally limited to hosts with a shared monotonic clock.
        panic!("lifecycle capture requires a Unix monotonic clock")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::prelude::*;

    // Different detail modes install different callsite interests. Keep these
    // temporary registries isolated from concurrent interest-cache rebuilds.
    static CAPTURE_TEST: Mutex<()> = Mutex::new(());

    fn as_value(record: CaptureRecord) -> Value {
        let mut bytes = Vec::new();
        record.write_json(&mut bytes).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn static_field_keys_match_owned_sorted_maps_across_tree_nodes() {
        let keys = [
            "worker_jobs_targets_33_plus",
            "stage",
            "backlog",
            "height",
            "transactions",
            "queued_jobs",
            "bytes",
            "gas_used",
            "view",
            "epoch",
            "workers",
            "channel",
            "sequence",
            "execution_ns",
            "wait_ns",
            "receipt_ns",
            "number",
            "block_count",
        ];
        for length in [0, 1, 2, 11, 12, keys.len()] {
            let mut fields = FieldMap::new();
            let mut owned = serde_json::Map::new();
            for (index, key) in keys[..length].iter().copied().enumerate() {
                let value = if key == "stage" {
                    Value::String("backpressure_start".into())
                } else {
                    Value::from(u64::MAX - index as u64)
                };
                fields.insert(key, value.clone());
                owned.insert(key.to_owned(), value);
            }
            // Repeated metadata fields overwrite the value, never duplicate the key.
            if length > 0 {
                fields.insert(keys[0], 0u64.into());
                owned.insert(keys[0].to_owned(), 0u64.into());
            }
            assert_eq!(serde_json::to_vec(&fields).unwrap(), serde_json::to_vec(&owned).unwrap());
            for (kind, name) in
                [(FieldRecordKind::Event, "event"), (FieldRecordKind::Fields, "fields")]
            {
                let expected = json!({"fields":owned,"id":5,"thread":6,"ts":7,"type":name});
                let record = CaptureRecord::Fields(Box::new(FieldRecord {
                    kind,
                    id: 5,
                    thread: 6,
                    ts: 7,
                    fields: fields.clone(),
                }));
                let mut bytes = Vec::new();
                record.write_json(&mut bytes).unwrap();
                assert_eq!(bytes, serde_json::to_vec(&expected).unwrap());
                assert_eq!(record.is_backpressure(), length > 1);
            }
        }
        assert_eq!(
            std::mem::size_of::<FieldMap>(),
            std::mem::size_of::<serde_json::Map<String, Value>>()
        );
    }

    #[test]
    fn static_field_aliases_preserve_privacy_and_owned_values() {
        let fields = {
            let key = [9; 32];
            let mut visitor = SafeFields { key: &key, values: FieldMap::new() };
            let private =
                "0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_owned();
            visitor.text("proposal.digest", &private);
            visitor.text("block.digest", &private.to_lowercase());
            visitor.text("proposal.height", "1");
            visitor.text("block.height", "2");
            visitor.text("height", "3");
            visitor.text("stage", "backpressure_start");
            visitor.text("stage", "DO_NOT_EXPORT");
            visitor.text("worker_jobs", "-1");
            visitor.text("worker_target_max", "18446744073709551616");
            visitor.text("private_key", "DO_NOT_EXPORT");
            visitor.values
        };
        let expected_hash = blake3::keyed_hash(&[9; 32], "a".repeat(64).as_bytes());
        let expected = json!({"block_hash":&expected_hash.to_hex()[..24], "height":3,
            "stage":"backpressure_start"});
        assert_eq!(serde_json::to_vec(&fields).unwrap(), serde_json::to_vec(&expected).unwrap());
        assert_eq!(fields.len(), 3);
    }

    #[test]
    fn typed_start_preserves_json_escaping_and_optional_parent() {
        for id in [0, 1, u64::MAX] {
            for parent in [None, Some(0), Some(u64::MAX)] {
                for name in ["storage_worker", "quoted\"\\\n\t\u{0000}λ"] {
                    let fields = FieldMap::from_iter([
                        ("block_hash", Value::String("pseudonymous".into())),
                        ("transactions", id.into()),
                    ]);
                    let expected = json!({"type":"start", "id":id, "ts":id,
                        "thread":id, "name":name, "category":"trie", "parent":parent,
                        "fields":fields});
                    let record = CaptureRecord::Start(Box::new(SpanStart {
                        id,
                        ts: id,
                        thread: id,
                        name,
                        category: "trie",
                        parent,
                        fields,
                    }));
                    let mut bytes = Vec::new();
                    record.write_json(&mut bytes).unwrap();
                    assert_eq!(bytes, serde_json::to_vec(&expected).unwrap());
                    assert_eq!(as_value(record), expected);
                }
            }
        }
    }

    #[test]
    fn typed_start_propagates_short_writer_failure() {
        let record = CaptureRecord::Start(Box::new(SpanStart {
            id: 1,
            ts: 2,
            thread: 3,
            name: "storage_worker",
            category: "trie",
            parent: None,
            fields: FieldMap::new(),
        }));
        // Fail both in a literal and in a serde-escaped value, never accept truncation.
        for capacity in [0, 14, 22, 70] {
            let mut buffer = vec![0; capacity];
            assert!(record.write_json(&mut io::Cursor::new(buffer.as_mut_slice())).is_err());
        }
    }

    #[test]
    fn typed_fields_preserve_json_and_flush_boundary() {
        for id in [0, 1, u64::MAX] {
            for (kind, name) in
                [(FieldRecordKind::Event, "event"), (FieldRecordKind::Fields, "fields")]
            {
                for stage in [None, Some("backpressure_start"), Some("quoted\"\\\n\t\u{0000}λ")] {
                    let mut fields = FieldMap::from_iter([("transactions", id.into())]);
                    if let Some(stage) = stage {
                        fields.insert("stage", stage.into());
                    }
                    let expected = json!({"type":name,"id":id,"ts":id,"thread":id,"fields":fields});
                    let record = CaptureRecord::Fields(Box::new(FieldRecord {
                        kind,
                        id,
                        ts: id,
                        thread: id,
                        fields,
                    }));
                    let mut bytes = Vec::new();
                    record.write_json(&mut bytes).unwrap();
                    assert_eq!(bytes, serde_json::to_vec(&expected).unwrap());
                    assert_eq!(record.is_backpressure(), stage == Some("backpressure_start"));
                    for capacity in [0, 5, 13, bytes.len() - 1] {
                        let mut buffer = vec![0; capacity];
                        assert!(record
                            .write_json(&mut io::Cursor::new(buffer.as_mut_slice()))
                            .is_err());
                    }
                    assert_eq!(as_value(record), expected);
                }
            }
        }
    }

    #[test]
    fn typed_stamps_match_json_bytes_and_queue_size() {
        for value in [0, 1, u64::MAX] {
            for (kind, record) in [
                ("enter", CaptureRecord::Enter { id: value, ts: value, thread: value }),
                ("exit", CaptureRecord::Exit { id: value, ts: value, thread: value }),
                ("end", CaptureRecord::End { id: value, ts: value, thread: value }),
            ] {
                let expected = json!({"type":kind,"id":value,"ts":value,"thread":value});
                let mut bytes = Vec::new();
                record.write_json(&mut bytes).unwrap();
                assert_eq!(bytes, serde_json::to_vec(&expected).unwrap());
                assert_eq!(as_value(record), expected);
            }
        }
        // Bound any inline enum overhead independently of the queue capacity.
        assert!(
            std::mem::size_of::<Option<CaptureRecord>>() <=
                std::mem::size_of::<Option<Value>>() + std::mem::size_of::<u64>()
        );
        eprintln!(
            "queue element bytes: JSON={}, typed={}",
            std::mem::size_of::<Option<Value>>(),
            std::mem::size_of::<Option<CaptureRecord>>()
        );
    }

    #[test]
    fn typed_stamps_keep_fifo_and_count_queue_failures() {
        let (tx, rx) = mpsc::sync_channel(2);
        let dropped = Arc::new(AtomicU64::new(0));
        let writer = Writer {
            tx,
            dropped: Arc::clone(&dropped),
            prewarm_failures: Arc::new(AtomicU64::new(0)),
        };
        writer.send(CaptureRecord::Enter { id: 9, ts: 10, thread: 11 });
        writer.send(json!({"type":"event","id":9,"ts":12}));
        writer.send(CaptureRecord::Exit { id: 9, ts: 13, thread: 11 });
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(as_value(rx.recv().unwrap().unwrap())["ts"], 10);
        assert_eq!(as_value(rx.recv().unwrap().unwrap())["ts"], 12);
        drop(rx);
        writer.send(CaptureRecord::End { id: 9, ts: 14, thread: 11 });
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn typed_stamps_preserve_active_intervals_and_retained_lifetime() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        let path = std::env::temp_dir().join(format!("lifecycle-active-{}.jsonl", monotonic_ns()));
        let (layer, guard) = LifecycleLayer::start(
            File::create(&path).unwrap(),
            [7; 32],
            monotonic_ns(),
            CaptureDetail::Full,
        )
        .unwrap();
        let subscriber = tracing_subscriber::registry()
            .with(layer.with_filter(tracing_subscriber::filter::filter_fn(capture_metadata)));
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(target: "lifecycle", "retained_operation");
            let retained = span.clone();
            span.in_scope(|| tracing::info!(target: "lifecycle", stage="operation_completed"));
            span.in_scope(|| {});
            drop(span);
            tracing::info!(target: "lifecycle", stage="load_end");
            drop(retained);
        });
        drop(guard);
        let data = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let rows: Vec<Value> = data.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        let kinds: Vec<_> = rows.iter().map(|r| r["type"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            [
                "header", "start", "enter", "event", "exit", "enter", "exit", "event", "end",
                "footer"
            ]
        );
        let id = rows[1]["id"].clone();
        for index in [2, 3, 4, 5, 6, 8] {
            assert_eq!(rows[index]["id"], id);
            assert_eq!(rows[index]["thread"], rows[1]["thread"]);
        }
        for pair in rows[1..9].windows(2) {
            assert!(pair[0]["ts"].as_u64().unwrap() <= pair[1]["ts"].as_u64().unwrap());
        }
        assert_eq!(rows[3]["fields"]["stage"], "operation_completed");
        assert_eq!(rows[9]["written"], 9);
        assert_eq!(rows[9]["dropped"], 0);
        assert_eq!(rows[9]["io_error"], false);
    }

    #[test]
    fn loop_resource_counts_are_numeric_optional_in_both_details() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for detail in [CaptureDetail::Full, CaptureDetail::Milestones] {
            let path =
                std::env::temp_dir().join(format!("lifecycle-resources-{}.jsonl", monotonic_ns()));
            let (layer, guard) = LifecycleLayer::start(
                File::create(&path).unwrap(),
                [7; 32],
                monotonic_ns(),
                detail,
            )
            .unwrap();
            let subscriber = tracing_subscriber::registry().with(layer.with_filter(
                tracing_subscriber::filter::filter_fn(move |meta| detail.capture_metadata(meta)),
            ));
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(target: "lifecycle", stage="execution_totals",
                    execution_resources_measured=1u64,
                    execution_voluntary_context_switches=Some(0u64),
                    execution_involuntary_context_switches=Some(1u64),
                    execution_minor_page_faults=Some(2u64),
                    execution_major_page_faults=Some(3u64),
                    execution_block_input_operations=Some(4u64),
                    execution_block_output_operations=Some(5u64));
                tracing::info!(target: "lifecycle", stage="execution_totals",
                    execution_resources_measured=0u64,
                    execution_voluntary_context_switches=None::<u64>,
                    execution_involuntary_context_switches=None::<u64>,
                    execution_minor_page_faults=None::<u64>,
                    execution_major_page_faults=None::<u64>,
                    execution_block_input_operations=None::<u64>,
                    execution_block_output_operations=None::<u64>);
                tracing::info!(target: "lifecycle", stage="execution_totals",
                    execution_minor_page_faults="must-not-export-private-text", worker_jobs_storage_only_single_group="must-not-export-private-text");
            });
            drop(guard);
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            let rows: Vec<Value> = text.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
            let measured =
                rows.iter().find(|r| r["fields"]["execution_resources_measured"] == 1).unwrap();
            let unavailable =
                rows.iter().find(|r| r["fields"]["execution_resources_measured"] == 0).unwrap();
            for (value, name) in [
                "execution_voluntary_context_switches",
                "execution_involuntary_context_switches",
                "execution_minor_page_faults",
                "execution_major_page_faults",
                "execution_block_input_operations",
                "execution_block_output_operations",
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(measured["fields"][name], value as u64);
                assert!(unavailable["fields"].get(name).is_none());
            }
            assert!(!text.contains("must-not-export-private-text"));
            assert_eq!(rows.last().unwrap()["dropped"], 0);
        }
    }

    #[test]
    fn detail_mode_rejects_unknown_values() {
        assert_eq!(CaptureDetail::parse(None).unwrap(), CaptureDetail::Full);
        assert_eq!(CaptureDetail::parse(Some("full")).unwrap(), CaptureDetail::Full);
        assert_eq!(CaptureDetail::parse(Some("milestones")).unwrap(), CaptureDetail::Milestones);
        assert!(CaptureDetail::parse(Some("off")).is_err());
        assert!(CaptureDetail::parse(Some("")).is_err());
    }

    #[test]
    fn proof_worker_totals_preserve_explicit_parent_and_optional_cpu() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        let path = std::env::temp_dir().join(format!("worker-cpu-{}.jsonl", monotonic_ns()));
        let (layer, guard) = LifecycleLayer::start(
            File::create(&path).unwrap(),
            [7; 32],
            monotonic_ns(),
            CaptureDetail::Full,
        )
        .unwrap();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink))
            .with(layer.with_filter(tracing_subscriber::filter::filter_fn(|meta| {
                CaptureDetail::Full.capture_metadata(meta)
            })));
        tracing::subscriber::with_default(subscriber, || {
            let parent =
                tracing::debug_span!(target: "engine::tree::payload_validator", "execute_block");
            let dispatch = tracing::dispatcher::get_default(Clone::clone);
            std::thread::spawn(move || {
                tracing::dispatcher::with_default(&dispatch, || {
                    let worker = tracing::debug_span!(target: "trie::proof_task", parent: &parent, "storage_worker");
                    // The explicit event parent must work without a thread-local entered span.
                    assert!(tracing::Span::current().is_none());
                    tracing::info!(target: "lifecycle", parent: &worker, stage="proof_storage_worker_totals", worker_run_ns=50u64, worker_thread_cpu_ns=Some(0u64), worker_cpu_measured=1u64, worker_success=1u64, worker_job_counts_measured=1u64, worker_jobs=Some(1u64), worker_storage_targets=Some(0u64), worker_account_targets=None::<u64>, worker_storage_groups=None::<u64>, worker_root_requests=Some(1u64), worker_target_max=Some(0u64), worker_jobs_targets_0=Some(1u64), worker_jobs_targets_1=Some(0u64), worker_jobs_targets_2_8=Some(0u64), worker_jobs_targets_9_32=Some(0u64), worker_jobs_targets_33_plus=Some(0u64), worker_job_counts_saturated=Some(0u64), native_tid=77777u64);
                    tracing::info!(target: "lifecycle", parent: &worker, stage="proof_account_worker_totals", worker_run_ns=60u64, worker_thread_cpu_ns=None::<u64>, worker_cpu_measured=0u64, worker_success=0u64, worker_job_counts_measured=1u64, worker_jobs=Some(0u64), worker_account_targets=Some(0u64), worker_storage_groups=Some(0u64), worker_jobs_storage_only_single_group=Some(0u64), worker_storage_targets=None::<u64>, worker_root_requests=None::<u64>);
                });
            }).join().unwrap();
        });
        drop(guard);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(!text.contains("native_tid"));
        let rows: Vec<Value> = text.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        let parent = rows.iter().find(|r| r["name"] == "execute_block").unwrap();
        let worker = rows.iter().find(|r| r["name"] == "storage_worker").unwrap();
        assert_eq!(worker["parent"], parent["id"]);
        let storage =
            rows.iter().find(|r| r["fields"]["stage"] == "proof_storage_worker_totals").unwrap();
        assert_eq!(storage["id"], worker["id"]);
        assert_eq!(storage["fields"]["worker_run_ns"], 50);
        assert_eq!(storage["fields"]["worker_thread_cpu_ns"], 0);
        assert_eq!(storage["fields"]["worker_cpu_measured"], 1);
        assert_eq!(storage["fields"]["worker_success"], 1);
        assert_eq!(storage["fields"]["worker_job_counts_measured"], 1);
        assert_eq!(storage["fields"]["worker_jobs"], 1);
        assert_eq!(storage["fields"]["worker_root_requests"], 1);
        assert_eq!(storage["fields"]["worker_jobs_targets_0"], 1);
        for field in [
            "worker_storage_targets",
            "worker_target_max",
            "worker_jobs_targets_1",
            "worker_jobs_targets_2_8",
            "worker_jobs_targets_9_32",
            "worker_jobs_targets_33_plus",
            "worker_job_counts_saturated",
        ] {
            assert_eq!(storage["fields"][field], 0, "{field}");
        }
        assert!(storage["fields"].get("worker_account_targets").is_none());
        assert!(storage["fields"].get("worker_storage_groups").is_none());
        assert!(storage["fields"].get("worker_jobs_storage_only_single_group").is_none());
        let account =
            rows.iter().find(|r| r["fields"]["stage"] == "proof_account_worker_totals").unwrap();
        assert_eq!(account["id"], worker["id"]);
        assert_eq!(account["fields"]["worker_run_ns"], 60);
        assert_eq!(account["fields"]["worker_cpu_measured"], 0);
        assert_eq!(account["fields"]["worker_success"], 0);
        assert_eq!(account["fields"]["worker_job_counts_measured"], 1);
        assert_eq!(account["fields"]["worker_account_targets"], 0);
        assert_eq!(account["fields"]["worker_storage_groups"], 0);
        assert_eq!(account["fields"]["worker_jobs_storage_only_single_group"], 0);
        assert!(account["fields"].get("worker_storage_targets").is_none());
        assert!(account["fields"].get("worker_root_requests").is_none());
        assert!(account["fields"].get("worker_thread_cpu_ns").is_none());
        assert_eq!(rows.last().unwrap()["dropped"], 0);
    }

    #[test]
    fn milestone_accounting_uses_event_enablement() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for detail in [CaptureDetail::Full, CaptureDetail::Milestones] {
            let subscriber = tracing_subscriber::registry().with(
                tracing_subscriber::fmt::layer().with_writer(std::io::sink).with_filter(
                    tracing_subscriber::filter::filter_fn(move |meta| {
                        detail.capture_metadata(meta)
                    }),
                ),
            );
            tracing::subscriber::with_default(subscriber, || {
                assert!(tracing::event_enabled!(target: "lifecycle", tracing::Level::INFO));
                if detail == CaptureDetail::Milestones {
                    assert!(!tracing::enabled!(target: "lifecycle", tracing::Level::INFO));
                    assert!(tracing::debug_span!(target: "lifecycle", "proof.storage.work")
                        .is_disabled());
                }
            });
        }
    }

    #[test]
    fn milestones_keep_worker_parents_through_other_subscriber_scopes() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        let path = std::env::temp_dir().join(format!("coarse-worker-cpu-{}.jsonl", monotonic_ns()));
        let detail = CaptureDetail::Milestones;
        let (layer, guard) =
            LifecycleLayer::start(File::create(&path).unwrap(), [7; 32], monotonic_ns(), detail)
                .unwrap();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink))
            .with(layer.with_filter(tracing_subscriber::filter::filter_fn(move |meta| {
                detail.capture_metadata(meta)
            })));
        tracing::subscriber::with_default(subscriber, || {
            let execution = tracing::debug_span!(target: "engine::tree::payload_validator", "execute_block", block_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            execution.in_scope(|| {
                let filtered = tracing::debug_span!(target: "trie::proof_task", "ProofWorkerHandle::new");
                // The formatting subscriber enables this span, but lifecycle excludes it.
                assert!(!filtered.is_disabled());
                filtered.in_scope(|| {
                    let parent = tracing::Span::current();
                    assert_eq!(parent.id(), filtered.id());
                    let dispatch = tracing::dispatcher::get_default(Clone::clone);
                    std::thread::spawn(move || tracing::dispatcher::with_default(&dispatch, || {
                        let storage = tracing::debug_span!(target: "trie::proof_task", parent: &parent, "storage_worker");
                        let account = tracing::debug_span!(target: "trie::proof_task", parent: &parent, "account_worker");
                        // Even an unrelated current scope cannot replace the explicit parent.
                        let unrelated = tracing::debug_span!(target: "engine::tree::payload_validator", "execute_block_bal");
                        unrelated.in_scope(|| {
                            tracing::debug_span!(target: "lifecycle", "proof.storage.work").in_scope(|| {
                                tracing::info!(target: "lifecycle", parent: &storage, stage="proof_storage_worker_totals", worker_run_ns=50u64, worker_thread_cpu_ns=Some(0u64), worker_cpu_measured=1u64, worker_success=1u64);
                                tracing::info!(target: "lifecycle", parent: &account, stage="proof_account_worker_totals", worker_run_ns=60u64, worker_thread_cpu_ns=None::<u64>, worker_cpu_measured=0u64, worker_success=0u64);
                                tracing::info!(target: "lifecycle", stage="operation_completed");
                            });
                        });
                    })).join().unwrap();
                });
            });
        });
        drop(guard);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let rows: Vec<Value> =
            text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        let execution = rows.iter().find(|r| r["name"] == "execute_block").unwrap();
        assert_eq!(execution["fields"]["block_hash"].as_str().unwrap().len(), 24);
        for (name, stage) in [
            ("storage_worker", "proof_storage_worker_totals"),
            ("account_worker", "proof_account_worker_totals"),
        ] {
            let worker = rows.iter().find(|r| r["name"] == name).unwrap();
            assert_eq!(worker["parent"], execution["id"]);
            let totals = rows.iter().find(|r| r["fields"]["stage"] == stage).unwrap();
            assert_eq!(totals["id"], worker["id"]);
            if name == "storage_worker" {
                assert_eq!(totals["fields"]["worker_thread_cpu_ns"], 0);
                assert_eq!(totals["fields"]["worker_cpu_measured"], 1);
            } else {
                assert!(totals["fields"].get("worker_thread_cpu_ns").is_none());
                assert_eq!(totals["fields"]["worker_cpu_measured"], 0);
            }
        }
        assert_eq!(rows.iter().filter(|r| r["type"] == "start").count(), 4);
        assert_eq!(rows.iter().filter(|r| r["type"] == "end").count(), 4);
        assert!(!rows
            .iter()
            .any(|r| matches!(r["type"].as_str(), Some("enter" | "exit" | "aggregate"))));
        for omitted in [
            "ProofWorkerHandle::new",
            "proof.storage.work",
            "operation_completed",
            "aaaaaaaaaaaaaaaa",
        ] {
            assert!(!text.contains(omitted), "unexpected detailed or private value: {omitted}");
        }
        assert_eq!(rows.last().unwrap()["dropped"], 0);
    }

    #[test]
    fn milestones_keep_workers_spawned_before_execution() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for other_subscriber in [false, true] {
            let path =
                std::env::temp_dir().join(format!("worker-sibling-{}.jsonl", monotonic_ns()));
            let detail = CaptureDetail::Milestones;
            let (layer, guard) = LifecycleLayer::start(
                File::create(&path).unwrap(),
                [7; 32],
                monotonic_ns(),
                detail,
            )
            .unwrap();
            let formatting = other_subscriber
                .then(|| tracing_subscriber::fmt::layer().with_writer(std::io::sink));
            let subscriber = tracing_subscriber::registry().with(formatting).with(
                layer.with_filter(tracing_subscriber::filter::filter_fn(move |meta| {
                    detail.capture_metadata(meta)
                })),
            );
            tracing::subscriber::with_default(subscriber, || {
                // Real receiver topology: workers are created before execute_block,
                // under validate_block_with_state through two filtered setup scopes.
                let validation = tracing::debug_span!(target: "engine::tree::payload_validator", "validate_block_with_state", block_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", type_name="PRIVATE_PAYLOAD_KIND");
                validation.in_scope(|| {
                    let setup = tracing::debug_span!(target: "engine::tree::payload_processor", "spawn_state_root");
                    setup.in_scope(|| {
                        let handle = tracing::debug_span!(target: "trie::proof_task", "ProofWorkerHandle::new");
                        handle.in_scope(|| {
                            let parent = tracing::Span::current();
                            let dispatch = tracing::dispatcher::get_default(Clone::clone);
                            std::thread::spawn(move || tracing::dispatcher::with_default(&dispatch, || {
                                for (name, stage) in [("storage", "proof_storage_worker_totals"), ("account", "proof_account_worker_totals")] {
                                    let worker = if name == "storage" {
                                        tracing::debug_span!(target: "trie::proof_task", parent: &parent, "storage_worker")
                                    } else {
                                        tracing::debug_span!(target: "trie::proof_task", parent: &parent, "account_worker")
                                    };
                                    tracing::info!(target: "lifecycle", parent: &worker, stage, worker_run_ns=50u64, worker_thread_cpu_ns=Some(1u64), worker_cpu_measured=1u64, worker_success=1u64, worker_job_counts_measured=1u64, worker_jobs=1u64, worker_job_counts_saturated=0u64, worker_jobs_storage_only_single_group=(name == "account").then_some(1u64));
                                }
                            })).join().unwrap();
                        });
                    });
                    // Execution is a later sibling, not the worker's ancestor.
                    tracing::debug_span!(target: "engine::tree::payload_validator", "execute_block").in_scope(|| {
                        tracing::info!(target: "lifecycle", stage="replay_start");
                    });
                });
            });
            drop(guard);
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            let rows: Vec<Value> =
                text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            let validation = rows
                .iter()
                .find(|r| r["name"] == "validate_block_with_state")
                .expect("pre-execution validation identity must survive milestone filtering");
            assert_eq!(validation["fields"]["block_hash"].as_str().unwrap().len(), 24);
            let execution = rows.iter().find(|r| r["name"] == "execute_block").unwrap();
            assert_eq!(execution["parent"], validation["id"]);
            for (name, stage) in [
                ("storage_worker", "proof_storage_worker_totals"),
                ("account_worker", "proof_account_worker_totals"),
            ] {
                let worker = rows.iter().find(|r| r["name"] == name).unwrap();
                assert_eq!(worker["parent"], validation["id"]);
                assert!(worker["id"].as_u64().unwrap() < execution["id"].as_u64().unwrap());
                let total = rows.iter().find(|r| r["fields"]["stage"] == stage).unwrap();
                assert_eq!(total["id"], worker["id"]);
                assert_eq!(total["fields"]["worker_job_counts_measured"], 1);
                assert_eq!(total["fields"]["worker_jobs"], 1);
                assert_eq!(total["fields"]["worker_job_counts_saturated"], 0);
                if name == "account_worker" {
                    assert_eq!(total["fields"]["worker_jobs_storage_only_single_group"], 1);
                } else {
                    assert!(total["fields"].get("worker_jobs_storage_only_single_group").is_none());
                }
            }
            assert_eq!(rows.iter().filter(|r| r["type"] == "start").count(), 4);
            assert!(!rows
                .iter()
                .any(|r| matches!(r["type"].as_str(), Some("enter" | "exit" | "aggregate"))));
            for omitted in [
                "spawn_state_root",
                "ProofWorkerHandle::new",
                "PRIVATE_PAYLOAD_KIND",
                "aaaaaaaaaaaaaaaa",
            ] {
                assert!(!text.contains(omitted));
            }
            assert_eq!(rows.last().unwrap()["dropped"], 0);
        }
    }

    #[test]
    fn milestones_preserve_identity_without_false_ancestor_completion() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        let path =
            std::env::temp_dir().join(format!("lifecycle-milestones-{}.jsonl", monotonic_ns()));
        let detail = CaptureDetail::Milestones;
        let (layer, guard) =
            LifecycleLayer::start(File::create(&path).unwrap(), [7; 32], monotonic_ns(), detail)
                .unwrap();
        let subscriber = tracing_subscriber::registry().with(layer.with_filter(
            tracing_subscriber::filter::filter_fn(move |meta| detail.capture_metadata(meta)),
        ));
        tracing::subscriber::with_default(subscriber, || {
            let proposal = tracing::info_span!(target: "tempo_consensus::consensus::application::actor", "handle_propose", epoch=1u64, view=2u64, block_hash=tracing::field::Empty);
            proposal.in_scope(|| {
                tracing::info!(target: "lifecycle", stage="proposal_start");
                let child = tracing::debug_span!(target: "lifecycle", "proposal.persist");
                assert!(child.is_disabled());
                child.in_scope(|| {
                    // Mirrors a future wrapper that captures Span::current after
                    // entering a disabled child. Its completion is not the parent's.
                    let captured_parent = tracing::Span::current();
                    assert_eq!(captured_parent.id(), proposal.id());
                    tracing::info!(target: "lifecycle", parent: &captured_parent, stage="operation_completed");
                    tracing::info!(target: "lifecycle", parent: &captured_parent, stage="operation_abandoned");
                    tracing::info!(target: "lifecycle", stage="proposal_ready", block_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
                });
                proposal.record("block_hash", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
                tracing::info!(target: "lifecycle", stage="digest_released", block_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            });
            let execution =
                tracing::debug_span!(target: "engine::tree::payload_validator", "execute_block");
            execution.in_scope(|| {
                tracing::info!(target: "lifecycle", stage="replay_start", block_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
                tracing::debug_span!(target: "engine::tree", "execution").in_scope(|| {
                    tracing::info!(target: "lifecycle", stage="execution_totals", execution_ns=12u64, transactions=3u64, execution_loop_ns=20u64, execution_thread_cpu_ns=Some(15u64), execution_cpu_measured=1u64);
                    tracing::info!(target: "lifecycle", stage="execution_totals", execution_loop_ns=21u64, execution_thread_cpu_ns=None::<u64>, execution_cpu_measured=0u64);
                });
                tracing::info!(target: "lifecycle", stage="replay_done", block_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            });
            tracing::info_span!(target: "tempo_consensus::consensus::application::actor", "handle_propose", epoch=1u64, view=3u64).in_scope(|| {
                tracing::info!(target: "lifecycle", stage="proposal_start");
                tracing::info!(target: "lifecycle", stage="cancelled");
            });
        });
        drop(guard);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let rows: Vec<Value> = text.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        assert_eq!(rows[0]["detail"], "milestones");
        assert_eq!(rows.iter().filter(|r| r["type"] == "start").count(), 3);
        assert!(!rows
            .iter()
            .any(|r| matches!(r["type"].as_str(), Some("enter" | "exit" | "aggregate"))));
        assert!(!text.contains("operation_completed") && !text.contains("operation_abandoned"));
        assert!(!text.contains("aaaaaaaaaaaaaaaa"));
        let proposal = rows.iter().find(|r| r["name"] == "handle_propose").unwrap();
        let ready = rows.iter().find(|r| r["fields"]["stage"] == "proposal_ready").unwrap();
        assert_eq!(ready["id"], proposal["id"]);
        let replay = rows.iter().find(|r| r["fields"]["stage"] == "replay_start").unwrap();
        let totals = rows.iter().find(|r| r["fields"]["stage"] == "execution_totals").unwrap();
        assert_eq!(totals["id"], replay["id"]);
        assert_eq!(replay["fields"]["block_hash"], ready["fields"]["block_hash"]);
        assert_eq!(totals["fields"]["transactions"], 3);
        assert_eq!(totals["fields"]["execution_loop_ns"], 20);
        assert_eq!(totals["fields"]["execution_thread_cpu_ns"], 15);
        assert_eq!(totals["fields"]["execution_cpu_measured"], 1);
        let unmeasured = rows.iter().find(|r| r["fields"]["execution_cpu_measured"] == 0).unwrap();
        assert_eq!(unmeasured["fields"]["execution_loop_ns"], 21);
        assert!(unmeasured["fields"].get("execution_thread_cpu_ns").is_none());
        let cancelled = rows.iter().find(|r| r["fields"]["stage"] == "cancelled").unwrap();
        assert_ne!(cancelled["id"], proposal["id"]);
        assert!(rows.iter().any(|r| r["id"] == cancelled["id"] && r["name"] == "handle_propose"));
        assert_eq!(rows.last().unwrap()["dropped"], 0);
    }

    #[test]
    fn milestones_filter_hot_span_values_and_completion_fields() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        struct MustNotFormat;
        impl fmt::Debug for MustNotFormat {
            fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
                panic!("excluded fields must not be formatted")
            }
        }
        let path = std::env::temp_dir().join(format!("lifecycle-filter-{}.jsonl", monotonic_ns()));
        let detail = CaptureDetail::Milestones;
        let (layer, guard) =
            LifecycleLayer::start(File::create(&path).unwrap(), [7; 32], monotonic_ns(), detail)
                .unwrap();
        let subscriber = tracing_subscriber::registry().with(layer.with_filter(
            tracing_subscriber::filter::filter_fn(move |meta| detail.capture_metadata(meta)),
        ));
        tracing::subscriber::with_default(subscriber, || {
            let evaluations = Cell::new(0);
            for _ in 0..100_000 {
                let scope = tracing::debug_span!(target: "lifecycle", "proof.account.work", queued_jobs={evaluations.set(evaluations.get()+1); 1u64});
                assert!(scope.is_disabled());
                tracing::info!(target: "lifecycle", parent: &scope, stage="operation_completed", block_hash=?MustNotFormat);
            }
            tracing::info!(target: "lifecycle", stage="frame_send", frame_hash=?MustNotFormat);
            assert_eq!(evaluations.get(), 0);
        });
        drop(guard);
        let rows: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        std::fs::remove_file(path).unwrap();
        assert_eq!(rows.len(), 2, "only the header and footer should reach the writer");
        assert_eq!(rows.last().unwrap()["dropped"], 0);
    }

    #[test]
    fn capture_preserves_lifecycle_without_private_fields() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        let path = std::env::temp_dir().join(format!(
            "lifecycle-test-{}-{}.jsonl",
            std::process::id(),
            monotonic_ns()
        ));
        let (layer, guard) = LifecycleLayer::start(
            File::create(&path).unwrap(),
            [7; 32],
            monotonic_ns(),
            CaptureDetail::Full,
        )
        .unwrap();
        let subscriber = tracing_subscriber::registry()
            .with(layer.with_filter(tracing_subscriber::filter::filter_fn(capture_metadata)));
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(target:"lifecycle", "proposal", height=42u64, block_count=7u64, persisted_height=41u64, block_hash="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", secret="DO_NOT_EXPORT", ip="192.0.2.1", error=?"credential");
            let _entered = span.enter();
            tracing::info!(target:"lifecycle", stage="marshal_enqueued", block_hash="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            tracing::info!(target:"lifecycle", stage="DO_NOT_EXPORT", payload="PRIVATE_TRANSACTION", private_key="DO_NOT_EXPORT");
            tracing::info!(target: "lifecycle", stage = "operation_completed", queued_jobs = 3u64, worker_jobs="DO_NOT_EXPORT", worker_target_max=?"DO_NOT_EXPORT", worker_job_counts_saturated=Some(1u64), worker_storage_targets=Some(u64::MAX));
            tracing::info!(target: "lifecycle", stage = "operation_abandoned");
            tracing::info!("DO_NOT_EXPORT");
        });
        drop(guard);
        let data = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        for private in
            ["DO_NOT_EXPORT", "192.0.2.1", "credential", "PRIVATE_TRANSACTION", "aaaaaaaaaaaaaaaa"]
        {
            assert!(!data.contains(private));
        }
        let values: Vec<Value> = data.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        assert!(values.iter().any(|v| v["fields"]["height"] == 42));
        assert!(values.iter().any(|v| v["fields"]["stage"] == "marshal_enqueued"));
        assert!(values
            .iter()
            .any(|v| v["fields"]["block_count"] == 7 && v["fields"]["persisted_height"] == 41));
        assert!(values
            .iter()
            .any(|v| v["fields"]["stage"] == "operation_completed" &&
                v["fields"]["queued_jobs"] == 3));
        assert!(values.iter().any(|v| v["fields"]["worker_job_counts_saturated"] == 1 &&
            v["fields"]["worker_storage_targets"] == u64::MAX));
        assert!(!values.iter().any(|v| v["fields"].get("worker_jobs").is_some() ||
            v["fields"].get("worker_target_max").is_some()));
        assert!(values.iter().any(|v| v["fields"]["stage"] == "operation_abandoned"));
        assert_eq!(values.last().unwrap()["dropped"], 0);
        assert_eq!(
            values.iter().filter(|v| v["type"] == "start").count(),
            values.iter().filter(|v| v["type"] == "end").count()
        );
    }
    #[test]
    fn prewarm_admission_parent_privacy_and_failure_footer() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for detail in [CaptureDetail::Full, CaptureDetail::Milestones] {
            for enabled in [false, true] {
                let path = std::env::temp_dir()
                    .join(format!("lifecycle-prewarm-{}.jsonl", monotonic_ns()));
                let (layer, guard) = LifecycleLayer::start_prewarm(
                    File::create(&path).unwrap(),
                    [9; 32],
                    monotonic_ns(),
                    detail,
                    enabled,
                )
                .unwrap();
                let subscriber = tracing_subscriber::registry().with(layer.with_filter(
                    tracing_subscriber::filter::filter_fn(move |meta| {
                        detail.capture_metadata(meta)
                    }),
                ));
                tracing::subscriber::with_default(subscriber, || {
                    let parent = tracing::debug_span!(target:"engine::tree::payload_validator", "validate_block_with_state", block_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
                    let context = tracing::debug_span!(target:"lifecycle", parent:&parent, "prewarm.context", prewarm_role=1u64, prewarm_mode=1u64, secret="private-value");
                    tracing::info!(target:"lifecycle", parent:&context, stage="prewarm_context_started", prewarm_role=1u64, prewarm_mode=1u64);
                    tracing::info!(target:"lifecycle", parent:&context, stage="prewarm_leaf_started", prewarm_leaf=1u64);
                    tracing::info!(target:"lifecycle", parent:&context, stage="prewarm_leaf_completed", prewarm_leaf=1u64, prewarm_cpu_measured=1u64, prewarm_thread_cpu_ns=0u64, prewarm_outcome=5u64, secret="private-value");
                    tracing::info!(target:"lifecycle", parent:&context, stage="prewarm_coverage_failure", prewarm_failure=1u64);
                    tracing::info!(target:"lifecycle", parent:&context, stage="prewarm_context_completed", prewarm_dispatched=1u64, prewarm_started=1u64, prewarm_completed=1u64, prewarm_context_outcome=0u64);
                });
                drop(guard);
                let text = std::fs::read_to_string(&path).unwrap();
                std::fs::remove_file(path).unwrap();
                let rows: Vec<Value> =
                    text.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
                assert_eq!(rows[0]["prewarm_cpu"], if enabled { "leaf_v1" } else { "disabled" });
                assert_eq!(rows.last().unwrap()["prewarm_coverage_failures"], u64::from(enabled));
                assert!(!text.contains("private-value") && !text.contains("aaaaaaaaaaaaaaaa"));
                let context = rows.iter().find(|r| r["name"] == "prewarm.context");
                assert_eq!(context.is_some(), enabled);
                if let Some(context) = context {
                    let parent =
                        rows.iter().find(|r| r["name"] == "validate_block_with_state").unwrap();
                    assert_eq!(context["parent"], parent["id"]);
                    let events: Vec<_> = rows.iter().filter(|r| r["type"] == "event").collect();
                    assert_eq!(events.len(), 5);
                    assert!(events.iter().all(|e| e["id"] == context["id"]));
                    let end = events
                        .iter()
                        .find(|e| e["fields"]["stage"] == "prewarm_leaf_completed")
                        .unwrap();
                    assert_eq!(end["fields"]["prewarm_thread_cpu_ns"], 0);
                } else {
                    assert!(!rows.iter().any(|r| r["type"] == "event"));
                }
            }
        }
    }

    #[test]
    fn prewarm_identity_uses_retained_parent_on_unrelated_worker_scope() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for detail in [CaptureDetail::Full, CaptureDetail::Milestones] {
            let path =
                std::env::temp_dir().join(format!("prewarm-parent-{}.jsonl", monotonic_ns()));
            let (layer, guard) = LifecycleLayer::start_prewarm(
                File::create(&path).unwrap(),
                [7; 32],
                monotonic_ns(),
                detail,
                true,
            )
            .unwrap();
            let subscriber = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink))
                .with(layer.with_filter(tracing_subscriber::filter::filter_fn(move |meta| {
                    detail.capture_metadata(meta)
                })));
            tracing::subscriber::with_default(subscriber, || {
                let root = tracing::debug_span!(target:"engine::tree::payload_validator","validate_block_with_state");
                let parent=root.in_scope(||tracing::debug_span!(target:"engine::tree::payload_processor::prewarm","prewarm and caching"));
                let dispatch = tracing::dispatcher::get_default(Clone::clone);
                drop(root);
                std::thread::spawn(move ||tracing::dispatcher::with_default(&dispatch,||{
                    let unrelated=tracing::debug_span!(target:"engine::tree::payload_validator","execute_block_bal");
                    unrelated.in_scope(||{
                        let context=tracing::debug_span!(target:"lifecycle",parent:&parent,"prewarm.context",prewarm_role=1u64,prewarm_mode=3u64);
                        tracing::info!(target:"lifecycle",parent:&context,stage="prewarm_context_started",prewarm_role=1u64,prewarm_mode=3u64);
                        tracing::info!(target:"lifecycle",parent:&context,stage="prewarm_context_completed",prewarm_dispatched=0u64,prewarm_started=0u64,prewarm_completed=0u64,prewarm_context_outcome=0u64);
                    });
                })).join().unwrap();
            });
            drop(guard);
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            let rows: Vec<Value> =
                text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            let identity = rows.iter().find(|r| r["name"] == "prewarm.context").unwrap();
            let mut parent = identity["parent"].clone();
            loop {
                let found =
                    rows.iter().find(|r| r["type"] == "start" && r["id"] == parent).unwrap();
                assert_ne!(found["name"], "execute_block_bal");
                if found["name"] == "validate_block_with_state" {
                    break
                }
                parent = found["parent"].clone();
            }
            assert!(rows
                .iter()
                .filter(|r| r["type"] == "event")
                .all(|r| r["id"] == identity["id"]));
            assert_eq!(rows.last().unwrap()["dropped"], 0);
        }
    }

    #[test]
    fn prewarm_header_is_available_before_any_work() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for enabled in [false, true] {
            let path =
                std::env::temp_dir().join(format!("lifecycle-admission-{}.jsonl", monotonic_ns()));
            let (_layer, guard) = LifecycleLayer::start_prewarm(
                File::create(&path).unwrap(),
                [7; 32],
                monotonic_ns(),
                CaptureDetail::Milestones,
                enabled,
            )
            .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                let data = std::fs::read_to_string(&path).unwrap();
                if let Some(header) =
                    data.lines().next().and_then(|line| serde_json::from_str::<Value>(line).ok())
                {
                    assert_eq!(header["type"], "header");
                    assert_eq!(header["prewarm_cpu"], if enabled { "leaf_v1" } else { "disabled" });
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "header stayed buffered");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            drop(guard);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn prewarm_failure_count_survives_queue_loss() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        let (tx, _rx) = mpsc::sync_channel(1);
        tx.send(Some(CaptureRecord::Json(json!({"type":"header"})))).unwrap();
        let dropped = Arc::new(AtomicU64::new(0));
        let failures = Arc::new(AtomicU64::new(0));
        let layer = LifecycleLayer {
            detail: CaptureDetail::Milestones,
            prewarm_cpu: true,
            writer: Arc::new(Writer {
                tx,
                dropped: Arc::clone(&dropped),
                prewarm_failures: Arc::clone(&failures),
            }),
            key: [7; 32],
            epoch: monotonic_ns(),
            next_span: AtomicU64::new(1),
            backpressure_seen: AtomicBool::new(false),
            root_aggregates: Aggregates::default(),
        };
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target:"lifecycle",stage="prewarm_coverage_failure",prewarm_failure=1u64);
        });
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(failures.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn mandatory_boundary_survives_a_full_queue_in_both_modes() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for detail in [CaptureDetail::Full, CaptureDetail::Milestones] {
            let (tx, rx) = mpsc::sync_channel(1);
            tx.send(Some(CaptureRecord::Json(json!({"type":"header"})))).unwrap();
            let dropped = Arc::new(AtomicU64::new(0));
            let layer = LifecycleLayer {
                detail,
                prewarm_cpu: false,
                writer: Arc::new(Writer {
                    tx,
                    dropped: Arc::clone(&dropped),
                    prewarm_failures: Arc::new(AtomicU64::new(0)),
                }),
                key: [7; 32],
                epoch: monotonic_ns(),
                next_span: AtomicU64::new(1),
                backpressure_seen: AtomicBool::new(false),
                root_aggregates: Aggregates::default(),
            };
            let subscriber = tracing_subscriber::registry().with(layer.with_filter(
                tracing_subscriber::filter::filter_fn(move |meta| detail.capture_metadata(meta)),
            ));
            let reader = thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(30));
                assert_eq!(as_value(rx.recv().unwrap().unwrap())["type"], "header");
                rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap().unwrap()
            });
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(target: "lifecycle", stage="backpressure_start", backlog=42u64);
            });
            let boundary = as_value(reader.join().unwrap());
            assert_eq!(boundary["fields"]["stage"], "backpressure_start");
            assert_eq!(boundary["fields"]["backlog"], 42);
            assert_eq!(dropped.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn backpressure_boundary_is_visible_before_shutdown() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        for detail in [CaptureDetail::Full, CaptureDetail::Milestones] {
            let path =
                std::env::temp_dir().join(format!("lifecycle-boundary-{}.jsonl", monotonic_ns()));
            let (layer, guard) = LifecycleLayer::start(
                File::create(&path).unwrap(),
                [1; 32],
                monotonic_ns(),
                detail,
            )
            .unwrap();
            let subscriber = tracing_subscriber::registry().with(layer.with_filter(
                tracing_subscriber::filter::filter_fn(move |meta| detail.capture_metadata(meta)),
            ));
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(target: "lifecycle", stage = "backpressure_start");
            });
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                let data = std::fs::read_to_string(&path).unwrap();
                if data.lines().any(|line| {
                    serde_json::from_str::<Value>(line)
                        .is_ok_and(|v| v["fields"]["stage"] == "backpressure_start")
                }) {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "boundary stayed buffered");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            drop(guard);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn hot_accessors_preserve_counts_and_owner_without_queue_pressure() {
        let _serial = CAPTURE_TEST.lock().unwrap();
        let path =
            std::env::temp_dir().join(format!("lifecycle-aggregate-{}.jsonl", monotonic_ns()));
        let (layer, guard) = LifecycleLayer::start(
            File::create(&path).unwrap(),
            [1; 32],
            monotonic_ns(),
            CaptureDetail::Full,
        )
        .unwrap();
        let subscriber = tracing_subscriber::registry()
            .with(layer.with_filter(tracing_subscriber::filter::filter_fn(capture_metadata)));
        tracing::subscriber::with_default(subscriber, || {
            let _owner = tracing::info_span!(target:"lifecycle", "block_work").entered();
            for _ in 0..100_000 {
                let _read =
                    tracing::debug_span!(target:"lifecycle", "state.overlay.execution_overlay")
                        .entered();
                let _nested =
                    tracing::debug_span!(target:"providers::state", "database_provider_ro")
                        .entered();
            }
        });
        drop(guard);
        let data = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let values: Vec<Value> = data.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        let owner = values.iter().find(|v| v["name"] == "block_work").unwrap()["id"].clone();
        let summaries: Vec<_> = values.iter().filter(|v| v["type"] == "aggregate").collect();
        assert_eq!(summaries.len(), 2);
        for summary in summaries {
            assert_eq!(summary["id"], owner);
            assert_eq!(summary["count"], 100_000);
            assert!(summary["elapsed_ns"].as_u64().unwrap() > 0);
        }
        assert!(values.len() < 20);
        assert_eq!(values.last().unwrap()["dropped"], 0);
    }
}
