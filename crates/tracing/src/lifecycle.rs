//! Opt-in benchmark capture. Only structural metadata and allowlisted fields reach disk.

use serde_json::{json, Map, Value};
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
                matches!(meta.name(), "execute_block" | "execute_block_bal"))
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
    Enter { id: u64, ts: u64, thread: u64 },
    Exit { id: u64, ts: u64, thread: u64 },
    End { id: u64, ts: u64, thread: u64 },
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
            Self::Enter { id, ts, thread } => ("enter", id, ts, thread),
            Self::Exit { id, ts, thread } => ("exit", id, ts, thread),
            Self::End { id, ts, thread } => ("end", id, ts, thread),
        };
        // Only fixed literals and unsigned integers are interpolated. Match the
        // existing sorted JSON keys without allocating a temporary object.
        write!(out, "{{\"id\":{id},\"thread\":{thread},\"ts\":{ts},\"type\":\"{kind}\"}}")
    }

    fn is_backpressure(&self) -> bool {
        matches!(self, Self::Json(value) if value["fields"]["stage"] == "backpressure_start")
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
}

/// Drains the capture and records loss status before the process exits.
pub(crate) struct LifecycleGuard {
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
        let Some(path) = std::env::var_os("RETH_LIFECYCLE_FILE") else { return Ok(None) };
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
        Ok(Some(Self::start(file, key, epoch, detail)?))
    }

    pub(crate) const fn detail(&self) -> CaptureDetail {
        self.detail
    }

    fn start(
        file: File,
        key: [u8; 32],
        epoch: u64,
        detail: CaptureDetail,
    ) -> eyre::Result<(Self, LifecycleGuard)> {
        let (tx, rx) = mpsc::sync_channel(QUEUE_CAPACITY);
        let dropped = Arc::new(AtomicU64::new(0));
        let writer = Arc::new(Writer { tx, dropped: Arc::clone(&dropped) });
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
                // Make the stop boundary visible even when the stream is otherwise idle.
                if value.is_backpressure() && out.flush().is_err() {
                    failed = true;
                }
                written += 1;
            }
            let footer = json!({"type":"footer", "written":written,
                "dropped":dropped.load(Ordering::Relaxed), "io_error":failed});
            let _ = serde_json::to_writer(&mut out, &footer);
            let _ = out.write_all(b"\n");
            let _ = out.flush();
        })?;
        writer.send(json!({"type":"header", "schema":1, "clock":"shared_monotonic_relative_ns", "detail":detail.label()}));
        let root_aggregates = Aggregates::default();
        let guard = LifecycleGuard {
            writer: Arc::clone(&writer),
            worker: Some(worker),
            root_aggregates: Arc::clone(&root_aggregates),
        };
        Ok((
            Self {
                detail,
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

    fn fields(&self) -> SafeFields<'_> {
        SafeFields { key: &self.key, values: Map::new() }
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
        let mut value = self.stamp("start", capture_id);
        value["name"] = attrs.metadata().name().into();
        value["category"] = category(attrs.metadata().target()).into();
        value["parent"] = parent.into();
        value["fields"] = Value::Object(fields.values);
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
            let mut value = self.stamp("fields", span.id);
            value["fields"] = Value::Object(fields.values);
            self.writer.send(value);
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
        let mut value = self.stamp("event", parent.unwrap_or(0));
        value["fields"] = Value::Object(fields.values);
        if value["fields"]["stage"] == "backpressure_start" &&
            !self.backpressure_seen.swap(true, Ordering::Relaxed)
        {
            // A full queue must not drop the first stop boundary. This can block only
            // after the measured pre-backpressure interval has already ended.
            if self.writer.tx.send(Some(value.into())).is_err() {
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
    values: Map<String, Value>,
}

impl SafeFields<'_> {
    fn text(&mut self, name: &str, value: &str) {
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
                self.values.insert(name.into(), hash.to_hex()[..24].to_string().into());
            }
        } else if numeric_field(name) {
            if let Ok(number) = value.parse::<u64>() {
                self.values.insert(name.into(), number.into());
            }
        } else if name == "stage" && STAGES.contains(&value) {
            self.values.insert(name.into(), value.into());
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
            self.values.insert(canonical_field(field.name()).into(), value.into());
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
        let writer = Writer { tx, dropped: Arc::clone(&dropped) };
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
                    tracing::info!(target: "lifecycle", parent: &worker, stage="proof_storage_worker_totals", worker_run_ns=50u64, worker_thread_cpu_ns=Some(0u64), worker_cpu_measured=1u64, worker_success=1u64, native_tid=77777u64);
                    tracing::info!(target: "lifecycle", parent: &worker, stage="proof_account_worker_totals", worker_run_ns=60u64, worker_thread_cpu_ns=None::<u64>, worker_cpu_measured=0u64, worker_success=0u64);
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
        let account =
            rows.iter().find(|r| r["fields"]["stage"] == "proof_account_worker_totals").unwrap();
        assert_eq!(account["id"], worker["id"]);
        assert_eq!(account["fields"]["worker_run_ns"], 60);
        assert_eq!(account["fields"]["worker_cpu_measured"], 0);
        assert_eq!(account["fields"]["worker_success"], 0);
        assert!(account["fields"].get("worker_thread_cpu_ns").is_none());
        assert_eq!(rows.last().unwrap()["dropped"], 0);
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
            tracing::info!(target: "lifecycle", stage = "operation_completed", queued_jobs = 3u64);
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
        assert!(values.iter().any(|v| v["fields"]["stage"] == "operation_abandoned"));
        assert_eq!(values.last().unwrap()["dropped"], 0);
        assert_eq!(
            values.iter().filter(|v| v["type"] == "start").count(),
            values.iter().filter(|v| v["type"] == "end").count()
        );
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
                writer: Arc::new(Writer { tx, dropped: Arc::clone(&dropped) }),
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
