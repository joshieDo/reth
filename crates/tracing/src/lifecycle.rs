//! Opt-in benchmark capture. Only structural metadata and allowlisted fields reach disk.

use serde_json::{json, Map, Value};
use std::{
    cell::Cell,
    collections::BTreeMap,
    fmt,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    sync::{
        atomic::{AtomicU64, Ordering},
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

const QUEUE_CAPACITY: usize = 65_536;
static NEXT_THREAD: AtomicU64 = AtomicU64::new(1);
thread_local! { static THREAD: Cell<u64> = const { Cell::new(0) }; }

/// Captures a source-timestamped, privacy-filtered benchmark stream.
pub(crate) struct LifecycleLayer {
    writer: Arc<Writer>,
    key: [u8; 32],
    epoch: u64,
    next_span: AtomicU64,
    root_aggregates: Aggregates,
}

struct Writer {
    tx: mpsc::SyncSender<Option<Value>>,
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
        Ok(Some(Self::start(file, key, epoch)?))
    }

    fn start(file: File, key: [u8; 32], epoch: u64) -> eyre::Result<(Self, LifecycleGuard)> {
        let (tx, rx) = mpsc::sync_channel(QUEUE_CAPACITY);
        let dropped = Arc::new(AtomicU64::new(0));
        let writer = Arc::new(Writer { tx, dropped: Arc::clone(&dropped) });
        let worker = thread::Builder::new().name("lifecycle-writer".into()).spawn(move || {
            let mut out = BufWriter::new(file);
            let mut written = 0u64;
            let mut failed = false;
            while let Ok(Some(value)) = rx.recv() {
                if serde_json::to_writer(&mut out, &value).is_err() || out.write_all(b"\n").is_err()
                {
                    failed = true;
                    // Continue draining so shutdown never waits on a full queue after an I/O error.
                    continue
                }
                written += 1;
            }
            let footer = json!({"type":"footer", "written":written,
                "dropped":dropped.load(Ordering::Relaxed), "io_error":failed});
            let _ = serde_json::to_writer(&mut out, &footer);
            let _ = out.write_all(b"\n");
            let _ = out.flush();
        })?;
        writer.send(json!({"type":"header", "schema":1, "clock":"shared_monotonic_relative_ns"}));
        let root_aggregates = Aggregates::default();
        let guard = LifecycleGuard {
            writer: Arc::clone(&writer),
            worker: Some(worker),
            root_aggregates: Arc::clone(&root_aggregates),
        };
        Ok((Self { writer, key, epoch, next_span: AtomicU64::new(1), root_aggregates }, guard))
    }

    fn stamp(&self, kind: &str, id: u64) -> Value {
        json!({"type":kind, "id":id, "ts":monotonic_ns().saturating_sub(self.epoch), "thread":thread_id()})
    }

    fn fields(&self) -> SafeFields<'_> {
        SafeFields { key: &self.key, values: Map::new() }
    }
}

impl Writer {
    fn send(&self, value: Value) {
        if self.tx.try_send(Some(value)).is_err() {
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
        let parent =
            ctx.event_span(event).and_then(|s| s.extensions().get::<CapturedSpan>().map(|s| s.id));
        let mut fields = self.fields();
        event.record(&mut fields);
        let mut value = self.stamp("event", parent.unwrap_or(0));
        value["fields"] = Value::Object(fields.values);
        self.writer.send(value);
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        self.span_event("enter", id, ctx);
    }
    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        self.span_event("exit", id, ctx);
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
        self.span_event("end", &id, ctx);
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
    fn span_event<S>(&self, kind: &str, id: &Id, ctx: Context<'_, S>)
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        if let Some(span) = ctx.span(id) {
            if let Some(span) = span.extensions().get::<CapturedSpan>() {
                if span.sample.is_none() {
                    self.writer.send(self.stamp(kind, span.id));
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
            "receipt_ns" |
            "bookkeeping_ns" |
            "wait_ns" |
            "head_block_height"
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

    #[test]
    fn capture_preserves_lifecycle_without_private_fields() {
        let path = std::env::temp_dir().join(format!(
            "lifecycle-test-{}-{}.jsonl",
            std::process::id(),
            monotonic_ns()
        ));
        let (layer, guard) =
            LifecycleLayer::start(File::create(&path).unwrap(), [7; 32], monotonic_ns()).unwrap();
        let subscriber = tracing_subscriber::registry()
            .with(layer.with_filter(tracing_subscriber::filter::filter_fn(capture_metadata)));
        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!(target:"lifecycle", "proposal", height=42u64, block_hash="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", secret="DO_NOT_EXPORT", ip="192.0.2.1", error=?"credential");
            let _entered = span.enter();
            tracing::info!(target:"lifecycle", stage="body_ready", block_hash="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            tracing::info!(target:"lifecycle", stage="DO_NOT_EXPORT", payload="PRIVATE_TRANSACTION", private_key="DO_NOT_EXPORT");
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
        assert!(values.iter().any(|v| v["fields"]["stage"] == "body_ready"));
        assert_eq!(values.last().unwrap()["dropped"], 0);
        assert_eq!(
            values.iter().filter(|v| v["type"] == "start").count(),
            values.iter().filter(|v| v["type"] == "end").count()
        );
    }
    #[test]
    fn hot_accessors_preserve_counts_and_owner_without_queue_pressure() {
        let path =
            std::env::temp_dir().join(format!("lifecycle-aggregate-{}.jsonl", monotonic_ns()));
        let (layer, guard) =
            LifecycleLayer::start(File::create(&path).unwrap(), [1; 32], monotonic_ns()).unwrap();
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
