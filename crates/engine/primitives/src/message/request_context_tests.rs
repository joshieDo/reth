use super::*;
use alloy_rpc_types_engine::{ExecutionPayloadSidecar, ExecutionPayloadV1};
use reth_ethereum_engine_primitives::EthPayloadTypes;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id, Record},
    Subscriber,
};
use tracing_subscriber::{
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
    Layer,
};

#[derive(Default, Debug)]
struct Records {
    names: BTreeMap<u64, &'static str>,
    parents: BTreeMap<u64, Option<u64>>,
    fields: BTreeMap<u64, u64>,
    links: Vec<(u64, u64)>,
    steps: Vec<(&'static str, u64)>,
}
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Records>>);
struct Accepted<'a>(&'a mut BTreeMap<u64, u64>, u64);
impl Visit for Accepted<'_> {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "accepted" {
            self.0.insert(self.1, value);
        }
    }
    fn record_debug(&mut self, _: &Field, _: &dyn fmt::Debug) {}
}
impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut r = self.0.lock().unwrap();
        let id = id.into_u64();
        r.names.insert(id, attrs.metadata().name());
        let parent = attrs.parent().map(Id::into_u64).or_else(|| {
            attrs.is_contextual().then(|| ctx.current_span().id().map(Id::into_u64)).flatten()
        });
        r.parents.insert(id, parent);
        attrs.record(&mut Accepted(&mut r.fields, id));
    }
    fn on_record(&self, id: &Id, values: &Record<'_>, _: Context<'_, S>) {
        values.record(&mut Accepted(&mut self.0.lock().unwrap().fields, id.into_u64()));
    }
    fn on_follows_from(&self, id: &Id, other: &Id, _: Context<'_, S>) {
        let mut r = self.0.lock().unwrap();
        // Like the lifecycle exporter, require both exact endpoints to be captured.
        if r.names.contains_key(&id.into_u64()) && r.names.contains_key(&other.into_u64()) {
            r.links.push((id.into_u64(), other.into_u64()));
        }
    }
    fn on_enter(&self, id: &Id, _: Context<'_, S>) {
        self.0.lock().unwrap().steps.push(("enter", id.into_u64()));
    }
    fn on_close(&self, id: Id, _: Context<'_, S>) {
        self.0.lock().unwrap().steps.push(("close", id.into_u64()));
    }
}
fn payload() -> ExecutionData {
    ExecutionData {
        payload: ExecutionPayloadV1::from_block_slow(&reth_ethereum_primitives::Block::default())
            .into(),
        sidecar: ExecutionPayloadSidecar::none(),
    }
}
fn poll_once<F: Future>(f: Pin<&mut F>) -> Poll<F::Output> {
    f.poll(&mut core::task::Context::from_waker(futures::task::noop_waker_ref()))
}

#[test]
fn same_payload_reversed_service_keeps_requests_and_cancellation_separate() {
    let capture = Capture::default();
    tracing::subscriber::with_default(tracing_subscriber::registry().with(capture.clone()), || {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = ConsensusEngineHandle::<EthPayloadTypes>::new(tx);
        let first = tracing::info_span!("verify");
        let second = tracing::info_span!("verify");
        let first_id = first.id().unwrap().into_u64();
        let second_id = second.id().unwrap().into_u64();
        let mut f1 = Box::pin(handle.new_payload_with_parent(payload(), first));
        let mut f2 = Box::pin(handle.new_payload_with_parent(payload(), second));
        assert!(poll_once(f1.as_mut()).is_pending());
        assert!(poll_once(f2.as_mut()).is_pending());
        let m1 = rx.try_recv().unwrap();
        let m2 = rx.try_recv().unwrap();
        drop(f2); // Cancellation leaves the queued context attached to the original request.
        for (message, expected, canceled) in [(m2, second_id, true), (m1, first_id, false)] {
            let BeaconEngineMessage::NewPayload { payload: _, tx, context } = message else {
                panic!()
            };
            let service = context.unwrap().start();
            let sid = service.span().id().unwrap().into_u64();
            let qid = {
                let r = capture.0.lock().unwrap();
                assert!(r.links.contains(&(sid, expected)));
                r.links
                    .iter()
                    .find(|(a, b)| *a == sid && r.names[b] == "engine.new_payload.queue")
                    .unwrap()
                    .1
            };
            assert!(
                capture.0.lock().unwrap().steps.contains(&("close", qid)),
                "queue must close before service entry"
            );
            service.in_scope(|| {
                let child = tracing::debug_span!("on_new_payload");
                assert_eq!(
                    capture.0.lock().unwrap().parents[&child.id().unwrap().into_u64()],
                    Some(sid)
                );
                let delivered = tx.send(Ok(PayloadStatus::from_status(PayloadStatusEnum::Syncing)));
                assert_eq!(delivered.is_err(), canceled);
                service.span().record("accepted", u64::from(delivered.is_ok()));
            });
            let r = capture.0.lock().unwrap();
            assert_eq!(r.fields[&qid], 1);
            assert_eq!(r.fields[&sid], u64::from(!canceled));
        }
        assert!(matches!(poll_once(f1.as_mut()), Poll::Ready(Ok(_))));
    });
}

#[test]
fn ordinary_and_disabled_requests_stay_untraced_and_closed_transport_fails() {
    let capture = Capture::default();
    tracing::subscriber::with_default(tracing_subscriber::registry().with(capture.clone()), || {
        let _ancestor = tracing::info_span!("ancestor").entered();
        assert!(NewPayloadContext::new(tracing::Span::none()).is_none());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = ConsensusEngineHandle::<EthPayloadTypes>::new(tx);
        let mut f = Box::pin(handle.new_payload(payload()));
        assert!(poll_once(f.as_mut()).is_pending());
        let BeaconEngineMessage::NewPayload { context, tx, .. } = rx.try_recv().unwrap() else {
            panic!()
        };
        assert!(context.is_none());
        drop(tx);
        assert!(matches!(
            poll_once(f.as_mut()),
            Poll::Ready(Err(BeaconOnNewPayloadError::EngineUnavailable))
        ));
        drop(rx);
        let parent = tracing::info_span!("verify");
        let mut f = Box::pin(handle.new_payload_with_parent(payload(), parent));
        assert!(matches!(
            poll_once(f.as_mut()),
            Poll::Ready(Err(BeaconOnNewPayloadError::EngineUnavailable))
        ));
        let r = capture.0.lock().unwrap();
        let queues: Vec<_> = r
            .names
            .iter()
            .filter(|(_, n)| **n == "engine.new_payload.queue")
            .map(|(id, _)| *id)
            .collect();
        assert_eq!(queues.len(), 1);
        assert_eq!(r.fields[&queues[0]], 0);
        assert!(r.steps.contains(&("close", queues[0])));
        assert!(!r.names.values().any(|n| *n == "engine.new_payload.service"));
    });
}

#[test]
fn filtered_request_cannot_be_replaced_by_enabled_ancestor_link() {
    let capture = Capture::default();
    let subscriber = tracing_subscriber::registry()
        .with(
            capture
                .clone()
                .with_filter(tracing_subscriber::filter::filter_fn(|m| m.name() != "verify")),
        )
        .with(tracing_subscriber::layer::Identity::new());
    tracing::subscriber::with_default(subscriber, || {
        let _ancestor = tracing::info_span!("ancestor").entered();
        let parent = tracing::info_span!("verify");
        assert!(!parent.is_disabled());
        let service = NewPayloadContext::new(parent).unwrap().start();
        service.in_scope(|| {});
    });
    let r = capture.0.lock().unwrap();
    assert!(!r.names.values().any(|n| *n == "verify"));
    assert!(r.links.iter().all(|(_, to)| r.names[to] == "engine.new_payload.queue"));
}

#[test]
fn globally_disabled_queue_does_not_create_context_for_enabled_request() {
    let subscriber =
        tracing_subscriber::registry().with(tracing_subscriber::filter::LevelFilter::INFO);
    tracing::subscriber::with_default(subscriber, || {
        let parent = tracing::info_span!("verify");
        assert!(!parent.is_disabled());
        assert!(NewPayloadContext::new(parent).is_none());
    });
}

#[test]
fn queue_and_service_use_explicit_parents_dispatcher() {
    let capture = Capture::default();
    let parent = tracing::subscriber::with_default(
        tracing_subscriber::registry().with(capture.clone()),
        || tracing::info_span!("verify"),
    );
    let parent_id = parent.id().unwrap().into_u64();
    let unrelated = Capture::default();
    tracing::subscriber::with_default(
        tracing_subscriber::registry().with(unrelated.clone()),
        || {
            let _ancestor = tracing::info_span!("unrelated").entered();
            let context = NewPayloadContext::new(parent).unwrap();
            // Service also works without the originating subscriber as the thread default.
            let service = std::thread::spawn(move || context.start()).join().unwrap();
            let service_id = service.span().id().unwrap().into_u64();
            service.span().record("accepted", 0_u64);
            drop(service);
            let r = capture.0.lock().unwrap();
            let queue_id =
                *r.names.iter().find(|(_, name)| **name == "engine.new_payload.queue").unwrap().0;
            assert_eq!(r.names.len(), 3);
            assert_eq!(r.parents[&queue_id], Some(parent_id));
            assert_eq!(r.parents[&service_id], Some(parent_id));
            assert!(r.links.contains(&(queue_id, parent_id)));
            assert!(r.links.contains(&(service_id, parent_id)));
            assert!(r.links.contains(&(service_id, queue_id)));
            assert_eq!(r.fields[&queue_id], 1);
            assert_eq!(r.fields[&service_id], 0);
            assert!(r.steps.contains(&("close", queue_id)));
        },
    );
    let r = unrelated.0.lock().unwrap();
    assert_eq!(r.names.values().copied().collect::<Vec<_>>(), ["unrelated"]);
    assert!(r.links.is_empty());
}

#[test]
fn queue_filter_uses_explicit_parents_dispatcher() {
    let parent = tracing::subscriber::with_default(
        tracing_subscriber::registry().with(tracing_subscriber::filter::LevelFilter::INFO),
        || {
            let _ancestor = tracing::info_span!("ancestor").entered();
            tracing::info_span!("verify")
        },
    );
    let unrelated = Capture::default();
    tracing::subscriber::with_default(
        tracing_subscriber::registry().with(unrelated.clone()),
        || assert!(NewPayloadContext::new(parent).is_none()),
    );
    assert!(unrelated.0.lock().unwrap().names.is_empty());
}

#[test]
fn canceled_transport_closes_context_under_explicit_parents_dispatcher() {
    let capture = Capture::default();
    let (mut response, receiver) = tracing::subscriber::with_default(
        tracing_subscriber::registry().with(capture.clone()),
        || {
            let _ancestor = tracing::info_span!("ancestor").entered();
            let parent = tracing::info_span!("verify");
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let handle = ConsensusEngineHandle::<EthPayloadTypes>::new(tx);
            let mut response =
                Box::pin(async move { handle.new_payload_with_parent(payload(), parent).await });
            assert!(poll_once(response.as_mut()).is_pending());
            (response, rx)
        },
    );
    let unrelated = Capture::default();
    tracing::subscriber::with_default(
        tracing_subscriber::registry().with(unrelated.clone()),
        || {
            let _ancestor = tracing::info_span!("unrelated").entered();
            drop(receiver);
            assert!(matches!(
                poll_once(response.as_mut()),
                Poll::Ready(Err(BeaconOnNewPayloadError::EngineUnavailable))
            ));
        },
    );
    let r = capture.0.lock().unwrap();
    assert_eq!(r.names.len(), 3);
    assert!(r.names.keys().all(|id| r.steps.contains(&("close", *id))));
    let queue_id = r.names.iter().find(|(_, name)| **name == "engine.new_payload.queue").unwrap().0;
    assert_eq!(r.fields[queue_id], 0);
    let r = unrelated.0.lock().unwrap();
    assert_eq!(r.names.values().copied().collect::<Vec<_>>(), ["unrelated"]);
    assert_eq!(r.steps.iter().filter(|(step, _)| *step == "close").count(), 1);
}
