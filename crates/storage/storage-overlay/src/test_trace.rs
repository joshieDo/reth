//! Test-only recorder for cold execution-overlay scopes.
use std::sync::{Arc, Mutex};
use tracing::{
    span::{Attributes, Id},
    Subscriber,
};
use tracing_subscriber::{layer::Context, prelude::*, registry::LookupSpan, Layer};

#[derive(Clone, Default)]
pub(crate) struct Capture(Arc<Mutex<Vec<SpanRecord>>>);
type SpanRecord = (&'static str, Option<&'static str>);

impl Capture {
    pub(crate) fn run<T>(&self, f: impl FnOnce() -> T) -> T {
        tracing::subscriber::with_default(tracing_subscriber::registry().with(self.clone()), f)
    }
    pub(crate) fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
    pub(crate) fn count(&self, name: &str) -> usize {
        self.0.lock().unwrap().iter().filter(|(n, _)| *n == name).count()
    }
    pub(crate) fn parents(&self, name: &str) -> Vec<&'static str> {
        self.0.lock().unwrap().iter().filter(|(n, _)| *n == name).filter_map(|(_, p)| *p).collect()
    }
}
impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if attrs.metadata().target() != "lifecycle" ||
            attrs.metadata().name() == "state.overlay.execution_overlay" ||
            attrs.metadata().name() == "state.overlay.state_trie_overlay"
        {
            return
        }
        let parent = ctx.span(id).and_then(|s| s.parent()).map(|s| s.metadata().name());
        self.0.lock().unwrap().push((attrs.metadata().name(), parent));
    }
}
