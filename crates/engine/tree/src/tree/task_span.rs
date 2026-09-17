//! Keep a queued task's parent alive until the tracing registry creates its child.
use tracing::{debug_span, trace_span, Span};

// A Span converted by value into an Id can close before new_span uses that Id.
// These functions own the queued handle but borrow it during child creation.
pub(super) fn sparse_trie(parent: Span) -> Span {
    debug_span!(target: "engine::tree::payload_processor", parent: &parent, "sparse_trie_task")
}

pub(super) fn hashing(parent: Span) -> Span {
    trace_span!(target: "reth_engine_tree::tree::state_root_strategy::sparse_trie", parent: &parent, "run_hashing_task")
}

pub(super) fn payload_conversion(parent: Span) -> Span {
    debug_span!(target: "engine::tree::payload_validator", parent: &parent, "convert_and_validate")
}

#[cfg(test)]
mod tests {
    use super::*;
    use reth_tracing::tracing_subscriber::{self, prelude::*};

    #[test]
    fn detached_tasks_retain_last_parent_after_cancellation() {
        for (make, target, name) in [
            (
                sparse_trie as fn(Span) -> Span,
                "engine::tree::payload_processor",
                "sparse_trie_task",
            ),
            (
                hashing,
                "reth_engine_tree::tree::state_root_strategy::sparse_trie",
                "run_hashing_task",
            ),
            (payload_conversion, "engine::tree::payload_validator", "convert_and_validate"),
        ] {
            let dispatch = tracing::Dispatch::new(
                tracing_subscriber::registry()
                    .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink)),
            );
            let queued = tracing::dispatcher::with_default(&dispatch, || {
                let caller = tracing::info_span!("caller");
                let queued = caller.in_scope(Span::current);
                let cancelled = caller.clone();
                drop(cancelled);
                drop(caller);
                queued
            });
            std::thread::spawn(move || {
                tracing::dispatcher::with_default(&dispatch, || {
                    let child = make(queued);
                    assert!(!child.is_disabled());
                    let metadata = child.metadata().expect("enabled child metadata");
                    assert_eq!(metadata.target(), target);
                    assert_eq!(metadata.name(), name);
                    let _entered = child.entered();
                })
            })
            .join()
            .expect("late task startup must retain its last parent until child creation");
        }
    }

    #[test]
    fn detached_tasks_accept_disabled_parent() {
        tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
            for make in [sparse_trie, hashing, payload_conversion] {
                assert!(make(Span::none()).is_disabled());
            }
        });
    }
}
