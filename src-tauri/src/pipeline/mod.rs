//! Ingest pipeline: stage wiring, bounded channels, and cancellation (`overview.md` §3).
//!
//! Five stages -- walk, decode, mel/features, embed, persist -- connected by bounded
//! channels whose depths are chosen so that backpressure, not memory growth, is what
//! happens when a stage falls behind. A `CancellationToken` threads through every stage;
//! cross-cutting rule 6 says every long operation is cancellable and reports progress on
//! the same throttle.
//!
//! Populated in Phases 2 and 4.

pub mod decode;
pub mod embed;
pub mod features;
pub mod progress;
pub mod walk;
