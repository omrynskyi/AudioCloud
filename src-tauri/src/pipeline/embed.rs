//! CLAP inference: batching accumulator over a shared `ort` session.
//!
//! One `Arc<Session>` shared across `rayon` workers -- never one session per thread.
//! Batches accumulate to a tuned size with a flush timeout, so the tail of a scan does not
//! stall waiting on a batch that will never fill. Outputs are L2-normalized on receipt,
//! before anything else touches them.
//!
//! Populated in Phase 4.
