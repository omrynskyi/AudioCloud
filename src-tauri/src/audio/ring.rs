//! Lock-free SPSC ring buffer between the decode-ahead task and the audio callback.
//!
//! A mutex here would be a priority-inversion glitch waiting for a slow scheduler. The
//! producer is a `tokio` task; the consumer is the real-time callback, which may not
//! block, allocate, or lock.
//!
//! Populated in Phase 8.
