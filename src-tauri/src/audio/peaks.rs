//! Waveform peak-summary generation for the inspector display.
//!
//! Summaries are computed once, cached, and served over the `abpeaks://` scheme rather
//! than shipped through `invoke` -- see `protocol::peaks`.
//!
//! Populated in Phase 8.
