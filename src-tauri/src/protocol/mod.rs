//! Custom URI scheme handlers (`overview.md` §6.4).
//!
//! Transport 3. Binary blobs the WebView should fetch and cache itself -- waveform peaks
//! above all -- travel over a registered scheme rather than the IPC channel, so they get
//! browser caching and streaming for free and never contend with command traffic.
//!
//! Populated in Phase 6.

pub mod peaks;
