//! Resumable, verified model download.
//!
//! Streams with `Range` so a kill -9 mid-download resumes rather than restarts, hashes
//! incrementally over the stream, and installs by `fsync` + atomic rename from `.partial`
//! only after the hash verifies. A half-written model that passes for complete is the
//! failure mode this ordering exists to make impossible.
//!
//! No network, hash mismatch, disk full, and interrupted-but-resumable are four distinct
//! outcomes with four distinct `AppError` variants and four distinct rendered states.
//!
//! Populated in Phase 3.
