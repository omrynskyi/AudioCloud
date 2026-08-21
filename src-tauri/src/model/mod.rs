//! CLAP model provisioning and inference session management (`overview.md` §3.5).
//!
//! The model is a release asset, not a repo artifact -- 200 MB does not belong in git.
//! It is downloaded on first run, verified by SHA-256, and versioned independently of the
//! app so a model revision is a download prompt rather than a full app update.
//!
//! Populated in Phase 3.

pub mod download;
pub mod session;
