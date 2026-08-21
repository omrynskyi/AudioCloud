//! The typed error surface (`overview.md` §6.7).
//!
//! Every failure mode the frontend must render differently is its own variant, tagged for
//! serde so TypeScript can exhaustively match on it. Cross-cutting rule 8: a new failure
//! mode means a new variant and a new rendered state -- never a `.unwrap()`, never a
//! stringly-typed error crossing the IPC boundary.
//!
//! Populated in Phase 6.
