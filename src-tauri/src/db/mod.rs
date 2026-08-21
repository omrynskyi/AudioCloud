//! SQLite data layer (`overview.md` §4).
//!
//! One writer, many readers. The invariant this module exists to protect: exactly one
//! write connection lives in the whole process (`writer`), and reads come from a small
//! read-only pool (`pool`). A second write connection appearing anywhere is a bug, not an
//! optimization.
//!
//! Populated in Phase 1.

pub mod embeddings;
pub mod pool;
pub mod queries;
pub mod writer;
