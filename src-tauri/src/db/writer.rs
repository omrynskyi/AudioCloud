//! The single SQLite writer thread.
//!
//! Owns the only write connection in the process. Accepts an `mpsc` command enum and
//! replies over oneshot channels. Writes are batched into transactions that flush at
//! 1000 rows **or** 250 ms, whichever comes first (`overview.md` §4.4).
//!
//! Populated in Phase 1.
