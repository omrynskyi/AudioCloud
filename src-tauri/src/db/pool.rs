//! The `r2d2` read pool (4 read-only connections) and the pragma customizer.
//!
//! Pragmas are per-connection, not per-database: `foreign_keys = ON` set once on one
//! connection does nothing for the others. The customizer applies the full pragma set
//! from `overview.md` §4.3 to **every** connection the pool hands out.
//!
//! Populated in Phase 1.
