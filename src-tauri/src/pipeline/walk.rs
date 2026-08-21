//! Filesystem discovery: `ignore::WalkBuilder::build_parallel()` plus `blake3` hashing.
//!
//! Deliberately **not** `jwalk`, which is deprecated upstream (`overview.md` §3.1).
//! A `(path, mtime, size)` fast-skip runs against existing rows before any hashing, so a
//! re-scan of an unchanged tree never reads file contents. Files over 64 MB are hashed by
//! head+tail+length sampling rather than in full.
//!
//! Populated in Phase 2.
