//! `embeddings.bin`: the embedding matrix lives outside SQLite (`overview.md` §4.2).
//!
//! Append-only, f16 storage via `half`, addressed by the `(offset, len)` pair recorded on
//! the sample row. Reads `mmap` the whole matrix rather than heap-loading it, so a 50k x
//! 512 corpus costs page cache instead of 51 MB of resident memory. `compact()` reclaims
//! space after bulk deletion.
//!
//! Populated in Phase 1.
