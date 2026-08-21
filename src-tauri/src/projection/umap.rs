//! UMAP via `annembed` over an `hnsw_rs` kNN graph.
//!
//! `annembed` is pinned to an exact version and **vendored into the repo**
//! (`overview.md` §10, risk 2) -- it is a small crate with a single maintainer and the
//! layout of the entire map depends on it.
//!
//! The embedding matrix is read through `mmap`, never heap-loaded: a 50k x 512 f16 matrix
//! is 51 MB that the page cache is better at managing than the allocator.
//!
//! Populated in Phase 5.
