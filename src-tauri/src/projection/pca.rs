//! Truncated-SVD PCA via `nalgebra`.
//!
//! Deterministic and cheap. Its neighborhoods are worse than UMAP's, but it is the
//! projector that always works, which makes it the right one to ship first and the right
//! fallback when a UMAP re-fit fails.
//!
//! Populated in Phase 5.
