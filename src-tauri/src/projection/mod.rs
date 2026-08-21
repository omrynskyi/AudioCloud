//! Dimensionality reduction to 3D coordinates (`overview.md` §3.7).
//!
//! Defines the `Projector` trait that PCA and UMAP both satisfy, so the map can be re-fit
//! under a different algorithm without the rest of the system knowing. PCA lands first:
//! it is fast, deterministic, and it unblocks the renderer with real coordinates while
//! UMAP is still being vendored.
//!
//! Populated in Phase 5.

pub mod pca;
pub mod procrustes;
pub mod umap;
