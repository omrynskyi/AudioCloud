//! Procrustes alignment of a new projection onto the previous one (`overview.md` §3.8).
//!
//! UMAP is not stable across runs: re-fitting the same corpus produces a rotated,
//! reflected, arbitrarily scaled cloud. Without alignment, adding 200 samples visually
//! teleports the user's entire library and destroys the spatial memory that makes the map
//! worth using.
//!
//! Center, cross-covariance, `R = V Uᵀ`, uniform scale -- and **reflection is allowed**.
//! Forcing `det(R) = +1` here is wrong: a reflected embedding is an equally valid UMAP
//! solution, and refusing to reflect leaves the cloud mirrored.
//!
//! Populated in Phase 5.
