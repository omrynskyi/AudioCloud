//! Dimensionality reduction to 3D coordinates (`overview.md` §3.7).
//!
//! Defines the [`Projector`] trait that PCA and UMAP both satisfy, so the map can be re-fit
//! under a different algorithm without the rest of the system knowing. PCA lands first:
//! it is fast, deterministic, and it unblocks the renderer with real coordinates while
//! UMAP is still being vendored.
//!
//! **What a projector is given.** An [`EmbeddingSet`], which is a memory-mapped view of
//! `embeddings.bin` plus the `(sample_id, location)` list that says which rows of it are
//! in play. Not a `Vec<Vec<f32>>` and not an `ndarray`: `overview.md` §4.2's whole
//! argument is that 50,000 × 512 f32 is 102 MB the page cache manages better than the
//! allocator, and a trait that took an owned matrix would make honouring that impossible
//! for every implementation at once.
//!
//! **What a projector is not responsible for.** Persistence, the single-active invariant,
//! the atomic swap, and alignment against the previous layout. Those are [`refit`]'s, and
//! they are the same for every algorithm -- which is why they are not on the trait.

pub mod incremental;
pub mod pca;
pub mod procrustes;
pub mod refit;
pub mod tsne;
pub mod umap;

use crate::{
    db::{DbError, EmbeddingLoc, EmbeddingMatrix},
    pipeline::CancellationToken,
};

pub use incremental::{place_incremental, IncrementalReport};
pub use pca::PcaProjector;
pub use procrustes::Alignment;
pub use refit::{
    plan, refit, refit_at_low_priority, ColorFit, Plan, Refit, RefitPhase, RefitProgress,
    RefitReport, RefitSnapshot,
};
pub use tsne::{fit_color, TsneParams, TsneProjector};
pub use umap::{UmapParams, UmapProjector};

/// Coordinates are three floats. Named so the intent survives a signature change.
pub type Point3 = [f32; 3];

/// Everything the projection layer can fail at.
///
/// Note what is *not* here: "the layout is bad". A projector that produces a useless
/// arrangement still produces coordinates, and no error type can tell the difference --
/// that is what `task.md` Phase 4's Risk 5 hand-inspection is for.
#[derive(Debug, thiserror::Error)]
pub enum ProjectionError {
    #[error(transparent)]
    Db(#[from] DbError),

    /// The caller asked to stop. Partial work is discarded, never half-published: a
    /// cancelled re-fit deletes its shadow run and leaves the active one exactly as it was.
    #[error("the projection was cancelled")]
    Cancelled,

    /// Fewer samples than the algorithm can say anything about.
    #[error("{algorithm} needs at least {need} embedded samples, found {have}")]
    TooFewSamples {
        algorithm: &'static str,
        have: usize,
        need: usize,
    },

    /// Rows in the set disagree about their own width, so there is no matrix to decompose.
    #[error("embedding dimensions are not uniform: row 0 has {expected}, row {row} has {actual}")]
    RaggedDimensions {
        expected: usize,
        row: usize,
        actual: usize,
    },

    /// The vectors carry no variance at all -- every row identical, or a corpus of one
    /// distinct sound under fifty filenames. There is no direction to project onto.
    #[error("the embeddings have no variance to project: {0}")]
    Degenerate(String),

    /// `annembed` said no. Its own error type is a bare `usize`, so the message is
    /// reconstructed here (see [`umap`]).
    #[error("umap: {0}")]
    Umap(String),

    /// There is no active projection to place new points into, or to align against.
    #[error("there is no active projection run")]
    NoActiveProjection,
}

/// The rows of `embeddings.bin` one projection is over.
///
/// A borrowed view, not a container: it holds a reference to the mmap and to the caller's
/// location list, and widens one row at a time into a caller-owned buffer. Constructing one
/// costs nothing and reading every row of it costs page cache, which is the property
/// `overview.md` §4.2 exists to protect.
///
/// Row order is the caller's. [`crate::db::queries::all_embedding_locs`] returns file order,
/// which makes a full pass sequential on disk; that is the order every re-fit uses.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingSet<'a> {
    matrix: &'a EmbeddingMatrix,
    rows: &'a [(i64, EmbeddingLoc)],
}

impl<'a> EmbeddingSet<'a> {
    /// Wraps a mapped matrix and the rows of it that are in play.
    ///
    /// The rows are *not* validated here -- an out-of-range location surfaces from
    /// [`Self::row_into`] as a typed [`DbError::EmbeddingOutOfRange`], because a re-fit that
    /// refuses to start because one of 50,000 offsets is stale is worse than one that
    /// reports which offset.
    pub fn new(matrix: &'a EmbeddingMatrix, rows: &'a [(i64, EmbeddingLoc)]) -> Self {
        Self { matrix, rows }
    }

    /// Number of vectors.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Width of a vector, taken from row 0. Zero for an empty set.
    ///
    /// The store enforces a single dimension on append, so this is a fact rather than a
    /// guess -- but a database that outlived a change to [`crate::EMBEDDING_DIM`] could
    /// still hold two widths, which is what [`Self::check_uniform_dimensions`] is for.
    pub fn dim(&self) -> usize {
        self.rows.first().map_or(0, |(_, loc)| loc.dims as usize)
    }

    /// Widens row `i` into `out`, reusing its allocation.
    pub fn row_into(&self, i: usize, out: &mut Vec<f32>) -> Result<(), ProjectionError> {
        let (_, loc) = self.rows.get(i).ok_or_else(|| {
            ProjectionError::Degenerate(format!("row {i} is past the end of the set"))
        })?;
        self.matrix.row_into(*loc, out)?;
        Ok(())
    }

    /// Rejects a set whose rows are not all the same width.
    ///
    /// Every projector calls this before it allocates anything: a ragged set produces a
    /// matrix whose shape is a lie, and the resulting failure would be an index panic
    /// somewhere inside a decomposition rather than a sentence naming the row.
    pub fn check_uniform_dimensions(&self) -> Result<usize, ProjectionError> {
        let expected = self.dim();
        for (row, (_, loc)) in self.rows.iter().enumerate() {
            if loc.dims as usize != expected {
                return Err(ProjectionError::RaggedDimensions {
                    expected,
                    row,
                    actual: loc.dims as usize,
                });
            }
        }
        Ok(expected)
    }

    /// `Err(Cancelled)` if the token has been tripped. Every stage that loops calls this.
    pub fn check_cancelled(cancel: &CancellationToken) -> Result<(), ProjectionError> {
        if cancel.is_cancelled() {
            return Err(ProjectionError::Cancelled);
        }
        Ok(())
    }
}

/// 512 dimensions to 3 (`overview.md` §3.7).
///
/// `Send + Sync` because a re-fit runs on a background thread and the projector is chosen
/// on whichever thread the command arrived on.
pub trait Projector: Send + Sync {
    /// Projects every row of `data`, returning one coordinate per row **in row order**.
    ///
    /// Row order is the contract, and it is load-bearing: [`refit`] pairs this output with
    /// the rows it built the set from *positionally*. A projector that reordered its output
    /// would scatter the entire library and fail nothing -- which is why `umap.rs` goes to
    /// the trouble of undoing `annembed`'s internal permutation rather than trusting it.
    fn fit_transform(
        &self,
        data: &EmbeddingSet<'_>,
        cancel: &CancellationToken,
    ) -> Result<Vec<Point3>, ProjectionError>;

    /// What goes in `projection_runs.algorithm`. `'umap'` or `'pca'`.
    fn name(&self) -> &'static str;

    /// What goes in `projection_runs.params_json`.
    ///
    /// Recorded rather than reconstructed, because the defaults will move and a run fitted
    /// under the old ones must still be able to say what it was.
    fn params_json(&self) -> String;
}

/// The axis-aligned extent of a point cloud, and the diagonal of it.
///
/// Displacement is only meaningful relative to the size of the thing that moved: three
/// units is the whole map on one layout and a rounding error on another. Every stability
/// claim in this phase is stated as a fraction of [`Self::diagonal`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min: Point3,
    pub max: Point3,
}

impl BoundingBox {
    /// The box containing every point, or `None` for an empty cloud.
    pub fn of(points: &[Point3]) -> Option<Self> {
        let mut iter = points.iter();
        let first = *iter.next()?;
        let mut bbox = Self {
            min: first,
            max: first,
        };
        for p in iter {
            for (axis, v) in p.iter().enumerate() {
                bbox.min[axis] = bbox.min[axis].min(*v);
                bbox.max[axis] = bbox.max[axis].max(*v);
            }
        }
        Some(bbox)
    }

    pub fn diagonal(&self) -> f32 {
        let mut sum = 0.0;
        for axis in 0..3 {
            let d = self.max[axis] - self.min[axis];
            sum += d * d;
        }
        sum.sqrt()
    }
}

/// Euclidean distance between two coordinates.
pub fn distance(a: Point3, b: Point3) -> f32 {
    let mut sum = 0.0;
    for axis in 0..3 {
        let d = a[axis] - b[axis];
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Fixtures for the projector unit tests.
    //!
    //! A projector reads an mmap, so a test that wants to project forty vectors has to put
    //! them in a real `embeddings.bin` first. That is three lines and it is the right
    //! amount of setup: a fake `EmbeddingSet` over a `Vec<Vec<f32>>` would test a code path
    //! nothing in the application uses, and the mmap is the part §4.2 is a claim about.

    #![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use tempfile::TempDir;

    use super::*;
    use crate::db::EmbeddingStore;

    /// A store holding `vectors`, and the locations they landed at.
    pub fn store_of(vectors: &[Vec<f32>]) -> (TempDir, EmbeddingStore, Vec<(i64, EmbeddingLoc)>) {
        let dim = vectors.first().map_or(0, Vec::len);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut store = EmbeddingStore::open(dir.path(), dim).expect("open store");
        let locs = store.append_batch(vectors).expect("append");
        let rows = locs
            .into_iter()
            .enumerate()
            // Sample ids start at 1, as SQLite's do, so a test that confuses an id with an
            // index fails instead of accidentally agreeing.
            .map(|(i, loc)| (i as i64 + 1, loc))
            .collect();
        (dir, store, rows)
    }

    /// Runs `projector` over `vectors`, returning the coordinates or the error.
    pub fn try_projected(
        projector: &dyn Projector,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<Point3>, ProjectionError> {
        let (_dir, store, rows) = store_of(vectors);
        let matrix = store.matrix().expect("map");
        let data = EmbeddingSet::new(&matrix, &rows);
        projector.fit_transform(&data, &CancellationToken::new())
    }

    /// [`try_projected`], for the tests that expect it to work.
    pub fn projected(projector: &dyn Projector, vectors: &[Vec<f32>]) -> Vec<Point3> {
        try_projected(projector, vectors).expect("projection failed")
    }

    /// `count` L2-normalized vectors of width `dim`, gathered into `clusters` clumps.
    ///
    /// Deterministic: the same seed is the same corpus, which is what lets a stability test
    /// add 5% new samples to a corpus and know the other 95% are byte-identical.
    ///
    /// **The clumps overlap on purpose.** Noise at a tenth of the cluster separation makes
    /// a corpus whose kNN graph falls into disconnected pieces, which `annembed` cannot
    /// embed at all (see [`crate::projection::umap`]) -- and which is also not what a real
    /// library looks like. A corpus with structure that a projector can find *and*
    /// neighborhoods that reach between clusters is the realistic case and the useful test.
    pub fn clustered(count: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        };
        (0..count)
            .map(|i| {
                let center = i % clusters;
                let mut v: Vec<f32> = (0..dim).map(|_| 0.45 * next()).collect();
                // Two hot dimensions per cluster, so the clusters are not simply the
                // coordinate axes -- a projector that happened to keep the first three
                // dimensions would otherwise look like it had found the structure.
                v[center % dim] += 1.0;
                v[(center * 7 + 3) % dim] += 0.6;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn a_bounding_box_diagonal_is_the_span_of_the_cloud() {
        let bbox = BoundingBox::of(&[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 1.0, 0.0]]).unwrap();
        assert_eq!(bbox.min, [0.0, 0.0, 0.0]);
        assert_eq!(bbox.max, [3.0, 4.0, 0.0]);
        assert!((bbox.diagonal() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn an_empty_cloud_has_no_bounding_box() {
        assert!(BoundingBox::of(&[]).is_none());
    }
}
