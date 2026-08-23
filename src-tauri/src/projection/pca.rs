//! Truncated-SVD PCA via `nalgebra`.
//!
//! Deterministic and cheap. Its neighborhoods are worse than UMAP's, but it is the
//! projector that always works, which makes it the right one to ship first and the right
//! fallback when a UMAP re-fit fails.
//!
//! **Why an eigendecomposition of the covariance and not an SVD of the data matrix.**
//! `task.md` asks for a truncated SVD, and this is one: the top-3 right singular vectors of
//! the centered n × d data matrix are exactly the top-3 eigenvectors of its d × d
//! covariance, so the two computations are the same computation. The difference is what
//! they need in memory. `nalgebra` has no truncated SVD -- `SVD::new` is a full
//! decomposition, and it takes a `DMatrix`, which means heap-loading all 50,000 × 512
//! values as f32 and then working over them. That is the 102 MB `overview.md` §4.2 exists
//! to refuse. Accumulating a 512 × 512 covariance instead touches every row of the mmap
//! exactly twice, allocates 2 MB of f64, and hands `nalgebra` a matrix whose size does not
//! depend on the size of the library at all.
//!
//! The cost is the usual one: forming `XᵀX` squares the condition number, so directions
//! whose singular values are near the square root of machine epsilon are lost. For the top
//! three axes of a cloud of L2-normalized 512-dimensional vectors that is not a real
//! concern, and the alternative costs 102 MB to fix a problem this projector does not have.
//!
//! **Determinism includes the signs.** An eigenvector is only defined up to sign, and which
//! one an eigensolver returns is an artifact of its iteration. Left alone, that is enough to
//! mirror the entire map between two runs over identical data. [`canonicalize_sign`] pins it:
//! the component with the largest magnitude is made positive. It costs nothing and it is
//! what makes a PCA re-fit land almost exactly on top of the layout it replaces, before
//! Procrustes is even asked.

use nalgebra::{DMatrix, DVector};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    pipeline::CancellationToken,
    projection::{EmbeddingSet, Point3, ProjectionError, Projector},
};

/// Components extracted. Three, because the renderer draws three.
const COMPONENTS: usize = 3;

/// Below this there is no cloud to find axes in.
///
/// Four rather than three: with exactly three points the top three components describe the
/// plane they happen to lie in perfectly, which is a fit with no residual and no meaning.
const MIN_SAMPLES: usize = 4;

/// An eigenvalue at or below this fraction of the largest is treated as noise rather than
/// as an axis.
///
/// Not zero: a rank-deficient cloud (every vector identical, or a library that is one sound
/// under many names) produces eigenvalues that are floating-point dust rather than exact
/// zeros, and projecting onto dust puts the whole corpus in an arbitrary line through the
/// origin. Such an axis is collapsed to zero instead, which draws a flat cloud -- visibly
/// wrong, rather than invisibly meaningless.
const RANK_EPSILON: f64 = 1e-9;

/// PCA onto the three directions of greatest variance.
#[derive(Debug, Clone, Copy, Default)]
pub struct PcaProjector;

impl PcaProjector {
    pub fn new() -> Self {
        Self
    }
}

impl Projector for PcaProjector {
    fn fit_transform(
        &self,
        data: &EmbeddingSet<'_>,
        cancel: &CancellationToken,
    ) -> Result<Vec<Point3>, ProjectionError> {
        let n = data.len();
        if n < MIN_SAMPLES {
            return Err(ProjectionError::TooFewSamples {
                algorithm: self.name(),
                have: n,
                need: MIN_SAMPLES,
            });
        }
        let dim = data.check_uniform_dimensions()?;
        if dim == 0 {
            return Err(ProjectionError::Degenerate(
                "the embeddings are zero-dimensional".into(),
            ));
        }

        let mean = mean_vector(data, cancel)?;
        let covariance = covariance_matrix(data, &mean, cancel)?;
        let axes = principal_axes(&covariance)?;
        project(data, &mean, &axes, cancel)
    }

    fn name(&self) -> &'static str {
        "pca"
    }

    fn params_json(&self) -> String {
        format!(r#"{{"components":{COMPONENTS}}}"#)
    }
}

/// Contiguous row ranges, one per core.
///
/// Contiguous rather than interleaved on purpose: [`crate::db::queries::all_embedding_locs`]
/// returns file order, so a thread that owns a contiguous range walks the mmap forwards and
/// the kernel's readahead is doing something useful. It also fixes the number of live
/// accumulators at the number of chunks -- a `rayon` fold over rows would allocate a d × d
/// f64 matrix per work-stealing task, and at 2 MB each that is a memory profile decided by
/// the scheduler.
fn chunks(n: usize) -> Vec<(usize, usize)> {
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(4);
    let per = n.div_ceil(cores).max(1);
    (0..n)
        .step_by(per)
        .map(|start| (start, (start + per).min(n)))
        .collect()
}

/// The centroid, in one pass.
///
/// f64 throughout: 50,000 additions into an f32 accumulator loses the low bits of the mean
/// exactly where the components are smallest, and the mean is subtracted from every row
/// twice afterwards.
fn mean_vector(
    data: &EmbeddingSet<'_>,
    cancel: &CancellationToken,
) -> Result<Vec<f64>, ProjectionError> {
    let dim = data.dim();
    let partials: Vec<Result<Vec<f64>, ProjectionError>> = chunks(data.len())
        .into_par_iter()
        .map(|(start, end)| {
            let mut sum = vec![0.0f64; dim];
            let mut row = Vec::with_capacity(dim);
            for i in start..end {
                if i % 4096 == 0 {
                    EmbeddingSet::check_cancelled(cancel)?;
                }
                data.row_into(i, &mut row)?;
                for (acc, v) in sum.iter_mut().zip(row.iter()) {
                    *acc += f64::from(*v);
                }
            }
            Ok(sum)
        })
        .collect();

    let mut mean = vec![0.0f64; dim];
    for partial in partials {
        for (acc, v) in mean.iter_mut().zip(partial?) {
            *acc += v;
        }
    }
    let n = data.len() as f64;
    for v in &mut mean {
        *v /= n;
    }
    Ok(mean)
}

/// The d × d covariance of the centered rows.
///
/// A second pass rather than the one-pass `XᵀX/n - μμᵀ` identity. The one-pass form
/// subtracts two numbers of similar size to get a small one, and for L2-normalized vectors
/// whose per-component means are the same order as their spread that is precisely the
/// regime where catastrophic cancellation eats the answer. The mean pass is O(n·d) and the
/// pages it touches are still resident when this one runs, so the second traversal costs
/// arithmetic, not IO.
///
/// Only the upper triangle is accumulated; the matrix is symmetric by construction and
/// filling both halves doubles the inner loop for no information.
fn covariance_matrix(
    data: &EmbeddingSet<'_>,
    mean: &[f64],
    cancel: &CancellationToken,
) -> Result<DMatrix<f64>, ProjectionError> {
    let dim = mean.len();
    let partials: Vec<Result<Vec<f64>, ProjectionError>> = chunks(data.len())
        .into_par_iter()
        .map(|(start, end)| {
            let mut acc = vec![0.0f64; dim * dim];
            let mut row = Vec::with_capacity(dim);
            let mut centered = vec![0.0f64; dim];
            for i in start..end {
                if i % 1024 == 0 {
                    EmbeddingSet::check_cancelled(cancel)?;
                }
                data.row_into(i, &mut row)?;
                for ((c, v), m) in centered.iter_mut().zip(row.iter()).zip(mean.iter()) {
                    *c = f64::from(*v) - m;
                }
                for a in 0..dim {
                    let ca = centered[a];
                    if ca == 0.0 {
                        continue;
                    }
                    let dst = &mut acc[a * dim + a..(a + 1) * dim];
                    for (slot, cb) in dst.iter_mut().zip(centered[a..].iter()) {
                        *slot += ca * cb;
                    }
                }
            }
            Ok(acc)
        })
        .collect();

    let mut flat = vec![0.0f64; dim * dim];
    for partial in partials {
        for (acc, v) in flat.iter_mut().zip(partial?) {
            *acc += v;
        }
    }

    // Sample covariance: n - 1, not n. The difference is invisible at 50,000 rows and
    // wrong at 5.
    let denominator = (data.len() as f64 - 1.0).max(1.0);
    let mut covariance = DMatrix::<f64>::zeros(dim, dim);
    for a in 0..dim {
        for b in a..dim {
            let v = flat[a * dim + b] / denominator;
            covariance[(a, b)] = v;
            covariance[(b, a)] = v;
        }
    }
    Ok(covariance)
}

/// The three eigenvectors of largest eigenvalue, sign-canonicalized, largest first.
///
/// An axis whose eigenvalue is below [`RANK_EPSILON`] of the leading one is returned as a
/// zero vector: a row projected onto it lands at zero, so the cloud is drawn flat in that
/// direction instead of being smeared along a numerically arbitrary one.
fn principal_axes(covariance: &DMatrix<f64>) -> Result<Vec<DVector<f64>>, ProjectionError> {
    let eigen = covariance.clone().symmetric_eigen();

    let mut order: Vec<usize> = (0..eigen.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));

    let leading = eigen.eigenvalues[order[0]];
    if leading <= 0.0 || !leading.is_finite() {
        return Err(ProjectionError::Degenerate(format!(
            "the largest eigenvalue of the covariance is {leading}"
        )));
    }

    Ok(order
        .into_iter()
        .take(COMPONENTS)
        .map(|k| {
            if eigen.eigenvalues[k] / leading <= RANK_EPSILON {
                return DVector::zeros(covariance.nrows());
            }
            let mut axis = eigen.eigenvectors.column(k).into_owned();
            canonicalize_sign(axis.as_mut_slice());
            axis
        })
        .collect())
}

/// Flips an eigenvector so its largest-magnitude component is positive.
///
/// Ties are broken by index, which matters more than it looks: without a total order two
/// runs over the same data could pick different components to key on and disagree about the
/// sign, which is the failure this function exists to prevent.
fn canonicalize_sign(axis: &mut [f64]) {
    let pivot =
        axis.iter().enumerate().fold(
            0usize,
            |best, (i, v)| {
                if v.abs() > axis[best].abs() {
                    i
                } else {
                    best
                }
            },
        );
    if axis.get(pivot).is_some_and(|v| *v < 0.0) {
        for v in axis.iter_mut() {
            *v = -*v;
        }
    }
}

/// Row order in, row order out: coordinate `i` belongs to `data.sample_id(i)`.
fn project(
    data: &EmbeddingSet<'_>,
    mean: &[f64],
    axes: &[DVector<f64>],
    cancel: &CancellationToken,
) -> Result<Vec<Point3>, ProjectionError> {
    let dim = mean.len();
    let mut out: Vec<Result<Vec<Point3>, ProjectionError>> = chunks(data.len())
        .into_par_iter()
        .map(|(start, end)| {
            let mut points = Vec::with_capacity(end - start);
            let mut row = Vec::with_capacity(dim);
            for i in start..end {
                if i % 4096 == 0 {
                    EmbeddingSet::check_cancelled(cancel)?;
                }
                data.row_into(i, &mut row)?;
                let mut coord = [0.0f32; COMPONENTS];
                for (k, axis) in axes.iter().enumerate() {
                    let mut acc = 0.0f64;
                    for ((v, m), a) in row.iter().zip(mean.iter()).zip(axis.iter()) {
                        acc += (f64::from(*v) - m) * a;
                    }
                    coord[k] = acc as f32;
                }
                points.push(coord);
            }
            Ok(points)
        })
        .collect();

    let mut coordinates = Vec::with_capacity(data.len());
    for chunk in out.drain(..) {
        coordinates.extend(chunk?);
    }
    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::projection::test_support::projected;
    use crate::projection::{distance, BoundingBox};

    /// The property PCA is for: the direction the data actually varies along becomes the
    /// first coordinate.
    #[test]
    fn the_first_component_is_the_direction_of_greatest_variance() {
        // A line through 8-dimensional space, thick in dimension 0 and thin in dimension 1.
        let vectors: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let t = i as f32 - 20.0;
                let mut v = vec![0.0f32; 8];
                v[0] = t;
                v[1] = 0.01 * ((i % 3) as f32 - 1.0);
                v
            })
            .collect();

        let points = projected(&PcaProjector::new(), &vectors);

        // The spread along x must dominate: the cloud is a line, and PCA found it.
        let bbox = BoundingBox::of(&points).unwrap();
        let x_span = bbox.max[0] - bbox.min[0];
        let y_span = bbox.max[1] - bbox.min[1];
        assert!(x_span > 30.0, "x span was {x_span}");
        assert!(y_span < 0.1, "y span was {y_span}, expected a thin axis");
    }

    /// Determinism is not "it happened to match": the same input twice must produce the
    /// same floats, including the signs an eigensolver is free to choose arbitrarily.
    #[test]
    fn the_same_corpus_projects_to_the_same_coordinates_twice() {
        let vectors: Vec<Vec<f32>> = (0..64)
            .map(|i| {
                (0..8)
                    .map(|d| ((i * 13 + d * 7) % 29) as f32 / 29.0 - 0.5)
                    .collect()
            })
            .collect();

        let first = projected(&PcaProjector::new(), &vectors);
        let second = projected(&PcaProjector::new(), &vectors);

        assert_eq!(first, second);
    }

    /// Three well-separated clumps must stay three clumps: within-cluster distance well
    /// below between-cluster distance. PCA does not promise good neighborhoods in general,
    /// but it must not scramble structure this coarse.
    #[test]
    fn separated_clusters_stay_separated() {
        let mut vectors = Vec::new();
        for cluster in 0..3usize {
            for i in 0..20 {
                let mut v = vec![0.0f32; 8];
                v[cluster] = 1.0 + 0.01 * (i as f32);
                v[3 + cluster] = 0.02 * ((i % 5) as f32);
                vectors.push(v);
            }
        }

        let points = projected(&PcaProjector::new(), &vectors);

        let within = distance(points[0], points[19]);
        let between = distance(points[0], points[20]);
        assert!(
            between > within * 5.0,
            "within {within}, between {between}: the clusters merged"
        );
    }

    /// A corpus of one distinct sound has no axes. It must produce a flat cloud and a
    /// typed error only when there is genuinely nothing at all -- not a panic inside
    /// `nalgebra`.
    #[test]
    fn identical_vectors_are_degenerate_rather_than_fatal() {
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| vec![0.5f32; 8]).collect();
        let err = crate::projection::test_support::try_projected(&PcaProjector::new(), &vectors)
            .unwrap_err();
        assert!(matches!(err, ProjectionError::Degenerate(_)), "got {err:?}");
    }

    #[test]
    fn a_corpus_too_small_to_have_a_shape_is_refused_by_name() {
        let vectors: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32; 8]).collect();
        let err = crate::projection::test_support::try_projected(&PcaProjector::new(), &vectors)
            .unwrap_err();
        assert!(matches!(
            err,
            ProjectionError::TooFewSamples {
                algorithm: "pca",
                have: 3,
                need: MIN_SAMPLES
            }
        ));
    }

    #[test]
    fn a_sign_flip_is_decided_by_the_largest_component_not_by_the_solver() {
        let mut a = vec![0.1, -0.9, 0.3];
        let mut b = vec![-0.1, 0.9, -0.3];
        canonicalize_sign(&mut a);
        canonicalize_sign(&mut b);
        assert_eq!(
            a, b,
            "two signs of one axis must canonicalize to the same one"
        );
        assert_eq!(a, vec![-0.1, 0.9, -0.3]);
    }
}
