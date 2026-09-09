//! Barnes-Hut t-SNE: PCA down to a modest number of dimensions, then Barnes-Hut t-SNE, fit
//! **independently** at whatever output dimensionality is asked for. [`TsneProjector`] fits
//! 2D for position; [`fit_color`] is a second, separate fit at 3D, normalized to `[0, 1]`
//! per axis for use as RGB. Neither is derived from the other -- flattening a 3D fit for
//! display was tried and measured worse than fitting the actual target dimensionality
//! directly (`umap::TARGET_DIM`'s doc), and the same reasoning applies here in the other
//! direction: color asks a different question of the data than position does, and answering
//! both from one fit would average two different notions of "close" together.
//!
//! **Why `nalgebra` for the PCA step and not `pca.rs`'s streaming version.** `pca.rs` earns
//! its chunked, mmap-driven accumulation because a UMAP re-fit must not hold 50,000 × 512
//! floats in memory at once (`overview.md` §4.2). That discipline buys nothing here: t-SNE's
//! affinity graph is pairwise over every row regardless, so the whole matrix is in memory the
//! moment `barnes_hut` is called. This PCA step reduces the matrix already in hand, and reuses
//! `pca.rs`'s covariance/eigen approach at a smaller scale.

use nalgebra::{DMatrix, DVector};

use crate::{
    pipeline::CancellationToken,
    projection::{EmbeddingSet, Point3, ProjectionError, Projector},
};

/// Below this there is no cloud to find structure in. Same floor as [`super::pca`]'s.
const MIN_SAMPLES: usize = 4;

/// PCA dimensions t-SNE actually sees: enough to keep the real structure of a
/// high-dimensional fingerprint or embedding, small enough that the affinity computation is
/// still cheap.
const INITIAL_DIMS: usize = 30;

/// What one t-SNE fit was asked for. Recorded verbatim in `projection_runs.params_json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsneParams {
    /// Effective neighborhood size of the conditional distribution.
    ///
    /// Van der Maaten's paper recommends 5-50; measured against this app's own library (real
    /// CLAP embeddings, `examples/retune_experiment.rs`, 5 repeats per value, swept 5-25), 12
    /// is the actual sweet spot here -- "is the sample nearest a hover truly one of its 15
    /// most similar" peaks at 94.8% around
    /// perplexity 8-12 and falls off on both sides (93.0% at 5, 90.3% at 25), while "are its
    /// 5 closest truly in that top-15" keeps climbing gently past it (peaks near 65.5% at
    /// 15-20). 12 sits at the point neither curve has started giving back what the other
    /// gained. Against the same corpus, this is a wholesale improvement over UMAP under its
    /// own tuned defaults (`umap::UmapParams`'s doc): 66% hit-rate@1 and 47% hit-rate@5 for
    /// UMAP, 94.8% and 65.4% here -- not a close call.
    pub perplexity: f32,
    /// Barnes-Hut approximation accuracy. Lower is more accurate and slower; `0.0` is exact
    /// t-SNE. `0.5` is `bhtsne`'s own default, and nothing in the sweep above found a reason
    /// to move it.
    pub theta: f32,
    /// Gradient descent iterations.
    pub epochs: usize,
}

impl Default for TsneParams {
    fn default() -> Self {
        Self {
            perplexity: 12.0,
            theta: 0.5,
            epochs: 1000,
        }
    }
}

/// t-SNE onto 2 dimensions, for the map's position.
#[derive(Debug, Clone, Copy, Default)]
pub struct TsneProjector {
    params: TsneParams,
}

impl TsneProjector {
    pub fn new(params: TsneParams) -> Self {
        Self { params }
    }
}

impl Projector for TsneProjector {
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
                "the vectors are zero-dimensional".into(),
            ));
        }

        let rows = materialize(data, cancel)?;
        let started = std::time::Instant::now();
        let reduced = pca_reduce(&rows, INITIAL_DIMS.min(dim))?;
        tracing::debug!(
            n,
            dim,
            elapsed_ms = started.elapsed().as_millis(),
            "tsne: pca reduce complete"
        );
        EmbeddingSet::check_cancelled(cancel)?;

        let embedded = fit::<2>(&reduced, &self.params);
        tracing::debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "tsne: barnes-hut fit complete"
        );
        Ok((0..n)
            .map(|i| [embedded[i * 2], embedded[i * 2 + 1], 0.0])
            .collect())
    }

    fn name(&self) -> &'static str {
        "tsne"
    }

    fn params_json(&self) -> String {
        format!(
            r#"{{"perplexity":{},"theta":{},"epochs":{},"initial_dims":{}}}"#,
            self.params.perplexity, self.params.theta, self.params.epochs, INITIAL_DIMS
        )
    }
}

/// A second, independent t-SNE fit at 3 dimensions, normalized to `[0, 1]` per axis for
/// direct use as RGB. A clustering-plus-colormap scheme is a plausible alternative, but a
/// plain normalized 3D t-SNE is simpler and needs no extra parameters (a cluster count, a
/// colormap choice) to tune.
pub fn fit_color(
    data: &EmbeddingSet<'_>,
    params: &TsneParams,
    cancel: &CancellationToken,
) -> Result<Vec<[f32; 3]>, ProjectionError> {
    let n = data.len();
    let dim = data.check_uniform_dimensions()?;
    if n < MIN_SAMPLES || dim == 0 {
        return Ok(vec![[0.5, 0.5, 0.5]; n]);
    }

    let rows = materialize(data, cancel)?;
    let reduced = pca_reduce(&rows, INITIAL_DIMS.min(dim))?;
    EmbeddingSet::check_cancelled(cancel)?;

    let embedded = fit::<3>(&reduced, params);
    let mut colors: Vec<[f32; 3]> = (0..n)
        .map(|i| [embedded[i * 3], embedded[i * 3 + 1], embedded[i * 3 + 2]])
        .collect();
    normalize_to_unit_range(&mut colors);
    Ok(colors)
}

/// Widens every row of `data` into an owned matrix.
///
/// Unlike `pca.rs`'s chunked accumulation, this is the one place in the crate that puts a
/// whole corpus in memory at once on purpose -- see the module doc for why that discipline
/// does not apply to t-SNE.
fn materialize(
    data: &EmbeddingSet<'_>,
    cancel: &CancellationToken,
) -> Result<Vec<Vec<f32>>, ProjectionError> {
    let mut rows = Vec::with_capacity(data.len());
    let mut row = Vec::new();
    for i in 0..data.len() {
        if i % 4096 == 0 {
            EmbeddingSet::check_cancelled(cancel)?;
        }
        data.row_into(i, &mut row)?;
        rows.push(row.clone());
    }
    Ok(rows)
}

/// Plain PCA down to `k` components: center, covariance, top-`k` eigenvectors, project.
///
/// `k` is at most a few dozen and the input is already in memory, so this is a direct
/// `nalgebra` eigendecomposition rather than `pca.rs`'s streaming accumulation -- see the
/// module doc. The eigendecomposition itself runs on whichever of the `n x n` or `dim x dim`
/// symmetric matrix is smaller: a fingerprint's `dim` (1,024) dwarfs a typical library's
/// sample count, and diagonalizing `dim x dim` to keep 30 columns is `dim`-cubed work thrown
/// away for nothing when the mathematically equivalent `n x n` Gram matrix carries the same
/// top eigenvalues at `n`-cubed cost instead (the standard dual-PCA identity: for
/// mean-centered rows `X` with Gram matrix `G = X X^T`, if `u` is a unit eigenvector of `G`
/// with eigenvalue `λ`, then `X^T u / sqrt(λ)` is a unit eigenvector of `X^T X` with the same
/// eigenvalue).
pub(crate) fn pca_reduce(rows: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>, ProjectionError> {
    let n = rows.len();
    let dim = rows.first().map_or(0, Vec::len);
    let k = k.min(dim).max(1);

    let mut mean = vec![0.0f64; dim];
    for row in rows {
        for (m, v) in mean.iter_mut().zip(row.iter()) {
            *m += f64::from(*v);
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    let centered: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .zip(mean.iter())
                .map(|(v, m)| f64::from(*v) - m)
                .collect()
        })
        .collect();
    let denominator = (n as f64 - 1.0).max(1.0);

    let axes: Vec<DVector<f64>> = if dim <= n {
        let mut covariance = DMatrix::<f64>::zeros(dim, dim);
        for row in &centered {
            for a in 0..dim {
                if row[a] == 0.0 {
                    continue;
                }
                for b in a..dim {
                    let contribution = row[a] * row[b];
                    covariance[(a, b)] += contribution;
                    if b != a {
                        covariance[(b, a)] += contribution;
                    }
                }
            }
        }
        covariance /= denominator;

        let eigen = covariance.symmetric_eigen();
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));
        order
            .into_iter()
            .take(k)
            .map(|i| eigen.eigenvectors.column(i).into_owned())
            .collect()
    } else {
        let mut gram = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let dot: f64 = centered[i]
                    .iter()
                    .zip(centered[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                gram[(i, j)] = dot;
                gram[(j, i)] = dot;
            }
        }
        gram /= denominator;

        let eigen = gram.symmetric_eigen();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));

        order
            .into_iter()
            .take(k)
            .filter(|&i| eigen.eigenvalues[i] > 1e-12)
            .map(|i| {
                let u = eigen.eigenvectors.column(i);
                let mut v = DVector::<f64>::zeros(dim);
                for (row, &weight) in centered.iter().zip(u.iter()) {
                    for (a, value) in v.iter_mut().zip(row.iter()) {
                        *a += weight * value;
                    }
                }
                v.normalize_mut();
                v
            })
            .collect()
    };

    Ok(rows
        .iter()
        .map(|row| {
            let centered: Vec<f64> = row
                .iter()
                .zip(mean.iter())
                .map(|(v, m)| f64::from(*v) - m)
                .collect();
            axes.iter()
                .map(|axis| {
                    let mut acc = 0.0f64;
                    for (c, a) in centered.iter().zip(axis.iter()) {
                        acc += c * a;
                    }
                    acc as f32
                })
                .collect()
        })
        .collect())
}

/// Runs Barnes-Hut t-SNE over `rows`, returning the flattened `n * D` embedding.
///
/// `Dim<D>: Morton<D>` is `bhtsne::barnes_hut`'s own bound -- its Z-order tree only covers
/// `D` in `2..=7`, which is every dimensionality this module ever calls `fit` with (2 for
/// position, 3 for color).
fn fit<const D: usize>(rows: &[Vec<f32>], params: &TsneParams) -> Vec<f32>
where
    bhtsne::Dim<D>: bhtsne::Morton<D>,
{
    let samples: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    let mut tsne = bhtsne::tSNE::<f32, &[f32], D>::new(&samples);
    tsne.perplexity(effective_perplexity(params.perplexity, rows.len()))
        .epochs(params.epochs);
    tsne.barnes_hut(params.theta, euclidean);
    tsne.embedding()
}

/// `bhtsne` panics -- a bare `panic!`, not a `Result` -- unless `n_samples - 1 >= 3 *
/// perplexity`. A library small enough to trip that must still get a layout, so this clamps
/// rather than propagating the crate's constraint as a caller-visible failure. The `- 1.0`
/// margin keeps the clamped value strictly inside the crate's own `>=`, so a small corpus
/// never lands exactly on the boundary a future off-by-one in either implementation could
/// cross.
fn effective_perplexity(requested: f32, n: usize) -> f32 {
    let ceiling = ((n as f32 - 1.0) / 3.0 - 1.0).max(1.0);
    requested.min(ceiling)
}

fn euclidean(a: &&[f32], b: &&[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Min-max normalizes each of the 3 columns to `[0, 1]` independently. A column with no
/// spread (every point landed at the same value on that axis) is left at `0.5` rather than
/// dividing by zero.
fn normalize_to_unit_range(colors: &mut [[f32; 3]]) {
    for axis in 0..3 {
        let min = colors.iter().map(|c| c[axis]).fold(f32::INFINITY, f32::min);
        let max = colors
            .iter()
            .map(|c| c[axis])
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        for c in colors.iter_mut() {
            c[axis] = if range > 0.0 && range.is_finite() {
                (c[axis] - min) / range
            } else {
                0.5
            };
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::projection::test_support::{clustered, projected, try_projected};
    use crate::projection::{distance, BoundingBox};

    #[test]
    fn separated_clusters_stay_separated() {
        let vectors = clustered(60, 16, 3, 42);
        let points = projected(&TsneProjector::default(), &vectors);

        let within = distance(points[0], points[3]);
        let between = distance(points[0], points[1]);
        assert!(
            between > within,
            "within {within}, between {between}: the clusters merged"
        );
    }

    #[test]
    fn every_point_lands_on_z_zero() {
        let vectors = clustered(40, 12, 2, 7);
        let points = projected(&TsneProjector::default(), &vectors);
        assert!(points.iter().all(|p| p[2] == 0.0));
    }

    #[test]
    fn the_layout_has_extent_and_is_finite() {
        let vectors = clustered(40, 12, 2, 11);
        let points = projected(&TsneProjector::default(), &vectors);
        let bbox = BoundingBox::of(&points).unwrap();
        assert!(bbox.diagonal() > 1e-6, "the cloud collapsed to a point");
        assert!(points.iter().all(|p| p.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn pca_reduce_finds_clusters_through_the_dual_branch() {
        // `dim` (20) exceeds `n` (18), forcing the Gram-matrix path -- the one the old code
        // never took, since it always diagonalized `dim x dim` regardless of `n`. Kept modest
        // rather than fingerprint-width (1,024): `clustered`'s two hot dimensions are a
        // shrinking fraction of a growing noise floor as `dim` climbs, so a wide corpus with
        // few points doesn't reliably separate under *any* PCA implementation -- that's a
        // property of the fixture, not of the branch under test (see
        // `pca_reduce_stays_fast_when_dim_dwarfs_the_sample_count` for the actual
        // fingerprint-scale case, which only needs to finish quickly, not separate cleanly).
        let vectors = clustered(18, 20, 2, 5);
        let reduced = pca_reduce(&vectors, 10).unwrap();
        let dist = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        // Indices 0 and 2 share a cluster (`center = i % clusters`, clusters=2); index 1 is
        // the other cluster.
        let within = dist(&reduced[0], &reduced[2]);
        let between = dist(&reduced[0], &reduced[1]);
        assert!(
            between > within,
            "within {within}, between {between}: clusters merged"
        );
    }

    #[test]
    fn pca_reduce_stays_fast_when_dim_dwarfs_the_sample_count() {
        // Regression guard for the bug this replaced: diagonalizing the full `dim x dim`
        // covariance matrix to keep 30 columns made a re-fit over fingerprint-width (1,024)
        // embeddings visibly hang (minutes, not seconds) in an unoptimized dev build, even
        // though a library-sized sample count is far smaller than that. The dual/Gram-matrix
        // path this test exercises is `n`-cubed instead of `dim`-cubed, so this must stay
        // fast even in debug mode.
        let vectors = clustered(600, 1024, 3, 9);
        let started = std::time::Instant::now();
        let reduced = pca_reduce(&vectors, 30).unwrap();
        assert!(
            started.elapsed().as_secs() < 60,
            "pca_reduce took {:?} on a 600x1024 corpus -- the dual-PCA path regressed",
            started.elapsed()
        );
        assert_eq!(reduced.len(), 600);
    }

    #[test]
    fn a_corpus_too_small_to_have_a_shape_is_refused_by_name() {
        let vectors: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32; 8]).collect();
        let err = try_projected(&TsneProjector::default(), &vectors).unwrap_err();
        assert!(matches!(
            err,
            ProjectionError::TooFewSamples {
                algorithm: "tsne",
                have: 3,
                need: MIN_SAMPLES
            }
        ));
    }

    #[test]
    fn perplexity_is_clamped_below_the_crates_panic_threshold() {
        // `bhtsne` panics unless `n - 1 >= 3 * perplexity`; the clamp must stay strictly
        // inside that, not land exactly on it.
        for n in [4, 10, 40, 200] {
            let p = effective_perplexity(1000.0, n);
            assert!(
                (n as f32 - 1.0) >= 3.0 * p,
                "n={n} perplexity={p} violates bhtsne's own bound"
            );
        }
        // A request already inside the bound is left alone.
        assert_eq!(effective_perplexity(5.0, 1000), 5.0);
    }

    #[test]
    fn color_lands_entirely_inside_the_unit_cube() {
        let vectors = clustered(30, 10, 3, 5);
        let (_dir, store, rows) = crate::projection::test_support::store_of(&vectors);
        let matrix = store.matrix().unwrap();
        let data = EmbeddingSet::new(&matrix, &rows);

        let colors = fit_color(&data, &TsneParams::default(), &CancellationToken::new()).unwrap();
        assert_eq!(colors.len(), vectors.len());
        for c in &colors {
            for &v in c {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "color channel out of range: {c:?}"
                );
            }
        }
    }
}
