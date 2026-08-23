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
//! **What this cannot do**, restated here because it is the thing most likely to be
//! forgotten at a bug report: Procrustes fixes the *global* transform. If a re-fit
//! legitimately decides two clusters should merge, no rigid alignment preserves the old
//! picture, because the old picture is now wrong. This turns "everything moved" into "most
//! things stayed, some things genuinely changed" and nothing more.

use nalgebra::{Matrix3, Vector3};

use crate::projection::Point3;

/// Below this many shared points the fit is noise, and applying it would move a whole
/// library on the evidence of a handful of samples.
///
/// Three is the minimum that determines a rotation in 3D at all; four insists on one point
/// of redundancy, so a degenerate triple cannot decide the transform by itself.
pub const MIN_CORRESPONDENCES: usize = 4;

/// A similarity transform: rotate (possibly reflecting), scale uniformly, translate.
///
/// Applied to the *entire* new layout, new points included -- the alignment is a property of
/// the two coordinate systems, not of the points that happened to be shared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alignment {
    /// `R`, in row-major order. Orthogonal; `det` may be -1.
    rotation: Matrix3<f32>,
    scale: f32,
    /// Centroid of the shared points in the *source* (new) layout.
    source_centroid: Vector3<f32>,
    /// Centroid of the same points in the *target* (previous) layout.
    target_centroid: Vector3<f32>,
}

impl Alignment {
    /// The transform that leaves every point exactly where it is.
    ///
    /// Returned rather than `None` wherever "do not move anything" is the right answer, so
    /// callers do not each grow a branch for the first re-fit of a fresh library.
    pub fn identity() -> Self {
        Self {
            rotation: Matrix3::identity(),
            scale: 1.0,
            source_centroid: Vector3::zeros(),
            target_centroid: Vector3::zeros(),
        }
    }

    /// Whether this transform does nothing.
    pub fn is_identity(&self) -> bool {
        *self == Self::identity()
    }

    /// `det(R)`. Negative means the fit chose to reflect, which is allowed and expected.
    pub fn determinant(&self) -> f32 {
        self.rotation.determinant()
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Maps one point from the source layout into the target's frame.
    pub fn apply(&self, point: Point3) -> Point3 {
        let centered = Vector3::new(point[0], point[1], point[2]) - self.source_centroid;
        let mapped = self.rotation * centered * self.scale + self.target_centroid;
        [mapped.x, mapped.y, mapped.z]
    }

    /// Maps a whole layout in place.
    pub fn apply_all(&self, points: &mut [Point3]) {
        if self.is_identity() {
            return;
        }
        for p in points.iter_mut() {
            *p = self.apply(*p);
        }
    }
}

/// Fits the similarity transform carrying `source` onto `target`.
///
/// The two slices are *correspondences*: `source[i]` and `target[i]` are the same sample in
/// the new and the previous layout. Order is the caller's contract, and getting it wrong
/// produces a perfectly valid transform onto the wrong thing -- which is why
/// [`crate::projection::refit`] builds both slices in one pass over one map rather than
/// sorting two lists and hoping.
///
/// Returns [`Alignment::identity`] when there is nothing to fit from: too few
/// correspondences, or a source cloud with no extent (every shared point in the same place,
/// so no rotation is determined).
pub fn fit(source: &[Point3], target: &[Point3]) -> Alignment {
    let n = source.len().min(target.len());
    if n < MIN_CORRESPONDENCES {
        return Alignment::identity();
    }

    let source_centroid = centroid(&source[..n]);
    let target_centroid = centroid(&target[..n]);

    // H = Aᵀ B over the centered clouds, accumulated as a sum of outer products so neither
    // n × 3 matrix is ever materialized.
    let mut h = Matrix3::<f64>::zeros();
    let mut source_variance = 0.0f64;
    for i in 0..n {
        let a = center(source[i], source_centroid);
        let b = center(target[i], target_centroid);
        source_variance += a.norm_squared();
        h += a * b.transpose();
    }

    if source_variance <= 0.0 || !source_variance.is_finite() {
        return Alignment::identity();
    }

    let svd = h.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        // `nalgebra`'s Jacobi SVD does not converge. Vanishingly unlikely on a 3 × 3 and
        // not worth failing a re-fit over: an unaligned layout is ugly, a lost re-fit is
        // an hour of the user's battery.
        tracing::warn!("procrustes: the 3x3 SVD did not converge; leaving the layout unaligned");
        return Alignment::identity();
    };

    // R = V Uᵀ. **No determinant correction.** The textbook Kabsch algorithm negates the
    // last column of V when det(V Uᵀ) < 0, because it is solving for a rigid body's
    // orientation and a reflected body is a different body. A UMAP layout has no
    // handedness: its axes carry no meaning, so a reflection is an equally valid solution
    // and forbidding it leaves the map mirrored, which is the exact failure this whole
    // module exists to prevent (`overview.md` §3.8, step 3).
    let rotation = v_t.transpose() * u.transpose();

    // Optimal uniform scale: the sum of singular values over the source's variance. The
    // singular values of H are the aligned per-axis covariances, so their sum is how much
    // of the source's spread the rotation actually carries onto the target.
    let scale = svd.singular_values.sum() / source_variance;
    if !scale.is_finite() || scale <= 0.0 {
        return Alignment::identity();
    }

    Alignment {
        rotation: rotation.map(|v| v as f32),
        scale: scale as f32,
        source_centroid: source_centroid.map(|v| v as f32),
        target_centroid: target_centroid.map(|v| v as f32),
    }
}

/// Mean position, in f64. Summing 50,000 f32 coordinates loses the centroid's low bits
/// exactly when the cloud is large, which is when the centroid matters most.
fn centroid(points: &[Point3]) -> Vector3<f64> {
    let mut sum = Vector3::<f64>::zeros();
    for p in points {
        sum += Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
    }
    sum / points.len() as f64
}

fn center(p: Point3, centroid: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])) - centroid
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::projection::distance;

    /// A cloud with enough extent in all three axes to determine a rotation.
    fn cloud() -> Vec<Point3> {
        (0..24)
            .map(|i| {
                let t = i as f32;
                [
                    (t * 0.7).sin() * 3.0,
                    (t * 1.3).cos() * 2.0,
                    (t * 0.31).sin() * 1.5 + t * 0.05,
                ]
            })
            .collect()
    }

    fn transformed(points: &[Point3], f: impl Fn(Point3) -> Point3) -> Vec<Point3> {
        points.iter().copied().map(f).collect()
    }

    fn max_error(a: &[Point3], b: &[Point3]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| distance(*x, *y))
            .fold(0.0f32, f32::max)
    }

    /// The core claim: a rotated, scaled, translated copy of a layout is recovered exactly.
    #[test]
    fn a_rotation_scale_and_translation_are_undone() {
        let target = cloud();
        // 90 degrees about z, doubled, shifted.
        let source = transformed(&target, |p| {
            [-p[1] * 2.0 + 10.0, p[0] * 2.0 - 4.0, p[2] * 2.0]
        });

        let alignment = fit(&source, &target);
        let mut aligned = source.clone();
        alignment.apply_all(&mut aligned);

        assert!(
            max_error(&aligned, &target) < 1e-3,
            "worst point off by {}",
            max_error(&aligned, &target)
        );
        assert!((alignment.scale() - 0.5).abs() < 1e-4);
        assert!(alignment.determinant() > 0.0);
    }

    /// **Reflection is allowed.** A mirrored layout must come back unmirrored; a Kabsch
    /// implementation that forces `det(R) = +1` fails this test, which is why it exists.
    #[test]
    fn a_reflected_layout_is_recovered_rather_than_left_mirrored() {
        let target = cloud();
        let source = transformed(&target, |p| [-p[0], p[1], p[2]]);

        let alignment = fit(&source, &target);
        let mut aligned = source.clone();
        alignment.apply_all(&mut aligned);

        assert!(
            alignment.determinant() < 0.0,
            "the fit refused to reflect: det = {}",
            alignment.determinant()
        );
        assert!(
            max_error(&aligned, &target) < 1e-3,
            "worst point off by {}",
            max_error(&aligned, &target)
        );
    }

    /// The transform is fitted on the shared points and applied to everything, including
    /// points the previous layout never saw.
    #[test]
    fn the_fit_carries_points_that_were_not_used_to_make_it() {
        let target = cloud();
        let map = |p: Point3| [p[2] + 1.0, p[0] - 2.0, p[1]];
        let source = transformed(&target, map);

        // Fitted on the first eight only.
        let alignment = fit(&source[..8], &target[..8]);
        let mut aligned = source.clone();
        alignment.apply_all(&mut aligned);

        assert!(
            max_error(&aligned[8..], &target[8..]) < 1e-3,
            "the unshared tail landed {} away",
            max_error(&aligned[8..], &target[8..])
        );
    }

    /// Not every re-fit has a predecessor worth aligning to. Too few correspondences must
    /// leave the layout exactly as the projector produced it, not move it somewhere
    /// arbitrary.
    #[test]
    fn too_few_correspondences_leave_the_layout_alone() {
        let target = cloud();
        let source = transformed(&target, |p| [p[0] + 5.0, p[1], p[2]]);

        let alignment = fit(&source[..3], &target[..3]);

        assert!(alignment.is_identity());
        let mut untouched = source.clone();
        alignment.apply_all(&mut untouched);
        assert_eq!(untouched, source);
    }

    /// A source cloud with no extent determines no rotation. The guard must be the
    /// identity, not a matrix of NaNs quietly applied to 50,000 points.
    #[test]
    fn a_source_cloud_with_no_extent_is_the_identity() {
        let source = vec![[1.0, 1.0, 1.0]; 8];
        let target = cloud()[..8].to_vec();

        let alignment = fit(&source, &target);

        assert!(alignment.is_identity());
    }

    /// Alignment of a layout onto itself must be a no-op, not a slow drift. A re-fit that
    /// reproduces the previous layout exactly is the best case, and it must stay put.
    #[test]
    fn aligning_a_layout_onto_itself_moves_nothing() {
        let points = cloud();

        let alignment = fit(&points, &points);
        let mut aligned = points.clone();
        alignment.apply_all(&mut aligned);

        assert!(
            max_error(&aligned, &points) < 1e-4,
            "self-alignment moved points by {}",
            max_error(&aligned, &points)
        );
    }
}
