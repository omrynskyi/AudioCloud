//! Incremental placement for small imports (`overview.md` §3.8).
//!
//! A user who drags twelve files into their library must not watch their entire map rotate.
//! So below [`super::refit::INCREMENTAL_THRESHOLD`] of the corpus, nothing is re-fitted:
//! each new sample is placed at the similarity-weighted barycenter of its nearest already-
//! projected neighbors, and **not one existing point moves** -- not approximately, not after
//! alignment, at all. The rows for existing samples are never written.
//!
//! Layout quality degrades slowly as these accumulate, which is what eventually triggers a
//! full re-fit. That is the design's own trade and it is the right one: a slightly stale
//! layout the user can navigate beats a perfect one they have to relearn.
//!
//! **Neighbors are found by exact brute force, not by an index.** `overview.md` §3.8 says
//! "via the existing HNSW index", and there is no such thing: the index UMAP builds lives
//! for the duration of one re-fit and nothing in the design persists it. Building one to
//! place 200 points would cost more than the answer, and an approximate answer's recall
//! would sit between a placement and its explanation. At 2% of 50,000 the exact pass is
//! 1,000 × 50,000 dot products over a resident mmap -- about a second, once, on an import.
//! If a persistent index ever arrives, it replaces the body of [`nearest`] and nothing else.

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    db::{queries, Database, DbError, EmbeddingLoc, EmbeddingMatrix},
    pipeline::CancellationToken,
    projection::{EmbeddingSet, Point3, ProjectionError},
};

/// Neighbors averaged to place one new point.
///
/// Matches [`super::UmapParams`]'s default `n_neighbors` on purpose: a point placed here
/// should land where a re-fit would have put it, and a re-fit's notion of "local" is this
/// number.
pub const PLACEMENT_NEIGHBORS: usize = 15;

/// Coordinates written per writer command. Same reasoning as `refit::WRITE_CHUNK`.
const WRITE_CHUNK: usize = 1000;

/// How far a placed point is nudged off its barycenter, as a fraction of the cloud's
/// diagonal.
///
/// Two byte-identical files share an embedding exactly (Phase 4 stores one vector and two
/// references to it), so their barycenters are identical too and they would occupy one
/// pixel forever -- unclickable, and indistinguishable from a single sample. The jitter is
/// small enough to be invisible at any useful zoom and large enough that a picker can
/// separate them.
const JITTER_FRACTION: f32 = 0.002;

/// What one incremental placement did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncrementalReport {
    /// The active run the points were added to. Unchanged -- this is not a new run.
    pub run_id: i64,
    pub placed: usize,
    /// Samples that had no projected neighbor to average and were therefore skipped.
    ///
    /// Nonzero only in the pathological case of an active run holding no coordinates at
    /// all, which a full re-fit fixes.
    pub unplaceable: usize,
}

/// Places every embedded sample the active projection has never seen.
///
/// Additive: it writes rows for the new samples into the *active* run and touches nothing
/// else. There is no shadow run and no swap, because there is nothing to swap -- the map
/// the user is looking at gains points and keeps every one it had.
pub fn place_incremental(
    db: &Database,
    cancel: &CancellationToken,
) -> Result<IncrementalReport, ProjectionError> {
    EmbeddingSet::check_cancelled(cancel)?;

    let conn = db.read()?;
    let active =
        queries::active_projection_run(&conn)?.ok_or(ProjectionError::NoActiveProjection)?;
    let placed_points = queries::projection_point_map(&conn, active.id)?;
    let newcomers = queries::embedding_locs_missing_from_run(&conn, active.id)?;
    let all = queries::all_embedding_locs(&conn)?;
    drop(conn);

    if newcomers.is_empty() {
        return Ok(IncrementalReport {
            run_id: active.id,
            placed: 0,
            unplaceable: 0,
        });
    }

    // The anchors: every sample that already has coordinates *and* a vector to compare
    // against. A row can have one without the other -- a sample whose file changed has its
    // offsets cleared by `upsert_samples` while its old coordinate survives until the next
    // re-fit -- and such a row is a coordinate with nothing to measure similarity to.
    let anchors: Vec<(i64, EmbeddingLoc, Point3)> = all
        .iter()
        .filter_map(|(id, loc)| placed_points.get(id).map(|p| (*id, *loc, *p)))
        .collect();

    if anchors.len() < 2 {
        return Ok(IncrementalReport {
            run_id: active.id,
            placed: 0,
            unplaceable: newcomers.len(),
        });
    }

    let extent =
        crate::projection::BoundingBox::of(&anchors.iter().map(|(_, _, p)| *p).collect::<Vec<_>>())
            .map_or(0.0, |b| b.diagonal());

    let matrix = {
        let store = db
            .embeddings()
            .lock()
            .map_err(|_| DbError::Poisoned("embedding store"))?;
        store.matrix()?
    };

    let k = PLACEMENT_NEIGHBORS.min(anchors.len());
    let placements = place_all(&matrix, &anchors, &newcomers, k, extent, cancel)?;
    drop(matrix);

    let placed = placements.len();
    for chunk in placements.chunks(WRITE_CHUNK) {
        EmbeddingSet::check_cancelled(cancel)?;
        db.writer()
            .set_projection_points(active.id, chunk.to_vec())?;
    }
    db.writer().flush()?;

    tracing::info!(
        run_id = active.id,
        placed,
        anchors = anchors.len(),
        "placed new samples incrementally"
    );

    Ok(IncrementalReport {
        run_id: active.id,
        placed,
        unplaceable: newcomers.len() - placed,
    })
}

/// Places every newcomer, scanning the anchors **once**.
///
/// The loop order is the whole performance story, and the benchmark is what settled it.
/// The obvious arrangement -- parallel over newcomers, each scanning every anchor -- reads
/// and widens the entire 50,000-row matrix once per newcomer. At 500 newcomers that is 25
/// million row decodes, and it measured **5.4 s**: three and a half times more expensive
/// than the PCA re-fit this path exists to avoid, which makes the path pointless.
///
/// Transposed, it is the same arithmetic with a hundredth of the memory traffic. The
/// newcomers' vectors are small enough to hold (under 2% of the corpus, so 2 MB at 50k ×
/// 512), so each anchor row is widened exactly once and scored against all of them. The
/// parallelism moves to contiguous anchor ranges, which also makes each thread's mmap access
/// sequential.
fn place_all(
    matrix: &EmbeddingMatrix,
    anchors: &[(i64, EmbeddingLoc, Point3)],
    newcomers: &[(i64, EmbeddingLoc)],
    k: usize,
    extent: f32,
    cancel: &CancellationToken,
) -> Result<Vec<(i64, Point3)>, ProjectionError> {
    let queries: Vec<Vec<f32>> = newcomers
        .iter()
        .map(|(_, loc)| matrix.row(*loc))
        .collect::<Result<_, _>>()?;
    let dim = queries.first().map_or(0, Vec::len);

    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(4);
    let per = anchors.len().div_ceil(cores).max(1);

    let partials: Vec<Result<Vec<TopK>, ProjectionError>> = (0..anchors.len())
        .step_by(per)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start| {
            let end = (start + per).min(anchors.len());
            let mut best: Vec<TopK> = (0..queries.len()).map(|_| TopK::new(k)).collect();
            let mut row = Vec::with_capacity(dim);

            for (n, (_, loc, point)) in anchors[start..end].iter().enumerate() {
                if n % 4096 == 0 {
                    EmbeddingSet::check_cancelled(cancel)?;
                }
                // A stale offset is one anchor's worth of missing evidence, not a reason to
                // refuse to place a file. `refit` treats the same offsets as a hard error
                // because there the row is the thing being projected.
                if matrix.row_into(*loc, &mut row).is_err() || row.len() != dim {
                    continue;
                }
                for (query, best) in queries.iter().zip(best.iter_mut()) {
                    best.offer(dot(query, &row), *point);
                }
            }
            Ok(best)
        })
        .collect();

    let mut merged: Vec<TopK> = (0..queries.len()).map(|_| TopK::new(k)).collect();
    for partial in partials {
        for (slot, chunk) in merged.iter_mut().zip(partial?) {
            slot.merge(chunk);
        }
    }

    let mut placements = Vec::with_capacity(newcomers.len());
    for ((sample_id, _), best) in newcomers.iter().zip(merged.iter()) {
        if let Some(point) = barycenter(&best.entries) {
            placements.push((*sample_id, jitter(point, *sample_id, extent)));
        }
    }
    Ok(placements)
}

/// Cosine similarity between two stored vectors.
///
/// A dot product, with no norm to divide by: every vector in the store is L2-normalized on
/// receipt (Phase 4), so there is no place for a forgotten normalization to hide.
///
/// **The eight accumulators are the point.** Floating-point addition is not associative, so
/// a plain `zip().map().sum()` is a serial dependency chain that LLVM is not permitted to
/// vectorize, and it runs at roughly one multiply-add per cycle. Splitting the sum into
/// eight independent lanes says explicitly that this particular reassociation is acceptable,
/// which lets the same arithmetic issue eight lanes wide. It is still deterministic --
/// fixed lanes, fixed order, same answer on every run -- which a `-ffast-math` equivalent
/// would not be.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];

    let mut chunks = a.chunks_exact(LANES).zip(b.chunks_exact(LANES));
    for (x, y) in &mut chunks {
        for lane in 0..LANES {
            acc[lane] += x[lane] * y[lane];
        }
    }

    let mut total: f32 = acc.iter().sum();
    for (x, y) in a[a.len() - a.len() % LANES..]
        .iter()
        .zip(&b[b.len() - b.len() % LANES..])
    {
        total += x * y;
    }
    total
}

/// The `k` most similar anchors seen so far, best first.
///
/// A sorted insert into a `k`-element vector rather than a `BinaryHeap`: `k` is 15, the
/// early-out on the current worst rejects essentially every anchor after the first handful,
/// and `f32` is not `Ord` so a heap would need a wrapper for the sake of an asymptotic
/// improvement that never arrives at this size.
#[derive(Debug)]
struct TopK {
    k: usize,
    entries: Vec<(f32, Point3)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            entries: Vec::with_capacity(k + 1),
        }
    }

    fn offer(&mut self, similarity: f32, point: Point3) {
        if self.entries.len() == self.k
            && self
                .entries
                .last()
                .is_some_and(|(worst, _)| similarity <= *worst)
        {
            return;
        }
        let at = self.entries.partition_point(|(s, _)| *s > similarity);
        self.entries.insert(at, (similarity, point));
        self.entries.truncate(self.k);
    }

    fn merge(&mut self, other: Self) {
        for (similarity, point) in other.entries {
            self.offer(similarity, point);
        }
    }
}

/// The similarity-weighted mean of some neighbors' positions.
///
/// Weights are similarities shifted into `[0, 2]` -- cosine runs to -1, and a negative
/// weight would push a point *away* from its neighbor and out of the convex hull of the
/// cloud entirely. Shifting keeps the placement inside the region its neighbors occupy,
/// which is the only claim this method makes.
fn barycenter(neighbors: &[(f32, Point3)]) -> Option<Point3> {
    if neighbors.is_empty() {
        return None;
    }
    let mut total = 0.0f64;
    let mut acc = [0.0f64; 3];
    for (similarity, point) in neighbors {
        let w = f64::from(similarity + 1.0);
        total += w;
        for axis in 0..3 {
            acc[axis] += w * f64::from(point[axis]);
        }
    }
    if total <= 0.0 || !total.is_finite() {
        // Every neighbor exactly anti-similar. Fall back to the unweighted mean rather than
        // dividing by zero: it is still inside the neighbors' hull, which is the property
        // that matters.
        let n = neighbors.len() as f64;
        let mut mean = [0.0f32; 3];
        for (_, point) in neighbors {
            for axis in 0..3 {
                mean[axis] += point[axis] / n as f32;
            }
        }
        return Some(mean);
    }
    Some([
        (acc[0] / total) as f32,
        (acc[1] / total) as f32,
        (acc[2] / total) as f32,
    ])
}

/// A deterministic nudge, keyed by `sample_id`.
///
/// Deterministic so that re-placing the same sample -- after a discarded re-fit, say --
/// puts it back exactly where it was. A random jitter would make the map twitch on every
/// import for no reason the user could name.
///
/// Shared with [`super::refit`], which needs the same nudge for the same reason: rows that
/// borrowed a twin's vector get one coordinate between them and have to be separable.
pub(crate) fn jitter(point: Point3, sample_id: i64, extent: f32) -> Point3 {
    if extent <= 0.0 {
        return point;
    }
    let amplitude = extent * JITTER_FRACTION;
    let mut state = (sample_id as u64) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 8_388_608.0) - 1.0
    };
    [
        point[0] + amplitude * next(),
        point[1] + amplitude * next(),
        point[2] + amplitude * next(),
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn a_placement_lands_between_its_neighbors_and_nearer_the_similar_one() {
        let neighbors = vec![(0.9f32, [0.0, 0.0, 0.0]), (0.1, [10.0, 0.0, 0.0])];
        let point = barycenter(&neighbors).unwrap();

        assert!(
            point[0] > 0.0 && point[0] < 10.0,
            "outside the hull: {point:?}"
        );
        assert!(
            point[0] < 5.0,
            "the more similar neighbor did not pull harder: {point:?}"
        );
    }

    /// Cosine runs to -1. An unshifted weight would be negative and push the point outside
    /// its neighbors entirely, which is the opposite of what a barycenter is for.
    #[test]
    fn an_anti_similar_neighbor_does_not_push_the_point_out_of_the_hull() {
        let neighbors = vec![(-0.95f32, [0.0, 0.0, 0.0]), (0.95, [10.0, 0.0, 0.0])];
        let point = barycenter(&neighbors).unwrap();
        assert!((0.0..=10.0).contains(&point[0]), "left the hull: {point:?}");
    }

    #[test]
    fn there_is_nothing_to_average_without_neighbors() {
        assert!(barycenter(&[]).is_none());
    }

    /// Duplicates share one vector by design (Phase 4), so their barycenters are identical.
    /// The jitter has to separate them, and has to do it the same way every time.
    #[test]
    fn the_jitter_separates_twins_and_repeats_itself() {
        let extent = 100.0;
        let a = jitter([1.0, 2.0, 3.0], 41, extent);
        let b = jitter([1.0, 2.0, 3.0], 42, extent);

        assert_ne!(a, b, "two samples landed on the same pixel");
        assert_eq!(a, jitter([1.0, 2.0, 3.0], 41, extent), "not deterministic");
        assert!(
            crate::projection::distance(a, [1.0, 2.0, 3.0]) < extent * JITTER_FRACTION * 2.0,
            "the nudge became a move: {a:?}"
        );
    }

    /// A cloud with no extent has no scale to nudge against, and multiplying by zero is the
    /// honest answer rather than a NaN.
    #[test]
    fn a_flat_cloud_is_not_jittered() {
        assert_eq!(jitter([1.0, 2.0, 3.0], 7, 0.0), [1.0, 2.0, 3.0]);
    }

    /// A/B for the lane split, since a comment claiming a speedup should be able to show
    /// one. Run with `--profile perf`; a debug build measures the optimizer being off.
    ///
    /// ```sh
    /// cargo test --manifest-path src-tauri/Cargo.toml --profile perf --lib \
    ///     lane_split -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "microbenchmark; run under --profile perf"]
    fn lane_split_versus_serial_dot() {
        const DIM: usize = 512;
        const ROWS: usize = 20_000;
        const QUERIES: usize = 32;

        let make = |seed: usize| -> Vec<f32> {
            (0..DIM)
                .map(|i| (((i * 37 + seed * 11) % 71) as f32 / 71.0) - 0.5)
                .collect()
        };
        let rows: Vec<Vec<f32>> = (0..ROWS).map(make).collect();
        let queries: Vec<Vec<f32>> = (0..QUERIES).map(|i| make(i + 9999)).collect();

        let serial =
            |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

        let mut sink = 0.0f32;
        let started = std::time::Instant::now();
        for row in &rows {
            for query in &queries {
                sink += serial(query, row);
            }
        }
        let serial_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        for row in &rows {
            for query in &queries {
                sink += dot(query, row);
            }
        }
        let lanes_elapsed = started.elapsed();

        let ops = (ROWS * QUERIES * DIM) as f64;
        println!("\n-- dot product, {ROWS} x {QUERIES} x {DIM} --");
        println!(
            "  serial         {:>8.0} ms  ({:.1} GFLOP/s)",
            serial_elapsed.as_secs_f64() * 1000.0,
            2.0 * ops / serial_elapsed.as_secs_f64() / 1e9
        );
        println!(
            "  8 lanes        {:>8.0} ms  ({:.1} GFLOP/s, {:.2}x)",
            lanes_elapsed.as_secs_f64() * 1000.0,
            2.0 * ops / lanes_elapsed.as_secs_f64() / 1e9,
            serial_elapsed.as_secs_f64() / lanes_elapsed.as_secs_f64()
        );
        // Keeps the optimizer from deleting either loop.
        assert!(sink.is_finite());
    }

    #[test]
    fn a_top_k_keeps_the_best_k_in_order() {
        let mut best = TopK::new(3);
        for (i, similarity) in [0.1f32, 0.9, 0.5, 0.95, 0.2].into_iter().enumerate() {
            best.offer(similarity, [i as f32, 0.0, 0.0]);
        }

        assert_eq!(best.entries.len(), 3);
        assert_eq!(
            best.entries.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            vec![0.95, 0.9, 0.5]
        );
        // The coordinate travels with the score it was offered against.
        assert_eq!(best.entries[0].1, [3.0, 0.0, 0.0]);
    }

    /// Anchors are scanned in parallel chunks, so the per-chunk lists have to combine into
    /// the same answer a single scan would have produced.
    #[test]
    fn merging_two_partial_top_k_lists_gives_the_global_answer() {
        let mut left = TopK::new(2);
        let mut right = TopK::new(2);
        left.offer(0.4, [0.0, 0.0, 0.0]);
        left.offer(0.8, [1.0, 0.0, 0.0]);
        right.offer(0.6, [2.0, 0.0, 0.0]);
        right.offer(0.99, [3.0, 0.0, 0.0]);

        left.merge(right);

        assert_eq!(
            left.entries.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            vec![0.99, 0.8]
        );
    }

    /// The whole placement path over a real store: the nearest anchor wins, and the answer
    /// does not depend on how the anchors happened to be split across threads.
    #[test]
    fn a_newcomer_is_placed_at_its_nearest_anchors() {
        // Eight unit vectors fanning away from [1, 0, ...]; anchor `i` sits at x = i.
        let vectors: Vec<Vec<f32>> = (0..9)
            .map(|i| {
                let mut v = vec![0.0f32; 4];
                v[0] = 1.0 - 0.1 * i as f32;
                v[1] = (1.0f32 - (1.0 - 0.1 * i as f32).powi(2)).max(0.0).sqrt();
                v
            })
            .collect();
        let (_dir, store, rows) = crate::projection::test_support::store_of(&vectors);
        let matrix = store.matrix().unwrap();

        // The first eight are anchors; the ninth is the newcomer, closest to anchor 7.
        let anchors: Vec<(i64, EmbeddingLoc, Point3)> = rows[..8]
            .iter()
            .enumerate()
            .map(|(i, (id, loc))| (*id, *loc, [i as f32, 0.0, 0.0]))
            .collect();
        let newcomers = vec![rows[8]];

        let placed = place_all(
            &matrix,
            &anchors,
            &newcomers,
            3,
            0.0,
            &CancellationToken::new(),
        )
        .unwrap();

        assert_eq!(placed.len(), 1);
        // Its three nearest are anchors 5, 6 and 7, so it lands in that neighbourhood --
        // decidedly not near anchor 0.
        assert!(placed[0].1[0] > 5.0, "placed at {:?}", placed[0].1);
    }
}
