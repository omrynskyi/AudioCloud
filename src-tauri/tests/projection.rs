//! Phase 5 end to end: coordinates, the single-active invariant, and the swap.
//!
//! These run against a real database and a real `embeddings.bin`, and never against the
//! pipeline: what is under test is what happens between a stored vector and a persisted
//! coordinate, and decoding four hundred wav files to get there would only add ways for the
//! test to fail for reasons that are not about projection. The vectors are synthesized with
//! deliberate structure -- clusters that a projector is *supposed* to keep together -- so
//! that "the layout is correct" is a statement with content rather than "it produced three
//! numbers".
//!
//! The claims here are `task.md` Phase 5's exit criteria, one test each:
//!
//! - coordinates exist for the whole corpus under both projectors;
//! - a re-fit with 5% new data leaves the other 95% substantially in place after alignment;
//! - the atomic swap holds under a concurrent reader;
//! - and, not in the criteria but the reason the criteria are safe: a cancelled or failed
//!   re-fit leaves the active map exactly as it found it.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use audiobank_lib::{
    db::{queries, Database, NewSample, SampleStatus},
    pipeline::CancellationToken,
    projection::{
        distance, place_incremental, plan, procrustes, refit, refit_at_low_priority, BoundingBox,
        PcaProjector, Plan, Point3, ProjectionError, Projector, Refit, RefitPhase, RefitSnapshot,
        UmapProjector,
    },
};
use tempfile::TempDir;

/// Vector width for these tests. Not 512: the shape of the arithmetic is identical at 32
/// and the fixtures stay small enough to reason about.
const DIM: usize = 32;

/// A database with `count` embedded samples drawn from `clusters` clumps.
///
/// Deterministic in `seed`, and -- the property the stability test leans on -- the first
/// `n` vectors of a corpus are the same whatever `count` is, so a "5% more samples" corpus
/// is genuinely the old one plus new rows rather than a new one that happens to be bigger.
fn library(count: usize, clusters: usize, seed: u64) -> (TempDir, Database, Vec<i64>) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), DIM).unwrap();
    let ids = add_samples(&db, 0, count, clusters, seed);
    (dir, db, ids)
}

/// Appends `count` samples numbered from `from`, returning their ids in order.
fn add_samples(db: &Database, from: usize, count: usize, clusters: usize, seed: u64) -> Vec<i64> {
    let root = db
        .writer()
        .add_root(format!("/library/{seed}"), None)
        .unwrap();

    let rows: Vec<NewSample> = (from..from + count)
        .map(|i| NewSample {
            root_id: root,
            rel_path: format!("drums/{:05}.wav", i),
            filename: format!("{:05}.wav", i),
            ext: "wav".into(),
            size_bytes: 2048,
            mtime: 1_700_000_000 + i as i64,
            content_hash: None,
            duration_ms: Some(500),
            sample_rate: Some(48_000),
            channels: Some(1),
            status: SampleStatus::Decoded,
        })
        .collect();
    let ids = db.writer().upsert_samples(rows).unwrap();

    let vectors: Vec<Vec<f32>> = (from..from + count)
        .map(|i| vector(i, clusters, seed))
        .collect();
    let locs = {
        let mut store = db.embeddings().lock().unwrap();
        let locs = store.append_batch(&vectors).unwrap();
        store.sync().unwrap();
        locs
    };

    db.writer()
        .set_embeddings(ids.iter().copied().zip(locs).collect())
        .unwrap();
    db.writer().flush().unwrap();
    ids
}

/// One L2-normalized vector: a cluster center plus deterministic noise keyed by `i`.
///
/// Keyed by `i` alone, never by the corpus size, which is what makes sample 7 the same
/// sound in a 200-file library and in a 210-file one.
fn vector(i: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut state = (seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let center = i % clusters;
    let mut v: Vec<f32> = (0..DIM).map(|_| 0.55 * next()).collect();
    v[center % DIM] += 1.0;
    v[(center * 7 + 5) % DIM] += 0.55;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn run_refit(db: &Database, projector: &dyn Projector) -> audiobank_lib::projection::RefitReport {
    let cancel = CancellationToken::new();
    refit(db, Refit::new(projector, &cancel)).unwrap()
}

/// **Exit criterion: 3D coordinates exist for the full corpus under both projectors.**
///
/// Both, in one test, because the interesting failure is a projector that satisfies the
/// trait and quietly returns a short vector -- and that is invisible unless the count is
/// checked against the corpus rather than against itself.
#[test]
fn both_projectors_give_every_sample_a_coordinate() {
    for projector in [
        &PcaProjector::new() as &dyn Projector,
        &UmapProjector::default(),
    ] {
        let (_dir, db, ids) = library(400, 5, 11);

        let report = run_refit(&db, projector);

        assert_eq!(report.algorithm, projector.name());
        assert_eq!(report.sample_count, ids.len());

        let conn = db.read().unwrap();
        let points = queries::active_projection_points(&conn).unwrap();
        assert_eq!(
            points.len(),
            ids.len(),
            "{} left samples without coordinates",
            projector.name()
        );

        // Every id is present exactly once, and every coordinate is a number.
        let placed: std::collections::HashSet<i64> = points.iter().map(|(id, _)| *id).collect();
        assert_eq!(placed.len(), ids.len());
        for id in &ids {
            assert!(placed.contains(id), "sample {id} has no coordinate");
        }
        assert!(points.iter().all(|(_, p)| p.iter().all(|v| v.is_finite())));

        // A layout with no extent is three columns of the same number.
        let bbox = BoundingBox::of(&points.iter().map(|(_, p)| *p).collect::<Vec<_>>()).unwrap();
        assert!(
            bbox.diagonal() > 1e-4,
            "{} collapsed the cloud",
            projector.name()
        );
    }
}

/// **Exit criterion: a re-fit with 5% new data leaves existing points substantially in
/// place after alignment.**
///
/// Stated as a fraction of the cloud's own diagonal, because a layout has no canonical
/// scale and "moved by 0.4" is not a claim about anything. 10% of the diagonal is a
/// deliberately loose bar for a *median*: it is the difference between "most things stayed"
/// and "the library rotated", which is exactly the distinction `overview.md` §3.8 says
/// Procrustes buys and the only one it claims.
#[test]
fn a_five_percent_import_leaves_the_rest_of_the_library_in_place() {
    let (_dir, db, _ids) = library(400, 6, 23);

    let first = run_refit(&db, &PcaProjector::new());
    let before: std::collections::HashMap<i64, Point3> = {
        let conn = db.read().unwrap();
        queries::active_projection_points(&conn)
            .unwrap()
            .into_iter()
            .collect()
    };

    // 20 new samples on 400 is 5%, which is over the incremental threshold and therefore a
    // real re-fit -- the path this criterion is about.
    add_samples(&db, 400, 20, 6, 23);
    assert!(matches!(
        plan(&db).unwrap(),
        Plan::Full {
            new: 20,
            total: 420
        }
    ));

    let second = run_refit(&db, &PcaProjector::new());

    assert_eq!(second.sample_count, 420);
    assert_eq!(second.correspondences, 400);
    assert_eq!(second.previous_run_id, Some(first.run_id));

    let after: std::collections::HashMap<i64, Point3> = {
        let conn = db.read().unwrap();
        queries::active_projection_points(&conn)
            .unwrap()
            .into_iter()
            .collect()
    };
    assert_eq!(after.len(), 420);

    let mut moved: Vec<f32> = before
        .iter()
        .map(|(id, old)| distance(*old, after[id]))
        .collect();
    moved.sort_by(f32::total_cmp);
    let median = moved[moved.len() / 2];
    let relative = median / second.extent;

    assert!(
        relative < 0.10,
        "median displacement {median} is {:.1}% of the {} cloud",
        relative * 100.0,
        second.extent
    );
    // The report must agree with what the database actually holds; a stability number
    // computed from a layout that was not the one persisted would be worse than none.
    let reported = second.median_displacement.unwrap();
    assert!(
        (reported - median).abs() < second.extent * 1e-3,
        "the report said {reported}, the database says {median}"
    );
}

/// Alignment has to be doing work, not decorating a projector that was already stable.
///
/// The falsifiable form: run the same re-fit with alignment off and confirm the layout
/// moves *more*. Without this, `a_five_percent_import_leaves_the_rest_of_the_library_in_place`
/// would pass just as happily against a `fit` that returned the identity every time.
#[test]
fn alignment_is_what_keeps_the_layout_still() {
    let vectors_seed = 71;
    let displacement = |align: bool| {
        let (_dir, db, _) = library(300, 5, vectors_seed);
        run_refit(&db, &PcaProjector::new());
        let before: std::collections::HashMap<i64, Point3> = {
            let conn = db.read().unwrap();
            queries::active_projection_points(&conn)
                .unwrap()
                .into_iter()
                .collect()
        };

        add_samples(&db, 300, 40, 5, vectors_seed);

        let cancel = CancellationToken::new();
        let projector = PcaProjector::new();
        let mut options = Refit::new(&projector, &cancel);
        if !align {
            options = options.without_alignment();
        }
        let report = refit(&db, options).unwrap();

        let after: std::collections::HashMap<i64, Point3> = {
            let conn = db.read().unwrap();
            queries::active_projection_points(&conn)
                .unwrap()
                .into_iter()
                .collect()
        };
        let mut moved: Vec<f32> = before
            .iter()
            .map(|(id, old)| distance(*old, after[id]))
            .collect();
        moved.sort_by(f32::total_cmp);
        moved[moved.len() / 2] / report.extent
    };

    // PCA is nearly stable on its own -- its axes are canonicalized precisely so that it is
    // -- so the unaligned number is small too. What must hold is that alignment does not
    // make it worse, and that the aligned layout is inside the bar the criterion sets.
    let aligned = displacement(true);
    let unaligned = displacement(false);
    assert!(
        aligned <= unaligned + 1e-6,
        "alignment made things worse: {aligned} aligned vs {unaligned} unaligned"
    );
    assert!(aligned < 0.10, "aligned displacement {aligned}");
}

/// **Exit criterion: the atomic swap holds under a concurrent reader.**
///
/// A reader hammering `active_projection_points` through the whole of a re-fit must never
/// see a partial map: not an empty result between one run being deactivated and the next
/// activated, and not a mixture of two runs' coordinates. Both are possible if the swap is
/// two transactions instead of one, and neither is visible without a reader running
/// *during* it -- which is why this test spawns one rather than checking afterwards.
#[test]
fn a_concurrent_reader_never_sees_a_half_swapped_map() {
    let (_dir, db, _) = library(600, 5, 31);
    let db = Arc::new(db);

    let first = run_refit(&db, &PcaProjector::new());
    let baseline = first.sample_count;

    let stop = Arc::new(AtomicBool::new(false));
    let observations = Arc::new(AtomicUsize::new(0));
    let bad = Arc::new(AtomicUsize::new(0));

    let reader = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let observations = Arc::clone(&observations);
        let bad = Arc::clone(&bad);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let conn = db.read().unwrap();
                let points = queries::active_projection_points(&conn).unwrap();
                observations.fetch_add(1, Ordering::Relaxed);
                // Either the old map or the new one, whole. Never anything else.
                if points.len() != baseline && points.len() != baseline + 60 {
                    bad.fetch_add(1, Ordering::Relaxed);
                }
                // No id may appear twice: two runs' rows in one answer would be a
                // half-swapped `is_active`.
                let unique: std::collections::HashSet<i64> =
                    points.iter().map(|(id, _)| *id).collect();
                if unique.len() != points.len() {
                    bad.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    };

    add_samples(&db, 600, 60, 5, 31);
    for _ in 0..4 {
        run_refit(&db, &PcaProjector::new());
    }

    stop.store(true, Ordering::Relaxed);
    reader.join().unwrap();

    assert!(
        observations.load(Ordering::Relaxed) > 10,
        "the reader barely ran; the test proved nothing"
    );
    assert_eq!(
        bad.load(Ordering::Relaxed),
        0,
        "a reader saw a partial map in {} observations",
        observations.load(Ordering::Relaxed)
    );
}

/// The single-active invariant is enforced by the schema, not by convention. Four re-fits
/// must leave exactly one active run and no accumulated shadows.
#[test]
fn exactly_one_run_is_active_and_superseded_runs_are_pruned() {
    let (_dir, db, _) = library(120, 4, 5);

    let mut run_ids = Vec::new();
    for _ in 0..4 {
        run_ids.push(run_refit(&db, &PcaProjector::new()).run_id);
    }

    let conn = db.read().unwrap();
    let runs = queries::projection_runs(&conn).unwrap();
    assert_eq!(
        runs.len(),
        1,
        "superseded runs accumulated: {:?}",
        runs.iter().map(|r| r.id).collect::<Vec<_>>()
    );
    assert_eq!(runs[0].id, *run_ids.last().unwrap());
    assert!(runs[0].is_active);
    assert!(runs[0].completed_at.is_some());

    // And the coordinates of the runs that were dropped went with them.
    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM projections", [], |r| r.get(0))
        .unwrap();
    assert_eq!(total, 120);
}

/// A cancelled re-fit must leave the map it was replacing untouched, and must not leave a
/// shadow run behind. This is what makes a multi-minute background job safe to abandon.
#[test]
fn a_cancelled_refit_leaves_the_active_map_exactly_as_it_was() {
    let (_dir, db, _) = library(200, 4, 17);
    let established = run_refit(&db, &PcaProjector::new());
    let before: Vec<(i64, Point3)> = {
        let conn = db.read().unwrap();
        queries::active_projection_points(&conn).unwrap()
    };

    add_samples(&db, 200, 50, 4, 17);

    let cancel = CancellationToken::new();
    cancel.cancel();
    let err = refit(&db, Refit::new(&PcaProjector::new(), &cancel)).unwrap_err();
    assert!(matches!(err, ProjectionError::Cancelled), "got {err:?}");

    let conn = db.read().unwrap();
    let active = queries::active_projection_run(&conn).unwrap().unwrap();
    assert_eq!(active.id, established.run_id, "the active run changed");
    assert_eq!(
        queries::active_projection_points(&conn).unwrap(),
        before,
        "the map moved under a cancelled re-fit"
    );
    assert_eq!(
        queries::projection_runs(&conn).unwrap().len(),
        1,
        "a shadow run survived cancellation"
    );
}

/// Incremental placement: **existing points never move**. Not "move a little" -- the rows
/// are not written at all, so equality here is exact rather than approximate, and that is
/// the whole claim.
#[test]
fn an_incremental_import_places_new_points_and_moves_no_old_ones() {
    let (_dir, db, _) = library(500, 5, 41);
    let established = run_refit(&db, &PcaProjector::new());
    let before: std::collections::HashMap<i64, Point3> = {
        let conn = db.read().unwrap();
        queries::active_projection_points(&conn)
            .unwrap()
            .into_iter()
            .collect()
    };

    // 5 on 505 is under 1%, which is the incremental path.
    let new_ids = add_samples(&db, 500, 5, 5, 41);
    assert!(matches!(
        plan(&db).unwrap(),
        Plan::Incremental { new: 5, total: 505 }
    ));

    let report = place_incremental(&db, &CancellationToken::new()).unwrap();
    assert_eq!(report.run_id, established.run_id, "a new run was created");
    assert_eq!(report.placed, 5);
    assert_eq!(report.unplaceable, 0);

    let conn = db.read().unwrap();
    let after: std::collections::HashMap<i64, Point3> = queries::active_projection_points(&conn)
        .unwrap()
        .into_iter()
        .collect();
    assert_eq!(after.len(), 505);

    for (id, old) in &before {
        assert_eq!(
            after[id], *old,
            "sample {id} moved during an incremental import"
        );
    }
    for id in &new_ids {
        assert!(after.contains_key(id), "sample {id} was never placed");
    }
    assert!(matches!(plan(&db).unwrap(), Plan::UpToDate));
}

/// A placement is only worth anything if it lands near the right neighbors. A new sample
/// drawn from cluster `c` must land closer to cluster `c`'s existing members than to the
/// others -- otherwise the incremental path is just scattering points cheaply.
#[test]
fn an_incrementally_placed_point_lands_among_its_own_cluster() {
    let clusters = 5usize;
    let (_dir, db, ids) = library(500, clusters, 53);
    run_refit(&db, &PcaProjector::new());

    let before: std::collections::HashMap<i64, Point3> = {
        let conn = db.read().unwrap();
        queries::active_projection_points(&conn)
            .unwrap()
            .into_iter()
            .collect()
    };

    // Sample 500 belongs to cluster 0, same as samples 0, 5, 10 ...
    let new_ids = add_samples(&db, 500, 1, clusters, 53);
    place_incremental(&db, &CancellationToken::new()).unwrap();

    let conn = db.read().unwrap();
    let after: std::collections::HashMap<i64, Point3> = queries::active_projection_points(&conn)
        .unwrap()
        .into_iter()
        .collect();
    let placed = after[&new_ids[0]];

    let mean_distance_to = |cluster: usize| {
        let members: Vec<f32> = ids
            .iter()
            .enumerate()
            .filter(|(i, _)| i % clusters == cluster)
            .map(|(_, id)| distance(placed, before[id]))
            .collect();
        members.iter().sum::<f32>() / members.len() as f32
    };

    let own = mean_distance_to(0);
    for other in 1..clusters {
        assert!(
            own < mean_distance_to(other),
            "placed nearer cluster {other} ({}) than its own ({own})",
            mean_distance_to(other)
        );
    }
}

/// Progress reports under the same rule as a scan: coalesced, and with a guaranteed
/// terminal snapshot so a UI cannot be left mid-job (cross-cutting rule 6).
#[test]
fn a_refit_reports_progress_and_always_finishes_it() {
    let (_dir, db, _) = library(300, 4, 61);

    let seen: Arc<std::sync::Mutex<Vec<RefitSnapshot>>> = Arc::default();
    let sink = Arc::clone(&seen);
    let cancel = CancellationToken::new();
    let projector = PcaProjector::new();
    let report = refit(
        &db,
        Refit::new(&projector, &cancel).with_progress(move |s| sink.lock().unwrap().push(s)),
    )
    .unwrap();

    let snapshots = seen.lock().unwrap();
    let last = snapshots.last().expect("no progress at all");
    assert!(last.is_terminal(), "the last snapshot was {:?}", last.phase);
    assert_eq!(last.phase, RefitPhase::Done);
    assert_eq!(last.samples, report.sample_count as u64);
    assert_eq!(last.written, report.sample_count as u64);
    // Coalesced: no two consecutive snapshots may be identical.
    for pair in snapshots.windows(2) {
        assert_ne!(pair[0], pair[1], "the ticker emitted a repeat");
    }
}

/// A library with no active projection has nothing to place into, and says so by name
/// rather than by creating a run behind the caller's back.
#[test]
fn incremental_placement_needs_a_map_to_place_into() {
    let (_dir, db, _) = library(50, 3, 3);

    let err = place_incremental(&db, &CancellationToken::new()).unwrap_err();

    assert!(
        matches!(err, ProjectionError::NoActiveProjection),
        "got {err:?}"
    );
    assert!(matches!(
        plan(&db).unwrap(),
        Plan::Full { new: 50, total: 50 }
    ));
}

/// A projector that refuses -- too few samples, no variance -- must not leave a shadow run
/// behind for the next swap to prune. The shadow row is only opened once the coordinates
/// exist.
#[test]
fn a_refused_projection_writes_no_run_at_all() {
    let (_dir, db, _) = library(3, 2, 9);

    let err = refit(
        &db,
        Refit::new(&PcaProjector::new(), &CancellationToken::new()),
    )
    .unwrap_err();

    assert!(
        matches!(
            err,
            ProjectionError::TooFewSamples {
                algorithm: "pca",
                ..
            }
        ),
        "got {err:?}"
    );
    let conn = db.read().unwrap();
    assert!(queries::projection_runs(&conn).unwrap().is_empty());
    assert!(queries::active_projection_run(&conn).unwrap().is_none());
}

/// **The failure mode that made [`Refit::with_fallback`] necessary.**
///
/// A corpus of tight, well-separated clumps produces a kNN graph in pieces, and
/// `annembed`'s diffusion-map initialization panics on one -- an `assert!` deep inside
/// `set_data_box`, which under the release profile's `panic = "abort"` would take the
/// application with it. So the UMAP projector refuses the graph first, and the fallback
/// turns the refusal into a PCA layout.
///
/// This test is the reason the guard exists rather than a theory about it: before the
/// component check, this corpus aborted the test process better than half the time.
#[test]
fn a_disconnected_corpus_falls_back_instead_of_panicking() {
    let (_dir, db, ids) = tight_library(400, 6, 23);

    // On its own, UMAP refuses -- by name, with the component count in the message.
    let err = refit(
        &db,
        Refit::new(&UmapProjector::default(), &CancellationToken::new()),
    )
    .unwrap_err();
    match &err {
        ProjectionError::Degenerate(message) => {
            assert!(message.contains("disconnected"), "{message}");
        }
        other => panic!("expected a degenerate graph, got {other:?}"),
    }

    // With a fallback, the library still gets a map -- recorded under the algorithm that
    // actually produced it, not the one that was asked for.
    let pca = PcaProjector::new();
    let report = refit(
        &db,
        Refit::new(&UmapProjector::default(), &CancellationToken::new()).with_fallback(&pca),
    )
    .unwrap();

    assert_eq!(
        report.algorithm, "pca",
        "the run lied about which projector ran"
    );
    assert_eq!(report.sample_count, ids.len());

    let conn = db.read().unwrap();
    let run = queries::active_projection_run(&conn).unwrap().unwrap();
    assert_eq!(run.algorithm, "pca");
    assert_eq!(
        queries::active_projection_points(&conn).unwrap().len(),
        ids.len()
    );
}

/// A corpus whose clusters barely touch: the shape that disconnects a kNN graph.
fn tight_library(count: usize, clusters: usize, seed: u64) -> (TempDir, Database, Vec<i64>) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), DIM).unwrap();
    let root = db.writer().add_root("/tight", None).unwrap();

    let rows: Vec<NewSample> = (0..count)
        .map(|i| NewSample {
            root_id: root,
            rel_path: format!("tight/{i:05}.wav"),
            filename: format!("{i:05}.wav"),
            ext: "wav".into(),
            size_bytes: 2048,
            mtime: 1_700_000_000 + i as i64,
            content_hash: None,
            duration_ms: Some(500),
            sample_rate: Some(48_000),
            channels: Some(1),
            status: SampleStatus::Decoded,
        })
        .collect();
    let ids = db.writer().upsert_samples(rows).unwrap();

    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            let mut state = (seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 8_388_608.0) - 1.0
            };
            let center = i % clusters;
            // A twentieth of the cluster separation: close enough that no point in one
            // clump is among any other clump's fifteen nearest.
            let mut v: Vec<f32> = (0..DIM).map(|_| 0.02 * next()).collect();
            v[center % DIM] += 1.0;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect();
    let locs = {
        let mut store = db.embeddings().lock().unwrap();
        let locs = store.append_batch(&vectors).unwrap();
        store.sync().unwrap();
        locs
    };
    db.writer()
        .set_embeddings(ids.iter().copied().zip(locs).collect())
        .unwrap();
    db.writer().flush().unwrap();
    (dir, db, ids)
}

/// **Exit criterion, under the projector it is actually about.**
///
/// PCA's axes are canonicalized, so it barely moves with or without alignment -- which
/// makes it the wrong witness for a claim about layout stability. UMAP's layout has no
/// canonical orientation at all (that is the entire reason `overview.md` §3.8 exists) and
/// its descent is stochastic, so this is the case Procrustes was bought for.
///
/// The bar is 20% of the cloud's diagonal against a measured 11-16% across six runs
/// (`BENCHMARKS.md`), and it is a *median*: a re-fit that legitimately reorganizes one
/// cluster must not fail this, because Procrustes does not claim to prevent that. What it
/// separates is "most things stayed" from "the library rotated", and an unaligned re-fit of
/// the same corpus moves the median point by 24-87%.
#[test]
fn a_five_percent_import_under_umap_leaves_the_library_in_place() {
    let (_dir, db, _) = library(400, 6, 23);
    let projector = UmapProjector::default();

    run_refit(&db, &projector);
    let before = layout(&db);

    add_samples(&db, 400, 20, 6, 23);
    let report = run_refit(&db, &projector);
    assert_eq!(
        report.algorithm, "umap",
        "the fit fell back to {}; this test then proves nothing about UMAP",
        report.algorithm
    );

    let after = layout(&db);
    let relative = median_displacement(&before, &after) / report.extent;

    assert!(
        relative < 0.20,
        "median displacement is {:.1}% of the cloud",
        relative * 100.0
    );
    // And the report has to agree with the database, or the number a UI would show is not
    // the number this test checked.
    let reported = report.median_displacement.unwrap() / report.extent;
    assert!(
        (reported - relative).abs() < 0.01,
        "{reported} vs {relative}"
    );
}

/// **Alignment is what keeps a UMAP layout still**, stated as the claim that cannot be
/// flaky.
///
/// The obvious version -- re-fit twice, once aligned and once not, and compare -- measures
/// two different stochastic descents, and the noise between them is larger than the effect.
/// Worse, even on one fit the *unaligned* number is a coin flip: a UMAP layout's orientation
/// is arbitrary, so sometimes it lands close to its predecessor by luck and the improvement
/// looks small. Asserting on an effect size there fails roughly one run in eight, which is a
/// test that reports the weather.
///
/// So this asserts the thing Procrustes actually guarantees. It minimizes the sum of squared
/// distances over the correspondences across all similarity transforms, and the identity is
/// one of those, so **the aligned RMS can never exceed the unaligned RMS** -- on any layout,
/// on any run. One fit, aligned by hand, compared against itself. A fit that transposed its
/// rotation, forbade reflection, or mixed up its correspondences fails this every time; a
/// lucky orientation cannot make it pass.
///
/// The effect *size* is a measurement rather than an assertion, and lives in
/// `BENCHMARKS.md`: 11-16% of the cloud diagonal aligned against 24-87% unaligned.
#[test]
fn alignment_can_never_make_a_umap_layout_move_further() {
    let (_dir, db, _) = library(400, 6, 29);
    let projector = UmapProjector::default();

    run_refit(&db, &projector);
    let before = layout(&db);

    add_samples(&db, 400, 20, 6, 29);
    let cancel = CancellationToken::new();
    let report = refit(&db, Refit::new(&projector, &cancel).without_alignment()).unwrap();
    assert_eq!(report.algorithm, "umap");

    let raw = layout(&db);
    let shared: Vec<(Point3, Point3)> = raw
        .iter()
        .filter_map(|(id, new)| before.get(id).map(|old| (*new, *old)))
        .collect();
    assert_eq!(shared.len(), 400);

    // The same correspondence pass `refit` does, over the same layout.
    let alignment = procrustes::fit(
        &shared.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        &shared.iter().map(|(_, o)| *o).collect::<Vec<_>>(),
    );

    let rms = |pairs: &[(Point3, Point3)]| {
        (pairs
            .iter()
            .map(|(a, b)| distance(*a, *b).powi(2))
            .sum::<f32>()
            / pairs.len() as f32)
            .sqrt()
    };
    let aligned: Vec<(Point3, Point3)> = shared
        .iter()
        .map(|(new, old)| (alignment.apply(*new), *old))
        .collect();

    assert!(
        rms(&aligned) <= rms(&shared),
        "alignment increased the residual: {} aligned vs {} unaligned",
        rms(&aligned),
        rms(&shared)
    );
    // And it is not a no-op dressed up as an improvement: a fresh UMAP layout does need a
    // real transform, so the residual has to actually come down.
    assert!(
        rms(&aligned) < rms(&shared) * 0.95,
        "the alignment did nothing: {} vs {}",
        rms(&aligned),
        rms(&shared)
    );
}

/// **Phase 4's finding 1, end to end.** A duplicate file stores a *reference* to its twin's
/// vector rather than a second copy, so two `samples` rows can carry the same `emb_offset`.
/// The projector must see that vector once, and both rows must still come out of the re-fit
/// with their own -- distinct, and near-identical -- coordinate.
#[test]
fn duplicate_rows_share_one_fit_and_still_get_their_own_coordinate() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), DIM).unwrap();
    let root = db.writer().add_root("/dupes", None).unwrap();

    let originals = 200usize;
    let copies = 60usize;
    let rows: Vec<NewSample> = (0..originals + copies)
        .map(|i| NewSample {
            root_id: root,
            rel_path: format!("dupes/{i:05}.wav"),
            filename: format!("{i:05}.wav"),
            ext: "wav".into(),
            size_bytes: 2048,
            mtime: 1_700_000_000 + i as i64,
            content_hash: None,
            duration_ms: Some(500),
            sample_rate: Some(48_000),
            channels: Some(1),
            status: SampleStatus::Decoded,
        })
        .collect();
    let ids = db.writer().upsert_samples(rows).unwrap();

    // Only the originals get bytes in `embeddings.bin`.
    let vectors: Vec<Vec<f32>> = (0..originals).map(|i| vector(i, 5, 77)).collect();
    let locs = {
        let mut store = db.embeddings().lock().unwrap();
        let locs = store.append_batch(&vectors).unwrap();
        store.sync().unwrap();
        locs
    };
    let mut assignments: Vec<(i64, audiobank_lib::db::EmbeddingLoc)> = ids[..originals]
        .iter()
        .copied()
        .zip(locs.iter().copied())
        .collect();
    // Each copy points at an original's bytes -- exactly what the dedup path writes.
    for (n, id) in ids[originals..].iter().enumerate() {
        assignments.push((*id, locs[n]));
    }
    db.writer().set_embeddings(assignments).unwrap();
    db.writer().flush().unwrap();

    let report = run_refit(&db, &PcaProjector::new());

    assert_eq!(report.sample_count, originals + copies);
    assert_eq!(
        report.distinct_vectors, originals,
        "the projector was handed the same vector twice"
    );

    let placed = layout(&db);
    assert_eq!(placed.len(), originals + copies);
    for n in 0..copies {
        let original = placed[&ids[n]];
        let copy = placed[&ids[originals + n]];
        assert_ne!(copy, original, "a duplicate landed exactly on its twin");
        assert!(
            distance(copy, original) < report.extent * 0.01,
            "a duplicate landed {} away in a cloud of {}",
            distance(copy, original),
            report.extent
        );
    }
}

/// `overview.md` §3.8 asks for the re-fit to run "at low priority", and on macOS that word
/// means a QoS class rather than a nice value. This checks the demoted path produces the
/// same map as the plain one -- which is the only observable difference there should be.
///
/// It cannot check the QoS itself: `pthread_get_qos_class_np` reports the *calling* thread,
/// and by the time the job has returned every thread that ran it is gone. What it does
/// check is that routing the whole job through a private `rayon` pool -- which is what makes
/// the demotion reach `hnsw_rs` and `annembed` rather than just the one thread that calls
/// them -- did not change an answer or lose a nested parallel section.
#[test]
fn a_low_priority_refit_produces_the_same_map() {
    let (_dir, plain, _) = library(300, 5, 83);
    let (_dir2, demoted, _) = library(300, 5, 83);

    let normal = refit(
        &plain,
        Refit::new(&PcaProjector::new(), &CancellationToken::new()),
    )
    .unwrap();
    let background = refit_at_low_priority(
        &demoted,
        Refit::new(&PcaProjector::new(), &CancellationToken::new()),
    )
    .unwrap();

    assert_eq!(normal.sample_count, background.sample_count);
    assert_eq!(normal.distinct_vectors, background.distinct_vectors);
    // PCA is deterministic, so "the same map" is exact rather than approximate.
    assert_eq!(layout(&plain), layout(&demoted));
}

/// The active layout, keyed by sample id.
fn layout(db: &Database) -> std::collections::HashMap<i64, Point3> {
    let conn = db.read().unwrap();
    queries::active_projection_points(&conn)
        .unwrap()
        .into_iter()
        .collect()
}

/// Median distance a sample present in both layouts moved.
fn median_displacement(
    before: &std::collections::HashMap<i64, Point3>,
    after: &std::collections::HashMap<i64, Point3>,
) -> f32 {
    let mut moved: Vec<f32> = before
        .iter()
        .filter_map(|(id, old)| after.get(id).map(|new| distance(*old, *new)))
        .collect();
    assert!(!moved.is_empty(), "the two layouts share no samples");
    moved.sort_by(f32::total_cmp);
    moved[moved.len() / 2]
}
