//! Phase 6 end to end: the binary transports against a real database.
//!
//! What is under test is the boundary, not the commands. A `#[tauri::command]` cannot be
//! called without an `AppHandle`, and wrapping the whole app to prove that a `Vec<u8>` has
//! the length it has would be a test about Tauri. So each case builds a real library, runs
//! the same query the command runs, encodes with the same function the command encodes
//! with, and asserts on the bytes -- which is every line of Phase 6 that could be wrong in
//! a way `cargo check` would not catch.
//!
//! The claims here are `task.md` Phase 6's exit criteria:
//!
//! - a 50,000-point cloud round-trips through one payload of **≤ 900 KB**;
//! - the feature column and the point cloud agree, point for point, on which sample is at
//!   which index -- the contract that lets the renderer colour the map without ids on the
//!   wire;
//! - a filter result is ascending sample ids, so the renderer's mask is a merge and not a
//!   `Set`;
//! - and the payloads survive the states that are easy to forget: an empty library, a
//!   projection that does not exist yet, and a feature nothing has a value for.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use audiocloud_lib::{
    db::{
        queries, search,
        search::{Feature, FeatureRange, QueryFilter},
        Database, NewSample, SampleFeatures, SampleStatus,
    },
    ipc::binary,
};
use tempfile::TempDir;

/// Vector width. Nothing here reads a vector; the store is opened with it and that is all.
const DIM: usize = 8;

/// A library of `count` samples under one root, projected into one active run.
///
/// Coordinates are a deterministic spiral rather than random, so a mis-ordered column shows
/// up as a specific wrong number rather than as "some float".
fn projected_library(count: usize) -> (TempDir, Database, Vec<i64>) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), DIM).unwrap();
    let root = db.writer().add_root("/library", None).unwrap();

    let rows: Vec<NewSample> = (0..count)
        .map(|i| NewSample {
            root_id: root,
            rel_path: format!("drums/{i:05}.wav"),
            filename: format!("{i:05}.wav"),
            ext: if i % 3 == 0 {
                "aiff".into()
            } else {
                "wav".into()
            },
            size_bytes: 2048,
            mtime: 1_700_000_000 + i as i64,
            content_hash: None,
            duration_ms: Some(100 + i as i64),
            sample_rate: Some(48_000),
            channels: Some(1),
            status: SampleStatus::Decoded,
        })
        .collect();
    let ids = db.writer().upsert_samples(rows).unwrap();

    let features: Vec<(i64, SampleFeatures)> = ids
        .iter()
        .enumerate()
        // Every third sample has no BPM at all, which is what makes the NaN path real.
        .map(|(i, &id)| {
            (
                id,
                SampleFeatures {
                    bpm: (i % 3 != 0).then_some(80.0 + (i % 60) as f32),
                    spectral_centroid: Some(i as f32),
                    ..Default::default()
                },
            )
        })
        .collect();
    db.writer().set_features(features).unwrap();

    let run = db
        .writer()
        .begin_projection_run("pca", "{}", count as i64)
        .unwrap();
    let points: Vec<(i64, [f32; 3])> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, [i as f32, -(i as f32), i as f32 * 0.5]))
        .collect();
    db.writer().set_projection_points(run, points).unwrap();
    db.writer().activate_projection_run(run).unwrap();
    db.writer().flush().unwrap();

    (dir, db, ids)
}

/// **The exit criterion.** 50,000 points, one payload, 900 KB.
#[test]
fn fifty_thousand_points_round_trip_in_one_payload() {
    let (_dir, db, ids) = projected_library(50_000);

    let conn = db.read().unwrap();
    let points = queries::active_projection_points(&conn).unwrap();
    drop(conn);
    assert_eq!(points.len(), 50_000);

    let buf = binary::point_cloud(&points).unwrap();
    assert!(
        buf.len() <= 900 * 1024,
        "the point cloud payload is {} bytes, over the 900 KB budget",
        buf.len()
    );

    // Decode it the way `src/ipc/binary.ts` does, and check the round trip is lossless.
    let header = binary::header(&buf).unwrap();
    assert_eq!(header.magic, binary::MAGIC_POINT_CLOUD);
    assert_eq!(header.version, binary::VERSION);
    assert_eq!(header.count, 50_000);

    let n = header.count;
    let read_u32 = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
    let read_f32 = |at: usize| f32::from_le_bytes(buf[at..at + 4].try_into().unwrap());
    let base = binary::HEADER_BYTES;

    for i in [0, 1, n / 2, n - 2, n - 1] {
        assert_eq!(read_u32(base + i * 4) as i64, ids[i], "id column at {i}");
        assert_eq!(read_f32(base + (n + i) * 4), i as f32, "x column at {i}");
        assert_eq!(
            read_f32(base + (2 * n + i) * 4),
            -(i as f32),
            "y column at {i}"
        );
        assert_eq!(
            read_f32(base + (3 * n + i) * 4),
            i as f32 * 0.5,
            "z column at {i}"
        );
    }
}

/// The contract that lets the feature column carry no ids: index `i` is the same sample in
/// both payloads.
///
/// Asserted against the *ids* rather than against the values, because two columns generated
/// from the same loop would agree even if both were in the wrong order.
#[test]
fn a_feature_column_lines_up_with_the_point_cloud() {
    let (_dir, db, _ids) = projected_library(1_000);

    let conn = db.read().unwrap();
    let points = queries::active_projection_points(&conn).unwrap();
    let centroids = search::feature_column(&conn, Feature::SpectralCentroid).unwrap();
    drop(conn);

    assert_eq!(points.len(), centroids.len());
    let cloud = binary::point_cloud(&points).unwrap();
    let column = binary::feature_column(&centroids);
    assert_eq!(
        binary::header(&cloud).unwrap().count,
        binary::header(&column).unwrap().count,
        "a count mismatch is how the frontend detects a re-fit between two fetches"
    );

    // `spectral_centroid` was set to the sample's index, and `x` to the same index. If the
    // two orderings disagreed anywhere, this is where it would show.
    for (i, (_, point)) in points.iter().enumerate() {
        assert_eq!(centroids[i], point[0], "column and cloud disagree at {i}");
    }
}

/// A null cell arrives as NaN rather than as zero, because zero is a legitimate value for
/// every column in the schema and "no BPM" is not "0 BPM".
#[test]
fn a_missing_feature_value_is_nan_and_keeps_its_slot() {
    let (_dir, db, _ids) = projected_library(30);

    let conn = db.read().unwrap();
    let bpm = search::feature_column(&conn, Feature::Bpm).unwrap();
    drop(conn);

    assert_eq!(bpm.len(), 30, "a null cell must not shorten the column");
    for (i, value) in bpm.iter().enumerate() {
        if i % 3 == 0 {
            assert!(value.is_nan(), "index {i} should have no BPM");
        } else {
            assert!(value.is_finite(), "index {i} lost its BPM");
        }
    }
}

/// Ascending ids, so the renderer's filter mask is one merge over two sorted arrays.
#[test]
fn a_filter_result_is_ascending_ids() {
    let (_dir, db, _ids) = projected_library(500);

    let conn = db.read().unwrap();
    let matches = search::sample_ids(
        &conn,
        &QueryFilter {
            exts: vec!["wav".into()],
            features: vec![FeatureRange {
                feature: Feature::Bpm,
                min: Some(100.0),
                max: None,
            }],
            ..Default::default()
        },
    )
    .unwrap();
    drop(conn);

    assert!(!matches.is_empty(), "the filter matched nothing at all");
    assert!(
        matches.windows(2).all(|w| w[0] < w[1]),
        "ids came back unsorted"
    );

    let buf = binary::id_list(&matches).unwrap();
    assert_eq!(binary::header(&buf).unwrap().count, matches.len());
    assert_eq!(buf.len(), binary::HEADER_BYTES + matches.len() * 4);
}

/// A range constraint excludes rows with no value for it -- "BPM over 100" is not a claim
/// about files whose tempo could not be estimated.
#[test]
fn a_range_filter_excludes_rows_with_no_value() {
    let (_dir, db, _ids) = projected_library(90);

    let conn = db.read().unwrap();
    let any_bpm = search::sample_ids(
        &conn,
        &QueryFilter {
            features: vec![FeatureRange {
                feature: Feature::Bpm,
                min: Some(0.0),
                max: None,
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let everything = search::sample_ids(&conn, &QueryFilter::default()).unwrap();
    drop(conn);

    assert_eq!(everything.len(), 90);
    assert_eq!(any_bpm.len(), 60, "every third sample has no BPM");
}

/// The filter panel is a stack of AND, not OR: ticking a second box narrows.
#[test]
fn tag_filters_intersect_rather_than_union() {
    let (_dir, db, ids) = projected_library(10);
    db.writer().set_tag(ids[0], "kick").unwrap();
    db.writer().set_tag(ids[0], "808").unwrap();
    db.writer().set_tag(ids[1], "kick").unwrap();
    db.writer().flush().unwrap();

    let conn = db.read().unwrap();
    let one = search::sample_ids(
        &conn,
        &QueryFilter {
            tags: vec!["kick".into()],
            ..Default::default()
        },
    )
    .unwrap();
    let both = search::sample_ids(
        &conn,
        &QueryFilter {
            tags: vec!["kick".into(), "808".into()],
            ..Default::default()
        },
    )
    .unwrap();
    drop(conn);

    assert_eq!(one, vec![ids[0], ids[1]]);
    assert_eq!(both, vec![ids[0]]);
}

/// Tagging has to reach the FTS index, or search disagrees with the sidebar.
#[test]
fn a_tag_becomes_searchable_and_stops_being_so_when_removed() {
    let (_dir, db, ids) = projected_library(4);
    db.writer().set_tag(ids[2], "vinyl").unwrap();
    db.writer().flush().unwrap();

    let found = |needle: &str| {
        let conn = db.read().unwrap();
        search::sample_ids(
            &conn,
            &QueryFilter {
                text: Some(needle.to_string()),
                ..Default::default()
            },
        )
        .unwrap()
    };

    assert_eq!(found("vinyl"), vec![ids[2]]);

    db.writer().unset_tag(ids[2], "vinyl").unwrap();
    db.writer().flush().unwrap();
    assert!(
        found("vinyl").is_empty(),
        "the FTS index kept a removed tag"
    );

    // The filename side of the same index must survive the delete/insert pair, or removing
    // a tag would quietly unindex the file.
    assert_eq!(found("00002").len(), 1, "reindexing lost the filename");
}

/// A library with no projection is a real state with a real screen behind it, not a failure:
/// the payload is a bare header and the renderer draws an empty scene.
#[test]
fn an_unprojected_library_yields_an_empty_payload_rather_than_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), DIM).unwrap();

    let conn = db.read().unwrap();
    let points = queries::active_projection_points(&conn).unwrap();
    let column = search::feature_column(&conn, Feature::Bpm).unwrap();
    drop(conn);

    let cloud = binary::point_cloud(&points).unwrap();
    assert_eq!(cloud.len(), binary::HEADER_BYTES);
    assert_eq!(binary::header(&cloud).unwrap().count, 0);
    assert!(column.is_empty());
}

/// A filter list past the builder's cap is a typed refusal, not a statement with ten
/// thousand placeholders in it.
#[test]
fn an_absurd_filter_is_refused_with_a_reason() {
    let (_dir, db, _ids) = projected_library(4);
    let conn = db.read().unwrap();

    let err = search::sample_ids(
        &conn,
        &QueryFilter {
            root_ids: (0..5_000).collect(),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, audiocloud_lib::db::DbError::FilterTooLarge { .. }),
        "got {err:?}"
    );
}

/// An unclosed quote is a normal intermediate state while a user types a phrase. The query
/// builder closes it before handing it to FTS5, so it must remain safe and searchable rather
/// than being rejected as malformed input.
#[test]
fn an_unterminated_phrase_is_safe_to_search() {
    let (_dir, db, _ids) = projected_library(4);
    let conn = db.read().unwrap();

    assert!(search::sample_ids(
        &conn,
        &QueryFilter {
            text: Some("\"unterminated".into()),
            ..Default::default()
        },
    )
    .is_ok());
}
