//! Read queries against the pool -- sample lookup, feature columns, FTS5 search,
//! nearest-neighbor fetch.
//!
//! Every function here takes a `&Connection` rather than reaching for the pool itself, so
//! the caller decides whether it is spending a pooled read connection or running inside the
//! writer's transaction. They are also, for the same reason, trivially unit-testable.
//!
//! Extended through Phases 6-9.

use std::collections::HashMap;

use rusqlite::Connection;

use super::{DbError, EmbeddingLoc, SampleFeatures, SampleStatus};

/// A library root as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryRoot {
    pub id: i64,
    pub path: String,
    pub label: Option<String>,
    pub enabled: bool,
    pub added_at: i64,
    pub last_scan_id: Option<i64>,
}

/// The subset of a sample row the discovery stage needs to decide "unchanged?" without
/// hashing the file (`overview.md` §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleStamp {
    pub id: i64,
    pub mtime: i64,
    pub size_bytes: i64,
    pub status: SampleStatus,
}

impl SampleStamp {
    /// Whether a file on disk matches this row cheaply enough to skip re-processing.
    pub fn unchanged(&self, mtime: i64, size_bytes: i64) -> bool {
        self.mtime == mtime && self.size_bytes == size_bytes
    }
}

/// Total sample rows, across every root.
pub fn count_samples(conn: &Connection) -> Result<i64, DbError> {
    Ok(conn.query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))?)
}

/// Sample rows in a given lifecycle state.
pub fn count_samples_with_status(conn: &Connection, status: SampleStatus) -> Result<i64, DbError> {
    Ok(conn.query_row(
        "SELECT COUNT(*) FROM samples WHERE status = ?1",
        [status.as_str()],
        |r| r.get(0),
    )?)
}

/// Every root the user has added, oldest first.
pub fn library_roots(conn: &Connection) -> Result<Vec<LibraryRoot>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, path, label, enabled, added_at, last_scan_id
         FROM library_roots ORDER BY added_at, id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(LibraryRoot {
            id: r.get(0)?,
            path: r.get(1)?,
            label: r.get(2)?,
            enabled: r.get::<_, i64>(3)? != 0,
            added_at: r.get(4)?,
            last_scan_id: r.get(5)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// The fast-skip lookup: one indexed hit on `(root_id, rel_path)`, no file IO.
pub fn sample_stamp(
    conn: &Connection,
    root_id: i64,
    rel_path: &str,
) -> Result<Option<SampleStamp>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, mtime, size_bytes, status FROM samples WHERE root_id = ?1 AND rel_path = ?2",
    )?;
    let found = stmt
        .query_row((root_id, rel_path), |r| {
            let status: String = r.get(3)?;
            Ok(SampleStamp {
                id: r.get(0)?,
                mtime: r.get(1)?,
                size_bytes: r.get(2)?,
                status: SampleStatus::parse(&status).unwrap_or(SampleStatus::Pending),
            })
        })
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// Every `(rel_path -> stamp)` under one root, for the discovery fast-skip.
///
/// One query instead of one per file. With four pooled read connections and a walker thread
/// per core, per-entry lookups spend most of a scan queued on `r2d2`; this takes the pool
/// out of the walk entirely. At 50,000 rows the map costs a few megabytes and is dropped
/// when the scan ends.
pub fn sample_stamps_for_root(
    conn: &Connection,
    root_id: i64,
) -> Result<HashMap<String, SampleStamp>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT rel_path, id, mtime, size_bytes, status FROM samples WHERE root_id = ?1",
    )?;
    let rows = stmt.query_map([root_id], |r| {
        let rel_path: String = r.get(0)?;
        let status: String = r.get(4)?;
        Ok((
            rel_path,
            SampleStamp {
                id: r.get(1)?,
                mtime: r.get(2)?,
                size_bytes: r.get(3)?,
                status: SampleStatus::parse(&status).unwrap_or(SampleStatus::Pending),
            },
        ))
    })?;
    rows.collect::<rusqlite::Result<HashMap<_, _>>>()
        .map_err(DbError::from)
}

/// What a previously processed sample determined about its own audio.
///
/// A matching content hash means byte-identical audio, so these values are exactly what
/// decoding the duplicate would produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedMetadata {
    pub duration_ms: Option<i64>,
    pub sample_rate: Option<i64>,
    pub channels: Option<i64>,
}

/// Finds an already-decoded sample with the same content, so a duplicate file can copy its
/// analysis instead of paying for a decode again (`overview.md` §3.1).
///
/// Restricted to rows that actually reached the decoder: a `pending` row has nothing to
/// copy, and a `decode_failed` one has nothing worth copying -- the duplicate must fail on
/// its own terms so its `error` column says what went wrong with *it*.
pub fn processed_sample_by_hash(
    conn: &Connection,
    content_hash: &[u8],
) -> Result<Option<(i64, DecodedMetadata)>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, duration_ms, sample_rate, channels FROM samples
         WHERE content_hash = ?1 AND status IN ('decoded', 'embedded')
         LIMIT 1",
    )?;
    let found = stmt
        .query_row([content_hash], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                DecodedMetadata {
                    duration_ms: r.get(1)?,
                    sample_rate: r.get(2)?,
                    channels: r.get(3)?,
                },
            ))
        })
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// Finds an already-embedded sample with the same content, so a duplicate file can borrow
/// its vector instead of paying for inference again (Phase 4 dedup).
///
/// `status = 'embedded'` is not redundant next to the two `NOT NULL`s. A row that was
/// re-processed after its file changed carries the new content hash while its old offsets
/// are cleared and its new ones have not been written yet, and the whole cost of getting
/// this wrong is a duplicate silently inheriting a vector for audio it does not contain.
/// The status column is what says the two agree.
pub fn embedded_sample_by_hash(
    conn: &Connection,
    content_hash: &[u8],
) -> Result<Option<(i64, EmbeddingLoc)>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, emb_offset, emb_len FROM samples
         WHERE content_hash = ?1 AND status = 'embedded'
           AND emb_offset IS NOT NULL AND emb_len IS NOT NULL
         LIMIT 1",
    )?;
    let found = stmt
        .query_row([content_hash], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                EmbeddingLoc {
                    offset: r.get::<_, i64>(1)? as u64,
                    dims: r.get::<_, i64>(2)? as u32,
                },
            ))
        })
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// Where one sample's vector lives, if it has been embedded.
pub fn embedding_loc(conn: &Connection, sample_id: i64) -> Result<Option<EmbeddingLoc>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT emb_offset, emb_len FROM samples
         WHERE id = ?1 AND emb_offset IS NOT NULL AND emb_len IS NOT NULL",
    )?;
    let found = stmt
        .query_row([sample_id], |r| {
            Ok(EmbeddingLoc {
                offset: r.get::<_, i64>(0)? as u64,
                dims: r.get::<_, i64>(1)? as u32,
            })
        })
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// Every embedded sample and its location, in file order.
///
/// File order matters: the projection re-fit walks the mmap sequentially, and compaction
/// rewrites in this order to keep it that way.
pub fn all_embedding_locs(conn: &Connection) -> Result<Vec<(i64, EmbeddingLoc)>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, emb_offset, emb_len FROM samples
         WHERE emb_offset IS NOT NULL AND emb_len IS NOT NULL
         ORDER BY emb_offset",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            EmbeddingLoc {
                offset: r.get::<_, i64>(1)? as u64,
                dims: r.get::<_, i64>(2)? as u32,
            },
        ))
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// How many samples have a vector in `embeddings.bin`.
///
/// A `COUNT(*)`, not the length of [`all_embedding_locs`]: the projection planner runs after
/// every scan and has no business materializing 50,000 rows to learn a number SQLite can
/// answer from an index.
pub fn count_embedded_samples(conn: &Connection) -> Result<i64, DbError> {
    Ok(conn.query_row(
        "SELECT COUNT(*) FROM samples WHERE emb_offset IS NOT NULL AND emb_len IS NOT NULL",
        [],
        |r| r.get(0),
    )?)
}

/// An embedded sample, with enough context to be recognizable in a neighbor list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedSample {
    pub id: i64,
    pub rel_path: String,
    pub duration_ms: Option<i64>,
    pub loc: EmbeddingLoc,
}

/// Every embedded sample and where its vector lives, in file order.
///
/// The brute-force neighbor pass reads this and walks the mmap once. Phase 5 replaces the
/// pass with an HNSW index; this stays as the exact answer to check the approximate one
/// against, and as what `task.md` Phase 4's Risk 5 evaluation inspects by hand.
pub fn embedded_samples(conn: &Connection) -> Result<Vec<EmbeddedSample>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, rel_path, duration_ms, emb_offset, emb_len FROM samples
         WHERE status = 'embedded' AND emb_offset IS NOT NULL AND emb_len IS NOT NULL
         ORDER BY emb_offset",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(EmbeddedSample {
            id: r.get(0)?,
            rel_path: r.get(1)?,
            duration_ms: r.get(2)?,
            loc: EmbeddingLoc {
                offset: r.get::<_, i64>(3)? as u64,
                dims: r.get::<_, i64>(4)? as u32,
            },
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// The DSP descriptor block for one sample.
pub fn sample_features(
    conn: &Connection,
    sample_id: i64,
) -> Result<Option<SampleFeatures>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT peak_db, rms_db, lufs_integrated, spectral_centroid, spectral_flatness,
                zero_crossing, onset_density, bpm, bpm_confidence, key_root, key_mode,
                key_confidence
         FROM sample_features WHERE sample_id = ?1",
    )?;
    let found = stmt
        .query_row([sample_id], |r| {
            Ok(SampleFeatures {
                peak_db: r.get(0)?,
                rms_db: r.get(1)?,
                lufs_integrated: r.get(2)?,
                spectral_centroid: r.get(3)?,
                spectral_flatness: r.get(4)?,
                zero_crossing: r.get(5)?,
                onset_density: r.get(6)?,
                bpm: r.get(7)?,
                bpm_confidence: r.get(8)?,
                key_root: r.get(9)?,
                key_mode: r.get(10)?,
                key_confidence: r.get(11)?,
            })
        })
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// A row of `projection_runs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionRun {
    pub id: i64,
    /// `'umap'` or `'pca'`.
    pub algorithm: String,
    pub params_json: String,
    pub sample_count: i64,
    pub created_at: i64,
    /// `None` while the run is still a shadow being built.
    pub completed_at: Option<i64>,
    pub is_active: bool,
}

/// The one active projection, or `None` before the first re-fit completes.
///
/// The schema's partial unique index makes "the one" a fact rather than a convention, so
/// this is a `query_row` and not a `LIMIT 1` over an ordering nobody chose.
pub fn active_projection_run(conn: &Connection) -> Result<Option<ProjectionRun>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, algorithm, params_json, sample_count, created_at, completed_at, is_active
         FROM projection_runs WHERE is_active = 1",
    )?;
    let found = stmt
        .query_row([], projection_run_from_row)
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// One run by id, active or not. What a test asserting the swap reads.
pub fn projection_run(conn: &Connection, run_id: i64) -> Result<Option<ProjectionRun>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, algorithm, params_json, sample_count, created_at, completed_at, is_active
         FROM projection_runs WHERE id = ?1",
    )?;
    let found = stmt
        .query_row([run_id], projection_run_from_row)
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// Every `projection_runs` row, newest first. For the dev inspector and for pruning.
pub fn projection_runs(conn: &Connection) -> Result<Vec<ProjectionRun>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT id, algorithm, params_json, sample_count, created_at, completed_at, is_active
         FROM projection_runs ORDER BY created_at DESC, id DESC",
    )?;
    let rows = stmt.query_map([], projection_run_from_row)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

fn projection_run_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ProjectionRun> {
    Ok(ProjectionRun {
        id: r.get(0)?,
        algorithm: r.get(1)?,
        params_json: r.get(2)?,
        sample_count: r.get(3)?,
        created_at: r.get(4)?,
        completed_at: r.get(5)?,
        is_active: r.get::<_, i64>(6)? != 0,
    })
}

/// Every coordinate belonging to one run, by `sample_id`.
pub fn projection_points(conn: &Connection, run_id: i64) -> Result<Vec<(i64, [f32; 3])>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT sample_id, x, y, z FROM projections WHERE run_id = ?1 ORDER BY sample_id",
    )?;
    let rows = stmt.query_map([run_id], |r| {
        Ok((r.get::<_, i64>(0)?, [r.get(1)?, r.get(2)?, r.get(3)?]))
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// The active layout, in one statement.
///
/// One statement rather than "read the run, then read its points" for a reason that
/// outlives this phase: the swap deletes the superseded run in the same transaction that
/// activates the new one, so a caller that issues two queries can read the old run id and
/// then find no rows under it. A single join is atomic against the swap under WAL and
/// cannot see the gap. Phase 6's `get_point_cloud` calls this.
pub fn active_projection_points(conn: &Connection) -> Result<Vec<(i64, [f32; 3])>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT p.sample_id, p.x, p.y, p.z
         FROM projections p
         JOIN projection_runs r ON r.id = p.run_id AND r.is_active = 1
         ORDER BY p.sample_id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((r.get::<_, i64>(0)?, [r.get(1)?, r.get(2)?, r.get(3)?]))
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// Coordinates in one run, as a lookup keyed by `sample_id`.
///
/// The Procrustes correspondence pass needs random access by id, not a sorted list.
pub fn projection_point_map(
    conn: &Connection,
    run_id: i64,
) -> Result<HashMap<i64, [f32; 3]>, DbError> {
    Ok(projection_points(conn, run_id)?.into_iter().collect())
}

/// How many coordinates a run holds.
pub fn count_projection_points(conn: &Connection, run_id: i64) -> Result<i64, DbError> {
    Ok(conn.query_row(
        "SELECT COUNT(*) FROM projections WHERE run_id = ?1",
        [run_id],
        |r| r.get(0),
    )?)
}

/// Embedded samples that `run_id` has no coordinate for, in file order.
///
/// The incremental placement path's input: everything the active layout has never seen.
/// `NOT EXISTS` rather than a `LEFT JOIN ... IS NULL` because `projections` is `WITHOUT
/// ROWID` with `(run_id, sample_id)` as its primary key, which makes the existence probe a
/// single index seek per candidate row.
pub fn embedding_locs_missing_from_run(
    conn: &Connection,
    run_id: i64,
) -> Result<Vec<(i64, EmbeddingLoc)>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT s.id, s.emb_offset, s.emb_len FROM samples s
         WHERE s.emb_offset IS NOT NULL AND s.emb_len IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM projections p WHERE p.run_id = ?1 AND p.sample_id = s.id
           )
         ORDER BY s.emb_offset",
    )?;
    let rows = stmt.query_map([run_id], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            EmbeddingLoc {
                offset: r.get::<_, i64>(1)? as u64,
                dims: r.get::<_, i64>(2)? as u32,
            },
        ))
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// Full-text search over filenames and tags, best match first.
///
/// The query string is passed to FTS5 as-is, so `KICK*` and `808 NOT snare` work. A
/// malformed query is a `DbError::Sqlite`, not a panic -- the search box will render it as
/// "no results" rather than taking the app down.
pub fn search_samples(conn: &Connection, query: &str, limit: u32) -> Result<Vec<i64>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT rowid FROM samples_fts WHERE samples_fts MATCH ?1 ORDER BY rank LIMIT ?2",
    )?;
    let rows = stmt.query_map((query, limit), |r| r.get::<_, i64>(0))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// Everything the inspector needs about one sample, minus its features and tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleRow {
    pub id: i64,
    pub root_id: i64,
    /// Absolute path of the root. Joined here rather than looked up separately so a
    /// detail fetch is one statement.
    pub root_path: String,
    pub rel_path: String,
    pub filename: String,
    pub ext: String,
    pub size_bytes: i64,
    pub duration_ms: Option<i64>,
    pub sample_rate: Option<i64>,
    pub channels: Option<i64>,
    pub status: SampleStatus,
    pub error: Option<String>,
    pub embedded: bool,
    pub updated_at: i64,
}

impl SampleRow {
    /// Where the file actually is.
    ///
    /// **The only place a filesystem path is constructed from database contents**, and the
    /// reason `rel_path` is stored relative to a root: moving a library is a one-row update
    /// to `library_roots.path`, and nothing the frontend sends is ever part of a path.
    /// The frontend has no filesystem permission at all (`overview.md` §2) and asks for
    /// samples by id, which is what keeps path traversal out of the threat model.
    pub fn absolute_path(&self) -> std::path::PathBuf {
        std::path::Path::new(&self.root_path).join(&self.rel_path)
    }
}

fn sample_row_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<SampleRow> {
    let status: String = r.get(10)?;
    Ok(SampleRow {
        id: r.get(0)?,
        root_id: r.get(1)?,
        root_path: r.get(2)?,
        rel_path: r.get(3)?,
        filename: r.get(4)?,
        ext: r.get(5)?,
        size_bytes: r.get(6)?,
        duration_ms: r.get(7)?,
        sample_rate: r.get(8)?,
        channels: r.get(9)?,
        status: SampleStatus::parse(&status).unwrap_or(SampleStatus::Pending),
        error: r.get(11)?,
        embedded: r.get::<_, Option<i64>>(12)?.is_some(),
        updated_at: r.get(13)?,
    })
}

const SAMPLE_ROW_COLUMNS: &str = "s.id, s.root_id, r.path, s.rel_path, s.filename, s.ext,
     s.size_bytes, s.duration_ms, s.sample_rate, s.channels, s.status, s.error,
     s.emb_offset, s.updated_at";

/// One sample row, joined to its root so the absolute path can be formed.
pub fn sample_row(conn: &Connection, sample_id: i64) -> Result<Option<SampleRow>, DbError> {
    let sql = format!(
        "SELECT {SAMPLE_ROW_COLUMNS} FROM samples s
         JOIN library_roots r ON r.id = s.root_id
         WHERE s.id = ?1"
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let found = stmt
        .query_row([sample_id], sample_row_from)
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// How many samples live under one root.
pub fn count_samples_for_root(conn: &Connection, root_id: i64) -> Result<i64, DbError> {
    Ok(conn.query_row(
        "SELECT COUNT(*) FROM samples WHERE root_id = ?1",
        [root_id],
        |r| r.get(0),
    )?)
}

/// Tag names on one sample, alphabetical.
pub fn tags_for_sample(conn: &Connection, sample_id: i64) -> Result<Vec<String>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT t.name FROM tags t
         JOIN sample_tags st ON st.tag_id = t.id
         WHERE st.sample_id = ?1
         ORDER BY t.name COLLATE NOCASE",
    )?;
    let rows = stmt.query_map([sample_id], |r| r.get::<_, String>(0))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// A tag row and how many samples carry it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagRow {
    pub id: i64,
    pub name: String,
    pub color: Option<String>,
    pub sample_count: i64,
}

/// Every tag, with its usage count, alphabetical.
///
/// `LEFT JOIN` rather than an inner one: a tag the user created and then removed from every
/// sample still exists, and dropping it from the list the moment its count hits zero would
/// make it look like the app forgot it.
pub fn all_tags(conn: &Connection) -> Result<Vec<TagRow>, DbError> {
    let mut stmt = conn.prepare(
        "SELECT t.id, t.name, t.color, COUNT(st.sample_id)
         FROM tags t LEFT JOIN sample_tags st ON st.tag_id = t.id
         GROUP BY t.id ORDER BY t.name COLLATE NOCASE",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok(TagRow {
            id: r.get(0)?,
            name: r.get(1)?,
            color: r.get(2)?,
            sample_count: r.get(3)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// A collection row and its size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionRow {
    pub id: i64,
    pub name: String,
    pub created_at: i64,
    pub sample_count: i64,
}

/// One collection by id, with its size.
pub fn collection(conn: &Connection, id: i64) -> Result<Option<CollectionRow>, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT c.id, c.name, c.created_at, COUNT(m.sample_id)
         FROM collections c LEFT JOIN collection_members m ON m.collection_id = c.id
         WHERE c.id = ?1 GROUP BY c.id",
    )?;
    let found = stmt
        .query_row([id], |r| {
            Ok(CollectionRow {
                id: r.get(0)?,
                name: r.get(1)?,
                created_at: r.get(2)?,
                sample_count: r.get(3)?,
            })
        })
        .map(Some)
        .or_else(no_rows_is_none)?;
    Ok(found)
}

/// Turns "no rows" into `None`, leaving every other error alone.
fn no_rows_is_none<T>(e: rusqlite::Error) -> rusqlite::Result<Option<T>> {
    match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    }
}
