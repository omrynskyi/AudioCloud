//! Read queries against the pool -- sample lookup, feature columns, FTS5 search,
//! nearest-neighbor fetch.
//!
//! Every function here takes a `&Connection` rather than reaching for the pool itself, so
//! the caller decides whether it is spending a pooled read connection or running inside the
//! writer's transaction. They are also, for the same reason, trivially unit-testable.
//!
//! Extended through Phases 5-9.

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

/// Turns "no rows" into `None`, leaving every other error alone.
fn no_rows_is_none<T>(e: rusqlite::Error) -> rusqlite::Result<Option<T>> {
    match e {
        rusqlite::Error::QueryReturnedNoRows => Ok(None),
        other => Err(other),
    }
}
