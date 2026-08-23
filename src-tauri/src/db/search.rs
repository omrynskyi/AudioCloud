//! Filtering and column extraction: the SQL behind `query_samples` and
//! `get_feature_column` (`overview.md` §6.1).
//!
//! Both are the same shape of problem -- take a description of what the user wants, produce
//! one statement, hand back a column of primitives -- and both are the reason those two
//! commands use the binary transport. A filter that matches 30,000 samples is 120 KB of
//! `u32` and 400 KB of JSON, and the JSON has to be parsed into 30,000 JavaScript numbers
//! before anything can be drawn.
//!
//! **The filter is built, not concatenated.** Every clause appends a bound parameter and
//! every column name is a `&'static str` chosen by a `match` on an enum. Nothing the
//! frontend sends becomes part of a statement's text, with one deliberate exception: the
//! FTS5 `MATCH` argument, which is passed through as written so `KICK*` and `808 NOT snare`
//! work. That argument is bound, not interpolated -- a malformed query is a
//! `DbError::Sqlite` the search box renders as "no results", not an injection and not a
//! panic.
//!
//! [`Feature`] and [`QueryFilter`] carry their own serde and `ts-rs` derives rather than
//! having DTO twins in `ipc::types`. They describe a *query*, not a record, and a
//! translation layer between two identical thirteen-variant enums would be a place for them
//! to drift rather than a boundary that protected anything.

use rusqlite::{types::Value, Connection, ToSql};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

use super::DbError;

/// Which scalar a feature column or a range filter is about.
///
/// An enum rather than a column name on the wire, for the obvious reason: a string that
/// reaches a `SELECT` is an injection, and a string that reaches a `SELECT` through a
/// `format!` is the injection. [`Feature::column`] is the only mapping from this to SQL and
/// it returns a `&'static str` picked from a `match`, so no caller-supplied text ever
/// becomes part of a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "Feature.ts")]
pub enum Feature {
    PeakDb,
    RmsDb,
    LufsIntegrated,
    SpectralCentroid,
    SpectralFlatness,
    ZeroCrossing,
    OnsetDensity,
    Bpm,
    BpmConfidence,
    KeyRoot,
    KeyMode,
    KeyConfidence,
    /// The one column that lives on `samples` rather than `sample_features`. Included
    /// because "colour the map by length" is the first thing anyone asks for, and because
    /// the join is the same either way.
    DurationMs,
}

impl Feature {
    /// The qualified column this feature reads, with `s` bound to `samples` and `f` to
    /// `sample_features`.
    pub fn column(self) -> &'static str {
        match self {
            Feature::PeakDb => "f.peak_db",
            Feature::RmsDb => "f.rms_db",
            Feature::LufsIntegrated => "f.lufs_integrated",
            Feature::SpectralCentroid => "f.spectral_centroid",
            Feature::SpectralFlatness => "f.spectral_flatness",
            Feature::ZeroCrossing => "f.zero_crossing",
            Feature::OnsetDensity => "f.onset_density",
            Feature::Bpm => "f.bpm",
            Feature::BpmConfidence => "f.bpm_confidence",
            Feature::KeyRoot => "f.key_root",
            Feature::KeyMode => "f.key_mode",
            Feature::KeyConfidence => "f.key_confidence",
            Feature::DurationMs => "s.duration_ms",
        }
    }
}

/// An inclusive range with either end optional. `{ min: 100 }` is "at least 100".
#[derive(Debug, Clone, Copy, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "FeatureRange.ts")]
pub struct FeatureRange {
    pub feature: Feature,
    #[ts(optional)]
    pub min: Option<f64>,
    #[ts(optional)]
    pub max: Option<f64>,
}

/// What `query_samples` filters on (`overview.md` §6.1).
///
/// Every field is optional and an omitted field is not a constraint, so `{}` matches the
/// whole library. Populated fields are combined with AND -- a filter panel where ticking a
/// second box widened the result would be a filter panel nobody could reason about.
#[derive(Debug, Clone, Default, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export_to = "QueryFilter.ts")]
pub struct QueryFilter {
    /// FTS5 query over filename and tags. Passed through as written, so `KICK*` and
    /// `808 NOT snare` work. A malformed query is a typed error, not a panic.
    #[ts(optional)]
    pub text: Option<String>,
    /// Empty means every root.
    pub root_ids: Vec<i64>,
    /// A sample must carry **all** of these.
    pub tags: Vec<String>,
    /// Lowercase, without the dot. Empty means every extension.
    pub exts: Vec<String>,
    /// Range constraints, one per feature. Two ranges over the same feature both apply,
    /// which is a way of writing an intersection and not worth forbidding.
    pub features: Vec<FeatureRange>,
    /// Restrict to samples the active projection has a coordinate for.
    ///
    /// The renderer's filter is a mask over the point cloud, so a result containing samples
    /// that are not *in* the point cloud is a result it has to filter again. Default is
    /// false, because the library list is a legitimate consumer of the unfiltered form.
    pub projected_only: bool,
    /// Hard cap on returned ids. `None` means every match -- which is the intended case for
    /// the renderer's mask, where 30,000 ids is 120 KB and the whole point of the binary
    /// transport.
    #[ts(optional)]
    pub limit: Option<u32>,
}

/// How many buckets a filter's `IN (...)` list may hold before it stops being a query and
/// starts being a denial of service.
///
/// The frontend builds these from checkbox lists -- roots, tags, extensions -- so the
/// realistic maximum is tens. A cap exists because "realistic" is not "enforced" and SQLite
/// has its own parameter limit that would surface as an opaque error rather than a
/// meaningful one.
const MAX_IN_LIST: usize = 512;

/// Ids matching `filter`, ascending.
///
/// Ascending is the contract, not an accident: the point cloud is also ordered by
/// `sample_id`, so the renderer turns "which points are in this filter" into one merge over
/// two sorted arrays rather than a 30,000-entry `Set` rebuilt on every keystroke.
pub fn sample_ids(conn: &Connection, filter: &QueryFilter) -> Result<Vec<i64>, DbError> {
    let mut sql = String::from(
        "SELECT s.id FROM samples s
         LEFT JOIN sample_features f ON f.sample_id = s.id",
    );
    let mut params: Vec<Value> = Vec::new();

    if filter.projected_only {
        sql.push_str(
            "\n         JOIN projections p ON p.sample_id = s.id
         JOIN projection_runs pr ON pr.id = p.run_id AND pr.is_active = 1",
        );
    }
    sql.push_str("\n         WHERE 1 = 1");

    if let Some(text) = filter
        .text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        // A subquery rather than a join: `samples_fts` is contentless, its `rowid` is the
        // sample id, and `IN` over a MATCH lets SQLite run the FTS scan once instead of
        // once per candidate row.
        sql.push_str(
            "\n           AND s.id IN (SELECT rowid FROM samples_fts WHERE samples_fts MATCH ?)",
        );
        params.push(Value::Text(text.to_string()));
    }

    push_in_list(
        &mut sql,
        &mut params,
        "s.root_id",
        filter.root_ids.iter().copied().map(Value::Integer),
    )?;
    push_in_list(
        &mut sql,
        &mut params,
        "s.ext",
        filter
            .exts
            .iter()
            .map(|e| Value::Text(e.trim().trim_start_matches('.').to_lowercase())),
    )?;

    // AND, not OR. A sample must carry every tag asked for, which is what a stack of
    // checkboxes means to the person ticking them: each one narrows.
    if !filter.tags.is_empty() {
        if filter.tags.len() > MAX_IN_LIST {
            return Err(too_many("tags", filter.tags.len()));
        }
        for tag in &filter.tags {
            sql.push_str(
                "\n           AND EXISTS (SELECT 1 FROM sample_tags st
                       JOIN tags t ON t.id = st.tag_id
                       WHERE st.sample_id = s.id AND t.name = ? COLLATE NOCASE)",
            );
            params.push(Value::Text(tag.clone()));
        }
    }

    for range in &filter.features {
        let column = range.feature.column();
        // A null cell fails both comparisons, so a range constraint excludes samples that
        // have no value for it. That is the right default -- "BPM between 90 and 100" is
        // not a claim about files whose tempo could not be estimated -- and it is why the
        // feature *column* transport sends NaN rather than dropping the row.
        if let Some(min) = range.min {
            sql.push_str(&format!("\n           AND {column} >= ?"));
            params.push(Value::Real(min));
        }
        if let Some(max) = range.max {
            sql.push_str(&format!("\n           AND {column} <= ?"));
            params.push(Value::Real(max));
        }
    }

    sql.push_str("\n         ORDER BY s.id");
    if let Some(limit) = filter.limit {
        sql.push_str("\n         LIMIT ?");
        params.push(Value::Integer(i64::from(limit)));
    }

    let mut stmt = conn.prepare(&sql)?;
    let bound: Vec<&dyn ToSql> = params.iter().map(|p| p as &dyn ToSql).collect();
    let rows = stmt.query_map(bound.as_slice(), |r| r.get::<_, i64>(0))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// One feature column over the active projection, **in the point cloud's order**.
///
/// The join and the `ORDER BY` are deliberately identical to
/// [`super::queries::active_projection_points`], because index `i` of this vector and index
/// `i` of that point cloud have to be the same sample. A missing cell -- no
/// `sample_features` row, or a column the analyzer declined to guess -- comes back as
/// `f32::NAN` rather than being dropped, so the two vectors stay the same length and the
/// frontend can test for absence per point.
///
/// One statement, atomic against the projection swap under WAL: a caller that read the run
/// id first and the coordinates second could see the old run's id and the new run's rows.
pub fn feature_column(conn: &Connection, feature: Feature) -> Result<Vec<f32>, DbError> {
    let sql = format!(
        "SELECT {} FROM projections p
         JOIN projection_runs r ON r.id = p.run_id AND r.is_active = 1
         JOIN samples s ON s.id = p.sample_id
         LEFT JOIN sample_features f ON f.sample_id = s.id
         ORDER BY p.sample_id",
        feature.column()
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let rows = stmt.query_map([], |r| {
        Ok(r.get::<_, Option<f64>>(0)?.unwrap_or(f64::NAN) as f32)
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DbError::from)
}

/// Appends `AND <column> IN (?, ?, ...)` with one bound parameter per value, or nothing at
/// all when the list is empty -- an empty checkbox list is "no constraint", not "match
/// nothing".
fn push_in_list(
    sql: &mut String,
    params: &mut Vec<Value>,
    column: &'static str,
    values: impl Iterator<Item = Value>,
) -> Result<(), DbError> {
    let values: Vec<Value> = values.collect();
    if values.is_empty() {
        return Ok(());
    }
    if values.len() > MAX_IN_LIST {
        return Err(too_many(column, values.len()));
    }

    sql.push_str("\n           AND ");
    sql.push_str(column);
    sql.push_str(" IN (");
    for i in 0..values.len() {
        if i > 0 {
            sql.push(',');
        }
        sql.push('?');
    }
    sql.push(')');
    params.extend(values);
    Ok(())
}

fn too_many(what: &'static str, count: usize) -> DbError {
    DbError::FilterTooLarge {
        what,
        count,
        max: MAX_IN_LIST,
    }
}
