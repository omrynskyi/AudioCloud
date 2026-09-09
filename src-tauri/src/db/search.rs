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
//! frontend sends becomes part of a statement's text. The FTS5 `MATCH` argument is the one
//! piece derived from free text, and it goes through [`fts_query`] first, which is what makes
//! typing a filename into the search box a search rather than a syntax error.
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

/// Turns what someone typed into a valid FTS5 query, or `None` if there is nothing to search
/// for.
///
/// ### Why this exists
///
/// The text used to reach `MATCH` verbatim, on the reasoning that passing it through is what
/// makes `KICK*` and `808 NOT snare` work. It does -- and it also makes the search box reject
/// the most ordinary things a person can type. Measured against the development library, whose
/// files are named like `@prodby.xero Snare - Crown.wav`:
///
/// | typed              | fts5 says                          |
/// | ------------------ | ---------------------------------- |
/// | `Snare - Crown`    | `no such column: Crown`            |
/// | `@prodby.xero`     | `syntax error near "@"`            |
/// | `crown.wav`        | `syntax error near "."`            |
/// | `kic`              | 0 rows -- no prefix match          |
///
/// You could copy a filename out of the app's own inspector, paste it into the app's own
/// search box, and get a syntax error. Raw passthrough is a good contract between programs and
/// a bad one between a program and a text field, so this sits in between: punctuation becomes
/// what fts5 already treats it as -- a separator -- and the tokens it separates are quoted, so
/// nothing a person types can be a syntax error.
///
/// ### What survives
///
/// - **Bare words** become quoted terms, ANDed, exactly as fts5 does implicitly.
/// - **The last word gets a `*`**, so results narrow as you type instead of appearing only on
///   the final keystroke. `kic` finds the kicks.
/// - **An explicit `*`** is honored wherever it appears: `kic* snare`.
/// - **`AND` / `OR` / `NOT`**, uppercase as fts5 requires, stay operators -- so `808 NOT snare`
///   still means what it did. Lowercase `and` is a word someone might be searching for, and is
///   treated as one. A dangling or doubled operator is dropped rather than becoming an error.
/// - **Double quotes** stay a phrase, and are the escape hatch from the automatic `*`:
///   `"kick"` is the exact word, `kick` is the prefix.
///
/// Everything else -- `@`, `.`, `-`, `(`, an unclosed quote -- is a separator, because that is
/// what `unicode61` made of it when the document was indexed. Searching for punctuation cannot
/// match anything, so dropping it loses nothing and buys a box that never errors.
pub fn fts_query(text: &str) -> Option<String> {
    let mut pieces: Vec<Piece> = Vec::new();
    for word in split_words(text) {
        if !word.quoted {
            if let Some(op) = as_operator(&word.text) {
                pieces.push(Piece::Operator(op));
                continue;
            }
        }
        // Splitting on "not alphanumeric" is the same cut `unicode61` makes, near enough: a
        // token that survives here is a token the index holds, and one that does not could not
        // have matched anything anyway.
        let tokens: Vec<&str> = word
            .text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.is_empty() {
            continue;
        }

        if word.quoted {
            // Order matters inside quotes -- that is the entire difference between a phrase
            // and a bag of words, and the reason someone reached for the quotes.
            pieces.push(Piece::Term {
                text: format!(
                    "\"{}\"{}",
                    tokens.join(" "),
                    if word.starred { "*" } else { "" }
                ),
                auto_prefixable: false,
            });
            continue;
        }

        for (token_index, token) in tokens.iter().enumerate() {
            let ends_word = token_index == tokens.len() - 1;
            let starred = ends_word && word.starred;
            pieces.push(Piece::Term {
                text: format!("\"{token}\"{}", if starred { "*" } else { "" }),
                auto_prefixable: ends_word && !starred,
            });
        }
    }

    drop_dangling_operators(&mut pieces);

    // The automatic prefix goes on the last surviving *term*, which is not always the last
    // word: half of `808 NOT snare` is `808 NOT`, and the token the cursor sits after there is
    // `808`. Applying it before the dangling operators were dropped put the prefix on nothing.
    if let Some(Piece::Term {
        text,
        auto_prefixable: true,
    }) = pieces.last_mut()
    {
        text.push('*');
    }

    if pieces.is_empty() {
        return None;
    }
    Some(
        pieces
            .iter()
            .map(|p| match p {
                Piece::Operator(op) => *op,
                Piece::Term { text, .. } => text.as_str(),
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// A term or an operator, kept apart so [`drop_dangling_operators`] can tell them apart after
/// the fact -- a term's text is already quoted by then and would be ambiguous to re-inspect.
enum Piece {
    Operator(&'static str),
    Term {
        /// Already quoted, and carrying an explicit `*` if the user typed one.
        text: String,
        /// Eligible for the automatic trailing `*` if it ends up being the last term. False
        /// for a phrase (quotes are how you opt out) and for a term that is starred already.
        auto_prefixable: bool,
    },
}

/// One whitespace- or quote-delimited chunk of what was typed, before tokenization.
struct Word {
    text: String,
    /// Came from inside double quotes: a phrase, and exempt from the automatic prefix.
    quoted: bool,
    /// Was followed by `*`.
    starred: bool,
}

/// Splits on whitespace, keeping a double-quoted span together as one word.
///
/// Nothing here can fail, including on input fts5 would reject. An unclosed quote closes at the
/// end of the input; a `*` with nothing before it marks the previous word instead of erroring;
/// stray punctuation rides along inside a word and is dropped later by tokenization.
fn split_words(text: &str) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut buf = String::new();
    let mut quoted = false;

    for c in text.chars() {
        match c {
            '"' => {
                if !buf.is_empty() {
                    words.push(Word {
                        text: std::mem::take(&mut buf),
                        quoted,
                        starred: false,
                    });
                }
                quoted = !quoted;
            }
            '*' if !quoted => {
                if !buf.is_empty() {
                    words.push(Word {
                        text: std::mem::take(&mut buf),
                        quoted: false,
                        starred: true,
                    });
                } else if let Some(last) = words.last_mut() {
                    // `"a b"*` -- the star trails the phrase that just closed.
                    last.starred = true;
                }
            }
            c if c.is_whitespace() && !quoted => {
                if !buf.is_empty() {
                    words.push(Word {
                        text: std::mem::take(&mut buf),
                        quoted: false,
                        starred: false,
                    });
                }
            }
            c => buf.push(c),
        }
    }
    if !buf.is_empty() {
        words.push(Word {
            text: buf,
            quoted,
            starred: false,
        });
    }
    words
}

/// `AND`, `OR` and `NOT` are operators only in the uppercase spelling fts5 itself requires.
/// Lowercase `and` is a word, and someone searching a library of loops for `kick and snare`
/// means three words, not a boolean.
fn as_operator(word: &str) -> Option<&'static str> {
    match word {
        "AND" => Some("AND"),
        "OR" => Some("OR"),
        "NOT" => Some("NOT"),
        _ => None,
    }
}

/// Removes operators that have nothing on one side of them -- leading, trailing, or doubled.
///
/// Every one of those is a syntax error in fts5 and all three are states a half-typed query
/// passes through: `kick NOT` exists for as long as it takes to type the next word. Dropping
/// them keeps the results showing what the finished part of the query asks for, rather than
/// blanking the panel until the sentence is complete.
fn drop_dangling_operators(pieces: &mut Vec<Piece>) {
    let mut previous_was_term = false;
    pieces.retain(|piece| match piece {
        Piece::Term { .. } => {
            previous_was_term = true;
            true
        }
        Piece::Operator(_) => {
            let keep = previous_was_term;
            previous_was_term = false;
            keep
        }
    });
    while matches!(pieces.last(), Some(Piece::Operator(_))) {
        pieces.pop();
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

    // `fts_query` rather than the text as typed -- see its doc comment for the syntax errors
    // that passthrough handed to anyone who typed a filename. `None` means nothing searchable
    // was typed (an empty box, or only punctuation), which is not a constraint.
    if let Some(query) = filter.text.as_deref().and_then(fts_query) {
        // A subquery rather than a join: `samples_fts` is contentless, its `rowid` is the
        // sample id, and `IN` over a MATCH lets SQLite run the FTS scan once instead of
        // once per candidate row.
        sql.push_str(
            "\n           AND s.id IN (SELECT rowid FROM samples_fts WHERE samples_fts MATCH ?)",
        );
        params.push(Value::Text(query));
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn a_bare_word_becomes_a_quoted_prefix_term() {
        // The prefix is the whole point of the last token: `kic` has to find the kicks while
        // the user is still typing, not only once they reach the `k`.
        assert_eq!(fts_query("kic").as_deref(), Some("\"kic\"*"));
        assert_eq!(fts_query("kick snare").as_deref(), Some("\"kick\" \"snare\"*"));
    }

    #[test]
    fn punctuation_is_a_separator_not_a_syntax_error() {
        // Every one of these was an fts5 error before, and every one is something a person
        // types when the files are named `@prodby.xero Snare - Crown.wav`.
        assert_eq!(
            fts_query("@prodby.xero").as_deref(),
            Some("\"prodby\" \"xero\"*")
        );
        assert_eq!(fts_query("crown.wav").as_deref(), Some("\"crown\" \"wav\"*"));
        assert_eq!(
            fts_query("Snare - Crown").as_deref(),
            Some("\"Snare\" \"Crown\"*")
        );
    }

    #[test]
    fn operators_survive_but_only_in_the_spelling_fts5_uses() {
        assert_eq!(
            fts_query("808 NOT snare").as_deref(),
            Some("\"808\" NOT \"snare\"*")
        );
        // Lowercase `and` is a word someone could be searching for, not a boolean.
        assert_eq!(
            fts_query("kick and snare").as_deref(),
            Some("\"kick\" \"and\" \"snare\"*")
        );
    }

    #[test]
    fn a_half_typed_boolean_does_not_blank_the_results() {
        // Each of these is a state `808 NOT snare` passes through on the way to being typed,
        // and each is a syntax error if handed to fts5 as-is.
        assert_eq!(fts_query("kick NOT").as_deref(), Some("\"kick\"*"));
        assert_eq!(fts_query("NOT kick").as_deref(), Some("\"kick\"*"));
        assert_eq!(fts_query("kick AND OR snare").as_deref(), Some("\"kick\" AND \"snare\"*"));
        assert_eq!(fts_query("NOT").as_deref(), None);
    }

    #[test]
    fn quotes_are_a_phrase_and_the_way_out_of_the_automatic_prefix() {
        assert_eq!(fts_query("\"kick\"").as_deref(), Some("\"kick\""));
        assert_eq!(
            fts_query("\"deep kick\"").as_deref(),
            Some("\"deep kick\"")
        );
        // An unclosed quote is what every phrase search looks like halfway through typing it.
        assert_eq!(fts_query("\"deep kick").as_deref(), Some("\"deep kick\""));
    }

    #[test]
    fn an_explicit_star_is_honored_wherever_it_falls() {
        assert_eq!(fts_query("kic* snare").as_deref(), Some("\"kic\"* \"snare\"*"));
    }

    #[test]
    fn nothing_searchable_is_not_a_constraint() {
        assert_eq!(fts_query(""), None);
        assert_eq!(fts_query("   "), None);
        // Punctuation cannot match a token, so a box holding only punctuation is an empty box.
        assert_eq!(fts_query("--- ... @"), None);
    }

    /// The index as `V7__fts_index_path.sql` builds it, with two rows shaped like the
    /// development library's.
    fn indexed() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE samples_fts USING fts5(
                 filename, path, tags, content = '',
                 tokenize = \"unicode61 remove_diacritics 2\");",
        )
        .unwrap();
        let rows = [
            (
                1i64,
                "@prodby.xero Snare - Crown.wav",
                "[FREE VERSION] @PRODBY.XERO UK UNDERGROUND DRUM KIT/Snares/@prodby.xero Snare - Crown.wav",
            ),
            (
                2,
                "@prodby.xero Kick - Apex.wav",
                "[FREE VERSION] @PRODBY.XERO UK UNDERGROUND DRUM KIT/Kicks/@prodby.xero Kick - Apex.wav",
            ),
        ];
        for (id, filename, path) in rows {
            conn.execute(
                "INSERT INTO samples_fts (rowid, filename, path, tags) VALUES (?1, ?2, ?3, '')",
                (id, filename, path),
            )
            .unwrap();
        }
        conn
    }

    fn hits(conn: &Connection, typed: &str) -> Vec<i64> {
        let Some(query) = fts_query(typed) else {
            return Vec::new();
        };
        conn.prepare("SELECT rowid FROM samples_fts WHERE samples_fts MATCH ?1 ORDER BY rowid")
            .unwrap()
            .query_map([query], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<i64>>>()
            .expect("a sanitized query must never be a syntax error")
    }

    #[test]
    fn a_folder_name_finds_what_is_filed_under_it() {
        // The whole reason V7 exists: `Kicks/` is where the kicks are, and before the `path`
        // column no query could reach it.
        let conn = indexed();
        assert_eq!(hits(&conn, "kicks"), [2]);
        assert_eq!(hits(&conn, "snares"), [1]);
        assert_eq!(hits(&conn, "underground"), [1, 2]);
    }

    #[test]
    fn a_filename_pasted_back_into_the_box_finds_its_own_file() {
        // Copying a name out of the inspector and pasting it into search was a syntax error.
        let conn = indexed();
        assert_eq!(hits(&conn, "@prodby.xero Snare - Crown.wav"), [1]);
    }

    #[test]
    fn nothing_a_person_can_type_reaches_fts5_as_an_error() {
        let conn = indexed();
        for typed in [
            "@prodby.xero",
            "crown.wav",
            "Snare - Crown",
            "kick NOT",
            "NOT kick",
            "(kick",
            "kick)",
            "\"unclosed",
            "* ",
            "^kick",
            "kick:snare",
            "a OR OR b",
            "---",
            "808 NOT snare",
        ] {
            // `hits` unwraps the query, so a syntax error fails the test by name.
            let _ = hits(&conn, typed);
        }
    }
}
