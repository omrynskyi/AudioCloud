//! SQLite data layer (`overview.md` §4).
//!
//! One writer, many readers. The invariant this module exists to protect: exactly one
//! write connection lives in the whole process ([`writer`]), and reads come from a small
//! read-only pool ([`pool`]). A second write connection appearing anywhere is a bug, not an
//! optimization.
//!
//! The pool's connections are opened `SQLITE_OPEN_READ_ONLY` and the write connection is
//! moved into the writer thread at construction, so the invariant is enforced by SQLite
//! and by ownership rather than by reviewer vigilance.

pub mod embeddings;
pub mod pool;
pub mod queries;
pub mod search;
pub mod writer;

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::OptionalExtension;

pub use embeddings::{EmbeddingLoc, EmbeddingMatrix, EmbeddingStore};
pub use pool::{ReadConn, ReadPool};
pub use writer::WriterHandle;

/// Migrations embedded at compile time from `src-tauri/migrations/` (`overview.md` §4.5).
mod embedded {
    refinery::embed_migrations!("./migrations");
}

/// Filename of the SQLite database inside the app data directory.
pub const DB_FILENAME: &str = "library.db";

/// Every failure the data layer can produce.
///
/// Kept local to the module rather than folded into [`crate::error::AppError`]: the IPC
/// error surface is Phase 6, and most of these never reach the frontend unmapped.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("read pool: {0}")]
    Pool(#[from] r2d2::Error),

    #[error("migration: {0}")]
    Migration(#[from] Box<refinery::Error>),

    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    /// The writer thread panicked or was shut down while a caller still held a handle.
    #[error("the database writer thread is no longer running")]
    WriterGone,

    #[error("embedding dimension mismatch: the store holds {expected}, got {actual}")]
    Dimension { expected: usize, actual: usize },

    #[error("embedding at byte {offset} (+{bytes}) lies outside embeddings.bin ({size} bytes)")]
    EmbeddingOutOfRange { offset: u64, bytes: u64, size: u64 },

    /// A filter arrived with more values in one `IN (...)` list than the query builder is
    /// willing to bind.
    ///
    /// Typed rather than folded into [`DbError::Sqlite`] because it is not SQLite's
    /// complaint: it is this crate refusing to build a statement, and the frontend can
    /// render "too many filters selected" only if it can tell the two apart.
    #[error("a filter listed {count} {what}, past the {max} this query builder binds")]
    FilterTooLarge {
        what: &'static str,
        count: usize,
        max: usize,
    },

    /// A `Mutex` guarding a data-layer resource was left poisoned by a panicking holder.
    ///
    /// Reported rather than unwrapped: the holder is a pipeline stage, and a poisoned
    /// embedding store means the scan cannot write vectors -- which is a scan that fails
    /// with a reason, not a process that aborts (cross-cutting rule 8).
    #[error("the {0} lock is poisoned")]
    Poisoned(&'static str),
}

impl From<refinery::Error> for DbError {
    fn from(e: refinery::Error) -> Self {
        // Boxed: `refinery::Error` is large enough that carrying it inline makes every
        // `Result<_, DbError>` in the crate pay for it.
        DbError::Migration(Box::new(e))
    }
}

/// Attaches the path being operated on to an IO failure, because "No such file or
/// directory" without one is not an error message, it is a riddle.
pub(crate) fn io_error(context: impl Into<String>, source: std::io::Error) -> DbError {
    DbError::Io {
        context: context.into(),
        source,
    }
}

/// Milliseconds since the Unix epoch. Every `*_at` column in the schema is one of these.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Lifecycle status of a sample row, mirroring the `samples.status` CHECK-less text column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStatus {
    Pending,
    Decoded,
    Embedded,
    DecodeFailed,
    Missing,
}

impl SampleStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            SampleStatus::Pending => "pending",
            SampleStatus::Decoded => "decoded",
            SampleStatus::Embedded => "embedded",
            SampleStatus::DecodeFailed => "decode_failed",
            SampleStatus::Missing => "missing",
        }
    }

    /// Whether this row is finished, for a scan that does or does not embed.
    ///
    /// The discovery fast-skip consults this: an unchanged file whose row is still
    /// `pending` was never actually read -- the last scan died before reaching it -- so
    /// "unchanged" says nothing useful and the file has to be processed. `DecodeFailed` and
    /// `Missing` count as finished, because retrying a corrupt file on every scan is a
    /// cost with no upside; a user who fixes the file changes its mtime, which is what
    /// brings it back.
    ///
    /// `embedding_required` is what makes resume work in Phase 4. A row reaches `decoded`
    /// when its DSP features land and `embedded` only once its vector is in
    /// `embeddings.bin`, so a scan that ran before the model was installed -- or one that
    /// was cancelled between the two -- leaves `decoded` rows behind. Those are finished
    /// for a scan with no session and unfinished for a scan with one, which is precisely
    /// the difference this argument carries. Without it a resumed scan would skip exactly
    /// the files it exists to finish.
    pub fn is_complete(self, embedding_required: bool) -> bool {
        match self {
            SampleStatus::Pending => false,
            SampleStatus::Decoded => !embedding_required,
            SampleStatus::Embedded | SampleStatus::DecodeFailed | SampleStatus::Missing => true,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pending" => SampleStatus::Pending,
            "decoded" => SampleStatus::Decoded,
            "embedded" => SampleStatus::Embedded,
            "decode_failed" => SampleStatus::DecodeFailed,
            "missing" => SampleStatus::Missing,
            _ => return None,
        })
    }
}

/// A row the discovery stage wants written, keyed by `(root_id, rel_path)`.
#[derive(Debug, Clone)]
pub struct NewSample {
    pub root_id: i64,
    pub rel_path: String,
    pub filename: String,
    pub ext: String,
    pub size_bytes: i64,
    pub mtime: i64,
    /// blake3, 32 bytes. `None` until the hashing stage runs.
    pub content_hash: Option<[u8; 32]>,
    pub duration_ms: Option<i64>,
    pub sample_rate: Option<i64>,
    pub channels: Option<i64>,
    pub status: SampleStatus,
}

/// The DSP descriptor block, written as a unit by the feature stage (Phase 2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SampleFeatures {
    pub peak_db: Option<f32>,
    pub rms_db: Option<f32>,
    pub lufs_integrated: Option<f32>,
    pub spectral_centroid: Option<f32>,
    pub spectral_flatness: Option<f32>,
    pub zero_crossing: Option<f32>,
    pub onset_density: Option<f32>,
    pub bpm: Option<f32>,
    pub bpm_confidence: Option<f32>,
    pub key_root: Option<i32>,
    pub key_mode: Option<i32>,
    pub key_confidence: Option<f32>,
}

/// Terminal (and initial) states of a `scan_runs` row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanStatus {
    Running,
    Completed,
    Cancelled,
    Failed,
}

impl ScanStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ScanStatus::Running => "running",
            ScanStatus::Completed => "completed",
            ScanStatus::Cancelled => "cancelled",
            ScanStatus::Failed => "failed",
        }
    }
}

/// Counters a scan reports when it finishes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanCounts {
    pub files_seen: i64,
    pub files_added: i64,
    pub files_skipped: i64,
    pub files_failed: i64,
}

/// The data layer as the rest of the app sees it: a writer handle, a read pool, and the
/// embedding store.
///
/// Constructed once at startup and handed to Tauri's managed state. Cloning is not
/// offered; share it as `&Database` or through `State<Database>`.
#[derive(Debug)]
pub struct Database {
    data_dir: PathBuf,
    writer: WriterHandle,
    reads: ReadPool,
    embeddings: Mutex<EmbeddingStore>,
}

impl Database {
    /// Opens (creating if absent) the database under `data_dir`.
    ///
    /// Order matters: the directory is created, migrations run to completion on a
    /// throwaway write connection, and only then are the writer thread and the read pool
    /// started. Opening the read pool first would race a migration that has not yet
    /// created the tables the readers expect -- and read-only connections cannot create
    /// the file at all.
    pub fn open(data_dir: impl AsRef<Path>, embedding_dim: usize) -> Result<Self, DbError> {
        let data_dir = data_dir.as_ref().to_path_buf();
        prepare_data_dir(&data_dir)?;

        let db_path = data_dir.join(DB_FILENAME);
        run_migrations(&db_path)?;

        let embeddings = EmbeddingStore::open(&data_dir, embedding_dim)?;
        let writer = WriterHandle::spawn(&db_path)?;
        let reads = pool::open_read_pool(&db_path)?;

        tracing::info!(
            path = %db_path.display(),
            read_pool = pool::READ_POOL_SIZE,
            "data layer ready"
        );

        Ok(Self {
            data_dir,
            writer,
            reads,
            embeddings: Mutex::new(embeddings),
        })
    }

    /// The single writer. Every mutation in the process goes through this.
    pub fn writer(&self) -> &WriterHandle {
        &self.writer
    }

    /// Checks out a read-only connection. Blocks only if all four are busy.
    pub fn read(&self) -> Result<ReadConn, DbError> {
        Ok(self.reads.get()?)
    }

    /// The append-only embedding store (`overview.md` §4.2).
    pub fn embeddings(&self) -> &Mutex<EmbeddingStore> {
        &self.embeddings
    }

    /// The app data directory this database lives in.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Flushes the writer's open batch and stops its thread. Called on app exit so the
    /// tail of a scan is not lost to a partial batch.
    pub fn shutdown(&self) {
        self.writer.shutdown();
    }
}

/// Creates the app data directory on first run, owner-only.
///
/// The directory holds the user's whole library index; 0700 keeps it out of reach of other
/// accounts on a shared Mac. `create_dir_all` is a no-op when it already exists, so this is
/// safe on every launch, not just the first.
pub fn prepare_data_dir(dir: &Path) -> Result<(), DbError> {
    std::fs::create_dir_all(dir)
        .map_err(|e| io_error(format!("creating app data dir {}", dir.display()), e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o700);
        std::fs::set_permissions(dir, perms)
            .map_err(|e| io_error(format!("securing app data dir {}", dir.display()), e))?;
    }

    Ok(())
}

/// Runs every pending migration on a connection that is closed before any other opens.
///
/// Forward-only and idempotent: `refinery` records applied migrations in
/// `refinery_schema_history` and skips them on the next run.
fn run_migrations(db_path: &Path) -> Result<(), DbError> {
    let mut conn = pool::open_write_connection(db_path)?;
    reconcile_legacy_v7(&mut conn)?;
    let report = embedded::migrations::runner().run(&mut conn)?;

    let applied = report.applied_migrations();
    if applied.is_empty() {
        tracing::debug!("schema up to date");
    } else {
        for m in applied {
            tracing::info!(version = m.version(), name = m.name(), "migration applied");
        }
    }

    Ok(())
}

/// Repairs exactly one historical migration-file edit before asking refinery to validate the
/// migration chain.
///
/// V7 was applied on a development build while it created the original three-column FTS table.
/// Its file was then extended in place with `contentless_delete` and a cleanup trigger. Refinery
/// rightly refuses to continue when an applied migration's checksum no longer matches, but the
/// correct data repair is forward-only: V8 rebuilds the index with the new definition. This small
/// bridge updates *only* that known history row, and only after proving the database still has the
/// original V7 table and lacks the new trigger. It never touches `samples` or any user metadata.
fn reconcile_legacy_v7(conn: &mut rusqlite::Connection) -> Result<(), DbError> {
    const VERSION: i32 = 7;
    const LEGACY_CHECKSUM: &str = "11538323785542997172";

    // A brand-new database has no refinery history yet. Let refinery create it while
    // applying V1 instead of treating that normal first-run state as a failed repair.
    let has_history: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'refinery_schema_history')",
        [],
        |row| row.get(0),
    )?;
    if !has_history {
        return Ok(());
    }

    let Some(current_checksum) = embedded::migrations::runner()
        .get_migrations()
        .iter()
        .find(|migration| migration.version() == VERSION)
        .map(|migration| migration.checksum().to_string())
    else {
        // The embedded list is compile-time generated. This is defensive only: without V7,
        // there is no checksum that could be reconciled, so refinery should report its normal
        // missing-migration error below.
        return Ok(());
    };

    let applied_checksum: Option<String> = conn
        .query_row(
            "SELECT checksum FROM refinery_schema_history WHERE version = ?1",
            [VERSION],
            |row| row.get(0),
        )
        .optional()?;
    if applied_checksum.as_deref() != Some(LEGACY_CHECKSUM) {
        return Ok(());
    }

    let table_sql: Option<String> = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'samples_fts'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let trigger_exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = 'samples_fts_delete')",
        [],
        |row| row.get(0),
    )?;

    let is_original_v7 = table_sql.as_deref().is_some_and(|sql| {
        sql.contains("filename")
            && sql.contains("path")
            && sql.contains("tags")
            && sql.contains("content = ''")
            && !sql.contains("contentless_delete")
    });
    if !is_original_v7 || trigger_exists {
        return Ok(());
    }

    conn.execute(
        "UPDATE refinery_schema_history SET checksum = ?1 WHERE version = ?2 AND checksum = ?3",
        (&current_checksum, VERSION, LEGACY_CHECKSUM),
    )?;
    tracing::warn!(
        version = VERSION,
        "reconciled legacy V7 migration history; V8 will rebuild the FTS index"
    );
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared fixtures. `TempDir` must outlive the `Database`, so every helper hands both
    //! back and the test binds them together.

    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use tempfile::TempDir;

    /// Small enough to keep fixtures readable; the real value is [`crate::EMBEDDING_DIM`].
    pub const TEST_DIM: usize = 8;

    pub fn temp_db() -> (TempDir, Database) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(dir.path(), TEST_DIM).expect("open database");
        (dir, db)
    }

    pub fn sample(root_id: i64, rel_path: &str) -> NewSample {
        let filename = rel_path.rsplit('/').next().unwrap_or(rel_path).to_string();
        let ext = filename
            .rsplit_once('.')
            .map(|(_, e)| e.to_string())
            .unwrap_or_default();
        NewSample {
            root_id,
            rel_path: rel_path.to_string(),
            filename,
            ext,
            size_bytes: 1024,
            mtime: 1_700_000_000,
            content_hash: None,
            duration_ms: Some(1500),
            sample_rate: Some(48_000),
            channels: Some(2),
            status: SampleStatus::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::test_support::*;
    use super::*;

    /// Exit criterion: migrations run twice with no effect the second time.
    #[test]
    fn migrations_are_idempotent() {
        let dir = tempfile::tempdir().unwrap();

        let first = Database::open(dir.path(), TEST_DIM).unwrap();
        let applied_first: i64 = first
            .read()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM refinery_schema_history", [], |r| {
                r.get(0)
            })
            .unwrap();
        first.shutdown();
        drop(first);

        let second = Database::open(dir.path(), TEST_DIM).unwrap();
        let conn = second.read().unwrap();
        let applied_second: i64 = conn
            .query_row("SELECT COUNT(*) FROM refinery_schema_history", [], |r| {
                r.get(0)
            })
            .unwrap();

        assert_eq!(
            applied_first, 8,
            "V1 through V8 should be the only migrations so far"
        );
        assert_eq!(
            applied_first, applied_second,
            "the second run applied a migration it should have skipped"
        );

        // The schema is still whole, not half-reapplied.
        let tables: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'samples'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(tables, 1);
    }

    #[test]
    fn legacy_v7_history_is_reconciled_before_v8_rebuilds_fts() {
        let dir = tempfile::tempdir().unwrap();
        let first = Database::open(dir.path(), TEST_DIM).unwrap();
        first.shutdown();
        drop(first);

        // Recreate the exact V7 state that escaped into the local development database:
        // its history checksum predates the later contentless-delete revision, and the table
        // has the original path index without a deletion trigger.
        let db_path = dir.path().join(DB_FILENAME);
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "
            DROP TRIGGER samples_fts_delete;
            DROP TABLE samples_fts;
            CREATE VIRTUAL TABLE samples_fts USING fts5(
                filename,
                path,
                tags,
                content = '',
                tokenize = \"unicode61 remove_diacritics 2\"
            );
            DELETE FROM refinery_schema_history WHERE version = 8;
            UPDATE refinery_schema_history
            SET checksum = '11538323785542997172'
            WHERE version = 7;
            ",
        )
        .unwrap();
        drop(conn);

        let repaired = Database::open(dir.path(), TEST_DIM).unwrap();
        let conn = repaired.read().unwrap();
        let checksum: String = conn
            .query_row(
                "SELECT checksum FROM refinery_schema_history WHERE version = 7",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_ne!(checksum, "11538323785542997172");

        let sql: String = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'samples_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(sql.contains("contentless_delete = 1"));
        let trigger_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name = 'samples_fts_delete'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(trigger_count, 1);
    }

    /// Exit criterion: `PRAGMA foreign_key_check` is clean.
    #[test]
    fn foreign_key_check_is_clean_on_a_fresh_database() {
        let (_dir, db) = temp_db();
        let conn = db.read().unwrap();
        let mut stmt = conn.prepare("PRAGMA foreign_key_check").unwrap();
        let violations = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .count();
        assert_eq!(violations, 0);
    }

    /// `foreign_keys` is per-connection and off by default: the whole reason the pool has a
    /// customizer. If this fails, orphan rows become possible on three of four readers.
    #[test]
    fn every_pooled_connection_enforces_foreign_keys() {
        let (_dir, db) = temp_db();

        // Check out every connection at once so the assertion covers all of them, not the
        // same one four times.
        let conns: Vec<_> = (0..pool::READ_POOL_SIZE)
            .map(|_| db.read().unwrap())
            .collect();

        for conn in &conns {
            let on: i64 = conn
                .query_row("PRAGMA foreign_keys", [], |r| r.get(0))
                .unwrap();
            assert_eq!(on, 1, "a pooled connection had foreign_keys off");

            let journal: String = conn
                .query_row("PRAGMA journal_mode", [], |r| r.get(0))
                .unwrap();
            assert_eq!(journal.to_lowercase(), "wal");
        }
    }

    /// Enforcement, not just declaration: an orphan sample must be rejected outright.
    #[test]
    fn foreign_keys_reject_an_orphan_sample() {
        let (_dir, db) = temp_db();

        let err = db
            .writer()
            .upsert_samples(vec![sample(9999, "nowhere/kick.wav")])
            .expect_err("a sample under a nonexistent root must not insert");

        match err {
            DbError::Sqlite(rusqlite::Error::SqliteFailure(e, _)) => {
                assert_eq!(e.code, rusqlite::ErrorCode::ConstraintViolation);
            }
            other => panic!("expected a constraint violation, got {other:?}"),
        }

        db.writer().flush().unwrap();
        assert_eq!(queries::count_samples(&db.read().unwrap()).unwrap(), 0);
    }

    /// A cascade delete has to reach the whole dependent graph, or removing a root leaves
    /// features pointing at samples that no longer exist.
    #[test]
    fn removing_a_root_cascades_to_its_samples() {
        let (_dir, db) = temp_db();
        let writer = db.writer();

        let root = writer.add_root("/Library/Samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![sample(root, "a.wav"), sample(root, "b.wav")])
            .unwrap();
        writer
            .set_features(vec![(ids[0], SampleFeatures::default())])
            .unwrap();
        writer.flush().unwrap();

        writer.remove_root(root).unwrap();

        let conn = db.read().unwrap();
        assert_eq!(queries::count_samples(&conn).unwrap(), 0);
        let features: i64 = conn
            .query_row("SELECT COUNT(*) FROM sample_features", [], |r| r.get(0))
            .unwrap();
        assert_eq!(features, 0);
    }

    #[cfg(unix)]
    #[test]
    fn the_data_dir_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("AudioCloud");
        prepare_data_dir(&data_dir).unwrap();

        let mode = std::fs::metadata(&data_dir).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o700);
    }

    #[test]
    fn sample_status_round_trips_through_its_column_text() {
        for status in [
            SampleStatus::Pending,
            SampleStatus::Decoded,
            SampleStatus::Embedded,
            SampleStatus::DecodeFailed,
            SampleStatus::Missing,
        ] {
            assert_eq!(SampleStatus::parse(status.as_str()), Some(status));
        }
        assert_eq!(SampleStatus::parse("nonsense"), None);
    }
}
