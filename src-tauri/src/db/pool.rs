//! The `r2d2` read pool (4 read-only connections) and the pragma customizer.
//!
//! Pragmas are per-connection, not per-database: `foreign_keys = ON` set once on one
//! connection does nothing for the others. The customizer applies the full pragma set
//! from `overview.md` §4.3 to **every** connection the pool hands out.

use std::path::Path;

use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{Connection, OpenFlags};

/// Read-only connections served concurrently with the writer. WAL is what makes four of
/// these safe alongside an open write transaction; see `overview.md` §4.4.
pub const READ_POOL_SIZE: u32 = 4;

/// A pooled read-only connection.
pub type ReadConn = r2d2::PooledConnection<SqliteConnectionManager>;

/// The pool of read-only connections.
pub type ReadPool = r2d2::Pool<SqliteConnectionManager>;

/// Which pragma set a connection gets.
///
/// `journal_mode` and `wal_autocheckpoint` are properties of the *database file*, not of
/// the connection, and a read-only connection cannot write the header to change them. The
/// writer sets them once; readers inherit them and would only error trying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Read,
    Write,
}

/// Per-connection pragmas that apply to every connection in the process (§4.3).
const COMMON_PRAGMAS: &str = "
    PRAGMA synchronous  = NORMAL;
    PRAGMA foreign_keys = ON;
    PRAGMA mmap_size    = 268435456;
    PRAGMA cache_size   = -65536;
    PRAGMA temp_store   = MEMORY;
    PRAGMA busy_timeout = 5000;
";

/// Database-level pragmas, settable only by a writable connection.
const WRITE_PRAGMAS: &str = "
    PRAGMA journal_mode       = WAL;
    PRAGMA wal_autocheckpoint = 1000;
";

/// Applies the pragma set for `mode` to `conn`.
///
/// `execute_batch` rather than `execute`: several of these return a row (`journal_mode`
/// reports the mode it settled on), which `execute` rejects outright.
pub fn apply_pragmas(conn: &Connection, mode: Mode) -> rusqlite::Result<()> {
    if mode == Mode::Write {
        conn.execute_batch(WRITE_PRAGMAS)?;
    }
    conn.execute_batch(COMMON_PRAGMAS)
}

/// Applies the read pragma set to every connection r2d2 creates.
#[derive(Debug)]
struct PragmaCustomizer;

impl r2d2::CustomizeConnection<Connection, rusqlite::Error> for PragmaCustomizer {
    fn on_acquire(&self, conn: &mut Connection) -> Result<(), rusqlite::Error> {
        apply_pragmas(conn, Mode::Read)
    }
}

/// Opens the process's read pool against an existing database file.
///
/// The connections are opened `SQLITE_OPEN_READ_ONLY`, so "no second writer"
/// (cross-cutting rule 2) is enforced by SQLite itself rather than by convention. The
/// file must already exist and be migrated -- see [`super::Database::open`].
pub fn open_read_pool(db_path: &Path) -> Result<ReadPool, r2d2::Error> {
    let manager = SqliteConnectionManager::file(db_path)
        .with_flags(OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX);

    r2d2::Pool::builder()
        .max_size(READ_POOL_SIZE)
        .connection_customizer(Box::new(PragmaCustomizer))
        .build(manager)
}

/// Opens the process's one and only write connection.
///
/// Private to the module tree on purpose: [`super::writer::Writer`] takes ownership of the
/// returned connection and never hands it out. Nothing else has a way to obtain one.
pub(super) fn open_write_connection(db_path: &Path) -> rusqlite::Result<Connection> {
    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    apply_pragmas(&conn, Mode::Write)?;
    Ok(conn)
}
