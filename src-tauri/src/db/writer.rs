//! The single SQLite writer thread.
//!
//! Owns the only write connection in the process. Accepts an `mpsc` command enum and
//! replies over oneshot channels. Writes are batched into transactions that flush at
//! 1000 rows **or** 250 ms, whichever comes first (`overview.md` §4.4).
//!
//! **Why a thread and not a mutex.** SQLite permits exactly one writer, and in WAL mode a
//! second concurrent writer does not queue -- it returns `SQLITE_BUSY`. A `Mutex<Connection>`
//! shared across a twelve-thread `rayon` pool would serialize anyway, but with every worker
//! parked on the lock and every insert paying its own transaction. Funnelling through a
//! channel lets the writer amortize the commit across a thousand rows and lets the
//! producers keep working.
//!
//! **Batching is deferred commits, not deferred work.** Statements execute the moment the
//! command arrives, inside a transaction that stays open; only the `COMMIT` waits. That is
//! why [`WriterHandle::upsert_samples`] can hand back row ids immediately -- they are
//! assigned by the insert, not by the commit.

use std::{
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use rusqlite::Connection;

use super::{
    io_error, now_ms, pool, DbError, EmbeddingLoc, NewSample, SampleFeatures, SampleStatus,
    ScanCounts, ScanStatus,
};

/// Commit once this many rows have accumulated in the open transaction.
pub const BATCH_ROWS: usize = 1000;

/// ...or once the oldest uncommitted row is this old, whichever comes first.
///
/// The time trigger matters as much as the count one: without it the last partial batch of
/// a scan sits uncommitted until something unrelated happens to push it over the line.
pub const BATCH_INTERVAL: Duration = Duration::from_millis(250);

/// Depth of the command queue. Deep enough to absorb a burst from the pipeline, shallow
/// enough that a stalled writer applies backpressure instead of growing without bound.
const QUEUE_DEPTH: usize = 256;

/// A oneshot reply channel. `sync_channel(1)` so the writer never blocks handing back a
/// result, even if the caller has already given up waiting.
type Reply<T> = SyncSender<Result<T, DbError>>;

/// Counters for what the writer actually did. Tests assert on these; Phase 10 logs them.
#[derive(Debug, Default)]
pub struct WriterMetrics {
    /// Transactions committed.
    pub commits: AtomicU64,
    /// Rows passed to the writer (not rows changed -- an upsert of an unchanged row counts).
    pub rows: AtomicU64,
    /// Commits triggered by reaching [`BATCH_ROWS`].
    pub flushes_by_count: AtomicU64,
    /// Commits triggered by [`BATCH_INTERVAL`] elapsing.
    pub flushes_by_timeout: AtomicU64,
    /// Commits forced by a durable command, an explicit flush, or shutdown.
    pub flushes_forced: AtomicU64,
}

impl WriterMetrics {
    pub fn commits(&self) -> u64 {
        self.commits.load(Ordering::Relaxed)
    }
    pub fn rows(&self) -> u64 {
        self.rows.load(Ordering::Relaxed)
    }
    pub fn flushes_by_count(&self) -> u64 {
        self.flushes_by_count.load(Ordering::Relaxed)
    }
    pub fn flushes_by_timeout(&self) -> u64 {
        self.flushes_by_timeout.load(Ordering::Relaxed)
    }
}

/// What made the writer commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlushReason {
    Count,
    Timeout,
    Forced,
}

/// A mutation request. Constructed only by [`WriterHandle`]'s methods.
#[derive(Debug)]
enum Command {
    UpsertSamples {
        rows: Vec<NewSample>,
        reply: Reply<Vec<i64>>,
    },
    SetFeatures {
        rows: Vec<(i64, SampleFeatures)>,
        reply: Reply<()>,
    },
    SetEmbeddings {
        rows: Vec<(i64, EmbeddingLoc)>,
        reply: Reply<()>,
    },
    MarkDecodeFailed {
        rows: Vec<(i64, String)>,
        reply: Reply<()>,
    },
    AddRoot {
        path: String,
        label: Option<String>,
        reply: Reply<i64>,
    },
    SetRootEnabled {
        root_id: i64,
        enabled: bool,
        reply: Reply<()>,
    },
    RemoveRoot {
        root_id: i64,
        reply: Reply<()>,
    },
    StartScan {
        root_id: Option<i64>,
        reply: Reply<i64>,
    },
    FinishScan {
        scan_id: i64,
        status: ScanStatus,
        counts: ScanCounts,
        error: Option<String>,
        reply: Reply<()>,
    },
    BeginProjectionRun {
        algorithm: String,
        params_json: String,
        sample_count: i64,
        reply: Reply<i64>,
    },
    SetProjectionPoints {
        run_id: i64,
        rows: Vec<(i64, [f32; 3])>,
        reply: Reply<()>,
    },
    ActivateProjectionRun {
        run_id: i64,
        reply: Reply<()>,
    },
    DiscardProjectionRun {
        run_id: i64,
        reply: Reply<()>,
    },
    Flush {
        reply: Reply<()>,
    },
    Shutdown,
}

impl Command {
    /// Rows this command contributes to the open batch.
    fn row_count(&self) -> usize {
        match self {
            Command::UpsertSamples { rows, .. } => rows.len(),
            Command::SetFeatures { rows, .. } => rows.len(),
            Command::SetEmbeddings { rows, .. } => rows.len(),
            Command::MarkDecodeFailed { rows, .. } => rows.len(),
            Command::SetProjectionPoints { rows, .. } => rows.len(),
            _ => 0,
        }
    }

    /// Whether the caller must not see a reply until the change is durable.
    ///
    /// Ingest rows are a derived cache -- losing the tail of a batch to a crash costs a
    /// rescan. `library_roots`, `tags`, and `collections` are real user work
    /// (`overview.md` §4.5), so anything touching them commits before it answers.
    fn is_durable(&self) -> bool {
        matches!(
            self,
            Command::AddRoot { .. }
                | Command::SetRootEnabled { .. }
                | Command::RemoveRoot { .. }
                | Command::StartScan { .. }
                | Command::FinishScan { .. }
                // The swap, and the two commands that bracket it. `is_durable` opens one
                // `BEGIN IMMEDIATE` around the command and commits before answering, which
                // is exactly `overview.md` §3.8 step 5's atomic swap -- readers never
                // observe a half-swapped map because there is no moment at which two runs
                // are active or none is.
                | Command::BeginProjectionRun { .. }
                | Command::ActivateProjectionRun { .. }
                | Command::DiscardProjectionRun { .. }
                | Command::Flush { .. }
        )
    }
}

/// The producer-side handle. Cheap to share; every method blocks until the writer has
/// executed the statement (not necessarily until it has committed -- see [`is_durable`]).
///
/// [`is_durable`]: Command::is_durable
#[derive(Debug)]
pub struct WriterHandle {
    tx: SyncSender<Command>,
    metrics: Arc<WriterMetrics>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl WriterHandle {
    /// Opens the write connection and moves it into a dedicated thread.
    pub fn spawn(db_path: &Path) -> Result<Self, DbError> {
        let conn = pool::open_write_connection(db_path)?;
        let metrics = Arc::new(WriterMetrics::default());
        let (tx, rx) = mpsc::sync_channel(QUEUE_DEPTH);

        let writer = Writer {
            conn,
            metrics: Arc::clone(&metrics),
            batch: None,
        };
        let thread = std::thread::Builder::new()
            .name("audiobank-db-writer".into())
            .spawn(move || writer.run(rx))
            .map_err(|e| {
                io_error(
                    format!("spawning the writer thread for {}", db_path.display()),
                    e,
                )
            })?;

        Ok(Self {
            tx,
            metrics,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Counters describing what the writer has committed so far.
    pub fn metrics(&self) -> &WriterMetrics {
        &self.metrics
    }

    /// Inserts or updates sample rows, returning their ids in the order given.
    pub fn upsert_samples(&self, rows: Vec<NewSample>) -> Result<Vec<i64>, DbError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        self.request(|reply| Command::UpsertSamples { rows, reply })
    }

    /// Writes the DSP descriptor block for already-inserted samples.
    pub fn set_features(&self, rows: Vec<(i64, SampleFeatures)>) -> Result<(), DbError> {
        if rows.is_empty() {
            return Ok(());
        }
        self.request(|reply| Command::SetFeatures { rows, reply })
    }

    /// Records where each sample's vector landed in `embeddings.bin` and marks it embedded.
    pub fn set_embeddings(&self, rows: Vec<(i64, EmbeddingLoc)>) -> Result<(), DbError> {
        if rows.is_empty() {
            return Ok(());
        }
        self.request(|reply| Command::SetEmbeddings { rows, reply })
    }

    /// Quarantines samples the decoder could not read, keeping the scan alive.
    pub fn mark_decode_failed(&self, rows: Vec<(i64, String)>) -> Result<(), DbError> {
        if rows.is_empty() {
            return Ok(());
        }
        self.request(|reply| Command::MarkDecodeFailed { rows, reply })
    }

    /// Adds a library root, or returns the existing id if the path is already a root.
    pub fn add_root(&self, path: impl Into<String>, label: Option<String>) -> Result<i64, DbError> {
        let path = path.into();
        self.request(|reply| Command::AddRoot { path, label, reply })
    }

    /// Enables or disables a root without touching its samples.
    pub fn set_root_enabled(&self, root_id: i64, enabled: bool) -> Result<(), DbError> {
        self.request(|reply| Command::SetRootEnabled {
            root_id,
            enabled,
            reply,
        })
    }

    /// Removes a root and, by cascade, its samples, features, and projections.
    pub fn remove_root(&self, root_id: i64) -> Result<(), DbError> {
        self.request(|reply| Command::RemoveRoot { root_id, reply })
    }

    /// Opens a `scan_runs` row and points the root's `last_scan_id` at it.
    pub fn start_scan(&self, root_id: Option<i64>) -> Result<i64, DbError> {
        self.request(|reply| Command::StartScan { root_id, reply })
    }

    /// Closes a `scan_runs` row with its terminal status and counters.
    pub fn finish_scan(
        &self,
        scan_id: i64,
        status: ScanStatus,
        counts: ScanCounts,
        error: Option<String>,
    ) -> Result<(), DbError> {
        self.request(|reply| Command::FinishScan {
            scan_id,
            status,
            counts,
            error,
            reply,
        })
    }

    /// Opens a shadow `projection_runs` row. Not active, not complete: just a home for the
    /// coordinates a re-fit is about to write.
    pub fn begin_projection_run(
        &self,
        algorithm: &str,
        params_json: &str,
        sample_count: i64,
    ) -> Result<i64, DbError> {
        let algorithm = algorithm.to_string();
        let params_json = params_json.to_string();
        self.request(|reply| Command::BeginProjectionRun {
            algorithm,
            params_json,
            sample_count,
            reply,
        })
    }

    /// Writes coordinates into a run. Upserts, so a re-run over the same run replaces
    /// rather than conflicting.
    pub fn set_projection_points(
        &self,
        run_id: i64,
        rows: Vec<(i64, [f32; 3])>,
    ) -> Result<(), DbError> {
        if rows.is_empty() {
            return Ok(());
        }
        self.request(|reply| Command::SetProjectionPoints {
            run_id,
            rows,
            reply,
        })
    }

    /// **The atomic swap** (`overview.md` §3.8, step 5).
    ///
    /// Deactivates whatever was active, activates `run_id`, stamps its `completed_at`, and
    /// drops the runs it supersedes -- all inside the single `BEGIN IMMEDIATE` the writer
    /// wraps a durable command in. A reader either sees the whole new map or the whole old
    /// one.
    pub fn activate_projection_run(&self, run_id: i64) -> Result<(), DbError> {
        self.request(|reply| Command::ActivateProjectionRun { run_id, reply })
    }

    /// Deletes a run and, by cascade, its coordinates.
    ///
    /// What a cancelled or failed re-fit calls on its shadow row. Refuses to delete the
    /// active run: dropping it would leave the app with no map at all, and no caller has a
    /// reason to want that.
    pub fn discard_projection_run(&self, run_id: i64) -> Result<(), DbError> {
        self.request(|reply| Command::DiscardProjectionRun { run_id, reply })
    }

    /// Commits the open batch and returns once it is durable.
    pub fn flush(&self) -> Result<(), DbError> {
        self.request(|reply| Command::Flush { reply })
    }

    /// Commits anything outstanding and stops the thread. Idempotent.
    pub fn shutdown(&self) {
        // A blocking send, not `try_send`: if the queue is full, a dropped stop command
        // would leave the writer parked on `recv()` with a live sender still held here, and
        // the join below would wait forever. `Err` means the thread is already gone, which
        // makes the join return immediately.
        let _ = self.tx.send(Command::Shutdown);
        if let Ok(mut slot) = self.thread.lock() {
            if let Some(handle) = slot.take() {
                if handle.join().is_err() {
                    tracing::error!("the database writer thread panicked");
                }
            }
        }
    }

    /// Sends a command and blocks on its oneshot reply.
    fn request<T>(&self, build: impl FnOnce(Reply<T>) -> Command) -> Result<T, DbError> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.tx.send(build(tx)).map_err(|_| DbError::WriterGone)?;
        rx.recv().map_err(|_| DbError::WriterGone)?
    }
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The state of an open, uncommitted transaction.
#[derive(Debug)]
struct Batch {
    rows: usize,
    /// When the transaction must commit even if it never reaches [`BATCH_ROWS`].
    deadline: Instant,
}

/// The consumer side: owns the connection, runs on its own thread.
#[derive(Debug)]
struct Writer {
    conn: Connection,
    metrics: Arc<WriterMetrics>,
    batch: Option<Batch>,
}

impl Writer {
    fn run(mut self, rx: Receiver<Command>) {
        loop {
            let received = match self.batch.as_ref().map(|b| b.deadline) {
                Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                    Some(remaining) => rx.recv_timeout(remaining),
                    None => {
                        self.flush(FlushReason::Timeout);
                        continue;
                    }
                },
                None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
            };

            match received {
                Ok(Command::Shutdown) => break,
                Ok(cmd) => self.handle(cmd),
                Err(RecvTimeoutError::Timeout) => self.flush(FlushReason::Timeout),
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        self.flush(FlushReason::Forced);
    }

    fn handle(&mut self, cmd: Command) {
        if cmd.is_durable() {
            // Commit whatever is pending so the durable change cannot be rolled back
            // together with a batch that fails later.
            self.flush(FlushReason::Forced);
            self.begin();
            let answer = self.dispatch(cmd);
            self.flush(FlushReason::Forced);
            answer();
            return;
        }

        let rows = cmd.row_count();
        self.begin();
        let answer = self.dispatch(cmd);

        if let Some(batch) = self.batch.as_mut() {
            batch.rows += rows;
            self.metrics.rows.fetch_add(rows as u64, Ordering::Relaxed);
            if batch.rows >= BATCH_ROWS {
                self.flush(FlushReason::Count);
            }
        }

        answer();
    }

    /// Executes one command, returning a closure that answers its caller.
    ///
    /// The answer is deferred rather than sent inline because a caller that has been told
    /// "done" will immediately read its own write back through a pooled connection -- and
    /// a pooled connection cannot see an uncommitted transaction. Answering before the
    /// `COMMIT` makes the writer's own durability guarantee a race. [`Self::handle`] calls
    /// the closure after any commit this command triggered.
    ///
    /// A command that fails part-way may have applied some of its rows: there is no
    /// per-command savepoint, because the rows it writes are all re-derivable by a rescan
    /// and a savepoint per command is a real cost at ingest rates. The caller sees the
    /// error either way.
    fn dispatch(&mut self, cmd: Command) -> Answer {
        let conn = &self.conn;
        match cmd {
            Command::UpsertSamples { rows, reply } => answer(reply, upsert_samples(conn, &rows)),
            Command::SetFeatures { rows, reply } => answer(reply, set_features(conn, &rows)),
            Command::SetEmbeddings { rows, reply } => answer(reply, set_embeddings(conn, &rows)),
            Command::MarkDecodeFailed { rows, reply } => {
                answer(reply, mark_decode_failed(conn, &rows))
            }
            Command::AddRoot { path, label, reply } => {
                answer(reply, add_root(conn, &path, label.as_deref()))
            }
            Command::SetRootEnabled {
                root_id,
                enabled,
                reply,
            } => answer(reply, set_root_enabled(conn, root_id, enabled)),
            Command::RemoveRoot { root_id, reply } => answer(reply, remove_root(conn, root_id)),
            Command::StartScan { root_id, reply } => answer(reply, start_scan(conn, root_id)),
            Command::FinishScan {
                scan_id,
                status,
                counts,
                error,
                reply,
            } => answer(
                reply,
                finish_scan(conn, scan_id, status, counts, error.as_deref()),
            ),
            Command::BeginProjectionRun {
                algorithm,
                params_json,
                sample_count,
                reply,
            } => answer(
                reply,
                begin_projection_run(conn, &algorithm, &params_json, sample_count),
            ),
            Command::SetProjectionPoints {
                run_id,
                rows,
                reply,
            } => answer(reply, set_projection_points(conn, run_id, &rows)),
            Command::ActivateProjectionRun { run_id, reply } => {
                answer(reply, activate_projection_run(conn, run_id))
            }
            Command::DiscardProjectionRun { run_id, reply } => {
                answer(reply, discard_projection_run(conn, run_id))
            }
            Command::Flush { reply } => answer(reply, Ok(())),
            Command::Shutdown => Box::new(|| {}),
        }
    }

    /// Opens a transaction if none is open. `IMMEDIATE` so the write lock is taken now
    /// rather than on the first write, turning a mid-transaction `SQLITE_BUSY` into a
    /// wait at a point where waiting is harmless.
    fn begin(&mut self) {
        if self.batch.is_some() {
            return;
        }
        match self.conn.execute_batch("BEGIN IMMEDIATE") {
            Ok(()) => {
                self.batch = Some(Batch {
                    rows: 0,
                    deadline: Instant::now() + BATCH_INTERVAL,
                });
            }
            Err(e) => tracing::error!(error = %e, "could not open a write transaction"),
        }
    }

    fn flush(&mut self, reason: FlushReason) {
        let Some(batch) = self.batch.take() else {
            return;
        };

        match self.conn.execute_batch("COMMIT") {
            Ok(()) => {
                self.metrics.commits.fetch_add(1, Ordering::Relaxed);
                let counter = match reason {
                    FlushReason::Count => &self.metrics.flushes_by_count,
                    FlushReason::Timeout => &self.metrics.flushes_by_timeout,
                    FlushReason::Forced => &self.metrics.flushes_forced,
                };
                counter.fetch_add(1, Ordering::Relaxed);
                tracing::trace!(rows = batch.rows, ?reason, "committed");
            }
            Err(e) => {
                tracing::error!(error = %e, rows = batch.rows, "commit failed, rolling back");
                if let Err(e) = self.conn.execute_batch("ROLLBACK") {
                    tracing::error!(error = %e, "rollback failed");
                }
            }
        }
    }
}

/// A pending answer to one caller, held until the writer decides it is safe to send.
type Answer = Box<dyn FnOnce()>;

/// Packages a result as an [`Answer`], tolerating a caller that has already hung up.
fn answer<T: 'static>(reply: Reply<T>, result: Result<T, DbError>) -> Answer {
    Box::new(move || {
        if let Err(e) = result.as_ref() {
            tracing::warn!(error = %e, "write command failed");
        }
        let _ = reply.send(result);
    })
}

fn upsert_samples(conn: &Connection, rows: &[NewSample]) -> Result<Vec<i64>, DbError> {
    let now = now_ms();
    let mut ids = Vec::with_capacity(rows.len());

    let mut find =
        conn.prepare_cached("SELECT id FROM samples WHERE root_id = ?1 AND rel_path = ?2")?;
    let mut insert = conn.prepare_cached(
        "INSERT INTO samples (
             root_id, rel_path, filename, ext, size_bytes, mtime, content_hash,
             duration_ms, sample_rate, channels, status, first_seen_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)",
    )?;
    // COALESCE on the optional columns: a `None` from this stage means "not determined
    // yet", not "known to be absent", and must not erase what an earlier stage learned.
    //
    // `emb_offset` and `emb_len` are the exception, and they are cleared rather than
    // coalesced. A row only reaches this statement because the file behind it changed --
    // an unchanged one is fast-skipped in the walker and never re-upserted -- so whatever
    // vector those offsets point at describes audio this file no longer contains. Left in
    // place they would survive as a stale-but-plausible embedding, and
    // `queries::embedded_sample_by_hash` would hand it to every duplicate of the *new*
    // content. The fresh vector lands moments later in the same batch.
    let mut update = conn.prepare_cached(
        "UPDATE samples SET
             filename     = ?2,
             ext          = ?3,
             size_bytes   = ?4,
             mtime        = ?5,
             content_hash = COALESCE(?6, content_hash),
             duration_ms  = COALESCE(?7, duration_ms),
             sample_rate  = COALESCE(?8, sample_rate),
             channels     = COALESCE(?9, channels),
             status       = ?10,
             emb_offset   = NULL,
             emb_len      = NULL,
             error        = NULL,
             updated_at   = ?11
         WHERE id = ?1",
    )?;
    // Contentless FTS: the row is written once, at insert. `filename` is a function of
    // `rel_path`, which together with `root_id` *is* the row's identity, so an update can
    // never change it. Tag maintenance (Phase 9) owns the `tags` column.
    let mut fts =
        conn.prepare_cached("INSERT INTO samples_fts (rowid, filename, tags) VALUES (?1, ?2, '')")?;

    for row in rows {
        let hash = row.content_hash.as_ref().map(|h| h.as_slice());
        let existing: Option<i64> = find
            .query_row((row.root_id, &row.rel_path), |r| r.get(0))
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;

        let id = match existing {
            Some(id) => {
                update.execute((
                    id,
                    &row.filename,
                    &row.ext,
                    row.size_bytes,
                    row.mtime,
                    hash,
                    row.duration_ms,
                    row.sample_rate,
                    row.channels,
                    row.status.as_str(),
                    now,
                ))?;
                id
            }
            None => {
                insert.execute((
                    row.root_id,
                    &row.rel_path,
                    &row.filename,
                    &row.ext,
                    row.size_bytes,
                    row.mtime,
                    hash,
                    row.duration_ms,
                    row.sample_rate,
                    row.channels,
                    row.status.as_str(),
                    now,
                ))?;
                let id = conn.last_insert_rowid();
                fts.execute((id, &row.filename))?;
                id
            }
        };
        ids.push(id);
    }

    Ok(ids)
}

fn set_features(conn: &Connection, rows: &[(i64, SampleFeatures)]) -> Result<(), DbError> {
    let mut stmt = conn.prepare_cached(
        "INSERT INTO sample_features (
             sample_id, peak_db, rms_db, lufs_integrated, spectral_centroid,
             spectral_flatness, zero_crossing, onset_density, bpm, bpm_confidence,
             key_root, key_mode, key_confidence
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
         ON CONFLICT(sample_id) DO UPDATE SET
             peak_db           = excluded.peak_db,
             rms_db            = excluded.rms_db,
             lufs_integrated   = excluded.lufs_integrated,
             spectral_centroid = excluded.spectral_centroid,
             spectral_flatness = excluded.spectral_flatness,
             zero_crossing     = excluded.zero_crossing,
             onset_density     = excluded.onset_density,
             bpm               = excluded.bpm,
             bpm_confidence    = excluded.bpm_confidence,
             key_root          = excluded.key_root,
             key_mode          = excluded.key_mode,
             key_confidence    = excluded.key_confidence",
    )?;

    for (sample_id, f) in rows {
        stmt.execute((
            sample_id,
            f.peak_db,
            f.rms_db,
            f.lufs_integrated,
            f.spectral_centroid,
            f.spectral_flatness,
            f.zero_crossing,
            f.onset_density,
            f.bpm,
            f.bpm_confidence,
            f.key_root,
            f.key_mode,
            f.key_confidence,
        ))?;
    }

    Ok(())
}

fn set_embeddings(conn: &Connection, rows: &[(i64, EmbeddingLoc)]) -> Result<(), DbError> {
    let now = now_ms();
    let mut stmt = conn.prepare_cached(
        "UPDATE samples SET emb_offset = ?2, emb_len = ?3, status = ?4, updated_at = ?5
         WHERE id = ?1",
    )?;

    for (sample_id, loc) in rows {
        stmt.execute((
            sample_id,
            loc.offset as i64,
            loc.dims as i64,
            SampleStatus::Embedded.as_str(),
            now,
        ))?;
    }

    Ok(())
}

fn mark_decode_failed(conn: &Connection, rows: &[(i64, String)]) -> Result<(), DbError> {
    let now = now_ms();
    let mut stmt = conn.prepare_cached(
        "UPDATE samples SET status = ?2, error = ?3, updated_at = ?4 WHERE id = ?1",
    )?;

    for (sample_id, error) in rows {
        stmt.execute((sample_id, SampleStatus::DecodeFailed.as_str(), error, now))?;
    }

    Ok(())
}

fn begin_projection_run(
    conn: &Connection,
    algorithm: &str,
    params_json: &str,
    sample_count: i64,
) -> Result<i64, DbError> {
    let id = conn.query_row(
        "INSERT INTO projection_runs (algorithm, params_json, sample_count, created_at, is_active)
         VALUES (?1, ?2, ?3, ?4, 0) RETURNING id",
        (algorithm, params_json, sample_count, now_ms()),
        |r| r.get(0),
    )?;
    Ok(id)
}

fn set_projection_points(
    conn: &Connection,
    run_id: i64,
    rows: &[(i64, [f32; 3])],
) -> Result<(), DbError> {
    let mut stmt = conn.prepare_cached(
        "INSERT INTO projections (run_id, sample_id, x, y, z) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(run_id, sample_id) DO UPDATE SET
             x = excluded.x, y = excluded.y, z = excluded.z",
    )?;
    for (sample_id, [x, y, z]) in rows {
        stmt.execute((run_id, sample_id, x, y, z))?;
    }
    Ok(())
}

/// Clear, set, prune -- in that order, in one transaction.
///
/// The order is forced by the schema. `idx_projection_active` is a partial unique index on
/// `is_active = 1`, so setting the new flag before clearing the old one is a constraint
/// violation rather than a momentary inconsistency. That the index makes the wrong order
/// *fail* rather than *corrupt* is the whole reason it is there.
///
/// The prune is deliberately limited to runs that finished. A shadow run being built by
/// another job has `completed_at IS NULL` and survives, so activating one re-fit cannot
/// delete another re-fit's work out from under it.
fn activate_projection_run(conn: &Connection, run_id: i64) -> Result<(), DbError> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM projection_runs WHERE id = ?1",
        [run_id],
        |r| r.get(0),
    )?;
    if exists == 0 {
        return Err(DbError::Sqlite(rusqlite::Error::QueryReturnedNoRows));
    }

    conn.execute(
        "UPDATE projection_runs SET is_active = 0 WHERE is_active = 1 AND id <> ?1",
        [run_id],
    )?;
    conn.execute(
        "UPDATE projection_runs SET is_active = 1, completed_at = COALESCE(completed_at, ?2)
         WHERE id = ?1",
        (run_id, now_ms()),
    )?;
    conn.execute(
        "DELETE FROM projection_runs WHERE id <> ?1 AND completed_at IS NOT NULL",
        [run_id],
    )?;
    Ok(())
}

fn discard_projection_run(conn: &Connection, run_id: i64) -> Result<(), DbError> {
    conn.execute(
        "DELETE FROM projection_runs WHERE id = ?1 AND is_active = 0",
        [run_id],
    )?;
    Ok(())
}

fn add_root(conn: &Connection, path: &str, label: Option<&str>) -> Result<i64, DbError> {
    let id = conn.query_row(
        "INSERT INTO library_roots (path, label, enabled, added_at) VALUES (?1, ?2, 1, ?3)
         ON CONFLICT(path) DO UPDATE SET label = COALESCE(?2, label)
         RETURNING id",
        (path, label, now_ms()),
        |r| r.get(0),
    )?;
    Ok(id)
}

fn set_root_enabled(conn: &Connection, root_id: i64, enabled: bool) -> Result<(), DbError> {
    conn.execute(
        "UPDATE library_roots SET enabled = ?2 WHERE id = ?1",
        (root_id, i64::from(enabled)),
    )?;
    Ok(())
}

fn remove_root(conn: &Connection, root_id: i64) -> Result<(), DbError> {
    conn.execute("DELETE FROM library_roots WHERE id = ?1", [root_id])?;
    Ok(())
}

fn start_scan(conn: &Connection, root_id: Option<i64>) -> Result<i64, DbError> {
    let scan_id: i64 = conn.query_row(
        "INSERT INTO scan_runs (root_id, started_at, status) VALUES (?1, ?2, ?3) RETURNING id",
        (root_id, now_ms(), ScanStatus::Running.as_str()),
        |r| r.get(0),
    )?;

    if let Some(root_id) = root_id {
        conn.execute(
            "UPDATE library_roots SET last_scan_id = ?2 WHERE id = ?1",
            (root_id, scan_id),
        )?;
    }

    Ok(scan_id)
}

fn finish_scan(
    conn: &Connection,
    scan_id: i64,
    status: ScanStatus,
    counts: ScanCounts,
    error: Option<&str>,
) -> Result<(), DbError> {
    conn.execute(
        "UPDATE scan_runs SET
             finished_at   = ?2,
             files_seen    = ?3,
             files_added   = ?4,
             files_skipped = ?5,
             files_failed  = ?6,
             status        = ?7,
             error         = ?8
         WHERE id = ?1",
        (
            scan_id,
            now_ms(),
            counts.files_seen,
            counts.files_added,
            counts.files_skipped,
            counts.files_failed,
            status.as_str(),
            error,
        ),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::thread::sleep;

    use super::super::{queries, test_support::*, SampleFeatures, SampleStatus};
    use super::*;

    /// Rows committed so far, seen through a *separate* read connection -- which in WAL
    /// mode cannot see the writer's open transaction. That is what makes this a test of
    /// commit timing rather than of statement execution.
    fn committed(db: &super::super::Database) -> i64 {
        queries::count_samples(&db.read().unwrap()).unwrap()
    }

    /// Batch trigger 1: 1000 rows.
    #[test]
    fn a_full_batch_commits_on_the_row_count() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();

        let before = writer.metrics().flushes_by_count();
        let rows: Vec<_> = (0..BATCH_ROWS)
            .map(|i| sample(root, &format!("kick_{i:04}.wav")))
            .collect();
        writer.upsert_samples(rows).unwrap();

        // No sleep: reaching BATCH_ROWS must commit before the writer takes another
        // command, so the rows are already visible.
        assert_eq!(committed(&db), BATCH_ROWS as i64);
        assert_eq!(writer.metrics().flushes_by_count(), before + 1);
    }

    /// Batch trigger 2: 250 ms. Without it the tail of a scan sits uncommitted.
    #[test]
    fn a_partial_batch_commits_on_the_timeout() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();

        let before = writer.metrics().flushes_by_timeout();
        let rows: Vec<_> = (0..10)
            .map(|i| sample(root, &format!("snare_{i}.wav")))
            .collect();
        writer.upsert_samples(rows).unwrap();

        assert_eq!(
            committed(&db),
            0,
            "a ten-row batch should still be open immediately after the call returns"
        );

        sleep(BATCH_INTERVAL * 2);

        assert_eq!(committed(&db), 10, "the timeout flush never fired");
        assert!(writer.metrics().flushes_by_timeout() > before);
    }

    /// Real user work must not sit in an open batch waiting for a timeout it may not
    /// survive: `add_root` answers only once its row is durable.
    #[test]
    fn a_durable_command_commits_before_it_replies() {
        let (_dir, db) = temp_db();
        let root = db
            .writer()
            .add_root("/Library/Audio/Samples", None)
            .unwrap();

        let roots = queries::library_roots(&db.read().unwrap()).unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, root);
        assert!(roots[0].enabled);
    }

    #[test]
    fn adding_the_same_root_twice_returns_the_same_id() {
        let (_dir, db) = temp_db();
        let writer = db.writer();

        let first = writer.add_root("/samples", None).unwrap();
        let second = writer.add_root("/samples", Some("Drums".into())).unwrap();

        assert_eq!(first, second);
        let roots = queries::library_roots(&db.read().unwrap()).unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].label.as_deref(), Some("Drums"));
    }

    /// A rescan re-upserts every row it sees. Ids have to be stable, or every foreign key
    /// pointing at a sample breaks on the second scan.
    #[test]
    fn re_upserting_a_sample_keeps_its_id_and_its_search_row() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();

        let first = writer
            .upsert_samples(vec![sample(root, "drums/KICK_808_Distorted-02.wav")])
            .unwrap();

        let mut changed = sample(root, "drums/KICK_808_Distorted-02.wav");
        changed.size_bytes = 4096;
        changed.status = SampleStatus::Decoded;
        let second = writer.upsert_samples(vec![changed]).unwrap();
        writer.flush().unwrap();

        assert_eq!(first, second);

        let conn = db.read().unwrap();
        assert_eq!(queries::count_samples(&conn).unwrap(), 1);

        // One FTS row, not two: the search index must not accumulate a duplicate per scan.
        let hits = queries::search_samples(&conn, "808", 10).unwrap();
        assert_eq!(hits, first);

        let stamp = queries::sample_stamp(&conn, root, "drums/KICK_808_Distorted-02.wav")
            .unwrap()
            .unwrap();
        assert_eq!(stamp.size_bytes, 4096);
        assert_eq!(stamp.status, SampleStatus::Decoded);
        assert!(stamp.unchanged(1_700_000_000, 4096));
    }

    /// Both halves of what §4.1's prose asks for: a fragment of a compound filename hits as
    /// a bareword, and the whole compound hits as a quoted phrase. See the tokenizer note in
    /// `V1__initial.sql` for why the DDL had to deviate to get this.
    #[test]
    fn search_finds_a_sample_by_a_fragment_of_its_filename() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![
                sample(root, "KICK_808_Distorted-02.wav"),
                sample(root, "Hat_Closed_01.wav"),
            ])
            .unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        assert_eq!(queries::search_samples(&conn, "808", 10).unwrap(), [ids[0]]);
        assert_eq!(
            queries::search_samples(&conn, "closed", 10).unwrap(),
            [ids[1]],
            "the tokenizer should split on underscores and ignore case"
        );
        assert_eq!(
            queries::search_samples(&conn, "\"KICK_808_Distorted-02.wav\"", 10).unwrap(),
            [ids[0]],
            "the compound form should match as a phrase"
        );
        assert!(queries::search_samples(&conn, "cowbell", 10)
            .unwrap()
            .is_empty());
    }

    /// An upsert must not erase what a later stage already learned about the row.
    #[test]
    fn an_upsert_preserves_columns_the_caller_left_unset() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();

        let mut hashed = sample(root, "loop.wav");
        hashed.content_hash = Some([7u8; 32]);
        let ids = writer.upsert_samples(vec![hashed]).unwrap();
        writer.flush().unwrap();

        // A rescan that has not hashed the file yet passes `None`.
        writer
            .upsert_samples(vec![sample(root, "loop.wav")])
            .unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        let hash: Option<Vec<u8>> = conn
            .query_row(
                "SELECT content_hash FROM samples WHERE id = ?1",
                [ids[0]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hash, Some(vec![7u8; 32]));
    }

    #[test]
    fn features_are_written_and_overwritten_as_a_block() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![sample(root, "clap.wav")])
            .unwrap();

        let features = SampleFeatures {
            peak_db: Some(-1.5),
            rms_db: Some(-12.0),
            bpm: Some(128.0),
            key_root: Some(9),
            key_mode: Some(0),
            ..SampleFeatures::default()
        };
        writer
            .set_features(vec![(ids[0], features.clone())])
            .unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        assert_eq!(
            queries::sample_features(&conn, ids[0]).unwrap(),
            Some(features)
        );

        let revised = SampleFeatures {
            bpm: Some(174.0),
            ..SampleFeatures::default()
        };
        writer
            .set_features(vec![(ids[0], revised.clone())])
            .unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        assert_eq!(
            queries::sample_features(&conn, ids[0]).unwrap(),
            Some(revised),
            "a second write should replace the block, not merge into it"
        );
    }

    /// Phase 2 quarantines a file it cannot decode and keeps scanning. The row has to
    /// survive with the reason attached.
    #[test]
    fn a_decode_failure_is_recorded_without_losing_the_row() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![sample(root, "truncated.wav")])
            .unwrap();

        writer
            .mark_decode_failed(vec![(ids[0], "unsupported codec: adpcm".into())])
            .unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        let (status, error): (String, Option<String>) = conn
            .query_row(
                "SELECT status, error FROM samples WHERE id = ?1",
                [ids[0]],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(status, SampleStatus::DecodeFailed.as_str());
        assert_eq!(error.as_deref(), Some("unsupported codec: adpcm"));
        assert_eq!(
            queries::count_samples_with_status(&conn, SampleStatus::DecodeFailed).unwrap(),
            1
        );
    }

    #[test]
    fn embedding_locations_mark_the_sample_embedded() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![sample(root, "pad.wav")])
            .unwrap();

        let loc = EmbeddingLoc {
            offset: 2048,
            dims: TEST_DIM as u32,
        };
        writer.set_embeddings(vec![(ids[0], loc)]).unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        assert_eq!(queries::embedding_loc(&conn, ids[0]).unwrap(), Some(loc));
        assert_eq!(
            queries::count_samples_with_status(&conn, SampleStatus::Embedded).unwrap(),
            1
        );
    }

    #[test]
    fn a_scan_run_records_its_outcome_and_points_the_root_at_it() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();

        let scan = writer.start_scan(Some(root)).unwrap();
        writer
            .finish_scan(
                scan,
                ScanStatus::Cancelled,
                ScanCounts {
                    files_seen: 42,
                    files_added: 40,
                    files_skipped: 1,
                    files_failed: 1,
                },
                None,
            )
            .unwrap();

        let conn = db.read().unwrap();
        let (status, seen, failed, finished): (String, i64, i64, Option<i64>) = conn
            .query_row(
                "SELECT status, files_seen, files_failed, finished_at FROM scan_runs WHERE id = ?1",
                [scan],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap();
        assert_eq!(status, "cancelled");
        assert_eq!(seen, 42);
        assert_eq!(failed, 1);
        assert!(finished.is_some());

        let roots = queries::library_roots(&conn).unwrap();
        assert_eq!(roots[0].last_scan_id, Some(scan));
    }

    /// Cross-cutting rule 2, made mechanical: the pool cannot produce a write connection
    /// even when asked directly.
    #[test]
    fn pooled_connections_cannot_write() {
        let (_dir, db) = temp_db();
        let conn = db.read().unwrap();
        let err = conn
            .execute("INSERT INTO tags (name) VALUES ('kick')", [])
            .expect_err("a pooled connection must be read-only");
        assert!(
            err.to_string().contains("readonly"),
            "unexpected error: {err}"
        );
    }

    /// Every producer thread shares one writer; the point of the channel is that they can
    /// do so without a `SQLITE_BUSY` storm.
    #[test]
    fn concurrent_producers_all_land() {
        let (_dir, db) = temp_db();
        let root = db.writer().add_root("/samples", None).unwrap();

        std::thread::scope(|scope| {
            for thread in 0..8 {
                let writer = db.writer();
                scope.spawn(move || {
                    for i in 0..100 {
                        writer
                            .upsert_samples(vec![sample(root, &format!("t{thread}/s{i}.wav"))])
                            .unwrap();
                    }
                });
            }
        });

        db.writer().flush().unwrap();
        assert_eq!(committed(&db), 800);
    }

    /// The invariant the schema enforces and the writer must not violate: setting the new
    /// active flag before clearing the old one is a constraint violation, not a momentary
    /// inconsistency. This is the test that fails if the two `UPDATE`s ever swap places.
    #[test]
    fn activating_a_run_deactivates_the_one_it_replaces() {
        let (_dir, db) = temp_db();
        let writer = db.writer();

        let first = writer.begin_projection_run("pca", "{}", 3).unwrap();
        writer.activate_projection_run(first).unwrap();
        let second = writer.begin_projection_run("umap", "{}", 3).unwrap();
        writer.activate_projection_run(second).unwrap();

        let conn = db.read().unwrap();
        let active: Vec<i64> = conn
            .prepare("SELECT id FROM projection_runs WHERE is_active = 1")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(active, vec![second]);

        let run = queries::projection_run(&conn, second).unwrap().unwrap();
        assert_eq!(run.algorithm, "umap");
        assert!(run.completed_at.is_some(), "an active run must be complete");
    }

    /// Activation prunes what it supersedes -- but only runs that finished. A shadow run
    /// another job is still filling has no `completed_at` and must survive, or two
    /// concurrent re-fits delete each other's work.
    #[test]
    fn activation_prunes_finished_runs_and_spares_shadows() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![sample(root, "a.wav"), sample(root, "b.wav")])
            .unwrap();

        let old = writer.begin_projection_run("pca", "{}", 2).unwrap();
        writer
            .set_projection_points(old, vec![(ids[0], [1.0, 2.0, 3.0])])
            .unwrap();
        writer.activate_projection_run(old).unwrap();

        let shadow = writer.begin_projection_run("umap", "{}", 2).unwrap();
        let fresh = writer.begin_projection_run("pca", "{}", 2).unwrap();
        writer
            .set_projection_points(fresh, vec![(ids[1], [4.0, 5.0, 6.0])])
            .unwrap();
        writer.activate_projection_run(fresh).unwrap();

        let conn = db.read().unwrap();
        let ids_left: Vec<i64> = queries::projection_runs(&conn)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert!(ids_left.contains(&fresh), "the new run was pruned");
        assert!(
            ids_left.contains(&shadow),
            "an unfinished shadow run was pruned"
        );
        assert!(!ids_left.contains(&old), "the superseded run survived");

        // And the pruned run's coordinates went with it, by cascade.
        assert_eq!(queries::count_projection_points(&conn, old).unwrap(), 0);
        assert_eq!(
            queries::active_projection_points(&conn).unwrap(),
            vec![(ids[1], [4.0, 5.0, 6.0])]
        );
    }

    /// Discarding is for shadows. The active run is the map the user is looking at, and no
    /// caller has a reason to want it gone.
    #[test]
    fn the_active_run_cannot_be_discarded() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let run = writer.begin_projection_run("pca", "{}", 0).unwrap();
        writer.activate_projection_run(run).unwrap();

        writer.discard_projection_run(run).unwrap();

        let conn = db.read().unwrap();
        assert_eq!(
            queries::active_projection_run(&conn).unwrap().map(|r| r.id),
            Some(run)
        );
    }
}
