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
    SetProjectionColors {
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
    SetTag {
        sample_id: i64,
        tag_name: String,
        reply: Reply<()>,
    },
    UnsetTag {
        sample_id: i64,
        tag_name: String,
        reply: Reply<()>,
    },
    SetTagColor {
        tag_id: i64,
        color: Option<String>,
        reply: Reply<()>,
    },
    CreateCollection {
        name: String,
        sample_ids: Vec<i64>,
        reply: Reply<i64>,
    },
    ReorderCollection {
        collection_id: i64,
        sample_ids: Vec<i64>,
        reply: Reply<()>,
    },
    DeleteCollection {
        collection_id: i64,
        reply: Reply<()>,
    },
    SetSetting {
        key: String,
        value: String,
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
            Command::SetProjectionColors { rows, .. } => rows.len(),
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
                // Tags and collections are real user work, not a derived cache. A tag the
                // user typed and a crash a moment later must not be a tag they have to type
                // again, and the FTS reindex that accompanies it has to land in the same
                // transaction or search would disagree with the sidebar.
                | Command::SetTag { .. }
                | Command::UnsetTag { .. }
                | Command::SetTagColor { .. }
                | Command::CreateCollection { .. }
                | Command::ReorderCollection { .. }
                | Command::DeleteCollection { .. }
                // A setting a user just changed in the Settings panel must not appear to
                // have taken effect and then evaporate on a crash before the next batch
                // flushes -- the same reasoning as a tag.
                | Command::SetSetting { .. }
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

    /// Writes colors into a run whose points already exist -- an `UPDATE`, not an upsert,
    /// since a color with no position to attach to is not a state this ever produces.
    pub fn set_projection_colors(
        &self,
        run_id: i64,
        rows: Vec<(i64, [f32; 3])>,
    ) -> Result<(), DbError> {
        if rows.is_empty() {
            return Ok(());
        }
        self.request(|reply| Command::SetProjectionColors {
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

    /// Attaches a tag to a sample, creating the tag if this is its first use.
    ///
    /// Idempotent: tagging an already-tagged sample succeeds and changes nothing.
    pub fn set_tag(&self, sample_id: i64, tag_name: impl Into<String>) -> Result<(), DbError> {
        let tag_name = tag_name.into();
        self.request(|reply| Command::SetTag {
            sample_id,
            tag_name,
            reply,
        })
    }

    /// Removes a tag from a sample. The tag itself survives with a count of zero -- see
    /// [`queries::all_tags`].
    ///
    /// [`queries::all_tags`]: super::queries::all_tags
    pub fn unset_tag(&self, sample_id: i64, tag_name: impl Into<String>) -> Result<(), DbError> {
        let tag_name = tag_name.into();
        self.request(|reply| Command::UnsetTag {
            sample_id,
            tag_name,
            reply,
        })
    }

    /// Sets (or clears, for `None`) a tag's display color.
    pub fn set_tag_color(&self, tag_id: i64, color: Option<String>) -> Result<(), DbError> {
        self.request(|reply| Command::SetTagColor {
            tag_id,
            color,
            reply,
        })
    }

    /// Creates a collection holding the given samples, in the order given.
    pub fn create_collection(
        &self,
        name: impl Into<String>,
        sample_ids: Vec<i64>,
    ) -> Result<i64, DbError> {
        let name = name.into();
        self.request(|reply| Command::CreateCollection {
            name,
            sample_ids,
            reply,
        })
    }

    /// Rewrites a collection's member order to match `sample_ids`.
    ///
    /// The caller -- `commands::collections::reorder_collection` -- has already checked
    /// `sample_ids` is exactly the collection's current membership; this just writes the new
    /// positions.
    pub fn reorder_collection(
        &self,
        collection_id: i64,
        sample_ids: Vec<i64>,
    ) -> Result<(), DbError> {
        self.request(|reply| Command::ReorderCollection {
            collection_id,
            sample_ids,
            reply,
        })
    }

    /// Deletes a collection and, by cascade, its membership rows. The samples themselves are
    /// untouched -- a collection is a saved arrangement, not ownership.
    pub fn delete_collection(&self, collection_id: i64) -> Result<(), DbError> {
        self.request(|reply| Command::DeleteCollection {
            collection_id,
            reply,
        })
    }

    /// Writes one key in the `app_settings` table, creating or overwriting it.
    pub fn set_setting(
        &self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), DbError> {
        let key = key.into();
        let value = value.into();
        self.request(|reply| Command::SetSetting { key, value, reply })
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
            Command::SetProjectionColors {
                run_id,
                rows,
                reply,
            } => answer(reply, set_projection_colors(conn, run_id, &rows)),
            Command::ActivateProjectionRun { run_id, reply } => {
                answer(reply, activate_projection_run(conn, run_id))
            }
            Command::DiscardProjectionRun { run_id, reply } => {
                answer(reply, discard_projection_run(conn, run_id))
            }
            Command::SetTag {
                sample_id,
                tag_name,
                reply,
            } => answer(reply, set_tag(conn, sample_id, &tag_name)),
            Command::UnsetTag {
                sample_id,
                tag_name,
                reply,
            } => answer(reply, unset_tag(conn, sample_id, &tag_name)),
            Command::SetTagColor {
                tag_id,
                color,
                reply,
            } => answer(reply, set_tag_color(conn, tag_id, color.as_deref())),
            Command::CreateCollection {
                name,
                sample_ids,
                reply,
            } => answer(reply, create_collection(conn, &name, &sample_ids)),
            Command::ReorderCollection {
                collection_id,
                sample_ids,
                reply,
            } => answer(reply, reorder_collection(conn, collection_id, &sample_ids)),
            Command::DeleteCollection {
                collection_id,
                reply,
            } => answer(reply, delete_collection(conn, collection_id)),
            Command::SetSetting { key, value, reply } => {
                answer(reply, set_setting(conn, &key, &value))
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
    // Contentless FTS: the row is written once, at insert. Both `filename` and `path` are
    // functions of `rel_path`, which together with `root_id` *is* the row's identity, so an
    // update can never change either. Tag maintenance (Phase 9) owns the `tags` column.
    let mut fts = conn.prepare_cached(
        "INSERT INTO samples_fts (rowid, filename, path, tags) VALUES (?1, ?2, ?3, '')",
    )?;

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
                fts.execute((id, &row.filename, &row.rel_path))?;
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

fn set_projection_colors(
    conn: &Connection,
    run_id: i64,
    rows: &[(i64, [f32; 3])],
) -> Result<(), DbError> {
    let mut stmt = conn.prepare_cached(
        "UPDATE projections SET r = ?1, g = ?2, b = ?3 WHERE run_id = ?4 AND sample_id = ?5",
    )?;
    for (sample_id, [r, g, b]) in rows {
        stmt.execute((r, g, b, run_id, sample_id))?;
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

/// The tag names indexed against one sample: joined by a space, ordered by name,
/// case-insensitively.
///
/// `V7__fts_index_path.sql`'s rebuild reproduces this expression in SQL, so the two have to
/// agree on the ordering and the separator or a migrated row indexes differently from one
/// written by a retag.
fn fts_tag_text(conn: &Connection, sample_id: i64) -> Result<String, DbError> {
    let mut stmt = conn.prepare_cached(
        "SELECT t.name FROM tags t
         JOIN sample_tags st ON st.tag_id = t.id
         WHERE st.sample_id = ?1
         ORDER BY t.name COLLATE NOCASE",
    )?;
    let names = stmt
        .query_map([sample_id], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(names.join(" "))
}

/// Rewrites one sample's `samples_fts` row so search sees the tags the sidebar shows.
///
/// A plain `DELETE` then an `INSERT`, which is only possible because V7 recreated the table
/// with `contentless_delete = 1`. Before that a contentless row was removed by re-supplying
/// the exact text it had been indexed with, so this took an `old_tags` argument that every
/// caller had to read *before* making its change -- and a caller that got it wrong corrupted
/// the index silently rather than failing. Nothing here needs to know what it is replacing
/// any more.
fn reindex_sample(conn: &Connection, sample_id: i64) -> Result<(), DbError> {
    let (filename, path): (String, String) = conn.query_row(
        "SELECT filename, rel_path FROM samples WHERE id = ?1",
        [sample_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;

    conn.prepare_cached("DELETE FROM samples_fts WHERE rowid = ?1")?
        .execute([sample_id])?;

    let tags = fts_tag_text(conn, sample_id)?;
    conn.prepare_cached(
        "INSERT INTO samples_fts (rowid, filename, path, tags) VALUES (?1, ?2, ?3, ?4)",
    )?
    .execute((sample_id, &filename, &path, &tags))?;

    Ok(())
}

/// Attaches a tag, creating it on first use.
///
/// The `ON CONFLICT DO NOTHING` plus a separate `SELECT` rather than `RETURNING id`: the
/// name column is `COLLATE NOCASE`, so "Kick" and "kick" are the same tag, and `RETURNING`
/// on a no-op conflict returns nothing at all. Two statements say what is meant.
fn set_tag(conn: &Connection, sample_id: i64, tag_name: &str) -> Result<(), DbError> {
    let name = tag_name.trim();
    if name.is_empty() {
        return Ok(());
    }

    conn.prepare_cached("INSERT INTO tags (name) VALUES (?1) ON CONFLICT(name) DO NOTHING")?
        .execute([name])?;
    let tag_id: i64 =
        conn.query_row("SELECT id FROM tags WHERE name = ?1", [name], |r| r.get(0))?;

    // The foreign key is what rejects a tag on a sample that does not exist, so an unknown
    // id is a constraint violation with a real message rather than a silent no-op.
    conn.prepare_cached(
        "INSERT INTO sample_tags (sample_id, tag_id) VALUES (?1, ?2)
         ON CONFLICT(sample_id, tag_id) DO NOTHING",
    )?
    .execute((sample_id, tag_id))?;

    reindex_sample(conn, sample_id)
}

/// Detaches a tag. The `tags` row survives with a count of zero -- a tag the user invented
/// and then cleared off every sample is still a tag they invented.
fn unset_tag(conn: &Connection, sample_id: i64, tag_name: &str) -> Result<(), DbError> {
    let name = tag_name.trim();
    let removed = conn
        .prepare_cached(
            "DELETE FROM sample_tags
             WHERE sample_id = ?1
               AND tag_id = (SELECT id FROM tags WHERE name = ?2)",
        )?
        .execute((sample_id, name))?;

    if removed == 0 {
        return Ok(());
    }
    reindex_sample(conn, sample_id)
}

/// Sets or clears one tag's display color.
fn set_tag_color(conn: &Connection, tag_id: i64, color: Option<&str>) -> Result<(), DbError> {
    conn.prepare_cached("UPDATE tags SET color = ?2 WHERE id = ?1")?
        .execute((tag_id, color))?;
    Ok(())
}

/// Creates a collection over the given samples, preserving the order they were given in.
///
/// `position` is that order, and it is the reason a collection is not just a tag: a tag is a
/// set, and a collection is a sequence the user arranged.
fn create_collection(conn: &Connection, name: &str, sample_ids: &[i64]) -> Result<i64, DbError> {
    conn.prepare_cached("INSERT INTO collections (name, created_at) VALUES (?1, ?2)")?
        .execute((name.trim(), now_ms()))?;
    let collection_id = conn.last_insert_rowid();

    let mut insert = conn.prepare_cached(
        "INSERT INTO collection_members (collection_id, sample_id, position)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(collection_id, sample_id) DO NOTHING",
    )?;
    for (position, sample_id) in sample_ids.iter().enumerate() {
        insert.execute((collection_id, sample_id, position as i64))?;
    }

    Ok(collection_id)
}

/// Rewrites `position` for every member of `collection_id` to match `sample_ids`'s order.
fn reorder_collection(
    conn: &Connection,
    collection_id: i64,
    sample_ids: &[i64],
) -> Result<(), DbError> {
    let mut update = conn.prepare_cached(
        "UPDATE collection_members SET position = ?3
         WHERE collection_id = ?1 AND sample_id = ?2",
    )?;
    for (position, sample_id) in sample_ids.iter().enumerate() {
        update.execute((collection_id, sample_id, position as i64))?;
    }
    Ok(())
}

/// Deletes a collection. `collection_members` cascades via its foreign key.
fn delete_collection(conn: &Connection, collection_id: i64) -> Result<(), DbError> {
    conn.prepare_cached("DELETE FROM collections WHERE id = ?1")?
        .execute([collection_id])?;
    Ok(())
}

/// Upserts one `app_settings` row.
fn set_setting(conn: &Connection, key: &str, value: &str) -> Result<(), DbError> {
    conn.prepare_cached(
        "INSERT INTO app_settings (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    )?
    .execute((key, value))?;
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

    /// The index has to cover the folders, because the folders are how a sample library is
    /// organized -- `Kicks/` is where the kicks are, and V1 could not see it.
    #[test]
    fn search_finds_a_sample_by_the_folder_it_is_filed_under() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![
                sample(root, "UK UNDERGROUND KIT/Kicks/Apex.wav"),
                sample(root, "UK UNDERGROUND KIT/Snares/Crown.wav"),
            ])
            .unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        assert_eq!(queries::search_samples(&conn, "kicks", 10).unwrap(), [ids[0]]);
        assert_eq!(
            queries::search_samples(&conn, "snares", 10).unwrap(),
            [ids[1]]
        );
        let mut both = queries::search_samples(&conn, "underground", 10).unwrap();
        both.sort_unstable();
        assert_eq!(both, ids, "a kit name should reach everything filed under it");
    }

    /// The corruption V7 exists to end.
    ///
    /// `samples_fts` is a virtual table, so the `ON DELETE CASCADE` from `library_roots` never
    /// reached it and nothing else deleted from it either. `samples.id` is a plain
    /// `INTEGER PRIMARY KEY`, so the next scan reissued the ids the removed library had been
    /// using and the index started answering with the *old* library's filenames -- on the
    /// development database, `kick` returned a file called `Snare - Razor.wav`. This is that
    /// exact sequence: fill a root, remove it, scan a different one into the reissued ids.
    #[test]
    fn removing_a_root_takes_its_rows_out_of_the_search_index() {
        let (_dir, db) = temp_db();
        let writer = db.writer();

        let first = writer.add_root("/first", None).unwrap();
        writer
            .upsert_samples(vec![sample(first, "kits/KICK_Distorted.wav")])
            .unwrap();
        writer.flush().unwrap();

        writer.remove_root(first).unwrap();
        writer.flush().unwrap();

        {
            let conn = db.read().unwrap();
            assert!(
                queries::search_samples(&conn, "kick", 10).unwrap().is_empty(),
                "a removed root's samples must not still be searchable"
            );
        }

        let second = writer.add_root("/second", None).unwrap();
        let reissued = writer
            .upsert_samples(vec![sample(second, "kits/Snare_Razor.wav")])
            .unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        assert!(
            queries::search_samples(&conn, "kick", 10).unwrap().is_empty(),
            "searching for the removed library's word must not return the new library's file"
        );
        assert_eq!(
            queries::search_samples(&conn, "snare", 10).unwrap(),
            reissued,
            "the new file must be findable by its own name"
        );
    }

    /// Retagging deletes and reinserts the fts row. Under `contentless_delete` that is a real
    /// delete; before V7 it was a re-supply of the old text, and getting it wrong left the
    /// index holding both versions at once.
    #[test]
    fn retagging_leaves_exactly_one_indexed_row() {
        let (_dir, db) = temp_db();
        let writer = db.writer();
        let root = writer.add_root("/samples", None).unwrap();
        let ids = writer
            .upsert_samples(vec![sample(root, "kits/Apex.wav")])
            .unwrap();
        writer.flush().unwrap();

        writer.set_tag(ids[0], "punchy").unwrap();
        writer.set_tag(ids[0], "vinyl").unwrap();
        writer.unset_tag(ids[0], "punchy").unwrap();
        writer.flush().unwrap();

        let conn = db.read().unwrap();
        assert_eq!(
            queries::search_samples(&conn, "vinyl", 10).unwrap(),
            ids,
            "a surviving tag must still be searchable"
        );
        assert!(
            queries::search_samples(&conn, "punchy", 10)
                .unwrap()
                .is_empty(),
            "a removed tag must leave no trace in the index"
        );
        assert_eq!(
            queries::search_samples(&conn, "apex", 10).unwrap(),
            ids,
            "one row, not three: the filename must match exactly once"
        );
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
