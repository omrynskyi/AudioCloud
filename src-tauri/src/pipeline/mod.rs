//! Ingest pipeline: stage wiring, bounded channels, and cancellation (`overview.md` §3).
//!
//! Five stages -- walk, decode, mel, embed, persist -- connected by bounded channels whose
//! depths are chosen so that backpressure, not memory growth, is what happens when a stage
//! falls behind. A [`CancellationToken`] threads through every stage; cross-cutting rule 6
//! says every long operation is cancellable and reports progress on the same throttle.
//!
//! **Decode and mel share a stage, and that is what keeps peak RSS down.** `overview.md`
//! §3 draws them as separate stages with a 256-deep queue between them, and then
//! immediately notes that a decoded window is 1.92 MB and that "decode output is actually
//! handed over as mel frames wherever possible". This is that: one `rayon` worker decodes,
//! analyzes and computes the spectrogram for a file, and what crosses the next channel is
//! the 256 KB mel tensor rather than the 1.92 MB window. The 1.92 MB buffer goes straight
//! back to its pool, so the number of them in existence is bounded by the number of cores
//! rather than by a queue depth.
//!
//! What that leaves is three queues:
//!
//! - **walk -> process**, 4096 deep. A path and a few integers.
//! - **process -> embed**, 64 deep ([`embed::EMBED_QUEUE_DEPTH`]). 256 KB each, 16 MB in
//!   flight. This is the queue that is actually full during a scan, because inference is
//!   the slowest stage, and it is therefore the one that bounds memory.
//! - **embed -> persist**, 1024 deep. A metadata struct plus a 512-float vector.
//!
//! A scan with no inference session is a first-class configuration, not a degraded one:
//! the model is a 200 MB download that may not have happened yet, and a library is worth
//! indexing before it arrives. Such a scan leaves rows at `decoded`, and the *next* scan
//! with a session picks them up -- see [`crate::db::SampleStatus::is_complete`], which is
//! what stops the fast-skip from skipping exactly the files that still owe a vector.

pub mod decode;
pub mod dsp_embed;
pub mod embed;
pub mod features;
pub mod mel;
pub mod progress;
pub mod walk;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
};

use crossbeam_channel::{bounded, Receiver, Sender};
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    db::{
        queries, Database, DbError, EmbeddingLoc, NewSample, SampleFeatures, SampleStatus,
        ScanCounts, ScanStatus,
    },
    model::session::ModelSession,
    pipeline::mel::Padding,
};

pub use decode::{BufferPool, Decoder, PooledBuffer};
pub use embed::{BatchConfig, Embed, EmbedError};
pub use progress::{ProgressSnapshot, ScanPhase, ScanProgress, Ticker};
pub use walk::{DiscoveredFile, Hasher};

/// Depth of the walk -> process queue (`overview.md` §3).
///
/// A [`DiscoveredFile`] is a path and a handful of integers, so 4096 of them is well under
/// a megabyte. Deep on purpose: discovery is orders of magnitude faster than decode, and a
/// deep queue here means the walker finishes early and stops competing for IO instead of
/// trickling along behind the decoders for the whole scan.
pub const WALK_QUEUE_DEPTH: usize = 4096;

/// Depth of the embed -> persist queue (`overview.md` §3).
pub const PERSIST_QUEUE_DEPTH: usize = 1024;

/// Rows the persist stage accumulates before handing them to the writer.
///
/// Matched to `db::writer::BATCH_ROWS` so one send fills exactly one of the writer's
/// transactions. Larger would straddle two commits; smaller would leave the writer
/// committing partial batches on its timeout.
const PERSIST_CHUNK: usize = 1000;

/// Cap on the in-scan dedup table (see [`TwinTable`]).
///
/// At the cap the table stops growing and later duplicates simply get decoded, which costs
/// time and never correctness. 200,000 entries is roughly 30 MB and covers four times the
/// corpus size every §7 target is stated against.
const TWIN_TABLE_CAPACITY: usize = 200_000;

/// Everything the ingest pipeline can fail at as a whole.
///
/// Note what is *not* here: a file that will not decode, and a batch that will not run.
/// The first is a row with `status = 'decode_failed'` (`overview.md` §3.2); the second is a
/// batch of rows left at `decoded` for the next scan to finish. A scan that aborts on the
/// first corrupt file -- or on one transient CoreML hiccup 40,000 files in -- is useless on
/// a real library.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error(transparent)]
    Db(#[from] DbError),

    #[error("library root {0} is not in the database")]
    UnknownRoot(i64),

    #[error("library root {path} is not a readable directory")]
    RootUnreadable { path: PathBuf },
}

/// A cooperative stop signal, shared by every stage of a long operation.
///
/// Cancellation is checked, never forced: a cancelled scan finishes the file it is holding,
/// stops taking new ones, and lets the writer commit what it already has. Cross-cutting
/// rule 6 asks for cancellable, not killable -- the difference is whether partial results
/// survive.
///
/// `Relaxed` throughout for the same reason as [`ScanProgress`]: the flag publishes no
/// other memory, and a worker that notices the cancellation one file late has done one
/// file of unnecessary work, not something incorrect.
#[derive(Debug, Default, Clone)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// How one scan should run: what can stop it, what embeds for it, and who watches it.
///
/// A struct rather than four more positional arguments, and constructed through builders so
/// that adding Phase 5's projection trigger or Phase 9's per-root overrides does not break
/// every call site again.
pub struct ScanOptions<'a> {
    cancel: &'a CancellationToken,
    /// What turns a spectrogram into a vector. `dyn` rather than a concrete session
    /// because CLAP is not the only answer: `dsp_embed::DspEmbedder` implements the same
    /// trait at a ten-thousandth of the cost, and `tests/evaluation.rs` exists to find out
    /// which one a drum library is actually better served by.
    embedder: Option<Arc<dyn Embed>>,
    padding: Padding,
    batch: BatchConfig,
    progress: Arc<ScanProgress>,
    /// Where the 100 ms ticker sends its snapshots. `None` means no ticker runs at all --
    /// the benchmarks and most tests do not want a thread waking up ten times a second in
    /// the middle of a measurement.
    #[allow(clippy::type_complexity)]
    sink: Option<Box<dyn FnMut(ProgressSnapshot) + Send + 'static>>,
    /// Called once per root, with the `scan_runs` id, the instant that row is opened.
    ///
    /// The seam Phase 6's `scan_library` needs. `overview.md` §6.1 has that command return
    /// a `scanId` while the scan itself keeps running, and the id is assigned by the insert
    /// in [`scan_root_with`] -- so the command spawns the job, waits on this callback for
    /// the id, registers the cancellation token under it, and answers. Milliseconds, not a
    /// scan's duration.
    #[allow(clippy::type_complexity)]
    on_start: Option<Box<dyn Fn(i64) + Send + Sync + 'static>>,
}

impl std::fmt::Debug for ScanOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanOptions")
            .field("cancelled", &self.cancel.is_cancelled())
            .field("embedding", &self.embedder.is_some())
            .field("padding", &self.padding)
            .field("batch", &self.batch)
            .field("ticking", &self.sink.is_some())
            .field("watched", &self.on_start.is_some())
            .finish()
    }
}

impl<'a> ScanOptions<'a> {
    /// A scan that decodes and analyzes but does not embed.
    pub fn new(cancel: &'a CancellationToken) -> Self {
        Self {
            cancel,
            embedder: None,
            padding: Padding::default(),
            batch: BatchConfig::default(),
            progress: Arc::new(ScanProgress::new()),
            sink: None,
            on_start: None,
        }
    }

    /// Attaches the CLAP session, turning this into a full five-stage scan.
    pub fn with_session(self, session: Arc<ModelSession>) -> Self {
        self.with_embedder(session)
    }

    /// Attaches any embedder. The model is one; `dsp_embed::DspEmbedder` is another.
    pub fn with_embedder(mut self, embedder: Arc<dyn Embed>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Chooses what the mel front-end does with audio shorter than its ten-second window.
    ///
    /// [`Padding::RepeatPad`] is CLAP's own and is what the parity gate is stated against,
    /// so it is the default and must stay so for any scan feeding the model.
    /// [`Padding::ZeroPad`] is for embedders that measure the envelope, where tiling a
    /// 200 ms kick twenty-five times would manufacture a decay the file does not have.
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    /// Overrides the batching parameters. The sweep in `tests/benchmarks.rs` is the only
    /// caller that should need this.
    pub fn with_batch(mut self, batch: BatchConfig) -> Self {
        self.batch = batch;
        self
    }

    /// Starts a 100 ms ticker that hands each coalesced snapshot to `sink`
    /// (`overview.md` §6.5). Phase 6's `Channel<ScanProgress>` is a sink like any other.
    pub fn with_progress<F>(mut self, sink: F) -> Self
    where
        F: FnMut(ProgressSnapshot) + Send + 'static,
    {
        self.sink = Some(Box::new(sink));
        self
    }

    /// Registers a callback that receives the `scan_runs` id as soon as the row is opened,
    /// before any file is touched.
    pub fn on_start<F>(mut self, f: F) -> Self
    where
        F: Fn(i64) + Send + Sync + 'static,
    {
        self.on_start = Some(Box::new(f));
        self
    }

    /// Shares the counter set, so a caller can read totals while the scan runs.
    pub fn progress(&self) -> Arc<ScanProgress> {
        Arc::clone(&self.progress)
    }

    fn embedding_required(&self) -> bool {
        self.embedder.is_some()
    }
}

/// What one scan did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanReport {
    /// The `scan_runs` row this scan wrote.
    pub scan_id: i64,
    pub root_id: i64,
    pub status: ScanStatus,
    pub counts: ScanCounts,
    /// Files whose content hash matched an already-processed sample, so no decoder ran.
    pub deduped: u64,
    /// Files this scan actually decoded.
    pub processed: u64,
    /// Files this scan produced a vector for, whether by inference or by borrowing a
    /// twin's.
    pub embedded: u64,
    /// Spectrograms sent through the model, and how many `run()` calls that took. Zero and
    /// zero for a scan with no session. The ratio is what the batch-size sweep reads.
    pub inference_batches: u64,
    pub inferred: u64,
}

/// What the embed stage decided should end up in `samples.emb_offset`.
///
/// A four-way split because the four cases cost wildly different amounts and only one of
/// them involves the model at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbedPlan {
    /// Nothing to store: a quarantined file, or any file at all when no session is
    /// available.
    None,
    /// A spectrogram was handed to the embed stage; the vector arrives alongside the row.
    Compute,
    /// A previous scan already embedded this exact content. The duplicate stores a second
    /// reference to the same bytes rather than a second copy of them.
    Existing(EmbeddingLoc),
    /// A twin *in this scan* owns the vector. The persist stage resolves it once the twin's
    /// bytes have actually landed, which may be several batches later or -- if the twin's
    /// batch failed -- never.
    CopyOf([u8; 32]),
}

/// A file that made it through decode and analysis, on its way to the writer.
#[derive(Debug)]
struct Analyzed {
    row: NewSample,
    /// `None` for a quarantined file.
    features: Option<SampleFeatures>,
    /// The message stored in `samples.error`, for a quarantined file.
    error: Option<String>,
    embed: EmbedPlan,
}

/// What an earlier file with the same content hash produced.
///
/// Copying this is the dedup path: a matching hash means byte-identical audio, so decoding
/// again would produce exactly these numbers at exactly the cost of decoding again.
#[derive(Debug, Clone)]
struct Twin {
    duration_ms: Option<i64>,
    sample_rate: Option<i64>,
    channels: Option<i64>,
    features: SampleFeatures,
    /// Where the twin's vector already lives, if it has one on disk. `None` for a twin
    /// produced by this scan, whose vector is still somewhere between the batcher and the
    /// writer -- those resolve through [`EmbedPlan::CopyOf`] instead.
    embedding: Option<EmbeddingLoc>,
}

/// In-scan deduplication, by content hash.
///
/// The `content_hash` index already catches duplicates of anything a previous scan
/// processed. What it cannot catch is the twin still sitting in the writer's open
/// transaction, or the twin being decoded on another core right now -- and a sample library
/// is full of the same 909 kick under six names in one folder, so both windows are hit
/// constantly.
///
/// **Claim, then publish.** A worker that is first to a hash reserves it and every other
/// worker that reaches the same hash waits for the result rather than decoding in parallel.
/// A memo that only recorded finished work would be decided by a race: four identical files
/// discovered together all start decoding before any of them finishes, and none of them
/// deduplicates. Which is exactly what happens, every time, on a small scan.
///
/// Waiting cannot deadlock. A claim is always published -- on success, on decode failure,
/// and on a panic, via [`TwinClaim`]'s `Drop` -- and the claiming worker never waits on
/// anything itself, so the wait graph has no cycle. In particular a claim is published
/// **before** inference, not after: making a duplicate wait for its twin's batch to run
/// would put a decode worker to sleep behind the slowest stage in the pipeline.
#[derive(Debug, Default)]
struct TwinTable {
    entries: Mutex<HashMap<[u8; 32], TwinState>>,
    published: Condvar,
}

#[derive(Debug, Clone)]
enum TwinState {
    /// A worker is decoding this content. Wait for it.
    InFlight,
    Ready(Twin),
    /// This content could not be decoded, so there is nothing to copy and no point waiting.
    Failed,
}

/// What [`TwinTable::claim`] decided this worker should do.
enum Claim<'a> {
    /// Another worker already produced the answer.
    Copy(Twin),
    /// Decode, then publish through this claim.
    Decode(TwinClaim<'a>),
    /// Decode, but do not publish -- the table is full, or a previous attempt at this
    /// content already failed.
    Unmemoized,
}

/// A reservation on one content hash. Publishes `Failed` if dropped without an answer, so a
/// worker that panics mid-decode releases everyone waiting on it instead of stranding them.
struct TwinClaim<'a> {
    table: &'a TwinTable,
    hash: [u8; 32],
    resolved: bool,
}

impl TwinClaim<'_> {
    fn publish(mut self, twin: Twin) {
        self.table.publish(self.hash, TwinState::Ready(twin));
        self.resolved = true;
    }
}

impl Drop for TwinClaim<'_> {
    fn drop(&mut self) {
        if !self.resolved {
            self.table.publish(self.hash, TwinState::Failed);
        }
    }
}

impl TwinTable {
    fn claim(&self, hash: &[u8; 32]) -> Claim<'_> {
        let Ok(mut entries) = self.entries.lock() else {
            return Claim::Unmemoized;
        };

        loop {
            match entries.get(hash) {
                Some(TwinState::Ready(twin)) => return Claim::Copy(twin.clone()),
                Some(TwinState::Failed) => return Claim::Unmemoized,
                Some(TwinState::InFlight) => match self.published.wait(entries) {
                    Ok(next) => entries = next,
                    Err(_) => return Claim::Unmemoized,
                },
                None => {
                    if entries.len() >= TWIN_TABLE_CAPACITY {
                        return Claim::Unmemoized;
                    }
                    entries.insert(*hash, TwinState::InFlight);
                    return Claim::Decode(TwinClaim {
                        table: self,
                        hash: *hash,
                        resolved: false,
                    });
                }
            }
        }
    }

    fn publish(&self, hash: [u8; 32], state: TwinState) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.insert(hash, state);
        }
        // One condvar for the whole table, so a publish wakes every waiter rather than just
        // the ones on this hash. With duplicates measured in handfuls, a per-hash condvar
        // would cost more in allocation than the spurious wakeups cost in cycles.
        self.published.notify_all();
    }
}

/// Scan-scoped state shared by every decode worker.
struct ScanState<'a> {
    db: &'a Database,
    cancel: &'a CancellationToken,
    progress: &'a ScanProgress,
    pool: Arc<BufferPool>,
    /// The 256 KB spectrogram buffers. `None` when there is no session, in which case no
    /// mel is computed at all -- the front-end is a third of the per-file cost and there is
    /// nothing downstream that would read it.
    mel_pool: Option<Arc<BufferPool>>,
    padding: Padding,
    twins: TwinTable,
}

impl ScanState<'_> {
    fn embedding_required(&self) -> bool {
        self.mel_pool.is_some()
    }
}

/// Per-worker scratch, built once per `rayon` thread rather than once per file.
struct Worker {
    hasher: Hasher,
    decoder: Decoder,
    analyzer: features::Analyzer,
    front_end: mel::FrontEnd,
}

/// Scans every enabled library root, decoding but not embedding.
pub fn scan_all_roots(
    db: &Database,
    cancel: &CancellationToken,
) -> Result<Vec<ScanReport>, PipelineError> {
    scan_all_roots_with(db, ScanOptions::new(cancel))
}

/// Scans every enabled library root under the given options.
///
/// One [`ScanOptions`] covers every root, which is what makes the session and the counter
/// set shared across them: a library with six roots builds one session and reports one
/// total, rather than six of each.
pub fn scan_all_roots_with(
    db: &Database,
    mut options: ScanOptions<'_>,
) -> Result<Vec<ScanReport>, PipelineError> {
    let conn = db.read()?;
    let roots = queries::library_roots(&conn)?;
    drop(conn);
    let mut reports = Vec::new();

    for root in roots.into_iter().filter(|r| r.enabled) {
        if options.cancel.is_cancelled() {
            break;
        }
        reports.push(scan_root_with(db, root.id, &mut options)?);
    }

    Ok(reports)
}

/// Scans one library root end to end, decoding but not embedding.
pub fn scan_root(
    db: &Database,
    root_id: i64,
    cancel: &CancellationToken,
) -> Result<ScanReport, PipelineError> {
    scan_root_with(db, root_id, &mut ScanOptions::new(cancel))
}

/// Scans one library root end to end, opening and closing its `scan_runs` row.
///
/// Returns `Ok` with a `cancelled` or `failed` status rather than `Err` for anything that
/// happened *during* the scan; `Err` is reserved for never having got started.
pub fn scan_root_with(
    db: &Database,
    root_id: i64,
    options: &mut ScanOptions<'_>,
) -> Result<ScanReport, PipelineError> {
    let cancel = options.cancel;
    let progress = options.progress();
    let conn = db.read()?;
    let root = queries::library_roots(&conn)?
        .into_iter()
        .find(|r| r.id == root_id)
        .ok_or(PipelineError::UnknownRoot(root_id))?;

    let path = PathBuf::from(&root.path);
    if !path.is_dir() {
        return Err(PipelineError::RootUnreadable { path });
    }

    // Every row this root already has, in one query. See `walk::WalkContext::known`.
    let known = queries::sample_stamps_for_root(&conn, root_id)?;
    drop(conn);
    tracing::info!(
        root_id,
        path = %path.display(),
        known = known.len(),
        embedding = options.embedding_required(),
        "scan starting"
    );

    let scan_id = db.writer().start_scan(Some(root_id))?;
    if let Some(on_start) = options.on_start.as_ref() {
        on_start(scan_id);
    }
    // The ticker starts before the stages and is dropped after them, so the terminal
    // snapshot reports the scan's real final counts rather than whatever the last 100 ms
    // boundary happened to catch (`overview.md` §6.5).
    let mut ticker = options
        .sink
        .take()
        .map(|sink| Ticker::spawn(Arc::clone(&progress), scan_id, sink));

    let outcome = run_stages(db, root_id, &path, &known, options, &progress);

    progress.enter(ScanPhase::Finishing);
    let counts = progress.counts();
    let status = match (&outcome, cancel.is_cancelled()) {
        (Err(_), _) => ScanStatus::Failed,
        (Ok(_), true) => ScanStatus::Cancelled,
        (Ok(_), false) => ScanStatus::Completed,
    };
    let error = outcome.as_ref().err().map(|e| e.to_string());

    // The scan_runs row closes even when the scan blew up -- a `running` row left behind
    // forever is how a UI ends up showing a progress bar with nothing behind it.
    db.writer().finish_scan(scan_id, status, counts, error)?;
    if let Some(ticker) = ticker.as_mut() {
        ticker.finish();
    }

    let (inferred, inference_batches) = outcome.as_ref().copied().unwrap_or((0, 0));
    tracing::info!(
        root_id,
        scan_id,
        ?status,
        seen = counts.files_seen,
        added = counts.files_added,
        skipped = counts.files_skipped,
        failed = counts.files_failed,
        deduped = ScanProgress::read(&progress.deduped),
        embedded = ScanProgress::read(&progress.embedded),
        inference_batches,
        "scan finished"
    );

    outcome?;

    Ok(ScanReport {
        scan_id,
        root_id,
        status,
        counts,
        deduped: ScanProgress::read(&progress.deduped),
        processed: ScanProgress::read(&progress.processed),
        embedded: ScanProgress::read(&progress.embedded),
        inferred,
        inference_batches,
    })
}

/// Wires the stages together and runs them to completion.
///
/// The topology, and where each stage's threads come from:
///
/// - **walk** -- one scoped thread, which hands the tree to `ignore`'s own parallel walker.
/// - **process** (hash, decode, DSP, mel) -- the calling thread, fanned out across the
///   global `rayon` pool by `par_bridge`. All of it is CPU-bound, which is what that pool
///   is for.
/// - **embed** -- one scoped thread. One, because there is one session and `run()` on it is
///   serialized; a second thread would only queue on the same mutex.
/// - **persist** -- one scoped thread, batching into the single writer.
///
/// Channel senders are what terminate the pipeline: the walk thread owns the only
/// `DiscoveredFile` sender, so finishing the walk ends the process stage, which drops the
/// only mel sender, which ends the embed stage, which drops the only row sender, which ends
/// the persist stage. There is no separate shutdown protocol to get wrong.
///
/// Returns the spectrogram and batch counts, for the batch-size sweep.
fn run_stages(
    db: &Database,
    root_id: i64,
    root: &std::path::Path,
    known: &HashMap<String, queries::SampleStamp>,
    options: &ScanOptions<'_>,
    progress: &ScanProgress,
) -> Result<(u64, u64), PipelineError> {
    // Everything the stage threads need, lifted out of `options` before the scope. A
    // `ScanOptions` holds a boxed sink and is therefore not `Sync`; the pieces are.
    let cancel = options.cancel;
    let embedder = options.embedder.clone();
    let padding = options.padding;
    let batch = options.batch;
    let embedding_required = options.embedding_required();

    let (found_tx, found_rx) = bounded::<DiscoveredFile>(WALK_QUEUE_DEPTH);
    let (mel_tx, mel_rx) = bounded::<embed::Pending<Analyzed>>(embed::EMBED_QUEUE_DEPTH);
    let (done_tx, done_rx) = bounded::<embed::Embedded<Analyzed>>(PERSIST_QUEUE_DEPTH);

    let state = ScanState {
        db,
        cancel,
        progress,
        pool: BufferPool::for_decode(),
        mel_pool: embedder.as_ref().map(|_| embed::mel_pool(batch)),
        padding,
        twins: TwinTable::default(),
    };

    std::thread::scope(|scope| -> Result<(u64, u64), PipelineError> {
        let persist = std::thread::Builder::new()
            .name("audiobank-persist".into())
            .spawn_scoped(scope, || persist_stage(db, done_rx, progress))
            .map_err(|e| DbError::Io {
                context: "spawning the persist stage".into(),
                source: e,
            })?;

        let embedder = std::thread::Builder::new()
            .name("audiobank-embed".into())
            .spawn_scoped(scope, {
                let embedder = embedder.clone();
                move || {
                    let counts = match embedder {
                        Some(session) => {
                            progress.enter(ScanPhase::Embedding);
                            embed::embed_stage(
                                &session,
                                batch,
                                cancel,
                                mel_rx,
                                &done_tx,
                                &progress.embedded,
                            )
                        }
                        // No session: the rows still have to reach the writer, so the stage
                        // runs as a pass-through rather than being wired out of the
                        // topology. One `if` here beats two shapes of pipeline.
                        None => {
                            for item in mel_rx {
                                if done_tx
                                    .send(embed::Embedded {
                                        payload: item.payload,
                                        embedding: None,
                                    })
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            (0, 0)
                        }
                    };
                    drop(done_tx);
                    counts
                }
            })
            .map_err(|e| DbError::Io {
                context: "spawning the embed stage".into(),
                source: e,
            })?;

        let walker = std::thread::Builder::new()
            .name("audiobank-walk".into())
            .spawn_scoped(scope, move || {
                let ctx = walk::WalkContext {
                    root_id,
                    root,
                    known,
                    embedding_required,
                    cancel,
                    progress,
                };
                walk::walk_root(&ctx, &found_tx);
                // Explicit, because it is load-bearing: this drop is what ends the process
                // stage. Letting it happen implicitly at the end of the closure would work
                // today and break the moment someone captures the sender by reference.
                drop(found_tx);
            })
            .map_err(|e| DbError::Io {
                context: "spawning the walk stage".into(),
                source: e,
            })?;

        process_stage(&state, found_rx, &mel_tx);
        drop(mel_tx);

        // A panicking stage is a bug, not a runtime condition, and there is no partial
        // result worth salvaging from one -- so it is logged and the scan is failed rather
        // than resumed. `join` on a panicked scoped thread returns `Err`; it does not
        // re-panic here.
        if walker.join().is_err() {
            tracing::error!("the walk stage panicked");
        }
        let inference = match embedder.join() {
            Ok(counts) => counts,
            Err(_) => {
                tracing::error!("the embed stage panicked");
                (0, 0)
            }
        };
        match persist.join() {
            Ok(result) => result.map(|()| inference),
            Err(_) => {
                tracing::error!("the persist stage panicked");
                Ok(inference)
            }
        }
    })
}

/// Hash, dedup, decode, analyze, and compute the spectrogram. Runs on the `rayon` pool, one
/// closure invocation per file.
fn process_stage(
    state: &ScanState<'_>,
    files: Receiver<DiscoveredFile>,
    out: &Sender<embed::Pending<Analyzed>>,
) {
    files.into_iter().par_bridge().for_each_init(
        || Worker {
            hasher: Hasher::new(),
            decoder: Decoder::new(Arc::clone(&state.pool)),
            analyzer: features::Analyzer::new(),
            front_end: mel::FrontEnd::with_padding(state.padding),
        },
        |worker, file| {
            // Draining rather than breaking: `par_bridge` has no early exit, so a cancelled
            // scan empties the queue as fast as it can be read instead of processing it.
            // The walker has already stopped filling it.
            if state.cancel.is_cancelled() {
                return;
            }
            state.progress.enter(ScanPhase::Decoding);
            state.progress.note_path(&file.rel_path);
            let analyzed = process_one(state, worker, file);
            let _ = out.send(analyzed);
        },
    );
}

/// One file: hash it, look for a twin, and decode only if there isn't one.
fn process_one(
    state: &ScanState<'_>,
    worker: &mut Worker,
    file: DiscoveredFile,
) -> embed::Pending<Analyzed> {
    let hash = match worker.hasher.hash_file(&file.path, file.size_bytes as u64) {
        Ok(hash) => hash,
        Err(e) => {
            ScanProgress::bump(&state.progress.failed);
            return quarantine(file, None, format!("could not read the file: {e}"));
        }
    };

    // The in-scan table first: it is the only one that can answer for a twin that is still
    // in flight or still in the writer's open transaction.
    let claim = match state.twins.claim(&hash) {
        Claim::Copy(twin) => return copy_of_twin(state, file, hash, twin),
        Claim::Decode(claim) => Some(claim),
        Claim::Unmemoized => None,
    };

    // ...then the `content_hash` index, for anything a previous scan already processed.
    if let Some(twin) = twin_from_database(state, &hash) {
        if let Some(claim) = claim {
            claim.publish(twin.clone());
        }
        return copy_of_twin(state, file, hash, twin);
    }

    let decoded = match worker.decoder.decode(&file.path, &file.ext) {
        Ok(decoded) => decoded,
        Err(e) => {
            tracing::debug!(path = %file.path.display(), error = %e, "quarantined");
            ScanProgress::bump(&state.progress.failed);
            // `claim` drops here, publishing `Failed`: the twins waiting on this content
            // must not inherit an analysis that does not exist. They will each fail on their
            // own file, which is what puts the right path in the right `error` column.
            return quarantine(file, Some(hash), e.to_string());
        }
    };

    let analysis = worker.analyzer.analyze(&decoded.samples);

    // The spectrogram, computed here rather than in a stage of its own so that the 1.92 MB
    // window can be released now instead of crossing another channel. It also reuses this
    // worker's FFT plan, which `analyze` has already paid for.
    let mel = state.mel_pool.as_ref().map(|pool| {
        let mut buf = pool.take();
        worker
            .front_end
            .compute(&decoded.samples, &mut worker.analyzer, buf.buffer_mut());
        buf
    });

    ScanProgress::bump(&state.progress.processed);

    let twin = Twin {
        duration_ms: decoded.duration_ms,
        sample_rate: Some(i64::from(decoded.source_sample_rate)),
        channels: Some(i64::from(decoded.channels)),
        features: analysis,
        // This scan's own vector does not exist yet; duplicates of it resolve through
        // `EmbedPlan::CopyOf` in the persist stage.
        embedding: None,
    };

    // Release the 1.92 MB buffer before the row goes into a queue: what crosses it is the
    // 256 KB spectrogram, which is the whole reason decode and mel share a stage.
    drop(decoded);

    if let Some(claim) = claim {
        claim.publish(twin.clone());
    }

    let embed = if mel.is_some() {
        EmbedPlan::Compute
    } else {
        EmbedPlan::None
    };

    embed::Pending {
        payload: Analyzed {
            row: NewSample {
                content_hash: Some(hash),
                duration_ms: twin.duration_ms,
                sample_rate: twin.sample_rate,
                channels: twin.channels,
                status: SampleStatus::Decoded,
                ..base_row(&file)
            },
            features: Some(twin.features),
            error: None,
            embed,
        },
        mel,
    }
}

/// A complete row for a file whose content was already analyzed. Not a stub: a duplicate is
/// as much a sample as its twin, it simply cost nothing to describe.
///
/// Costs nothing to *embed*, either, which is the part that matters at 50,000 files: the
/// row points at its twin's bytes rather than running the model on identical audio to
/// produce an identical vector.
fn copy_of_twin(
    state: &ScanState<'_>,
    file: DiscoveredFile,
    hash: [u8; 32],
    twin: Twin,
) -> embed::Pending<Analyzed> {
    ScanProgress::bump(&state.progress.deduped);

    let embed = match twin.embedding {
        Some(loc) => EmbedPlan::Existing(loc),
        None if state.embedding_required() => EmbedPlan::CopyOf(hash),
        None => EmbedPlan::None,
    };

    embed::Pending {
        payload: Analyzed {
            row: NewSample {
                content_hash: Some(hash),
                duration_ms: twin.duration_ms,
                sample_rate: twin.sample_rate,
                channels: twin.channels,
                status: SampleStatus::Decoded,
                ..base_row(&file)
            },
            features: Some(twin.features),
            error: None,
            embed,
        },
        mel: None,
    }
}

/// The columns that come from the directory entry alone.
fn base_row(file: &DiscoveredFile) -> NewSample {
    NewSample {
        root_id: file.root_id,
        rel_path: file.rel_path.clone(),
        filename: file.filename.clone(),
        ext: file.ext.clone(),
        size_bytes: file.size_bytes,
        mtime: file.mtime,
        content_hash: None,
        duration_ms: None,
        sample_rate: None,
        channels: None,
        status: SampleStatus::Pending,
    }
}

/// A row that records why the file could not be read, so the UI can list it
/// (`overview.md` §3.2).
fn quarantine(
    file: DiscoveredFile,
    hash: Option<[u8; 32]>,
    error: String,
) -> embed::Pending<Analyzed> {
    embed::Pending {
        payload: Analyzed {
            row: NewSample {
                content_hash: hash,
                status: SampleStatus::DecodeFailed,
                ..base_row(&file)
            },
            features: None,
            error: Some(error),
            embed: EmbedPlan::None,
        },
        mel: None,
    }
}

/// A twin from a previous scan, found through the `content_hash` index.
///
/// When this scan is embedding, only an already-*embedded* twin will do. A twin that is
/// merely `decoded` saves one decode and leaves both rows owing a vector, so this scan
/// would end with the corpus half-embedded and the user would have to run it again to
/// converge. Falling through to the decoder instead costs one decode and finishes the job:
/// the file gets a real vector, and the in-scan [`TwinTable`] gives its duplicates a
/// [`EmbedPlan::CopyOf`] reference to it.
fn twin_from_database(state: &ScanState<'_>, hash: &[u8; 32]) -> Option<Twin> {
    let conn = state.db.read().ok()?;

    let (id, embedding) = if state.embedding_required() {
        let (id, loc) = queries::embedded_sample_by_hash(&conn, hash).ok()??;
        (id, Some(loc))
    } else {
        let (id, _) = queries::processed_sample_by_hash(&conn, hash).ok()??;
        (id, None)
    };

    // Metadata is read off the row the vector came from rather than carried along from the
    // lookup, so both branches describe the same sample.
    let (_, meta) = queries::processed_sample_by_hash(&conn, hash).ok()??;
    let features = queries::sample_features(&conn, id).ok().flatten()?;

    Some(Twin {
        duration_ms: meta.duration_ms,
        sample_rate: meta.sample_rate,
        channels: meta.channels,
        features,
        embedding,
    })
}

/// Vectors this scan has already written, by content hash.
///
/// Bounded like [`TwinTable`]: past the cap a duplicate simply stores its own copy of the
/// vector, which costs a kilobyte of `embeddings.bin` and never correctness.
#[derive(Debug, Default)]
struct EmbeddingLedger {
    by_hash: HashMap<[u8; 32], EmbeddingLoc>,
    /// Rows whose twin had not landed yet when they were written. Resolved once, at the end
    /// of the scan.
    deferred: Vec<(i64, [u8; 32])>,
}

/// Accumulates analyzed rows and hands them to the single writer in writer-sized batches.
///
/// One thread, because the writer is one thread: fanning this out would only produce more
/// callers queued on the same channel.
fn persist_stage(
    db: &Database,
    rows: Receiver<embed::Embedded<Analyzed>>,
    progress: &ScanProgress,
) -> Result<(), PipelineError> {
    let mut batch: Vec<embed::Embedded<Analyzed>> = Vec::with_capacity(PERSIST_CHUNK);
    let mut ledger = EmbeddingLedger::default();

    for row in rows {
        batch.push(row);
        if batch.len() >= PERSIST_CHUNK {
            flush(db, &mut batch, &mut ledger, progress)?;
        }
    }

    flush(db, &mut batch, &mut ledger, progress)?;
    resolve_deferred(db, &mut ledger, progress)?;

    // One `fsync` per scan rather than one per batch. The file is append-only and the
    // database is the index into it, so the ordering that matters is bytes-before-offsets,
    // and that is what this call establishes before `finish_scan` closes the run.
    if let Ok(store) = db.embeddings().lock() {
        store.sync()?;
    }
    Ok(())
}

/// Writes one batch: the sample rows first, then the features, quarantine messages and
/// vectors that need the ids the sample rows just returned.
fn flush(
    db: &Database,
    batch: &mut Vec<embed::Embedded<Analyzed>>,
    ledger: &mut EmbeddingLedger,
    progress: &ScanProgress,
) -> Result<(), PipelineError> {
    if batch.is_empty() {
        return Ok(());
    }

    let writer = db.writer();
    let rows: Vec<NewSample> = batch.iter().map(|a| a.payload.row.clone()).collect();
    let ids = writer.upsert_samples(rows)?;

    let mut features = Vec::new();
    let mut failures = Vec::new();
    // Vectors to append, with the row and content hash each one belongs to.
    let mut fresh: Vec<(i64, Option<[u8; 32]>, Vec<f32>)> = Vec::new();
    // Rows that point at bytes already in the file.
    let mut locs: Vec<(i64, EmbeddingLoc)> = Vec::new();
    // Rows that got a vector without one being computed for them. The embed stage counts
    // the ones it ran; these would otherwise be invisible, and on a library that is 40%
    // duplicates that is most of the progress bar.
    let mut borrowed = 0u64;

    for (id, item) in ids.iter().zip(batch.iter_mut()) {
        let analyzed = &mut item.payload;
        if let Some(f) = analyzed.features.take() {
            features.push((*id, f));
        }
        if let Some(e) = analyzed.error.take() {
            failures.push((*id, e));
        }

        match analyzed.embed {
            EmbedPlan::None => {}
            EmbedPlan::Compute => {
                // `None` here is an inference failure, already logged by the embed stage.
                // The row stays `decoded` and the next scan finishes it.
                if let Some(vector) = item.embedding.take() {
                    fresh.push((*id, analyzed.row.content_hash, vector));
                }
            }
            EmbedPlan::Existing(loc) => {
                locs.push((*id, loc));
                borrowed += 1;
            }
            EmbedPlan::CopyOf(hash) => match ledger.by_hash.get(&hash) {
                Some(loc) => {
                    locs.push((*id, *loc));
                    borrowed += 1;
                }
                // The twin is in a batch that has not been persisted yet -- the embed stage
                // preserves order, but the *decode* stage does not, so a duplicate can
                // legitimately overtake the file it copies from.
                None => ledger.deferred.push((*id, hash)),
            },
        }
    }

    if !fresh.is_empty() {
        let mut store = db
            .embeddings()
            .lock()
            .map_err(|_| DbError::Poisoned("embedding store"))?;
        let vectors: Vec<&[f32]> = fresh.iter().map(|(_, _, v)| v.as_slice()).collect();
        let appended = store.append_batch(&vectors)?;
        for ((id, hash, _), loc) in fresh.iter().zip(appended) {
            locs.push((*id, loc));
            if let Some(hash) = hash {
                if ledger.by_hash.len() < TWIN_TABLE_CAPACITY {
                    ledger.by_hash.insert(*hash, loc);
                }
            }
        }
    }

    writer.set_features(features)?;
    writer.mark_decode_failed(failures)?;
    // Last, and separately: this is the statement that flips a row to `embedded`, and it
    // must not run before the bytes it points at are in the file.
    writer.set_embeddings(locs)?;

    ScanProgress::bump_by(&progress.persisted, ids.len() as u64);
    ScanProgress::bump_by(&progress.embedded, borrowed);
    batch.clear();
    Ok(())
}

/// Points the duplicates that overtook their twins at the right bytes.
///
/// Runs once, after every row of the scan has been written. Anything still unresolved --
/// because its twin's batch failed inference, or because the scan was cancelled before the
/// twin got there -- is left `decoded` rather than guessed at, and the next scan embeds it
/// on its own terms.
fn resolve_deferred(
    db: &Database,
    ledger: &mut EmbeddingLedger,
    progress: &ScanProgress,
) -> Result<(), PipelineError> {
    if ledger.deferred.is_empty() {
        return Ok(());
    }

    let conn = db.read()?;
    let mut locs = Vec::with_capacity(ledger.deferred.len());
    let mut unresolved = 0u64;

    for (id, hash) in ledger.deferred.drain(..) {
        match ledger.by_hash.get(&hash) {
            Some(loc) => locs.push((id, *loc)),
            None => match queries::embedded_sample_by_hash(&conn, &hash)? {
                Some((_, loc)) => locs.push((id, loc)),
                None => unresolved += 1,
            },
        }
    }
    drop(conn);

    if unresolved > 0 {
        tracing::warn!(
            unresolved,
            "duplicates whose twin never produced a vector; left for the next scan"
        );
    }

    ScanProgress::bump_by(&progress.embedded, locs.len() as u64);
    db.writer().set_embeddings(locs)?;
    Ok(())
}
