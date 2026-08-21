//! Ingest pipeline: stage wiring, bounded channels, and cancellation (`overview.md` §3).
//!
//! Five stages -- walk, decode, mel/features, embed, persist -- connected by bounded
//! channels whose depths are chosen so that backpressure, not memory growth, is what
//! happens when a stage falls behind. A [`CancellationToken`] threads through every stage;
//! cross-cutting rule 6 says every long operation is cancellable and reports progress on
//! the same throttle.
//!
//! **Phase 2 wires three of the five.** Walk feeds a decode-and-analyze stage which feeds
//! the writer; there is no mel or embed stage yet, so the two queues between them
//! (`overview.md` §3's 256 and 64) do not exist and the decode stage hands its 1.92 MB
//! buffer straight back to the pool instead of passing it on. That is why the queue between
//! process and persist can be 1024 deep here: the payload crossing it is a metadata struct
//! of a couple of hundred bytes, not audio. Phase 4 inserts the missing stages between
//! them, and the depths in §3 apply from that point.

pub mod decode;
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

use crate::db::{
    queries, Database, DbError, NewSample, SampleFeatures, SampleStatus, ScanCounts, ScanStatus,
};

pub use decode::{BufferPool, Decoder};
pub use progress::ScanProgress;
pub use walk::{DiscoveredFile, Hasher};

/// Depth of the walk -> process queue (`overview.md` §3).
///
/// A [`DiscoveredFile`] is a path and a handful of integers, so 4096 of them is well under
/// a megabyte. Deep on purpose: discovery is orders of magnitude faster than decode, and a
/// deep queue here means the walker finishes early and stops competing for IO instead of
/// trickling along behind the decoders for the whole scan.
pub const WALK_QUEUE_DEPTH: usize = 4096;

/// Depth of the process -> persist queue (`overview.md` §3).
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
/// Note what is *not* here: a file that will not decode. That is a row with
/// `status = 'decode_failed'`, not an error -- a scan that aborts on the first corrupt
/// file is useless on a real library (`overview.md` §3.2).
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
}

/// A file that made it through decode and analysis, on its way to the writer.
#[derive(Debug)]
struct Analyzed {
    row: NewSample,
    /// `None` for a quarantined file.
    features: Option<SampleFeatures>,
    /// The message stored in `samples.error`, for a quarantined file.
    error: Option<String>,
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
/// anything itself, so the wait graph has no cycle.
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
    twins: TwinTable,
}

/// Per-worker scratch, built once per `rayon` thread rather than once per file.
struct Worker {
    hasher: Hasher,
    decoder: Decoder,
    analyzer: features::Analyzer,
}

/// Scans every enabled library root.
pub fn scan_all_roots(
    db: &Database,
    cancel: &CancellationToken,
) -> Result<Vec<ScanReport>, PipelineError> {
    let conn = db.read()?;
    let roots = queries::library_roots(&conn)?;
    drop(conn);
    let mut reports = Vec::new();

    for root in roots.into_iter().filter(|r| r.enabled) {
        if cancel.is_cancelled() {
            break;
        }
        reports.push(scan_root(db, root.id, cancel)?);
    }

    Ok(reports)
}

/// Scans one library root end to end, opening and closing its `scan_runs` row.
///
/// Returns `Ok` with a `cancelled` or `failed` status rather than `Err` for anything that
/// happened *during* the scan; `Err` is reserved for never having got started.
pub fn scan_root(
    db: &Database,
    root_id: i64,
    cancel: &CancellationToken,
) -> Result<ScanReport, PipelineError> {
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
        "scan starting"
    );

    let scan_id = db.writer().start_scan(Some(root_id))?;
    let progress = ScanProgress::new();

    let outcome = run_stages(db, root_id, &path, &known, cancel, &progress);

    let counts = progress.counts();
    let status = match (&outcome, cancel.is_cancelled()) {
        (Err(_), _) => ScanStatus::Failed,
        (Ok(()), true) => ScanStatus::Cancelled,
        (Ok(()), false) => ScanStatus::Completed,
    };
    let error = outcome.as_ref().err().map(|e| e.to_string());

    // The scan_runs row closes even when the scan blew up -- a `running` row left behind
    // forever is how a UI ends up showing a progress bar with nothing behind it.
    db.writer().finish_scan(scan_id, status, counts, error)?;

    tracing::info!(
        root_id,
        scan_id,
        ?status,
        seen = counts.files_seen,
        added = counts.files_added,
        skipped = counts.files_skipped,
        failed = counts.files_failed,
        deduped = ScanProgress::read(&progress.deduped),
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
    })
}

/// Wires the stages together and runs them to completion.
///
/// The topology, and where each stage's threads come from:
///
/// - **walk** -- one scoped thread, which hands the tree to `ignore`'s own parallel walker.
/// - **process** -- the calling thread, fanned out across the global `rayon` pool by
///   `par_bridge`. Decode and DSP are CPU-bound, which is what that pool is for.
/// - **persist** -- one scoped thread, batching into the single writer.
///
/// Channel senders are what terminate the pipeline: the walk thread owns the only
/// `DiscoveredFile` sender, so finishing the walk ends the process stage, which drops the
/// only `Analyzed` sender, which ends the persist stage. There is no separate shutdown
/// protocol to get wrong.
fn run_stages(
    db: &Database,
    root_id: i64,
    root: &std::path::Path,
    known: &HashMap<String, queries::SampleStamp>,
    cancel: &CancellationToken,
    progress: &ScanProgress,
) -> Result<(), PipelineError> {
    let (found_tx, found_rx) = bounded::<DiscoveredFile>(WALK_QUEUE_DEPTH);
    let (done_tx, done_rx) = bounded::<Analyzed>(PERSIST_QUEUE_DEPTH);

    let state = ScanState {
        db,
        cancel,
        progress,
        pool: BufferPool::for_decode(),
        twins: TwinTable::default(),
    };

    std::thread::scope(|scope| -> Result<(), PipelineError> {
        let persist = std::thread::Builder::new()
            .name("audiobank-persist".into())
            .spawn_scoped(scope, || persist_stage(db, done_rx, progress))
            .map_err(|e| DbError::Io {
                context: "spawning the persist stage".into(),
                source: e,
            })?;

        let walker = std::thread::Builder::new()
            .name("audiobank-walk".into())
            .spawn_scoped(scope, move || {
                let ctx = walk::WalkContext {
                    root_id,
                    root,
                    known,
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

        process_stage(&state, found_rx, &done_tx);
        drop(done_tx);

        // A panicking stage is a bug, not a runtime condition, and there is no partial
        // result worth salvaging from one -- so it is logged and the scan is failed rather
        // than resumed. `join` on a panicked scoped thread returns `Err`; it does not
        // re-panic here.
        if walker.join().is_err() {
            tracing::error!("the walk stage panicked");
        }
        match persist.join() {
            Ok(result) => result,
            Err(_) => {
                tracing::error!("the persist stage panicked");
                Ok(())
            }
        }
    })
}

/// Hash, dedup, decode, analyze. Runs on the `rayon` pool, one closure invocation per file.
fn process_stage(state: &ScanState<'_>, files: Receiver<DiscoveredFile>, out: &Sender<Analyzed>) {
    files.into_iter().par_bridge().for_each_init(
        || Worker {
            hasher: Hasher::new(),
            decoder: Decoder::new(Arc::clone(&state.pool)),
            analyzer: features::Analyzer::new(),
        },
        |worker, file| {
            // Draining rather than breaking: `par_bridge` has no early exit, so a cancelled
            // scan empties the queue as fast as it can be read instead of processing it.
            // The walker has already stopped filling it.
            if state.cancel.is_cancelled() {
                return;
            }
            let analyzed = process_one(state, worker, file);
            let _ = out.send(analyzed);
        },
    );
}

/// One file: hash it, look for a twin, and decode only if there isn't one.
fn process_one(state: &ScanState<'_>, worker: &mut Worker, file: DiscoveredFile) -> Analyzed {
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
    ScanProgress::bump(&state.progress.processed);

    let twin = Twin {
        duration_ms: decoded.duration_ms,
        sample_rate: Some(i64::from(decoded.source_sample_rate)),
        channels: Some(i64::from(decoded.channels)),
        features: analysis,
    };

    // Release the 1.92 MB buffer before the row goes into a queue 1024 deep. Phase 4 passes
    // it on to the mel stage instead; today nothing downstream wants it.
    drop(decoded);

    if let Some(claim) = claim {
        claim.publish(twin.clone());
    }

    Analyzed {
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
    }
}

/// A complete row for a file whose content was already analyzed. Not a stub: a duplicate is
/// as much a sample as its twin, it simply cost nothing to describe.
fn copy_of_twin(
    state: &ScanState<'_>,
    file: DiscoveredFile,
    hash: [u8; 32],
    twin: Twin,
) -> Analyzed {
    ScanProgress::bump(&state.progress.deduped);
    Analyzed {
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
fn quarantine(file: DiscoveredFile, hash: Option<[u8; 32]>, error: String) -> Analyzed {
    Analyzed {
        row: NewSample {
            content_hash: hash,
            status: SampleStatus::DecodeFailed,
            ..base_row(&file)
        },
        features: None,
        error: Some(error),
    }
}

/// A twin from a previous scan, found through the `content_hash` index.
fn twin_from_database(state: &ScanState<'_>, hash: &[u8; 32]) -> Option<Twin> {
    let conn = state.db.read().ok()?;
    let (id, meta) = queries::processed_sample_by_hash(&conn, hash).ok()??;
    let features = queries::sample_features(&conn, id).ok().flatten()?;

    Some(Twin {
        duration_ms: meta.duration_ms,
        sample_rate: meta.sample_rate,
        channels: meta.channels,
        features,
    })
}

/// Accumulates analyzed rows and hands them to the single writer in writer-sized batches.
///
/// One thread, because the writer is one thread: fanning this out would only produce more
/// callers queued on the same channel.
fn persist_stage(
    db: &Database,
    rows: Receiver<Analyzed>,
    progress: &ScanProgress,
) -> Result<(), PipelineError> {
    let mut batch: Vec<Analyzed> = Vec::with_capacity(PERSIST_CHUNK);

    for row in rows {
        batch.push(row);
        if batch.len() >= PERSIST_CHUNK {
            flush(db, &mut batch, progress)?;
        }
    }

    flush(db, &mut batch, progress)
}

/// Writes one batch: the sample rows first, then the features and quarantine messages that
/// need the ids the sample rows just returned.
fn flush(
    db: &Database,
    batch: &mut Vec<Analyzed>,
    progress: &ScanProgress,
) -> Result<(), PipelineError> {
    if batch.is_empty() {
        return Ok(());
    }

    let writer = db.writer();
    let rows: Vec<NewSample> = batch.iter().map(|a| a.row.clone()).collect();
    let ids = writer.upsert_samples(rows)?;

    let mut features = Vec::new();
    let mut failures = Vec::new();
    for (id, analyzed) in ids.iter().zip(batch.iter_mut()) {
        if let Some(f) = analyzed.features.take() {
            features.push((*id, f));
        }
        if let Some(e) = analyzed.error.take() {
            failures.push((*id, e));
        }
    }

    writer.set_features(features)?;
    writer.mark_decode_failed(failures)?;

    ScanProgress::bump_by(&progress.persisted, ids.len() as u64);
    batch.clear();
    Ok(())
}
