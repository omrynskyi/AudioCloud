//! Progress reporting: atomic counters per stage, drained by a single 100 ms ticker.
//!
//! Cross-cutting rule 3: no per-file IPC events. 50,000 files emitting individually would
//! saturate the WebView message port and stall the render loop; the counters are cheap and
//! the ticker coalesces them to <= 10 Hz (`overview.md` §6.5).
//!
//! The split is deliberate. [`ScanProgress`] is written by every stage and knows nothing
//! about who is watching; [`Ticker`] is one thread that samples it and hands a
//! [`ProgressSnapshot`] to a sink. Phase 6 makes that sink a `tauri::ipc::Channel` and
//! changes nothing on the producer side, which is the point -- the throttle is a property
//! of the pipeline, not a favour the IPC layer does for it.

use serde::Serialize;
use ts_rs::TS;

use std::{
    sync::{
        atomic::{AtomicU64, AtomicU8, Ordering},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::db::ScanCounts;

/// How often the ticker samples the counters (`overview.md` §6.5's <= 10 Hz).
pub const TICK: Duration = Duration::from_millis(100);

/// Which stage a scan is mostly in, for a label the UI can show.
///
/// "Mostly", because the stages overlap by design -- the walker is still finding files
/// while the embedder is running batches. The phase is the furthest stage that has started,
/// which is what a progress label is actually asking about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "ScanPhase.ts")]
#[repr(u8)]
pub enum ScanPhase {
    Walking = 0,
    Decoding = 1,
    Embedding = 2,
    Finishing = 3,
}

impl ScanPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            ScanPhase::Walking => "walking",
            ScanPhase::Decoding => "decoding",
            ScanPhase::Embedding => "embedding",
            ScanPhase::Finishing => "finishing",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => ScanPhase::Decoding,
            2 => ScanPhase::Embedding,
            3 => ScanPhase::Finishing,
            _ => ScanPhase::Walking,
        }
    }
}

/// Per-stage counters for one scan.
///
/// Every field is written by many threads and read by one, so `Relaxed` is the right
/// ordering throughout: the counters are a progress display, not a synchronization
/// mechanism, and no other memory is published through them. The final read happens after
/// every producing thread has been joined, and that join is the barrier which makes the
/// totals exact rather than eventually-consistent.
#[derive(Debug, Default)]
pub struct ScanProgress {
    /// Directory entries the walker considered, including the ones it filtered out.
    pub seen: AtomicU64,
    /// Files that passed the extension and size filters and were not fast-skipped.
    pub queued: AtomicU64,
    /// Files whose `(mtime, size)` matched an existing row, so nothing was read.
    pub skipped: AtomicU64,
    /// Files whose content hash matched an already-processed sample.
    pub deduped: AtomicU64,
    /// Files this scan decoded and analyzed.
    pub processed: AtomicU64,
    /// Files this scan gave a vector to, whether the model ran for them or they borrowed a
    /// byte-identical twin's. Lags [`Self::processed`] while batches are in flight, and can
    /// exceed it on a library full of duplicates -- a copy is embedded without being
    /// decoded.
    pub embedded: AtomicU64,
    /// Files quarantined: unreadable, undecodable, or not audio after all.
    pub failed: AtomicU64,
    /// Rows handed to the writer and committed.
    pub persisted: AtomicU64,

    /// The furthest stage that has started. Written with `fetch_max`, so a late walker
    /// cannot drag the label back to `Walking` after inference has begun.
    phase: AtomicU8,
    /// A recently-processed path, for texture. Sampled, never exhaustive
    /// (`overview.md` §6.5) -- writers use `try_lock` and skip on contention, so this
    /// costs a stage nothing and is allowed to miss.
    current: Mutex<Option<String>>,
    /// When the scan started, for the ETA.
    started: Mutex<Option<Instant>>,
}

impl ScanProgress {
    pub fn new() -> Self {
        let progress = Self::default();
        if let Ok(mut started) = progress.started.lock() {
            *started = Some(Instant::now());
        }
        progress
    }

    /// Adds one to a counter. A free function rather than a method per field so the
    /// ordering decision lives in exactly one place.
    #[inline]
    pub fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn bump_by(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    #[inline]
    pub fn read(counter: &AtomicU64) -> u64 {
        counter.load(Ordering::Relaxed)
    }

    /// Advances the reported phase, never rewinding it.
    pub fn enter(&self, phase: ScanPhase) {
        self.phase.fetch_max(phase as u8, Ordering::Relaxed);
    }

    pub fn phase(&self) -> ScanPhase {
        ScanPhase::from_u8(self.phase.load(Ordering::Relaxed))
    }

    /// Offers a path for the "currently processing" label, dropping it if the lock is
    /// busy. A progress label is not worth blocking a decode worker for.
    pub fn note_path(&self, path: &str) {
        if let Ok(mut current) = self.current.try_lock() {
            *current = Some(path.to_string());
        }
    }

    /// The subset of these counters that a `scan_runs` row stores.
    ///
    /// `files_added` counts every row this scan wrote, deduplicated files included -- they
    /// are new rows even though no decoder ran for them.
    pub fn counts(&self) -> ScanCounts {
        ScanCounts {
            files_seen: Self::read(&self.seen) as i64,
            files_added: Self::read(&self.persisted) as i64,
            files_skipped: Self::read(&self.skipped) as i64,
            files_failed: Self::read(&self.failed) as i64,
        }
    }

    /// One coalesced reading of every counter (`overview.md` §6.5).
    pub fn snapshot(&self, scan_id: i64) -> ProgressSnapshot {
        let queued = Self::read(&self.queued);
        let done = Self::read(&self.persisted);
        let elapsed = self
            .started
            .lock()
            .ok()
            .and_then(|s| *s)
            .map(|t| t.elapsed())
            .unwrap_or_default();

        ProgressSnapshot {
            scan_id,
            phase: self.phase(),
            files_seen: Self::read(&self.seen),
            files_queued: queued,
            files_done: done,
            files_skipped: Self::read(&self.skipped),
            files_failed: Self::read(&self.failed),
            files_embedded: Self::read(&self.embedded),
            current_path: self.current.lock().ok().and_then(|c| c.clone()),
            eta_seconds: eta_seconds(queued, done, elapsed),
        }
    }
}

/// What one tick reports.
///
/// The shape is `overview.md` §6.5's, which is why Phase 6 could add derives here rather
/// than a translation layer: this *is* the wire type, carried by `Channel<ScanEvent>` as
/// the `progress` variant and exported to TypeScript as `ScanProgress`. It is named
/// `ProgressSnapshot` in Rust only because [`ScanProgress`] -- the counter set -- got the
/// obvious name first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(rename = "ScanProgress", export_to = "ScanProgress.ts")]
pub struct ProgressSnapshot {
    pub scan_id: i64,
    pub phase: ScanPhase,
    pub files_seen: u64,
    pub files_queued: u64,
    pub files_done: u64,
    pub files_skipped: u64,
    pub files_failed: u64,
    pub files_embedded: u64,
    /// Sampled, not exhaustive -- for texture, not accounting.
    pub current_path: Option<String>,
    pub eta_seconds: Option<u64>,
}

impl ProgressSnapshot {
    /// Whether this is the last snapshot of the scan.
    pub fn is_terminal(&self) -> bool {
        self.phase == ScanPhase::Finishing
    }
}

/// Remaining seconds at the rate achieved so far, or `None` when there is nothing to
/// extrapolate from.
///
/// Deliberately naive. An ETA computed from a windowed rate looks better and is wrong in a
/// more confusing way when the corpus is mixed -- a folder of 200 ms one-shots followed by
/// a folder of 10 s loops -- and this number's job is to keep a progress bar from lying
/// about being nearly done, not to be accurate to the second.
fn eta_seconds(queued: u64, done: u64, elapsed: Duration) -> Option<u64> {
    if done == 0 || queued <= done {
        return None;
    }
    let per_file = elapsed.as_secs_f64() / done as f64;
    let remaining = (queued - done) as f64 * per_file;
    remaining.is_finite().then(|| remaining.ceil() as u64)
}

/// The single ticker: one thread, one sink, one message every [`TICK`].
///
/// Guarantees a terminal snapshot on drop regardless of the tick schedule, so a UI can
/// never be left sitting at 99% (`overview.md` §6.5). The scan sets
/// [`ScanPhase::Finishing`] before the ticker is dropped, which is what makes that last
/// snapshot identifiable as terminal.
#[derive(Debug)]
pub struct Ticker {
    stop: Arc<std::sync::atomic::AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Ticker {
    /// Starts a ticker over `progress`, calling `sink` at most once per [`TICK`].
    ///
    /// The sink runs on the ticker's own thread and must not block for long: it is between
    /// the counters and the UI, and a slow sink turns a 10 Hz throttle into whatever the
    /// sink's latency is.
    pub fn spawn<F>(progress: Arc<ScanProgress>, scan_id: i64, sink: F) -> Self
    where
        F: FnMut(ProgressSnapshot) + Send + 'static,
    {
        Self::watch(progress, move |p| p.snapshot(scan_id), sink)
    }

    /// The general form: sample any shared state through `snapshot`, hand the result to
    /// `sink`, at most once per [`TICK`], skipping repeats and guaranteeing a last one.
    ///
    /// Generic because the throttle is a property of the *pipeline*, not of scanning:
    /// Phase 5's projection re-fit is a long operation that reports progress under the same
    /// rule (cross-cutting rule 6), and it deserves the coalescing and the terminal-event
    /// guarantee rather than a second thread that reimplements them slightly differently.
    ///
    /// The `snapshot` closure and the sink both run on the ticker's own thread. Neither may
    /// block for long: they sit between the counters and the UI, and a slow one turns a
    /// 10 Hz throttle into whatever its latency is.
    pub fn watch<T, S, N, F>(state: Arc<T>, snapshot: N, mut sink: F) -> Self
    where
        T: Send + Sync + 'static,
        S: Clone + PartialEq + Send + 'static,
        N: Fn(&T) -> S + Send + 'static,
        F: FnMut(S) + Send + 'static,
    {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&stop);

        let handle = std::thread::Builder::new()
            .name("audiobank-progress".into())
            .spawn(move || {
                let mut last: Option<S> = None;
                loop {
                    let stopping = flag.load(Ordering::Relaxed);
                    let current = snapshot(&state);

                    // Identical snapshots are dropped: an idle stage should cost the
                    // WebView nothing at all, and a progress bar that re-renders 10 times a
                    // second with the same numbers is the anti-pattern in a smaller hat.
                    if stopping || last.as_ref() != Some(&current) {
                        sink(current.clone());
                        last = Some(current);
                    }
                    if stopping {
                        break;
                    }
                    std::thread::sleep(TICK);
                }
            })
            .ok();

        Self { stop, handle }
    }

    /// Emits one final snapshot and stops. Idempotent; [`Drop`] calls it.
    pub fn finish(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn the_phase_never_rewinds() {
        let progress = ScanProgress::new();
        assert_eq!(progress.phase(), ScanPhase::Walking);

        progress.enter(ScanPhase::Embedding);
        progress.enter(ScanPhase::Decoding);
        assert_eq!(
            progress.phase(),
            ScanPhase::Embedding,
            "a still-running walker dragged the label backwards"
        );
    }

    #[test]
    fn an_eta_needs_something_to_extrapolate_from() {
        assert_eq!(eta_seconds(100, 0, Duration::from_secs(1)), None);
        assert_eq!(eta_seconds(100, 100, Duration::from_secs(1)), None);
        assert_eq!(eta_seconds(0, 0, Duration::from_secs(1)), None);
        // Half done in 10 s means about 10 s left.
        assert_eq!(eta_seconds(100, 50, Duration::from_secs(10)), Some(10));
    }

    /// The guarantee that keeps a progress bar off 99%: a terminal snapshot regardless of
    /// where the tick boundary happened to fall.
    #[test]
    fn a_terminal_snapshot_always_arrives() {
        let progress = Arc::new(ScanProgress::new());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);

        let mut ticker = Ticker::spawn(Arc::clone(&progress), 7, move |s| {
            sink.lock().unwrap().push(s);
        });

        ScanProgress::bump_by(&progress.persisted, 5);
        progress.enter(ScanPhase::Finishing);
        ticker.finish();

        let snapshots = seen.lock().unwrap();
        let last = snapshots.last().expect("no snapshot at all");
        assert!(last.is_terminal());
        assert_eq!(last.files_done, 5);
        assert_eq!(last.scan_id, 7);
    }

    /// <= 10 Hz is a ceiling on the wire, and an unchanged scan should be well under it.
    #[test]
    fn an_idle_scan_emits_once_rather_than_ten_times_a_second() {
        let progress = Arc::new(ScanProgress::new());
        let count = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let sink = Arc::clone(&count);

        let mut ticker = Ticker::spawn(Arc::clone(&progress), 1, move |_| {
            sink.fetch_add(1, Ordering::Relaxed);
        });
        std::thread::sleep(TICK * 6);
        ticker.finish();

        // One for the first sample, one terminal. Anything more is a repeat of numbers
        // that did not change.
        assert!(
            count.load(Ordering::Relaxed) <= 2,
            "idle ticker emitted {} snapshots",
            count.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn a_sampled_path_is_allowed_to_be_missing_but_not_wrong() {
        let progress = ScanProgress::new();
        assert!(progress.snapshot(1).current_path.is_none());
        progress.note_path("kicks/909.wav");
        assert_eq!(
            progress.snapshot(1).current_path.as_deref(),
            Some("kicks/909.wav")
        );
    }
}
