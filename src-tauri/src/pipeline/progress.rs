//! Progress reporting: atomic counters per stage, drained by a single 100 ms ticker.
//!
//! Cross-cutting rule 3: no per-file IPC events. 50,000 files emitting individually would
//! saturate the WebView message port and stall the render loop; the counters are cheap and
//! the ticker coalesces them to <= 10 Hz (`overview.md` §6.5).
//!
//! Phase 2 populates the counters and reads them once the scan has joined every producer.
//! The ticker and the `Channel<ScanProgress>` that drains them to the frontend are Phase 4
//! and Phase 6 -- the shape here is chosen so that adding them changes nothing on the
//! producer side.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::db::ScanCounts;

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
    /// Files quarantined: unreadable, undecodable, or not audio after all.
    pub failed: AtomicU64,
    /// Rows handed to the writer and committed.
    pub persisted: AtomicU64,
}

impl ScanProgress {
    pub fn new() -> Self {
        Self::default()
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
}
