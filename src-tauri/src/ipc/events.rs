//! What the progress `Channel`s carry (`overview.md` §6.5).
//!
//! Three long operations stream: a scan, a projection re-fit, and the model download. All
//! three follow the same two rules, and both rules are properties of the *pipeline* rather
//! than favours the IPC layer does for it -- the throttle already lives in
//! [`crate::pipeline::Ticker`], and this module only decides what the throttled snapshot
//! looks like on the wire.
//!
//! 1. **Coalesced to ≤ 10 Hz.** One ticker samples atomic counters every 100 ms and emits
//!    one message. Not one message per file: 50,000 of those is 50,000 wakeups of the
//!    WebView during precisely the operation where responsiveness matters most, and nobody
//!    reads 50,000 filenames.
//! 2. **A guaranteed terminal event.** Every stream ends in a `finished` variant --
//!    completion, cancellation, and failure alike -- emitted outside the tick schedule, so
//!    a progress bar can never be left sitting at 99% because the job died between ticks.
//!
//! Each stream is one internally-tagged union rather than two channels, so the terminal
//! event arrives in order behind the last progress event on the same channel. Two channels
//! would let "finished" overtake the progress message that says what finished.

use serde::Serialize;
use ts_rs::TS;

use crate::{
    db::{ScanCounts, ScanStatus},
    error::AppError,
    ipc::types::RefitOutcome,
    pipeline::ProgressSnapshot,
    projection::{RefitPhase, RefitSnapshot},
};

/// What a scan streams back to its caller.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(tag = "event", rename_all = "camelCase")]
#[ts(export_to = "ScanEvent.ts")]
pub enum ScanEvent {
    /// One coalesced reading of the counters. At most ten a second, and skipped entirely
    /// when nothing changed since the last one.
    Progress(ProgressSnapshot),
    /// The terminal event. Always arrives, whatever the outcome.
    Finished(ScanOutcome),
}

/// How a scan ended and what it did.
///
/// A DTO over `db::ScanCounts` plus the counters that are not in the `scan_runs` row.
/// `status` is what the UI renders: a cancelled scan kept everything it wrote and should
/// say so, a failed one has an `error` to show, and a completed one has neither.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "ScanOutcome.ts")]
pub struct ScanOutcome {
    pub scan_id: i64,
    pub root_id: i64,
    /// `running` never appears here; the other three of [`ScanStatus`] do.
    pub status: String,
    pub files_seen: i64,
    pub files_added: i64,
    pub files_skipped: i64,
    pub files_failed: i64,
    /// Files that got a vector, whether the model ran for them or they borrowed a
    /// byte-identical twin's.
    pub files_embedded: u64,
    /// Set only for `status = "failed"`, and only for a failure that stopped the scan --
    /// a file that would not decode is a quarantined row, not a failed scan.
    pub error: Option<AppError>,
}

impl ScanOutcome {
    pub fn new(
        scan_id: i64,
        root_id: i64,
        status: ScanStatus,
        counts: ScanCounts,
        files_embedded: u64,
        error: Option<AppError>,
    ) -> Self {
        Self {
            scan_id,
            root_id,
            status: status.as_str().to_string(),
            files_seen: counts.files_seen,
            files_added: counts.files_added,
            files_skipped: counts.files_skipped,
            files_failed: counts.files_failed,
            files_embedded,
            error,
        }
    }
}

/// What a projection re-fit streams back.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(tag = "event", rename_all = "camelCase")]
#[ts(export_to = "RefitEvent.ts")]
pub enum RefitEvent {
    Progress(RefitProgressEvent),
    Finished(RefitFinished),
}

/// One coalesced reading of a re-fit's counters.
///
/// Carries `job_id` because `RefitSnapshot` does not and cannot: the `projection_runs` row
/// a re-fit ends up writing does not exist while it is fitting (`overview.md` §3.8 creates
/// the shadow run only once there are coordinates to put in it), so the id the caller was
/// given at the start is an app-assigned job id and this is where it gets attached.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "RefitProgressEvent.ts")]
pub struct RefitProgressEvent {
    pub job_id: i64,
    /// `reading` | `fitting` | `aligning` | `writing` | `swapping` | `done`. `fitting` is
    /// the long one and the opaque one -- it is inside the projector, and no counter
    /// crosses that boundary.
    pub phase: String,
    pub samples: u64,
    /// Coordinates committed to the shadow run. Zero until `writing`.
    pub written: u64,
}

impl RefitProgressEvent {
    pub fn new(job_id: i64, snapshot: RefitSnapshot) -> Self {
        Self {
            job_id,
            phase: snapshot.phase.as_str().to_string(),
            samples: snapshot.samples,
            written: snapshot.written,
        }
    }

    /// The terminal phase, for the event that is emitted outside the tick schedule.
    pub fn done(job_id: i64, samples: u64, written: u64) -> Self {
        Self {
            job_id,
            phase: RefitPhase::Done.as_str().to_string(),
            samples,
            written,
        }
    }
}

/// The terminal event of a re-fit.
///
/// Exactly one of `outcome` and `error` is set, and `error` being `Cancelled` is the
/// ordinary case rather than a problem: a cancelled re-fit discards its shadow run and
/// leaves the map the user is looking at untouched (`overview.md` §3.8).
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "RefitFinished.ts")]
pub struct RefitFinished {
    pub job_id: i64,
    pub outcome: Option<RefitOutcome>,
    pub error: Option<AppError>,
}

/// What the model download streams back (`overview.md` §3.5).
#[derive(Debug, Clone, Serialize, TS)]
#[serde(tag = "event", rename_all = "camelCase")]
#[ts(export_to = "DownloadEvent.ts")]
pub enum DownloadEvent {
    Progress(DownloadProgressEvent),
    Finished(DownloadFinished),
}

/// Bytes so far, and the total when the server admits to one.
///
/// `total` is `None` for a chunked response, and the first-run screen has to render that
/// as a live byte count rather than a bar stuck at zero -- which is why it is an `Option`
/// on the wire and not a hopeful guess made in Rust.
#[derive(Debug, Clone, Copy, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "DownloadProgressEvent.ts")]
pub struct DownloadProgressEvent {
    pub downloaded: u64,
    pub total: Option<u64>,
}

impl From<crate::model::download::DownloadProgress> for DownloadProgressEvent {
    fn from(p: crate::model::download::DownloadProgress) -> Self {
        Self {
            downloaded: p.downloaded,
            total: p.total,
        }
    }
}

/// The terminal event of a download.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "DownloadFinished.ts")]
pub struct DownloadFinished {
    /// Where the verified model landed, on success.
    pub path: Option<String>,
    pub error: Option<AppError>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::pipeline::{ProgressSnapshot, ScanPhase};

    fn snapshot() -> ProgressSnapshot {
        ProgressSnapshot {
            scan_id: 7,
            phase: ScanPhase::Embedding,
            files_seen: 100,
            files_queued: 80,
            files_done: 40,
            files_skipped: 20,
            files_failed: 1,
            files_embedded: 39,
            current_path: Some("kicks/909.wav".into()),
            eta_seconds: Some(12),
        }
    }

    /// Internally tagged, so the terminal event and the progress events arrive in order on
    /// one channel. Two channels would let "finished" overtake the message saying what
    /// finished.
    #[test]
    fn a_scan_event_is_one_internally_tagged_union() {
        let json = serde_json::to_value(ScanEvent::Progress(snapshot())).unwrap();
        assert_eq!(json["event"], "progress");
        assert_eq!(json["scanId"], 7);
        assert_eq!(json["phase"], "embedding");
        assert_eq!(json["filesDone"], 40);
        assert_eq!(json["currentPath"], "kicks/909.wav");
        assert_eq!(json["etaSeconds"], 12);

        let json = serde_json::to_value(ScanEvent::Finished(ScanOutcome::new(
            7,
            1,
            ScanStatus::Cancelled,
            ScanCounts {
                files_seen: 100,
                files_added: 40,
                files_skipped: 20,
                files_failed: 1,
            },
            39,
            None,
        )))
        .unwrap();
        assert_eq!(json["event"], "finished");
        assert_eq!(json["status"], "cancelled");
        assert_eq!(json["filesEmbedded"], 39);
        // A cancelled scan kept everything it wrote, so it carries no error. The frontend
        // renders it as "stopped", not as a failure.
        assert!(json["error"].is_null());
    }

    /// The error a terminal event carries is the same tagged union every command rejects
    /// with, so one `switch (error.kind)` covers both paths.
    #[test]
    fn a_failed_stream_carries_the_same_error_type_a_command_would_reject_with() {
        let json = serde_json::to_value(RefitEvent::Finished(RefitFinished {
            job_id: 3,
            outcome: None,
            error: Some(crate::error::AppError::Cancelled),
        }))
        .unwrap();
        assert_eq!(json["event"], "finished");
        assert_eq!(json["error"]["kind"], "cancelled");
        assert!(json["outcome"].is_null());
    }

    #[test]
    fn a_download_with_no_declared_total_says_so_rather_than_guessing() {
        let json = serde_json::to_value(DownloadEvent::Progress(DownloadProgressEvent {
            downloaded: 1024,
            total: None,
        }))
        .unwrap();
        assert_eq!(json["event"], "progress");
        assert_eq!(json["downloaded"], 1024);
        assert!(json["total"].is_null());
    }
}
