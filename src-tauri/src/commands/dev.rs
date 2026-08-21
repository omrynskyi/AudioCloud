//! Temporary development commands.
//!
//! **These are not the IPC surface.** `overview.md` §6.1 is, and Phase 6 builds it: typed
//! errors, `ts-rs` bindings, binary transports, throttled progress channels. What is here is
//! the minimum needed to point Phase 2's pipeline at a real folder and read back counts,
//! and it is compiled out of release builds so it cannot quietly become the thing Phase 6
//! was supposed to replace.
//!
//! Deliberate shortcuts, each of which Phase 6 removes:
//!
//! - errors are strings, not [`crate::error::AppError`] variants;
//! - the scan runs on the calling thread's Tauri worker rather than a tracked job, so there
//!   is no handle to cancel it through;
//! - progress is a return value at the end, not a `Channel` during.

#![cfg(debug_assertions)]

use serde::Serialize;
use tauri::State;

use crate::{
    db::{queries, Database, SampleStatus},
    pipeline::{scan_all_roots, CancellationToken, ScanReport},
};

/// What a dev scan did, flattened for the console.
#[derive(Debug, Serialize)]
pub struct DevScanSummary {
    pub roots_scanned: usize,
    pub files_seen: i64,
    pub files_added: i64,
    pub files_skipped: i64,
    pub files_failed: i64,
    pub decoded: u64,
    pub deduped: u64,
    /// Totals across the whole database afterwards, not just this scan.
    pub total_samples: i64,
    pub total_quarantined: i64,
    pub elapsed_ms: u128,
}

impl DevScanSummary {
    fn from_reports(reports: &[ScanReport], db: &Database, elapsed_ms: u128) -> Self {
        let conn = db.read();
        let (total_samples, total_quarantined) = match &conn {
            Ok(conn) => (
                queries::count_samples(conn).unwrap_or(-1),
                queries::count_samples_with_status(conn, SampleStatus::DecodeFailed).unwrap_or(-1),
            ),
            Err(_) => (-1, -1),
        };

        Self {
            roots_scanned: reports.len(),
            files_seen: reports.iter().map(|r| r.counts.files_seen).sum(),
            files_added: reports.iter().map(|r| r.counts.files_added).sum(),
            files_skipped: reports.iter().map(|r| r.counts.files_skipped).sum(),
            files_failed: reports.iter().map(|r| r.counts.files_failed).sum(),
            decoded: reports.iter().map(|r| r.processed).sum(),
            deduped: reports.iter().map(|r| r.deduped).sum(),
            total_samples,
            total_quarantined,
            elapsed_ms,
        }
    }
}

/// Registers a library root. Idempotent: an existing path returns its existing id.
#[tauri::command]
pub fn dev_add_root(db: State<'_, Database>, path: String) -> Result<i64, String> {
    if !std::path::Path::new(&path).is_dir() {
        return Err(format!("{path} is not a directory"));
    }
    db.writer().add_root(path, None).map_err(|e| e.to_string())
}

/// Every root the user has added.
#[tauri::command]
pub fn dev_list_roots(db: State<'_, Database>) -> Result<Vec<String>, String> {
    let conn = db.read().map_err(|e| e.to_string())?;
    Ok(queries::library_roots(&conn)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|r| format!("{} ({}) enabled={}", r.path, r.id, r.enabled))
        .collect())
}

/// Scans every enabled root and reports counts.
///
/// Blocking, and therefore `async`: Tauri runs an `async` command on its own runtime rather
/// than on the main thread, and cross-cutting rule 1 says nothing that blocks may run there.
#[tauri::command]
pub async fn dev_scan(db: State<'_, Database>) -> Result<DevScanSummary, String> {
    let started = std::time::Instant::now();
    let reports = scan_all_roots(&db, &CancellationToken::new()).map_err(|e| e.to_string())?;
    Ok(DevScanSummary::from_reports(
        &reports,
        &db,
        started.elapsed().as_millis(),
    ))
}
