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
    model::{Model, ModelStatus},
    pipeline::{scan_all_roots_with, CancellationToken, ScanOptions, ScanReport},
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
    pub embedded: u64,
    /// Spectrograms sent through the model, and the `run()` calls that took.
    pub inferred: u64,
    pub inference_batches: u64,
    /// Which execution provider bound, or why no session was used.
    pub inference: String,
    /// Totals across the whole database afterwards, not just this scan.
    pub total_samples: i64,
    pub total_quarantined: i64,
    pub elapsed_ms: u128,
}

impl DevScanSummary {
    fn from_reports(
        reports: &[ScanReport],
        db: &Database,
        inference: String,
        elapsed_ms: u128,
    ) -> Self {
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
            embedded: reports.iter().map(|r| r.embedded).sum(),
            inferred: reports.iter().map(|r| r.inferred).sum(),
            inference_batches: reports.iter().map(|r| r.inference_batches).sum(),
            inference,
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
///
/// Embeds if -- and only if -- a model is installed. A missing model downgrades the scan to
/// decode and DSP rather than failing it: the library is worth indexing before the 200 MB
/// download finishes, and the next scan finishes what this one starts. Progress goes to the
/// log through the real 100 ms ticker, so the throttle is exercised even though Phase 6 owns
/// the `Channel` it will eventually feed.
#[tauri::command]
pub async fn dev_scan(
    db: State<'_, Database>,
    model: State<'_, Model>,
) -> Result<DevScanSummary, String> {
    let started = std::time::Instant::now();
    let cancel = CancellationToken::new();

    let (session, inference) = match model.status() {
        ModelStatus::Installed => match model.session().get() {
            Ok(session) => {
                let provider = session.provider().as_str().to_string();
                (Some(session), provider)
            }
            Err(e) => (None, format!("session init failed: {e}")),
        },
        other => (None, format!("no model: {other:?}")),
    };

    let mut options = ScanOptions::new(&cancel).with_progress(|snapshot| {
        tracing::info!(
            phase = snapshot.phase.as_str(),
            seen = snapshot.files_seen,
            done = snapshot.files_done,
            embedded = snapshot.files_embedded,
            failed = snapshot.files_failed,
            eta_s = snapshot.eta_seconds,
            "scan progress"
        );
    });
    if let Some(session) = session {
        options = options.with_session(session);
    }

    let reports = scan_all_roots_with(&db, options).map_err(|e| e.to_string())?;
    Ok(DevScanSummary::from_reports(
        &reports,
        &db,
        inference,
        started.elapsed().as_millis(),
    ))
}

/// One row of a neighbor listing.
#[derive(Debug, Serialize)]
pub struct DevNeighbor {
    pub rel_path: String,
    pub duration_ms: Option<i64>,
    /// Cosine similarity. Every stored vector is L2-normalized, so this is a dot product.
    pub similarity: f32,
}

/// Nearest neighbors of one sample, by exact cosine over every stored vector.
///
/// **This is the instrument for `overview.md` risk 5**, the open question of whether CLAP
/// says anything useful about a 200 ms hi-hat. `task.md` Phase 4 asks for a hand inspection
/// of a real one-shot library -- do kicks retrieve kicks -- and a hand inspection needs
/// something to look at. Point the app at a drum library, scan it, and call this on a kick.
///
/// Brute force on purpose: an approximate index would put its own recall between the
/// question and the answer, and at 50,000 x 512 f16 one pass over the mmap is well under a
/// second. Phase 5's HNSW is the fast path, and this stays as the exact answer to check it
/// against.
#[tauri::command]
pub async fn dev_neighbors(
    db: State<'_, Database>,
    query: String,
    limit: usize,
) -> Result<Vec<DevNeighbor>, String> {
    let conn = db.read().map_err(|e| e.to_string())?;
    let samples = queries::embedded_samples(&conn).map_err(|e| e.to_string())?;
    drop(conn);

    let needle = query.to_lowercase();
    let target = samples
        .iter()
        .find(|s| s.rel_path.to_lowercase().contains(&needle))
        .ok_or_else(|| format!("no embedded sample matching {query:?}"))?;

    let store = db.embeddings().lock().map_err(|_| "embedding store lock")?;
    let matrix = store.matrix().map_err(|e| e.to_string())?;
    let mut query_vector = Vec::new();
    matrix
        .row_into(target.loc, &mut query_vector)
        .map_err(|e| e.to_string())?;

    let mut scored = Vec::with_capacity(samples.len());
    let mut row = Vec::new();
    for sample in &samples {
        if sample.id == target.id {
            continue;
        }
        if matrix.row_into(sample.loc, &mut row).is_err() {
            continue;
        }
        let similarity: f32 = query_vector
            .iter()
            .zip(row.iter())
            .map(|(a, b)| a * b)
            .sum();
        scored.push((similarity, sample));
    }

    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(scored
        .into_iter()
        .take(limit.clamp(1, 200))
        .map(|(similarity, sample)| DevNeighbor {
            rel_path: sample.rel_path.clone(),
            duration_ms: sample.duration_ms,
            similarity,
        })
        .collect())
}

/// What the app knows about the model without touching the network or the graph.
#[derive(Debug, Serialize)]
pub struct DevModelStatus {
    pub version: String,
    pub status: String,
    pub installed_path: String,
    pub partial_bytes: Option<u64>,
    pub session_initialized: bool,
}

/// Reports model provisioning state. Cheap enough to call on every keystroke.
#[tauri::command]
pub fn dev_model_status(model: State<'_, Model>) -> DevModelStatus {
    DevModelStatus {
        version: model.release().version.to_string(),
        status: match model.status() {
            ModelStatus::Installed => "installed",
            ModelStatus::Downloadable => "downloadable",
            ModelStatus::Unpinned => "unpinned",
        }
        .to_string(),
        installed_path: model.paths().installed().display().to_string(),
        partial_bytes: std::fs::metadata(model.paths().partial())
            .ok()
            .map(|m| m.len()),
        session_initialized: model.session().is_initialized(),
    }
}

/// Downloads and verifies the model, logging progress rather than streaming it.
///
/// Phase 6 replaces the `tracing` calls with a throttled `Channel<DownloadProgress>`; the
/// throttle itself already lives in `model::download` and does not move.
#[tauri::command]
pub async fn dev_download_model(
    db: State<'_, Database>,
    model: State<'_, Model>,
) -> Result<String, String> {
    let downloader = model.downloader(db.data_dir()).map_err(|e| e.to_string())?;
    let path = downloader
        .ensure(&CancellationToken::new(), |p| {
            tracing::info!(
                downloaded = p.downloaded,
                total = p.total,
                "model download progress"
            );
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(path.display().to_string())
}

/// Forces the lazy session to build and reports which execution provider it bound.
///
/// Blocking (session construction is hundreds of milliseconds plus a warmup), hence
/// `async`: cross-cutting rule 1.
#[tauri::command]
pub async fn dev_session_info(model: State<'_, Model>) -> Result<String, String> {
    let session = model.session().get().map_err(|e| e.to_string())?;
    Ok(format!(
        "provider={} embedding_dim={}",
        session.provider().as_str(),
        session.embedding_dim()
    ))
}
