//! Library roots and scanning (`overview.md` §6.1).
//!
//! The scan commands are the shape every long operation in this app takes: admit the job or
//! report the one already running, hand the caller an id, run the work on a blocking task,
//! and stream coalesced progress down a `Channel` that is guaranteed to end in a terminal
//! event (`overview.md` §6.5, §6.6).

use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc, Mutex,
};

use tauri::{ipc::Channel, AppHandle, Manager, State};

use crate::{
    audio::peaks::PeakCache,
    commands::Jobs,
    db::{queries, Database, ScanStatus},
    error::AppError,
    ipc::{
        events::{ScanEvent, ScanOutcome},
        types::LibraryRoot,
    },
    model::Model,
    pipeline::{scan_root_with, ScanOptions, ScanProgress},
};

/// Registers a folder as a library root. Idempotent: an existing path returns its row.
///
/// The path is validated here rather than at scan time so the user finds out that they
/// picked a file, or a folder they cannot read, while the folder picker is still on screen.
#[tauri::command]
pub async fn add_library_root(
    db: State<'_, Database>,
    path: String,
    label: Option<String>,
) -> Result<LibraryRoot, AppError> {
    let dir = std::path::Path::new(&path);
    if !dir.is_dir() {
        return Err(AppError::NotFound(path));
    }
    // Canonicalized so the same folder reached two ways -- through a symlink, or with a
    // trailing slash -- is one root rather than two. `library_roots.path` is UNIQUE, and
    // that constraint is only meaningful over a normalized form.
    let path = dir
        .canonicalize()
        .map_err(|e| AppError::internal("canonicalizing a library root", e))?
        .display()
        .to_string();

    let id = db.writer().add_root(path, label)?;
    root_by_id(&db, id)
}

/// Every root the user has added, oldest first, with its sample count.
#[tauri::command]
pub async fn list_library_roots(db: State<'_, Database>) -> Result<Vec<LibraryRoot>, AppError> {
    let conn = db.read()?;
    let roots = queries::library_roots(&conn)?;
    let mut out = Vec::with_capacity(roots.len());
    for root in roots {
        let count = queries::count_samples_for_root(&conn, root.id)?;
        out.push(LibraryRoot::new(root, count));
    }
    Ok(out)
}

/// Removes a root and, by cascade, its samples, features, tags and coordinates.
///
/// Destructive, and deliberately not softened: `status = 'missing'` is how a temporarily
/// unmounted drive is handled (`overview.md` risk 12), and this is the other thing -- the
/// user saying they are done with a folder. The frontend owes this a confirmation, which is
/// Phase 9's.
#[tauri::command]
pub async fn remove_library_root(
    db: State<'_, Database>,
    peaks: State<'_, PeakCache>,
    root_id: i64,
) -> Result<(), AppError> {
    db.writer().remove_root(root_id)?;
    // Sample ids are reused by SQLite, so a cached summary keyed by a freed id would be
    // served for whatever row inherits it next.
    peaks.clear();
    Ok(())
}

/// Enables or disables a root without touching its samples.
///
/// **Not in `overview.md` §6.1**, and added because the column it drives has been in the
/// schema since Phase 1 with nothing able to set it. A disabled root is skipped by
/// `scan_library`'s "every enabled root" pass while keeping its rows, its tags and its place
/// on the map, which is what a user with an external drive that is not currently plugged in
/// actually wants.
#[tauri::command]
pub async fn set_root_enabled(
    db: State<'_, Database>,
    root_id: i64,
    enabled: bool,
) -> Result<LibraryRoot, AppError> {
    db.writer().set_root_enabled(root_id, enabled)?;
    root_by_id(&db, root_id)
}

/// Starts a scan of one root and returns its `scan_runs` id immediately.
///
/// The scan itself runs on a blocking task -- cross-cutting rule 1 -- and reports through
/// `on_progress` at no more than 10 Hz. The command returns as soon as the `scan_runs` row
/// exists, which is one `INSERT`, so the UI gets an id to cancel with without waiting for a
/// walk of 50,000 files.
///
/// Embeds if -- and only if -- a model is installed. A missing model downgrades the scan to
/// decode and DSP rather than failing it: a library is worth indexing before a 200 MB
/// download finishes, and the next scan finishes what this one starts.
#[tauri::command]
pub async fn scan_library<R: tauri::Runtime>(
    app: AppHandle<R>,
    jobs: State<'_, Jobs>,
    root_id: i64,
    on_progress: Channel<ScanEvent>,
) -> Result<i64, AppError> {
    let jobs = jobs.inner().clone();
    let (cancel, slot) = jobs.begin_scan()?;

    // Two things need the scan id and neither can wait for the other: this command, which
    // owes the caller an id, and the blocking task, which owes the channel a terminal event
    // naming the scan that failed. `announce` is the shared oneshot -- resolved by
    // `ScanOptions::on_start` the instant the row is opened, or by the task's own error path
    // if the scan never got that far -- and `assigned` is the plain copy the task reads
    // afterwards.
    let (id_tx, id_rx) = tokio::sync::oneshot::channel::<Result<i64, AppError>>();
    let announce = Arc::new(Mutex::new(Some(id_tx)));
    let assigned = Arc::new(AtomicI64::new(0));

    let on_start = {
        let jobs = jobs.clone();
        let announce = Arc::clone(&announce);
        let assigned = Arc::clone(&assigned);
        move |scan_id: i64| {
            assigned.store(scan_id, Ordering::Relaxed);
            jobs.attach_scan_id(scan_id);
            resolve(&announce, Ok(scan_id));
        }
    };

    tauri::async_runtime::spawn_blocking(move || {
        // `slot` is moved in so the admission is released however this task ends.
        let _slot = slot;
        let db = app.state::<Database>();
        let model = app.state::<Model>();

        let session = match model.status() {
            crate::model::ModelStatus::Installed => match model.session().get() {
                Ok(session) => Some(session),
                Err(e) => {
                    // A session that will not build does not fail the scan. The rows land at
                    // `decoded` and the next scan with a working session finishes them,
                    // which is the same path a scan with no model at all takes.
                    tracing::warn!(error = %e, "scanning without inference: the session would not build");
                    None
                }
            },
            _ => None,
        };

        let mut options = ScanOptions::new(&cancel).on_start(on_start).with_progress({
            let channel = on_progress.clone();
            move |snapshot| {
                // A failed send means the WebView reloaded and the receiver is gone. The
                // scan keeps going: its work is durable, and the reloaded page reads the
                // `scan_runs` row rather than the channel.
                let _ = channel.send(ScanEvent::Progress(snapshot));
            }
        });
        if let Some(session) = session {
            options = options.with_session(session);
        }
        let counters = options.progress();

        let outcome = scan_root_with(&db, root_id, &mut options);

        // A rescanned file is different audio under the same sample id.
        app.state::<PeakCache>().clear();

        let event = match outcome {
            Ok(report) => ScanOutcome::new(
                report.scan_id,
                report.root_id,
                report.status,
                report.counts,
                report.embedded,
                None,
            ),
            // `scan_root_with` returns `Err` only for a scan that never got started -- an
            // unknown root, an unreadable path. A scan that died part-way returns `Ok` with
            // a `failed` status. Either way a terminal event fires, and either way the
            // command's caller learns the outcome (`overview.md` §6.5).
            Err(e) => {
                let error = AppError::from(e);
                resolve(&announce, Err(error.clone()));
                ScanOutcome::new(
                    assigned.load(Ordering::Relaxed),
                    root_id,
                    ScanStatus::Failed,
                    counters.counts(),
                    ScanProgress::read(&counters.embedded),
                    Some(error),
                )
            }
        };
        let _ = on_progress.send(ScanEvent::Finished(event));
    });

    // Waits on the id, not on the scan. `Err` from the receiver means the task ended without
    // resolving it at all, which is a panic in the blocking task and nothing else.
    match id_rx.await {
        Ok(result) => result,
        Err(_) => Err(AppError::internal(
            "starting a scan",
            "the scan task ended without reporting a scan id",
        )),
    }
}

/// Fills a oneshot at most once, tolerating a caller that has already stopped listening.
fn resolve(
    slot: &Mutex<Option<tokio::sync::oneshot::Sender<Result<i64, AppError>>>>,
    value: Result<i64, AppError>,
) {
    if let Ok(mut slot) = slot.lock() {
        if let Some(tx) = slot.take() {
            let _ = tx.send(value);
        }
    }
}

/// Signals the running scan to stop.
///
/// Cooperative, not forced (`overview.md` §6.6): the scan finishes the file it is holding,
/// stops taking new ones, and lets the writer commit what it already has. The `scan_runs`
/// row is marked `cancelled`, partial results are kept -- a half-scanned library is still
/// useful, and a resumed scan skips what is already `embedded`.
#[tauri::command]
pub async fn cancel_scan(jobs: State<'_, Jobs>, scan_id: i64) -> Result<(), AppError> {
    jobs.cancel_scan(scan_id)
}

fn root_by_id(db: &Database, id: i64) -> Result<LibraryRoot, AppError> {
    let conn = db.read()?;
    let root = queries::library_roots(&conn)?
        .into_iter()
        .find(|r| r.id == id)
        .ok_or_else(|| AppError::not_found("library root", id))?;
    let count = queries::count_samples_for_root(&conn, id)?;
    Ok(LibraryRoot::new(root, count))
}
