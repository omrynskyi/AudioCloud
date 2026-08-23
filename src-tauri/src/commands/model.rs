//! Model provisioning commands (`overview.md` §3.5, §6.1).

use tauri::{ipc::Channel, AppHandle, Manager, State};

use crate::{
    commands::Jobs,
    db::Database,
    error::AppError,
    ipc::{
        events::{DownloadEvent, DownloadFinished, DownloadProgressEvent},
        types::{ModelState, ModelStatus},
    },
    model::Model,
};

/// What the app knows about the model, without touching the network or the graph.
///
/// Cheap enough to poll: three path stats and no IO beyond them. In particular it does
/// **not** build the `ort` session -- `overview.md` §7 budgets cold start to interactive
/// under two seconds *excluding* session init, which is only honest if a status badge
/// cannot trigger it.
#[tauri::command]
pub async fn get_model_status(model: State<'_, Model>) -> Result<ModelStatus, AppError> {
    let release = model.release();
    Ok(ModelStatus {
        version: release.version.to_string(),
        state: ModelState::from(model.status()),
        path: model.paths().installed().display().to_string(),
        total_bytes: release.bytes,
        downloaded_bytes: std::fs::metadata(model.paths().partial())
            .ok()
            .map(|m| m.len()),
        session_ready: model.session().is_initialized(),
    })
}

/// Downloads and verifies the model, streaming progress.
///
/// Resumable, SHA-256 verified, installed by an atomic rename -- all of that lives in
/// [`crate::model::download`] and none of it moves here. What this adds is the `Channel`:
/// the downloader already throttles its own callback, so the events on the wire are the
/// same 10 Hz every other progress stream in the app uses (cross-cutting rule 3).
///
/// Returns as soon as the download is admitted. The outcome arrives as the terminal event.
#[tauri::command]
pub async fn download_model<R: tauri::Runtime>(
    app: AppHandle<R>,
    jobs: State<'_, Jobs>,
    on_progress: Channel<DownloadEvent>,
) -> Result<(), AppError> {
    let jobs = jobs.inner().clone();
    let (cancel, slot) = jobs.begin_download()?;

    // `spawn`, not `spawn_blocking`: `Downloader::ensure` is genuinely async -- it awaits a
    // `reqwest` stream and writes through a `tokio::fs` handle -- and putting an async
    // future on a blocking thread would occupy the thread while it waits on the network.
    tauri::async_runtime::spawn(async move {
        let _slot = slot;
        let db = app.state::<Database>();
        let model = app.state::<Model>();

        let result = async {
            let downloader = model.downloader(db.data_dir())?;
            downloader
                .ensure(&cancel, |p| {
                    let _ =
                        on_progress.send(DownloadEvent::Progress(DownloadProgressEvent::from(p)));
                })
                .await
        }
        .await;

        let finished = match result {
            Ok(path) => DownloadFinished {
                path: Some(path.display().to_string()),
                error: None,
            },
            Err(e) => DownloadFinished {
                path: None,
                error: Some(e.into()),
            },
        };
        let _ = on_progress.send(DownloadEvent::Finished(finished));
    });

    Ok(())
}

/// Stops the running download.
///
/// Cross-cutting rule 6: every long operation is cancellable. **Not in `overview.md` §6.1**,
/// which lists `cancel_scan` and stops there -- a 200 MB transfer on a metered connection is
/// exactly the kind of long operation that rule is about, and the download layer already
/// keeps its `.partial` on cancel so the next call resumes rather than restarting.
#[tauri::command]
pub async fn cancel_download(jobs: State<'_, Jobs>) -> Result<(), AppError> {
    jobs.cancel_download()
}
