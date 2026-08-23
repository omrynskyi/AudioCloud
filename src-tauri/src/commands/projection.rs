//! The projection re-fit command (`overview.md` §3.7, §3.8, §6.1).

use tauri::{ipc::Channel, AppHandle, Manager, State};

use crate::{
    commands::Jobs,
    db::Database,
    error::AppError,
    ipc::{
        events::{RefitEvent, RefitFinished, RefitProgressEvent},
        types::{Algorithm, RefitOutcome, RefitParams},
    },
    projection::{
        place_incremental, plan, refit_at_low_priority, PcaProjector, Plan, Projector, Refit,
        UmapParams, UmapProjector,
    },
};

/// Rebuilds the 3D layout and swaps it in, streaming progress.
///
/// Returns a **job id**, not a `projection_runs` id. `overview.md` §6.1 says `runId`, and it
/// cannot be one: §3.8 creates the shadow run only once there are coordinates to put in it,
/// which is minutes into a 50,000-point UMAP fit, so returning it would mean blocking the
/// command for the length of the job. The real run id arrives in the terminal
/// [`RefitEvent`], which is where the frontend needs it anyway -- it is the id of the map
/// that is now on screen, and until the swap there is no such map.
///
/// **The planner decides between placing and re-fitting**, exactly as §3.8 says to, so this
/// is what a "the library changed" hook calls after a scan. A small import is placed into
/// the existing layout without moving a single existing point; a large one, or a library
/// with no map at all, gets a full re-fit aligned onto whatever was there before.
/// `force_full` overrides the planner, because incremental placement accumulates drift and
/// the user is the one who can see that the map has gone crooked.
#[tauri::command]
pub async fn start_refit<R: tauri::Runtime>(
    app: AppHandle<R>,
    jobs: State<'_, Jobs>,
    params: RefitParams,
    on_progress: Channel<RefitEvent>,
) -> Result<i64, AppError> {
    let jobs = jobs.inner().clone();
    let (job_id, cancel, slot) = jobs.begin_refit()?;

    tauri::async_runtime::spawn_blocking(move || {
        let _slot = slot;
        let db = app.state::<Database>();
        let started = std::time::Instant::now();

        let result = run(&db, &params, &cancel, job_id, &on_progress, started);

        let finished = match result {
            Ok(outcome) => RefitFinished {
                job_id,
                outcome: Some(outcome),
                error: None,
            },
            Err(error) => RefitFinished {
                job_id,
                outcome: None,
                error: Some(error),
            },
        };
        // The terminal event, outside the tick schedule and on every path
        // (`overview.md` §6.5). The `Ticker` guarantees one for a job that reached the
        // projector; this guarantees one for a job that failed before it.
        let _ = on_progress.send(RefitEvent::Finished(finished));
    });

    Ok(job_id)
}

/// Cancels the running re-fit.
///
/// Cross-cutting rule 6. A cancelled re-fit discards its shadow run and leaves the map the
/// user is looking at exactly as it was -- there is no half-swapped state to recover from,
/// because the swap is one transaction.
#[tauri::command]
pub async fn cancel_refit(jobs: State<'_, Jobs>, job_id: i64) -> Result<(), AppError> {
    jobs.cancel_refit(job_id)
}

/// The body of the job, off the command so it is readable.
fn run(
    db: &Database,
    params: &RefitParams,
    cancel: &crate::pipeline::CancellationToken,
    job_id: i64,
    channel: &Channel<RefitEvent>,
    started: std::time::Instant,
) -> Result<RefitOutcome, AppError> {
    let decision = plan(db)?;

    // Incremental placement is additive and fast -- there is nothing to stream, and no
    // shadow run to swap. It reports as a single terminal event with `incremental: true`,
    // which is what tells the UI that nothing on the map moved.
    if !params.force_full {
        match decision {
            Plan::UpToDate => {
                let conn = db.read()?;
                let active = crate::db::queries::active_projection_run(&conn)?
                    .ok_or(AppError::NoProjection)?;
                return Ok(RefitOutcome {
                    run_id: active.id,
                    algorithm: active.algorithm,
                    sample_count: active.sample_count as usize,
                    correspondences: 0,
                    relative_displacement: Some(0.0),
                    incremental: true,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                });
            }
            Plan::Incremental { .. } => {
                let report = place_incremental(db, cancel)?;
                return Ok(RefitOutcome {
                    run_id: report.run_id,
                    algorithm: "incremental".into(),
                    sample_count: report.placed,
                    correspondences: 0,
                    relative_displacement: Some(0.0),
                    incremental: true,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                });
            }
            Plan::Full { .. } => {}
        }
    }

    let pca = PcaProjector::new();
    let umap = UmapProjector::new(UmapParams {
        n_neighbors: params
            .n_neighbors
            .unwrap_or_else(|| UmapParams::default().n_neighbors),
        ..Default::default()
    });
    let primary: &dyn Projector = match params.algorithm {
        Algorithm::Umap => &umap,
        Algorithm::Pca => &pca,
    };

    // PCA is always the fallback, including when PCA is what was asked for -- the second
    // attempt then costs nothing and the branch stays uniform. A library with no map at all
    // is worse than a library with a plainer one (`overview.md` §3.7).
    let options = Refit::new(primary, cancel)
        .with_fallback(&pca)
        .with_progress({
            let channel = channel.clone();
            move |snapshot| {
                let _ = channel.send(RefitEvent::Progress(RefitProgressEvent::new(
                    job_id, snapshot,
                )));
            }
        });

    let report = refit_at_low_priority(db, options)?;
    let relative_displacement = report.relative_displacement();
    Ok(RefitOutcome {
        run_id: report.run_id,
        algorithm: report.algorithm.to_string(),
        sample_count: report.sample_count,
        correspondences: report.correspondences,
        relative_displacement,
        incremental: false,
        elapsed_ms: report.elapsed_ms as u64,
    })
}
