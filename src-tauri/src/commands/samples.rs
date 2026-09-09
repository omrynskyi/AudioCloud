//! Per-sample commands: detail, neighbors, tags, and the two audio ones (`overview.md` §6.1).
//! Collection commands are `commands::collections`.

use std::sync::{mpsc, Arc};

use tauri::{Runtime, State, Window};

use crate::{
    audio::AudioPlayer,
    db::{queries, Database},
    error::AppError,
    ipc::types::{Neighbor, SampleDetail, SampleFeatureBlock, Tag},
};

/// Everything the inspector shows for one sample.
#[tauri::command]
pub async fn get_sample_detail(
    db: State<'_, Database>,
    sample_id: i64,
) -> Result<SampleDetail, AppError> {
    let conn = db.read()?;
    let row = queries::sample_row(&conn, sample_id)?
        .ok_or_else(|| AppError::not_found("sample", sample_id))?;
    let features = queries::sample_features(&conn, sample_id)?.map(SampleFeatureBlock::from);
    let tags = queries::tags_for_sample(&conn, sample_id)?;
    drop(conn);

    Ok(SampleDetail {
        id: row.id,
        root_id: row.root_id,
        root_path: row.root_path,
        rel_path: row.rel_path,
        filename: row.filename,
        ext: row.ext,
        size_bytes: row.size_bytes,
        duration_ms: row.duration_ms,
        sample_rate: row.sample_rate,
        channels: row.channels,
        status: row.status.as_str().to_string(),
        error: row.error,
        embedded: row.embedded,
        features,
        tags,
        updated_at: row.updated_at,
    })
}

/// Starts an operating-system file drag for one sample.
///
/// HTML drag data is enough for reordering rows inside the WebView, but Finder and DAWs need
/// a native file-list pasteboard. The frontend still supplies only a sample id: the absolute
/// path is resolved and checked here, preserving the same boundary as [`reveal_in_finder`].
/// `drag::start_drag` must be called on AppKit's main thread, while a Tauri async command is
/// not guaranteed to run there, hence the small one-shot channel around `run_on_main_thread`.
#[tauri::command]
pub async fn start_sample_drag<R: Runtime>(
    window: Window<R>,
    db: State<'_, Database>,
    sample_id: i64,
) -> Result<(), AppError> {
    let conn = db.read()?;
    let row = queries::sample_row(&conn, sample_id)?
        .ok_or_else(|| AppError::not_found("sample", sample_id))?;
    drop(conn);

    let path = row.absolute_path();
    if !path.is_file() {
        return Err(AppError::NotFound(path.display().to_string()));
    }

    let drag_window = window.clone();
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    window
        .run_on_main_thread(move || {
            let result = drag::start_drag(
                &drag_window,
                drag::DragItem::Files(vec![path]),
                drag::Image::Raw(include_bytes!("../../icons/128x128.png").to_vec()),
                |_result, _cursor_position| {},
                drag::Options::default(),
            );
            let _ = started_tx.send(result);
        })
        .map_err(|e| AppError::internal("dispatching a sample drag", e))?;

    started_rx
        .recv()
        .map_err(|e| AppError::internal("waiting for a sample drag to start", e))?
        .map_err(|e| AppError::internal("starting a sample drag", e))
}

/// The `k` nearest samples to `sample_id` by cosine similarity.
///
/// **Exact, by one pass over the memory-mapped matrix.** Phase 5 built an HNSW index, and it
/// belongs to the projection: it is constructed inside a re-fit, over a corpus snapshot,
/// tuned for embedding quality rather than query latency, and it does not outlive the job.
/// Building a second, persistent index to answer a query the user issues a few times a
/// minute would put an approximate answer's recall between the question and the truth for no
/// gain -- 50,000 x 512 f16 is 51 MB of sequential page-cache reads and 25 million multiply-
/// adds, which is tens of milliseconds. If a future phase needs this at interactive rates
/// over a much larger corpus, the index is the answer; today it would be a cache with a
/// coherence problem.
///
/// `async` because that pass is real work and cross-cutting rule 1 keeps it off the main
/// thread.
#[tauri::command]
pub async fn get_similar(
    db: State<'_, Database>,
    sample_id: i64,
    k: usize,
) -> Result<Vec<Neighbor>, AppError> {
    let k = k.clamp(1, 200);

    let conn = db.read()?;
    let target = queries::embedding_loc(&conn, sample_id)?
        .ok_or_else(|| AppError::NotFound(format!("sample {sample_id} has no embedding")))?;
    let samples = queries::embedded_samples(&conn)?;
    drop(conn);

    let matrix = {
        let store = db
            .embeddings()
            .lock()
            .map_err(|_| AppError::internal("locking the embedding store", "poisoned"))?;
        store.matrix()?
    };

    let mut query = Vec::new();
    matrix.row_into(target, &mut query)?;

    let mut scored: Vec<(f32, &queries::EmbeddedSample)> = Vec::with_capacity(samples.len());
    let mut row = Vec::new();
    for sample in &samples {
        if sample.id == sample_id {
            continue;
        }
        // A stale offset is one row missing from one neighbor list, not a failed command.
        // The alternative -- refusing to answer because one of 50,000 locations is out of
        // range -- is worse for the user and no more correct.
        if matrix.row_into(sample.loc, &mut row).is_err() {
            continue;
        }
        let similarity: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        scored.push((similarity, sample));
    }

    // `total_cmp` rather than `partial_cmp`: a NaN similarity would make a `partial_cmp`
    // sort's comparator inconsistent, and an inconsistent comparator is a panic in the
    // standard library's sort, not a wrong order.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(scored
        .into_iter()
        .take(k)
        .map(|(similarity, sample)| Neighbor {
            sample_id: sample.id,
            filename: sample
                .rel_path
                .rsplit('/')
                .next()
                .unwrap_or(&sample.rel_path)
                .to_string(),
            rel_path: sample.rel_path.clone(),
            duration_ms: sample.duration_ms,
            similarity,
        })
        .collect())
}

/// Attaches a tag, creating it on first use. Idempotent.
///
/// Both arguments are checked before the writer sees them, and the reason is worth stating:
/// the writer wraps a durable command in one transaction and **commits it even when the
/// command fails**, because the rows the ingest path writes are all re-derivable and a
/// savepoint per command is a real cost at ingest rates. Tags are not ingest. An unknown
/// `sample_id` would fail on `sample_tags`' foreign key *after* the `tags` row had been
/// created, leaving a tag the user never finished making. Validating here is what keeps that
/// out of the database, and it turns a `Database` error into the two typed ones the UI can
/// actually render.
#[tauri::command]
pub async fn set_tag(
    db: State<'_, Database>,
    sample_id: i64,
    tag_name: String,
) -> Result<Vec<String>, AppError> {
    let name = require_tag_name(&tag_name)?;
    require_sample(&db, sample_id)?;

    db.writer().set_tag(sample_id, name)?;
    let conn = db.read()?;
    Ok(queries::tags_for_sample(&conn, sample_id)?)
}

/// Removes a tag from a sample. The tag itself survives with a count of zero.
#[tauri::command]
pub async fn unset_tag(
    db: State<'_, Database>,
    sample_id: i64,
    tag_name: String,
) -> Result<Vec<String>, AppError> {
    let name = require_tag_name(&tag_name)?;
    require_sample(&db, sample_id)?;

    db.writer().unset_tag(sample_id, name)?;
    let conn = db.read()?;
    Ok(queries::tags_for_sample(&conn, sample_id)?)
}

/// Rejects a blank tag name. The writer treats one as a no-op, which would leave the caller
/// unable to tell "tagged with nothing" from "tagged".
fn require_tag_name(tag_name: &str) -> Result<&str, AppError> {
    let name = tag_name.trim();
    if name.is_empty() {
        return Err(AppError::invalid("tagName", "a tag needs a name"));
    }
    Ok(name)
}

/// Rejects a sample id the database does not have. See [`set_tag`] on why this is not left
/// to the foreign key.
///
/// `pub(crate)`: `commands::collections` validates the same way before touching the writer.
pub(crate) fn require_sample(db: &Database, sample_id: i64) -> Result<(), AppError> {
    let conn = db.read()?;
    if queries::sample_row(&conn, sample_id)?.is_none() {
        return Err(AppError::not_found("sample", sample_id));
    }
    Ok(())
}

/// Every tag, with its usage count.
#[tauri::command]
pub async fn list_tags(db: State<'_, Database>) -> Result<Vec<Tag>, AppError> {
    let conn = db.read()?;
    Ok(queries::all_tags(&conn)?.into_iter().map(tag_dto).collect())
}

/// Sets (or clears, for `None`) a tag's display color.
#[tauri::command]
pub async fn set_tag_color(
    db: State<'_, Database>,
    tag_id: i64,
    color: Option<String>,
) -> Result<Tag, AppError> {
    let conn = db.read()?;
    queries::tag(&conn, tag_id)?.ok_or_else(|| AppError::not_found("tag", tag_id))?;
    drop(conn);

    db.writer().set_tag_color(tag_id, color)?;

    let conn = db.read()?;
    let row = queries::tag(&conn, tag_id)?.ok_or_else(|| AppError::not_found("tag", tag_id))?;
    Ok(tag_dto(row))
}

fn tag_dto(row: queries::TagRow) -> Tag {
    Tag {
        id: row.id,
        name: row.name,
        color: row.color,
        sample_count: row.sample_count,
    }
}

/// Selects the sample's file in Finder.
///
/// **The path is built in Rust, from the database, and never comes from the frontend.** The
/// WebView has no filesystem permission at all (`overview.md` §2) and asks for samples by
/// id; this command is what "reveal" means without handing the renderer a path argument it
/// could have made up.
#[tauri::command]
pub async fn reveal_in_finder(db: State<'_, Database>, sample_id: i64) -> Result<(), AppError> {
    let conn = db.read()?;
    let row = queries::sample_row(&conn, sample_id)?
        .ok_or_else(|| AppError::not_found("sample", sample_id))?;
    drop(conn);

    let path = row.absolute_path();
    if !path.exists() {
        return Err(AppError::NotFound(path.display().to_string()));
    }
    tauri_plugin_opener::reveal_item_in_dir(&path)
        .map_err(|e| AppError::internal("revealing a sample in Finder", e))
}

/// Starts previewing a sample, retriggering cleanly if one is already playing.
///
/// Hover-to-audition's debounce and click-to-play's immediacy are both the frontend's call --
/// `overview.md` §6.1 fixes this command at `(sampleId, gain)`, with nothing on the wire to
/// tell the two apart -- but every call here is safe to make as fast as the frontend likes:
/// [`AudioPlayer::play`] always retriggers through a fresh envelope rather than clicking.
#[tauri::command]
pub async fn play_sample(
    player: State<'_, Arc<AudioPlayer>>,
    db: State<'_, Database>,
    sample_id: i64,
    gain: f32,
) -> Result<(), AppError> {
    player.play(&db, sample_id, gain).await?;
    Ok(())
}

/// Stops whatever is playing.
#[tauri::command]
pub async fn stop_playback(player: State<'_, Arc<AudioPlayer>>) -> Result<(), AppError> {
    player.stop()?;
    Ok(())
}

/// Warms the decode cache for a sample the user has not asked to hear yet -- typically a
/// neighbor of whatever they are currently hovering, so that hovering it next is a cache hit
/// instead of a cold decode. A no-op, not an error, if no device has been opened yet or the
/// sample is gone; see [`AudioPlayer::prefetch`] for why.
#[tauri::command]
pub async fn prefetch_sample(
    player: State<'_, Arc<AudioPlayer>>,
    db: State<'_, Database>,
    sample_id: i64,
) -> Result<(), AppError> {
    player.prefetch(&db, sample_id).await?;
    Ok(())
}
