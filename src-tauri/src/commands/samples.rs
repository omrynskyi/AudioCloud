//! Per-sample commands: detail, neighbors, tags, collections, and the two audio ones
//! (`overview.md` §6.1).

use tauri::State;

use crate::{
    db::{queries, Database},
    error::AppError,
    ipc::types::{Collection, Neighbor, SampleDetail, SampleFeatureBlock, Tag},
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
fn require_sample(db: &Database, sample_id: i64) -> Result<(), AppError> {
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
    Ok(queries::all_tags(&conn)?
        .into_iter()
        .map(|t| Tag {
            id: t.id,
            name: t.name,
            color: t.color,
            sample_count: t.sample_count,
        })
        .collect())
}

/// Creates a collection over the given samples, preserving their order.
#[tauri::command]
pub async fn create_collection(
    db: State<'_, Database>,
    name: String,
    sample_ids: Vec<i64>,
) -> Result<Collection, AppError> {
    if name.trim().is_empty() {
        return Err(AppError::invalid("name", "a collection needs a name"));
    }
    // Same reasoning as `set_tag`: a durable command commits even when it fails, so an
    // unknown id would leave a half-populated collection behind rather than nothing.
    for &sample_id in &sample_ids {
        require_sample(&db, sample_id)?;
    }
    let id = db.writer().create_collection(name, sample_ids)?;

    let conn = db.read()?;
    let row =
        queries::collection(&conn, id)?.ok_or_else(|| AppError::not_found("collection", id))?;
    Ok(Collection {
        id: row.id,
        name: row.name,
        created_at: row.created_at,
        sample_count: row.sample_count,
    })
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

/// Starts previewing a sample.
///
/// **Phase 8 builds the engine behind this** (`task.md` Phase 8: `cpal` stream, lock-free
/// ring, no allocation on the audio thread). The command exists now because its contract is
/// settled and the generated bindings are Phase 6's deliverable -- `task.md`'s parallelism
/// note has Phase 9's shell built against this surface while Phases 4-6 are still running,
/// which is only possible if the surface is complete. Returning
/// [`AppError::Unavailable`] is the honest form of "not yet": the frontend renders the
/// transport controls disabled rather than discovering at runtime that a command it was
/// promised does not exist.
#[tauri::command]
pub async fn play_sample(_sample_id: i64, _gain: f32) -> Result<(), AppError> {
    Err(AppError::Unavailable {
        feature: "audio preview".into(),
    })
}

/// Stops whatever is playing. See [`play_sample`] on why this is not built yet.
#[tauri::command]
pub async fn stop_playback() -> Result<(), AppError> {
    Err(AppError::Unavailable {
        feature: "audio preview".into(),
    })
}
