//! Collections: create, list, reorder, export, delete (`task.md` Phase 9).
//!
//! A collection is a sequence the user arranged, not a set -- `db::writer::create_collection`'s
//! doc comment says why `position` exists at all. Every command here that touches an existing
//! collection's membership validates against [`queries::collection_member_ids`] first, for the
//! same reason `set_tag` validates its sample id before the writer sees it: a durable command
//! commits even when it fails, and validating here is what keeps a bad request from reaching
//! the database at all.

use std::path::Path;

use tauri::State;

use crate::{
    commands::samples::require_sample,
    db::{queries, Database},
    error::AppError,
    ipc::types::{Collection, CollectionDetail},
};

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
    collection_by_id(&db, id)
}

/// Every collection, newest first.
#[tauri::command]
pub async fn list_collections(db: State<'_, Database>) -> Result<Vec<Collection>, AppError> {
    let conn = db.read()?;
    Ok(queries::all_collections(&conn)?
        .into_iter()
        .map(|row| Collection {
            id: row.id,
            name: row.name,
            created_at: row.created_at,
            sample_count: row.sample_count,
        })
        .collect())
}

/// One collection with its members, in order.
#[tauri::command]
pub async fn get_collection(
    db: State<'_, Database>,
    collection_id: i64,
) -> Result<CollectionDetail, AppError> {
    let conn = db.read()?;
    let row = queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    let members = queries::collection_members(&conn, collection_id)?;
    Ok(CollectionDetail::new(row, members))
}

/// Rewrites a collection's member order.
///
/// `sample_ids` must be exactly the collection's current membership, reordered. Adding or
/// dropping members has dedicated commands, so a stale reorder cannot quietly make a member
/// disappear. Rejecting a mismatch here, rather than silently reordering the intersection, is what
/// keeps a stale drag-and-drop from quietly losing a sample the frontend forgot it was
/// holding.
#[tauri::command]
pub async fn reorder_collection(
    db: State<'_, Database>,
    collection_id: i64,
    sample_ids: Vec<i64>,
) -> Result<CollectionDetail, AppError> {
    let conn = db.read()?;
    queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    let mut current = queries::collection_member_ids(&conn, collection_id)?;
    drop(conn);

    let mut given = sample_ids.clone();
    current.sort_unstable();
    given.sort_unstable();
    if current != given {
        return Err(AppError::invalid(
            "sampleIds",
            "must be exactly the collection's current members, reordered",
        ));
    }

    db.writer().reorder_collection(collection_id, sample_ids)?;
    get_collection_detail(&db, collection_id)
}

/// Appends samples to an existing collection in the supplied order.
///
/// Repeating an add is safe: current members are preserved and duplicate ids are ignored.
/// That makes the optimistic-looking “Add selection” affordance honest even when the user
/// has already put part of that selection in the collection.
#[tauri::command]
pub async fn add_to_collection(
    db: State<'_, Database>,
    collection_id: i64,
    sample_ids: Vec<i64>,
) -> Result<CollectionDetail, AppError> {
    let conn = db.read()?;
    queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    drop(conn);
    for &sample_id in &sample_ids {
        require_sample(&db, sample_id)?;
    }

    db.writer().add_to_collection(collection_id, sample_ids)?;
    get_collection_detail(&db, collection_id)
}

/// Removes a member without deleting the underlying sound.
#[tauri::command]
pub async fn remove_from_collection(
    db: State<'_, Database>,
    collection_id: i64,
    sample_id: i64,
) -> Result<CollectionDetail, AppError> {
    let conn = db.read()?;
    queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    drop(conn);

    db.writer()
        .remove_from_collection(collection_id, sample_id)?;
    get_collection_detail(&db, collection_id)
}

/// Changes a collection's name without disturbing its saved order.
#[tauri::command]
pub async fn rename_collection(
    db: State<'_, Database>,
    collection_id: i64,
    name: String,
) -> Result<Collection, AppError> {
    if name.trim().is_empty() {
        return Err(AppError::invalid("name", "a collection needs a name"));
    }
    let conn = db.read()?;
    queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    drop(conn);

    db.writer().rename_collection(collection_id, name)?;
    collection_by_id(&db, collection_id)
}

/// Deletes a collection. The samples themselves are untouched.
#[tauri::command]
pub async fn delete_collection(
    db: State<'_, Database>,
    collection_id: i64,
) -> Result<(), AppError> {
    let conn = db.read()?;
    queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    drop(conn);
    db.writer().delete_collection(collection_id)?;
    Ok(())
}

/// Writes a collection's absolute file paths, one per line, to `dest_path`.
///
/// `dest_path` comes from the frontend's native save dialog -- the user chose it through OS
/// UI, the same shape `add_library_root`'s `path` argument already takes. The frontend never
/// discovers or constructs a path on its own; it only ever hands back one the OS dialog gave
/// it.
#[tauri::command]
pub async fn export_collection(
    db: State<'_, Database>,
    collection_id: i64,
    dest_path: String,
) -> Result<(), AppError> {
    let conn = db.read()?;
    queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    let members = queries::collection_members(&conn, collection_id)?;

    let mut lines = Vec::with_capacity(members.len());
    for member in &members {
        let row = queries::sample_row(&conn, member.sample_id)?
            .ok_or_else(|| AppError::not_found("sample", member.sample_id))?;
        lines.push(row.absolute_path().display().to_string());
    }
    drop(conn);

    let dest = Path::new(&dest_path);
    if let Some(parent) = dest.parent() {
        if !parent.is_dir() {
            return Err(AppError::invalid(
                "destPath",
                "the destination folder does not exist",
            ));
        }
    }
    std::fs::write(dest, lines.join("\n"))
        .map_err(|e| AppError::internal("exporting a collection's file list", e))
}

fn collection_by_id(db: &Database, id: i64) -> Result<Collection, AppError> {
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

fn get_collection_detail(db: &Database, collection_id: i64) -> Result<CollectionDetail, AppError> {
    let conn = db.read()?;
    let row = queries::collection(&conn, collection_id)?
        .ok_or_else(|| AppError::not_found("collection", collection_id))?;
    let members = queries::collection_members(&conn, collection_id)?;
    Ok(CollectionDetail::new(row, members))
}
