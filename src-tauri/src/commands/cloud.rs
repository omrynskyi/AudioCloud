//! The three commands that answer in raw bytes (`overview.md` §6.3).
//!
//! Every one of them returns a column of primitives whose length is the size of the library,
//! and JSON is the wrong container for all three. The point cloud is the clearest case: 800
//! KB of `f32` becomes 3-4 MB of text that Rust has to generate, JavaScript has to parse,
//! and the garbage collector has to clean up as 50,000 short-lived objects -- hundreds of
//! milliseconds and a stutter on load, for a payload that arrives as an `ArrayBuffer` and
//! becomes typed-array views with no copying and no parsing at all.
//!
//! These are `async` and blocking-free by different means than the scan commands: each one
//! is a single indexed statement plus a memcpy, measured in single-digit milliseconds at
//! 50,000 rows, so it runs inline on Tauri's worker rather than paying for a
//! `spawn_blocking` hop.

use tauri::{ipc::Response, State};

use crate::{
    db::{queries, search, Database},
    error::AppError,
    ipc::{binary, types::QueryFilter},
};

/// The active layout, as one `ArrayBuffer`.
///
/// **One statement, not two.** The swap that activates a re-fit deletes the run it
/// supersedes in the same transaction, so a caller that read the active run's id and then
/// read its points could see the old id and find no rows under it. The join in
/// [`queries::active_projection_points`] is atomic against the swap under WAL and cannot
/// observe that gap.
///
/// An empty payload -- a bare sixteen-byte header -- is the honest answer before the first
/// re-fit completes, and not an error: a library that has been scanned but not projected is
/// a real state with a real screen behind it, and `NoProjection` would make the renderer
/// treat "no map yet" as a failure.
#[tauri::command]
pub async fn get_point_cloud(db: State<'_, Database>) -> Result<Response, AppError> {
    let conn = db.read()?;
    let points = queries::active_projection_points(&conn)?;
    drop(conn);

    let buf = binary::point_cloud(&points)?;
    tracing::debug!(points = points.len(), bytes = buf.len(), "point cloud");
    Ok(Response::new(buf))
}

/// One scalar column over the active layout, in the point cloud's order.
///
/// The order *is* the join between the two payloads; see [`search::feature_column`]. A
/// `count` that disagrees with the cloud the frontend is holding means a re-fit landed
/// between the two fetches, and the frontend refetches rather than colouring the map by
/// somebody else's numbers.
#[tauri::command]
pub async fn get_feature_column(
    db: State<'_, Database>,
    feature: crate::ipc::types::Feature,
) -> Result<Response, AppError> {
    let conn = db.read()?;
    let values = search::feature_column(&conn, feature)?;
    drop(conn);

    Ok(Response::new(binary::feature_column(&values)))
}

/// Sample ids matching a filter, ascending, as one `ArrayBuffer`.
///
/// Ascending because the point cloud is too, so the renderer's filter mask is one merge over
/// two sorted arrays instead of a `Set` rebuilt on every keystroke.
#[tauri::command]
pub async fn query_samples(
    db: State<'_, Database>,
    filter: QueryFilter,
) -> Result<Response, AppError> {
    let conn = db.read()?;
    let ids = search::sample_ids(&conn, &filter)?;
    drop(conn);

    Ok(Response::new(binary::id_list(&ids)?))
}
