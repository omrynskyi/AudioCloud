//! Application assembly: the Tauri builder, plugin registration, and managed state.
//!
//! Nothing that can block belongs here. Long-lived resources (the SQLite writer thread,
//! the read pool, the `ort` session, the audio engine) are constructed lazily and handed
//! to `.manage()`; see `overview.md` §7 on keeping cold start off the critical path.

pub mod audio;
pub mod commands;
pub mod db;
pub mod error;
pub mod model;
pub mod pipeline;
pub mod projection;
pub mod protocol;

use tauri::Manager;

use crate::db::Database;

/// Dimensionality of a CLAP audio-tower embedding (`overview.md` §3.4).
///
/// Fixed here rather than discovered from the ONNX graph so the data layer can be built and
/// benchmarked before the model exists (Phase 3). Phase 3's parity gate is what proves the
/// two agree.
pub const EMBEDDING_DIM: usize = 512;

/// Builds and runs the desktop application.
///
/// # Panics
/// Panics only if Tauri itself fails to initialize, which is unrecoverable.
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "audiobank=info,warn".into()),
        )
        .init();

    // Deliberate exception to `clippy::expect_used`. If the Tauri runtime itself fails to
    // start there is no window to render an error into and no state worth preserving, so
    // there is nothing for a typed `AppError` to be recovered *by*. Every other failure in
    // this crate is a variant in `error.rs` -- see cross-cutting rule 8.
    #[allow(clippy::expect_used)]
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // ~/Library/Application Support/<bundle-id>/, created on first run.
            let data_dir = app.path().app_data_dir()?;
            let db = Database::open(&data_dir, EMBEDDING_DIM)?;
            app.manage(db);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while starting AudioBank");

    app.run(|app, event| {
        // The writer holds up to 250 ms of uncommitted rows by design. Exiting without
        // draining it throws away the tail of whatever scan was running.
        if matches!(event, tauri::RunEvent::Exit) {
            if let Some(db) = app.try_state::<Database>() {
                db.shutdown();
            }
        }
    });
}
