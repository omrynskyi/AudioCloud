//! Application assembly: the Tauri builder, plugin registration, and managed state.
//!
//! Nothing that can block belongs here. Long-lived resources (the SQLite writer thread,
//! the read pool, the audio engine) are constructed lazily and handed
//! to `.manage()`; see `overview.md` §7 on keeping cold start off the critical path.

pub mod audio;
pub mod commands;
pub mod db;
pub mod error;
pub mod ipc;
pub mod pipeline;
pub mod projection;
pub mod protocol;

use std::sync::Arc;

use tauri::Manager;

use crate::{
    audio::{peaks::PeakCache, AudioPlayer},
    commands::Jobs,
    db::Database,
    protocol::peaks,
};

/// Dimensionality of the stored per-sample vector.
///
/// [`pipeline::fingerprint_embed::FINGERPRINT_DIM`] -- the active embedder and the width
/// `embeddings.bin` is opened at.
pub const EMBEDDING_DIM: usize = pipeline::fingerprint_embed::FINGERPRINT_DIM;

/// The command surface (`overview.md` §6.1).
///
/// Every command the frontend can call, in one list. Four of them -- `get_point_cloud`,
/// `get_point_colors`, `get_feature_column`, `query_samples` -- answer in raw bytes; the
/// rest are JSON. Waveform
/// peaks are not here at all: they travel over the `abpeaks://` scheme registered below, so
/// bulk asset traffic never competes with commands (`overview.md` §6.4).
///
/// `pub` so `tests/surface.rs` can mount the same list on a mock runtime. A test that
/// registered its own subset would prove that the subset works and say nothing about the
/// surface the app actually exposes.
pub fn command_handler<R: tauri::Runtime>(
) -> impl Fn(tauri::ipc::Invoke<R>) -> bool + Send + Sync + 'static {
    tauri::generate_handler![
        commands::library::add_library_root,
        commands::library::list_library_roots,
        commands::library::remove_library_root,
        commands::library::set_root_enabled,
        commands::library::scan_library,
        commands::library::cancel_scan,
        commands::cloud::get_point_cloud,
        commands::cloud::get_point_colors,
        commands::cloud::get_feature_column,
        commands::cloud::query_samples,
        commands::samples::get_sample_detail,
        commands::samples::start_sample_drag,
        commands::samples::get_similar,
        commands::samples::set_tag,
        commands::samples::unset_tag,
        commands::samples::list_tags,
        commands::samples::set_tag_color,
        commands::samples::rename_tag,
        commands::samples::delete_tag,
        commands::samples::reveal_in_finder,
        commands::samples::play_sample,
        commands::samples::stop_playback,
        commands::samples::prefetch_sample,
        commands::collections::create_collection,
        commands::collections::list_collections,
        commands::collections::get_collection,
        commands::collections::add_to_collection,
        commands::collections::remove_from_collection,
        commands::collections::rename_collection,
        commands::collections::reorder_collection,
        commands::collections::delete_collection,
        commands::collections::export_collection,
        commands::projection::start_refit,
        commands::projection::cancel_refit,
        commands::settings::get_settings,
        commands::settings::list_audio_devices,
        commands::settings::set_audio_device,
        commands::settings::set_gain,
        commands::settings::reveal_data_dir,
        commands::settings::reset_database,
    ]
}

/// Builds and runs the desktop application.
///
/// # Panics
/// Panics only if Tauri itself fails to initialize, which is unrecoverable.
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "audiocloud=info,warn".into()),
        )
        .init();

    // Deliberate exception to `clippy::expect_used`. If the Tauri runtime itself fails to
    // start there is no window to render an error into and no state worth preserving, so
    // there is nothing for a typed `AppError` to be recovered *by*. Every other failure in
    // this crate is a variant in `error.rs` -- see cross-cutting rule 8.
    #[allow(clippy::expect_used)]
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        // Folder picker for "add a library root" and the destination picker for
        // "export collection" (`task.md` Phase 9) -- see `capabilities/main.json` for why
        // this was not a dependency before this phase.
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(command_handler())
        // Transport 3 (`overview.md` §6.4). Registered on the builder rather than served
        // from a command so waveform fetches get the WebView's own HTTP cache and never
        // queue behind the point cloud on the IPC handler.
        .register_uri_scheme_protocol(peaks::SCHEME, peaks::handle)
        .setup(|app| {
            // ~/Library/Application Support/<bundle-id>/, created on first run.
            let data_dir = app.path().app_data_dir()?;
            let db = Database::open(&data_dir, EMBEDDING_DIM)?;
            app.manage(db);

            // Three empty containers. `Jobs` is three mutexes; `PeakCache` holds one decoder
            // whose buffer pool allocates lazily; `AudioPlayer` holds a `LazyEngine` that does
            // not open an audio device until the first `play_sample`. None of the three reads
            // a file, touches the network, or opens a device during setup.
            app.manage(Jobs::new());
            app.manage(PeakCache::new());

            let player = AudioPlayer::new();
            // Priming the preference is a Mutex write, not a device open -- `AudioPlayer`
            // stays lazy (`overview.md` §7's cold-start budget), and the name just sits
            // ready for whenever the first `play_sample` actually opens a stream.
            //
            // `db` was already moved into `app.manage` above, so this reads it back through
            // the app handle rather than the local binding.
            if let Ok(conn) = app.state::<Database>().read() {
                if let Ok(Some(name)) = crate::db::queries::setting(&conn, "audio_device") {
                    if !name.is_empty() {
                        player.set_preferred_device(Some(name));
                    }
                }
            }
            app.manage(Arc::new(player));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while starting AudioCloud");

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
