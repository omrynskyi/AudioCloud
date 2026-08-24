//! Application assembly: the Tauri builder, plugin registration, and managed state.
//!
//! Nothing that can block belongs here. Long-lived resources (the SQLite writer thread,
//! the read pool, the `ort` session, the audio engine) are constructed lazily and handed
//! to `.manage()`; see `overview.md` §7 on keeping cold start off the critical path.

pub mod audio;
pub mod commands;
pub mod db;
pub mod error;
pub mod ipc;
pub mod model;
pub mod pipeline;
pub mod projection;
pub mod protocol;

use std::sync::Arc;

use tauri::Manager;

use crate::{
    audio::{peaks::PeakCache, AudioPlayer},
    commands::Jobs,
    db::Database,
    model::Model,
    protocol::peaks,
};

/// Dimensionality of a CLAP audio-tower embedding (`overview.md` §3.4).
///
/// Fixed here rather than discovered from the ONNX graph so the data layer can be built and
/// benchmarked before the model exists (Phase 3). Phase 3's parity gate is what proves the
/// two agree.
pub const EMBEDDING_DIM: usize = 512;

/// The command surface (`overview.md` §6.1).
///
/// Every command the frontend can call, in one list. Three of them -- `get_point_cloud`,
/// `get_feature_column`, `query_samples` -- answer in raw bytes; the rest are JSON. Waveform
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
        commands::cloud::get_feature_column,
        commands::cloud::query_samples,
        commands::samples::get_sample_detail,
        commands::samples::get_similar,
        commands::samples::set_tag,
        commands::samples::unset_tag,
        commands::samples::list_tags,
        commands::samples::create_collection,
        commands::samples::reveal_in_finder,
        commands::samples::play_sample,
        commands::samples::stop_playback,
        commands::projection::start_refit,
        commands::projection::cancel_refit,
        commands::model::get_model_status,
        commands::model::download_model,
        commands::model::cancel_download,
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

            // Path joins and an empty cell. The model is not read, the network is not
            // touched, and no `ort` session is built -- `overview.md` §7 budgets cold start
            // to interactive at under two seconds *excluding* ML session init, which is
            // only honest if init genuinely happens somewhere else. See
            // `model::session::LazySession`.
            app.manage(Model::new(&data_dir));

            // Three empty containers. `Jobs` is three mutexes; `PeakCache` holds one decoder
            // whose buffer pool allocates lazily; `AudioPlayer` holds a `LazyEngine` that does
            // not open an audio device until the first `play_sample`. None of the three reads
            // a file, touches the network, or opens a device during setup.
            app.manage(Jobs::new());
            app.manage(PeakCache::new());
            app.manage(Arc::new(AudioPlayer::new()));
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
