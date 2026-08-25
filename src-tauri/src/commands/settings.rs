//! Settings: persisted audio device/gain, device enumeration, data-dir reveal, and
//! reset-database (`task.md` Phase 9).

use std::sync::Arc;

use tauri::{AppHandle, State};

use crate::{
    audio::{self, peaks::PeakCache, AudioPlayer},
    commands::Jobs,
    db::{queries, Database},
    error::AppError,
    ipc::types::AudioDeviceInfo,
};

pub use crate::ipc::types::AppSettings;

const KEY_AUDIO_DEVICE: &str = "audio_device";
const KEY_GAIN: &str = "gain";

/// Default gain when nothing has been persisted yet. Unity: a first-run user should hear a
/// sample at the level it was recorded at, not at some house-picked loudness.
const DEFAULT_GAIN: f32 = 1.0;

/// The settings that survive a restart, read with defaults.
#[tauri::command]
pub async fn get_settings(db: State<'_, Database>) -> Result<AppSettings, AppError> {
    read_settings(&db)
}

/// Every output device this machine can see.
#[tauri::command]
pub async fn list_audio_devices() -> Result<Vec<AudioDeviceInfo>, AppError> {
    Ok(audio::list_output_devices()
        .map_err(|e| AppError::internal("listing audio output devices", e))?
        .into_iter()
        .map(AudioDeviceInfo::from)
        .collect())
}

/// Persists a preferred output device (or clears it, for `None` = "use the OS default") and
/// switches the live stream if one is already open.
#[tauri::command]
pub async fn set_audio_device(
    db: State<'_, Database>,
    player: State<'_, Arc<AudioPlayer>>,
    name: Option<String>,
) -> Result<AppSettings, AppError> {
    db.writer()
        .set_setting(KEY_AUDIO_DEVICE, name.clone().unwrap_or_default())?;
    player.set_preferred_device(name);
    read_settings(&db)
}

/// Persists the master gain. `play_sample`'s own `gain` argument is still what actually gets
/// applied to a given playback -- this is only what the Settings slider remembers.
#[tauri::command]
pub async fn set_gain(db: State<'_, Database>, gain: f32) -> Result<AppSettings, AppError> {
    if !gain.is_finite() || gain < 0.0 {
        return Err(AppError::invalid("gain", "must be a non-negative number"));
    }
    db.writer().set_setting(KEY_GAIN, gain.to_string())?;
    read_settings(&db)
}

/// Reveals the app's data directory (the database, the embedding store) in Finder.
#[tauri::command]
pub async fn reveal_data_dir(db: State<'_, Database>) -> Result<(), AppError> {
    tauri_plugin_opener::reveal_item_in_dir(db.data_dir())
        .map_err(|e| AppError::internal("revealing the data directory", e))
}

/// Deletes the whole library and relaunches the app.
///
/// **Why a relaunch, not a live rebuild.** `Database`'s writer, read pool and embedding store
/// are relied on by every command in the crate to simply exist; making them replaceable while
/// the app keeps running would put a lock around state that today needs none. Tearing the
/// process down and starting fresh gets the same result -- an empty library on next use --
/// without that. `AppHandle::restart` is not used here because of an open upstream bug where
/// it only quits on macOS without relaunching; the relaunch is done by hand.
///
/// **This command's reply may never arrive.** The process exits before Tauri can serialize
/// one back. The frontend must treat "invoked it, then nothing" as the expected success path,
/// not a hang.
#[tauri::command]
pub async fn reset_database<R: tauri::Runtime>(
    app: AppHandle<R>,
    db: State<'_, Database>,
    jobs: State<'_, Jobs>,
    peaks: State<'_, PeakCache>,
) -> Result<(), AppError> {
    if jobs.any_running() {
        return Err(AppError::Unavailable {
            feature: "resetting the database while a scan, re-fit or download is running".into(),
        });
    }

    db.shutdown();
    peaks.clear();

    let dir = db.data_dir();
    for name in [
        crate::db::DB_FILENAME,
        "library.db-wal",
        "library.db-shm",
        "library.db-journal",
        crate::db::embeddings::EMBEDDINGS_FILENAME,
    ] {
        if let Err(e) = std::fs::remove_file(dir.join(name)) {
            if e.kind() != std::io::ErrorKind::NotFound {
                return Err(AppError::internal(
                    "resetting the database",
                    format!("removing {name}: {e}"),
                ));
            }
        }
    }

    let exe = std::env::current_exe()
        .map_err(|e| AppError::internal("resetting the database", format!("current_exe: {e}")))?;
    std::process::Command::new(exe)
        .spawn()
        .map_err(|e| AppError::internal("resetting the database", format!("relaunch: {e}")))?;

    app.exit(0);
    Ok(())
}

fn read_settings(db: &Database) -> Result<AppSettings, AppError> {
    let conn = db.read()?;
    let audio_device = queries::setting(&conn, KEY_AUDIO_DEVICE)?.filter(|s| !s.is_empty());
    let gain = queries::setting(&conn, KEY_GAIN)?
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(DEFAULT_GAIN);
    Ok(AppSettings { audio_device, gain })
}
