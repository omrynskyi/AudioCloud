//! The typed error surface (`overview.md` §6.7).
//!
//! Every failure mode the frontend must render differently is its own variant, tagged for
//! serde so TypeScript can exhaustively match on it. Cross-cutting rule 8: a new failure
//! mode means a new variant and a new rendered state -- never a `.unwrap()`, never a
//! stringly-typed error crossing the IPC boundary.
//!
//! **This is a translation layer, not the error type the crate uses internally.** `DbError`,
//! `PipelineError`, `ProjectionError` and `ModelError` stay where they are, because they
//! carry detail that is useful to a caller inside the process and meaningless to a user --
//! an r2d2 timeout, a ragged embedding row, a `SQLITE_BUSY`. The `From` impls below are the
//! only place that detail is collapsed, and each one is a decision about what recovery the
//! frontend can actually offer.
//!
//! **`Internal` carries a correlation id, not a message.** Anything that reaches it is a bug
//! or an environment failure nobody can act on from a dialog, so the message goes to the log
//! under a short random id and the id goes to the user. "Something went wrong (a3f19c2b)" is
//! a bug report; a serialized `r2d2::Error` is a screenshot nobody can search for.

use serde::Serialize;
use ts_rs::TS;

use crate::{
    audio::{engine::AudioError, PlaybackError},
    db::DbError,
    model::{download::ModelError, ModelStatus},
    pipeline::PipelineError,
    projection::ProjectionError,
};

/// Everything that can cross the IPC boundary as a failure.
///
/// `#[serde(tag = "kind", content = "detail")]` makes this a TypeScript discriminated union:
/// `{ kind: "modelMissing" }`, `{ kind: "decode", detail: { path, reason } }`. The frontend
/// switches on `kind` and renders a recovery action per case.
#[derive(Debug, Clone, thiserror::Error, Serialize, TS)]
#[serde(
    tag = "kind",
    content = "detail",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
#[ts(export_to = "AppError.ts")]
pub enum AppError {
    /// No model is installed, so nothing can be embedded. Recovery: download it.
    #[error("model not installed")]
    ModelMissing,

    /// The download did not finish. Recovery depends on `resumable`, which is why it is on
    /// the wire rather than inferred from the message: an interrupted transfer offers
    /// "Resume", an HTTP 404 offers "Check for a newer release", and the frontend cannot
    /// tell those apart from a string.
    #[error("model download failed: {message}")]
    ModelDownload { message: String, resumable: bool },

    /// The downloaded bytes are not the model. Recovery: discard and re-download; the
    /// partial is already gone.
    #[error("checksum mismatch: expected {expected}, got {got}")]
    ChecksumMismatch { expected: String, got: String },

    /// The build has no published digest to verify a download against
    /// (`crate::model::UNPINNED`). Recovery: none available to the user -- this is a build
    /// that shipped without a pinned model, and saying so is more useful than offering a
    /// download that would refuse itself.
    #[error("model release {version} has no published checksum in this build")]
    ModelUnpinned { version: String },

    /// The data layer failed in a way the user might survive by retrying.
    #[error("database error: {0}")]
    Database(String),

    /// One file would not decode. Carries the path because the quarantine list is a list of
    /// paths, and a decode failure with no path in it cannot be rendered as a row.
    #[error("could not decode {path}: {reason}")]
    Decode { path: String, reason: String },

    /// A sample id, root id, or path that does not exist. Recovery: refresh the list -- the
    /// frontend is holding an id the database has since dropped.
    #[error("not found: {0}")]
    NotFound(String),

    /// A second scan was requested while one was running. Carries the running scan's id so
    /// the frontend can offer "show me the one that is already going" rather than an error.
    #[error("a scan is already running")]
    ScanInProgress { scan_id: i64 },

    /// A second re-fit was requested while one was running.
    #[error("a projection re-fit is already running")]
    RefitInProgress { job_id: i64 },

    /// The user asked to stop. Not a failure; the frontend renders it as "cancelled" and
    /// keeps whatever partial work survived (`overview.md` §6.6).
    #[error("cancelled")]
    Cancelled,

    /// There is no active projection, so there is no map to read, filter or add points to.
    /// Recovery: run a scan, then a re-fit.
    #[error("there is no projection yet")]
    NoProjection,

    /// The library holds fewer embedded samples than the requested operation needs.
    #[error("{operation} needs at least {need} embedded samples, and there are {have}")]
    TooFewSamples {
        operation: String,
        have: usize,
        need: usize,
    },

    /// A well-typed argument whose *value* is not usable -- an empty collection name, a
    /// blank tag.
    ///
    /// **A second deviation from `overview.md` §6.7's variant list.** `NotFound` is about an
    /// id the database does not have; this is about a value the frontend should not have
    /// sent. Collapsing the two produces "Not found: a collection needs a name", which is
    /// both wrong and unactionable. `field` names the input so the UI can highlight it
    /// rather than raising a dialog about the form as a whole.
    #[error("{field}: {reason}")]
    InvalidArgument { field: String, reason: String },

    /// The command exists and its contract is fixed, but the subsystem behind it is not
    /// built yet.
    ///
    /// **A deviation from `overview.md` §6.7's variant list, added deliberately.** The audio
    /// commands in §6.1 were Phase 6's to type and Phase 8's to build; before Phase 8 landed
    /// this was `play_sample`'s and `stop_playback`'s only possible answer. Its remaining
    /// constructor is `start_download`'s "a second download is already running" -- see
    /// `commands::Jobs::begin_download` -- which is a request refused because the feature it
    /// asked for is already in flight, not a subsystem that is missing.
    #[error("{feature} is not available in this build yet")]
    Unavailable { feature: String },

    /// No audio output device could be opened, or the one in use disappeared and a
    /// replacement could not be built.
    ///
    /// Distinct from [`AppError::Internal`] on purpose: a machine with no output device (or
    /// one mid-permission-dialog) is an environment condition a user can plausibly fix --
    /// plug something in, grant the permission -- not a bug to file a correlation id about.
    #[error("no audio output device is available: {0}")]
    AudioDevice(String),

    /// A bug, or an environment failure nobody can act on. The only variant the frontend
    /// renders as "something went wrong", and it carries a log correlation id.
    #[error("internal error ({0})")]
    Internal(String),
}

impl AppError {
    /// Logs `error` under a fresh correlation id and returns the id to the user.
    ///
    /// The `context` is what the log line is *about* -- "reading the point cloud", not the
    /// error's own message, which is already carried by `error`.
    pub fn internal(context: &str, error: impl std::fmt::Display) -> Self {
        let id = correlation_id();
        tracing::error!(correlation_id = %id, context, error = %error, "internal error");
        AppError::Internal(id)
    }

    /// The variant a missing row of table `what` with key `id` should produce.
    pub fn not_found(what: &str, id: impl std::fmt::Display) -> Self {
        AppError::NotFound(format!("{what} {id}"))
    }

    /// A value the frontend should not have sent.
    pub fn invalid(field: &str, reason: &str) -> Self {
        AppError::InvalidArgument {
            field: field.to_string(),
            reason: reason.to_string(),
        }
    }
}

/// A short, searchable id for one logged failure.
///
/// Eight hex characters from the system clock's nanoseconds, hashed so that two failures a
/// microsecond apart do not share a visually confusable prefix. Not a UUID: this is a
/// grep key for a log file that lives on one machine, and sixteen bytes of ceremony would
/// buy nothing a user is going to type into a bug report.
fn correlation_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mixed = nanos.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31);
    format!("{:08x}", (mixed >> 32) as u32)
}

impl From<DbError> for AppError {
    fn from(e: DbError) -> Self {
        match e {
            // A poisoned lock or a dead writer thread means a previous panic. Nothing the
            // user does fixes it and the message would only alarm; it is a bug report.
            DbError::Poisoned(_) | DbError::WriterGone => {
                AppError::internal("the data layer is in an unrecoverable state", e)
            }
            // Everything else -- a busy database, a failed migration, a stale embedding
            // offset -- is at least plausibly transient or fixable by a rescan, and the
            // message says which.
            other => AppError::Database(other.to_string()),
        }
    }
}

impl From<PipelineError> for AppError {
    fn from(e: PipelineError) -> Self {
        match e {
            PipelineError::Db(db) => db.into(),
            PipelineError::UnknownRoot(id) => AppError::not_found("library root", id),
            PipelineError::RootUnreadable { ref path } => {
                AppError::NotFound(path.display().to_string())
            }
        }
    }
}

impl From<ProjectionError> for AppError {
    fn from(e: ProjectionError) -> Self {
        match e {
            ProjectionError::Db(db) => db.into(),
            ProjectionError::Cancelled => AppError::Cancelled,
            ProjectionError::NoActiveProjection => AppError::NoProjection,
            ProjectionError::TooFewSamples {
                algorithm,
                have,
                need,
            } => AppError::TooFewSamples {
                operation: algorithm.to_string(),
                have,
                need,
            },
            // A ragged matrix, a degenerate corpus, or `annembed` refusing. All three are
            // states no button fixes: they describe the *data*, and the honest rendering is
            // "the map could not be built", with the reason in the log.
            other => AppError::internal("the projection could not be built", other),
        }
    }
}

impl From<AudioError> for AppError {
    fn from(e: AudioError) -> Self {
        AppError::AudioDevice(e.to_string())
    }
}

impl From<PlaybackError> for AppError {
    fn from(e: PlaybackError) -> Self {
        match e {
            PlaybackError::NotFound(id) => AppError::not_found("sample", id),
            PlaybackError::Db(db) => db.into(),
            PlaybackError::Decode { path, reason } => AppError::Decode { path, reason },
            PlaybackError::Device(audio) => audio.into(),
            // A resample failure is a `rubato` internal error over a well-formed buffer this
            // crate built -- nothing about it is the user's file or the user's device, so it
            // gets the same treatment as any other "should not happen" failure.
            PlaybackError::Resample(reason) => {
                AppError::internal("resampling audio for playback", reason)
            }
        }
    }
}

impl From<ModelError> for AppError {
    fn from(e: ModelError) -> Self {
        let resumable = e.is_resumable();
        match e {
            ModelError::ReleaseNotPinned { version } => AppError::ModelUnpinned { version },
            ModelError::HashMismatch { expected, actual } => AppError::ChecksumMismatch {
                expected,
                got: actual,
            },
            ModelError::Cancelled { .. } => AppError::Cancelled,
            other => AppError::ModelDownload {
                message: other.to_string(),
                resumable,
            },
        }
    }
}

/// What a command that needs an installed model should fail with, given the model's state.
///
/// Two states, two different screens: "download it" is an action, "this build shipped
/// without a pinned release" is not, and collapsing them into one error would put a button
/// in front of the user that cannot work.
pub fn require_model(status: ModelStatus) -> Result<(), AppError> {
    match status {
        ModelStatus::Installed => Ok(()),
        ModelStatus::Downloadable => Err(AppError::ModelMissing),
        ModelStatus::Unpinned => Err(AppError::ModelUnpinned {
            version: crate::model::ModelRelease::CURRENT.version.to_string(),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// The shape the frontend switches on. A change here is a breaking change to
    /// `src/bindings/AppError.ts`, which is what the CI bindings check exists to catch --
    /// but a tagged enum that silently stopped being tagged would generate *valid*
    /// TypeScript for the wrong wire format, so the tag itself is asserted here.
    #[test]
    fn errors_serialize_as_a_tagged_union() {
        let json = serde_json::to_value(AppError::ModelMissing).unwrap();
        assert_eq!(json, serde_json::json!({ "kind": "modelMissing" }));

        let json = serde_json::to_value(AppError::Decode {
            path: "kicks/909.wav".into(),
            reason: "the stream is damaged".into(),
        })
        .unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "kind": "decode",
                "detail": { "path": "kicks/909.wav", "reason": "the stream is damaged" }
            })
        );

        let json = serde_json::to_value(AppError::NotFound("sample 12".into())).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "kind": "notFound", "detail": "sample 12" })
        );
    }

    /// A dead writer is a bug, not a database message to show the user.
    #[test]
    fn an_unrecoverable_data_layer_becomes_internal() {
        assert!(matches!(
            AppError::from(DbError::WriterGone),
            AppError::Internal(_)
        ));
        assert!(matches!(
            AppError::from(DbError::Dimension {
                expected: 512,
                actual: 8
            }),
            AppError::Database(_)
        ));
    }

    /// Cancellation is not a download failure, and the frontend must not offer "Retry" for
    /// something the user chose.
    #[test]
    fn a_cancelled_download_is_cancellation() {
        assert!(matches!(
            AppError::from(ModelError::Cancelled { downloaded: 17 }),
            AppError::Cancelled
        ));
    }

    /// Resumability is a field, not a phrase inside a message.
    #[test]
    fn a_download_error_says_whether_resuming_would_help() {
        let err = AppError::from(ModelError::Http {
            status: 404,
            url: "https://example.invalid/model.onnx".into(),
        });
        match err {
            AppError::ModelDownload { resumable, .. } => assert!(!resumable),
            other => panic!("expected a download error, got {other:?}"),
        }
    }

    #[test]
    fn a_correlation_id_is_short_and_hex() {
        let id = correlation_id();
        assert_eq!(id.len(), 8);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
