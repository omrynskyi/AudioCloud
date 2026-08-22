//! CLAP model provisioning and inference session management (`overview.md` §3.5).
//!
//! The model is a release asset, not a repo artifact -- 200 MB does not belong in git.
//! It is downloaded on first run, verified by SHA-256, and versioned independently of the
//! app so a model revision is a download prompt rather than a full app update.
//!
//! Two halves, deliberately separate:
//!
//! - [`download`] gets the bytes onto disk and proves they are the right bytes. It knows
//!   nothing about inference.
//! - [`session`] turns those bytes into a warm `ort` session and is the **only** file in
//!   the crate permitted to name an `ort` type (cross-cutting rule 7).
//!
//! What ties them together is [`ModelRelease`]: a version, a URL, and a hash compiled into
//! the binary. The hash is the whole security and correctness story -- a truncated,
//! corrupted, or substituted model that still loads produces embeddings that are wrong and
//! plausible, which is the failure mode this module exists to make impossible.

pub mod download;
pub mod session;

use std::path::{Path, PathBuf};

/// A published model artifact: what to fetch, and how to know it arrived intact.
///
/// Versioned independently of the app (`task.md` Phase 11) so shipping a better checkpoint
/// is a download prompt rather than a release. [`ModelPaths`] puts the version in the
/// filename for the same reason: two versions can coexist on disk while one is in use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRelease {
    /// Model version, independent of the app's. Appears in the on-disk filename.
    pub version: &'static str,
    /// Where the asset is published.
    pub url: &'static str,
    /// Lowercase hex SHA-256 of the asset. See [`ModelRelease::is_pinned`].
    pub sha256: &'static str,
    /// Expected size in bytes, for progress and for a cheap early rejection of an
    /// obviously wrong asset. `None` when the release has not been measured.
    pub bytes: Option<u64>,
}

/// The sentinel [`ModelRelease::sha256`] carries until the export has actually been run and
/// the asset published.
///
/// It exists so the un-run state is a *typed, rendered* state rather than a lie. The
/// alternative -- putting a plausible-looking hash in the source before anything has been
/// hashed -- fails at the moment of verification with "the model you downloaded is
/// corrupt", which is both wrong and unactionable. This fails at the moment of the request
/// with [`download::ModelError::ReleaseNotPinned`], which says what is actually true.
pub const UNPINNED: &str = "0000000000000000000000000000000000000000000000000000000000000000";

impl ModelRelease {
    /// The release this build expects.
    ///
    /// `sha256` and `bytes` are filled in from the output of
    /// `scripts/export_clap_onnx.py`, which prints both, and `url` from wherever the asset
    /// was published. Until then [`ModelRelease::is_pinned`] is false and
    /// [`download::Downloader`] refuses to fetch anything -- see [`UNPINNED`].
    pub const CURRENT: ModelRelease = ModelRelease {
        version: "clap-audio-v1",
        // TODO: this repository has no remote yet, so the asset is not published anywhere
        // and this URL 404s. The digest below is real -- it is the export that passed the
        // parity gate -- so `status()` reports `Downloadable` and a download attempt fails
        // with `ModelError::Http` rather than silently fetching something unverifiable.
        // Point this at the release asset once it exists.
        url: "https://github.com/audiobank/audiobank/releases/download/model-clap-audio-v1/clap_audio.onnx",
        sha256: "757910cd3aee90c95db4f6f3cca5252af6328c8c5296b306f105d5d6c2d7e1ae",
        bytes: Some(117_325_762),
    };

    /// Whether this release has a real hash behind it.
    ///
    /// A download that cannot be verified is not a download worth starting.
    pub fn is_pinned(&self) -> bool {
        self.sha256.len() == 64
            && self.sha256 != UNPINNED
            && self
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    /// On-disk name, version included so two revisions can sit side by side while one is
    /// mapped into a live session.
    pub fn filename(&self) -> String {
        format!("{}.onnx", self.version)
    }
}

/// Where a release lives under the app data dir, and where it lives while it is still
/// arriving.
///
/// The two paths differ by a suffix on purpose: `.partial` is in the same directory as its
/// destination, which is what makes the final [`std::fs::rename`] a same-filesystem atomic
/// operation rather than a copy that can be interrupted halfway.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    dir: PathBuf,
    installed: PathBuf,
    partial: PathBuf,
}

impl ModelPaths {
    /// `<data_dir>/models/<version>.onnx`, with its `.partial` sibling.
    pub fn new(data_dir: impl AsRef<Path>, release: &ModelRelease) -> Self {
        let dir = data_dir.as_ref().join("models");
        let installed = dir.join(release.filename());
        let partial = dir.join(format!("{}.partial", release.filename()));
        Self {
            dir,
            installed,
            partial,
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The verified model, present only if a download completed and its hash matched.
    pub fn installed(&self) -> &Path {
        &self.installed
    }

    /// The in-flight download. Never loaded by [`session`]; that is the entire point of
    /// the rename.
    pub fn partial(&self) -> &Path {
        &self.partial
    }

    /// Whether a verified model is on disk.
    ///
    /// A file at [`Self::installed`] got there by surviving verification, so its existence
    /// is the check. Re-hashing 200 MB on every launch would cost roughly half a second of
    /// cold start (`overview.md` §7) to re-prove something already proved; the settings
    /// screen can ask for that explicitly via [`download::Downloader::verify_installed`].
    pub fn is_installed(&self) -> bool {
        self.installed.is_file()
    }
}

/// What the app can say about the model right now, without touching the network or the
/// graph.
///
/// Three states because they have three recoveries, and Phase 9 owes each one a rendered
/// screen: install it, download it, or ship a build with a pinned release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// A verified model is on disk. A session can be built from it.
    Installed,
    /// Nothing on disk yet, but the release is pinned and fetchable.
    Downloadable,
    /// This build has no digest to verify against; see [`UNPINNED`].
    Unpinned,
}

/// The managed handle: which release this build wants, where it goes, and the lazily-built
/// session over it.
///
/// Constructing one is a couple of path joins. That is the point -- it is `.manage()`d
/// during setup, and nothing it owns touches the disk or the network until something asks
/// it to (`overview.md` §7).
#[derive(Debug)]
pub struct Model {
    release: ModelRelease,
    paths: ModelPaths,
    session: session::LazySession,
}

impl Model {
    /// The handle for [`ModelRelease::CURRENT`] under `data_dir`.
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self::with_release(data_dir, ModelRelease::CURRENT)
    }

    pub fn with_release(data_dir: impl AsRef<Path>, release: ModelRelease) -> Self {
        let paths = ModelPaths::new(data_dir, &release);
        let session = session::LazySession::new(paths.installed());
        Self {
            release,
            paths,
            session,
        }
    }

    pub fn release(&self) -> &ModelRelease {
        &self.release
    }

    pub fn paths(&self) -> &ModelPaths {
        &self.paths
    }

    /// The lazily-built inference session. The first call to
    /// [`session::LazySession::get`] is what pays for session construction.
    pub fn session(&self) -> &session::LazySession {
        &self.session
    }

    pub fn status(&self) -> ModelStatus {
        if self.paths.is_installed() {
            ModelStatus::Installed
        } else if self.release.is_pinned() {
            ModelStatus::Downloadable
        } else {
            ModelStatus::Unpinned
        }
    }

    /// A downloader for this release. Cheap; build one per download rather than holding a
    /// `reqwest::Client` open for the life of the app to make one request on first run.
    pub fn downloader(
        &self,
        data_dir: impl AsRef<Path>,
    ) -> Result<download::Downloader, download::ModelError> {
        download::Downloader::new(data_dir, self.release.clone())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_release_is_either_pinned_or_obviously_not() {
        // Not an assertion that it *is* pinned -- until the export runs it is not, and that
        // is a legitimate state. This asserts the two states are distinguishable, so a
        // half-edited constant (a real-looking hash of the wrong length, say) cannot pass
        // for either.
        let current = ModelRelease::CURRENT;
        assert!(
            current.sha256 == UNPINNED || current.is_pinned(),
            "MODEL sha256 is neither the UNPINNED sentinel nor a valid 64-char lowercase \
             hex digest: {}",
            current.sha256
        );
        assert!(
            current.url.starts_with("https://"),
            "model URL must be https"
        );
    }

    /// Exercises the predicate, not the shipped constant's current value. An earlier
    /// version of this test started from `ModelRelease::CURRENT` and asserted it was
    /// unpinned, so it passed for as long as the export had not been run and failed the
    /// moment it was -- testing the state of the world rather than the logic.
    #[test]
    fn pinning_rejects_the_sentinel_and_malformed_digests() {
        let mut release = ModelRelease {
            sha256: UNPINNED,
            ..ModelRelease::CURRENT
        };
        assert!(!release.is_pinned(), "the sentinel must not pass");

        release.sha256 = "ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert!(!release.is_pinned(), "uppercase hex must not pass");

        release.sha256 = "abc";
        assert!(!release.is_pinned(), "short digest must not pass");

        release.sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert!(release.is_pinned());
    }

    /// The three states must be distinguishable without touching the network, because the
    /// first-run screen renders one of them before anything has been fetched.
    #[test]
    fn status_distinguishes_unpinned_from_merely_not_downloaded() {
        let dir = tempfile::tempdir().unwrap();
        let unpinned = Model::with_release(
            dir.path(),
            ModelRelease {
                sha256: UNPINNED,
                ..ModelRelease::CURRENT
            },
        );
        assert_eq!(unpinned.status(), ModelStatus::Unpinned);

        let pinned = ModelRelease {
            sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            ..ModelRelease::CURRENT
        };
        let model = Model::with_release(dir.path(), pinned);
        assert_eq!(model.status(), ModelStatus::Downloadable);

        std::fs::create_dir_all(model.paths().dir()).unwrap();
        std::fs::write(model.paths().installed(), b"not really a model").unwrap();
        assert_eq!(model.status(), ModelStatus::Installed);
    }

    /// `.manage()` happens in `setup`, on the main thread. Whatever this does, it must not
    /// be able to block it.
    #[test]
    fn constructing_the_handle_builds_no_session_and_touches_no_disk() {
        let dir = tempfile::tempdir().unwrap();
        let model = Model::new(dir.path());
        assert!(!model.session().is_initialized());
        assert!(!model.paths().dir().exists(), "setup created directories");
    }

    #[test]
    fn partial_is_a_sibling_of_the_installed_file() {
        let paths = ModelPaths::new("/tmp/appdata", &ModelRelease::CURRENT);
        assert_eq!(paths.installed().parent(), paths.partial().parent());
        assert_eq!(paths.installed().parent(), Some(paths.dir()));
        assert_ne!(paths.installed(), paths.partial());
    }
}
