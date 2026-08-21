//! Resumable, verified model download.
//!
//! Streams with `Range` so a kill -9 mid-download resumes rather than restarts, hashes
//! incrementally over the stream, and installs by `fsync` + atomic rename from `.partial`
//! only after the hash verifies. A half-written model that passes for complete is the
//! failure mode this ordering exists to make impossible.
//!
//! **The ordering, and why each step is where it is:**
//!
//! 1. Bytes land in `<version>.onnx.partial`, never at the destination name. Nothing else
//!    in the app will ever open that path, so an interrupted download is inert rather than
//!    dangerous.
//! 2. SHA-256 runs over the stream as it arrives, not over the file afterwards -- a second
//!    200 MB read to verify what was just written is pure latency.
//! 3. On resume the partial's existing bytes are re-hashed from disk first. A `Sha256`
//!    state cannot be serialized across a process restart, so this is the cost of resuming
//!    at all: one sequential read of what is already on disk, against re-downloading it.
//! 4. Verification happens **before** the rename. A mismatch deletes the partial, because
//!    a partial whose hash is wrong will still be wrong after the next resume and keeping
//!    it would wedge the app in a loop of resuming into the same failure.
//! 5. `fsync` the file, rename, then `fsync` the directory. Without the directory sync the
//!    rename can be lost by a crash even though the data was durable.
//!
//! No network, hash mismatch, disk full, and interrupted-but-resumable are four distinct
//! outcomes with four distinct [`ModelError`] variants and four distinct rendered states.

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use super::{ModelPaths, ModelRelease};
use crate::pipeline::CancellationToken;

/// Minimum gap between progress callbacks (cross-cutting rule 3: coalesce to <= 10 Hz).
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// Read size when re-hashing an existing partial on resume.
const REHASH_CHUNK: usize = 1 << 20;

/// How long to wait for the server to say anything at all.
///
/// Separate from any total timeout, of which there is none: a 200 MB download over a slow
/// connection is legitimately long, and killing it at an arbitrary deadline would turn a
/// working download into a failure. What is not legitimate is a socket that has gone quiet.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// `ENOSPC`. `ErrorKind::StorageFull` is the mapping this should arrive as, but the
/// mapping is not guaranteed for every syscall on every platform and "disk full" is a
/// state with its own UI, so the raw code is checked too.
const ENOSPC: i32 = 28;

/// Everything that can go wrong between "the user pressed download" and "a verified model
/// is on disk".
///
/// Kept local to the module rather than folded into [`crate::error::AppError`], which is
/// Phase 6 -- the same split `DbError` makes. What matters now is that the four outcomes
/// `overview.md` §3.5 calls out are four *distinguishable values*, because each one has a
/// different recovery: retry later, re-download from scratch, free space, resume.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The build has no real hash to verify against, so there is nothing safe to fetch.
    /// See [`super::UNPINNED`].
    #[error(
        "model release {version} has no published SHA-256 in this build; \
         run scripts/export_clap_onnx.py and pin its digest in ModelRelease::CURRENT"
    )]
    ReleaseNotPinned { version: String },

    /// Could not reach the server. Retry when there is a network; nothing is wrong on
    /// disk, and whatever was already downloaded is still resumable.
    #[error("cannot reach {url}: {source}")]
    Offline {
        url: String,
        #[source]
        source: Box<reqwest::Error>,
    },

    /// The server answered, and the answer was not a model. A 404 here means the release
    /// asset moved, which no amount of retrying fixes.
    #[error("{url} returned HTTP {status}")]
    Http { status: u16, url: String },

    /// The bytes are not the model. Unrecoverable by resuming -- the partial is discarded.
    #[error("model failed verification: expected SHA-256 {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    /// Out of space. The partial is kept: freeing space and resuming is the recovery, and
    /// throwing away 150 MB the user already paid for would be a strange way to help.
    #[error("no space left on device while writing {path}")]
    DiskFull { path: PathBuf },

    /// The transfer died mid-stream. The partial is kept and the next call resumes from
    /// where this one stopped.
    #[error("transfer interrupted after {downloaded} bytes: {source}")]
    Interrupted {
        downloaded: u64,
        total: Option<u64>,
        #[source]
        source: Box<reqwest::Error>,
    },

    /// The user cancelled. Not a failure; the partial is kept and resumes.
    #[error("download cancelled after {downloaded} bytes")]
    Cancelled { downloaded: u64 },

    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
}

impl ModelError {
    /// Whether calling [`Downloader::ensure`] again could reasonably succeed without the
    /// user doing anything but waiting.
    ///
    /// Drives whether the first-run screen offers "Retry" or something more specific.
    pub fn is_resumable(&self) -> bool {
        matches!(
            self,
            ModelError::Offline { .. }
                | ModelError::Interrupted { .. }
                | ModelError::Cancelled { .. }
                | ModelError::DiskFull { .. }
        )
    }

    fn io(context: impl Into<String>, source: std::io::Error, path: &Path) -> Self {
        if source.kind() == ErrorKind::StorageFull || source.raw_os_error() == Some(ENOSPC) {
            return ModelError::DiskFull {
                path: path.to_path_buf(),
            };
        }
        ModelError::Io {
            context: context.into(),
            source,
        }
    }
}

/// How far along a download is.
///
/// `total` is `None` when the server declines to say -- a chunked response, or a release
/// whose size was never recorded. The UI needs to render that case as a live byte count
/// rather than a progress bar stuck at zero, which is why it is an `Option` and not a
/// hopeful guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
}

impl DownloadProgress {
    /// Completion in `[0, 1]`, or `None` when the total is unknown.
    pub fn fraction(&self) -> Option<f32> {
        self.total
            .filter(|&t| t > 0)
            .map(|t| (self.downloaded as f64 / t as f64).min(1.0) as f32)
    }
}

/// Fetches and verifies one [`ModelRelease`] into one [`ModelPaths`].
#[derive(Debug)]
pub struct Downloader {
    client: reqwest::Client,
    release: ModelRelease,
    paths: ModelPaths,
}

impl Downloader {
    /// Builds a downloader for `release` under `data_dir`.
    pub fn new(data_dir: impl AsRef<Path>, release: ModelRelease) -> Result<Self, ModelError> {
        let paths = ModelPaths::new(data_dir, &release);
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            // A 200 MB body over a slow link is not a stalled request, so there is no
            // total timeout. `read_timeout` catches the case that actually matters: a
            // connection that stops delivering and never closes.
            .read_timeout(Duration::from_secs(60))
            .user_agent(concat!("audiobank/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|source| ModelError::Offline {
                url: release.url.to_string(),
                source: Box::new(source),
            })?;

        Ok(Self {
            client,
            release,
            paths,
        })
    }

    pub fn paths(&self) -> &ModelPaths {
        &self.paths
    }

    /// Ensures a verified model is on disk, returning its path.
    ///
    /// Idempotent and cheap when the model is already installed: it returns without
    /// touching the network. Otherwise it resumes or starts the download, reporting
    /// progress through `on_progress` at no more than 10 Hz plus one guaranteed final
    /// event, and checking `cancel` between chunks.
    pub async fn ensure(
        &self,
        cancel: &CancellationToken,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<PathBuf, ModelError> {
        if self.paths.is_installed() {
            return Ok(self.paths.installed().to_path_buf());
        }
        if !self.release.is_pinned() {
            return Err(ModelError::ReleaseNotPinned {
                version: self.release.version.to_string(),
            });
        }

        std::fs::create_dir_all(self.paths.dir())
            .map_err(|e| ModelError::io("creating the model directory", e, self.paths.dir()))?;

        let (mut hasher, mut downloaded) = self.resume_state().await?;
        let mut response = self.request(downloaded).await?;

        // A server that ignores `Range` answers 200 with the whole file. Honour that
        // rather than appending its bytes to the ones already on disk, which would produce
        // a longer file that fails verification for a reason nobody could diagnose.
        if downloaded > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            tracing::info!(
                downloaded,
                status = response.status().as_u16(),
                "server did not honour Range; restarting the download"
            );
            hasher = Sha256::new();
            downloaded = 0;
            response = self.request(0).await?;
        }

        if !response.status().is_success() {
            return Err(ModelError::Http {
                status: response.status().as_u16(),
                url: self.release.url.to_string(),
            });
        }

        // `content_length` is what is *left* to transfer on a 206, so the total is that
        // plus what is already here. The release's own recorded size wins when present,
        // since it is the only figure that was ever verified.
        let total = self
            .release
            .bytes
            .or_else(|| response.content_length().map(|len| len + downloaded));

        // Append when resuming; truncate when starting over. Opening for append after a
        // restart is what turns "the server ignored Range" into a file that is the right
        // bytes twice over and fails verification for an undiagnosable reason.
        let mut options = tokio::fs::OpenOptions::new();
        options.create(true);
        if downloaded > 0 {
            options.append(true);
        } else {
            options.write(true).truncate(true);
        }
        let mut file = options
            .open(self.paths.partial())
            .await
            .map_err(|e| ModelError::io("opening the partial download", e, self.paths.partial()))?;

        on_progress(DownloadProgress { downloaded, total });
        let outcome = self
            .pump(
                response,
                &mut file,
                &mut hasher,
                &mut downloaded,
                total,
                cancel,
                &mut on_progress,
            )
            .await;

        // `tokio::fs::File` queues its writes, so the file is only as long as the byte
        // count above once it has been flushed -- and on every error path that byte count
        // is exactly what the next resume will ask the server to continue from. Flushing
        // only on success would make every interruption re-download the tail it already
        // had.
        let flushed = file
            .flush()
            .await
            .map_err(|e| ModelError::io("flushing the partial download", e, self.paths.partial()));
        outcome?;
        flushed?;
        on_progress(DownloadProgress { downloaded, total });

        let digest = hex(&hasher.finalize());
        if digest != self.release.sha256 {
            // Poisoned: resuming would append to bytes that are already wrong. Deleting is
            // what turns this from a permanent wedge into one wasted download.
            let _ = std::fs::remove_file(self.paths.partial());
            return Err(ModelError::HashMismatch {
                expected: self.release.sha256.to_string(),
                actual: digest,
            });
        }

        self.install(file).await?;
        tracing::info!(
            version = self.release.version,
            bytes = downloaded,
            "model verified and installed"
        );
        Ok(self.paths.installed().to_path_buf())
    }

    /// Drains the response body into `file`, hashing as it goes.
    ///
    /// Split out from [`Self::ensure`] purely so that every way out of the loop -- a torn
    /// connection, a cancel, a write failure -- lands in one place that flushes before the
    /// error escapes. `downloaded` is `&mut` for the same reason: the caller needs the
    /// count that corresponds to what is on disk even when this returns an error.
    #[allow(clippy::too_many_arguments)]
    async fn pump(
        &self,
        response: reqwest::Response,
        file: &mut tokio::fs::File,
        hasher: &mut Sha256,
        downloaded: &mut u64,
        total: Option<u64>,
        cancel: &CancellationToken,
        on_progress: &mut impl FnMut(DownloadProgress),
    ) -> Result<(), ModelError> {
        let mut stream = response.bytes_stream();
        let mut last_tick = Instant::now();

        while let Some(chunk) = stream.next().await {
            if cancel.is_cancelled() {
                return Err(ModelError::Cancelled {
                    downloaded: *downloaded,
                });
            }

            let chunk = chunk.map_err(|source| ModelError::Interrupted {
                downloaded: *downloaded,
                total,
                source: Box::new(source),
            })?;

            file.write_all(&chunk).await.map_err(|e| {
                ModelError::io("writing the partial download", e, self.paths.partial())
            })?;
            hasher.update(&chunk);
            *downloaded += chunk.len() as u64;

            if last_tick.elapsed() >= PROGRESS_INTERVAL {
                last_tick = Instant::now();
                on_progress(DownloadProgress {
                    downloaded: *downloaded,
                    total,
                });
            }
        }
        Ok(())
    }

    /// Re-hashes whatever is already in the partial file, so the stream can be appended to
    /// an accurate digest.
    ///
    /// Returns a fresh hasher and zero when there is nothing to resume.
    async fn resume_state(&self) -> Result<(Sha256, u64), ModelError> {
        let mut hasher = Sha256::new();
        let mut file = match tokio::fs::File::open(self.paths.partial()).await {
            Ok(file) => file,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok((hasher, 0)),
            Err(e) => {
                return Err(ModelError::io(
                    "opening the partial download",
                    e,
                    self.paths.partial(),
                ))
            }
        };

        let mut downloaded = 0u64;
        let mut buf = vec![0u8; REHASH_CHUNK];
        loop {
            let read = tokio::io::AsyncReadExt::read(&mut file, &mut buf)
                .await
                .map_err(|e| {
                    ModelError::io("re-hashing the partial download", e, self.paths.partial())
                })?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
            downloaded += read as u64;
        }

        if downloaded > 0 {
            tracing::info!(downloaded, "resuming model download");
        }
        Ok((hasher, downloaded))
    }

    /// Issues the GET, with a `Range` header when there is something to resume from.
    async fn request(&self, from: u64) -> Result<reqwest::Response, ModelError> {
        let mut request = self.client.get(self.release.url);
        if from > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={from}-"));
        }
        request.send().await.map_err(|source| {
            // A request that got as far as a response and then died mid-body is
            // `Interrupted`; one that never connected is `Offline`. They are the same
            // `reqwest::Error` type and very different states to be in.
            if source.is_connect() || source.is_timeout() || source.is_request() {
                ModelError::Offline {
                    url: self.release.url.to_string(),
                    source: Box::new(source),
                }
            } else {
                ModelError::Interrupted {
                    downloaded: from,
                    total: self.release.bytes,
                    source: Box::new(source),
                }
            }
        })
    }

    /// Durably moves the verified partial into place.
    async fn install(&self, mut file: tokio::fs::File) -> Result<(), ModelError> {
        file.flush()
            .await
            .map_err(|e| ModelError::io("flushing the model", e, self.paths.partial()))?;
        file.sync_all()
            .await
            .map_err(|e| ModelError::io("fsyncing the model", e, self.paths.partial()))?;
        drop(file);

        std::fs::rename(self.paths.partial(), self.paths.installed())
            .map_err(|e| ModelError::io("installing the model", e, self.paths.installed()))?;

        // The rename itself is metadata, and metadata is not durable until the *directory*
        // is synced. Skipping this is how a verified model disappears across a power loss.
        if let Ok(dir) = std::fs::File::open(self.paths.dir()) {
            let _ = dir.sync_all();
        }
        Ok(())
    }

    /// Re-hashes the installed model and checks it against the release.
    ///
    /// Not on any startup path -- see [`ModelPaths::is_installed`]. This is what the
    /// settings screen's "verify model" action calls, and what to reach for when a session
    /// fails to load a file that is supposedly fine.
    pub fn verify_installed(&self) -> Result<(), ModelError> {
        use std::io::Read;

        let mut file = std::fs::File::open(self.paths.installed())
            .map_err(|e| ModelError::io("opening the model", e, self.paths.installed()))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; REHASH_CHUNK];
        loop {
            let read = file
                .read(&mut buf)
                .map_err(|e| ModelError::io("reading the model", e, self.paths.installed()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
        }

        let digest = hex(&hasher.finalize());
        if digest != self.release.sha256 {
            return Err(ModelError::HashMismatch {
                expected: self.release.sha256.to_string(),
                actual: digest,
            });
        }
        Ok(())
    }
}

/// Lowercase hex. Small enough that a dependency for it would cost more to audit than to
/// write.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        // Writing to a `String` cannot fail.
        let _ = write!(out, "{b:02x}");
        out
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::{
        io::{BufRead, BufReader, Write},
        net::{TcpListener, TcpStream},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use super::*;

    /// A one-connection-at-a-time HTTP/1.1 server that can be told to misbehave.
    ///
    /// Hand-rolled rather than pulled in: what the tests need is a server that *breaks* in
    /// specific ways -- truncates a body, ignores `Range`, returns 404 -- and a real HTTP
    /// server is built to do none of those. Fifty lines here beats a dependency plus the
    /// configuration to make it fail on command.
    struct TestServer {
        port: u16,
        /// Every `Range` header value seen, in order. `None` for a request without one.
        ranges: Arc<Mutex<Vec<Option<String>>>>,
        requests: Arc<AtomicUsize>,
    }

    #[derive(Clone, Copy)]
    struct Behaviour {
        supports_range: bool,
        /// Close the connection after this many body bytes, mid-transfer.
        cut_after: Option<usize>,
        status: u16,
    }

    impl Default for Behaviour {
        fn default() -> Self {
            Self {
                supports_range: true,
                cut_after: None,
                status: 200,
            }
        }
    }

    impl TestServer {
        fn spawn(body: Vec<u8>, behaviour: Behaviour) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let ranges = Arc::new(Mutex::new(Vec::new()));
            let requests = Arc::new(AtomicUsize::new(0));

            let seen = Arc::clone(&ranges);
            let count = Arc::clone(&requests);
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    count.fetch_add(1, Ordering::SeqCst);
                    Self::serve(stream, &body, behaviour, &seen);
                }
            });

            Self {
                port,
                ranges,
                requests,
            }
        }

        fn serve(
            mut stream: TcpStream,
            body: &[u8],
            behaviour: Behaviour,
            seen: &Mutex<Vec<Option<String>>>,
        ) {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut range = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("range:") {
                    range = Some(value.trim().to_string());
                }
            }
            seen.lock().unwrap().push(range.clone());

            if behaviour.status != 200 {
                let _ = write!(
                    stream,
                    "HTTP/1.1 {} Nope\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    behaviour.status
                );
                return;
            }

            let from = range
                .filter(|_| behaviour.supports_range)
                .and_then(|r| r.strip_prefix("bytes=").map(str::to_string))
                .and_then(|r| r.trim_end_matches('-').parse::<usize>().ok())
                .unwrap_or(0)
                .min(body.len());

            let slice = &body[from..];
            let (code, reason) = if from > 0 {
                (206, "Partial Content")
            } else {
                (200, "OK")
            };
            let _ = write!(
                stream,
                "HTTP/1.1 {code} {reason}\r\nContent-Length: {}\r\n\
                 Accept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                slice.len()
            );

            match behaviour.cut_after {
                // Write a prefix and drop the socket without the rest. `reqwest` sees a
                // body shorter than `Content-Length` and errors, which is exactly the
                // mid-transfer failure a real network produces.
                Some(n) => {
                    let _ = stream.write_all(&slice[..n.min(slice.len())]);
                    let _ = stream.flush();
                }
                None => {
                    let _ = stream.write_all(slice);
                    let _ = stream.flush();
                }
            }
        }

        fn url(&self) -> String {
            format!("http://127.0.0.1:{}/clap_audio.onnx", self.port)
        }

        fn ranges(&self) -> Vec<Option<String>> {
            self.ranges.lock().unwrap().clone()
        }

        fn request_count(&self) -> usize {
            self.requests.load(Ordering::SeqCst)
        }
    }

    /// A deterministic stand-in for a 200 MB model.
    fn model_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn digest(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex(&hasher.finalize())
    }

    /// `ModelRelease` is `&'static str` all the way down, which a test cannot construct at
    /// runtime -- so it leaks. Bounded by the number of test cases, and the process is
    /// about to exit.
    fn release(url: String, sha256: String, bytes: Option<u64>) -> ModelRelease {
        ModelRelease {
            version: "clap-audio-test",
            url: Box::leak(url.into_boxed_str()),
            sha256: Box::leak(sha256.into_boxed_str()),
            bytes,
        }
    }

    #[tokio::test]
    async fn downloads_verifies_and_installs_atomically() {
        let body = model_bytes(300_000);
        let server = TestServer::spawn(body.clone(), Behaviour::default());
        let dir = tempfile::tempdir().unwrap();

        let downloader = Downloader::new(
            dir.path(),
            release(server.url(), digest(&body), Some(body.len() as u64)),
        )
        .unwrap();

        let mut ticks = Vec::new();
        let path = downloader
            .ensure(&CancellationToken::new(), |p| ticks.push(p))
            .await
            .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert!(!downloader.paths().partial().exists(), "partial survived");
        assert_eq!(
            ticks.last().unwrap(),
            &DownloadProgress {
                downloaded: body.len() as u64,
                total: Some(body.len() as u64),
            },
            "the terminal progress event is not the completed one"
        );
        downloader.verify_installed().unwrap();
    }

    /// An already-installed model must not cost a request. This is the check that keeps
    /// the model off the cold-start path (`overview.md` §7).
    #[tokio::test]
    async fn an_installed_model_short_circuits_before_the_network() {
        let body = model_bytes(4096);
        let server = TestServer::spawn(body.clone(), Behaviour::default());
        let dir = tempfile::tempdir().unwrap();
        let downloader =
            Downloader::new(dir.path(), release(server.url(), digest(&body), None)).unwrap();

        downloader
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap();
        assert_eq!(server.request_count(), 1);

        downloader
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap();
        assert_eq!(server.request_count(), 1, "a second ensure hit the network");
    }

    /// The headline requirement: survive a kill mid-download and resume. The first server
    /// truncates the body; the second serves the rest, and must be asked for exactly the
    /// right offset.
    #[tokio::test]
    async fn an_interrupted_download_resumes_from_where_it_stopped() {
        let body = model_bytes(200_000);
        let dir = tempfile::tempdir().unwrap();
        let sha = digest(&body);

        let broken = TestServer::spawn(
            body.clone(),
            Behaviour {
                cut_after: Some(50_000),
                ..Behaviour::default()
            },
        );
        let first = Downloader::new(
            dir.path(),
            release(broken.url(), sha.clone(), Some(body.len() as u64)),
        )
        .unwrap();
        let err = first
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap_err();
        assert!(
            matches!(err, ModelError::Interrupted { .. }),
            "expected Interrupted, got {err:?}"
        );
        assert!(err.is_resumable());

        let partial = std::fs::metadata(first.paths().partial()).unwrap().len();
        assert!(
            partial > 0 && partial < body.len() as u64,
            "{partial} bytes"
        );
        assert!(!first.paths().installed().exists(), "installed a torn file");

        let good = TestServer::spawn(body.clone(), Behaviour::default());
        let second = Downloader::new(
            dir.path(),
            release(good.url(), sha, Some(body.len() as u64)),
        )
        .unwrap();
        let path = second
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(
            good.ranges(),
            vec![Some(format!("bytes={partial}-"))],
            "resume did not ask for the right range"
        );
    }

    /// A server that answers a `Range` request with the whole file must not have its bytes
    /// appended to the partial. Restarting is correct; concatenating produces a file that
    /// fails verification for a reason nobody could diagnose from the error.
    #[tokio::test]
    async fn a_server_that_ignores_range_causes_a_clean_restart() {
        let body = model_bytes(120_000);
        let dir = tempfile::tempdir().unwrap();
        let sha = digest(&body);

        let broken = TestServer::spawn(
            body.clone(),
            Behaviour {
                cut_after: Some(30_000),
                ..Behaviour::default()
            },
        );
        let first = Downloader::new(dir.path(), release(broken.url(), sha.clone(), None)).unwrap();
        first
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap_err();
        assert!(std::fs::metadata(first.paths().partial()).unwrap().len() > 0);

        let stubborn = TestServer::spawn(
            body.clone(),
            Behaviour {
                supports_range: false,
                ..Behaviour::default()
            },
        );
        let second = Downloader::new(dir.path(), release(stubborn.url(), sha, None)).unwrap();
        let path = second
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), body, "restart concatenated");
        assert_eq!(
            stubborn.request_count(),
            2,
            "expected a re-request from zero"
        );
    }

    /// Wrong bytes must never reach the destination name, and the partial must not survive
    /// to be resumed into the same failure forever.
    #[tokio::test]
    async fn a_hash_mismatch_installs_nothing_and_discards_the_partial() {
        let body = model_bytes(64_000);
        let server = TestServer::spawn(body.clone(), Behaviour::default());
        let dir = tempfile::tempdir().unwrap();
        let downloader = Downloader::new(
            dir.path(),
            release(server.url(), digest(b"a different model"), None),
        )
        .unwrap();

        let err = downloader
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap_err();

        match err {
            ModelError::HashMismatch { ref actual, .. } => assert_eq!(actual, &digest(&body)),
            other => panic!("expected HashMismatch, got {other:?}"),
        }
        assert!(!downloader.paths().installed().exists());
        assert!(
            !downloader.paths().partial().exists(),
            "poisoned partial kept"
        );
        assert!(!err.is_resumable());
    }

    #[tokio::test]
    async fn a_missing_release_asset_is_an_http_error_not_a_retry_loop() {
        let server = TestServer::spawn(
            Vec::new(),
            Behaviour {
                status: 404,
                ..Behaviour::default()
            },
        );
        let dir = tempfile::tempdir().unwrap();
        let downloader =
            Downloader::new(dir.path(), release(server.url(), digest(b""), None)).unwrap();

        let err = downloader
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap_err();
        assert!(
            matches!(err, ModelError::Http { status: 404, .. }),
            "{err:?}"
        );
        assert!(!err.is_resumable(), "a 404 will not fix itself");
    }

    #[tokio::test]
    async fn no_server_at_all_is_offline_and_resumable() {
        let dir = tempfile::tempdir().unwrap();
        // Port 1 on loopback: nothing is listening and nothing can be.
        let downloader = Downloader::new(
            dir.path(),
            release("http://127.0.0.1:1/model.onnx".into(), digest(b""), None),
        )
        .unwrap();

        let err = downloader
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap_err();
        assert!(matches!(err, ModelError::Offline { .. }), "{err:?}");
        assert!(err.is_resumable());
    }

    /// Cancellation keeps what was downloaded. Cross-cutting rule 6: cancellable, not
    /// killable -- the difference is whether partial results survive.
    #[tokio::test]
    async fn cancellation_keeps_the_partial_for_the_next_attempt() {
        let body = model_bytes(2_000_000);
        let server = TestServer::spawn(body.clone(), Behaviour::default());
        let dir = tempfile::tempdir().unwrap();
        let downloader = Downloader::new(
            dir.path(),
            release(server.url(), digest(&body), Some(body.len() as u64)),
        )
        .unwrap();

        let cancel = CancellationToken::new();
        let err = downloader
            .ensure(&cancel, |_| cancel.cancel())
            .await
            .unwrap_err();

        assert!(matches!(err, ModelError::Cancelled { .. }), "{err:?}");
        assert!(err.is_resumable());
        assert!(!downloader.paths().installed().exists());
    }

    /// An unpinned build must refuse before it opens a socket. Fetching 200 MB that cannot
    /// be verified is worse than not fetching it.
    #[tokio::test]
    async fn an_unpinned_release_refuses_to_download() {
        let server = TestServer::spawn(model_bytes(1024), Behaviour::default());
        let dir = tempfile::tempdir().unwrap();
        let downloader = Downloader::new(
            dir.path(),
            release(server.url(), super::super::UNPINNED.to_string(), None),
        )
        .unwrap();

        let err = downloader
            .ensure(&CancellationToken::new(), |_| {})
            .await
            .unwrap_err();
        assert!(
            matches!(err, ModelError::ReleaseNotPinned { .. }),
            "{err:?}"
        );
        assert_eq!(server.request_count(), 0, "it opened a socket anyway");
    }

    #[test]
    fn hex_matches_the_known_sha256_of_the_empty_string() {
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn progress_fraction_is_none_without_a_total() {
        let unknown = DownloadProgress {
            downloaded: 10,
            total: None,
        };
        assert_eq!(unknown.fraction(), None);

        let known = DownloadProgress {
            downloaded: 50,
            total: Some(200),
        };
        assert_eq!(known.fraction(), Some(0.25));
    }
}
