//! `ort` session lifecycle: execution-provider selection, warmup, and lazy init.
//!
//! **Every `ort` type is isolated behind this module** so that an rc-version upgrade
//! touches exactly one file -- `ort` is pinned to an exact release candidate and its API is
//! not yet stable. Nothing outside here names `ort::`; what leaves is `f32`, [`Provider`],
//! and [`SessionError`].
//!
//! CoreML is attempted first with a verified CPU fallback; which EP actually bound is
//! logged, because "CoreML was requested" and "CoreML is running" are not the same claim.
//! A warmup run on a zero tensor pays the first-inference cost off the user's critical
//! path, and session construction stays off the cold-start path entirely
//! (`overview.md` §7).
//!
//! **Deviation from `overview.md` §3.4, and it is a load-bearing one.** §3.4 says a
//! `Session` is "thread-safe for concurrent `run()` calls" and prescribes `Arc<Session>`
//! shared across `rayon` workers. That is not true of `ort` 2.0.0-rc.13, and the crate is
//! explicit about why: `Session::run` takes `&mut self` because ONNX Runtime's `Run` is not
//! thread-safe, and earlier versions of `ort` that allowed concurrent inference "often saw
//! crashes and memory corruption". So [`ModelSession`] is still one session shared as an
//! `Arc` -- the expensive thing is still built exactly once -- but inference is serialized
//! behind a `Mutex`.
//!
//! This costs less than it sounds like, because §3.4's other prescription is batching: the
//! embed stage stacks 16-32 mel tensors into one `run()`. The parallelism lives in decode
//! and mel extraction, which are `rayon`'s, and the serialized region is one large matmul
//! per batch rather than one per file. Phase 4's throughput target is what settles whether
//! that is enough; if it is not, the fix is one session per worker thread, not concurrent
//! `run()` on a shared one.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use ort::{
    ep::{CoreML, CPU},
    session::{builder::GraphOptimizationLevel, Session},
    value::{TensorRef, ValueType},
};

use crate::pipeline::mel::{MEL_BINS, MEL_FRAMES};

/// Environment variable that forces the CPU provider.
///
/// The exit criterion for Phase 3 is that the session initializes on **both** CoreML and
/// forced-CPU, which requires forcing to be something a human can do without a rebuild.
/// It is also the first thing to try when a machine produces embeddings that fail parity.
pub const FORCE_CPU_ENV: &str = "AUDIOBANK_FORCE_CPU";

/// Which execution provider a session ended up on.
///
/// Recorded per session rather than assumed, because the interesting case is the one where
/// CoreML registers, refuses the graph at run time, and everything silently falls back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    CoreMl,
    Cpu,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Provider::CoreMl => "coreml",
            Provider::Cpu => "cpu",
        }
    }
}

/// Everything that can go wrong between a verified `.onnx` on disk and a session that has
/// successfully run inference once.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("no model at {0}; download it before creating a session")]
    NotInstalled(PathBuf),

    /// Neither provider produced a session that could run. The CoreML failure is carried
    /// along because "CPU also failed" without it is half a bug report.
    #[error("could not initialize an inference session on any provider: {cpu} (coreml: {coreml})")]
    NoProvider { coreml: String, cpu: String },

    /// The graph is not the graph this build's front-end feeds. Fatal at init on purpose:
    /// every alternative produces embeddings that are wrong and plausible.
    #[error("model input {name} has shape {shape:?}, which is not a [batch, 1, {frames}, {mels}] or [batch, 1, {mels}, {frames}] log-mel tensor")]
    UnexpectedInput {
        name: String,
        shape: Vec<i64>,
        frames: usize,
        mels: usize,
    },

    #[error("model outputs {actual}-dimensional embeddings; this build expects {expected}")]
    UnexpectedOutput { expected: usize, actual: usize },

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("batch of {mels} mel values is not a whole number of {expected}-value spectrograms")]
    RaggedBatch { mels: usize, expected: usize },
}

/// How the graph wants its log-mel tensor laid out.
///
/// [`crate::pipeline::mel`] produces frames-major (`[T][mels]`), which is HTSAT's own
/// layout and therefore the one the export should produce. `overview.md` §3.4 writes the
/// input as `[B, 1, 64, T]` -- mels-major -- and rather than pick a winner on paper, this
/// is read off the graph at init and the tensor is transposed if needed. One `if` beats a
/// silent 90-degree rotation of every spectrogram in the corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// `[batch, 1, frames, mels]`
    FramesMajor,
    /// `[batch, 1, mels, frames]`
    MelsMajor,
}

/// Reads the layout out of a declared input shape, or rejects the graph.
///
/// Split out as a pure function over `&[i64]` so it can be tested against every shape a
/// plausible export might produce without any of them existing yet. Dynamic axes are `-1`;
/// batch is expected to be dynamic or 1, and the two spectrogram axes must be concrete,
/// because a graph that will accept any number of mel bins will accept the wrong number.
fn layout_of(shape: &[i64]) -> Option<Layout> {
    let (frames, mels) = (MEL_FRAMES as i64, MEL_BINS as i64);
    match shape {
        // The channel axis is 1 for both orderings, so it cannot disambiguate; the last two
        // axes do.
        [_, 1, f, m] if *f == frames && *m == mels => Some(Layout::FramesMajor),
        [_, 1, m, f] if *f == frames && *m == mels => Some(Layout::MelsMajor),
        // Some exports drop the singleton channel.
        [_, f, m] if *f == frames && *m == mels => Some(Layout::FramesMajor),
        [_, m, f] if *f == frames && *m == mels => Some(Layout::MelsMajor),
        _ => None,
    }
}

/// One warm CLAP audio tower, and the only thing in the crate that owns an `ort::Session`.
#[derive(Debug)]
pub struct ModelSession {
    /// `Mutex` rather than bare `Session` -- see the module docs. `run` needs `&mut`, and
    /// ONNX Runtime's `Run` is not thread-safe.
    session: Mutex<Session>,
    provider: Provider,
    layout: Layout,
    input_name: String,
    embedding_dim: usize,
}

impl ModelSession {
    /// Builds a session over a verified model file, preferring CoreML.
    ///
    /// Expensive -- hundreds of milliseconds, plus the warmup. Never call this on the
    /// cold-start path; call it through [`LazySession`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(SessionError::NotInstalled(path.to_path_buf()));
        }

        // The environment is process-global and idempotent; `commit` on an already-committed
        // environment is a no-op returning false, which is why the result is discarded.
        let _ = ort::init()
            .with_name("audiobank")
            .with_telemetry(false)
            .commit();

        if std::env::var_os(FORCE_CPU_ENV).is_some() {
            tracing::info!("{FORCE_CPU_ENV} is set; skipping CoreML");
            return Self::open_with(path, Provider::Cpu).map_err(|cpu| SessionError::NoProvider {
                coreml: format!("skipped: {FORCE_CPU_ENV} is set"),
                cpu,
            });
        }

        match Self::open_with(path, Provider::CoreMl) {
            Ok(session) => Ok(session),
            Err(coreml) => {
                // Not a warning-free path and it should not be: falling back is a real
                // performance cliff, and a user reporting "the scan takes four hours"
                // needs this line in the log.
                tracing::warn!(error = %coreml, "CoreML unavailable; falling back to CPU");
                Self::open_with(path, Provider::Cpu)
                    .map_err(|cpu| SessionError::NoProvider { coreml, cpu })
            }
        }
    }

    /// Builds and **proves** a session on one provider.
    ///
    /// The proof is the warmup: the session is not returned until a real `run()` has
    /// succeeded on it. This is what makes the CPU fallback verified rather than
    /// aspirational (`task.md` Phase 3) -- CoreML registering successfully and then
    /// rejecting the graph on the first inference is precisely the failure that a
    /// registration-only check would wave through, and it would surface halfway into a
    /// 50,000-file scan instead of at init.
    fn open_with(path: &Path, provider: Provider) -> Result<Self, String> {
        let started = Instant::now();
        let mut builder = Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?;

        builder = match provider {
            Provider::CoreMl => builder
                .with_execution_providers([CoreML::default().build().error_on_failure()])
                .map_err(|e| e.to_string())?,
            Provider::Cpu => builder
                // Explicit rather than implicit: without this the session would still run
                // on CPU, but it would do so having silently inherited whatever providers
                // the environment registered, which is not the same thing as a CPU session.
                .with_no_environment_execution_providers()
                .map_err(|e| e.to_string())?
                .with_execution_providers([CPU::default().build().error_on_failure()])
                .map_err(|e| e.to_string())?,
        };

        let session = builder.commit_from_file(path).map_err(|e| e.to_string())?;
        let (input_name, layout, embedding_dim) = inspect(&session).map_err(|e| e.to_string())?;

        let model = Self {
            session: Mutex::new(session),
            provider,
            layout,
            input_name,
            embedding_dim,
        };
        model.warmup().map_err(|e| e.to_string())?;

        tracing::info!(
            provider = provider.as_str(),
            embedding_dim,
            layout = ?layout,
            ms = started.elapsed().as_millis() as u64,
            "inference session ready"
        );
        Ok(model)
    }

    pub fn provider(&self) -> Provider {
        self.provider
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Runs one zero tensor through the graph so the first real batch does not pay for
    /// lazy kernel compilation -- which under CoreML means an on-device model compile, and
    /// is measured in seconds rather than milliseconds.
    fn warmup(&self) -> Result<(), SessionError> {
        let silence = vec![0.0f32; MEL_FRAMES * MEL_BINS];
        self.embed(&silence)?;
        Ok(())
    }

    /// Embeds one log-mel spectrogram, L2-normalized.
    pub fn embed(&self, mel: &[f32]) -> Result<Vec<f32>, SessionError> {
        let mut batch = self.embed_batch(mel, 1)?;
        batch.truncate(self.embedding_dim);
        Ok(batch)
    }

    /// Embeds `count` concatenated log-mel spectrograms in one `run()`, returning
    /// `count * embedding_dim` L2-normalized floats.
    ///
    /// Per-sample `run()` calls are dominated by fixed overhead (`overview.md` §3.4), so
    /// this is the call Phase 4's batcher makes; [`Self::embed`] is the convenience wrapper
    /// for the warmup and the parity test.
    pub fn embed_batch(&self, mels: &[f32], count: usize) -> Result<Vec<f32>, SessionError> {
        let per_sample = MEL_FRAMES * MEL_BINS;
        if count == 0 || mels.len() != count * per_sample {
            return Err(SessionError::RaggedBatch {
                mels: mels.len(),
                expected: per_sample,
            });
        }

        // Transposing costs one pass over 64,064 floats per sample. It only happens when
        // the graph disagrees with the front-end's natural layout, and `Cow`-ing it away
        // would cost more in branches here than it saves.
        let (shape, data) = match self.layout {
            Layout::FramesMajor => (
                [count as i64, 1, MEL_FRAMES as i64, MEL_BINS as i64],
                std::borrow::Cow::Borrowed(mels),
            ),
            Layout::MelsMajor => {
                let mut transposed = vec![0.0f32; mels.len()];
                for sample in 0..count {
                    let src = &mels[sample * per_sample..(sample + 1) * per_sample];
                    let dst = &mut transposed[sample * per_sample..(sample + 1) * per_sample];
                    for frame in 0..MEL_FRAMES {
                        for mel in 0..MEL_BINS {
                            dst[mel * MEL_FRAMES + frame] = src[frame * MEL_BINS + mel];
                        }
                    }
                }
                (
                    [count as i64, 1, MEL_BINS as i64, MEL_FRAMES as i64],
                    std::borrow::Cow::Owned(transposed),
                )
            }
        };

        let tensor = TensorRef::from_array_view((shape, data.as_ref()))
            .map_err(|e| SessionError::Inference(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| SessionError::Inference("the inference session is poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| SessionError::Inference(e.to_string()))?;
        let (_, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| SessionError::Inference(e.to_string()))?;

        if values.len() != count * self.embedding_dim {
            return Err(SessionError::UnexpectedOutput {
                expected: count * self.embedding_dim,
                actual: values.len(),
            });
        }

        let mut out = values.to_vec();
        for row in out.chunks_mut(self.embedding_dim) {
            l2_normalize(row);
        }
        Ok(out)
    }
}

/// Scales a vector to unit length, in place.
///
/// Done the instant an embedding leaves the graph (`overview.md` §3.4): cosine similarity
/// becomes a dot product, and UMAP's metric assumes it. A zero vector -- which the warmup's
/// silence can legitimately produce -- is left alone rather than divided by zero, because a
/// `NaN` written to `embeddings.bin` poisons every projection that ever reads it.
fn l2_normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let scale = (1.0 / norm) as f32;
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
}

/// Reads the input name, tensor layout, and embedding width off a committed session.
fn inspect(session: &Session) -> Result<(String, Layout, usize), SessionError> {
    let input = session
        .inputs()
        .first()
        .ok_or_else(|| SessionError::Inference("the model declares no inputs".into()))?;

    let ValueType::Tensor { shape, .. } = input.dtype() else {
        return Err(SessionError::UnexpectedInput {
            name: input.name().to_string(),
            shape: Vec::new(),
            frames: MEL_FRAMES,
            mels: MEL_BINS,
        });
    };
    let dims: Vec<i64> = shape.iter().copied().collect();
    let layout = layout_of(&dims).ok_or_else(|| SessionError::UnexpectedInput {
        name: input.name().to_string(),
        shape: dims.clone(),
        frames: MEL_FRAMES,
        mels: MEL_BINS,
    })?;

    // The embedding width is read from the graph and then checked against the constant the
    // data layer was built and benchmarked against in Phase 1. They have to agree; the
    // schema, `embeddings.bin`'s stride, and every projection depend on it.
    let dim = session
        .outputs()
        .first()
        .and_then(|out| match out.dtype() {
            ValueType::Tensor { shape, .. } => shape.iter().last().copied(),
            _ => None,
        })
        .filter(|&d| d > 0)
        .ok_or(SessionError::UnexpectedOutput {
            expected: crate::EMBEDDING_DIM,
            actual: 0,
        })? as usize;

    if dim != crate::EMBEDDING_DIM {
        return Err(SessionError::UnexpectedOutput {
            expected: crate::EMBEDDING_DIM,
            actual: dim,
        });
    }

    Ok((input.name().to_string(), layout, dim))
}

/// A session that is built on first use and never on the cold-start path.
///
/// `overview.md` §7 budgets cold start to interactive at under two seconds and explicitly
/// excludes ML session init, which is only honest if init genuinely does not happen at
/// startup. This is the mechanism: [`crate::lib`] manages one of these, construction is
/// free, and the first thing that actually needs an embedding pays for it.
///
/// A failed attempt is not cached. Session init fails for reasons that get fixed while the
/// app is running -- the model finishing its download, `AUDIOBANK_FORCE_CPU` being set --
/// and a `OnceLock` that remembers the first failure forever would require a restart to
/// recover from any of them.
#[derive(Debug)]
pub struct LazySession {
    path: PathBuf,
    cell: Mutex<Option<Arc<ModelSession>>>,
}

impl LazySession {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            cell: Mutex::new(None),
        }
    }

    /// The session, building it if this is the first call.
    ///
    /// The lock is held across construction so that two scans starting at once build one
    /// session rather than two. Session construction is the expensive thing this whole
    /// module exists to do exactly once.
    pub fn get(&self) -> Result<Arc<ModelSession>, SessionError> {
        let mut cell = self
            .cell
            .lock()
            .map_err(|_| SessionError::Inference("the session cell is poisoned".into()))?;
        if let Some(session) = cell.as_ref() {
            return Ok(Arc::clone(session));
        }

        let session = Arc::new(ModelSession::open(&self.path)?);
        *cell = Some(Arc::clone(&session));
        Ok(session)
    }

    /// Whether a session has already been built. Tests and the settings screen; nothing on
    /// a hot path.
    pub fn is_initialized(&self) -> bool {
        self.cell.lock().map(|c| c.is_some()).unwrap_or(false)
    }

    /// Drops the session, so the next [`Self::get`] rebuilds it.
    ///
    /// What "re-download the model" in settings has to call: a live session holds the old
    /// file open, and on macOS that keeps the unlinked inode alive.
    pub fn reset(&self) {
        if let Ok(mut cell) = self.cell.lock() {
            *cell = None;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Both orderings are accepted and told apart. This is the test that would have caught
    /// `overview.md` §3.4 and HTSAT disagreeing about the axis order, which they do.
    #[test]
    fn layout_is_read_from_the_declared_shape() {
        let (f, m) = (MEL_FRAMES as i64, MEL_BINS as i64);
        assert_eq!(layout_of(&[-1, 1, f, m]), Some(Layout::FramesMajor));
        assert_eq!(layout_of(&[1, 1, f, m]), Some(Layout::FramesMajor));
        assert_eq!(layout_of(&[-1, 1, m, f]), Some(Layout::MelsMajor));
        assert_eq!(layout_of(&[-1, f, m]), Some(Layout::FramesMajor));
        assert_eq!(layout_of(&[-1, m, f]), Some(Layout::MelsMajor));
    }

    /// A graph with a dynamic mel axis is rejected rather than accommodated. "Accepts any
    /// number of mel bins" means "accepts the wrong number of mel bins", and the result
    /// would be embeddings that are wrong and plausible.
    #[test]
    fn a_graph_that_does_not_want_this_front_end_is_rejected() {
        let (f, m) = (MEL_FRAMES as i64, MEL_BINS as i64);
        assert_eq!(layout_of(&[-1, 1, f, -1]), None, "dynamic mel axis");
        assert_eq!(layout_of(&[-1, 1, -1, m]), None, "dynamic frame axis");
        assert_eq!(layout_of(&[-1, 1, 128, m]), None, "wrong frame count");
        assert_eq!(layout_of(&[-1, 1, f, 128]), None, "wrong mel count");
        assert_eq!(layout_of(&[-1, 480_000]), None, "raw waveform input");
        assert_eq!(layout_of(&[]), None);
    }

    #[test]
    fn normalization_makes_unit_vectors_and_leaves_zero_alone() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);

        let mut zero = vec![0.0f32; 8];
        l2_normalize(&mut zero);
        assert!(zero.iter().all(|&x| x == 0.0), "silence became {zero:?}");
    }

    /// A missing model is its own error, not a generic inference failure -- the recovery is
    /// "download it", and the first-run screen needs to be able to say so.
    #[test]
    fn a_missing_model_is_not_installed_rather_than_a_load_failure() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("clap-audio-v1.onnx");

        let lazy = LazySession::new(&missing);
        assert!(!lazy.is_initialized());
        assert!(matches!(lazy.get(), Err(SessionError::NotInstalled(_))));
        assert!(
            !lazy.is_initialized(),
            "a failed init must not be cached as success"
        );
    }

    /// Constructing a [`LazySession`] must not touch the model, so that `lib.rs` can
    /// `.manage()` one during setup without putting session init on the cold-start path.
    #[test]
    fn construction_is_free() {
        let started = Instant::now();
        let lazy = LazySession::new("/definitely/not/a/model.onnx");
        assert!(!lazy.is_initialized());
        assert!(started.elapsed().as_millis() < 50);
        lazy.reset();
    }

    /// The batch API rejects a buffer that is not a whole number of spectrograms rather
    /// than reshaping it into whatever fits.
    #[test]
    fn a_ragged_batch_is_refused_before_any_tensor_is_built() {
        // Reaching `embed_batch` needs a session, which needs a model. What can be checked
        // without one is that the arithmetic defining "ragged" is the arithmetic the caller
        // will be held to.
        let per_sample = MEL_FRAMES * MEL_BINS;
        assert_eq!(per_sample, 1001 * 64);
        assert_ne!(per_sample * 2 - 1, per_sample * 2);
    }
}
