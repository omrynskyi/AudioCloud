//! Embedding: the batching accumulator over one shared embedder.
//!
//! Batching keeps the spectrogram-to-vector stage bounded and lets the active embedder
//! process several samples at a time.
//!
//! Two things this module does *not* do, both deliberate:
//!
//! - It talks only to the [`Embed`] trait, keeping the batching logic independent of the
//!   concrete fingerprint implementation.
//! - **It does not decide what a batch means for the database.** It attaches vectors to
//!   rows and passes them on; `pipeline::persist_stage` owns `embeddings.bin` and the
//!   `samples` update.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};

use crate::{
    pipeline::{
        decode::{BufferPool, PooledBuffer},
        mel::{MEL_BINS, MEL_FRAMES},
    },
};

/// Values in one log-mel spectrogram: 1001 frames x 64 bands, 256 KB of f32.
pub const MEL_VALUES: usize = MEL_FRAMES * MEL_BINS;

/// Depth of the mel -> embed queue (`overview.md` §3).
///
/// 64 mel tensors is 16 MB in flight. This is the queue that bounds peak RSS during a
/// scan: inference is the slowest stage, so this is the one that is actually full, and
/// backpressure from it is what stops the decoders -- and through them the walker -- from
/// running ahead of the model.
pub const EMBED_QUEUE_DEPTH: usize = 64;

/// How a batch is filled and when it is given up on.
///
/// `size` is `overview.md` §3.4's 16-32, starting at the bottom of that range;
/// `benchmarks::embed_batch_size_sweep` is what moves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchConfig {
    /// Spectrograms per `run()`.
    pub size: usize,
    /// How long a partly-filled batch waits for company before running anyway.
    pub timeout: Duration,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            size: 16,
            timeout: Duration::from_millis(250),
        }
    }
}

impl BatchConfig {
    /// A batch size of `size`, with the default timeout. For the sweep and for tests.
    pub fn of_size(size: usize) -> Self {
        Self {
            size: size.max(1),
            ..Self::default()
        }
    }
}

/// A batch of mel spectrograms, and the vectors that come back.
///
/// The narrowest possible view of inference, for two reasons. It keeps `ort` behind
/// [`crate::model::session`] (cross-cutting rule 7), and it makes the accumulator testable
/// without a model: `tests` below drive the whole batching path through a fake that counts
/// how many spectrograms each `run` received, which is the property the batcher exists to
/// have and the one a real session would make almost impossible to observe.
pub trait Embed: Send + Sync {
    /// Embeds `count` concatenated [`MEL_VALUES`]-value spectrograms, returning
    /// `count * dim` L2-normalized floats.
    fn embed_batch(&self, mels: &[f32], count: usize) -> Result<Vec<f32>, EmbedError>;

    /// Width of one vector. Checked against [`crate::EMBEDDING_DIM`] at session init.
    fn embedding_dim(&self) -> usize;
}

/// What inference can fail at, from the pipeline's side of the trait.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {

    #[error("the model returned {actual} values for a batch of {count} x {dim}")]
    ShortBatch {
        count: usize,
        dim: usize,
        actual: usize,
    },
}

/// A pool of mel buffers, sized for the queue between the mel and embed stages.
///
/// 256 KB each (`overview.md` §3.6). Per-file allocation of these across a 50,000-file
/// scan is 12 GB of churn for a buffer whose size is a compile-time constant.
///
/// `max_held` covers the steady state rather than the high-water mark: buffers are taken
/// by `rayon` workers one at a time and released by the embed stage a whole batch at a
/// time, so the free list swings by a batch. Sized under that and every release past the
/// cap is a free followed immediately by a fresh 256 KB allocation, which is the churn the
/// pool exists to avoid.
pub fn mel_pool(batch: BatchConfig) -> Arc<BufferPool> {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    BufferPool::new(MEL_VALUES, cores * 2 + batch.size * 2)
}

/// One item on its way through the embed stage: a payload of the caller's choosing, and
/// optionally a spectrogram to embed.
///
/// Generic over the payload so this module never has to know what a `samples` row is. The
/// pipeline passes its `Analyzed`; the tests pass an integer.
#[derive(Debug)]
pub struct Pending<T> {
    pub payload: T,
    /// `None` for anything that does not need inference -- a quarantined file, a duplicate
    /// borrowing its twin's vector, or any row at all when no session is available.
    pub mel: Option<PooledBuffer>,
}

/// What came back for one item.
#[derive(Debug)]
pub struct Embedded<T> {
    pub payload: T,
    /// The L2-normalized vector, or `None` if this item never asked for one.
    ///
    /// Also `None` when inference *failed*: a batch that errors is logged once and its
    /// rows continue unembedded rather than taking the scan down with them. They stay
    /// `decoded`, which is exactly the state a later scan picks back up
    /// ([`crate::db::SampleStatus::is_complete`]), so the cost of a transient inference
    /// failure is a rescan and not a hole in the corpus.
    pub embedding: Option<Vec<f32>>,
}

/// Drives the batching accumulator over a channel, until the sender side is dropped.
///
/// The loop is a `recv_timeout` rather than a `recv`: without the timeout the tail of a
/// scan -- the last nine files of a library, say, with a batch size of sixteen -- would sit
/// in the accumulator until the walk stage happened to drop its sender, which on a slow
/// filesystem is seconds later. The deadline is measured from the first item of the batch,
/// so a steadily-fed stage never pays for it and a starved one gives up exactly once.
///
/// Cancellation is checked between batches. An in-flight `run()` is not interruptible --
/// ONNX Runtime offers no such thing -- so a cancelled scan finishes the batch it is
/// holding, which is bounded by [`BatchConfig::size`] and is the difference between
/// cancellable and killable (cross-cutting rule 6).
pub fn embed_stage<T, E: Embed + ?Sized>(
    model: &Arc<E>,
    config: BatchConfig,
    cancel: &crate::pipeline::CancellationToken,
    input: Receiver<Pending<T>>,
    out: &Sender<Embedded<T>>,
    embedded_counter: &std::sync::atomic::AtomicU64,
) -> (u64, u64) {
    let dim = model.embedding_dim();
    let mut payloads: Vec<T> = Vec::with_capacity(config.size);
    let mut mels: Vec<PooledBuffer> = Vec::with_capacity(config.size);
    let mut wants: Vec<bool> = Vec::with_capacity(config.size);
    let mut staging: Vec<f32> = Vec::with_capacity(config.size * MEL_VALUES);
    let (mut ran, mut runs) = (0u64, 0u64);

    // `None` until the batch has its first member: a deadline that starts ticking on an
    // empty batch would fire repeatedly through every idle stretch of the scan.
    let mut deadline: Option<Instant> = None;

    loop {
        let timeout = match deadline {
            Some(at) => at.saturating_duration_since(Instant::now()),
            None => Duration::from_millis(100),
        };

        match input.recv_timeout(timeout) {
            Ok(item) => {
                if deadline.is_none() {
                    deadline = Some(Instant::now() + config.timeout);
                }
                wants.push(item.mel.is_some());
                if let Some(mel) = item.mel {
                    mels.push(mel);
                }
                payloads.push(item.payload);

                if mels.len() >= config.size {
                    run_batch(
                        model,
                        dim,
                        &mut payloads,
                        &mut mels,
                        &mut wants,
                        &mut staging,
                        out,
                        embedded_counter,
                        &mut ran,
                        &mut runs,
                    );
                    deadline = None;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                // Either the batch aged out, or nothing has arrived at all and this is the
                // idle tick that lets cancellation be noticed.
                if !payloads.is_empty() {
                    run_batch(
                        model,
                        dim,
                        &mut payloads,
                        &mut mels,
                        &mut wants,
                        &mut staging,
                        out,
                        embedded_counter,
                        &mut ran,
                        &mut runs,
                    );
                    deadline = None;
                }
                if cancel.is_cancelled() && input.is_empty() {
                    // Nothing upstream is going to arrive that this stage would keep. The
                    // sender is still alive until the process stage unwinds, so waiting for
                    // it here would hold the whole teardown behind a full walk queue.
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    // Whatever the last partial batch holds is still owed to the persist stage.
    if !payloads.is_empty() {
        run_batch(
            model,
            dim,
            &mut payloads,
            &mut mels,
            &mut wants,
            &mut staging,
            out,
            embedded_counter,
            &mut ran,
            &mut runs,
        );
    }

    (ran, runs)
}

/// Stacks the batch into one tensor, runs it, and redistributes the rows.
///
/// An inference failure is logged and the whole batch continues unembedded -- see
/// [`Embedded::embedding`] on why that is a recoverable state rather than a scan-ending
/// one.
#[allow(clippy::too_many_arguments)]
fn run_batch<T, E: Embed + ?Sized>(
    model: &Arc<E>,
    dim: usize,
    payloads: &mut Vec<T>,
    mels: &mut Vec<PooledBuffer>,
    wants: &mut Vec<bool>,
    staging: &mut Vec<f32>,
    out: &Sender<Embedded<T>>,
    embedded_counter: &std::sync::atomic::AtomicU64,
    ran: &mut u64,
    runs: &mut u64,
) {
    let count = mels.len();
    let vectors = if count == 0 {
        Vec::new()
    } else {
        staging.clear();
        for mel in mels.iter() {
            staging.extend_from_slice(mel);
        }
        *runs += 1;
        *ran += count as u64;

        match model.embed_batch(staging, count) {
            Ok(values) if values.len() == count * dim => values,
            Ok(values) => {
                tracing::error!(
                    count,
                    dim,
                    actual = values.len(),
                    "inference returned the wrong number of values; batch left unembedded"
                );
                Vec::new()
            }
            Err(e) => {
                tracing::error!(error = %e, count, "inference failed; batch left unembedded");
                Vec::new()
            }
        }
    };

    // Release the mel buffers before the rows go into a queue 1024 deep. They are 256 KB
    // each and nothing downstream wants them; holding them to the end of the send loop
    // would put a batch's worth of spectrograms behind the persist queue's backpressure.
    mels.clear();

    let mut taken = 0usize;
    for (payload, wanted) in payloads.drain(..).zip(wants.drain(..)) {
        let embedding = if wanted {
            let slice = vectors
                .get(taken * dim..(taken + 1) * dim)
                .map(<[f32]>::to_vec);
            taken += 1;
            slice
        } else {
            None
        };
        if embedding.is_some() {
            embedded_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if out.send(Embedded { payload, embedding }).is_err() {
            // The persist stage is gone, which happens only during teardown.
            break;
        }
    }
    payloads.clear();
    wants.clear();
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::pipeline::CancellationToken;
    use crossbeam_channel::bounded;
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    };

    const DIM: usize = 8;

    /// A model that records the shape of every batch it is handed and returns a vector
    /// whose first component identifies the spectrogram it came from.
    ///
    /// The identifying value is what proves redistribution: a batcher that handed row 3's
    /// vector to row 5 would still produce plausible, normalized, correctly-counted output.
    #[derive(Debug, Default)]
    struct CountingModel {
        batches: Mutex<Vec<usize>>,
        fail: bool,
    }

    impl Embed for CountingModel {
        fn embed_batch(&self, mels: &[f32], count: usize) -> Result<Vec<f32>, EmbedError> {
            assert_eq!(
                mels.len(),
                count * MEL_VALUES,
                "ragged batch reached the model"
            );
            self.batches.lock().unwrap().push(count);
            if self.fail {
                return Err(EmbedError::ShortBatch {
                    count,
                    dim: DIM,
                    actual: 0,
                });
            }
            let mut out = vec![0.0f32; count * DIM];
            for (i, row) in out.chunks_mut(DIM).enumerate() {
                // The tag is the first mel value of the i-th spectrogram, so a
                // mis-strided stack shows up here rather than nowhere.
                row[0] = mels[i * MEL_VALUES];
            }
            Ok(out)
        }

        fn embedding_dim(&self) -> usize {
            DIM
        }
    }

    fn mel_tagged(pool: &Arc<BufferPool>, tag: f32) -> PooledBuffer {
        let mut buf = pool.take();
        let v = buf.buffer_mut();
        v.clear();
        v.resize(MEL_VALUES, tag);
        buf
    }

    /// Runs `n` tagged items through the stage and returns what came out.
    fn run_stage(model: Arc<CountingModel>, config: BatchConfig, n: usize) -> Vec<Embedded<usize>> {
        let pool = mel_pool(config);
        let (tx, rx) = bounded(EMBED_QUEUE_DEPTH);
        let (out_tx, out_rx) = bounded(1024);
        let counter = AtomicU64::new(0);
        let cancel = CancellationToken::new();

        std::thread::scope(|scope| {
            scope.spawn(|| {
                for i in 0..n {
                    tx.send(Pending {
                        payload: i,
                        mel: Some(mel_tagged(&pool, i as f32)),
                    })
                    .unwrap();
                }
                drop(tx);
            });
            embed_stage(&model, config, &cancel, rx, &out_tx, &counter);
            drop(out_tx);
            out_rx.into_iter().collect()
        })
    }

    #[test]
    fn a_full_batch_runs_as_one_call() {
        let model = Arc::new(CountingModel::default());
        let out = run_stage(Arc::clone(&model), BatchConfig::of_size(16), 32);

        assert_eq!(out.len(), 32);
        assert_eq!(
            *model.batches.lock().unwrap(),
            vec![16, 16],
            "32 items at a batch size of 16 must be two run() calls, not 32"
        );
    }

    /// The property the whole module exists for: every row gets *its own* vector back.
    #[test]
    fn vectors_are_returned_to_the_rows_they_came_from() {
        let model = Arc::new(CountingModel::default());
        let out = run_stage(Arc::clone(&model), BatchConfig::of_size(4), 10);

        assert_eq!(out.len(), 10);
        for (i, item) in out.iter().enumerate() {
            assert_eq!(item.payload, i, "rows arrived out of order");
            let tag = item.embedding.as_ref().unwrap()[0];
            assert_eq!(tag, i as f32, "row {i} was handed another row's vector");
        }
    }

    /// The tail of a scan: nine files and a batch size of sixteen must not wait for seven
    /// files that are never coming.
    #[test]
    fn a_partial_batch_flushes_on_the_timeout_rather_than_waiting() {
        let model = Arc::new(CountingModel::default());
        let config = BatchConfig {
            size: 16,
            timeout: Duration::from_millis(30),
        };
        let pool = mel_pool(config);
        let (tx, rx) = bounded::<Pending<usize>>(EMBED_QUEUE_DEPTH);
        let (out_tx, out_rx) = bounded(64);
        let counter = AtomicU64::new(0);
        let cancel = CancellationToken::new();

        std::thread::scope(|scope| {
            let stage = scope.spawn(|| {
                embed_stage(&model, config, &cancel, rx, &out_tx, &counter);
            });

            for i in 0..9 {
                tx.send(Pending {
                    payload: i,
                    mel: Some(mel_tagged(&pool, i as f32)),
                })
                .unwrap();
            }

            // The sender is deliberately still alive: this asserts the timeout flushed the
            // batch, not that dropping the channel did.
            let first = out_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("the tail of the batch never flushed");
            assert_eq!(first.payload, 0);
            assert!(first.embedding.is_some());
            assert_eq!(*model.batches.lock().unwrap(), vec![9]);

            drop(tx);
            stage.join().unwrap();
        });
    }

    /// A quarantined file has no spectrogram. It must still come out, in order, so the
    /// persist stage sees one stream of rows rather than two.
    #[test]
    fn items_with_no_spectrogram_pass_through_in_order() {
        let model = Arc::new(CountingModel::default());
        let config = BatchConfig::of_size(4);
        let pool = mel_pool(config);
        let (tx, rx) = bounded::<Pending<usize>>(64);
        let (out_tx, out_rx) = bounded(64);
        let counter = AtomicU64::new(0);
        let cancel = CancellationToken::new();

        let out: Vec<_> = std::thread::scope(|scope| {
            scope.spawn(|| {
                for i in 0..8 {
                    let mel = (i % 2 == 0).then(|| mel_tagged(&pool, i as f32));
                    tx.send(Pending { payload: i, mel }).unwrap();
                }
                drop(tx);
            });
            embed_stage(&model, config, &cancel, rx, &out_tx, &counter);
            drop(out_tx);
            out_rx.into_iter().collect()
        });

        assert_eq!(out.len(), 8);
        for (i, item) in out.iter().enumerate() {
            assert_eq!(item.payload, i);
            assert_eq!(item.embedding.is_some(), i % 2 == 0);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 4, "counted the passengers");
        // Four spectrograms at a batch size of four is one run, and the four items with no
        // mel must not have inflated it.
        assert_eq!(*model.batches.lock().unwrap(), vec![4]);
    }

    /// An inference failure loses the vectors, not the rows.
    #[test]
    fn a_failed_batch_leaves_its_rows_unembedded_rather_than_dropping_them() {
        let model = Arc::new(CountingModel {
            fail: true,
            ..CountingModel::default()
        });
        let out = run_stage(model, BatchConfig::of_size(4), 6);

        assert_eq!(out.len(), 6, "rows were dropped by a failed batch");
        assert!(out.iter().all(|e| e.embedding.is_none()));
    }

    #[test]
    fn the_default_batch_size_is_inside_the_range_overview_prescribes() {
        let default = BatchConfig::default();
        assert!((16..=32).contains(&default.size), "§3.4 says 16-32");
        assert!(default.timeout <= Duration::from_millis(500));
    }
}
