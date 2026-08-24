//! Real-time audio preview.
//!
//! Cross-cutting rule 5: **no allocation on the audio thread**, enforced by a debug-only
//! guard allocator that panics rather than by good intentions. The callback copies from a
//! lock-free ring and applies gain; everything else -- decode, cache, envelope scheduling --
//! happens off it.
//!
//! **What lives where.** [`engine`] is device lifecycle and the real-time callback -- it knows
//! nothing about samples, the database, or files. [`ring`] is the SPSC channel between them.
//! [`guard`] is the allocator that makes "no allocation" checkable rather than aspirational.
//! This module is the layer above all three: [`AudioPlayer`] turns a `sample_id` into decoded,
//! device-formatted PCM and hands it to an [`Engine`](engine::Engine), the way
//! [`peaks::PeakCache`] turns one into a waveform summary -- same shape, same reason the two
//! are siblings rather than one merged into the other.

pub mod engine;
pub mod guard;
pub mod peaks;
pub mod ring;

use std::{
    collections::{HashMap, VecDeque},
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use rubato::{audioadapter_buffers::direct::InterleavedSlice, Async, FixedAsync, Resampler};

use crate::{
    db::{queries, Database, DbError},
    pipeline::{
        decode::{sinc_parameters, DecodeError, TARGET_SAMPLE_RATE},
        BufferPool, Decoder,
    },
};

use engine::{AudioError, DeviceFormat, Engine};

/// Resampler chunk size, matched to `pipeline::decode`'s own -- see that module for the
/// reasoning; it applies identically here.
const RESAMPLE_CHUNK: usize = 1024;

/// Device-formatted PCM buffers held at once. Small: each entry is at most ten seconds of
/// stereo `f32` at a typical interface's rate (≈3.8 MB), and a listening session touches a
/// handful of samples in quick succession, not hundreds.
const PCM_CACHE_ENTRIES: usize = 24;

/// What can go wrong turning a `sample_id` into sound.
#[derive(Debug, thiserror::Error)]
pub enum PlaybackError {
    #[error("sample {0} was not found")]
    NotFound(i64),
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("could not decode {path}: {reason}")]
    Decode { path: String, reason: String },
    #[error(transparent)]
    Device(#[from] AudioError),
    #[error("could not resample to the output device's rate: {0}")]
    Resample(String),
}

/// A session's worth of pre-decoded buffers, keyed to the format they were built for.
///
/// FIFO with a cap, exactly like `peaks::PeakCache`'s `Entries` -- see that module for why a
/// true LRU is not worth the extra bookkeeping at this entry count and this access pattern.
#[derive(Debug, Default)]
struct PcmCache {
    by_key: HashMap<(i64, u64), Arc<Vec<f32>>>,
    order: VecDeque<(i64, u64)>,
}

impl PcmCache {
    fn get(&self, sample_id: i64, format_epoch: u64) -> Option<Arc<Vec<f32>>> {
        self.by_key.get(&(sample_id, format_epoch)).cloned()
    }

    fn insert(&mut self, sample_id: i64, format_epoch: u64, pcm: Arc<Vec<f32>>) {
        let key = (sample_id, format_epoch);
        if self.by_key.insert(key, pcm).is_none() {
            self.order.push_back(key);
        }
        while self.order.len() > PCM_CACHE_ENTRIES {
            if let Some(evicted) = self.order.pop_front() {
                self.by_key.remove(&evicted);
            }
        }
    }
}

/// An [`Engine`] that opens its device on first use, never on the cold-start path.
///
/// The exact shape of `model::session::LazySession`, for the exact reason: `overview.md` §7
/// excludes device/session init from the two-second cold-start budget, which is only honest if
/// init genuinely happens later. A failed open is not cached, because the reasons it fails --
/// no device plugged in yet, a permission dialog not yet answered -- are reasons that get fixed
/// while the app keeps running.
#[derive(Debug, Default)]
pub struct LazyEngine {
    cell: Mutex<Option<Arc<Engine>>>,
}

impl LazyEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// The engine, opening the default output device if this is the first call.
    pub fn get(&self) -> Result<Arc<Engine>, AudioError> {
        let mut cell = self
            .cell
            .lock()
            .map_err(|_| AudioError::Stream("the engine cell is poisoned".into()))?;
        if let Some(engine) = cell.as_ref() {
            return Ok(Arc::clone(engine));
        }
        let engine = Engine::open()?;
        *cell = Some(Arc::clone(&engine));
        Ok(engine)
    }

    /// Whether a device has already been opened. What [`AudioPlayer::stop`] checks so that
    /// stopping playback that was never started does not itself open a device.
    pub fn is_initialized(&self) -> bool {
        self.cell.lock().map(|c| c.is_some()).unwrap_or(false)
    }
}

/// Turns sample ids into sound: decode, resample and channel-format for the output device,
/// cache, and hand the result to the [`Engine`].
///
/// One [`Decoder`] behind one mutex, exactly like `peaks::PeakCache` -- the reasoning is
/// identical: a `Decoder` owns a resampler per source rate and a pool holding a 1.92 MB window,
/// and building one per hover would allocate megabytes on a cursor sweep.
pub struct AudioPlayer {
    lazy: LazyEngine,
    decoder: Mutex<Decoder>,
    /// One cached resampler, keyed by the device rate it was built for. A single slot rather
    /// than a map like `Decoder`'s: the device's output rate does not vary per file the way a
    /// library's source rates do, so there is only ever one rate worth caching against.
    resampler: Mutex<Option<(u32, Async<f32>)>>,
    cache: Mutex<PcmCache>,
}

impl Default for AudioPlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioPlayer {
    pub fn new() -> Self {
        Self {
            lazy: LazyEngine::new(),
            decoder: Mutex::new(Decoder::new(BufferPool::for_decode())),
            resampler: Mutex::new(None),
            cache: Mutex::new(PcmCache::default()),
        }
    }

    /// Starts (or retriggers) playback of `sample_id`.
    ///
    /// Hover debounce lives in the frontend -- `overview.md` §6.1 fixes this command's
    /// signature at `(sampleId, gain)`, with no room for a hover/click distinction on the wire
    /// -- but retriggering here is always clean regardless of how fast the caller calls
    /// again: [`Engine::play_pcm`] bumps the generation, which is what makes the previous
    /// clip's tail fade out instead of clicking.
    pub async fn play(
        self: &Arc<Self>,
        db: &Database,
        sample_id: i64,
        gain: f32,
    ) -> Result<(), PlaybackError> {
        // The database is checked before the device is opened, deliberately: a bad id is the
        // common shape of "the frontend is holding a stale selection," and answering it must
        // not depend on -- or pay the cost of -- a machine having audio hardware at all.
        let conn = db.read()?;
        let row =
            queries::sample_row(&conn, sample_id)?.ok_or(PlaybackError::NotFound(sample_id))?;
        drop(conn);

        let engine = self.lazy.get()?;
        let format = engine.format();

        let pcm = match self
            .cache
            .lock()
            .ok()
            .and_then(|c| c.get(sample_id, format.epoch))
        {
            Some(pcm) => pcm,
            None => self.decode_for(row, format).await?,
        };

        let generation = engine.play_pcm(pcm, gain);
        log_latency(&engine, generation, sample_id);
        Ok(())
    }

    /// Releases whatever is currently playing. A no-op, not an error, if no device has ever
    /// been opened -- there is nothing to release, and opening one just to release nothing
    /// would be the cold-start cost this module exists to avoid.
    pub fn stop(&self) -> Result<(), PlaybackError> {
        if self.lazy.is_initialized() {
            self.lazy.get()?.stop();
        }
        Ok(())
    }

    /// Decodes, resamples and caches an already-fetched sample row for `format`, off the async
    /// runtime's worker threads: decode is real CPU work over a file, and `overview.md` §2
    /// forbids sustained CPU work on the `tokio` pool exactly as it forbids it on the main
    /// thread.
    async fn decode_for(
        self: &Arc<Self>,
        row: queries::SampleRow,
        format: DeviceFormat,
    ) -> Result<Arc<Vec<f32>>, PlaybackError> {
        let sample_id = row.id;
        let path = row.absolute_path();
        let ext = row.ext.clone();
        let rel_path = row.rel_path.clone();

        let this = Arc::clone(self);
        let pcm = tokio::task::spawn_blocking(move || {
            this.decode_and_format(&path, &ext, &rel_path, format)
        })
        .await
        .map_err(|e| {
            PlaybackError::Device(AudioError::Stream(format!("the decode task panicked: {e}")))
        })??;

        let pcm = Arc::new(pcm);
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(sample_id, format.epoch, Arc::clone(&pcm));
        }
        Ok(pcm)
    }

    /// The blocking half: decode the file's analysis window, then resample and interleave it
    /// to `format`. Runs on a `spawn_blocking` thread, never on an async worker.
    fn decode_and_format(
        &self,
        path: &Path,
        ext: &str,
        rel_path: &str,
        format: DeviceFormat,
    ) -> Result<Vec<f32>, PlaybackError> {
        let decoded = {
            let mut decoder = self.decoder.lock().map_err(|_| {
                PlaybackError::Device(AudioError::Stream("the decoder is poisoned".into()))
            })?;
            decoder
                .decode(path, ext)
                .map_err(|e| decode_error(rel_path, e))?
        };

        let mut resampler = self.resampler.lock().map_err(|_| {
            PlaybackError::Device(AudioError::Stream("the resampler is poisoned".into()))
        })?;
        to_device_format(&decoded.samples, format, &mut resampler)
    }
}

impl std::fmt::Debug for AudioPlayer {
    // Manual rather than derived: `rubato::Async` does not implement `Debug`, and wrapping it
    // in `Mutex` does not change that. Nothing here is worth printing per-field anyway --
    // there is exactly one of these, managed by Tauri, and what a caller wants from its debug
    // representation is "this exists," not its resampler's internal state.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioPlayer").finish_non_exhaustive()
    }
}

/// Resamples `mono` (at [`TARGET_SAMPLE_RATE`]) to `format`'s rate if it differs, then
/// interleaves it to `format`'s channel count by simple duplication.
///
/// **Duplication, not a real downmix or pan law.** The source is already mono -- Phase 2's
/// decode downmixes on the way in -- so every output channel is meant to carry the same
/// signal; there is no stereo image to preserve or lose.
fn to_device_format(
    mono: &[f32],
    format: DeviceFormat,
    resampler: &mut Option<(u32, Async<f32>)>,
) -> Result<Vec<f32>, PlaybackError> {
    let at_rate: std::borrow::Cow<'_, [f32]> = if format.sample_rate == TARGET_SAMPLE_RATE {
        std::borrow::Cow::Borrowed(mono)
    } else {
        std::borrow::Cow::Owned(resample(mono, format.sample_rate, resampler)?)
    };

    let channels = format.channels.max(1) as usize;
    let mut out = Vec::with_capacity(at_rate.len() * channels);
    for &sample in at_rate.iter() {
        out.extend(std::iter::repeat_n(sample, channels));
    }
    Ok(out)
}

/// Resamples `mono` from [`TARGET_SAMPLE_RATE`] to `target_rate`, reusing `cached` when it was
/// already built for this rate. Mirrors `pipeline::decode::Decoder::resample_into`'s approach,
/// mirrored rather than shared because that one resamples *to* a fixed rate from a varying
/// source and this one resamples *from* a fixed rate to a varying (but session-stable) one --
/// the caching key is on the opposite side in each.
fn resample(
    mono: &[f32],
    target_rate: u32,
    cached: &mut Option<(u32, Async<f32>)>,
) -> Result<Vec<f32>, PlaybackError> {
    let needs_rebuild = !matches!(cached, Some((rate, _)) if *rate == target_rate);
    if needs_rebuild {
        let ratio = f64::from(target_rate) / f64::from(TARGET_SAMPLE_RATE);
        let built = Async::new_sinc(
            ratio,
            1.0,
            &sinc_parameters(),
            RESAMPLE_CHUNK,
            1,
            FixedAsync::Input,
        )
        .map_err(|e| PlaybackError::Resample(e.to_string()))?;
        *cached = Some((target_rate, built));
    }
    // Present by construction: either it already matched, or the branch above just built one.
    let (_, resampler) = cached
        .as_mut()
        .ok_or_else(|| PlaybackError::Resample("the resampler was not built".into()))?;
    if !needs_rebuild {
        // A cached resampler carries the previous clip's delay-line tail; not resetting bleeds
        // its decay into the next one's opening samples.
        resampler.reset();
    }

    let input_len = mono.len();
    let needed = resampler.process_all_needed_output_len(input_len);
    let mut out = vec![0.0f32; needed];

    let adapter = InterleavedSlice::new(mono, 1, input_len)
        .map_err(|e| PlaybackError::Resample(e.to_string()))?;
    let mut sink = InterleavedSlice::new_mut(out.as_mut_slice(), 1, needed)
        .map_err(|e| PlaybackError::Resample(e.to_string()))?;

    let (_, produced) = resampler
        .process_all_into_buffer(&adapter, &mut sink, input_len, None)
        .map_err(|e| PlaybackError::Resample(e.to_string()))?;
    out.truncate(produced);
    Ok(out)
}

/// A file that will not decode is a `decode` error with its path in it -- the same shape
/// `peaks::PeakCache` produces, because a sample whose peaks will not generate is a sample
/// whose audio will not play either, and the inspector should say the same thing either way.
fn decode_error(rel_path: &str, e: DecodeError) -> PlaybackError {
    PlaybackError::Decode {
        path: rel_path.to_string(),
        reason: e.to_string(),
    }
}

/// Watches for the first audible sample of `generation` and logs the latency once it arrives.
///
/// **What this measures, precisely.** From the moment [`AudioPlayer::play`] was called to the
/// first non-silent sample the real-time callback emitted -- not from the pointer event that
/// caused the call. `overview.md` §7's target is stated as hover-to-audible end to end, and the
/// gap between the two is the IPC round trip and whatever debounce the frontend applies before
/// calling `play_sample` at all; neither is observable from here. What this number bounds is
/// the half of the budget this module is actually responsible for: decode, resample, queue,
/// ramp.
fn log_latency(engine: &Arc<Engine>, generation: u64, sample_id: i64) {
    let engine = Arc::clone(engine);
    tokio::spawn(async move {
        // 20 x 10ms = 200ms, several times the 50ms budget this is checking against; a
        // generation that has not gone audible by then is not going to, and polling forever
        // over a superseded or stopped play would just leak a task per hover.
        for _ in 0..20 {
            if let Some(latency) = engine.latency_since(generation) {
                tracing::info!(
                    sample_id,
                    generation,
                    latency_ms = latency.as_secs_f64() * 1000.0,
                    "audio: command-received to first audible sample"
                );
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn format(sample_rate: u32, channels: u16) -> DeviceFormat {
        DeviceFormat {
            sample_rate,
            channels,
            epoch: 1,
        }
    }

    #[test]
    fn a_matching_rate_only_duplicates_channels() {
        let mut resampler = None;
        let mono = [0.1f32, 0.2, 0.3];
        let out = to_device_format(&mono, format(TARGET_SAMPLE_RATE, 2), &mut resampler).unwrap();
        assert_eq!(out, [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
        assert!(
            resampler.is_none(),
            "no resample was needed, so none should have been built"
        );
    }

    #[test]
    fn mono_stays_mono_at_one_channel() {
        let mut resampler = None;
        let mono = [1.0f32, -1.0];
        let out = to_device_format(&mono, format(TARGET_SAMPLE_RATE, 1), &mut resampler).unwrap();
        assert_eq!(out, [1.0, -1.0]);
    }

    #[test]
    fn a_different_rate_resamples_before_interleaving() {
        let mut resampler = None;
        // A half-second of a 440 Hz tone at the pipeline's native rate.
        let mono: Vec<f32> = (0..TARGET_SAMPLE_RATE / 2)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / TARGET_SAMPLE_RATE as f32).sin()
            })
            .collect();

        let out = to_device_format(&mono, format(44_100, 2), &mut resampler).unwrap();
        assert!(resampler.is_some(), "a rate change must build a resampler");
        // Stereo, and roughly proportional to the rate ratio -- exact frame counts are an
        // implementation detail of the sinc filter's edge handling.
        assert_eq!(out.len() % 2, 0);
        let frames = out.len() / 2;
        let expected = (mono.len() as f64 * 44_100.0 / TARGET_SAMPLE_RATE as f64) as usize;
        assert!(
            frames.abs_diff(expected) < 200,
            "resampled length {frames} should be close to {expected}"
        );
        // Every output frame's two channels must agree -- duplication, not a real stereo mix.
        for pair in out.chunks_exact(2) {
            assert_eq!(pair[0], pair[1]);
        }
    }

    #[test]
    fn a_cached_resampler_is_reused_for_the_same_rate() {
        let mut resampler = None;
        let mono = vec![0.0f32; 4_800];
        to_device_format(&mono, format(44_100, 1), &mut resampler).unwrap();
        let built_once = resampler.is_some();
        // Same rate again: `resample` must reuse the cached resampler rather than rebuilding.
        to_device_format(&mono, format(44_100, 1), &mut resampler).unwrap();
        assert!(built_once && resampler.is_some());
        assert_eq!(resampler.as_ref().unwrap().0, 44_100);
    }

    #[test]
    fn pcm_cache_evicts_oldest_first() {
        let mut cache = PcmCache::default();
        for id in 0..(PCM_CACHE_ENTRIES as i64 + 5) {
            cache.insert(id, 1, Arc::new(vec![0.0]));
        }
        assert_eq!(cache.by_key.len(), PCM_CACHE_ENTRIES);
        assert!(
            cache.get(0, 1).is_none(),
            "the oldest entry should have been evicted"
        );
        assert!(cache.get(PCM_CACHE_ENTRIES as i64 + 4, 1).is_some());
    }

    #[test]
    fn pcm_cache_keys_on_format_epoch_so_a_rebuild_invalidates_it() {
        let mut cache = PcmCache::default();
        cache.insert(7, 1, Arc::new(vec![1.0]));
        assert!(cache.get(7, 1).is_some());
        assert!(
            cache.get(7, 2).is_none(),
            "a new format epoch must miss, not return stale PCM"
        );
    }
}
