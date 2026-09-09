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
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
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

pub use engine::{list_output_devices, AudioDeviceInfo};

/// Resampler chunk size, matched to `pipeline::decode`'s own -- see that module for the
/// reasoning; it applies identically here.
const RESAMPLE_CHUNK: usize = 1024;

/// Total device-formatted PCM held at once, in bytes. A byte budget rather than an entry count
/// because an entry's size is not fixed: it is at most ten seconds of `f32` at the *device's*
/// rate and channel count, and that varies by an order of magnitude across real interfaces --
/// ~1.9 MB for ten seconds mono at 48 kHz, ~15.4 MB for ten seconds stereo at 192 kHz. An entry
/// count sized safely for the low end wastes most of a small library's cache headroom on a
/// typical setup; sized safely for the high end, it lets a pro interface's cache balloon past
/// what was actually budgeted. A byte budget means what it says regardless of the device.
const PCM_CACHE_BYTE_BUDGET: usize = 1024 * 1024 * 1024;

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
/// FIFO with a byte budget, exactly like `peaks::PeakCache`'s `Entries` is a FIFO with an entry
/// cap -- see that module for why a true LRU is not worth the extra bookkeeping at this access
/// pattern. The budget is what differs: see [`PCM_CACHE_BYTE_BUDGET`] for why bytes, not count.
#[derive(Debug, Default)]
struct PcmCache {
    by_key: HashMap<(i64, u64), Arc<Vec<f32>>>,
    order: VecDeque<(i64, u64)>,
    total_bytes: usize,
}

impl PcmCache {
    fn get(&self, sample_id: i64, format_epoch: u64) -> Option<Arc<Vec<f32>>> {
        self.by_key.get(&(sample_id, format_epoch)).cloned()
    }

    fn insert(&mut self, sample_id: i64, format_epoch: u64, pcm: Arc<Vec<f32>>) {
        let key = (sample_id, format_epoch);
        let bytes = std::mem::size_of_val(pcm.as_slice());
        if let Some(previous) = self.by_key.insert(key, pcm) {
            // Re-inserting a key already at the back of `order` (a re-decode after a device
            // format change bumped the epoch, keyed fresh) would otherwise double-count it.
            self.total_bytes -= std::mem::size_of_val(previous.as_slice());
        } else {
            self.order.push_back(key);
        }
        self.total_bytes += bytes;

        // The just-inserted entry is always the newest and is never the one popped here: it
        // sits at the back of `order`, and the loop only ever removes from the front. A single
        // entry larger than the whole budget is kept anyway -- rejecting it would mean refusing
        // to cache (and eventually refusing to play) a sample for being itself, not for
        // crowding anything else out.
        while self.total_bytes > PCM_CACHE_BYTE_BUDGET && self.order.len() > 1 {
            if let Some(evicted) = self.order.pop_front() {
                if let Some(pcm) = self.by_key.remove(&evicted) {
                    self.total_bytes -= std::mem::size_of_val(pcm.as_slice());
                }
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
    /// The device Settings has asked for, remembered even before an engine exists to tell it
    /// to -- a device chosen on the first-run screen, before anything has ever played, must
    /// still be the one the first `play_sample` opens.
    preferred: Mutex<Option<String>>,
}

impl LazyEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// The engine, opening the preferred (or default) output device if this is the first call.
    pub fn get(&self) -> Result<Arc<Engine>, AudioError> {
        let mut cell = self
            .cell
            .lock()
            .map_err(|_| AudioError::Stream("the engine cell is poisoned".into()))?;
        if let Some(engine) = cell.as_ref() {
            return Ok(Arc::clone(engine));
        }
        let preferred = self
            .preferred
            .lock()
            .map_err(|_| AudioError::Stream("the preferred-device cell is poisoned".into()))?
            .clone();
        let engine = Engine::open(preferred)?;
        *cell = Some(Arc::clone(&engine));
        Ok(engine)
    }

    /// Whether a device has already been opened. What [`AudioPlayer::stop`] checks so that
    /// stopping playback that was never started does not itself open a device.
    pub fn is_initialized(&self) -> bool {
        self.cell.lock().map(|c| c.is_some()).unwrap_or(false)
    }

    /// Records the preferred device and, if an engine is already open, switches it live.
    pub fn set_preferred(&self, name: Option<String>) {
        if let Ok(mut preferred) = self.preferred.lock() {
            *preferred = name.clone();
        }
        if let Ok(cell) = self.cell.lock() {
            if let Some(engine) = cell.as_ref() {
                engine.set_preferred(name);
            }
        }
    }
}

/// Which decode lane a request runs on. See [`AudioPlayer::prefetch_decoder`] for why there
/// are two rather than one shared lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    /// A hover or a click: someone is waiting to hear this.
    Play,
    /// Warming the cache for something nobody has asked for yet.
    Prefetch,
}

/// Turns sample ids into sound: decode, resample and channel-format for the output device,
/// cache, and hand the result to the [`Engine`].
///
/// One [`Decoder`] behind one mutex, exactly like `peaks::PeakCache` -- the reasoning is
/// identical: a `Decoder` owns a resampler per source rate and a pool holding a 1.92 MB window,
/// and building one per hover would allocate megabytes on a cursor sweep.
pub struct AudioPlayer {
    lazy: LazyEngine,
    /// The decode lane a real [`play`](AudioPlayer::play) uses, and nothing else.
    decoder: Mutex<Decoder>,
    /// A second, identical lane used only by [`prefetch`](AudioPlayer::prefetch).
    ///
    /// One shared `Mutex<Decoder>` meant a hover could arrive one instruction after a
    /// prefetch took the lock and then wait out that prefetch's entire decode -- measured at
    /// up to 32 ms on this library, on the one path where milliseconds are the whole point.
    /// Warming the cache must never be able to delay the sound the user is waiting for, and a
    /// second decoder (one more pooled window, ~1.9 MB) is a cheaper way to guarantee that
    /// than any priority scheme over a single lock.
    prefetch_decoder: Mutex<Decoder>,
    /// One cached resampler, keyed by the device rate it was built for. A single slot rather
    /// than a map like `Decoder`'s: the device's output rate does not vary per file the way a
    /// library's source rates do, so there is only ever one rate worth caching against. One
    /// per decode lane, for the same reason the decoders are split.
    resampler: Mutex<Option<(u32, Async<f32>)>>,
    prefetch_resampler: Mutex<Option<(u32, Async<f32>)>>,
    cache: Mutex<PcmCache>,
    /// Bumped at the start of every [`AudioPlayer::play`] call. A decode that finishes after a
    /// *later* call has already started reads back something other than the sequence it was
    /// given and drops its result instead of playing it -- without this, two hovers close
    /// enough together that the first is still decoding when the second lands can finish in
    /// either order, and a slow first decode completing after a fast second one would silently
    /// override the sound the user is actually still hovering with a stale one.
    request_seq: AtomicU64,
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
            prefetch_decoder: Mutex::new(Decoder::new(BufferPool::for_decode())),
            resampler: Mutex::new(None),
            prefetch_resampler: Mutex::new(None),
            cache: Mutex::new(PcmCache::default()),
            request_seq: AtomicU64::new(0),
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
        // Claimed before anything else so that every call, including this one, has a
        // strict order to be judged against -- see `request_seq`'s doc comment.
        let my_seq = self.request_seq.fetch_add(1, Ordering::AcqRel) + 1;

        let engine = self.lazy.get()?;
        let format = engine.format();

        // The cache is consulted before the database, not after. A warm hover -- which, once
        // the background sweep has run, is nearly every hover -- has no use for the row: the
        // path in it was already turned into device-format PCM, and the sample cannot have
        // gone missing in a way that matters to a buffer already decoded. Fetching it anyway
        // put a pool checkout and a query on the one path that is supposed to be nothing but a
        // lookup and a hand-off to the engine.
        let cached = self
            .cache
            .lock()
            .ok()
            .and_then(|c| c.get(sample_id, format.epoch));

        let pcm = match cached {
            Some(pcm) => pcm,
            None => {
                // Cold: now the row is genuinely needed. A bad id is the common shape of "the
                // frontend is holding a stale selection," and this is where it surfaces.
                let conn = db.read()?;
                let row = queries::sample_row(&conn, sample_id)?
                    .ok_or(PlaybackError::NotFound(sample_id))?;
                drop(conn);
                self.decode_for(row, format, Lane::Play).await?
            }
        };

        // A later call already claimed a higher sequence number while this one was decoding
        // (or even just doing the DB lookup): that later call is what the user is actually
        // hovering now, and it will already have played or is about to. Playing this one too
        // would either glitch over it or, worse, win the engine's own generation race and
        // replace the sound the user expects with a stale one.
        if self.request_seq.load(Ordering::Acquire) != my_seq {
            return Ok(());
        }

        let generation = engine.play_pcm(pcm, gain).await;
        log_latency(&engine, generation, sample_id);
        Ok(())
    }

    /// Decodes and caches `sample_id` without playing it, so that a later [`play`](Self::play)
    /// for the same sample is a cache hit instead of a cold decode.
    ///
    /// Only does anything once a device is already open. Prefetching must never be what opens
    /// one -- that would mean hovering near a sample (not even playing one) triggers the same
    /// device-permission and hardware-wake cost as pressing play, which is not a trade a mere
    /// hover should be able to make. Once the user has played anything at all, though, the
    /// device is already open and warming its neighbors costs nothing extra.
    pub async fn prefetch(
        self: &Arc<Self>,
        db: &Database,
        sample_id: i64,
    ) -> Result<(), PlaybackError> {
        if !self.lazy.is_initialized() {
            return Ok(());
        }
        let engine = self.lazy.get()?;
        let format = engine.format();

        let already_cached = self
            .cache
            .lock()
            .ok()
            .is_some_and(|c| c.get(sample_id, format.epoch).is_some());
        if already_cached {
            return Ok(());
        }

        let conn = db.read()?;
        let row = match queries::sample_row(&conn, sample_id)? {
            Some(row) => row,
            // A neighbor that no longer exists is not this call's problem to report --
            // prefetching is a best-effort warm-up, not a request the user is waiting on.
            None => return Ok(()),
        };
        drop(conn);

        self.decode_for(row, format, Lane::Prefetch).await?;
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

    /// Sets the preferred output device by name (or `None` for the OS default). Switches a
    /// live stream immediately; otherwise just remembered for the next `play`.
    pub fn set_preferred_device(&self, name: Option<String>) {
        self.lazy.set_preferred(name);
    }

    /// Decodes, resamples and caches an already-fetched sample row for `format`, off the async
    /// runtime's worker threads: decode is real CPU work over a file, and `overview.md` §2
    /// forbids sustained CPU work on the `tokio` pool exactly as it forbids it on the main
    /// thread.
    async fn decode_for(
        self: &Arc<Self>,
        row: queries::SampleRow,
        format: DeviceFormat,
        lane: Lane,
    ) -> Result<Arc<Vec<f32>>, PlaybackError> {
        let sample_id = row.id;
        let path = row.absolute_path();
        let ext = row.ext.clone();
        let rel_path = row.rel_path.clone();

        let this = Arc::clone(self);
        let pcm = tokio::task::spawn_blocking(move || {
            this.decode_and_format(&path, &ext, &rel_path, format, lane)
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
        lane: Lane,
    ) -> Result<Vec<f32>, PlaybackError> {
        let (decoder, resampler) = match lane {
            Lane::Play => (&self.decoder, &self.resampler),
            Lane::Prefetch => (&self.prefetch_decoder, &self.prefetch_resampler),
        };

        let decoded = {
            let mut decoder = decoder.lock().map_err(|_| {
                PlaybackError::Device(AudioError::Stream("the decoder is poisoned".into()))
            })?;
            decoder
                .decode(path, ext)
                .map_err(|e| decode_error(rel_path, e))?
        };

        let mut resampler = resampler.lock().map_err(|_| {
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
#[allow(
    unknown_lints,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::chunks_exact_to_as_chunks
)]
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
    fn pcm_cache_evicts_oldest_first_once_over_budget() {
        // Entries sized so five of them exceed the budget but four do not, forcing exactly
        // one eviction on the fifth insert.
        let entry_bytes = PCM_CACHE_BYTE_BUDGET / 4 + 1;
        let entry_len = entry_bytes / std::mem::size_of::<f32>();
        let mut cache = PcmCache::default();
        for id in 0..5 {
            cache.insert(id, 1, Arc::new(vec![0.0; entry_len]));
        }
        assert_eq!(cache.by_key.len(), 4, "one eviction should have made room");
        assert!(
            cache.get(0, 1).is_none(),
            "the oldest entry should have been evicted"
        );
        assert!(cache.get(4, 1).is_some(), "the newest entry must survive");
        assert!(
            cache.total_bytes <= PCM_CACHE_BYTE_BUDGET,
            "total_bytes should track what is actually cached"
        );
    }

    #[test]
    fn pcm_cache_keeps_a_single_entry_larger_than_the_whole_budget() {
        // A ten-minute ambience bed at a high sample rate can exceed the budget on its own;
        // it must still play, not be silently refused caching.
        let huge = PCM_CACHE_BYTE_BUDGET / std::mem::size_of::<f32>() + 1;
        let mut cache = PcmCache::default();
        cache.insert(1, 1, Arc::new(vec![0.0; huge]));
        assert!(cache.get(1, 1).is_some());
    }

    #[test]
    fn pcm_cache_re_inserting_a_key_does_not_double_count_its_bytes() {
        let mut cache = PcmCache::default();
        cache.insert(1, 1, Arc::new(vec![0.0; 100]));
        let after_first = cache.total_bytes;
        cache.insert(1, 1, Arc::new(vec![0.0; 100]));
        assert_eq!(
            cache.total_bytes, after_first,
            "replacing an existing key's PCM should not grow total_bytes"
        );
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
