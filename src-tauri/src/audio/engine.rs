//! `cpal` output stream and device lifecycle.
//!
//! Device change and sample-rate change are normal events, not errors: unplugging an
//! interface mid-audition must not panic the audio thread or take the app with it.
//!
//! **What the real-time callback is allowed to touch, and nothing else.** The audio thread
//! (`overview.md` §2) owns exactly one [`RingConsumer`] and reads a handful of atomics in
//! [`Transport`] -- no mutex, no allocation, no logging. [`render`] is the whole of what it
//! does, factored out as a free function over plain data so it can be proven correct against a
//! bare `HeapRb` in tests, with no `cpal` stream and no audio hardware involved.
//!
//! **Retriggering is a generation counter, not a cleared buffer.** [`Engine::play_pcm`] bumps
//! [`Transport::generation`] and hands a fresh decode-ahead task the *current* producer lock;
//! [`render`] notices the generation change, drops whatever the previous generation had queued
//! (`RingConsumer::clear`, which the audio thread is allowed to call because it is the
//! exclusive owner of that half), and starts a fresh attack ramp. A superseded push task
//! notices the same counter and stops writing -- there is never a moment where two tasks are
//! both trying to fill the ring, which is what keeps this correct as an SPSC channel without
//! either side ever locking.
//!
//! **One `tokio::sync::Mutex` guards the producer, and it is never touched by the audio
//! thread.** It exists so exactly one decode-ahead task is pushing at a time; a task that
//! finds itself superseded returns as soon as it next checks the generation, which bounds how
//! long a new play has to wait for the lock to whoever held it before.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    ErrorKind, SupportedStreamConfig, SupportedStreamConfigRange,
};
use tokio::sync::{Mutex as AsyncMutex, Notify};

use crate::{
    audio::{
        guard::AudioThreadGuard,
        ring::{self, Consumer, Observer, Producer, RingConsumer, RingProducer},
    },
    pipeline::decode::TARGET_SAMPLE_RATE,
};

/// Attack ramp on every new generation -- a fresh hover or a retrigger alike. Long enough that
/// a full-scale step function never reaches the DAC, short enough that a one-shot's own
/// transient is not audibly softened.
const ATTACK_MS: f32 = 5.0;

/// Release ramp on `stop()`, a retrigger, and the natural end of a clip. Slightly longer than
/// the attack: a fade-out is more forgiving of being a little slow than a fade-in is of being
/// audible as a swell.
const RELEASE_MS: f32 = 8.0;

/// How long a decode-ahead task waits before retrying a ring that reported no room. The ring
/// holds `ring::RING_SECONDS` of audio; a few milliseconds of backoff is negligible against
/// that and keeps a full ring from becoming a busy-loop.
const PUSH_BACKOFF: Duration = Duration::from_millis(2);

/// Sane bounds on `gain`, independent of whatever the caller asked for. `2.0` is +6 dB of
/// headroom over unity, which is enough for "this file was recorded quiet" without being
/// enough to turn a typo into a speaker-damaging blast.
const GAIN_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;

/// Everything that can go wrong opening or rebuilding the output stream.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("no audio output device is available")]
    NoDevice,
    #[error("could not determine a usable output configuration: {0}")]
    Config(String),
    #[error("could not open the output stream: {0}")]
    Stream(String),
}

/// One enumerable output device, for the Settings picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

/// Every output device `cpal` can see on this host, default first.
///
/// A name, not a stable id: this `cpal` gives a device's name through `Display` rather than a
/// persistent identifier, and a name is exactly what [`choose_device`] needs back to find the
/// same device again.
pub fn list_output_devices() -> Result<Vec<AudioDeviceInfo>, AudioError> {
    let host = cpal::default_host();
    let default_name = host.default_output_device().map(|d| d.to_string());

    let mut devices: Vec<AudioDeviceInfo> = host
        .output_devices()
        .map_err(|e| AudioError::Config(e.to_string()))?
        .map(|d| {
            let name = d.to_string();
            let is_default = Some(&name) == default_name.as_ref();
            AudioDeviceInfo { name, is_default }
        })
        .collect();
    devices.sort_by(|a, b| b.is_default.cmp(&a.is_default).then(a.name.cmp(&b.name)));
    Ok(devices)
}

/// The output format a built stream ended up with, and which build it came from.
///
/// `epoch` is what lets [`crate::audio::PcmCache`] know a cached, device-formatted buffer is
/// still good: it increments on every rebuild, so a cache keyed on `(sample_id, epoch)` treats
/// a device or sample-rate change as what it is -- every previously-formatted buffer is stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub epoch: u64,
}

/// Lock-free state shared between control-side callers and the real-time callback.
///
/// Every field here is read or written with `Ordering::Relaxed`. Nothing in this struct
/// publishes other memory that a `Relaxed` load would need to see correctly ordered -- the PCM
/// bytes themselves cross the boundary through the ring buffer, whose own synchronization is
/// what `ringbuf` provides. These atomics are control signals (which generation, how loud, is
/// it still wanted), not a channel for data.
#[derive(Debug)]
struct Transport {
    /// Bumped by every `play_pcm` and by every stream rebuild. The audio thread treats a
    /// change as "forget what you were doing and start over"; a decode-ahead task treats a
    /// mismatch as "someone else owns playback now, stop pushing."
    generation: AtomicU64,
    /// `f32::to_bits` of the current target gain. Bits rather than a float `Atomic` type,
    /// which the standard library does not provide.
    gain_bits: AtomicU32,
    /// Explicit intent: true from `play_pcm` until `stop()` or a superseding `play_pcm`.
    want_playing: AtomicBool,
    /// The generation a decode-ahead task has finished pushing every frame of, or `0` if no
    /// generation has finished yet. Compared against the audio thread's current generation, not
    /// stored as a bool, because a stale `true` left over from the previous clip would end
    /// playback of the new one a frame after it starts.
    finished_generation: AtomicU64,
    /// Nanoseconds since `Engine`'s epoch when `play_pcm` was called for the generation that is
    /// current right now. Overwritten by every `play_pcm`; a reader must confirm the generation
    /// has not moved on before trusting it.
    started_at_nanos: AtomicU64,
    /// Nanoseconds since the epoch when the audio thread first emitted a non-silent sample for
    /// the current generation. `0` means "not yet" -- true silence at time zero is not a state
    /// this player can reach, since the epoch is fixed at construction and playback always
    /// starts some nonzero time after it.
    first_audible_nanos: AtomicU64,
    /// Incremented on every stream (re)build. See [`DeviceFormat::epoch`].
    format_epoch: AtomicU64,
}

impl Transport {
    fn new() -> Self {
        Self {
            generation: AtomicU64::new(0),
            gain_bits: AtomicU32::new(1.0f32.to_bits()),
            want_playing: AtomicBool::new(false),
            finished_generation: AtomicU64::new(0),
            started_at_nanos: AtomicU64::new(0),
            first_audible_nanos: AtomicU64::new(0),
            format_epoch: AtomicU64::new(0),
        }
    }
}

/// Where [`render`] is in its envelope, per generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Nothing has ever played, or the release ramp finished. Output is silence and the ring
    /// is not touched.
    Idle,
    /// Ramping `0.0 -> 1.0`, always exactly [`ATTACK_MS`] long regardless of what triggered it
    /// -- a fresh sample and a retrigger sound the same at the very start.
    Attack,
    /// Steady state: full envelope, popping and playing whatever the ring holds.
    Sustain,
    /// Ramping `1.0 -> 0.0`, entered from `stop()`, a retrigger, or the clip's natural end.
    Release,
}

/// Real-time-thread-only state for one stream. Rebuilt fresh by every call to `build_stream`;
/// never shared, never locked.
#[derive(Debug)]
struct CallbackState {
    /// The last generation this callback observed, so it can detect a change on the next call.
    generation: u64,
    phase: Phase,
    envelope: f32,
    /// Whether [`Transport::first_audible_nanos`] has already been written for `generation`.
    /// Local rather than a compare-exchange on the atomic itself: only this thread ever writes
    /// that field, so there is nothing to race with, and the flag avoids the atomic write past
    /// the first frame that has real audio in it.
    latency_recorded: bool,
    attack_step: f32,
    release_step: f32,
}

impl CallbackState {
    fn new(sample_rate: u32) -> Self {
        let step_for = |ms: f32| -> f32 {
            let frames = (sample_rate as f32 * ms / 1000.0).max(1.0);
            1.0 / frames
        };
        Self {
            generation: 0,
            phase: Phase::Idle,
            envelope: 0.0,
            latency_recorded: false,
            attack_step: step_for(ATTACK_MS),
            release_step: step_for(RELEASE_MS),
        }
    }
}

/// Renders one callback's worth of interleaved `output`: pops from `consumer`, drives the
/// attack/release envelope, applies gain. The entire body of the real-time callback, and nothing
/// else runs on that thread -- factored out so it is provable against a plain `HeapRb` with no
/// `cpal` stream and no audio hardware anywhere near the test.
///
/// `now_nanos` stands in for `Instant::now()` relative to the engine's epoch; a test supplies a
/// deterministic clock, the real callback supplies the genuine one.
fn render(
    state: &mut CallbackState,
    transport: &Transport,
    consumer: &mut RingConsumer,
    output: &mut [f32],
    channels: usize,
    now_nanos: impl Fn() -> u64,
) {
    let channels = channels.max(1);
    let generation = transport.generation.load(Ordering::Relaxed);
    if generation != state.generation {
        state.generation = generation;
        state.envelope = 0.0;
        state.phase = if generation == 0 {
            Phase::Idle
        } else {
            Phase::Attack
        };
        state.latency_recorded = false;
        // The exclusive-owner drop: whatever the previous generation queued belongs to a clip
        // nobody wants anymore, and this is the only place that can safely discard it.
        consumer.clear();
    }

    // The common case -- nothing has ever played -- costs one atomic load and a `fill`.
    if generation == 0 {
        output.fill(0.0);
        return;
    }

    let want = transport.want_playing.load(Ordering::Relaxed);
    let finished = transport.finished_generation.load(Ordering::Relaxed) == generation;
    let gain = f32::from_bits(transport.gain_bits.load(Ordering::Relaxed));

    for frame in output.chunks_mut(channels) {
        match state.phase {
            Phase::Attack => {
                state.envelope += state.attack_step;
                if state.envelope >= 1.0 {
                    state.envelope = 1.0;
                    state.phase = Phase::Sustain;
                }
            }
            Phase::Sustain => {
                // Explicitly stopped, or the clip is both finished producing and fully drained
                // -- not merely empty this instant, which an in-flight decode-ahead task can
                // also cause and which is an underrun, not an ending.
                if !want || (finished && consumer.is_empty()) {
                    state.phase = Phase::Release;
                }
            }
            Phase::Release => {
                state.envelope -= state.release_step;
                if state.envelope <= 0.0 {
                    state.envelope = 0.0;
                    state.phase = Phase::Idle;
                }
            }
            Phase::Idle => {}
        }

        if state.phase == Phase::Idle {
            frame.fill(0.0);
            continue;
        }

        let popped = consumer.pop_slice(frame);
        if popped < frame.len() {
            // An underrun during Attack/Sustain, or the tail of Release once the ring has
            // nothing left to give the fade: silence, not stale data.
            frame[popped..].fill(0.0);
        }
        if popped > 0 && !state.latency_recorded {
            transport
                .first_audible_nanos
                .store(now_nanos(), Ordering::Relaxed);
            state.latency_recorded = true;
        }

        let applied = gain * state.envelope;
        for sample in frame.iter_mut() {
            *sample *= applied;
        }
    }
}

/// Picks a device output config, preferring [`TARGET_SAMPLE_RATE`] so the common case needs no
/// resampling between the decode pipeline's own output and the speaker.
fn preferred_rate(
    ranges: impl Iterator<Item = SupportedStreamConfigRange>,
) -> Option<SupportedStreamConfig> {
    ranges
        .filter_map(|range| range.try_with_sample_rate(TARGET_SAMPLE_RATE))
        .next()
}

fn choose_config(device: &cpal::Device) -> Result<SupportedStreamConfig, AudioError> {
    if let Ok(ranges) = device.supported_output_configs() {
        if let Some(config) = preferred_rate(ranges) {
            return Ok(config);
        }
    }
    device
        .default_output_config()
        .map_err(|e| AudioError::Config(e.to_string()))
}

/// Picks `preferred` by name if it is still attached, falling back to the host default when it
/// is absent, gone, or `None` -- the same fallback `overview.md` §6.1's device commands
/// document: a device unplugged since the user chose it must not turn every play into an
/// error.
fn choose_device(host: &cpal::Host, preferred: Option<&str>) -> Result<cpal::Device, AudioError> {
    if let Some(name) = preferred {
        if let Ok(mut devices) = host.output_devices() {
            if let Some(device) = devices.find(|d| d.to_string() == name) {
                return Ok(device);
            }
        }
        tracing::warn!(
            device = name,
            "preferred output device not found; using the default"
        );
    }
    host.default_output_device().ok_or(AudioError::NoDevice)
}

/// Builds one stream: picks a device and config, allocates a fresh ring sized for it, and wires
/// the real-time callback to the given [`Transport`]. Used both for the very first stream and
/// for every rebuild after a device change, so the two paths cannot drift apart.
///
/// Bumps `transport`'s generation and clears `want_playing`: whatever a decode-ahead task was
/// mid-push on the *previous* ring is sized for a format this new ring may not share, and must
/// stop rather than write frames that no longer line up with the channel count.
fn build_stream(
    host: &cpal::Host,
    preferred: Option<&str>,
    transport: &Arc<Transport>,
    rebuild: &Arc<Notify>,
    epoch: Instant,
) -> Result<(cpal::Stream, RingProducer, DeviceFormat), AudioError> {
    let device = choose_device(host, preferred)?;
    let supported = choose_config(&device)?;
    let config = supported.config();
    let channels = config.channels;
    let sample_rate = config.sample_rate;

    let (producer, mut consumer) = ring::ring(sample_rate, channels);

    transport.generation.fetch_add(1, Ordering::AcqRel);
    transport.want_playing.store(false, Ordering::Relaxed);
    let format_epoch = transport.format_epoch.fetch_add(1, Ordering::AcqRel) + 1;

    let mut state = CallbackState::new(sample_rate);
    let cb_transport = Arc::clone(transport);
    let channels_usize = channels as usize;

    let data_callback = move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
        let _guard = AudioThreadGuard::enter();
        render(
            &mut state,
            &cb_transport,
            &mut consumer,
            output,
            channels_usize,
            || epoch.elapsed().as_nanos() as u64,
        );
    };

    let err_rebuild = Arc::clone(rebuild);
    let error_callback = move |err: cpal::Error| match err.kind() {
        // The stream is still running against the new default device; nothing to do.
        ErrorKind::DeviceChanged => {
            tracing::info!("audio route changed; the stream followed automatically");
        }
        ErrorKind::DeviceNotAvailable | ErrorKind::StreamInvalidated => {
            tracing::warn!(error = %err, "audio output needs a new stream");
            err_rebuild.notify_one();
        }
        // An `Xrun`, a backend hiccup: worth the log line, not worth tearing the stream down.
        _ => tracing::warn!(error = %err, "audio stream error"),
    };

    let stream = device
        .build_output_stream::<f32, _, _>(config, data_callback, error_callback, None)
        .map_err(|e| AudioError::Stream(e.to_string()))?;
    stream
        .play()
        .map_err(|e| AudioError::Stream(e.to_string()))?;

    Ok((
        stream,
        producer,
        DeviceFormat {
            sample_rate,
            channels,
            epoch: format_epoch,
        },
    ))
}

/// One output stream, its device, and everything a decode-ahead task needs to fill it.
///
/// Built lazily -- see [`crate::audio::LazyEngine`] -- because opening a device is exactly the
/// kind of thing `overview.md` §7's cold-start budget says should not happen before the first
/// frame. Once built, it outlives device changes: `stream`, `producer` and `format` are the
/// parts a rebuild replaces, behind locks that only ever see control-side contention.
pub struct Engine {
    host: cpal::Host,
    /// The device name Settings last asked for, or `None` for "whatever the OS default is."
    /// Read by every rebuild, not just the first build, so a device chosen while the app is
    /// running survives a later `DeviceChanged`/`DeviceNotAvailable` rebuild rather than being
    /// silently forgotten in favor of the default.
    preferred: Mutex<Option<String>>,
    stream: Mutex<Option<cpal::Stream>>,
    producer: AsyncMutex<RingProducer>,
    transport: Arc<Transport>,
    format: Mutex<DeviceFormat>,
    rebuild: Arc<Notify>,
    epoch: Instant,
}

impl std::fmt::Debug for Engine {
    // Manual rather than derived: neither `cpal::Host`/`Stream` nor a `ringbuf` producer
    // implement `Debug`. What is actually useful to see is the live format, which `format()`
    // already exposes as a small `Copy` struct.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("format", &self.format())
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Opens `preferred`'s device (or the OS default, if `None` or not found) and starts the
    /// background task that rebuilds the stream when `cpal` reports it needs one.
    pub fn open(preferred: Option<String>) -> Result<Arc<Self>, AudioError> {
        let host = cpal::default_host();
        let transport = Arc::new(Transport::new());
        let rebuild = Arc::new(Notify::new());
        let epoch = Instant::now();

        let (stream, producer, format) =
            build_stream(&host, preferred.as_deref(), &transport, &rebuild, epoch)?;

        let engine = Arc::new(Self {
            host,
            preferred: Mutex::new(preferred),
            stream: Mutex::new(Some(stream)),
            producer: AsyncMutex::new(producer),
            transport,
            format: Mutex::new(format),
            rebuild,
            epoch,
        });

        tokio::spawn(watch_for_rebuild(Arc::clone(&engine)));
        Ok(engine)
    }

    /// Switches the stream to a named device (or back to the OS default, for `None`),
    /// rebuilding immediately rather than waiting for the next `DeviceChanged` error.
    ///
    /// Reuses the same [`Notify`] the error callback signals on: a device switch and a device
    /// failure both mean "the current stream is no longer the right one," and
    /// [`rebuild_stream`] already knows how to open a fresh one against whatever `preferred`
    /// currently says.
    ///
    /// [`rebuild_stream`]: Engine::rebuild_stream
    pub fn set_preferred(&self, name: Option<String>) {
        *self
            .preferred
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = name;
        self.rebuild.notify_one();
    }

    /// The format the currently open stream renders to, for [`crate::audio::PcmCache`] to key
    /// on and for the decode-ahead path to resample and interleave against.
    pub fn format(&self) -> DeviceFormat {
        *self
            .format
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Starts playback of an already device-formatted PCM buffer, returning the generation this
    /// play was assigned.
    ///
    /// `pcm` is interleaved `f32` at exactly `self.format()`'s rate and channel count --
    /// [`crate::audio::to_device_format`] is what produces that, off this thread. Spawns the
    /// decode-ahead push task and returns immediately; the caller does not wait for a single
    /// frame to reach the speaker.
    pub fn play_pcm(self: &Arc<Self>, pcm: Arc<Vec<f32>>, gain: f32) -> u64 {
        let gain = gain.clamp(*GAIN_RANGE.start(), *GAIN_RANGE.end());
        let generation = self.transport.generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.transport
            .gain_bits
            .store(gain.to_bits(), Ordering::Relaxed);
        self.transport
            .finished_generation
            .store(0, Ordering::Relaxed);
        self.transport
            .first_audible_nanos
            .store(0, Ordering::Relaxed);
        self.transport
            .started_at_nanos
            .store(self.now_nanos(), Ordering::Relaxed);
        // Ordered last: the moment this is visible, the audio thread may start reading a
        // generation whose gain and timestamps must already be in place.
        self.transport.want_playing.store(true, Ordering::Relaxed);

        let engine = Arc::clone(self);
        tokio::spawn(async move { engine.push_pcm(generation, pcm).await });
        generation
    }

    /// Releases whatever is currently playing. Cooperative, like a scan's cancellation: the
    /// audio thread ramps down over [`RELEASE_MS`] rather than cutting to silence, and a
    /// decode-ahead task still filling the ring notices on its next loop iteration and stops.
    pub fn stop(&self) {
        self.transport.want_playing.store(false, Ordering::Relaxed);
    }

    /// Elapsed time from `play_pcm` to the first audible sample of `generation`, if that
    /// generation is still the one playing and has produced sound. `None` covers both "not
    /// audible yet" and "superseded by a later play" -- a caller polling this in a loop cannot
    /// tell those apart from one read and should not need to.
    pub fn latency_since(&self, generation: u64) -> Option<Duration> {
        if self.transport.generation.load(Ordering::Relaxed) != generation {
            return None;
        }
        let first = self.transport.first_audible_nanos.load(Ordering::Relaxed);
        if first == 0 {
            return None;
        }
        let started = self.transport.started_at_nanos.load(Ordering::Relaxed);
        Some(Duration::from_nanos(first.saturating_sub(started)))
    }

    fn now_nanos(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// Pushes `pcm` into the ring in whatever chunks it accepts, stopping early if superseded
    /// or stopped. Marks `generation` finished once every frame has been handed over -- not
    /// once every frame has been *played*, which is [`render`]'s job to notice by draining the
    /// ring.
    async fn push_pcm(self: Arc<Self>, generation: u64, pcm: Arc<Vec<f32>>) {
        let mut producer = self.producer.lock().await;
        let mut offset = 0;
        while offset < pcm.len() {
            if self.transport.generation.load(Ordering::Relaxed) != generation
                || !self.transport.want_playing.load(Ordering::Relaxed)
            {
                return;
            }
            let pushed = producer.push_slice(&pcm[offset..]);
            offset += pushed;
            if pushed == 0 {
                // Held across the await deliberately: `tokio::sync::Mutex`'s guard is `Send`
                // for exactly this, and releasing it here would let a *third*, not-yet-spawned
                // task race this one for no benefit -- a superseding task still has to wait
                // for a lock either way, and this keeps the wait bounded by one sleep rather
                // than by however the runtime happens to schedule a re-lock.
                tokio::time::sleep(PUSH_BACKOFF).await;
            }
        }
        if self.transport.generation.load(Ordering::Relaxed) == generation {
            self.transport
                .finished_generation
                .store(generation, Ordering::Relaxed);
        }
    }

    async fn rebuild_stream(self: &Arc<Self>) {
        let preferred = self
            .preferred
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        match build_stream(
            &self.host,
            preferred.as_deref(),
            &self.transport,
            &self.rebuild,
            self.epoch,
        ) {
            Ok((stream, producer, format)) => {
                *self.producer.lock().await = producer;
                *self
                    .stream
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(stream);
                *self
                    .format
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = format;
                tracing::info!(
                    sample_rate = format.sample_rate,
                    channels = format.channels,
                    "audio output stream rebuilt"
                );
            }
            Err(e) => {
                tracing::error!(error = %e, "could not rebuild the audio output stream");
                *self
                    .stream
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
            }
        }
    }
}

/// Rebuilds the stream every time `cpal` reports it needs one. One task for the engine's whole
/// lifetime; `Notify::notified` coalesces bursts of the same signal into one wakeup, so a
/// device flapping does not queue up a rebuild per error.
async fn watch_for_rebuild(engine: Arc<Engine>) {
    loop {
        engine.rebuild.notified().await;
        engine.rebuild_stream().await;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A deterministic stand-in for `Instant::now()`, so latency assertions do not depend on
    /// how fast the test machine happens to be.
    fn clock(nanos: &Cell<u64>) -> impl Fn() -> u64 + '_ {
        || nanos.get()
    }

    fn playing(transport: &Transport, generation: u64, gain: f32) {
        transport.generation.store(generation, Ordering::Relaxed);
        transport.gain_bits.store(gain.to_bits(), Ordering::Relaxed);
        transport.want_playing.store(true, Ordering::Relaxed);
        transport.finished_generation.store(0, Ordering::Relaxed);
        transport.first_audible_nanos.store(0, Ordering::Relaxed);
    }

    /// A fresh [`CallbackState`] that has already observed `transport`'s current generation,
    /// against an empty ring.
    ///
    /// In production the real-time callback runs continuously from the moment the stream opens,
    /// so by the time a decode-ahead task has anything to push, the callback has long since
    /// caught up to the current (at-rest) generation -- the clear-on-change branch only ever
    /// runs against an empty ring. A test that pre-fills the ring and only then calls `render`
    /// for the first time would see that clear discard real data, which is not a bug in
    /// `render`; it is a scenario `render` is never actually asked to handle. Priming with a
    /// zero-length buffer reproduces the real ordering: the generation transition (and its
    /// `consumer.clear()`) happens once, against nothing, before the caller pushes anything.
    fn primed(transport: &Transport, cons: &mut RingConsumer, sample_rate: u32) -> CallbackState {
        let mut state = CallbackState::new(sample_rate);
        render(&mut state, transport, cons, &mut [], 1, || 0);
        state
    }

    #[test]
    fn silence_before_anything_ever_played() {
        let transport = Transport::new();
        let (_prod, mut cons) = ring::ring(48_000, 1);
        let mut state = CallbackState::new(48_000);
        let mut out = [1.0f32, 1.0, 1.0];

        render(&mut state, &transport, &mut cons, &mut out, 1, || 0);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_attack_ramp_reaches_full_scale_and_then_holds() {
        let transport = Transport::new();
        let (mut prod, mut cons) = ring::ring(48_000, 1);
        playing(&transport, 1, 1.0);
        let mut state = primed(&transport, &mut cons, 48_000);
        prod.push_slice(&vec![1.0f32; 4096]);

        let attack_frames = (1.0 / state.attack_step).ceil() as usize;

        // One frame at a time so the ramp's own shape is visible to the assertions.
        let mut envelopes = Vec::new();
        for _ in 0..attack_frames + 10 {
            let mut out = [0.0f32];
            render(&mut state, &transport, &mut cons, &mut out, 1, || 0);
            envelopes.push(out[0]);
        }

        assert!(
            envelopes[0] > 0.0 && envelopes[0] < 1.0,
            "first frame is mid-ramp, not a step"
        );
        assert!(
            envelopes.windows(2).all(|w| w[1] + 1e-6 >= w[0]),
            "envelope must not decrease during attack: {envelopes:?}"
        );
        assert!(
            // A frame of slack: summing `attack_step` in `f32` does not necessarily cross
            // `1.0` in exactly `ceil(1.0 / attack_step)` additions.
            envelopes[attack_frames + 1..]
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-4),
            "envelope should be at unity gain once attack completes: {envelopes:?}"
        );
    }

    #[test]
    fn stopping_releases_rather_than_cutting_to_silence() {
        let transport = Transport::new();
        let (mut prod, mut cons) = ring::ring(48_000, 1);
        playing(&transport, 1, 1.0);
        let mut state = primed(&transport, &mut cons, 48_000);
        prod.push_slice(&vec![1.0f32; 8192]);

        // Run past the attack so we are in Sustain. A couple of frames of slack past the
        // arithmetic minimum: summing `attack_step` in `f32` does not necessarily cross `1.0`
        // in exactly `ceil(1.0 / attack_step)` additions.
        let attack_frames = (1.0 / state.attack_step).ceil() as usize + 2;
        let mut out = vec![0.0f32; attack_frames];
        render(&mut state, &transport, &mut cons, &mut out, 1, || 0);
        assert_eq!(state.phase, Phase::Sustain);

        transport.want_playing.store(false, Ordering::Relaxed);
        let release_frames = (1.0 / state.release_step).ceil() as usize;
        let mut samples = Vec::new();
        for _ in 0..release_frames + 5 {
            let mut out = [0.0f32];
            render(&mut state, &transport, &mut cons, &mut out, 1, || 0);
            samples.push(out[0]);
        }

        assert_eq!(state.phase, Phase::Idle);
        assert!(
            samples[0] > 0.0,
            "release starts from full volume, not silence"
        );
        assert!(
            samples.windows(2).all(|w| w[1] <= w[0] + 1e-6),
            "envelope must not increase during release: {samples:?}"
        );
        assert_eq!(*samples.last().unwrap(), 0.0);
    }

    #[test]
    fn a_finished_and_drained_clip_ends_on_its_own() {
        let transport = Transport::new();
        let (mut prod, mut cons) = ring::ring(48_000, 1);
        prod.push_slice(&[1.0, 1.0, 1.0, 1.0]);
        playing(&transport, 1, 1.0);
        transport.finished_generation.store(1, Ordering::Relaxed);

        let mut state = CallbackState::new(48_000);
        state.phase = Phase::Sustain; // skip the attack ramp for this assertion
        state.envelope = 1.0;
        state.generation = 1;

        // Drain the four queued samples, then run well past the release ramp.
        let mut out = vec![0.0f32; 4 + (1.0 / state.release_step).ceil() as usize + 5];
        render(&mut state, &transport, &mut cons, &mut out, 1, || 0);

        assert_eq!(
            state.phase,
            Phase::Idle,
            "a finished, empty ring must end playback"
        );
        assert_eq!(*out.last().unwrap(), 0.0);
    }

    #[test]
    fn an_underrun_is_silence_not_an_ending() {
        // Finished is false and the ring is empty: this is a decode-ahead task lagging, not a
        // clip that is over, and Sustain must not give up on it.
        let transport = Transport::new();
        let (_prod, mut cons) = ring::ring(48_000, 1);
        playing(&transport, 1, 1.0);

        let mut state = CallbackState::new(48_000);
        state.phase = Phase::Sustain;
        state.envelope = 1.0;
        state.generation = 1;

        let mut out = [1.0f32; 8];
        render(&mut state, &transport, &mut cons, &mut out, 1, || 0);

        assert_eq!(
            out, [0.0; 8],
            "an underrun must output silence, not stale data"
        );
        assert_eq!(
            state.phase,
            Phase::Sustain,
            "an underrun must not end playback"
        );
    }

    #[test]
    fn a_new_generation_clears_the_previous_ones_stale_audio() {
        let transport = Transport::new();
        let (mut prod, mut cons) = ring::ring(48_000, 1);
        // Generation 1's leftovers, still sitting in the ring.
        prod.push_slice(&[9.0, 9.0, 9.0]);

        let mut state = CallbackState::new(48_000);
        state.generation = 1;
        state.phase = Phase::Sustain;
        state.envelope = 1.0;

        // Generation 2 has started but has not pushed anything yet.
        transport.generation.store(2, Ordering::Relaxed);
        transport.want_playing.store(true, Ordering::Relaxed);

        let mut out = [0.0f32];
        render(&mut state, &transport, &mut cons, &mut out, 1, || 0);

        assert!(
            cons.is_empty(),
            "generation 1's leftover frames must be dropped, not played"
        );
        assert_eq!(
            state.phase,
            Phase::Attack,
            "a new generation always starts with an attack"
        );
    }

    #[test]
    fn latency_is_recorded_once_at_the_first_real_sample() {
        let transport = Transport::new();
        let (mut prod, mut cons) = ring::ring(48_000, 1);
        playing(&transport, 1, 1.0);
        let mut state = primed(&transport, &mut cons, 48_000);
        prod.push_slice(&[1.0, 1.0]);

        let nanos = Cell::new(1_000);

        let mut out = [0.0f32];
        render(
            &mut state,
            &transport,
            &mut cons,
            &mut out,
            1,
            clock(&nanos),
        );
        assert_eq!(transport.first_audible_nanos.load(Ordering::Relaxed), 1_000);

        // A later callback must not overwrite the first timestamp.
        nanos.set(9_000);
        render(
            &mut state,
            &transport,
            &mut cons,
            &mut out,
            1,
            clock(&nanos),
        );
        assert_eq!(transport.first_audible_nanos.load(Ordering::Relaxed), 1_000);
    }

    #[test]
    fn preferred_rate_picks_the_target_when_a_range_covers_it() {
        let ranges = vec![
            SupportedStreamConfigRange::new(
                2,
                8_000,
                16_000,
                cpal::SupportedBufferSize::Unknown,
                cpal::SampleFormat::F32,
            ),
            SupportedStreamConfigRange::new(
                2,
                44_100,
                96_000,
                cpal::SupportedBufferSize::Unknown,
                cpal::SampleFormat::F32,
            ),
        ];
        let picked = preferred_rate(ranges.into_iter()).expect("a covering range exists");
        assert_eq!(picked.sample_rate(), TARGET_SAMPLE_RATE);
        assert_eq!(picked.channels(), 2);
    }

    #[test]
    fn preferred_rate_is_none_when_nothing_covers_the_target() {
        let ranges = vec![SupportedStreamConfigRange::new(
            2,
            8_000,
            16_000,
            cpal::SupportedBufferSize::Unknown,
            cpal::SampleFormat::F32,
        )];
        assert!(preferred_rate(ranges.into_iter()).is_none());
    }
}
