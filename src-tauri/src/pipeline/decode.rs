//! Audio decode: `symphonia` probe and streaming decode, `rubato` resample.
//!
//! Hard stop at 10 s of 48 kHz output -- a 20-minute stem must not decode in full to be
//! embedded. Downmix to mono happens during decode, not after. Output buffers come from a
//! pool; allocating the 1.92 MB buffer per file is the difference between flat and
//! sawtooth RSS across a scan (`overview.md` §3.6).

use std::{
    collections::HashMap,
    fs::File,
    ops::Deref,
    path::Path,
    sync::{Arc, Mutex},
};

use rubato::{
    audioadapter_buffers::direct::InterleavedSlice, Async, FixedAsync, Resampler,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::{
    audio::GenericAudioBufferRef,
    codecs::audio::AudioDecoderOptions,
    codecs::CodecParameters,
    errors::Error as SymphoniaError,
    formats::{probe::Hint, FormatOptions, TrackType},
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};

/// Everything downstream -- the mel front-end, the DSP descriptors, the preview engine --
/// assumes this rate.
pub const TARGET_SAMPLE_RATE: u32 = 48_000;

/// CLAP's audio window (`overview.md` §3.2). Decoding past it is work whose output is
/// discarded.
pub const WINDOW_SECONDS: usize = 10;

/// The decode cap: 10 s at 48 kHz, mono.
pub const MAX_OUTPUT_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * WINDOW_SECONDS;

/// Resampler chunk size. `process_all_into_buffer` loops over the input in chunks of this
/// many frames; the value trades per-call overhead against the size of rubato's internal
/// scratch.
const RESAMPLE_CHUNK: usize = 1024;

/// The "middle profile" of `overview.md` §3.2, made concrete.
///
/// This feeds an ML model, not a mastering chain. `sinc_len` 64 with 128x oversampling and
/// a Blackman-Harris window puts the resampling error far below the quantization floor of
/// the f16 embedding it eventually becomes, at a quarter the cost of rubato's default
/// 256-tap profile -- which matters, because a majority of any real sample library is
/// 44.1 kHz and therefore takes this path.
///
/// `f_cutoff: None` is rubato's recommended setting: it picks the highest cutoff that keeps
/// aliasing under the window's sidelobe level for this filter length, which is a better
/// answer than any constant guessed here.
fn sinc_parameters() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 64,
        f_cutoff: None,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
    }
}

/// Why a file could not be turned into samples.
///
/// Each variant is a distinct thing to tell the user in the quarantine list, which is the
/// whole reason a decode failure is a row rather than a log line (`overview.md` §3.2).
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("could not open the file: {0}")]
    Open(#[source] std::io::Error),

    #[error("no format reader recognized the container: {0}")]
    UnknownFormat(#[source] SymphoniaError),

    #[error("the file contains no audio track")]
    NoAudioTrack,

    #[error("no decoder for this codec: {0}")]
    UnsupportedCodec(#[source] SymphoniaError),

    #[error("the track declares no sample rate")]
    UnknownSampleRate,

    #[error("the stream is damaged: {0}")]
    Damaged(#[source] SymphoniaError),

    #[error("the track decoded to no audio at all")]
    Empty,

    #[error("resampling {from} Hz to {TARGET_SAMPLE_RATE} Hz failed: {source}")]
    Resample {
        from: u32,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Mono 48 kHz audio plus what the container said about the source.
#[derive(Debug)]
pub struct Decoded {
    /// At most [`MAX_OUTPUT_SAMPLES`] mono samples at [`TARGET_SAMPLE_RATE`], from the
    /// pool.
    pub samples: PooledBuffer,
    /// Duration of the **whole** file, not of the decoded window. This is what the UI shows
    /// and what the duration filter sorts on, so truncating it to 10 s would be a lie.
    /// `None` when the container declares neither a frame count nor a duration.
    pub duration_ms: Option<i64>,
    /// Sample rate of the source, before resampling.
    pub source_sample_rate: u32,
    /// Channel count of the source, before downmix.
    pub channels: u16,
    /// Whether the decoder stopped at the window rather than at the end of the file.
    pub truncated: bool,
}

/// A recycled `Vec<f32>`, returned to its pool on drop.
///
/// The buffer is 1.92 MB. At 50,000 samples, allocating one per file is ~100 GB of churn
/// through the allocator for no reason (`overview.md` §3.6).
#[derive(Debug)]
pub struct PooledBuffer {
    // `Option` only so `Drop` can move the `Vec` out. It is `Some` for the whole visible
    // lifetime of the value.
    buf: Option<Vec<f32>>,
    pool: Arc<BufferPool>,
}

impl Deref for PooledBuffer {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        self.buf.as_ref().unwrap_or(&EMPTY)
    }
}

/// Stand-in for the unreachable `None` case in [`PooledBuffer::deref`], so the accessor
/// needs no `unwrap`.
static EMPTY: Vec<f32> = Vec::new();

impl PooledBuffer {
    /// The buffer, writable. How a stage fills a pooled buffer -- the decoder writes its
    /// window through this, and the mel stage writes its spectrogram through it.
    pub fn buffer_mut(&mut self) -> &mut Vec<f32> {
        self.buf.get_or_insert_with(Vec::new)
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.put(buf);
        }
    }
}

/// A fixed-capacity free list of decode buffers.
///
/// Bounded on purpose: an unbounded pool grows to the high-water mark of concurrent
/// decodes and never shrinks, which on a 16-core machine is 30 MB held forever. Past the
/// cap, returned buffers are dropped and the allocator reclaims them.
#[derive(Debug)]
pub struct BufferPool {
    free: Mutex<Vec<Vec<f32>>>,
    capacity: usize,
    max_held: usize,
}

impl BufferPool {
    /// A pool of buffers each able to hold `capacity` samples, holding at most `max_held`
    /// of them idle.
    pub fn new(capacity: usize, max_held: usize) -> Arc<Self> {
        Arc::new(Self {
            free: Mutex::new(Vec::new()),
            capacity,
            max_held,
        })
    }

    /// A pool sized for the decode stage: one full window per buffer, one idle buffer per
    /// core plus headroom for the handful in flight between stages.
    pub fn for_decode() -> Arc<Self> {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self::new(MAX_OUTPUT_SAMPLES, cores * 2)
    }

    /// Takes a buffer, empty and with the pool's capacity reserved.
    pub fn take(self: &Arc<Self>) -> PooledBuffer {
        let buf = self
            .free
            .lock()
            .ok()
            .and_then(|mut free| free.pop())
            .unwrap_or_else(|| Vec::with_capacity(self.capacity));

        PooledBuffer {
            buf: Some(buf),
            pool: Arc::clone(self),
        }
    }

    /// Number of buffers currently idle. Tests assert on this; nothing else should care.
    pub fn idle(&self) -> usize {
        self.free.lock().map(|f| f.len()).unwrap_or(0)
    }

    fn put(&self, mut buf: Vec<f32>) {
        // A buffer that grew past the pool's capacity is not worth holding: it came from a
        // pathological source rate and keeping it pins that memory for the whole scan.
        if buf.capacity() > self.capacity * 2 {
            return;
        }
        buf.clear();
        if let Ok(mut free) = self.free.lock() {
            if free.len() < self.max_held {
                free.push(buf);
            }
        }
    }
}

/// Per-worker decode state: the buffer pool, and one resampler per source rate.
///
/// Resamplers are cached because building one means computing a 64 x 128 sinc table, and a
/// library is overwhelmingly two or three distinct sample rates. Each worker holds its own
/// so the cache needs no lock.
#[derive(Debug)]
pub struct Decoder {
    pool: Arc<BufferPool>,
    resamplers: HashMap<u32, Async<f32>>,
    /// Interleaved f32 staging for one decoded packet, reused across packets and files.
    packet: Vec<f32>,
    /// Mono at the *source* rate, before resampling. Reused; not from the pool, because it
    /// is sized by the source rate rather than the target.
    mono: Vec<f32>,
}

impl Decoder {
    pub fn new(pool: Arc<BufferPool>) -> Self {
        Self {
            pool,
            resamplers: HashMap::new(),
            packet: Vec::new(),
            mono: Vec::new(),
        }
    }

    /// Decodes the first [`WINDOW_SECONDS`] of `path` to mono 48 kHz.
    ///
    /// `ext` is passed to the probe as a hint. It is a hint only: a `.wav` that is really a
    /// FLAC still decodes, because `symphonia` falls back to content sniffing.
    pub fn decode(&mut self, path: &Path, ext: &str) -> Result<Decoded, DecodeError> {
        let file = File::open(path).map_err(DecodeError::Open)?;
        let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());

        let mut hint = Hint::new();
        hint.with_extension(ext);

        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(DecodeError::UnknownFormat)?;

        let track = format
            .default_track(TrackType::Audio)
            .ok_or(DecodeError::NoAudioTrack)?;
        let track_id = track.id;

        let Some(CodecParameters::Audio(params)) = track.codec_params.clone() else {
            return Err(DecodeError::NoAudioTrack);
        };
        let source_rate = params.sample_rate.ok_or(DecodeError::UnknownSampleRate)?;
        if source_rate == 0 {
            return Err(DecodeError::UnknownSampleRate);
        }

        let duration_ms = track
            .num_frames
            .map(|frames| (frames as u128 * 1000 / u128::from(source_rate)) as i64)
            .or_else(|| {
                let time = track.time_base?.calc_duration(track.duration?)?;
                i64::try_from(time.as_millis()).ok()
            });

        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(DecodeError::UnsupportedCodec)?;

        // How many source-rate frames are needed to fill the window after resampling. Ceil,
        // plus a hair, so a rounding shortfall never leaves the last few milliseconds off.
        let needed = (MAX_OUTPUT_SAMPLES as u64 * u64::from(source_rate))
            .div_ceil(u64::from(TARGET_SAMPLE_RATE)) as usize
            + 1;

        self.mono.clear();
        self.mono.reserve(needed.min(MAX_OUTPUT_SAMPLES * 4));

        let mut channels = 0u16;
        let mut truncated = false;

        loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                // A truncated file is the common case in a sample library -- an interrupted
                // copy, a bad export. Whatever decoded before the tear is still usable, so
                // stop here rather than discarding it.
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break
                }
                Err(e) => return Err(DecodeError::Damaged(e)),
            };

            if packet.track_id != track_id {
                continue;
            }

            let audio = match decoder.decode(&packet) {
                Ok(audio) => audio,
                // One bad packet is not a bad file. `symphonia` documents `DecodeError` and
                // `IoError` as per-packet and recoverable; keep going.
                Err(SymphoniaError::DecodeError(_)) | Err(SymphoniaError::IoError(_)) => continue,
                Err(e) => return Err(DecodeError::Damaged(e)),
            };

            channels = audio.spec().channels().count() as u16;
            downmix_into(&audio, &mut self.packet, &mut self.mono);

            if self.mono.len() >= needed {
                self.mono.truncate(needed);
                truncated = true;
                break;
            }
        }

        if self.mono.is_empty() {
            return Err(DecodeError::Empty);
        }

        // Only a file whose content genuinely ran past the window counts as truncated: a
        // 3-second one-shot that filled `needed` exactly did not.
        truncated &= duration_ms.is_none_or(|d| d > (WINDOW_SECONDS as i64) * 1000);

        let mut samples = self.pool.take();
        if source_rate == TARGET_SAMPLE_RATE {
            samples.buffer_mut().extend_from_slice(&self.mono);
        } else {
            self.resample_into(source_rate, &mut samples)?;
        }
        samples.buffer_mut().truncate(MAX_OUTPUT_SAMPLES);

        Ok(Decoded {
            samples,
            duration_ms,
            source_sample_rate: source_rate,
            channels: channels.max(1),
            truncated,
        })
    }

    /// Resamples `self.mono` from `source_rate` into `out`.
    fn resample_into(
        &mut self,
        source_rate: u32,
        out: &mut PooledBuffer,
    ) -> Result<(), DecodeError> {
        let ratio = f64::from(TARGET_SAMPLE_RATE) / f64::from(source_rate);

        let resampler = match self.resamplers.entry(source_rate) {
            std::collections::hash_map::Entry::Occupied(e) => {
                let r = e.into_mut();
                // Cached resamplers carry the tail of the previous file in their delay
                // line. Not resetting bleeds one sample's decay into the next.
                r.reset();
                r
            }
            std::collections::hash_map::Entry::Vacant(e) => e.insert(
                Async::new_sinc(
                    ratio,
                    1.0,
                    &sinc_parameters(),
                    RESAMPLE_CHUNK,
                    1,
                    FixedAsync::Input,
                )
                .map_err(|e| DecodeError::Resample {
                    from: source_rate,
                    source: Box::new(e),
                })?,
            ),
        };

        let input_len = self.mono.len();
        let needed = resampler.process_all_needed_output_len(input_len);

        let buf = out.buffer_mut();
        buf.clear();
        buf.resize(needed, 0.0);

        let adapter = InterleavedSlice::new(self.mono.as_slice(), 1, input_len).map_err(|e| {
            DecodeError::Resample {
                from: source_rate,
                source: Box::new(e),
            }
        })?;
        let mut sink = InterleavedSlice::new_mut(buf.as_mut_slice(), 1, needed).map_err(|e| {
            DecodeError::Resample {
                from: source_rate,
                source: Box::new(e),
            }
        })?;

        let (_, produced) = resampler
            .process_all_into_buffer(&adapter, &mut sink, input_len, None)
            .map_err(|e| DecodeError::Resample {
                from: source_rate,
                source: Box::new(e),
            })?;

        // The buffer was sized for the worst case; everything past `produced` is padding.
        buf.truncate(produced);
        Ok(())
    }
}

/// Sums a decoded packet's channels into `mono`, scaling by the channel count.
///
/// Downmixing here rather than after the fact is what keeps peak memory to one mono buffer
/// instead of a stereo (or 5.1) one (`overview.md` §3.2). `scratch` is the caller's reused
/// interleaved staging buffer.
fn downmix_into(audio: &GenericAudioBufferRef<'_>, scratch: &mut Vec<f32>, mono: &mut Vec<f32>) {
    let channels = audio.spec().channels().count();
    if channels == 0 {
        return;
    }

    // One conversion point for every sample format symphonia can hand back, rather than a
    // ten-armed match repeated per codec. `copy_to_vec_interleaved` reuses the Vec's
    // allocation, so this is a memcpy-and-convert, not an allocation, after the first
    // packet.
    audio.copy_to_vec_interleaved::<f32>(scratch);

    if channels == 1 {
        mono.extend_from_slice(scratch);
        return;
    }

    let scale = 1.0 / channels as f32;
    mono.extend(
        scratch
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() * scale),
    );
}
