//! DSP feature extraction: `realfft` framing plus the descriptor set.
//!
//! Peak, RMS, LUFS, spectral centroid, spectral flatness, ZCR, onset density, BPM and key
//! all come out of a single pass over the same frames -- the mel front-end that feeds
//! inference is computed here too, and Phase 3's parity gate exists because a front-end
//! that quietly disagrees with the ONNX graph produces embeddings that are wrong in a way
//! nothing downstream can detect.
//!
//! **Framing is the CLAP contract** (`overview.md` §3.3): 1024-point Hann window, hop 480,
//! at 48 kHz. The DSP descriptors below are computed from those frames rather than from
//! their own analysis because the transform is already paid for. Phase 3 adds the mel
//! filterbank on top of the same [`Analyzer::power`] output; nothing here has to move for
//! it, which is the point of paying for the framing now.
//!
//! **Key is the one exception, and it has to be.** A 1024-point window at 48 kHz has 46.9 Hz
//! bins; a semitone at middle C is 15 Hz. Pitch is simply not resolvable in those frames in
//! the register where pitch lives -- run a C major triad through them and the answer comes
//! back C# minor, confidently. `overview.md` §3.3 shares the FFT to avoid paying for a
//! transform twice, which is an efficiency argument; it is not an argument for populating a
//! column with numbers that are wrong. So chroma gets its own [`CHROMA_FRAME_SIZE`]-point
//! pass over the same buffer, at roughly the cost of the main one and a fraction of the cost
//! of the decode that precedes it.

use std::sync::Arc;

use realfft::{num_complex::Complex32, RealFftPlanner, RealToComplex};

use super::decode::TARGET_SAMPLE_RATE;
use crate::db::SampleFeatures;

/// FFT size, in samples (`overview.md` §3.3).
pub const FRAME_SIZE: usize = 1024;

/// Hop between frames, in samples. 10 ms at 48 kHz.
pub const HOP_SIZE: usize = 480;

/// Number of magnitude bins a frame produces.
pub const NUM_BINS: usize = FRAME_SIZE / 2 + 1;

/// Frames per second of audio, which is what turns an onset count into a density and a
/// lag into a tempo.
const FRAME_RATE: f32 = TARGET_SAMPLE_RATE as f32 / HOP_SIZE as f32;

/// Floor for anything that becomes a decibel or a logarithm.
///
/// -200 dBFS: far below 24-bit silence, so it never truncates real signal, and finite, so
/// a digitally silent file yields a number instead of `-inf` and a `NULL` column.
const EPSILON: f32 = 1e-10;

/// The tempo range the BPM estimator searches.
///
/// Beyond it, autocorrelation of a ten-second window has too few periods to be worth
/// believing, and octave errors dominate. A drum loop outside 60-180 gets reported at its
/// half- or double-time, which is the conventional failure and better than a NULL.
const MIN_BPM: f32 = 60.0;
const MAX_BPM: f32 = 180.0;

/// FFT size for the chroma pass.
///
/// 8192 points is 5.86 Hz per bin, which is narrower than a semitone (5.95 Hz) at 100 Hz --
/// so every pitch from G2 up is resolvable, and below that a bass note's second harmonic
/// carries the same pitch class anyway. The window is 170 ms, which is long enough to hold a
/// note and short enough not to smear a chord change across the whole analysis.
pub const CHROMA_FRAME_SIZE: usize = 8192;

/// Hop for the chroma pass: 50% overlap, ~85 ms.
pub const CHROMA_HOP_SIZE: usize = CHROMA_FRAME_SIZE / 2;

/// The band the chroma estimate reads.
///
/// Below 80 Hz the transform can no longer separate semitones; above ~5 kHz the partials of
/// a pitched note are drowned by noise and cymbals, and including them flattens the profile
/// on every percussive sample.
const CHROMA_MIN_HZ: f32 = 80.0;
const CHROMA_MAX_HZ: f32 = 5000.0;

/// How far a spectral peak must stand above the frame's mean power to be treated as a
/// partial rather than as noise. Broadband noise has local maxima everywhere; without this
/// they scatter across all twelve pitch classes and every sample acquires a key.
const PEAK_PROMINENCE: f32 = 8.0;

/// Fraction of chroma frames that must contain at least one resolved partial before the
/// sample is considered pitched at all.
///
/// This is the gate that keeps drums out of the key column, and it is a far better one than
/// a correlation threshold. Noise correlates *well* with some key profile by accident: a
/// handful of spurious peaks land in two pitch classes, the profile looks concentrated, and
/// white noise comes back as G# minor at 0.84. What noise cannot fake is persistence. A
/// pitched sample produces partials in essentially every frame; measured across white noise,
/// filtered noise, and a pitch-swept kick, none clears 25%.
const CHROMA_COVERAGE_FLOOR: f32 = 0.5;

/// Share of the chroma a pitch class needs before it counts as present.
const KEY_CLASS_SHARE: f32 = 0.10;

/// Distinct pitch classes needed before major and minor are distinguishable.
///
/// Below three, there is a note or an interval but not a key, and the winning mode is a coin
/// flip -- so `key_root` is reported and `key_mode` is left `NULL`. This is the case a bass
/// one-shot falls into, and getting it right is why the schema makes the two columns
/// separately nullable.
const KEY_MIN_CLASSES: usize = 3;

/// Reference frequency for chroma binning: A4 = 440 Hz, pitch class 9.
const A4_HZ: f32 = 440.0;

/// Correlation a key profile must reach before a mode is claimed. A clean triad correlates
/// around 0.85; below this the chroma is pitched but not diatonic, and only the root is
/// reported.
const KEY_CORRELATION_FLOOR: f32 = 0.5;

/// Krumhansl-Kessler major and minor key profiles, the standard perceptual weights for
/// correlating a chroma vector against the 24 keys.
const MAJOR_PROFILE: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const MINOR_PROFILE: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Reusable analysis state: one FFT plan, one window, and the scratch every frame needs.
///
/// Built once per decode worker. Planning a 1024-point real FFT is not free, and neither is
/// allocating four buffers per file across a 50,000-file scan (`overview.md` §3.6).
pub struct Analyzer {
    fft: Arc<dyn RealToComplex<f32>>,
    /// Periodic Hann window, precomputed.
    window: Vec<f32>,
    /// The chroma pass: its own plan, window, and buffers. See the module docs for why it
    /// cannot share the frames above.
    chroma_fft: Arc<dyn RealToComplex<f32>>,
    chroma_window: Vec<f32>,
    chroma_frame: Vec<f32>,
    chroma_spectrum: Vec<Complex32>,
    chroma_scratch: Vec<Complex32>,
    chroma_power: Vec<f32>,
    /// The windowed frame handed to the FFT. `process` mutates its input, so it cannot be a
    /// view into the caller's samples.
    frame: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    /// Power spectrum of the frame just transformed. Phase 3's mel filterbank reads this.
    power: Vec<f32>,
    /// Magnitude of the previous frame, for spectral flux.
    previous: Vec<f32>,
    /// The onset detection function, one value per frame.
    flux: Vec<f32>,
}

// `RealToComplex` is not `Debug`, and `missing_debug_implementations` is a warning this
// crate turns on. A manual impl beats either suppressing the lint or leaking the planner's
// internals.
impl std::fmt::Debug for Analyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Analyzer")
            .field("frame_size", &FRAME_SIZE)
            .field("hop_size", &HOP_SIZE)
            .finish_non_exhaustive()
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    pub fn new() -> Self {
        // One planner for both sizes: `rustfft` caches twiddle factors inside it, so
        // planning the second transform from the same planner shares what it can.
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME_SIZE);
        let chroma_fft = planner.plan_fft_forward(CHROMA_FRAME_SIZE);

        Self {
            frame: fft.make_input_vec(),
            spectrum: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window: hann_window(FRAME_SIZE),
            power: vec![0.0; NUM_BINS],
            previous: vec![0.0; NUM_BINS],
            flux: Vec::new(),
            chroma_frame: chroma_fft.make_input_vec(),
            chroma_spectrum: chroma_fft.make_output_vec(),
            chroma_scratch: chroma_fft.make_scratch_vec(),
            chroma_power: vec![0.0; CHROMA_FRAME_SIZE / 2 + 1],
            chroma_window: hann_window(CHROMA_FRAME_SIZE),
            chroma_fft,
        }
    }

    /// Every DSP descriptor for one decoded buffer, in a single pass over its frames.
    ///
    /// `samples` is mono at [`TARGET_SAMPLE_RATE`]. A buffer shorter than one frame still
    /// produces the time-domain descriptors -- a 5 ms transient is a legitimate sample and
    /// deserves a peak and an RMS -- with the spectral ones left `None`.
    pub fn analyze(&mut self, samples: &[f32]) -> SampleFeatures {
        let mut features = SampleFeatures {
            peak_db: Some(to_db(peak(samples))),
            rms_db: Some(to_db(rms(samples))),
            zero_crossing: Some(zero_crossing_rate(samples)),
            lufs_integrated: loudness(samples),
            ..SampleFeatures::default()
        };

        if samples.len() < FRAME_SIZE {
            return features;
        }

        // Accumulators over frames. Centroid and flatness are averaged with each frame
        // weighted by its energy: an unweighted mean lets the silent tail of a one-shot,
        // where the "centroid" is whatever the noise floor happens to be, drag the answer
        // toward nonsense.
        let mut centroid_sum = 0.0f64;
        let mut flatness_sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        self.flux.clear();
        self.previous.fill(0.0);
        let mut first_frame = true;

        for start in (0..=samples.len() - FRAME_SIZE).step_by(HOP_SIZE) {
            self.transform(&samples[start..start + FRAME_SIZE]);

            let energy: f64 = self.power.iter().map(|&p| f64::from(p)).sum();
            if energy > f64::from(EPSILON) {
                centroid_sum += f64::from(spectral_centroid(&self.power)) * energy;
                flatness_sum += f64::from(spectral_flatness(&self.power)) * energy;
                weight_sum += energy;
            }

            // Spectral flux over magnitudes, half-wave rectified: only bins that *gained*
            // energy count, because an onset is energy appearing, and a note ending would
            // otherwise register as one.
            let mut flux = 0.0f32;
            for (bin, &p) in self.power.iter().enumerate() {
                let magnitude = p.sqrt();
                if !first_frame {
                    flux += (magnitude - self.previous[bin]).max(0.0);
                }
                self.previous[bin] = magnitude;
            }
            self.flux.push(flux);
            first_frame = false;
        }

        if weight_sum > 0.0 {
            features.spectral_centroid = Some((centroid_sum / weight_sum) as f32);
            features.spectral_flatness = Some((flatness_sum / weight_sum) as f32);
        }

        let onsets = pick_onsets(&self.flux);
        let seconds = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
        if seconds > 0.0 {
            features.onset_density = Some(onsets.len() as f32 / seconds);
        }

        if let Some((bpm, confidence)) = estimate_tempo(&self.flux) {
            features.bpm = Some(bpm);
            features.bpm_confidence = Some(confidence);
        }

        let (chroma, coverage) = self.chroma(samples);
        if let Some(key) = estimate_key(&chroma, coverage) {
            features.key_root = Some(key.root);
            features.key_mode = key.mode;
            features.key_confidence = Some(key.confidence);
        }

        features
    }

    /// The power spectrum of the most recently transformed frame.
    ///
    /// Exposed for Phase 3's mel filterbank, which is a weighted sum over exactly these
    /// bins and has no business recomputing the transform.
    pub fn power(&self) -> &[f32] {
        &self.power
    }

    /// The pitch-class profile of the whole buffer, from its own longer transform, and the
    /// fraction of frames that contributed a resolved partial to it.
    fn chroma(&mut self, samples: &[f32]) -> ([f32; 12], f32) {
        let mut chroma = [0.0f32; 12];
        let (mut frames, mut with_peaks) = (0usize, 0usize);
        if samples.len() < CHROMA_FRAME_SIZE {
            return (chroma, 0.0);
        }

        for start in (0..=samples.len() - CHROMA_FRAME_SIZE).step_by(CHROMA_HOP_SIZE) {
            for (dst, (&src, &w)) in self
                .chroma_frame
                .iter_mut()
                .zip(samples[start..].iter().zip(self.chroma_window.iter()))
            {
                *dst = src * w;
            }

            frames += 1;
            if self
                .chroma_fft
                .process_with_scratch(
                    &mut self.chroma_frame,
                    &mut self.chroma_spectrum,
                    &mut self.chroma_scratch,
                )
                .is_err()
            {
                continue;
            }

            for (dst, c) in self
                .chroma_power
                .iter_mut()
                .zip(self.chroma_spectrum.iter())
            {
                *dst = c.norm_sqr();
            }
            if accumulate_chroma(&self.chroma_power, CHROMA_FRAME_SIZE, &mut chroma) > 0 {
                with_peaks += 1;
            }
        }

        let coverage = if frames == 0 {
            0.0
        } else {
            with_peaks as f32 / frames as f32
        };
        (chroma, coverage)
    }

    /// Windows one frame, transforms it, and fills [`Self::power`].
    fn transform(&mut self, samples: &[f32]) {
        for (dst, (&src, &w)) in self
            .frame
            .iter_mut()
            .zip(samples.iter().zip(self.window.iter()))
        {
            *dst = src * w;
        }

        // The only failure mode is a length mismatch between the buffers and the plan, and
        // all four came from the plan itself. Dropping the frame keeps this off the
        // `Result` path of every caller.
        if self
            .fft
            .process_with_scratch(&mut self.frame, &mut self.spectrum, &mut self.scratch)
            .is_err()
        {
            self.power.fill(0.0);
            return;
        }

        for (dst, c) in self.power.iter_mut().zip(self.spectrum.iter()) {
            *dst = c.norm_sqr();
        }
    }
}

/// Periodic (not symmetric) Hann: the correct choice for STFT analysis, where consecutive
/// frames must sum to a constant.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = std::f32::consts::TAU * i as f32 / n as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()))
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Full-scale decibels, floored rather than allowed to reach `-inf`.
fn to_db(amplitude: f32) -> f32 {
    20.0 * amplitude.max(EPSILON).log10()
}

/// Fraction of adjacent sample pairs that change sign.
///
/// Normalized to [0, 1] rather than reported as crossings per second, so the filter slider
/// in Phase 9 has a fixed range that does not depend on the sample rate.
fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

/// EBU R 128 integrated loudness, in LUFS.
///
/// `None` for a buffer shorter than one 400 ms gating block, or one whose every block gates
/// out as silence -- both of which say "this number does not exist for this file", which is
/// a different statement from -70 LUFS.
fn loudness(samples: &[f32]) -> Option<f32> {
    let mut meter = ebur128::EbuR128::new(1, TARGET_SAMPLE_RATE, ebur128::Mode::I).ok()?;
    meter.add_frames_f32(samples).ok()?;
    let lufs = meter.loudness_global().ok()?;
    lufs.is_finite().then_some(lufs as f32)
}

/// The power-weighted mean frequency of one frame, in Hz.
fn spectral_centroid(power: &[f32]) -> f32 {
    let mut weighted = 0.0f64;
    let mut total = 0.0f64;
    for (bin, &p) in power.iter().enumerate() {
        weighted += f64::from(bin_hz(bin)) * f64::from(p);
        total += f64::from(p);
    }
    if total <= 0.0 {
        return 0.0;
    }
    (weighted / total) as f32
}

/// Geometric mean over arithmetic mean of the power spectrum: 1 for white noise, near 0 for
/// a pure tone. The geometric mean is computed in the log domain because the direct product
/// of 513 small numbers underflows f32 long before it finishes.
fn spectral_flatness(power: &[f32]) -> f32 {
    if power.is_empty() {
        return 0.0;
    }
    let mut log_sum = 0.0f64;
    let mut sum = 0.0f64;
    for &p in power {
        let p = f64::from(p.max(EPSILON));
        log_sum += p.ln();
        sum += p;
    }
    let n = power.len() as f64;
    let geometric = (log_sum / n).exp();
    let arithmetic = sum / n;
    if arithmetic <= 0.0 {
        return 0.0;
    }
    (geometric / arithmetic).clamp(0.0, 1.0) as f32
}

/// Centre frequency of an FFT bin.
fn bin_hz(bin: usize) -> f32 {
    bin as f32 * TARGET_SAMPLE_RATE as f32 / FRAME_SIZE as f32
}

/// Folds one frame's spectral peaks into a 12-bin pitch-class profile.
///
/// **Peaks, not bins.** A 1024-point FFT at 48 kHz has 46.9 Hz bins, while a semitone at
/// middle C is 15 Hz -- so assigning each bin's power to the pitch class of its centre
/// frequency is meaningless in exactly the register where pitch lives, and reports C major
/// triads as C# minor. What is resolvable is the *location* of a peak: a windowed sinusoid
/// spreads over three bins in a known shape, and interpolating that shape recovers its true
/// frequency to a fraction of a bin.
///
/// Reading peaks instead of every bin also fixes the noise case for free. Broadband noise
/// has no partials that clear [`PEAK_PROMINENCE`], so it contributes almost nothing and its
/// chroma stays flat -- which is what makes an unpitched sample report no key rather than
/// an arbitrary one.
fn accumulate_chroma(power: &[f32], frame_size: usize, chroma: &mut [f32; 12]) -> usize {
    let mut peaks = 0usize;
    let hz_per_bin = TARGET_SAMPLE_RATE as f32 / frame_size as f32;

    let mean = power.iter().map(|&p| f64::from(p)).sum::<f64>() / power.len() as f64;
    let threshold = (mean * f64::from(PEAK_PROMINENCE)) as f32;

    let lo = ((CHROMA_MIN_HZ / hz_per_bin).ceil() as usize).max(1);
    let hi = ((CHROMA_MAX_HZ / hz_per_bin).floor() as usize).min(power.len().saturating_sub(2));

    for bin in lo..=hi {
        let (a, b, c) = (power[bin - 1], power[bin], power[bin + 1]);
        if b <= a || b < c || b < threshold {
            continue;
        }

        let Some(offset) = interpolate_peak(a, b, c) else {
            continue;
        };
        let hz = (bin as f32 + offset) * hz_per_bin;

        // 12-TET: pitch class is the semitone distance from A4, modulo an octave, offset so
        // that index 0 is C.
        let semitones = 12.0 * (hz / A4_HZ).log2();
        let class = (semitones.round() as i32 + 9).rem_euclid(12) as usize;
        chroma[class] += a + b + c;
        peaks += 1;
    }
    peaks
}

/// Refines a three-bin peak to a sub-bin offset by fitting a parabola to the log magnitudes.
///
/// The standard estimator: for a Hann-windowed sinusoid the log-magnitude around the peak is
/// close to parabolic, and its vertex sits at the real frequency. `None` when the fit says
/// the vertex is more than half a bin away, which means the shape was not a resolved partial
/// in the first place.
fn interpolate_peak(a: f32, b: f32, c: f32) -> Option<f32> {
    // Half of each log-power is the log-magnitude; the constant factor cancels in the
    // vertex formula, so the halving is left out.
    let (a, b, c) = (
        a.max(EPSILON).ln(),
        b.max(EPSILON).ln(),
        c.max(EPSILON).ln(),
    );

    let curvature = a - 2.0 * b + c;
    if curvature.abs() <= f32::EPSILON {
        return None;
    }

    let delta = 0.5 * (a - c) / curvature;
    (delta.abs() <= 0.5).then_some(delta)
}

/// Peak-picks the onset detection function.
///
/// An onset is a local maximum that stands above a running mean of the surrounding flux by
/// a margin. The adaptive threshold is what makes one function work on both a sparse
/// one-shot and a dense breakbeat: a fixed threshold tuned for one reports nothing on the
/// other.
fn pick_onsets(flux: &[f32]) -> Vec<usize> {
    /// Frames of context either side of the candidate for the running mean. ~100 ms.
    const WINDOW: usize = 10;
    /// How far above local mean flux a peak must rise.
    const MARGIN: f32 = 1.5;
    /// Minimum gap between onsets, in frames. ~50 ms -- below that it is one transient
    /// being counted twice.
    const MIN_GAP: usize = 5;

    let mut onsets = Vec::new();
    if flux.len() < 3 {
        return onsets;
    }

    let mean = flux.iter().sum::<f32>() / flux.len() as f32;
    if mean <= EPSILON {
        return onsets;
    }

    let mut last = 0usize;
    for i in 1..flux.len() - 1 {
        let value = flux[i];
        if value <= flux[i - 1] || value < flux[i + 1] {
            continue;
        }

        let lo = i.saturating_sub(WINDOW);
        let hi = (i + WINDOW + 1).min(flux.len());
        let local: f32 = flux[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;

        if value > local * MARGIN
            && value > mean * 0.5
            && (onsets.is_empty() || i - last >= MIN_GAP)
        {
            onsets.push(i);
            last = i;
        }
    }

    onsets
}

/// Estimates tempo by autocorrelating the onset detection function.
///
/// Returns `None` when the function is too flat to have a period -- a sustained pad has no
/// tempo, and inventing one for it would populate a BPM filter with noise.
///
/// Confidence is the winning lag's correlation relative to the mean over the searched
/// range, squashed into [0, 1]. It is an ordering, not a probability: it exists so the UI
/// can grey out a guess, and so Phase 9's BPM filter can offer "confident matches only".
fn estimate_tempo(flux: &[f32]) -> Option<(f32, f32)> {
    let min_lag = (FRAME_RATE * 60.0 / MAX_BPM).round() as usize;
    let max_lag = (FRAME_RATE * 60.0 / MIN_BPM).round() as usize;
    if flux.len() < max_lag * 2 || min_lag == 0 {
        return None;
    }

    // Autocorrelating the raw function finds the DC component, not the beat. Subtracting
    // the mean makes the correlation measure periodicity of the *fluctuation*.
    let mean = flux.iter().sum::<f32>() / flux.len() as f32;
    let centered: Vec<f32> = flux.iter().map(|&f| f - mean).collect();

    let energy: f32 = centered.iter().map(|c| c * c).sum();
    if energy <= EPSILON {
        return None;
    }

    let mut best = (0usize, f32::MIN);
    let mut total = 0.0f32;
    let mut count = 0usize;

    for lag in min_lag..=max_lag {
        let correlation: f32 = centered[lag..]
            .iter()
            .zip(centered.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / energy;

        total += correlation;
        count += 1;
        if correlation > best.1 {
            best = (lag, correlation);
        }
    }

    let (lag, peak) = best;
    if lag == 0 || peak <= 0.0 {
        return None;
    }

    let average = total / count as f32;
    let confidence = ((peak - average) / peak.abs().max(EPSILON)).clamp(0.0, 1.0);
    let bpm = FRAME_RATE * 60.0 / lag as f32;

    Some((bpm, confidence))
}

/// What the chroma pass concluded about pitch.
#[derive(Debug, Clone, Copy, PartialEq)]
struct KeyEstimate {
    /// Pitch class, 0 = C.
    root: i32,
    /// 0 minor, 1 major. `None` when there is a root but not enough distinct pitch classes
    /// to tell the two apart -- a single bass note, an interval, a tuned percussion hit.
    mode: Option<i32>,
    confidence: f32,
}

/// Correlates a chroma vector against all 24 Krumhansl-Kessler key profiles.
///
/// `None` when the sample is unpitched, which most of a drum library is -- that is the
/// "NULL if unpitched" case the schema comment describes.
fn estimate_key(chroma: &[f32; 12], coverage: f32) -> Option<KeyEstimate> {
    if coverage < CHROMA_COVERAGE_FLOOR {
        return None;
    }

    let total: f32 = chroma.iter().sum();
    if total <= EPSILON {
        return None;
    }
    let normalized: Vec<f32> = chroma.iter().map(|&c| c / total).collect();

    let present = normalized.iter().filter(|&&c| c >= KEY_CLASS_SHARE).count();
    if present >= KEY_MIN_CLASSES {
        if let Some((root, mode, r)) = best_key_profile(&normalized) {
            if r >= KEY_CORRELATION_FLOOR {
                return Some(KeyEstimate {
                    root,
                    mode: Some(mode),
                    confidence: r,
                });
            }
        }
    }

    // Pitched, but not diatonic enough to name a mode. The dominant pitch class is still a
    // real and useful answer: it is what a bass one-shot's root note is.
    let (root, share) = normalized
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))?;

    Some(KeyEstimate {
        root: root as i32,
        mode: None,
        confidence: *share,
    })
}

/// The best-correlating of the 24 profiles, as `(root, mode, correlation)`.
fn best_key_profile(normalized: &[f32]) -> Option<(i32, i32, f32)> {
    let mut best: Option<(i32, i32, f32)> = None;
    for root in 0..12 {
        for (mode, profile) in [(0, &MINOR_PROFILE), (1, &MAJOR_PROFILE)] {
            // The profile is written with the tonic first, so rotating it by `root` lines it
            // up with a chroma vector indexed from C.
            let rotated: Vec<f32> = (0..12)
                .map(|i| profile[(i + 12 - root as usize) % 12])
                .collect();
            let r = pearson(normalized, &rotated);
            if best.is_none_or(|(_, _, b)| r > b) {
                best = Some((root, mode, r));
            }
        }
    }
    best
}

/// Pearson correlation of two equal-length vectors, 0 when either is constant.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }

    let denom = (var_a * var_b).sqrt();
    if denom <= EPSILON {
        return 0.0;
    }
    cov / denom
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// A sine at `hz` and amplitude `amp`, `seconds` long.
    fn sine(hz: f32, amp: f32, seconds: f32) -> Vec<f32> {
        let n = (TARGET_SAMPLE_RATE as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / TARGET_SAMPLE_RATE as f32;
                amp * (std::f32::consts::TAU * hz * t).sin()
            })
            .collect()
    }

    #[test]
    fn peak_and_rms_match_the_arithmetic_for_a_sine() {
        let s = sine(1000.0, 0.5, 1.0);
        let f = Analyzer::new().analyze(&s);

        // A sine's peak is its amplitude and its RMS is amplitude / sqrt(2).
        assert!((f.peak_db.unwrap() - to_db(0.5)).abs() < 0.1);
        assert!((f.rms_db.unwrap() - to_db(0.5 / 2f32.sqrt())).abs() < 0.1);
    }

    #[test]
    fn the_centroid_lands_on_the_tone() {
        let f = Analyzer::new().analyze(&sine(1000.0, 0.8, 1.0));
        let centroid = f.spectral_centroid.unwrap();
        assert!(
            (centroid - 1000.0).abs() < 100.0,
            "centroid of a 1 kHz sine was {centroid} Hz"
        );
    }

    /// Flatness is the one descriptor with an unambiguous ordering: noise is flat, a tone is
    /// not. Asserting the ordering rather than absolute values keeps the test from encoding
    /// the window's leakage as if it were a requirement.
    #[test]
    fn flatness_separates_a_tone_from_noise() {
        let tone = Analyzer::new().analyze(&sine(1000.0, 0.8, 1.0));

        let mut rng = 0x243f_6a88_85a3_08d3u64;
        let noise: Vec<f32> = (0..TARGET_SAMPLE_RATE as usize)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect();
        let noise = Analyzer::new().analyze(&noise);

        assert!(
            noise.spectral_flatness.unwrap() > tone.spectral_flatness.unwrap() * 10.0,
            "noise {:?} was not decisively flatter than a tone {:?}",
            noise.spectral_flatness,
            tone.spectral_flatness
        );
    }

    #[test]
    fn zero_crossing_rate_tracks_frequency() {
        let low = Analyzer::new().analyze(&sine(100.0, 0.8, 1.0));
        let high = Analyzer::new().analyze(&sine(8000.0, 0.8, 1.0));
        assert!(high.zero_crossing.unwrap() > low.zero_crossing.unwrap() * 10.0);

        // 2 crossings per cycle: 100 Hz over 48 kHz is 200/48000.
        let expected = 200.0 / TARGET_SAMPLE_RATE as f32;
        assert!((low.zero_crossing.unwrap() - expected).abs() < expected * 0.05);
    }

    /// Digital silence must produce numbers, not `-inf` and not `NaN`. Every one of these
    /// ends up in a REAL column that the filter UI reads.
    #[test]
    fn silence_produces_finite_descriptors() {
        let f = Analyzer::new().analyze(&vec![0.0; TARGET_SAMPLE_RATE as usize]);

        assert!(f.peak_db.unwrap().is_finite());
        assert!(f.rms_db.unwrap().is_finite());
        assert_eq!(f.zero_crossing, Some(0.0));
        for value in [f.spectral_centroid, f.spectral_flatness, f.bpm] {
            assert!(
                value.is_none_or(|v| v.is_finite()),
                "{value:?} was not finite"
            );
        }
    }

    /// A buffer shorter than one FFT frame still has a peak and an RMS.
    #[test]
    fn a_sub_frame_buffer_still_yields_time_domain_descriptors() {
        let f = Analyzer::new().analyze(&sine(1000.0, 0.5, 0.005));

        assert!(f.peak_db.is_some());
        assert!(f.rms_db.is_some());
        assert!(f.spectral_centroid.is_none());
        assert!(f.bpm.is_none());
    }

    #[test]
    fn an_empty_buffer_does_not_panic() {
        let f = Analyzer::new().analyze(&[]);
        assert_eq!(f.zero_crossing, Some(0.0));
        assert!(f.peak_db.unwrap().is_finite());
    }

    /// A click train at a known tempo is the only BPM test worth writing: real music has
    /// swing, ambiguity, and octave errors, and pinning this estimator's behavior on it
    /// would be pinning its bugs.
    #[test]
    fn tempo_estimation_finds_a_click_train() {
        let bpm = 120.0f32;
        let period = (TARGET_SAMPLE_RATE as f32 * 60.0 / bpm) as usize;
        let mut samples = vec![0.0f32; TARGET_SAMPLE_RATE as usize * 8];
        for start in (0..samples.len()).step_by(period) {
            // A short decaying burst, so the click has spectral content across the band
            // rather than being a single-sample impulse.
            for i in 0..480.min(samples.len() - start) {
                let decay = 1.0 - (i as f32 / 480.0);
                samples[start + i] = decay * decay * if i % 3 == 0 { 0.9 } else { -0.9 };
            }
        }

        let f = Analyzer::new().analyze(&samples);
        let estimated = f.bpm.expect("a click train has a tempo");

        // Accept the octave errors that any autocorrelation tempo estimator makes; what is
        // being tested is that the period is found at all.
        let ratios = [
            estimated / bpm,
            estimated / (bpm / 2.0),
            estimated / (bpm * 2.0),
        ];
        assert!(
            ratios.iter().any(|r| (r - 1.0).abs() < 0.06),
            "estimated {estimated} BPM for a {bpm} BPM click train"
        );
    }

    #[test]
    fn onset_density_rises_with_the_number_of_transients() {
        let make = |clicks: usize| {
            let mut samples = vec![0.0f32; TARGET_SAMPLE_RATE as usize * 2];
            let step = samples.len() / clicks;
            for start in (0..samples.len()).step_by(step) {
                for i in 0..240.min(samples.len() - start) {
                    let decay = 1.0 - (i as f32 / 240.0);
                    samples[start + i] = decay * if i % 2 == 0 { 0.9 } else { -0.9 };
                }
            }
            Analyzer::new().analyze(&samples).onset_density.unwrap()
        };

        assert!(make(16) > make(4));
    }

    /// A C major triad must report C major. This is the case the estimator exists for; the
    /// interesting negative cases -- percussion reporting no key, a single note reporting no
    /// mode -- follow it.
    #[test]
    fn a_major_triad_reports_its_root_and_mode() {
        // C4, E4, G4.
        let mut samples = sine(261.63, 0.3, 3.0);
        for (dst, src) in samples.iter_mut().zip(sine(329.63, 0.3, 3.0)) {
            *dst += src;
        }
        for (dst, src) in samples.iter_mut().zip(sine(392.0, 0.3, 3.0)) {
            *dst += src;
        }

        let f = Analyzer::new().analyze(&samples);
        assert_eq!(f.key_root, Some(0), "expected C");
        assert_eq!(f.key_mode, Some(1), "expected major");
        assert!(f.key_confidence.unwrap() >= KEY_CORRELATION_FLOOR);
    }

    #[test]
    fn a_minor_triad_reports_minor() {
        let mut samples = sine(220.0, 0.3, 3.0);
        for (dst, src) in samples.iter_mut().zip(sine(261.63, 0.3, 3.0)) {
            *dst += src;
        }
        for (dst, src) in samples.iter_mut().zip(sine(329.63, 0.3, 3.0)) {
            *dst += src;
        }

        let f = Analyzer::new().analyze(&samples);
        assert_eq!(f.key_root, Some(9), "expected A");
        assert_eq!(f.key_mode, Some(0), "expected minor");
    }

    /// A single sustained note has a root and no mode. Reporting "A major" for a bass
    /// one-shot would put half a library's bass hits on the wrong side of a key filter.
    #[test]
    fn a_single_note_reports_a_root_but_no_mode() {
        // A sawtooth at A2, built from its first twenty harmonics.
        let n = TARGET_SAMPLE_RATE as usize * 3;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / TARGET_SAMPLE_RATE as f32;
                (1..=20)
                    .map(|k| (std::f32::consts::TAU * 110.0 * k as f32 * t).sin() / k as f32)
                    .sum::<f32>()
                    * 0.2
            })
            .collect();

        let f = Analyzer::new().analyze(&samples);
        assert_eq!(f.key_root, Some(9), "expected A");
        assert_eq!(f.key_mode, None, "one note cannot decide major or minor");
    }

    /// The gate that keeps a drum library out of the key column. Three seeds, because a
    /// single noise buffer that happens to pass tells you nothing.
    #[test]
    fn broadband_noise_reports_no_key() {
        for seed in [
            0x9e37_79b9_7f4a_7c15u64,
            0x243f_6a88_85a3_08d3,
            0xdead_beef_cafe_1234,
        ] {
            let mut rng = seed;
            let noise: Vec<f32> = (0..TARGET_SAMPLE_RATE as usize * 2)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    ((rng >> 40) as f32 / 8_388_608.0) - 1.0
                })
                .collect();

            let f = Analyzer::new().analyze(&noise);
            assert_eq!(
                f.key_root, None,
                "noise (seed {seed:#x}) was assigned a key"
            );
        }
    }

    /// A pitch-swept sine -- the shape of every kick drum -- has no stable partial, so it
    /// has no key either.
    #[test]
    fn a_pitch_sweep_reports_no_key() {
        let n = TARGET_SAMPLE_RATE as usize * 3;
        let mut phase = 0.0f32;
        let kick: Vec<f32> = (0..n)
            .map(|i| {
                let hz = 200.0 * (-3.0 * (i as f32 / TARGET_SAMPLE_RATE as f32)).exp() + 40.0;
                phase += std::f32::consts::TAU * hz / TARGET_SAMPLE_RATE as f32;
                phase.sin() * 0.8
            })
            .collect();

        assert_eq!(Analyzer::new().analyze(&kick).key_root, None);
    }

    #[test]
    fn loudness_is_reported_in_lufs_and_tracks_level() {
        let loud = Analyzer::new().analyze(&sine(1000.0, 0.8, 2.0));
        let quiet = Analyzer::new().analyze(&sine(1000.0, 0.08, 2.0));

        let loud = loud
            .lufs_integrated
            .expect("2 s is longer than a gate block");
        let quiet = quiet
            .lufs_integrated
            .expect("2 s is longer than a gate block");

        // A 20 dB drop in amplitude is a 20 dB drop in loudness.
        assert!((loud - quiet - 20.0).abs() < 1.0, "{loud} vs {quiet} LUFS");
    }

    /// The analyzer is reused across every file a worker handles, so state left behind by
    /// one file must not colour the next. This is the regression test for the `previous`
    /// magnitude buffer and the flux vector.
    #[test]
    fn reusing_an_analyzer_gives_the_same_answer_as_a_fresh_one() {
        let mut reused = Analyzer::new();
        let noisy = sine(60.0, 0.9, 1.0);
        let target = sine(1000.0, 0.5, 1.0);

        reused.analyze(&noisy);
        let after_reuse = reused.analyze(&target);
        let fresh = Analyzer::new().analyze(&target);

        assert_eq!(after_reuse, fresh);
    }
}
