//! The embedding: a 45-number timbre-and-envelope description of a whole sound, computed
//! from the log-mel spectrogram the pipeline has already paid for plus the decoded audio.
//! No pretrained model, no learned weights.
//!
//! **What it encodes**, per sound:
//!
//! - **Timbre** -- 20 MFCCs (an orthonormal DCT-II down each log-mel column, coefficients
//!   0..20, so C0 -- overall level -- is kept), summarized as a mean and a standard deviation
//!   over the sound's *real* frames. 40 numbers.
//! - **Envelope** -- five numbers from a time-domain RMS envelope: attack time, decay time,
//!   the head-to-tail energy ratio, the log of the length, and whether the decoder hit the
//!   10 s window. These are what separate a loop from a hit and a pad from a pluck, and they
//!   turned out to matter more for classification than the MFCCs themselves.
//!
//! It is a two-step computation because the stages that run it see different things:
//! [`extract`] runs in the process stage, where the decoded audio still exists, and produces
//! *raw* features. [`MfccEmbedder`] runs in the embed stage and turns raw features into the
//! stored vector by standardizing each dimension against fixed constants
//! ([`super::mfcc_stats`]) and clipping at [`CLIP`] standard deviations.
//!
//! **The stored vectors are z-scores, and are deliberately not L2-normalized.** Normalizing
//! them moves a third of every sound's nearest neighbors (row norms span roughly 2 to 20,
//! and "how unusual is this sound" is real information for the map). Anything that compares
//! two stored vectors must therefore not assume unit length -- `get_similar` divides by the
//! norms and incremental placement uses Euclidean distance.
//!
//! **Why fixed constants.** Vectors are immutable once written -- duplicates share their
//! twin's bytes -- so they cannot be re-standardized as the library grows. Constants fitted
//! to one library generalize well to another of the same kind (stats from 300 of 499 samples
//! reproduced 95% of each sample's neighbors), and `examples/mfcc_stats.rs` regenerates them.

use std::sync::OnceLock;

use super::{
    decode::{MAX_OUTPUT_SAMPLES, TARGET_SAMPLE_RATE},
    embed::{Embed, EmbedError},
    features::{FRAME_SIZE, HOP_SIZE},
    mel::{MEL_BINS, MEL_FRAMES},
    mfcc_stats::{MEAN, STD},
};

/// MFCC coefficients kept, starting at C0.
pub const MFCC_COEFFS: usize = 20;

/// Envelope descriptors: attack, decay, head/tail ratio, log length, hit-the-cap flag.
pub const ENVELOPE_DIMS: usize = 5;

/// Width of a feature vector, raw or stored: mean and std of each MFCC, then the envelope.
pub const FEATURE_DIM: usize = MFCC_COEFFS * 2 + ENVELOPE_DIMS;

/// Stored z-scores are clipped to +-this many standard deviations, so one degenerate file
/// (an hours-long decay, a pure-DC offset) cannot stretch a dimension for the whole map.
pub const CLIP: f32 = 5.0;

/// The spectrogram's dynamic range, in dB below its loudest bin. Anything quieter is raised
/// to that floor before the DCT (`librosa.power_to_db(top_db=80)`, which the approved map was
/// built with). Without it the -100 dB silence *inside* a sound -- a gap between hits, a
/// gated tail -- dominates the MFCC spread, and two sounds differ by how empty their quiet
/// parts are rather than by what they sound like.
const TOP_DB: f32 = 80.0;

/// Frames of the envelope counted as the sound's "head": the first 0.32 s at a 10 ms hop.
const HEAD_FRAMES: usize = 32;

/// A frame is "off" once its RMS is below this fraction of the peak.
const ACTIVE_FLOOR: f32 = 0.1;

/// Floor inside the head/tail energy ratio, so a silent tail does not divide by zero.
const ENERGY_EPS: f64 = 1e-12;

const SAMPLE_RATE: f32 = TARGET_SAMPLE_RATE as f32;

/// Raw, unstandardized features for one sound.
///
/// `samples` is the decoded mono window (at most 10 s at 48 kHz), `mel` the row-major
/// `[MEL_FRAMES][MEL_BINS]` log-mel spectrogram of it in dB -- computed with
/// [`super::mel::Padding::ZeroPad`], so frames past the end of the audio are the silent
/// floor and are ignored here in favour of the true frame count -- and `truncated` whether
/// the decoder stopped at the window rather than at the end of the file.
///
/// Digital silence and one-sample files come out finite; the layout is
/// `[mfcc mean x 20][mfcc std x 20][attack, decay, head/tail dB, log10 seconds, capped]`.
pub fn extract(samples: &[f32], mel: &[f32], truncated: bool) -> [f32; FEATURE_DIM] {
    let mut out = [0.0f32; FEATURE_DIM];
    let real_frames = real_frame_count(samples.len());
    if mel.len() < real_frames * MEL_BINS {
        return out;
    }

    // --- Timbre: MFCC mean and standard deviation over the real frames.
    let table = dct_table();
    let real = &mel[..real_frames * MEL_BINS];
    let floor = real.iter().copied().fold(f32::NEG_INFINITY, f32::max) - TOP_DB;
    let mut sum = [0.0f64; MFCC_COEFFS];
    let mut sum_sq = [0.0f64; MFCC_COEFFS];
    for frame in real.as_chunks::<MEL_BINS>().0 {
        for (k, basis) in table.iter().enumerate() {
            let c: f64 = basis
                .iter()
                .zip(frame)
                .map(|(&b, &v)| f64::from(b) * f64::from(v.max(floor)))
                .sum();
            sum[k] += c;
            sum_sq[k] += c * c;
        }
    }
    let n = real_frames as f64;
    for k in 0..MFCC_COEFFS {
        let mean = sum[k] / n;
        out[k] = mean as f32;
        out[MFCC_COEFFS + k] = (sum_sq[k] / n - mean * mean).max(0.0).sqrt() as f32;
    }

    // --- Envelope.
    let rms = rms_envelope(samples, real_frames);
    let env = &mut out[MFCC_COEFFS * 2..];
    let peak = rms.iter().copied().fold(0.0f32, f32::max);
    if peak > 0.0 {
        let peak_at = rms.iter().position(|&v| v == peak).unwrap_or(0);
        let floor = ACTIVE_FLOOR * peak;
        let onset = rms.iter().position(|&v| v > floor).unwrap_or(0);
        let decay = rms[peak_at..]
            .iter()
            .position(|&v| v < floor)
            .unwrap_or(rms.len() - peak_at);

        let energy = |part: &[f32]| {
            part.iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
        };
        let split = HEAD_FRAMES.min(rms.len());
        let head = energy(&rms[..split]) + ENERGY_EPS;
        let tail = energy(&rms[split..]) + ENERGY_EPS;

        env[0] = (peak_at.saturating_sub(onset) * HOP_SIZE) as f32 / SAMPLE_RATE;
        env[1] = (decay * HOP_SIZE) as f32 / SAMPLE_RATE;
        env[2] = (10.0 * (head / tail).log10()) as f32;
    }
    // Clamped to one frame so a one-sample file has a defined length.
    env[3] = (samples.len().max(FRAME_SIZE) as f32 / SAMPLE_RATE).log10();
    env[4] = f32::from(truncated || samples.len() >= MAX_OUTPUT_SAMPLES);
    out
}

/// Frames of `mel` that describe real audio rather than padding: `center = true` gives a
/// signal of `n` samples `1 + n / hop` frames.
fn real_frame_count(n_samples: usize) -> usize {
    (1 + n_samples / HOP_SIZE).clamp(1, MEL_FRAMES)
}

/// RMS of each `FRAME_SIZE` window centered on `frame * HOP_SIZE`, zero-extended past the
/// ends -- the same framing as the mel front-end, so envelope frame `i` and mel frame `i`
/// describe the same instant.
fn rms_envelope(samples: &[f32], frames: usize) -> Vec<f32> {
    let half = (FRAME_SIZE / 2) as isize;
    (0..frames)
        .map(|i| {
            let centre = (i * HOP_SIZE) as isize;
            let lo = (centre - half).max(0) as usize;
            let hi = ((centre + half).max(0) as usize).min(samples.len());
            let energy: f64 = samples
                .get(lo..hi.max(lo))
                .unwrap_or(&[])
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum();
            (energy / FRAME_SIZE as f64).sqrt() as f32
        })
        .collect()
}

/// The orthonormal DCT-II basis, `MFCC_COEFFS` rows of `MEL_BINS` weights
/// (`scipy.fft.dct(type=2, norm="ortho")`, which is what `librosa.feature.mfcc` uses).
fn dct_table() -> &'static [[f32; MEL_BINS]; MFCC_COEFFS] {
    static TABLE: OnceLock<[[f32; MEL_BINS]; MFCC_COEFFS]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0f32; MEL_BINS]; MFCC_COEFFS];
        let n = MEL_BINS as f64;
        for (k, row) in table.iter_mut().enumerate() {
            let scale = if k == 0 {
                (1.0 / n).sqrt()
            } else {
                (2.0 / n).sqrt()
            };
            for (i, w) in row.iter_mut().enumerate() {
                *w =
                    (scale * (std::f64::consts::PI / n * (i as f64 + 0.5) * k as f64).cos()) as f32;
            }
        }
        table
    })
}

/// Raw features -> stored vector: standardize against [`MEAN`]/[`STD`], clip at [`CLIP`].
#[derive(Debug, Default, Clone, Copy)]
pub struct MfccEmbedder;

impl MfccEmbedder {
    pub fn new() -> Self {
        Self
    }
}

impl Embed for MfccEmbedder {
    fn embed_batch(&self, features: &[f32], count: usize) -> Result<Vec<f32>, EmbedError> {
        let mut out = Vec::with_capacity(count * FEATURE_DIM);
        for raw in features.as_chunks::<FEATURE_DIM>().0.iter().take(count) {
            out.extend(standardize(raw));
        }
        if out.len() != count * FEATURE_DIM {
            return Err(EmbedError::ShortBatch {
                count,
                dim: FEATURE_DIM,
                actual: out.len(),
            });
        }
        Ok(out)
    }

    fn embedding_dim(&self) -> usize {
        FEATURE_DIM
    }
}

/// One raw feature vector -> z-scores. A non-finite or zero-spread dimension maps to 0,
/// the corpus mean, rather than poisoning every distance it is part of.
pub fn standardize(raw: &[f32]) -> [f32; FEATURE_DIM] {
    let mut out = [0.0f32; FEATURE_DIM];
    for (i, (&x, o)) in raw.iter().zip(out.iter_mut()).enumerate() {
        let z = (x - MEAN[i]) / STD[i];
        *o = if z.is_finite() {
            z.clamp(-CLIP, CLIP)
        } else {
            0.0
        };
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::pipeline::{
        features::Analyzer,
        mel::{FrontEnd, Padding},
    };

    /// A decaying sine burst of `seconds` at `hz`, `tau` seconds of decay.
    fn burst(seconds: f32, hz: f32, tau: f32) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE;
                (2.0 * std::f32::consts::PI * hz * t).sin() * (-t / tau).exp() * 0.5
            })
            .collect()
    }

    fn features_of(samples: &[f32], truncated: bool) -> [f32; FEATURE_DIM] {
        let mut analyzer = Analyzer::new();
        let mut front_end = FrontEnd::with_padding(Padding::ZeroPad);
        let mut mel = Vec::new();
        front_end.compute(samples, &mut analyzer, &mut mel);
        extract(samples, &mel, truncated)
    }

    #[test]
    fn the_layout_is_45_finite_numbers() {
        assert_eq!(FEATURE_DIM, 45);
        let f = features_of(&burst(0.4, 220.0, 0.1), false);
        assert!(f.iter().all(|v| v.is_finite()));
        assert_eq!(MfccEmbedder::new().embedding_dim(), FEATURE_DIM);
    }

    #[test]
    fn digital_silence_and_tiny_files_stay_finite() {
        for samples in [vec![0.0f32; 24_000], vec![0.3f32], Vec::new()] {
            let f = features_of(&samples, false);
            assert!(
                f.iter().all(|v| v.is_finite()),
                "non-finite for {} samples",
                samples.len()
            );
            let z = standardize(&f);
            assert!(z.iter().all(|v| v.is_finite() && v.abs() <= CLIP));
        }
    }

    #[test]
    fn identical_audio_gives_an_identical_vector() {
        let a = features_of(&burst(0.5, 330.0, 0.08), false);
        let b = features_of(&burst(0.5, 330.0, 0.08), false);
        assert_eq!(a, b);
    }

    /// A short tick and a long tone can share a spectrum; the envelope block is what tells
    /// them apart, and it is the reason this embedding beats the MFCCs alone.
    #[test]
    fn length_and_decay_separate_a_tick_from_a_pad() {
        let tick = features_of(&burst(0.2, 440.0, 0.02), false);
        let pad = features_of(&burst(4.0, 440.0, 1.5), false);
        let base = MFCC_COEFFS * 2;
        assert!(
            pad[base + 1] > tick[base + 1] + 0.3,
            "decay: {} vs {}",
            pad[base + 1],
            tick[base + 1]
        );
        assert!(
            pad[base + 3] > tick[base + 3] + 1.0,
            "log length: {} vs {}",
            pad[base + 3],
            tick[base + 3]
        );
    }

    /// Frames past the end of the audio are the silent floor in a zero-padded mel; averaging
    /// them in would make every short sound describe its own padding. Whatever sits in the
    /// padded frames must not reach the features.
    #[test]
    fn padding_does_not_leak_into_the_mfcc_statistics() {
        let samples = burst(0.3, 220.0, 0.1);
        let mut analyzer = Analyzer::new();
        let mut front_end = FrontEnd::with_padding(Padding::ZeroPad);
        let mut mel = Vec::new();
        front_end.compute(&samples, &mut analyzer, &mut mel);
        let clean = extract(&samples, &mel, false);

        let real = real_frame_count(samples.len());
        assert!(
            real < MEL_FRAMES / 10,
            "the fixture should be mostly padding"
        );
        for v in &mut mel[real * MEL_BINS..] {
            *v = 60.0;
        }
        assert_eq!(extract(&samples, &mel, false), clean);
    }

    #[test]
    fn the_cap_flag_follows_truncation_or_a_full_window() {
        let base = MFCC_COEFFS * 2 + 4;
        assert_eq!(features_of(&burst(0.3, 220.0, 0.1), false)[base], 0.0);
        assert_eq!(features_of(&burst(0.3, 220.0, 0.1), true)[base], 1.0);
    }

    #[test]
    fn stored_vectors_are_clipped_and_finite() {
        let mut raw = [0.0f32; FEATURE_DIM];
        raw[0] = 1.0e9;
        raw[1] = f32::NAN;
        raw[2] = f32::NEG_INFINITY;
        let z = standardize(&raw);
        assert_eq!(z[0], CLIP);
        assert_eq!(z[1], 0.0);
        assert_eq!(z[2], 0.0);
    }

    #[test]
    fn a_batch_embeds_every_sample_independently() {
        let embedder = MfccEmbedder::new();
        let mut raw = features_of(&burst(0.3, 220.0, 0.05), false).to_vec();
        raw.extend(features_of(&burst(2.0, 1200.0, 0.8), false));
        let out = embedder.embed_batch(&raw, 2).unwrap();
        assert_eq!(out.len(), 2 * FEATURE_DIM);
        assert_ne!(out[..FEATURE_DIM], out[FEATURE_DIM..]);
    }

    #[test]
    fn a_short_batch_is_an_error_not_a_panic() {
        let err = MfccEmbedder::new().embed_batch(&[0.0; FEATURE_DIM], 2);
        assert!(err.is_err());
    }
}
