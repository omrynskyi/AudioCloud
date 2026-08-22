//! A timbre embedding computed directly from the log-mel spectrogram, with no model.
//!
//! The cheap half of the `overview.md` risk 5 question. CLAP costs ~62 ms per sample; this
//! costs microseconds, on a spectrogram the pipeline has already paid for. If it retrieves
//! drums as well as CLAP does, the inference is not earning its keep on a one-shot library
//! -- and `tests/evaluation.rs` is what settles that, per category, with a number.
//!
//! It implements [`Embed`] like the model does, so it drops into the same five-stage
//! pipeline, the same `embeddings.bin`, and the same projection with nothing downstream
//! aware of the difference.
//!
//! **What it encodes**, in three blocks that mirror the axes XLN's XO exposes to users
//! (Frequency, Length, Drumminess) because those are the axes a drum library actually
//! varies along:
//!
//! 1. **Spectral shape** -- the average mel-band profile, with each frame's own mean
//!    removed so the vector describes *colour* rather than *loudness*. A kick's energy sits
//!    low, a hat's high; this is the block that separates them.
//! 2. **MFCCs** -- a DCT of the log-mel, coefficients 1..=19, mean and standard deviation
//!    over time. C0 is dropped deliberately: it is overall energy, which is a mixing
//!    decision rather than a timbre.
//! 3. **Envelope** -- how long the sample sounds for and how it decays. A 200 ms tick and a
//!    two-second crash can have similar spectra and are not similar sounds.
//!
//! Each block is L2-normalized before concatenation, so a 64-dimensional block and a
//! 4-dimensional one contribute comparably instead of the longest block deciding
//! everything. The block weights are the one genuinely arbitrary thing here; they are
//! constants rather than a tuning API because `tests/evaluation.rs` can measure a change to
//! them, and a knob nobody measures is a knob nobody should have.

use crate::pipeline::{
    embed::{Embed, EmbedError, MEL_VALUES},
    mel::{MEL_BINS, MEL_FRAMES},
};

/// MFCC coefficients kept, starting at C1. C0 is overall energy, not timbre.
const MFCC_COEFFS: usize = 19;

/// Envelope descriptors: active length, decay slope, peak position, energy spread.
const ENVELOPE_DIMS: usize = 4;

/// Dimensionality of the vector this produces.
pub const DSP_EMBEDDING_DIM: usize = MEL_BINS + MFCC_COEFFS * 2 + ENVELOPE_DIMS;

/// Frames quieter than this far below the loudest frame are treated as silence.
///
/// The front-end pads every sample to a fixed ten-second window, so without a gate the
/// statistics of a 200 ms hi-hat would be almost entirely a description of the padding.
const ACTIVE_FLOOR_DB: f32 = 60.0;

/// Relative weight of each block after individual normalization.
const SHAPE_WEIGHT: f32 = 1.0;
const MFCC_WEIGHT: f32 = 1.0;
const ENVELOPE_WEIGHT: f32 = 0.7;

/// A stateless log-mel -> timbre-vector transform.
#[derive(Debug, Default, Clone, Copy)]
pub struct DspEmbedder {
    _private: (),
}

impl DspEmbedder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Embeds one `[MEL_FRAMES][MEL_BINS]` log-mel spectrogram, row-major, in dB.
    pub fn embed_one(&self, mel: &[f32], out: &mut Vec<f32>) {
        out.clear();

        // Per-frame energy, as the loudest band in the frame. Cheap, and robust to the
        // floor: a silent frame is at the log floor in every band.
        let mut frame_peak = [f32::NEG_INFINITY; MEL_FRAMES];
        let mut frame_mean = [0.0f32; MEL_FRAMES];
        for (f, peak) in frame_peak.iter_mut().enumerate() {
            let row = &mel[f * MEL_BINS..(f + 1) * MEL_BINS];
            let mut sum = 0.0f32;
            for &v in row {
                if v > *peak {
                    *peak = v;
                }
                sum += v;
            }
            frame_mean[f] = sum / MEL_BINS as f32;
        }

        let loudest = frame_peak.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let gate = loudest - ACTIVE_FLOOR_DB;
        let active: Vec<usize> = (0..MEL_FRAMES).filter(|&f| frame_peak[f] >= gate).collect();
        // Digital silence gates everything out; describe it as the floor rather than
        // dividing by zero.
        let active: Vec<usize> = if active.is_empty() {
            (0..MEL_FRAMES).collect()
        } else {
            active
        };
        let n = active.len() as f32;

        // --- Block 1: loudness-normalized spectral shape.
        let mut shape = vec![0.0f32; MEL_BINS];
        for &f in &active {
            let row = &mel[f * MEL_BINS..(f + 1) * MEL_BINS];
            for (b, &v) in row.iter().enumerate() {
                shape[b] += v - frame_mean[f];
            }
        }
        for v in shape.iter_mut() {
            *v /= n;
        }

        // --- Block 2: MFCC mean and standard deviation over the active frames.
        let mut mfcc_sum = [0.0f64; MFCC_COEFFS];
        let mut mfcc_sq = [0.0f64; MFCC_COEFFS];
        let mut coeffs = [0.0f32; MFCC_COEFFS];
        for &f in &active {
            let row = &mel[f * MEL_BINS..(f + 1) * MEL_BINS];
            dct_ii(row, &mut coeffs);
            for k in 0..MFCC_COEFFS {
                mfcc_sum[k] += f64::from(coeffs[k]);
                mfcc_sq[k] += f64::from(coeffs[k]) * f64::from(coeffs[k]);
            }
        }
        let mut mfcc = Vec::with_capacity(MFCC_COEFFS * 2);
        for k in 0..MFCC_COEFFS {
            let mean = mfcc_sum[k] / f64::from(n);
            let var = (mfcc_sq[k] / f64::from(n) - mean * mean).max(0.0);
            mfcc.push(mean as f32);
            mfcc.push(var.sqrt() as f32);
        }

        // --- Block 3: envelope.
        let first = active[0];
        let last = active[active.len() - 1];
        let peak_frame = frame_peak
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (f, &v)| {
                if v > best.1 {
                    (f, v)
                } else {
                    best
                }
            })
            .0;

        // Log length: a drum library spans three orders of magnitude in duration, and the
        // difference between 50 ms and 100 ms matters as much as between 1 s and 2 s.
        let length = ((last - first + 1) as f32).ln() / (MEL_FRAMES as f32).ln();
        // Decay: dB lost per active frame from the peak onward, normalized.
        let decay = if last > peak_frame {
            (frame_peak[peak_frame] - frame_peak[last]) / (last - peak_frame) as f32
        } else {
            0.0
        };
        let peak_position = peak_frame as f32 / MEL_FRAMES as f32;
        let spread = {
            let mean: f32 = active.iter().map(|&f| frame_peak[f]).sum::<f32>() / n;
            let var: f32 = active
                .iter()
                .map(|&f| (frame_peak[f] - mean).powi(2))
                .sum::<f32>()
                / n;
            var.sqrt() / 40.0
        };
        let envelope = [length, decay, peak_position, spread];

        // --- Assemble: each block normalized, then weighted, then the whole thing
        // normalized. Without the per-block step the 64-wide shape block would drown the
        // 4-wide envelope purely by being longer.
        push_normalized(out, &shape, SHAPE_WEIGHT);
        push_normalized(out, &mfcc, MFCC_WEIGHT);
        push_normalized(out, &envelope, ENVELOPE_WEIGHT);
        l2_normalize(out);
    }
}

impl Embed for DspEmbedder {
    fn embed_batch(&self, mels: &[f32], count: usize) -> Result<Vec<f32>, EmbedError> {
        if count == 0 || mels.len() != count * MEL_VALUES {
            return Err(EmbedError::ShortBatch {
                count,
                dim: DSP_EMBEDDING_DIM,
                actual: mels.len(),
            });
        }
        let mut out = Vec::with_capacity(count * DSP_EMBEDDING_DIM);
        let mut one = Vec::with_capacity(DSP_EMBEDDING_DIM);
        for i in 0..count {
            self.embed_one(&mels[i * MEL_VALUES..(i + 1) * MEL_VALUES], &mut one);
            out.extend_from_slice(&one);
        }
        Ok(out)
    }

    fn embedding_dim(&self) -> usize {
        DSP_EMBEDDING_DIM
    }
}

/// DCT-II of one log-mel frame, coefficients 1..=`MFCC_COEFFS` (C0 dropped).
fn dct_ii(row: &[f32], out: &mut [f32; MFCC_COEFFS]) {
    let n = row.len() as f32;
    for (k, slot) in out.iter_mut().enumerate() {
        let k = (k + 1) as f32;
        let mut sum = 0.0f32;
        for (b, &v) in row.iter().enumerate() {
            sum += v * (std::f32::consts::PI * k * (b as f32 + 0.5) / n).cos();
        }
        *slot = sum / n;
    }
}

/// Appends `block`, scaled to `weight` times unit length. A zero block contributes zeros
/// rather than NaNs.
fn push_normalized(out: &mut Vec<f32>, block: &[f32], weight: f32) {
    let norm = block
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let scale = if norm > 0.0 {
        weight / norm as f32
    } else {
        0.0
    };
    out.extend(block.iter().map(|&x| x * scale));
}

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// A synthetic log-mel with energy concentrated in `band`, lasting `frames`.
    fn spectrogram(band: usize, frames: usize) -> Vec<f32> {
        let mut mel = vec![-100.0f32; MEL_VALUES];
        for f in 0..frames.min(MEL_FRAMES) {
            for b in 0..MEL_BINS {
                let distance = (b as f32 - band as f32).abs();
                mel[f * MEL_BINS + b] = -distance * 2.0 - f as f32 * 0.05;
            }
        }
        mel
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn embed(mel: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        DspEmbedder::new().embed_one(mel, &mut out);
        out
    }

    #[test]
    fn the_vector_is_unit_length_and_the_right_width() {
        let v = embed(&spectrogram(10, 40));
        assert_eq!(v.len(), DSP_EMBEDDING_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    /// The property the whole module exists for: something low and short (a kick) must be
    /// closer to another low short thing than to something high and short (a hat).
    #[test]
    fn low_sounds_are_nearer_to_low_sounds_than_to_high_ones() {
        let kick = embed(&spectrogram(6, 40));
        let other_kick = embed(&spectrogram(8, 45));
        let hat = embed(&spectrogram(55, 40));

        assert!(
            cosine(&kick, &other_kick) > cosine(&kick, &hat),
            "kick/kick {:.3} was not above kick/hat {:.3}",
            cosine(&kick, &other_kick),
            cosine(&kick, &hat)
        );
    }

    /// And length has to matter, or a tick and a crash with the same colour collide.
    #[test]
    fn length_separates_otherwise_identical_spectra() {
        let short = embed(&spectrogram(30, 20));
        let similar_short = embed(&spectrogram(30, 25));
        let long = embed(&spectrogram(30, 800));

        assert!(
            cosine(&short, &similar_short) > cosine(&short, &long),
            "short/short {:.3} was not above short/long {:.3}",
            cosine(&short, &similar_short),
            cosine(&short, &long)
        );
    }

    /// Silence must produce a finite vector rather than a NaN that poisons every
    /// projection that ever reads it.
    #[test]
    fn digital_silence_does_not_produce_nans() {
        let v = embed(&vec![-100.0f32; MEL_VALUES]);
        assert_eq!(v.len(), DSP_EMBEDDING_DIM);
        assert!(v.iter().all(|x| x.is_finite()), "silence produced {v:?}");
    }

    /// Loudness is a mixing decision, not a timbre: the same sound 12 dB down must land in
    /// essentially the same place.
    #[test]
    fn overall_level_barely_moves_the_vector() {
        let quiet = embed(&spectrogram(20, 60));
        let loud: Vec<f32> = spectrogram(20, 60).iter().map(|v| v + 12.0).collect();
        let loud = embed(&loud);

        assert!(
            cosine(&quiet, &loud) > 0.99,
            "a 12 dB gain change moved the vector to {:.4}",
            cosine(&quiet, &loud)
        );
    }
}
