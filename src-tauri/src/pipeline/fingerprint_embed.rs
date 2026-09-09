//! A raw fingerprint embedding: a small, downsampled image of a sound's first fraction of a
//! second -- no pretrained model, no learned weights, just the shape of the attack and
//! early decay as the log-mel front-end already renders it.
//!
//! Implements [`Embed`] like the model does, so it drops into the same pipeline stage, the
//! same `embeddings.bin`, and the same projection with nothing downstream aware of the
//! difference -- the same approach [`super::dsp_embed::DspEmbedder`] takes, and this reuses
//! its module's log-mel input rather than computing a second spectrogram.
//!
//! **Why this and not [`super::dsp_embed::DspEmbedder`].** That module is the more
//! sophisticated of the two -- MFCCs, a loudness-normalized spectral-shape block, an
//! envelope block -- and this embedder is deliberately simpler: no
//! MFCCs, no envelope descriptors, just a cropped and downsampled spectrogram treated as an
//! image. It exists because the choice to use it is not about winning that evaluation --
//! it is a deliberate trade of some measured retrieval accuracy for a pipeline with no
//! external model to download, license, or version, and a similarity signal that is exactly
//! and only "what does the first fraction of a second of this sound look like."

use super::{
    embed::{Embed, EmbedError, MEL_VALUES},
    mel::MEL_BINS,
};

/// Frequency bands after block-averaging [`MEL_BINS`] mel bands down.
pub const FP_ROWS: usize = 32;

/// Time frames, cropped from the front of the mel spectrogram.
///
/// The mel front-end's hop is 10 ms (`pipeline::features::HOP_SIZE` at 48 kHz), so 32
/// frames is the sample's first ~0.32 s -- long enough to hold a one-shot's attack and the
/// start of its decay, short enough that a long tail or a loop's later content cannot dilute
/// it. No cropping-for-shortness logic is needed here the way the raw-audio version would
/// need it: the mel buffer is always the model's full padded window
/// ([`super::mel::MEL_FRAMES`]) long regardless of how short the source file was.
pub const FP_COLS: usize = 32;

/// The flattened fingerprint length.
pub const FINGERPRINT_DIM: usize = FP_ROWS * FP_COLS;

/// A stateless log-mel -> fingerprint transform.
#[derive(Debug, Default, Clone, Copy)]
pub struct FingerprintEmbedder;

impl FingerprintEmbedder {
    pub fn new() -> Self {
        Self
    }
}

impl Embed for FingerprintEmbedder {
    fn embed_batch(&self, mels: &[f32], count: usize) -> Result<Vec<f32>, EmbedError> {
        let mut out = Vec::with_capacity(count * FINGERPRINT_DIM);
        for i in 0..count {
            let sample = &mels[i * MEL_VALUES..(i + 1) * MEL_VALUES];
            out.extend_from_slice(&fingerprint(sample));
        }
        Ok(out)
    }

    fn embedding_dim(&self) -> usize {
        FINGERPRINT_DIM
    }
}

/// One fingerprint from one [`MEL_VALUES`]-long log-mel buffer, row-major `[frames][mels]`
/// (`mel::FrontEnd::compute`'s own layout).
///
/// L2-normalized, per [`Embed::embed_batch`]'s contract -- everything downstream (cosine
/// similarity in `get_similar`, the projectors) assumes a unit vector.
fn fingerprint(mels: &[f32]) -> [f32; FINGERPRINT_DIM] {
    let mut grid = [0.0f32; FINGERPRINT_DIM];
    for (col, frame) in mels.chunks_exact(MEL_BINS).take(FP_COLS).enumerate() {
        for (row, band) in reduce_bands(frame).iter().enumerate() {
            grid[row * FP_COLS + col] = *band;
        }
    }

    // Per-sample shape normalization -- min-max to [0, 1] -- makes the fingerprint
    // invariant to gain before the final L2 normalization makes it a unit vector: two hits
    // of the same drum at different recorded levels should fingerprint alike, because their
    // *shape* is alike and shape is all this is meant to capture.
    normalize_shape(&mut grid);
    l2_normalize(&mut grid);
    grid
}

/// Averages [`MEL_BINS`] mel bands down into [`FP_ROWS`] bands.
fn reduce_bands(frame: &[f32]) -> [f32; FP_ROWS] {
    let mut bands = [0.0f32; FP_ROWS];
    let per_band = MEL_BINS.div_ceil(FP_ROWS);
    for (row, band) in bands.iter_mut().enumerate() {
        let start = row * per_band;
        let end = (start + per_band).min(MEL_BINS);
        if start >= end {
            continue;
        }
        let sum: f32 = frame[start..end].iter().sum();
        *band = sum / (end - start) as f32;
    }
    bands
}

/// Per-sample min-max normalization to `[0, 1]`. Digital silence, or a buffer too short to
/// have any spectral shape at all, leaves every cell identical; filling with zero is the
/// honest answer rather than dividing by a zero range.
fn normalize_shape(grid: &mut [f32; FINGERPRINT_DIM]) {
    let min = grid.iter().copied().fold(f32::INFINITY, f32::min);
    let max = grid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range <= 0.0 || !range.is_finite() {
        grid.fill(0.0);
        return;
    }
    for v in grid.iter_mut() {
        *v = (*v - min) / range;
    }
}

fn l2_normalize(grid: &mut [f32; FINGERPRINT_DIM]) {
    let norm = grid
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        let scale = (1.0 / norm) as f32;
        for v in grid.iter_mut() {
            *v *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn mel_buffer(fill: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut v = vec![0.0f32; MEL_VALUES];
        for frame in 0..MEL_VALUES / MEL_BINS {
            for bin in 0..MEL_BINS {
                v[frame * MEL_BINS + bin] = fill(frame, bin);
            }
        }
        v
    }

    #[test]
    fn every_fingerprint_is_a_unit_vector() {
        let embedder = FingerprintEmbedder::new();
        let mels = mel_buffer(|frame, bin| ((frame * 3 + bin) % 17) as f32 * 0.1 - 0.5);
        let out = embedder.embed_batch(&mels, 1).unwrap();
        assert_eq!(out.len(), FINGERPRINT_DIM);
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm was {norm}");
    }

    #[test]
    fn digital_silence_is_the_zero_vector_not_nan() {
        let embedder = FingerprintEmbedder::new();
        let mels = mel_buffer(|_, _| 0.0);
        let out = embedder.embed_batch(&mels, 1).unwrap();
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_batch_embeds_every_sample_independently() {
        let embedder = FingerprintEmbedder::new();
        let mut mels = mel_buffer(|frame, bin| ((frame + bin) % 7) as f32);
        mels.extend(mel_buffer(|frame, bin| ((frame * bin) % 11) as f32));
        let out = embedder.embed_batch(&mels, 2).unwrap();
        assert_eq!(out.len(), 2 * FINGERPRINT_DIM);
        assert_ne!(
            out[..FINGERPRINT_DIM],
            out[FINGERPRINT_DIM..],
            "two different mel buffers embedded identically"
        );
    }

    /// Gain invariance: the same shape at a different loudness fingerprints the same, since
    /// shape -- not absolute level -- is what this embedder is for.
    #[test]
    fn a_quieter_copy_of_the_same_shape_fingerprints_alike() {
        let embedder = FingerprintEmbedder::new();
        let loud = mel_buffer(|frame, bin| ((frame * 5 + bin * 3) % 13) as f32 - 6.0);
        let quiet: Vec<f32> = loud.iter().map(|v| v * 0.3 - 4.0).collect();

        let a = embedder.embed_batch(&loud, 1).unwrap();
        let b = embedder.embed_batch(&quiet, 1).unwrap();
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            dot > 0.999,
            "cosine similarity between the two was only {dot}"
        );
    }
}
