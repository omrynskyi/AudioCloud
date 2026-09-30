//! A limited decode is the front half of a head-first preview start, and it is only correct if
//! the head is a prefix of the full window: the player plays the head, then continues from the
//! full decode at the seam, so any disagreement between the two is an audible click.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use audiocloud_lib::pipeline::{
    decode::{MAX_OUTPUT_SAMPLES, TARGET_SAMPLE_RATE},
    BufferPool, Decoder,
};
use support::{noise, sine, write_wav, WavFormat};
use tempfile::TempDir;

const HEAD: usize = TARGET_SAMPLE_RATE as usize / 2;
/// Output samples at the end of the head that may differ: the resampler's edge.
const SEAM_GUARD: usize = 256;

fn programme(rate: u32, seconds: f32) -> Vec<f32> {
    let tone = sine(330.0, 0.4, seconds, rate);
    let hiss = noise(7, tone.len());
    tone.iter().zip(&hiss).map(|(t, n)| t + 0.05 * n).collect()
}

#[test]
fn the_head_is_a_prefix_of_the_full_window_at_every_source_rate() {
    let dir = TempDir::new().unwrap();
    for rate in [22_050u32, 44_100, 48_000, 96_000] {
        let path = dir.path().join(format!("{rate}.wav"));
        write_wav(&path, WavFormat::Float32, rate, 1, &programme(rate, 12.0));

        let mut decoder = Decoder::new(BufferPool::for_decode());
        let head = decoder.decode_limited(&path, "wav", HEAD).unwrap();
        let head_samples: Vec<f32> = head.samples.to_vec();
        let full = decoder.decode(&path, "wav").unwrap();

        assert_eq!(head_samples.len(), HEAD, "{rate} Hz: head length");
        assert!(
            head.truncated,
            "{rate} Hz: a 12 s file is longer than its head"
        );
        assert_eq!(
            full.samples.len(),
            MAX_OUTPUT_SAMPLES,
            "{rate} Hz: full length"
        );
        assert_eq!(
            head.duration_ms, full.duration_ms,
            "{rate} Hz: duration is the file's"
        );

        let worst = head_samples[..HEAD - SEAM_GUARD]
            .iter()
            .zip(full.samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "{rate} Hz: head and full disagree by {worst}");
    }
}

#[test]
fn a_file_shorter_than_its_head_is_not_truncated() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("short.wav");
    write_wav(&path, WavFormat::Pcm16, 44_100, 1, &programme(44_100, 0.2));

    let mut decoder = Decoder::new(BufferPool::for_decode());
    let head = decoder.decode_limited(&path, "wav", HEAD).unwrap();
    assert!(!head.truncated);
    assert!(head.samples.len() < HEAD);
}

#[test]
fn a_limit_past_the_window_is_clamped_to_it() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("long.wav");
    write_wav(
        &path,
        WavFormat::Float32,
        48_000,
        1,
        &programme(48_000, 12.0),
    );

    let mut decoder = Decoder::new(BufferPool::for_decode());
    let d = decoder.decode_limited(&path, "wav", usize::MAX).unwrap();
    assert_eq!(d.samples.len(), MAX_OUTPUT_SAMPLES);
}
