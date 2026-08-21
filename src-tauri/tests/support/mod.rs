//! Fixtures shared by the integration tests and the benchmarks.
//!
//! Audio is generated rather than committed. A committed corpus would be tens of megabytes
//! of binary in git for content that is fully described by four lines of arithmetic, and it
//! would leave every test's expected values unexplainable by reading the test.
//!
//! The WAV writer here is deliberate too: `symphonia` decodes, it does not encode, and
//! pulling in an encoder crate to produce a 44-byte header is not a trade worth making.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod fixtures;

use std::{f32::consts::TAU, io::Write, path::Path};

/// Sample formats the fixture writer can emit, so tests can cover more than one decode path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavFormat {
    /// 16-bit signed PCM: `WAVE_FORMAT_PCM`. The overwhelming majority of a sample library.
    Pcm16,
    /// 32-bit IEEE float: `WAVE_FORMAT_IEEE_FLOAT`. What a DAW bounces.
    Float32,
}

impl WavFormat {
    fn tag(self) -> u16 {
        match self {
            WavFormat::Pcm16 => 1,
            WavFormat::Float32 => 3,
        }
    }

    fn bits(self) -> u16 {
        match self {
            WavFormat::Pcm16 => 16,
            WavFormat::Float32 => 32,
        }
    }
}

/// Writes a canonical RIFF/WAVE file.
///
/// `samples` is interleaved across `channels`.
pub fn write_wav(path: &Path, format: WavFormat, sample_rate: u32, channels: u16, samples: &[f32]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }

    let bytes_per_sample = u32::from(format.bits() / 8);
    let data_len = samples.len() as u32 * bytes_per_sample;
    let block_align = channels * format.bits() / 8;
    let byte_rate = sample_rate * u32::from(block_align);

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format.tag().to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&format.bits().to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        match format {
            WavFormat::Pcm16 => {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                out.extend_from_slice(&v.to_le_bytes());
            }
            WavFormat::Float32 => out.extend_from_slice(&s.to_le_bytes()),
        }
    }

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&out).unwrap();
    file.sync_all().unwrap();
}

/// A 16-bit mono sine at 48 kHz -- the fixture most tests want.
pub fn write_sine(path: &Path, hz: f32, seconds: f32) {
    write_wav(
        path,
        WavFormat::Pcm16,
        48_000,
        1,
        &sine(hz, 0.5, seconds, 48_000),
    );
}

/// A sine of `seconds` at `rate`.
pub fn sine(hz: f32, amplitude: f32, seconds: f32, rate: u32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .map(|i| amplitude * (TAU * hz * i as f32 / rate as f32).sin())
        .collect()
}

/// Deterministic, dependency-free noise in [-1, 1).
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}
