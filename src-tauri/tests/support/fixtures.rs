//! The Rust half of the CLAP parity fixtures.
//!
//! `scripts/export_clap_onnx.py` defines twenty synthetic signals as recipes, runs them
//! through Python CLAP, and commits the resulting embeddings as the parity oracle. This
//! module regenerates the same waveforms in Rust so the oracle can be checked without
//! nineteen megabytes of wav files in git.
//!
//! **Two generators means they can drift**, which would show up as a parity failure with a
//! completely misleading cause. So `fixtures.json` carries a probe of each waveform --
//! sixteen samples and an RMS -- and [`Fixture::check_probe`] verifies the recipes still
//! produce the same audio before anything looks at an embedding. A hash would be stricter
//! and wrong: `sin` may differ in the last bit between libms, and that is not a bug.
//!
//! Keep the arithmetic here identical to `synthesize` in the export script. Both compute in
//! `f64` and cast to `f32` exactly once, at the end, for the same reason.

#![allow(dead_code)]

use std::{f64::consts::PI, path::Path};

use serde::Deserialize;

/// Where the committed fixtures live, relative to the crate root.
pub const FIXTURE_DIR: &str = "tests/fixtures/clap";

/// One synthetic signal, deserialized straight from `fixtures.json`.
///
/// The flat shape mirrors the Python dicts: not every field applies to every `kind`, and a
/// per-kind enum would make the two files harder to read against each other than the
/// `Option`s cost.
#[derive(Debug, Clone, Deserialize)]
pub struct Fixture {
    pub name: String,
    pub kind: String,
    pub seconds: f64,
    #[serde(default)]
    pub amp: f64,
    #[serde(default)]
    pub hz: f64,
    #[serde(default)]
    pub hz_end: f64,
    #[serde(default)]
    pub decay: f64,
    #[serde(default)]
    pub bpm: f64,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub partials: usize,
    #[serde(default)]
    pub semitones: Vec<f64>,
    pub probe: Probe,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Probe {
    pub len: usize,
    pub at: Vec<f64>,
    pub rms: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub sample_rate: u32,
    pub embedding_dim: usize,
    pub fixtures: Vec<Fixture>,
}

impl Manifest {
    /// Loads `fixtures.json`, or explains that the export has never been run.
    pub fn load(dir: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(dir.join("fixtures.json")).ok()?;
        match serde_json::from_str(&raw) {
            Ok(manifest) => Some(manifest),
            Err(e) => panic!("tests/fixtures/clap/fixtures.json is unreadable: {e}"),
        }
    }
}

const SAMPLE_RATE: f64 = 48_000.0;

impl Fixture {
    /// The waveform, mono at 48 kHz.
    pub fn synthesize(&self) -> Vec<f32> {
        let n = (self.seconds * SAMPLE_RATE) as usize;
        let sr = SAMPLE_RATE;
        let amp = self.amp;

        let samples: Vec<f64> = match self.kind.as_str() {
            "silence" => vec![0.0; n],
            "dc" => vec![amp; n],
            "sine" => (0..n)
                .map(|i| amp * (2.0 * PI * self.hz * i as f64 / sr).sin())
                .collect(),
            "harmonics" => (0..n)
                .map(|i| {
                    let mut acc = 0.0;
                    for k in 1..=self.partials {
                        let hz = self.hz * k as f64;
                        if hz >= sr / 2.0 {
                            break;
                        }
                        acc += (2.0 * PI * hz * i as f64 / sr).sin() / k as f64;
                    }
                    amp * acc
                })
                .collect(),
            "chord" => {
                let freqs: Vec<f64> = self
                    .semitones
                    .iter()
                    .map(|s| self.hz * 2f64.powf(s / 12.0))
                    .collect();
                (0..n)
                    .map(|i| {
                        amp * freqs
                            .iter()
                            .map(|f| (2.0 * PI * f * i as f64 / sr).sin())
                            .sum::<f64>()
                    })
                    .collect()
            }
            "noise" => xorshift(self.seed, n)
                .into_iter()
                .map(|v| amp * v)
                .collect(),
            "kick" => {
                let mut phase = 0.0;
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        let hz = self.hz * (1.0 + 3.0 * (-40.0 * t).exp());
                        phase += 2.0 * PI * hz / sr;
                        amp * (-self.decay * t).exp() * phase.sin()
                    })
                    .collect()
            }
            "snare" => {
                let noise = xorshift(self.seed, n);
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        let env = (-self.decay * t).exp();
                        let body = (2.0 * PI * self.hz * t).sin();
                        amp * env * (0.6 * noise[i] + 0.4 * body)
                    })
                    .collect()
            }
            "hat" => {
                let noise = xorshift(self.seed, n);
                let mut prev = 0.0;
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sr;
                        let high = noise[i] - prev;
                        prev = noise[i];
                        amp * (-self.decay * t).exp() * high
                    })
                    .collect()
            }
            "clicks" => {
                let period = (sr * 60.0 / self.bpm) as usize;
                (0..n)
                    .map(|i| {
                        let t = (i % period) as f64 / sr;
                        amp * (-self.decay * t).exp()
                    })
                    .collect()
            }
            "chirp" => (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    let phase = 2.0
                        * PI
                        * (self.hz * t + 0.5 * (self.hz_end - self.hz) * t * t / self.seconds);
                    amp * phase.sin()
                })
                .collect(),
            other => panic!("{}: unknown fixture kind {other:?}", self.name),
        };

        samples.into_iter().map(|s| s as f32).collect()
    }

    /// Asserts this generator still agrees with the one that produced the oracle.
    ///
    /// Runs before any embedding comparison so that a drifted generator reports itself
    /// rather than showing up as a mysterious cosine of 0.8.
    pub fn check_probe(&self, samples: &[f32]) {
        assert_eq!(
            samples.len(),
            self.probe.len,
            "{}: the Rust and Python fixture generators disagree on length",
            self.name
        );

        let step = (samples.len() / 16).max(1);
        for (i, &want) in self.probe.at.iter().enumerate() {
            let index = (i * step).min(samples.len() - 1);
            let got = f64::from(samples[index]);
            assert!(
                (got - want).abs() < 1e-5,
                "{}: sample {index} is {got}, the oracle was generated from {want}. \
                 The Rust and Python fixture generators have drifted -- reconcile \
                 tests/support/fixtures.rs with synthesize() in scripts/export_clap_onnx.py \
                 before trusting any parity result.",
                self.name
            );
        }

        let rms = (samples
            .iter()
            .map(|&s| f64::from(s) * f64::from(s))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt();
        assert!(
            (rms - self.probe.rms).abs() < 1e-5,
            "{}: RMS is {rms}, the oracle was generated from {}",
            self.name,
            self.probe.rms
        );
    }
}

/// The generator in [`super::noise`], in `f64`, so the Python mirror can reproduce it
/// exactly without either side depending on a PRNG crate.
fn xorshift(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f64 / 8_388_608.0) - 1.0
        })
        .collect()
}

/// Reads the committed reference embeddings, or `None` if the export has not been run.
///
/// `count * dim` f32, little-endian, in `fixtures.json` order.
pub fn reference_embeddings(dir: &Path, count: usize, dim: usize) -> Option<Vec<f32>> {
    let raw = std::fs::read(dir.join("reference_embeddings.f32")).ok()?;
    assert_eq!(
        raw.len(),
        count * dim * 4,
        "reference_embeddings.f32 holds {} bytes, expected {} fixtures x {dim} f32",
        raw.len(),
        count
    );
    Some(
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

/// Cosine similarity of two equal-length vectors.
///
/// Both sides are L2-normalized already, so this is a dot product -- but computing it
/// properly costs nothing and means a normalization bug shows up here instead of hiding.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}
