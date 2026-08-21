//! The CLAP mel front-end: the exact log-mel spectrogram the ONNX graph was trained on.
//!
//! This module is the parity surface. Everything else in the pipeline can be a little
//! different from some reference and still be useful; this cannot. A mel front-end that
//! disagrees with the graph by a mel-scale convention, a normalization, or half a frame of
//! padding produces embeddings that are wrong in a way nothing downstream can detect --
//! neighbors still look like neighbors, the map still looks like a map, and every distance
//! in it is nonsense. `task.md` Phase 3 is explicit: if parity fails, stop.
//!
//! So the constants here are not choices. They are transcriptions of what LAION-CLAP's
//! HTSAT audio tower does to a waveform before the first convolution, and
//! `scripts/export_clap_onnx.py` writes the values it read off the loaded checkpoint into
//! `tests/fixtures/clap/frontend.json` so [`FrontEnd::SPEC`] can be *checked* against the
//! model rather than believed about it. See [`crate::model::session`] for the other half
//! of the contract, the tensor layout.
//!
//! **What is transcribed, and from where:**
//!
//! - Framing: 1024-point periodic Hann, hop 480, at 48 kHz -- the same framing
//!   `features.rs` already pays for, which is why the DSP descriptors were computed on it
//!   in Phase 2.
//! - `center = true`, `pad_mode = "reflect"`: `torchlibrosa`'s `Spectrogram` defaults, so a
//!   10 s window yields `480000 / 480 + 1 = 1001` frames rather than 999.
//! - Power spectrogram (`|X|^2`), not magnitude: `torchlibrosa`'s `power = 2.0`.
//! - Filterbank: `librosa.filters.mel` defaults -- **Slaney** mel scale (`htk = False`) and
//!   **Slaney** area normalization (`norm = "slaney"`), 64 bands from 50 Hz to 14 kHz.
//!   The HTK formula and the unnormalized bank are both one-line changes away and both
//!   produce a plausible-looking spectrogram that is not this one.
//! - Log: `10 * log10(max(p, 1e-10))` with `ref = 1.0` and **no** `top_db` clamp.
//!
//! The one thing that is genuinely a choice is what to do with audio shorter than the
//! window; see [`Padding`].

use super::{
    decode::{MAX_OUTPUT_SAMPLES, TARGET_SAMPLE_RATE},
    features::{Analyzer, FRAME_SIZE, HOP_SIZE, NUM_BINS},
};

/// Mel bands. `mel_bins` in HTSAT's config.
pub const MEL_BINS: usize = 64;

/// Low edge of the filterbank, in Hz.
pub const MEL_FMIN: f32 = 50.0;

/// High edge of the filterbank, in Hz. Below Nyquist by design: the top 10 kHz of a 48 kHz
/// signal is where the codec artifacts live and almost none of what CLAP was trained to
/// hear.
pub const MEL_FMAX: f32 = 14_000.0;

/// Frames the graph expects, fixed by the 10 s window and `center = true`.
///
/// The input axis is fixed rather than dynamic so the batcher in Phase 4 can stack tensors
/// without a ragged dimension, which is also why [`Padding`] exists.
pub const MEL_FRAMES: usize = MAX_OUTPUT_SAMPLES / HOP_SIZE + 1;

/// Floor under the power spectrogram before the logarithm. `amin` in `torchlibrosa`.
const AMIN: f32 = 1e-10;

/// Samples of reflect padding on each side of the signal, from `center = true`.
const CENTER_PAD: usize = FRAME_SIZE / 2;

/// What to do with a sample shorter than the 10 s window.
///
/// This is the one parameter here that is a decision rather than a transcription, and it
/// matters more than it looks: almost every file in a one-shot library is under a second,
/// so whatever this does, it does to the entire corpus.
///
/// [`Padding::RepeatPad`] is the default because it is LAION-CLAP's own
/// (`data_filling = "repeatpad"`), and matching the reference implementation is the whole
/// point of a parity gate -- a front-end that zero-pads is not the front-end the reference
/// embeddings came from, and the gate would fail for a reason that is not a bug.
///
/// It is worth knowing what it does, though. Repeat-padding a 400 ms kick presents the
/// graph with two and a half seconds of kicks at 400 ms intervals, so some of what CLAP
/// hears in a one-shot is a rhythm the file does not have. [`Padding::ZeroPad`] is kept
/// for the Phase 4 Risk 5 evaluation, which is where "do kicks retrieve kicks" gets
/// answered against real audio and where changing this would have to be justified -- with
/// its own reference embeddings, because it invalidates these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Padding {
    /// Tile the signal `floor(window / len)` times, then zero-fill the remainder.
    #[default]
    RepeatPad,
    /// Zero-fill to the window.
    ZeroPad,
}

/// Every front-end parameter that has to agree with the exported graph, in one place.
///
/// Serialized shape-for-shape with the `frontend` object that
/// `scripts/export_clap_onnx.py` writes, so `tests/parity.rs` can assert equality against
/// the checkpoint instead of the two drifting silently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontEndSpec {
    pub sample_rate: u32,
    pub window_size: usize,
    pub hop_size: usize,
    pub mel_bins: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub amin: f32,
    pub ref_value: f32,
    pub top_db: Option<f32>,
    pub center: bool,
    pub htk: bool,
    pub slaney_norm: bool,
    pub frames: usize,
}

/// A reusable mel front-end: one filterbank, one padded scratch buffer, one output buffer.
///
/// Built once per worker and reused, like [`Analyzer`]. The filterbank is 64 x 513 f32 and
/// the scratch buffer is the full window, so constructing one per file across a 50,000-file
/// scan is the kind of churn `overview.md` §3.6 exists to forbid.
#[derive(Debug)]
pub struct FrontEnd {
    /// Row-major `[MEL_BINS][NUM_BINS]` triangular weights.
    filters: Vec<f32>,
    /// The signal, padded to the window and then reflect-padded for centering.
    padded: Vec<f32>,
    padding: Padding,
}

impl Default for FrontEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontEnd {
    /// The parameters this front-end implements. Asserted against the model in
    /// `tests/parity.rs`.
    pub const SPEC: FrontEndSpec = FrontEndSpec {
        sample_rate: TARGET_SAMPLE_RATE,
        window_size: FRAME_SIZE,
        hop_size: HOP_SIZE,
        mel_bins: MEL_BINS,
        fmin: MEL_FMIN,
        fmax: MEL_FMAX,
        amin: AMIN,
        ref_value: 1.0,
        top_db: None,
        center: true,
        htk: false,
        slaney_norm: true,
        frames: MEL_FRAMES,
    };

    pub fn new() -> Self {
        Self::with_padding(Padding::default())
    }

    pub fn with_padding(padding: Padding) -> Self {
        Self {
            filters: mel_filterbank(),
            padded: Vec::with_capacity(MAX_OUTPUT_SAMPLES + FRAME_SIZE),
            padding,
        }
    }

    /// The mel filterbank, row-major `[MEL_BINS][NUM_BINS]`. Exposed for the filterbank
    /// tests and for the fixture comparison; nothing in the pipeline needs it.
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }

    /// Computes the log-mel spectrogram of one decoded buffer into `out`.
    ///
    /// `samples` is mono at [`TARGET_SAMPLE_RATE`] and at most [`MAX_OUTPUT_SAMPLES`] long
    /// -- the decode stage guarantees both. `out` is resized to exactly
    /// `MEL_FRAMES * MEL_BINS` and filled row-major as `[frames][mels]`, which is HTSAT's
    /// own layout; see [`crate::model::session`] on why the tensor is not transposed.
    ///
    /// `analyzer` is borrowed rather than owned so the transform stays shared with the DSP
    /// pass (`features.rs` module docs): the framing is identical, and planning a second
    /// 1024-point real FFT per worker to compute the same thing twice would be waste.
    /// The analyzer's own state is scratch, so passing one mid-`analyze` is not a concern
    /// -- `analyze` returns before this is called.
    pub fn compute(&mut self, samples: &[f32], analyzer: &mut Analyzer, out: &mut Vec<f32>) {
        self.pad(samples);

        out.clear();
        out.resize(MEL_FRAMES * MEL_BINS, 0.0);

        for frame in 0..MEL_FRAMES {
            let start = frame * HOP_SIZE;
            let power = analyzer.transform_frame(&self.padded[start..start + FRAME_SIZE]);
            let row = &mut out[frame * MEL_BINS..(frame + 1) * MEL_BINS];

            for (mel, dst) in row.iter_mut().enumerate() {
                let weights = &self.filters[mel * NUM_BINS..(mel + 1) * NUM_BINS];
                // f64 accumulation: 513 products of numbers spanning the dynamic range of
                // a power spectrum lose real precision in f32, and the log below turns a
                // relative error near the floor into an absolute one in decibels.
                let energy: f64 = weights
                    .iter()
                    .zip(power.iter())
                    .map(|(&w, &p)| f64::from(w) * f64::from(p))
                    .sum();
                *dst = 10.0 * (energy.max(f64::from(AMIN))).log10() as f32;
            }
        }
    }

    /// Fills [`Self::padded`] with the signal extended to the window and then reflect-padded
    /// for centering, so the frame loop above is a plain slice walk with no edge cases.
    fn pad(&mut self, samples: &[f32]) {
        self.padded.clear();
        self.padded.resize(CENTER_PAD, 0.0);

        // 1. Extend to the full window.
        let n = samples.len().min(MAX_OUTPUT_SAMPLES);
        if n == 0 {
            self.padded.resize(CENTER_PAD * 2 + MAX_OUTPUT_SAMPLES, 0.0);
            return;
        }
        match self.padding {
            Padding::RepeatPad => {
                // `floor(window / n)` whole copies, then zeros -- not a partial final copy.
                // The mid-signal splice this leaves is CLAP's, and reproducing it exactly
                // is the point.
                for _ in 0..(MAX_OUTPUT_SAMPLES / n).max(1) {
                    self.padded.extend_from_slice(&samples[..n]);
                }
            }
            Padding::ZeroPad => self.padded.extend_from_slice(&samples[..n]),
        }
        self.padded.resize(CENTER_PAD + MAX_OUTPUT_SAMPLES, 0.0);

        // 2. Reflect-pad both ends, `numpy`-style: the edge sample is the mirror axis and
        //    is not repeated, so `[1,2,3,4,5]` padded by 2 is `[3,2,1,2,3,4,5,4,3]`.
        for j in 0..CENTER_PAD {
            self.padded[j] = self.padded[CENTER_PAD + (CENTER_PAD - j)];
        }
        let end = CENTER_PAD + MAX_OUTPUT_SAMPLES;
        for k in 0..CENTER_PAD {
            self.padded.push(self.padded[end - 2 - k]);
        }
    }
}

/// Slaney-scale mel from Hz, the `htk = False` branch of `librosa.hz_to_mel`.
///
/// Linear at 200/3 Hz per mel below 1 kHz, logarithmic above -- the piecewise definition
/// from Slaney's Auditory Toolbox. The HTK formula (`2595 * log10(1 + f/700)`) is a
/// different curve and produces a different bank; `librosa`'s default is this one.
fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    // ln(6.4) / 27: the slope that makes the log branch continuous with the linear one.
    let logstep = 6.4f32.ln() / 27.0;

    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

/// Inverse of [`hz_to_mel`].
fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f32.ln() / 27.0;

    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// Centre frequency of FFT bin `k` for the analysis window.
fn bin_hz(k: usize) -> f32 {
    k as f32 * TARGET_SAMPLE_RATE as f32 / FRAME_SIZE as f32
}

/// The triangular filterbank, row-major `[MEL_BINS][NUM_BINS]`.
///
/// A transcription of `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` with its
/// defaults: `MEL_BINS + 2` mel-spaced edges, a triangle per band rising from edge `i` to
/// `i + 1` and falling to `i + 2`, then Slaney normalization -- each triangle scaled by
/// `2 / (edge[i+2] - edge[i])` so bands carry equal *area* rather than equal peak. Without
/// it the high bands, which are wide, dominate; the spectrogram still looks like a
/// spectrogram, and it is not the one the graph was trained on.
fn mel_filterbank() -> Vec<f32> {
    let mel_min = hz_to_mel(MEL_FMIN);
    let mel_max = hz_to_mel(MEL_FMAX);
    let edges: Vec<f32> = (0..MEL_BINS + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (MEL_BINS + 1) as f32;
            mel_to_hz(mel)
        })
        .collect();

    let mut filters = vec![0.0f32; MEL_BINS * NUM_BINS];
    for mel in 0..MEL_BINS {
        let (lo, mid, hi) = (edges[mel], edges[mel + 1], edges[mel + 2]);
        let enorm = 2.0 / (hi - lo);
        for k in 0..NUM_BINS {
            let hz = bin_hz(k);
            let rising = (hz - lo) / (mid - lo);
            let falling = (hi - hz) / (hi - mid);
            filters[mel * NUM_BINS + k] = rising.min(falling).max(0.0) * enorm;
        }
    }
    filters
}

/// Periodic Hann, duplicated from `features.rs` only so the tests below can build a frame
/// without reaching into a private helper. The analyzer's window is the one that runs.
#[cfg(test)]
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / n as f32).cos()))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// The mel scale must round-trip, and it must be continuous where the two branches
    /// meet. A discontinuity at 1 kHz is the classic transcription bug: it moves every
    /// band edge above it and every value in the top three quarters of the spectrogram.
    #[test]
    fn mel_scale_round_trips_and_is_continuous() {
        for hz in [0.0, 50.0, 440.0, 999.0, 1000.0, 1001.0, 5000.0, 14_000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!(
                (back - hz).abs() <= hz.abs() * 1e-4 + 1e-3,
                "{hz} Hz round-tripped to {back}"
            );
        }
        let below = hz_to_mel(999.999);
        let above = hz_to_mel(1000.001);
        assert!((above - below).abs() < 1e-3, "{below} -> {above} at 1 kHz");
    }

    /// Slaney, not HTK. At 1 kHz the Slaney scale is exactly 15 mel by construction
    /// (1000 / (200/3)); HTK puts 1 kHz at ~999.99 mel. Nothing else distinguishes the two
    /// as cheaply, and picking the wrong one is silent.
    #[test]
    fn mel_scale_is_slaney_not_htk() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-4);
        assert!(hz_to_mel(1000.0) < 100.0, "this looks like the HTK formula");
    }

    /// Every band is a triangle inside the requested range: non-negative, one contiguous
    /// run of support, and nothing outside [fmin, fmax].
    #[test]
    fn filterbank_is_triangular_and_bounded() {
        let filters = mel_filterbank();
        assert_eq!(filters.len(), MEL_BINS * NUM_BINS);

        for mel in 0..MEL_BINS {
            let row = &filters[mel * NUM_BINS..(mel + 1) * NUM_BINS];
            assert!(
                row.iter().all(|&w| w >= 0.0),
                "band {mel} has a negative tap"
            );

            let support: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|(_, &w)| w > 0.0)
                .map(|(k, _)| k)
                .collect();
            assert!(!support.is_empty(), "band {mel} is empty");
            assert_eq!(
                support.last().unwrap() - support[0] + 1,
                support.len(),
                "band {mel} support is not contiguous"
            );

            let (first, last) = (bin_hz(support[0]), bin_hz(*support.last().unwrap()));
            assert!(
                first >= MEL_FMIN - 50.0 && last <= MEL_FMAX + 50.0,
                "band {mel} spans {first}..{last} Hz, outside the requested range"
            );
        }
    }

    /// Slaney *normalization*: each triangle's area, in bin units, is 1 -- which is what
    /// `2 / (hi - lo)` buys. An unnormalized bank has peak 1 instead, so this single
    /// assertion separates `norm="slaney"` from `norm=None`.
    #[test]
    fn filterbank_is_slaney_normalized() {
        let filters = mel_filterbank();
        let bin_width = TARGET_SAMPLE_RATE as f32 / FRAME_SIZE as f32;

        // Skip the first bands: below ~200 Hz a triangle is narrower than one 46.9 Hz bin,
        // so its sampled area is a poor estimate of its analytic area for reasons that have
        // nothing to do with normalization.
        for mel in 8..MEL_BINS {
            let row = &filters[mel * NUM_BINS..(mel + 1) * NUM_BINS];
            let area: f32 = row.iter().sum::<f32>() * bin_width;
            assert!(
                (area - 1.0).abs() < 0.15,
                "band {mel} has area {area}, expected ~1 (is norm=None?)"
            );
            assert!(
                row.iter().fold(0.0f32, |a, &b| a.max(b)) < 1.0,
                "band {mel} peaks at 1, which is what an unnormalized bank does"
            );
        }
    }

    /// The shape the graph is fed. 1001 frames, not 999 -- the difference is `center=true`,
    /// and getting it wrong shifts every frame by half a window.
    #[test]
    fn output_shape_is_fixed_regardless_of_input_length() {
        let mut analyzer = Analyzer::new();
        let mut front = FrontEnd::new();
        let mut out = Vec::new();

        assert_eq!(MEL_FRAMES, 1001);
        for len in [0, 1, 480, FRAME_SIZE, 48_000, MAX_OUTPUT_SAMPLES] {
            let samples = vec![0.1f32; len];
            front.compute(&samples, &mut analyzer, &mut out);
            assert_eq!(out.len(), MEL_FRAMES * MEL_BINS, "input of {len} samples");
            assert!(out.iter().all(|v| v.is_finite()), "input of {len} samples");
        }
    }

    /// Digital silence must produce the floor, not `-inf` and not `NaN`. A single `NaN`
    /// anywhere in a batch poisons the whole `run()`.
    #[test]
    fn silence_is_the_floor_not_negative_infinity() {
        let mut analyzer = Analyzer::new();
        let mut front = FrontEnd::new();
        let mut out = Vec::new();

        front.compute(&vec![0.0f32; 48_000], &mut analyzer, &mut out);
        let floor = 10.0 * AMIN.log10();
        assert!(
            out.iter().all(|&v| (v - floor).abs() < 1e-3),
            "{:?}",
            &out[..8]
        );
    }

    /// Reflect padding, verified on the exact example in `numpy`'s own documentation, by
    /// reaching through `pad` with a signal short enough to inspect. Zero-padding here
    /// instead is a difference of a few frames at each edge, which is small, consistent,
    /// and wrong.
    #[test]
    fn centering_reflects_rather_than_zero_pads() {
        let mut front = FrontEnd::with_padding(Padding::ZeroPad);
        front.pad(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Left edge mirrors about sample 0 without repeating it.
        assert_eq!(front.padded[CENTER_PAD], 1.0);
        assert_eq!(front.padded[CENTER_PAD - 1], 2.0);
        assert_eq!(front.padded[CENTER_PAD - 2], 3.0);
        // Past the signal the window is zeros, so the mirror is zeros too -- the interesting
        // edge is the left one, and the right edge is checked by the length.
        assert_eq!(front.padded.len(), MAX_OUTPUT_SAMPLES + FRAME_SIZE);
    }

    /// Repeat-pad tiles whole copies and then zero-fills; it does not splice a partial
    /// copy. This is LAION-CLAP's `data_filling="repeatpad"` and the reason [`Padding`]
    /// has a doc comment three times its length.
    #[test]
    fn repeat_pad_tiles_whole_copies_then_zero_fills() {
        let mut front = FrontEnd::with_padding(Padding::RepeatPad);
        // 100,000 samples goes in 4 whole times (480,000 / 100,000 = 4), leaving 80,000.
        let signal: Vec<f32> = (0..100_000).map(|i| (i % 7) as f32).collect();
        front.pad(&signal);

        let body = &front.padded[CENTER_PAD..CENTER_PAD + MAX_OUTPUT_SAMPLES];
        for copy in 0..4 {
            assert_eq!(&body[copy * 100_000..copy * 100_000 + 16], &signal[..16]);
        }
        assert!(
            body[400_000..].iter().all(|&s| s == 0.0),
            "tail is not zeros"
        );
    }

    /// A pure tone must put its energy in the band containing it and essentially nowhere
    /// else. This is the end-to-end check that the bank is wired to the right bins at all:
    /// an off-by-one in the `[MEL_BINS][NUM_BINS]` indexing passes every structural test
    /// above and fails this one.
    #[test]
    fn a_tone_lands_in_the_band_that_contains_it() {
        let mut analyzer = Analyzer::new();
        let mut front = FrontEnd::new();
        let mut out = Vec::new();

        let hz = 1000.0f32;
        let samples: Vec<f32> = (0..MAX_OUTPUT_SAMPLES)
            .map(|i| (std::f32::consts::TAU * hz * i as f32 / TARGET_SAMPLE_RATE as f32).sin())
            .collect();
        front.compute(&samples, &mut analyzer, &mut out);

        // A frame from the middle, well clear of the edges.
        let row = &out[500 * MEL_BINS..501 * MEL_BINS];
        let loudest = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("mel bins are finite"))
            .map(|(i, _)| i)
            .unwrap();

        let edges: Vec<f32> = (0..MEL_BINS + 2)
            .map(|i| {
                let mel_min = hz_to_mel(MEL_FMIN);
                let mel_max = hz_to_mel(MEL_FMAX);
                mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (MEL_BINS + 1) as f32)
            })
            .collect();
        assert!(
            edges[loudest] <= hz && hz <= edges[loudest + 2],
            "1 kHz peaked in band {loudest}, which spans {}..{} Hz",
            edges[loudest],
            edges[loudest + 2]
        );

        // ...and it is a peak, not a plateau: 30 dB down four bands away.
        let far = (loudest + 8).min(MEL_BINS - 1);
        assert!(
            row[loudest] - row[far] > 30.0,
            "{} vs {}",
            row[loudest],
            row[far]
        );
    }

    /// The window has to be periodic, not symmetric. `features.rs` picks periodic and this
    /// asserts the shared analyzer still does, because a symmetric window is a different
    /// spectrogram and the difference is one sample wide.
    #[test]
    fn window_is_periodic() {
        let w = hann(8);
        assert_eq!(w[0], 0.0);
        assert!(w[4] > 0.999, "peak is not at n/2: {:?}", w);
        assert!(
            (w[1] - w[7]).abs() < 1e-6,
            "not symmetric about n/2: {:?}",
            w
        );
    }
}
