//! **The Phase 3 parity gate.**
//!
//! `task.md` is blunt about why this exists: if the Rust mel front-end disagrees with the
//! ONNX graph, every embedding is silently wrong, the map still looks completely plausible,
//! and nothing downstream can detect it. So the front-end is not trusted -- it is checked,
//! against embeddings produced by Python CLAP itself, at cosine similarity **> 0.999**.
//!
//! Three tiers, in increasing order of what they need on disk:
//!
//! 1. [`the_two_fixture_generators_agree`] needs only the repo. It proves the twenty
//!    synthetic signals Rust builds are the ones the oracle was recorded from.
//! 2. [`the_front_end_matches_the_checkpoint`] needs `frontend.json`, which
//!    `scripts/export_clap_onnx.py` reads off the loaded checkpoint. It proves
//!    `FrontEnd::SPEC` is not a guess.
//! 3. [`clap_embeddings_match_the_reference`] needs the reference embeddings **and** the
//!    200 MB model, so it does not run in CI. It is the gate itself.
//!
//! What keeps tiers 2 and 3 from quietly never running is
//! [`the_parity_gate_is_not_silently_disabled`]: a build whose `ModelRelease::CURRENT` is
//! pinned **must** have the oracle committed. Pinning a release without recording what it
//! produces is exactly the state `task.md`'s "if parity fails, stop" is written to prevent,
//! and this test makes it impossible to reach by accident.
//!
//! To run tier 3 locally, once the model is downloaded:
//!
//! ```sh
//! AUDIOBANK_MODEL_PATH=~/Library/Application\ Support/com.audiobank.app/models/clap-audio-v1.onnx \
//!   cargo test --manifest-path src-tauri/Cargo.toml --test parity -- --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::path::{Path, PathBuf};

use audiobank_lib::{
    model::{session::ModelSession, ModelRelease},
    pipeline::{
        features::Analyzer,
        mel::{FrontEnd, MEL_BINS, MEL_FRAMES},
    },
    EMBEDDING_DIM,
};
use support::fixtures::{self, Manifest};

/// Points the gate at an already-downloaded model. Without it, tier 3 skips.
const MODEL_PATH_ENV: &str = "AUDIOBANK_MODEL_PATH";

/// Cosine similarity the front-end must reach against Python CLAP (`task.md` Phase 3).
const PARITY_FLOOR: f64 = 0.999;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(fixtures::FIXTURE_DIR)
}

fn manifest() -> Manifest {
    Manifest::load(&fixture_dir()).expect(
        "tests/fixtures/clap/fixtures.json is missing. Regenerate it with \
         `python3 scripts/export_clap_onnx.py --recipes-only`.",
    )
}

/// Tier 1. The oracle is only an oracle if both sides are looking at the same audio.
#[test]
fn the_two_fixture_generators_agree() {
    let manifest = manifest();
    assert_eq!(manifest.sample_rate, 48_000);
    assert_eq!(manifest.embedding_dim, EMBEDDING_DIM);
    assert!(
        manifest.fixtures.len() >= 20,
        "task.md asks for 20 diverse fixtures; the manifest has {}",
        manifest.fixtures.len()
    );

    for fixture in &manifest.fixtures {
        let samples = fixture.synthesize();
        fixture.check_probe(&samples);
    }
}

/// Every fixture must survive the front-end without producing a value that would poison a
/// batch. Runs without any of the fixture files, and catches the whole class of "one
/// pathological input NaNs the entire `run()`".
#[test]
fn every_fixture_produces_a_finite_spectrogram() {
    let mut analyzer = Analyzer::new();
    let mut front = FrontEnd::new();
    let mut mel = Vec::new();

    for fixture in &manifest().fixtures {
        let samples = fixture.synthesize();
        front.compute(&samples, &mut analyzer, &mut mel);

        assert_eq!(mel.len(), MEL_FRAMES * MEL_BINS, "{}", fixture.name);
        assert!(
            mel.iter().all(|v| v.is_finite()),
            "{}: the front-end produced a non-finite mel value",
            fixture.name
        );
        // Everything is in decibels against a 1.0 reference with a -100 dB floor, so a
        // value above 0 dB means the filterbank normalization has gone wrong, and one below
        // the floor means the log has.
        let (lo, hi) = mel
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(
            (-100.5..=30.0).contains(&lo) && (-100.5..=30.0).contains(&hi),
            "{}: mel range {lo}..{hi} dB is outside anything the log-mel can produce",
            fixture.name
        );
    }
}

/// Tier 2. `FrontEnd::SPEC` versus what the checkpoint actually does.
///
/// The export script writes `frontend.json` by *reading* the loaded model rather than by
/// restating its own constants, which is what makes this an independent check instead of a
/// tautology.
#[test]
fn the_front_end_matches_the_checkpoint() {
    let path = fixture_dir().join("frontend.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        println!(
            "SKIP: {} has not been generated. Run scripts/export_clap_onnx.py against the \
             real checkpoint; until then the front-end constants are unverified.",
            path.display()
        );
        return;
    };

    let json: serde_json::Value = serde_json::from_str(&raw).expect("frontend.json is malformed");
    let spec = FrontEnd::SPEC;

    let integer = |key: &str| -> u64 {
        json[key]
            .as_u64()
            .unwrap_or_else(|| panic!("frontend.json has no integer {key}"))
    };
    let float = |key: &str| -> f64 {
        json[key]
            .as_f64()
            .unwrap_or_else(|| panic!("frontend.json has no number {key}"))
    };
    let boolean = |key: &str| -> bool {
        json[key]
            .as_bool()
            .unwrap_or_else(|| panic!("frontend.json has no boolean {key}"))
    };

    assert_eq!(u64::from(spec.sample_rate), integer("sample_rate"));
    assert_eq!(spec.window_size as u64, integer("window_size"));
    assert_eq!(spec.hop_size as u64, integer("hop_size"));
    assert_eq!(spec.mel_bins as u64, integer("mel_bins"));
    assert_eq!(spec.frames as u64, integer("frames"));
    assert!((f64::from(spec.fmin) - float("fmin")).abs() < 1e-6);
    assert!((f64::from(spec.fmax) - float("fmax")).abs() < 1e-6);
    assert!((f64::from(spec.amin) - float("amin")).abs() < 1e-16);
    assert!((f64::from(spec.ref_value) - float("ref_value")).abs() < 1e-9);
    assert_eq!(spec.center, boolean("center"));
    assert_eq!(spec.htk, boolean("htk"));
    assert_eq!(spec.slaney_norm, boolean("slaney_norm"));
    assert!(
        spec.top_db.is_none() == json["top_db"].is_null(),
        "top_db: this build clamps at {:?}, the checkpoint at {}",
        spec.top_db,
        json["top_db"]
    );
    assert_eq!(json["pad_mode"], "reflect", "the front-end reflect-pads");
    assert_eq!(json["power"], 2.0, "the front-end uses a power spectrogram");
}

/// Tier 3. **The gate.**
#[test]
fn clap_embeddings_match_the_reference() {
    let dir = fixture_dir();
    let manifest = manifest();

    let Some(reference) =
        fixtures::reference_embeddings(&dir, manifest.fixtures.len(), EMBEDDING_DIM)
    else {
        println!(
            "SKIP: no reference embeddings at {}. Run scripts/export_clap_onnx.py to \
             record the parity oracle.",
            dir.join("reference_embeddings.f32").display()
        );
        return;
    };

    let Some(model) = model_path() else {
        println!(
            "SKIP: no model. Set {MODEL_PATH_ENV} to a downloaded clap_audio.onnx to run \
             the parity gate."
        );
        return;
    };

    let session = ModelSession::open(&model).expect("could not open the model");
    println!(
        "parity gate on {} (provider {})",
        model.display(),
        session.provider().as_str()
    );

    let mut analyzer = Analyzer::new();
    let mut front = FrontEnd::new();
    let mut mel = Vec::new();
    let mut worst = (f64::MAX, String::new());

    for (index, fixture) in manifest.fixtures.iter().enumerate() {
        let samples = fixture.synthesize();
        // Tier 1's check, again, right here: a drifted generator would otherwise present as
        // a parity failure, and the two have completely different fixes.
        fixture.check_probe(&samples);

        front.compute(&samples, &mut analyzer, &mut mel);
        let ours = session.embed(&mel).expect("inference failed");
        let theirs = &reference[index * EMBEDDING_DIM..(index + 1) * EMBEDDING_DIM];
        let similarity = fixtures::cosine(&ours, theirs);

        println!("  {:<14} cos={similarity:.6}", fixture.name);
        if similarity < worst.0 {
            worst = (similarity, fixture.name.clone());
        }
    }

    assert!(
        worst.0 > PARITY_FLOOR,
        "PARITY FAILED: {} reached only cosine {:.6} against Python CLAP (floor {PARITY_FLOOR}).\n\
         Do not proceed to Phase 4. Every embedding produced by a mismatched front-end is \
         silently wrong and the map will look completely plausible.\n\
         Start with src-tauri/src/pipeline/mel.rs: the mel scale (Slaney vs HTK), the \
         filterbank normalization, centering, and the padding mode are the four that \
         produce a plausible-looking spectrogram that is not the right one.",
        worst.1,
        worst.0
    );
}

/// The gate cannot be disabled by omission.
///
/// Tiers 2 and 3 skip when their fixtures are absent, which is correct while the export has
/// never been run -- and would be a disaster if it persisted past the point where a real
/// model ships. So: a pinned release requires a committed oracle. There is no state in
/// which AudioBank downloads a verified model and has never checked what it produces.
#[test]
fn the_parity_gate_is_not_silently_disabled() {
    let dir = fixture_dir();
    let has_oracle =
        dir.join("reference_embeddings.f32").is_file() && dir.join("frontend.json").is_file();

    assert!(
        has_oracle || !ModelRelease::CURRENT.is_pinned(),
        "ModelRelease::CURRENT is pinned to a published model, but {} has no committed \
         parity oracle. Run scripts/export_clap_onnx.py against that exact checkpoint and \
         commit frontend.json and reference_embeddings.f32 -- a pinned model whose \
         front-end was never verified is precisely what task.md Phase 3 forbids shipping.",
        dir.display()
    );
}

/// The model, from the environment or from wherever a real install would have put it.
fn model_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(MODEL_PATH_ENV) {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }

    let home = std::env::var_os("HOME")?;
    let path = Path::new(&home)
        .join("Library/Application Support/com.audiobank.app/models")
        .join(ModelRelease::CURRENT.filename());
    path.is_file().then_some(path)
}
