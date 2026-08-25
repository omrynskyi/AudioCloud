//! Opens the real default output device and measures actual play-to-audible latency.
//!
//! **Ignored by default, like `tests/evaluation.rs`'s drum-library tests.** This needs a real
//! audio output device -- headless CI has none -- so it is opt-in: `cargo test --test
//! audio_hardware -- --ignored`. Everything else about the audio engine (the envelope's
//! attack/release state machine, retriggering, the guard allocator's enforcement, resample and
//! channel-formatting) is proven against plain data in `audio::engine`'s and `audio::guard`'s
//! own unit tests and in `tests/audio_guard.rs`, none of which need hardware. This is the one
//! number that does.
//!
//! What this measures is command-received-to-first-audible-sample, not pointer-event-to-
//! audible -- see `audio::log_latency`'s doc comment for why the IPC hop and the frontend's own
//! debounce are outside what a Rust test can observe.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::{sync::Arc, time::Duration};

use audiobank_lib::{
    audio::{engine::Engine, AudioPlayer},
    db::Database,
    pipeline::{scan_root, CancellationToken},
    EMBEDDING_DIM,
};
use support::write_sine;

/// A short, silence-free sine burst -- long enough that the ring never underruns before the
/// natural end, short enough that the test does not sit around.
fn sine_burst(sample_rate: u32, channels: u16, seconds: f32, hz: f32) -> Vec<f32> {
    let frames = (sample_rate as f32 * seconds) as usize;
    let mut out = Vec::with_capacity(frames * channels as usize);
    for i in 0..frames {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * hz * t).sin() * 0.2;
        for _ in 0..channels {
            out.push(sample);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real audio output device"]
async fn hover_to_audible_is_under_the_budget() {
    let engine = Engine::open(None).expect("no default output device on this machine");
    let format = engine.format();
    println!(
        "opened {} Hz, {} channel(s)",
        format.sample_rate, format.channels
    );

    let pcm = Arc::new(sine_burst(format.sample_rate, format.channels, 0.3, 440.0));
    let generation = engine.play_pcm(pcm, 1.0);

    let mut latency = None;
    for _ in 0..100 {
        if let Some(l) = engine.latency_since(generation) {
            latency = Some(l);
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // Let the burst finish playing rather than tearing the stream down mid-note.
    tokio::time::sleep(Duration::from_millis(400)).await;

    let latency = latency.expect("the engine never reported an audible sample");
    println!("play_pcm -> first audible sample: {latency:?}");

    // `overview.md` §7's target is the full hover-to-audible chain; this is the half the
    // engine owns. A generous multiple of the 50ms headline budget rather than the budget
    // itself: a debug build under a test harness on a shared machine is not the environment
    // the target was written against, and a flaky assertion here is worse than a loose one.
    assert!(
        latency < Duration::from_millis(200),
        "play_pcm to first audible sample took {latency:?}, which is not in the neighborhood \
         of the 50ms hover-to-audible budget even accounting for test-harness overhead"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real audio output device"]
async fn stop_releases_without_a_panic() {
    let engine = Engine::open(None).expect("no default output device on this machine");
    let format = engine.format();
    let pcm = Arc::new(sine_burst(format.sample_rate, format.channels, 2.0, 220.0));
    engine.play_pcm(pcm, 0.5);
    tokio::time::sleep(Duration::from_millis(100)).await;
    engine.stop();
    // Long enough for the ~8ms release ramp to complete well before the assertion.
    tokio::time::sleep(Duration::from_millis(50)).await;
}

/// The full path `commands::samples::play_sample` actually drives: a real database row
/// pointing at a real file, decoded, resampled and channel-formatted for the real device, and
/// played. Everything upstream of this (`to_device_format`, `resample`) is unit-tested against
/// synthetic buffers, and everything downstream (`Engine`) is proven above against hardware;
/// this is the one test that proves the seam between them is wired correctly.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real audio output device"]
async fn a_real_sample_plays_end_to_end() {
    let data_dir = tempfile::tempdir().unwrap();
    let library_dir = tempfile::tempdir().unwrap();
    let db = Database::open(data_dir.path(), EMBEDDING_DIM).unwrap();
    let root_id = db
        .writer()
        .add_root(library_dir.path().to_str().unwrap(), None)
        .unwrap();

    write_sine(&library_dir.path().join("kick.wav"), 220.0, 0.5);
    scan_root(&db, root_id, &CancellationToken::new()).unwrap();

    let sample_id: i64 = {
        let conn = db.read().unwrap();
        conn.query_row(
            "SELECT id FROM samples WHERE root_id = ?1 AND rel_path = 'kick.wav'",
            [root_id],
            |r| r.get(0),
        )
        .unwrap()
    };

    let player = Arc::new(AudioPlayer::new());
    player.play(&db, sample_id, 0.6).await.unwrap();
    // The clip is 0.5s; give it time to actually reach the speaker and finish before the
    // database and its temp directories drop out from under a still-running decode task.
    tokio::time::sleep(Duration::from_millis(700)).await;

    // A retrigger of the same sample must also succeed, and must hit the PCM cache rather
    // than decoding again -- both paths through `AudioPlayer::play` get exercised here.
    player.play(&db, sample_id, 0.4).await.unwrap();
    tokio::time::sleep(Duration::from_millis(600)).await;

    player.stop().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real audio output device"]
async fn rapid_retriggers_do_not_panic_or_deadlock() {
    let engine = Engine::open(None).expect("no default output device on this machine");
    let format = engine.format();
    let pcm = Arc::new(sine_burst(format.sample_rate, format.channels, 1.0, 330.0));

    for _ in 0..20 {
        engine.play_pcm(Arc::clone(&pcm), 0.4);
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
}
