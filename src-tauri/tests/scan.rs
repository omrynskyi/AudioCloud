//! Phase 2 end-to-end: point the pipeline at a real folder and check what lands in the
//! database.
//!
//! These are integration tests rather than unit tests on purpose. Every exit criterion in
//! `task.md` Phase 2 is a statement about the whole of walk -> decode -> features -> writer,
//! and testing the stages in isolation would leave the wiring between them -- which is where
//! the interesting failures live -- unexercised.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::{path::Path, time::Instant};

use audiobank_lib::{
    db::{queries, Database, SampleStatus},
    pipeline::{scan_root, CancellationToken, ScanReport},
    EMBEDDING_DIM,
};
use support::{noise, sine, write_sine, write_wav, WavFormat};
use tempfile::TempDir;

/// A database and a library root, both under temporary directories that the caller must
/// keep alive.
struct Fixture {
    _data: TempDir,
    library: TempDir,
    db: Database,
    root_id: i64,
}

impl Fixture {
    fn new() -> Self {
        let data = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();
        let db = Database::open(data.path(), EMBEDDING_DIM).unwrap();
        let root_id = db
            .writer()
            .add_root(library.path().to_str().unwrap(), Some("test".into()))
            .unwrap();
        Self {
            _data: data,
            library,
            db,
            root_id,
        }
    }

    fn path(&self, rel: &str) -> std::path::PathBuf {
        self.library.path().join(rel)
    }

    fn scan(&self) -> ScanReport {
        scan_root(&self.db, self.root_id, &CancellationToken::new()).unwrap()
    }

    fn row(&self, rel_path: &str) -> Row {
        let conn = self.db.read().unwrap();
        conn.query_row(
            "SELECT id, filename, ext, status, duration_ms, sample_rate, channels, error,
                    length(content_hash)
             FROM samples WHERE root_id = ?1 AND rel_path = ?2",
            (self.root_id, rel_path),
            |r| {
                Ok(Row {
                    id: r.get(0)?,
                    filename: r.get(1)?,
                    ext: r.get(2)?,
                    status: r.get::<_, String>(3)?,
                    duration_ms: r.get(4)?,
                    sample_rate: r.get(5)?,
                    channels: r.get(6)?,
                    error: r.get(7)?,
                    hash_len: r.get(8)?,
                })
            },
        )
        .unwrap_or_else(|e| panic!("no row for {rel_path}: {e}"))
    }

    fn count(&self) -> i64 {
        queries::count_samples(&self.db.read().unwrap()).unwrap()
    }
}

#[derive(Debug)]
struct Row {
    id: i64,
    filename: String,
    ext: String,
    status: String,
    duration_ms: Option<i64>,
    sample_rate: Option<i64>,
    channels: Option<i64>,
    error: Option<String>,
    hash_len: Option<i64>,
}

/// The headline exit criterion: a real folder scans to the database with correct metadata
/// and DSP features.
#[test]
fn a_folder_of_audio_scans_to_rows_with_metadata_and_features() {
    let fx = Fixture::new();

    write_wav(
        &fx.path("drums/kick.wav"),
        WavFormat::Pcm16,
        48_000,
        1,
        &sine(60.0, 0.9, 0.5, 48_000),
    );
    write_wav(
        &fx.path("keys/chord.wav"),
        WavFormat::Float32,
        44_100,
        2,
        // Interleaved stereo: the same tone in both channels.
        &sine(440.0, 0.4, 2.0, 44_100)
            .iter()
            .flat_map(|&s| [s, s])
            .collect::<Vec<f32>>(),
    );
    // Non-audio cruft, which every sample library is full of.
    std::fs::write(fx.path("drums/kick.wav.asd"), b"analysis").unwrap();
    std::fs::write(fx.path("readme.txt"), b"notes").unwrap();
    std::fs::write(fx.path(".DS_Store"), b"finder").unwrap();

    let report = fx.scan();

    assert_eq!(fx.count(), 2, "only the two audio files should have rows");
    assert_eq!(report.counts.files_added, 2);
    assert_eq!(report.counts.files_failed, 0);
    assert_eq!(report.status, audiobank_lib::db::ScanStatus::Completed);

    let kick = fx.row("drums/kick.wav");
    assert_eq!(kick.filename, "kick.wav");
    assert_eq!(kick.ext, "wav");
    assert_eq!(kick.status, "decoded");
    assert_eq!(kick.sample_rate, Some(48_000));
    assert_eq!(kick.channels, Some(1));
    assert_eq!(kick.duration_ms, Some(500));
    assert_eq!(kick.hash_len, Some(32), "blake3 is 32 bytes");
    assert!(kick.error.is_none());

    let chord = fx.row("keys/chord.wav");
    assert_eq!(
        chord.sample_rate,
        Some(44_100),
        "the source rate is recorded"
    );
    assert_eq!(chord.channels, Some(2));
    assert_eq!(chord.duration_ms, Some(2000));

    // Features are written for both, and describe the audio rather than being placeholders.
    let conn = fx.db.read().unwrap();
    let kick_features = queries::sample_features(&conn, kick.id).unwrap().unwrap();
    let chord_features = queries::sample_features(&conn, chord.id).unwrap().unwrap();

    assert!(
        kick_features.spectral_centroid.unwrap() < chord_features.spectral_centroid.unwrap(),
        "a 60 Hz sine must have a lower centroid than a 440 Hz one"
    );
    assert!((kick_features.peak_db.unwrap() - (-0.9)).abs() < 0.5);
    assert!(kick_features.lufs_integrated.is_some());
    assert!(kick_features.rms_db.unwrap() < 0.0);
}

/// The other headline criterion: corrupt and unsupported files are quarantined without
/// aborting the scan.
#[test]
fn unreadable_files_are_quarantined_and_the_scan_continues() {
    let fx = Fixture::new();

    write_sine(&fx.path("good_before.wav"), 440.0, 0.3);
    // A plausible RIFF header followed by nothing decodable. Long enough to clear the
    // walker's minimum size, so it reaches the decoder and is quarantined rather than
    // filtered out as too small to be audio.
    let mut truncated = b"RIFF\x24\x10\x00\x00WAVEfmt junk".to_vec();
    truncated.extend(std::iter::repeat_n(0xCDu8, 4000));
    std::fs::write(fx.path("truncated.wav"), &truncated).unwrap();
    // An extension we accept over content we cannot possibly decode.
    std::fs::write(fx.path("lying.flac"), vec![0xABu8; 5000]).unwrap();
    write_sine(&fx.path("good_after.wav"), 880.0, 0.3);

    let report = fx.scan();

    assert_eq!(fx.count(), 4, "quarantined files still get rows");
    assert_eq!(report.status, audiobank_lib::db::ScanStatus::Completed);
    assert_eq!(report.counts.files_failed, 2);

    for name in ["truncated.wav", "lying.flac"] {
        let row = fx.row(name);
        assert_eq!(row.status, "decode_failed", "{name}");
        assert!(
            row.error.as_deref().is_some_and(|e| !e.is_empty()),
            "{name} was quarantined with no explanation"
        );
    }

    // The files either side of the failures decoded normally.
    for name in ["good_before.wav", "good_after.wav"] {
        assert_eq!(fx.row(name).status, "decoded", "{name}");
    }

    let conn = fx.db.read().unwrap();
    assert_eq!(
        queries::count_samples_with_status(&conn, SampleStatus::Decoded).unwrap(),
        2
    );
}

/// Exit criterion: a re-scan of an unchanged folder completes in under 5% of the original
/// time.
///
/// The threshold is checked against wall clock, which is why the fixture is 60 files rather
/// than 6 -- at six files the constant costs of opening a scan swamp the ratio and the test
/// measures nothing. The comparison is still conservative: it asserts 10%, not 5%, because a
/// CI runner's IO variance is larger than the margin the real criterion has, and a flaky
/// performance test gets deleted rather than fixed. The recorded 5% number lives in
/// `BENCHMARKS.md`, measured on a real library.
#[test]
fn rescanning_an_unchanged_folder_skips_everything() {
    let fx = Fixture::new();

    for i in 0..60 {
        write_wav(
            &fx.path(&format!("bank{}/hit_{i:03}.wav", i % 4)),
            WavFormat::Pcm16,
            48_000,
            1,
            &sine(100.0 + i as f32 * 20.0, 0.7, 1.0, 48_000),
        );
    }

    let started = Instant::now();
    let first = fx.scan();
    let cold = started.elapsed();
    assert_eq!(first.counts.files_added, 60);
    assert_eq!(first.processed, 60);

    let started = Instant::now();
    let second = fx.scan();
    let warm = started.elapsed();

    assert_eq!(second.counts.files_skipped, 60, "nothing should be re-read");
    assert_eq!(second.counts.files_added, 0);
    assert_eq!(second.processed, 0);
    assert_eq!(fx.count(), 60, "a rescan must not duplicate rows");

    assert!(
        warm.as_secs_f64() < cold.as_secs_f64() * 0.10,
        "rescan took {warm:?} against a cold scan of {cold:?}"
    );
}

/// A file whose contents changed must be re-read even though its path did not.
#[test]
fn a_modified_file_is_reprocessed() {
    let fx = Fixture::new();
    let path = fx.path("moving_target.wav");

    write_wav(
        &path,
        WavFormat::Pcm16,
        48_000,
        1,
        &sine(200.0, 0.8, 1.0, 48_000),
    );
    fx.scan();
    let before = fx.row("moving_target.wav");
    let centroid_before = queries::sample_features(&fx.db.read().unwrap(), before.id)
        .unwrap()
        .unwrap()
        .spectral_centroid
        .unwrap();

    // A different tone, a different length -- so both halves of the (mtime, size) check see
    // a change even on a filesystem with coarse timestamps.
    write_wav(
        &path,
        WavFormat::Pcm16,
        48_000,
        1,
        &sine(4000.0, 0.8, 1.5, 48_000),
    );

    let report = fx.scan();
    assert_eq!(report.counts.files_skipped, 0);
    assert_eq!(report.processed, 1);
    assert_eq!(fx.count(), 1, "the row is updated, not duplicated");

    let after = fx.row("moving_target.wav");
    assert_eq!(after.id, before.id);
    assert_eq!(after.duration_ms, Some(1500));

    let centroid_after = queries::sample_features(&fx.db.read().unwrap(), after.id)
        .unwrap()
        .unwrap()
        .spectral_centroid
        .unwrap();
    assert!(
        centroid_after > centroid_before * 5.0,
        "features were not recomputed: {centroid_before} -> {centroid_after}"
    );
}

/// The same 909 kick under six names is what a sample library actually looks like. Only one
/// of them should be decoded; the rest copy its analysis.
#[test]
fn identical_files_are_deduplicated_within_one_scan() {
    let fx = Fixture::new();
    let audio = sine(330.0, 0.6, 1.0, 48_000);

    for name in [
        "a/kick.wav",
        "b/kick_copy.wav",
        "c/KICK 2.wav",
        "d/kick-03.wav",
    ] {
        write_wav(&fx.path(name), WavFormat::Pcm16, 48_000, 1, &audio);
    }
    // ...plus one that is genuinely different, to prove dedup is keyed on content.
    write_wav(
        &fx.path("e/snare.wav"),
        WavFormat::Pcm16,
        48_000,
        1,
        &noise(42, 48_000),
    );

    let report = fx.scan();

    assert_eq!(fx.count(), 5, "every duplicate still gets its own row");
    assert_eq!(report.counts.files_added, 5);
    // Exactly two decodes -- one kick, one snare -- and three copies. This is an equality,
    // not a lower bound, because the claim-then-publish table makes it deterministic: a memo
    // that only recorded finished work would have all four kicks decoding at once.
    assert_eq!(
        report.processed, 2,
        "the same audio was decoded more than once"
    );
    assert_eq!(report.deduped, 3);

    // Deduplicated rows are complete rows, not stubs: same metadata, same features.
    let conn = fx.db.read().unwrap();
    let reference = fx.row("a/kick.wav");
    let reference_features = queries::sample_features(&conn, reference.id)
        .unwrap()
        .unwrap();

    for name in ["b/kick_copy.wav", "c/KICK 2.wav", "d/kick-03.wav"] {
        let row = fx.row(name);
        assert_eq!(row.status, "decoded", "{name}");
        assert_eq!(row.duration_ms, reference.duration_ms, "{name}");
        assert_eq!(row.sample_rate, reference.sample_rate, "{name}");
        assert_eq!(
            queries::sample_features(&conn, row.id).unwrap().unwrap(),
            reference_features,
            "{name} did not inherit the analysis of its twin"
        );
    }
}

/// Dedup has to survive across scans too: the second scan finds the twin through the
/// `content_hash` index rather than through the in-scan memo.
#[test]
fn a_copy_added_later_is_deduplicated_against_the_original() {
    let fx = Fixture::new();
    let audio = sine(220.0, 0.6, 1.0, 48_000);

    write_wav(
        &fx.path("original.wav"),
        WavFormat::Pcm16,
        48_000,
        1,
        &audio,
    );
    let first = fx.scan();
    assert_eq!(first.processed, 1);
    assert_eq!(first.deduped, 0);

    write_wav(&fx.path("copy.wav"), WavFormat::Pcm16, 48_000, 1, &audio);
    let second = fx.scan();

    assert_eq!(second.counts.files_skipped, 1, "the original is unchanged");
    assert_eq!(second.deduped, 1, "the copy should not have been decoded");
    assert_eq!(second.processed, 0);

    let conn = fx.db.read().unwrap();
    assert_eq!(
        queries::sample_features(&conn, fx.row("copy.wav").id).unwrap(),
        queries::sample_features(&conn, fx.row("original.wav").id).unwrap()
    );
}

/// Cross-cutting rule 6. A cancelled scan keeps what it already wrote and marks its
/// `scan_runs` row `cancelled` -- it does not roll back and it does not hang.
#[test]
fn a_cancelled_scan_stops_cleanly_and_keeps_partial_results() {
    let fx = Fixture::new();
    for i in 0..40 {
        write_wav(
            &fx.path(&format!("hit_{i:03}.wav")),
            WavFormat::Pcm16,
            48_000,
            1,
            &sine(300.0, 0.7, 1.0, 48_000),
        );
    }

    let cancel = CancellationToken::new();
    cancel.cancel();

    let report = scan_root(&fx.db, fx.root_id, &cancel).unwrap();
    assert_eq!(report.status, audiobank_lib::db::ScanStatus::Cancelled);

    let conn = fx.db.read().unwrap();
    let status: String = conn
        .query_row(
            "SELECT status FROM scan_runs WHERE id = ?1",
            [report.scan_id],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "cancelled");

    // Whatever it got through is committed and consistent; the point is that the database
    // is usable, not that a particular number of files landed.
    let violations: i64 = conn
        .query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(violations, 0);

    // ...and the next scan picks up the rest.
    let resumed = fx.scan();
    assert_eq!(resumed.status, audiobank_lib::db::ScanStatus::Completed);
    assert_eq!(fx.count(), 40);
}

/// Every supported source rate has to reach the pipeline as 48 kHz mono, because everything
/// downstream assumes it. The source rate is recorded, the resampled duration is not
/// distorted, and the tone survives the resampler.
#[test]
fn files_at_other_sample_rates_resample_correctly() {
    let fx = Fixture::new();

    for (name, rate) in [
        ("r22050.wav", 22_050u32),
        ("r44100.wav", 44_100),
        ("r48000.wav", 48_000),
        ("r96000.wav", 96_000),
    ] {
        write_wav(
            &fx.path(name),
            WavFormat::Pcm16,
            rate,
            1,
            &sine(1000.0, 0.7, 1.0, rate),
        );
    }

    fx.scan();
    let conn = fx.db.read().unwrap();

    for (name, rate) in [
        ("r22050.wav", 22_050i64),
        ("r44100.wav", 44_100),
        ("r48000.wav", 48_000),
        ("r96000.wav", 96_000),
    ] {
        let row = fx.row(name);
        assert_eq!(row.status, "decoded", "{name}");
        assert_eq!(row.sample_rate, Some(rate), "{name}");
        assert_eq!(row.duration_ms, Some(1000), "{name}");

        // The 1 kHz tone must still be at 1 kHz after resampling. This is the assertion that
        // catches a ratio inverted somewhere in the resampler wiring.
        let centroid = queries::sample_features(&conn, row.id)
            .unwrap()
            .unwrap()
            .spectral_centroid
            .unwrap();
        assert!(
            (centroid - 1000.0).abs() < 150.0,
            "{name} decoded to a centroid of {centroid} Hz"
        );
    }
}

/// A file longer than the 10 s window decodes only its window, but reports its real length.
/// Truncating `duration_ms` to the window would put every long sample in the wrong bucket of
/// the duration filter.
#[test]
fn a_long_file_is_truncated_to_the_window_but_reports_its_full_duration() {
    let fx = Fixture::new();
    write_wav(
        &fx.path("stem.wav"),
        WavFormat::Pcm16,
        48_000,
        1,
        &sine(440.0, 0.5, 25.0, 48_000),
    );

    fx.scan();
    assert_eq!(fx.row("stem.wav").duration_ms, Some(25_000));
}

/// A root that is a symlink loop must not hang the walker or the test suite.
#[cfg(unix)]
#[test]
fn a_symlink_loop_does_not_trap_the_walker() {
    let fx = Fixture::new();
    write_sine(&fx.path("real/kick.wav"), 440.0, 0.2);
    std::os::unix::fs::symlink(fx.library.path(), fx.path("real/loop")).unwrap();

    let report = fx.scan();
    assert_eq!(report.status, audiobank_lib::db::ScanStatus::Completed);
    // The one real file is found; the loop contributes nothing beyond it.
    assert!(fx.count() >= 1);
    assert_eq!(fx.row("real/kick.wav").status, "decoded");
}

/// A `.gitignore` in a sample library is a `.gitignore` for source code that happens to be
/// nearby. Honoring it would scan a perfectly good library to zero samples, silently.
#[test]
fn a_gitignore_does_not_hide_the_library() {
    let fx = Fixture::new();
    std::fs::write(fx.path(".gitignore"), b"*.wav\n").unwrap();
    std::fs::create_dir_all(fx.path(".git")).unwrap();
    write_sine(&fx.path("kick.wav"), 440.0, 0.2);

    fx.scan();
    assert_eq!(fx.count(), 1, "the .gitignore hid the library");
}

/// Zero-byte files are not audio and must not become quarantine rows -- a placeholder
/// directory would otherwise fill the "could not be read" list with noise.
#[test]
fn empty_and_undersized_files_are_ignored_rather_than_quarantined() {
    let fx = Fixture::new();
    std::fs::write(fx.path("empty.wav"), b"").unwrap();
    std::fs::write(fx.path("stub.aiff"), b"FORM").unwrap();
    write_sine(&fx.path("real.wav"), 440.0, 0.2);

    let report = fx.scan();
    assert_eq!(fx.count(), 1);
    assert_eq!(report.counts.files_failed, 0);
}

/// `scan_root` on a root that no longer exists must fail with the typed error, not a panic
/// and not a silently empty scan.
#[test]
fn scanning_a_missing_root_is_a_typed_error() {
    let fx = Fixture::new();
    let gone = tempfile::tempdir().unwrap();
    let root_id = fx
        .db
        .writer()
        .add_root(gone.path().to_str().unwrap(), None)
        .unwrap();
    drop(gone);

    let err = scan_root(&fx.db, root_id, &CancellationToken::new()).unwrap_err();
    assert!(
        matches!(
            err,
            audiobank_lib::pipeline::PipelineError::RootUnreadable { .. }
        ),
        "got {err:?}"
    );
}

/// Discovery must be able to name every file it finds. This is a smoke test that the walk
/// handles the punctuation real sample packs use.
#[test]
fn filenames_with_awkward_characters_round_trip() {
    let fx = Fixture::new();
    let names = [
        "KICK_808_Distorted-02.wav",
        "Hat [closed] #3.wav",
        "café — loop 120bpm.wav",
        "100% wet.wav",
    ];
    for name in names {
        write_sine(&Path::new(&fx.path("pack")).join(name), 500.0, 0.2);
    }

    fx.scan();
    assert_eq!(fx.count(), names.len() as i64);
    for name in names {
        assert_eq!(fx.row(&format!("pack/{name}")).status, "decoded", "{name}");
    }
}
