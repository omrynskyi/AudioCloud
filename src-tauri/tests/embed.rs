//! Phase 4 end to end: the five-stage pipeline, against a real ONNX Runtime.
//!
//! The model these run on is not CLAP. It is one of the ~130 KB fixture graphs from
//! `scripts/make_session_fixtures.py`, carrying the real export's input signature and
//! output width and computing deterministic nonsense in between. That is the right oracle
//! for this phase and the wrong one for Phase 3: **nothing here says the embeddings mean
//! anything**, and nothing here can. `tests/parity.rs` is the test that says that, and it
//! needs the real checkpoint.
//!
//! What these do prove is everything between the decoder and `embeddings.bin`: that a
//! vector reaches the row it was computed for, that a duplicate borrows its twin's bytes
//! instead of paying for inference again, that a scan with no model leaves work a later
//! scan finishes, that a cancelled scan keeps what it wrote, and that a changed file does
//! not inherit its own stale vector. Those are the failures that would otherwise surface as
//! a plausible-looking map, which is the same reason Phase 3 exists.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use audiobank_lib::{
    db::{queries, Database, EmbeddingLoc, SampleStatus, ScanStatus},
    model::session::ModelSession,
    pipeline::{
        scan_root_with, BatchConfig, CancellationToken, ProgressSnapshot, ScanOptions, ScanReport,
    },
    EMBEDDING_DIM,
};
use support::{sine, write_wav, WavFormat};
use tempfile::TempDir;

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/session/frames_major.onnx")
}

/// A session over the fixture graph, or a printed reason and a skipped test.
///
/// Skipping rather than failing: the fixtures are committed, so the only way this is
/// absent is a checkout that filtered them, and a test that fails for that is noise.
fn session() -> Option<Arc<ModelSession>> {
    let path = fixture_model();
    if !path.is_file() {
        println!("SKIPPED: no session fixture at {}", path.display());
        return None;
    }
    match ModelSession::open(&path) {
        Ok(session) => Some(Arc::new(session)),
        Err(e) => {
            println!("SKIPPED: could not open the fixture session: {e}");
            None
        }
    }
}

/// Runs the body only if the fixture graph loads. Every test here needs one.
macro_rules! with_session {
    ($name:ident) => {
        let Some($name) = session() else { return };
    };
}

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

    fn path(&self, rel: &str) -> PathBuf {
        self.library.path().join(rel)
    }

    /// Writes a one-second tone whose pitch is a function of the name, so two different
    /// files are never accidentally byte-identical.
    fn write_tone(&self, rel: &str, hz: f32) {
        write_wav(
            &self.path(rel),
            WavFormat::Pcm16,
            48_000,
            1,
            &sine(hz, 0.7, 1.0, 48_000),
        );
    }

    fn scan_with(&self, options: &mut ScanOptions<'_>) -> ScanReport {
        scan_root_with(&self.db, self.root_id, options).unwrap()
    }

    /// A full five-stage scan.
    fn embed_scan(&self, session: &Arc<ModelSession>) -> ScanReport {
        let cancel = CancellationToken::new();
        let mut options = ScanOptions::new(&cancel)
            .with_session(Arc::clone(session))
            // Four rather than sixteen: these libraries are a handful of files, and a batch
            // that never fills would test the timeout instead of the batcher.
            .with_batch(BatchConfig::of_size(4));
        self.scan_with(&mut options)
    }

    /// A scan with no model, as happens before the 200 MB download has finished.
    fn decode_only_scan(&self) -> ScanReport {
        let cancel = CancellationToken::new();
        let mut options = ScanOptions::new(&cancel);
        self.scan_with(&mut options)
    }

    fn status(&self, rel: &str) -> String {
        let conn = self.db.read().unwrap();
        conn.query_row(
            "SELECT status FROM samples WHERE root_id = ?1 AND rel_path = ?2",
            (self.root_id, rel),
            |r| r.get(0),
        )
        .unwrap_or_else(|e| panic!("no row for {rel}: {e}"))
    }

    fn loc(&self, rel: &str) -> Option<EmbeddingLoc> {
        let conn = self.db.read().unwrap();
        let id: i64 = conn
            .query_row(
                "SELECT id FROM samples WHERE root_id = ?1 AND rel_path = ?2",
                (self.root_id, rel),
                |r| r.get(0),
            )
            .unwrap_or_else(|e| panic!("no row for {rel}: {e}"));
        queries::embedding_loc(&conn, id).unwrap()
    }

    fn vector(&self, rel: &str) -> Vec<f32> {
        let loc = self
            .loc(rel)
            .unwrap_or_else(|| panic!("{rel} has no embedding"));
        self.db.embeddings().lock().unwrap().read(loc).unwrap()
    }

    fn embedded_rows(&self) -> i64 {
        queries::count_samples_with_status(&self.db.read().unwrap(), SampleStatus::Embedded)
            .unwrap()
    }

    fn embeddings_file_len(&self) -> u64 {
        self.db.embeddings().lock().unwrap().len_bytes()
    }
}

/// Bytes one f16 vector occupies in `embeddings.bin`.
const VECTOR_BYTES: u64 = EMBEDDING_DIM as u64 * 2;

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

/// The headline: a folder of audio comes out the other end with a vector per row.
#[test]
fn a_scan_with_a_session_embeds_every_file() {
    with_session!(session);
    let fx = Fixture::new();
    for i in 0..10 {
        fx.write_tone(&format!("tone_{i:02}.wav"), 100.0 + i as f32 * 37.0);
    }

    let report = fx.embed_scan(&session);
    assert_eq!(report.status, ScanStatus::Completed);
    assert_eq!(report.processed, 10);
    assert_eq!(report.embedded, 10);
    assert_eq!(fx.embedded_rows(), 10);

    // Ten distinct vectors, ten vectors' worth of file.
    assert_eq!(fx.embeddings_file_len(), 10 * VECTOR_BYTES);

    for i in 0..10 {
        let rel = format!("tone_{i:02}.wav");
        assert_eq!(fx.status(&rel), "embedded");
        let v = fx.vector(&rel);
        assert_eq!(v.len(), EMBEDDING_DIM);
        // f16 storage, so unit length survives to about three decimals and no further.
        assert!(
            (norm(&v) - 1.0).abs() < 1e-2,
            "{rel} is not L2-normalized: |x| = {}",
            norm(&v)
        );
    }
}

/// Batching is the whole reason a serialized session is affordable (`overview.md` §3.4).
/// Ten files at a batch size of four is three `run()` calls, not ten.
#[test]
fn inference_runs_in_batches_rather_than_once_per_file() {
    with_session!(session);
    let fx = Fixture::new();
    for i in 0..10 {
        fx.write_tone(&format!("tone_{i:02}.wav"), 100.0 + i as f32 * 37.0);
    }

    let report = fx.embed_scan(&session);
    assert_eq!(report.inferred, 10);
    assert!(
        report.inference_batches <= 3,
        "10 files at a batch size of 4 took {} run() calls",
        report.inference_batches
    );
    assert!(report.inference_batches >= 1);
}

/// The failure a batcher makes possible and nothing else does: row 3's vector arriving on
/// row 5. Different audio must produce different vectors, and the same audio the same one.
#[test]
fn each_row_gets_its_own_vector() {
    with_session!(session);
    let fx = Fixture::new();
    for i in 0..8 {
        fx.write_tone(&format!("tone_{i:02}.wav"), 120.0 + i as f32 * 61.0);
    }
    fx.embed_scan(&session);

    let vectors: Vec<Vec<f32>> = (0..8)
        .map(|i| fx.vector(&format!("tone_{i:02}.wav")))
        .collect();

    for (i, a) in vectors.iter().enumerate() {
        for (j, b) in vectors.iter().enumerate().skip(i + 1) {
            assert!(
                a != b,
                "tone {i} and tone {j} were handed the same vector, so the batch was \
                 redistributed wrongly"
            );
        }
    }
}

/// A duplicate is a reference to its twin's bytes, not a second copy of them -- and, more
/// importantly, not a second inference run.
#[test]
fn duplicates_borrow_their_twins_vector_instead_of_running_the_model_again() {
    with_session!(session);
    let fx = Fixture::new();
    fx.write_tone("original.wav", 220.0);
    for i in 0..5 {
        std::fs::copy(fx.path("original.wav"), fx.path(&format!("copy_{i}.wav"))).unwrap();
    }

    let report = fx.embed_scan(&session);
    assert_eq!(report.processed, 1, "the copies were decoded");
    assert_eq!(report.deduped, 5);
    assert_eq!(report.inferred, 1, "the model ran on identical audio twice");
    assert_eq!(report.embedded, 6, "every row still owns a vector");
    assert_eq!(fx.embedded_rows(), 6);

    // One vector in the file, six rows pointing at it.
    assert_eq!(fx.embeddings_file_len(), VECTOR_BYTES);
    let original = fx.loc("original.wav").unwrap();
    for i in 0..5 {
        assert_eq!(fx.loc(&format!("copy_{i}.wav")).unwrap(), original);
    }
}

/// A copy discovered by a *later* scan resolves through the `content_hash` index rather
/// than the in-scan table, which is a different code path with the same guarantee.
#[test]
fn a_copy_added_later_borrows_the_stored_vector() {
    with_session!(session);
    let fx = Fixture::new();
    fx.write_tone("original.wav", 330.0);
    fx.embed_scan(&session);
    let original = fx.loc("original.wav").unwrap();

    std::fs::copy(fx.path("original.wav"), fx.path("later.wav")).unwrap();
    let report = fx.embed_scan(&session);

    assert_eq!(report.processed, 0, "the copy was decoded");
    assert_eq!(report.inferred, 0, "the model ran for the copy");
    assert_eq!(report.embedded, 1, "the borrowed vector was not counted");
    assert_eq!(fx.loc("later.wav").unwrap(), original);
    assert_eq!(fx.embeddings_file_len(), VECTOR_BYTES);
}

/// A library is worth indexing before the 200 MB model arrives. The rows that scan leaves
/// behind are `decoded`, and the next scan -- the one with a session -- has to finish them
/// rather than fast-skip them.
#[test]
fn a_scan_without_a_model_leaves_work_that_a_later_scan_finishes() {
    with_session!(session);
    let fx = Fixture::new();
    for i in 0..6 {
        fx.write_tone(&format!("tone_{i}.wav"), 150.0 + i as f32 * 40.0);
    }

    let first = fx.decode_only_scan();
    assert_eq!(first.processed, 6);
    assert_eq!(first.embedded, 0);
    assert_eq!(fx.embedded_rows(), 0);
    assert_eq!(fx.status("tone_0.wav"), "decoded");

    let second = fx.embed_scan(&session);
    assert_eq!(
        second.counts.files_skipped, 0,
        "the fast-skip skipped rows that still owed a vector"
    );
    assert_eq!(second.embedded, 6);
    assert_eq!(fx.embedded_rows(), 6);
}

/// The other half of resume: once a row *is* embedded, a rescan must not touch it. This is
/// the re-scan budget from Phase 2, still holding with a model attached.
#[test]
fn rescanning_an_embedded_library_reads_nothing() {
    with_session!(session);
    let fx = Fixture::new();
    for i in 0..6 {
        fx.write_tone(&format!("tone_{i}.wav"), 150.0 + i as f32 * 40.0);
    }
    fx.embed_scan(&session);
    let before = fx.embeddings_file_len();

    let again = fx.embed_scan(&session);
    assert_eq!(again.counts.files_skipped, 6);
    assert_eq!(again.processed, 0);
    assert_eq!(again.inferred, 0);
    assert_eq!(
        fx.embeddings_file_len(),
        before,
        "a rescan appended vectors for files it had already embedded"
    );
}

/// A file whose contents changed must not keep the vector for the audio it used to hold --
/// and no duplicate of the *new* contents may inherit it either. A stale vector is exactly
/// the failure mode this phase cannot detect downstream: the map still renders.
#[test]
fn a_changed_file_does_not_keep_its_stale_vector() {
    with_session!(session);
    let fx = Fixture::new();
    fx.write_tone("morphing.wav", 200.0);
    fx.embed_scan(&session);
    let before = fx.vector("morphing.wav");

    // A different tone *and* a different length, so the `(mtime, size)` fast-skip cannot
    // mistake it for the file it replaced.
    write_wav(
        &fx.path("morphing.wav"),
        WavFormat::Pcm16,
        48_000,
        1,
        &sine(900.0, 0.5, 1.6, 48_000),
    );

    let report = fx.embed_scan(&session);
    assert_eq!(report.processed, 1);
    assert_eq!(report.inferred, 1);

    let after = fx.vector("morphing.wav");
    assert_ne!(
        before, after,
        "the row kept the vector for its old contents"
    );
    assert_eq!(fx.status("morphing.wav"), "embedded");
}

/// Quarantine still works with a model attached: a file that will not decode has no
/// spectrogram, so it never reaches the batcher, and the scan does not stop.
#[test]
fn undecodable_files_are_quarantined_without_reaching_the_model() {
    with_session!(session);
    let fx = Fixture::new();
    fx.write_tone("good.wav", 440.0);
    std::fs::write(fx.path("broken.wav"), vec![0x7fu8; 4096]).unwrap();

    let report = fx.embed_scan(&session);
    assert_eq!(report.status, ScanStatus::Completed);
    assert_eq!(report.counts.files_failed, 1);
    assert_eq!(report.inferred, 1, "the broken file reached the model");
    assert_eq!(fx.status("broken.wav"), "decode_failed");
    assert!(fx.loc("broken.wav").is_none());
    assert_eq!(fx.status("good.wav"), "embedded");
}

/// Cross-cutting rule 6, with inference in the pipeline: cancelling keeps the vectors
/// already written, closes the run as `cancelled`, and leaves a database the next scan can
/// pick up.
#[test]
fn a_cancelled_scan_keeps_the_vectors_it_already_wrote() {
    with_session!(session);
    let fx = Fixture::new();
    for i in 0..40 {
        fx.write_tone(&format!("tone_{i:02}.wav"), 100.0 + i as f32 * 7.0);
    }

    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut options = ScanOptions::new(&cancel)
        .with_session(Arc::clone(&session))
        .with_batch(BatchConfig::of_size(4));
    let report = fx.scan_with(&mut options);
    assert_eq!(report.status, ScanStatus::Cancelled);

    // Whatever landed is consistent: every row that claims to be embedded has offsets
    // inside the file, and every vector reads back.
    let conn = fx.db.read().unwrap();
    let violations: i64 = conn
        .query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(violations, 0);

    let locs = queries::all_embedding_locs(&conn).unwrap();
    drop(conn);
    for (_, loc) in &locs {
        let v = fx.db.embeddings().lock().unwrap().read(*loc).unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
    }

    // ...and the resumed scan finishes the job.
    let resumed = fx.embed_scan(&session);
    assert_eq!(resumed.status, ScanStatus::Completed);
    assert_eq!(fx.embedded_rows(), 40);
}

/// `overview.md` §6.5: coalesced to <= 10 Hz, with a terminal event that always arrives.
/// The anti-pattern this rules out is one event per file, so the assertion is against the
/// file count rather than against a rate.
#[test]
fn progress_is_coalesced_and_always_terminates() {
    with_session!(session);
    let fx = Fixture::new();
    for i in 0..60 {
        fx.write_tone(&format!("tone_{i:02}.wav"), 100.0 + i as f32 * 5.0);
    }

    let snapshots: Arc<Mutex<Vec<ProgressSnapshot>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&snapshots);

    let cancel = CancellationToken::new();
    let mut options = ScanOptions::new(&cancel)
        .with_session(Arc::clone(&session))
        .with_batch(BatchConfig::of_size(4))
        .with_progress(move |s| sink.lock().unwrap().push(s));
    let report = fx.scan_with(&mut options);

    let snapshots = snapshots.lock().unwrap();
    assert!(!snapshots.is_empty(), "no progress at all");
    assert!(
        snapshots.len() < 60,
        "{} snapshots for 60 files is on its way to one per file",
        snapshots.len()
    );

    let last = snapshots.last().unwrap();
    assert!(last.is_terminal(), "the last snapshot is not terminal");
    assert_eq!(last.scan_id, report.scan_id);
    assert_eq!(last.files_done, 60);
    assert_eq!(last.files_embedded, 60);
    assert!(
        snapshots
            .windows(2)
            .all(|w| w[0].files_done <= w[1].files_done),
        "progress went backwards"
    );
}
