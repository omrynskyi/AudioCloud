//! Phase 1 benchmarks (`task.md`, cross-cutting rule 9).
//!
//! These are `#[ignore]`d because they are only meaningful in an optimized build, and because
//! a 50,000-row insert has no business running on every `cargo test`. CI compiles them --
//! which is what keeps them from rotting -- but does not run them.
//!
//! ```sh
//! cargo test --manifest-path src-tauri/Cargo.toml --profile perf --test benchmarks \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is not optional: RSS is a property of the process, so two benchmarks
//! running at once measure each other.
//!
//! Recorded results live in `BENCHMARKS.md` next to the targets they are measured against.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Instant;

use audiobank_lib::{
    db::{queries, Database, EmbeddingStore, NewSample, SampleFeatures, SampleStatus},
    EMBEDDING_DIM,
};
use half::f16;

/// The corpus size every §7 target is stated against.
const CORPUS: usize = 50_000;

/// Rows per command. Matches `writer::BATCH_ROWS`, so each command fills exactly one batch.
const CHUNK: usize = 1000;

/// Deterministic, dependency-free noise. The values only have to be varied and
/// reproducible; a real distribution would not change what is being measured.
struct Xorshift(u64);

impl Xorshift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Map to [-1, 1): the range an L2-normalized embedding component lives in.
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

/// Resident set size in bytes, straight from the kernel.
///
/// `ps` rather than a crate: this is the number Activity Monitor shows, it needs no
/// dependency, and a benchmark that lies about its own instrumentation is worse than no
/// benchmark.
fn rss_bytes() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .expect("rss")
        * 1024
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn synthetic_sample(root_id: i64, i: usize) -> NewSample {
    let filename = format!("KICK_{i:05}_Distorted-{:02}.wav", i % 97);
    NewSample {
        root_id,
        rel_path: format!("drums/{}/{filename}", i % 512),
        filename,
        ext: "wav".into(),
        size_bytes: 512_000 + (i as i64 % 4096),
        mtime: 1_700_000_000 + i as i64,
        content_hash: Some([(i % 251) as u8; 32]),
        duration_ms: Some(1500),
        sample_rate: Some(48_000),
        channels: Some(2),
        status: SampleStatus::Decoded,
    }
}

fn synthetic_features(i: usize) -> SampleFeatures {
    SampleFeatures {
        peak_db: Some(-1.0 - (i % 20) as f32),
        rms_db: Some(-14.0 - (i % 9) as f32),
        lufs_integrated: Some(-18.0),
        spectral_centroid: Some(1200.0 + (i % 3000) as f32),
        spectral_flatness: Some(0.2),
        zero_crossing: Some(0.08),
        onset_density: Some(2.5),
        bpm: Some(90.0 + (i % 80) as f32),
        bpm_confidence: Some(0.7),
        key_root: Some((i % 12) as i32),
        key_mode: Some((i % 2) as i32),
        key_confidence: Some(0.5),
    }
}

/// Exit criterion: 50,000 synthetic rows insert in **< 2 s**.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn insert_50k_samples_and_features() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), EMBEDDING_DIM).unwrap();
    let writer = db.writer();
    let root = writer.add_root("/Library/Audio/Samples", None).unwrap();

    // Built up front: this measures the data layer, not `format!`.
    let rows: Vec<NewSample> = (0..CORPUS).map(|i| synthetic_sample(root, i)).collect();

    let rss_before = rss_bytes();
    let started = Instant::now();

    let mut ids = Vec::with_capacity(CORPUS);
    for chunk in rows.chunks(CHUNK) {
        ids.extend(writer.upsert_samples(chunk.to_vec()).unwrap());
    }
    writer.flush().unwrap();
    let samples_elapsed = started.elapsed();

    let features_started = Instant::now();
    for chunk in ids.chunks(CHUNK) {
        let batch: Vec<_> = chunk
            .iter()
            .enumerate()
            .map(|(n, id)| (*id, synthetic_features(n)))
            .collect();
        writer.set_features(batch).unwrap();
    }
    writer.flush().unwrap();
    let features_elapsed = features_started.elapsed();
    let total = started.elapsed();
    let rss_after = rss_bytes();

    let conn = db.read().unwrap();
    assert_eq!(queries::count_samples(&conn).unwrap(), CORPUS as i64);
    let db_bytes = std::fs::metadata(dir.path().join("library.db"))
        .unwrap()
        .len();

    println!("\n-- insert 50k samples + features --");
    println!(
        "  samples        {:>8.0} ms  ({:.0} rows/s)",
        samples_elapsed.as_secs_f64() * 1000.0,
        CORPUS as f64 / samples_elapsed.as_secs_f64()
    );
    println!(
        "  features       {:>8.0} ms  ({:.0} rows/s)",
        features_elapsed.as_secs_f64() * 1000.0,
        CORPUS as f64 / features_elapsed.as_secs_f64()
    );
    println!("  total          {:>8.0} ms", total.as_secs_f64() * 1000.0);
    println!("  commits        {:>8}", writer.metrics().commits());
    println!("  library.db     {:>8.1} MiB", mib(db_bytes));
    println!(
        "  RSS delta      {:>8.1} MiB",
        mib(rss_after.saturating_sub(rss_before))
    );

    assert!(
        samples_elapsed.as_secs_f64() < 2.0,
        "50k inserts took {samples_elapsed:?}, target < 2 s"
    );
}

/// Exit criterion: the embedding file round-trips bit-exactly, and reading it back does not
/// cost 51 MB of resident memory.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn append_and_mmap_50k_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = EmbeddingStore::open(dir.path(), EMBEDDING_DIM).unwrap();

    let mut rng = Xorshift(0x5eed_1234_9876_abcd);
    let vectors: Vec<Vec<f32>> = (0..CORPUS)
        .map(|_| (0..EMBEDDING_DIM).map(|_| rng.next_f32()).collect())
        .collect();

    let started = Instant::now();
    let mut locs = Vec::with_capacity(CORPUS);
    for chunk in vectors.chunks(CHUNK) {
        locs.extend(store.append_batch(chunk).unwrap());
    }
    store.sync().unwrap();
    let append_elapsed = started.elapsed();
    let file_bytes = store.len_bytes();

    // Reopen: the read path must not depend on state the writing handle happened to hold.
    drop(store);
    let store = EmbeddingStore::open(dir.path(), EMBEDDING_DIM).unwrap();

    // Sparse read first, while the mapping is still cold. A hundred rows scattered across
    // the file is the access pattern of the inspector and of a nearest-neighbor lookup --
    // and it is the one that proves the mmap is a view rather than a load.
    //
    // A row is 1 KiB and an Apple Silicon page is 16 KiB, so each scattered row costs a
    // whole page: 100 rows is ~1.6 MiB resident against a 48.8 MiB file. Reading many more
    // rows than that would page in most of the file and measure nothing.
    const SPARSE_ROWS: usize = 100;

    let rss_before = rss_bytes();
    let map_started = Instant::now();
    let matrix = store.matrix().unwrap();
    let map_elapsed = map_started.elapsed();

    let mut row = Vec::with_capacity(EMBEDDING_DIM);
    for i in (0..CORPUS).step_by(CORPUS / SPARSE_ROWS) {
        matrix.row_into(locs[i], &mut row).unwrap();
    }
    let rss_sparse = rss_bytes();

    // Now the whole matrix, one row at a time into the same buffer, as the projection
    // re-fit will read it.
    let verify_started = Instant::now();
    let mut mismatches = 0usize;
    for (i, loc) in locs.iter().enumerate() {
        matrix.row_into(*loc, &mut row).unwrap();
        for (value, stored) in vectors[i].iter().zip(&row) {
            if f16::from_f32(*value).to_f32() != *stored {
                mismatches += 1;
            }
        }
    }
    let verify_elapsed = verify_started.elapsed();
    let rss_full = rss_bytes();

    // And the single-row pread path agrees with the mapped one.
    assert_eq!(
        store.read(locs[CORPUS - 1]).unwrap(),
        matrix.row(locs[CORPUS - 1]).unwrap()
    );

    println!("\n-- append + mmap 50k x {EMBEDDING_DIM} f16 --");
    println!(
        "  append         {:>8.0} ms  ({:.1} MiB/s)",
        append_elapsed.as_secs_f64() * 1000.0,
        mib(file_bytes) / append_elapsed.as_secs_f64()
    );
    println!("  embeddings.bin {:>8.1} MiB", mib(file_bytes));
    println!(
        "  mmap           {:>8.3} ms",
        map_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  verify all     {:>8.0} ms",
        verify_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "  RSS, {SPARSE_ROWS} rows  {:>8.1} MiB",
        mib(rss_sparse.saturating_sub(rss_before))
    );
    println!(
        "  RSS, all rows  {:>8.1} MiB  (clean, file-backed, evictable)",
        mib(rss_full.saturating_sub(rss_before))
    );

    assert_eq!(mismatches, 0, "embeddings did not round-trip bit-exactly");
    assert_eq!(file_bytes, (CORPUS * EMBEDDING_DIM * 2) as u64);
    // The exit criterion is that reading the matrix does not cost 51 MB of resident memory.
    // Read as "no 51 MB heap load": a full traversal *does* bring the whole file into RSS,
    // but as clean file-backed pages the kernel can drop under pressure -- which is the
    // entire difference between a mapping and a `Vec<f32>`. The falsifiable form of the
    // claim is the sparse read: touch a hundred rows, stay near zero.
    assert!(
        rss_sparse.saturating_sub(rss_before) < 10 * 1024 * 1024,
        "mapping and sampling {SPARSE_ROWS} rows cost {:.1} MiB of RSS; this path is \
         supposed to page in only what it touches",
        mib(rss_sparse.saturating_sub(rss_before))
    );
}
