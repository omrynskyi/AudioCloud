//! Phase 1, 2, 4 and 5 benchmarks (`task.md`, cross-cutting rule 9).
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

use std::{sync::Arc, time::Instant};

mod support;

use audiobank_lib::{
    db::{
        queries,
        search::{self, Feature, FeatureRange, QueryFilter},
        Database, EmbeddingStore, NewSample, SampleFeatures, SampleStatus,
    },
    model::session::ModelSession,
    pipeline::{
        features::Analyzer, scan_root, scan_root_with, BatchConfig, CancellationToken, ScanOptions,
    },
    projection::{place_incremental, refit, PcaProjector, Point3, Projector, Refit, UmapProjector},
    EMBEDDING_DIM,
};
use half::f16;
use support::{noise, sine, write_wav, WavFormat};

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

/// High-water resident size across a span of work.
///
/// `rss_bytes()` reports *current* RSS, so the usual before/after delta measures whatever
/// the allocator happened not to have returned yet. For most of these benchmarks that is
/// close enough -- their peak is at the end. It is not close enough for the UMAP re-fit,
/// where the same fit reported 1.6 GiB one run and 2.9 GiB the next, and where the number is
/// the one Phase 10 will plan against. So that one samples.
///
/// 50 ms is well under the granularity of anything being measured here and costs one `ps`
/// per tick.
#[derive(Debug)]
struct PeakRss {
    stop: Arc<std::sync::atomic::AtomicBool>,
    peak: Arc<std::sync::atomic::AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakRss {
    fn watch() -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak = Arc::new(std::sync::atomic::AtomicU64::new(rss_bytes()));
        let (flag, high) = (Arc::clone(&stop), Arc::clone(&peak));

        let handle = std::thread::spawn(move || {
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                high.fetch_max(rss_bytes(), std::sync::atomic::Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            high.fetch_max(rss_bytes(), std::sync::atomic::Ordering::Relaxed);
        });

        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    /// Stops sampling and returns the high-water mark in bytes.
    fn finish(mut self) -> u64 {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }
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

// ---------------------------------------------------------------------------------------
// Phase 2 -- discovery and decode
// ---------------------------------------------------------------------------------------

/// Files in the synthetic library. Two orders of magnitude below the 5,000-file target in
/// `task.md`, because these are *generated* -- writing 5,000 WAVs to a temp directory
/// measures the temp directory. The throughput figure is per-file and scales; the recorded
/// number against a real library lives in `BENCHMARKS.md`.
const LIBRARY_FILES: usize = 400;

/// Builds a synthetic sample library on disk: mixed formats, mixed sample rates, mixed
/// lengths, and a realistic proportion of exact duplicates.
///
/// Duplicates matter to the measurement, not just to the dedup test. A real library is
/// perhaps 10% redundant, and a throughput number taken on 400 unique files overstates what
/// the pipeline does on 400 real ones.
fn write_library(root: &std::path::Path) -> (usize, usize) {
    let mut unique = 0usize;
    let mut duplicates = 0usize;

    for i in 0..LIBRARY_FILES {
        let name = format!("bank{:02}/HIT_{i:04}_Distorted-{:02}.wav", i % 16, i % 97);
        let path = root.join(&name);

        // Every tenth file is a byte-exact copy of the one before it.
        if i % 10 == 9 {
            let previous = root.join(format!(
                "bank{:02}/HIT_{:04}_Distorted-{:02}.wav",
                (i - 1) % 16,
                i - 1,
                (i - 1) % 97
            ));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::copy(&previous, &path).unwrap();
            duplicates += 1;
            continue;
        }

        // A spread of source rates, so the resampler is on the measured path for most files
        // exactly as it is in a real library.
        let (format, rate) = match i % 4 {
            0 => (WavFormat::Pcm16, 44_100u32),
            1 => (WavFormat::Pcm16, 48_000),
            2 => (WavFormat::Float32, 44_100),
            _ => (WavFormat::Pcm16, 96_000),
        };

        // Lengths a drum library actually has: mostly short one-shots, some loops, a few
        // long enough to exercise the 10 s decode cap.
        let seconds = match i % 20 {
            0 => 12.0,
            1..=3 => 4.0,
            _ => 0.8,
        };

        // The frequency is a function of `i` alone, so no two generated files are
        // byte-identical by accident. An earlier version derived it from `i % 40`, which
        // silently made 60% of the corpus duplicates and turned a decode benchmark into a
        // measurement of the dedup table.
        let audio = if i % 3 == 0 {
            noise(i as u64 + 1, (rate as f32 * seconds) as usize)
        } else {
            sine(80.0 + i as f32 * 2.7, 0.7, seconds, rate)
        };

        write_wav(&path, format, rate, 1, &audio);
        unique += 1;
    }

    (unique, duplicates)
}

/// Exit criteria: decode+DSP throughput >= 400 samples/s, a rescan of an unchanged folder
/// completes in < 5% of the original time, and peak RSS stays flat.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn scan_a_synthetic_library() {
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();

    let (unique, duplicates) = write_library(library.path());
    let bytes: u64 = walk_size(library.path());

    let db = Database::open(data.path(), EMBEDDING_DIM).unwrap();
    let root = db
        .writer()
        .add_root(library.path().to_str().unwrap(), None)
        .unwrap();

    let rss_before = rss_bytes();
    let started = Instant::now();
    let cold_report = scan_root(&db, root, &CancellationToken::new()).unwrap();
    let cold = started.elapsed();
    let rss_after_cold = rss_bytes();

    let started = Instant::now();
    let warm_report = scan_root(&db, root, &CancellationToken::new()).unwrap();
    let warm = started.elapsed();
    let rss_after_warm = rss_bytes();

    // Files actually decoded, not files seen. The exit criterion is about decode+DSP
    // throughput, and a deduplicated file passes through neither.
    let throughput = cold_report.processed as f64 / cold.as_secs_f64();
    let ratio = warm.as_secs_f64() / cold.as_secs_f64();

    println!("\n-- scan {LIBRARY_FILES} files ({unique} unique, {duplicates} duplicates) --");
    println!("  on disk        {:>8.1} MiB", mib(bytes));
    println!("  cold scan      {:>8.0} ms", cold.as_secs_f64() * 1000.0);
    println!(
        "  decoded        {:>8}  ({throughput:.0} samples/s)",
        cold_report.processed
    );
    println!("  deduplicated   {:>8}", cold_report.deduped);
    println!(
        "  rescan         {:>8.0} ms  ({:.1}% of cold)",
        warm.as_secs_f64() * 1000.0,
        ratio * 100.0
    );
    println!("  skipped        {:>8}", warm_report.counts.files_skipped);
    println!(
        "  RSS after cold {:>8.1} MiB",
        mib(rss_after_cold.saturating_sub(rss_before))
    );
    println!(
        "  RSS after warm {:>8.1} MiB",
        mib(rss_after_warm.saturating_sub(rss_before))
    );

    let conn = db.read().unwrap();
    assert_eq!(queries::count_samples(&conn).unwrap(), LIBRARY_FILES as i64);
    assert_eq!(cold_report.counts.files_failed, 0);
    // The generator's duplicates are the only duplicates: a throughput figure inflated by
    // files the decoder never touched is not a throughput figure.
    assert_eq!(cold_report.deduped as usize, duplicates);
    assert_eq!(cold_report.processed as usize, unique);
    assert_eq!(warm_report.counts.files_skipped, LIBRARY_FILES as i64);

    assert!(
        throughput >= 400.0,
        "decode+DSP throughput was {throughput:.0} files/s, target >= 400"
    );
    assert!(
        ratio < 0.05,
        "rescan took {:.1}% of the cold scan, target < 5%",
        ratio * 100.0
    );
}

/// The DSP stage on its own, so a regression can be attributed to it rather than to decode.
///
/// One 10 s buffer is 1000 frames of 1024-point FFT for the descriptors plus 116 frames of
/// 8192-point FFT for chroma. `overview.md` §3.3 shares the first transform across the
/// descriptors precisely so this number stays small next to decode.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn analyze_a_full_window() {
    const ITERATIONS: usize = 200;

    let samples = sine(440.0, 0.7, 10.0, 48_000);
    let mut analyzer = Analyzer::new();

    // One pass to pay the FFT planning and first-touch costs off the measurement.
    analyzer.analyze(&samples);

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(analyzer.analyze(std::hint::black_box(&samples)));
    }
    let elapsed = started.elapsed();

    let per_call = elapsed.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    println!("\n-- analyze one 10 s window --");
    println!("  per window     {per_call:>8.2} ms");
    println!(
        "  throughput     {:>8.0} windows/s",
        ITERATIONS as f64 / elapsed.as_secs_f64()
    );
}

/// Recursive size of a directory tree.
fn walk_size(root: &std::path::Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let meta = entry.metadata().unwrap();
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

/// Exit criterion: peak RSS during a scan stays flat as the file count grows.
///
/// The falsifiable form of "backpressure is the memory strategy" (`overview.md` §3). Scan a
/// library, then scan one four times the size in a fresh database, and compare the resident
/// memory each cost. Bounded queues times bounded payloads is a constant, so the second
/// number must not be four times the first -- if it is, some stage is accumulating instead
/// of streaming, and that is a bug that only shows up on a library nobody tested against.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn scan_memory_is_flat_as_the_library_grows() {
    fn scan_n(files: usize) -> u64 {
        let data = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();

        for i in 0..files {
            write_wav(
                &library
                    .path()
                    .join(format!("bank{:02}/hit_{i:05}.wav", i % 32)),
                WavFormat::Pcm16,
                48_000,
                1,
                &sine(80.0 + i as f32 * 1.3, 0.7, 1.0, 48_000),
            );
        }

        let db = Database::open(data.path(), EMBEDDING_DIM).unwrap();
        let root = db
            .writer()
            .add_root(library.path().to_str().unwrap(), None)
            .unwrap();

        let before = rss_bytes();
        let report = scan_root(&db, root, &CancellationToken::new()).unwrap();
        let after = rss_bytes();

        assert_eq!(report.processed as usize, files);
        after.saturating_sub(before)
    }

    // Small first, so the allocator's high-water mark from the large scan cannot be what the
    // small one is measured against.
    let small = scan_n(250);
    let large = scan_n(1000);

    println!("\n-- scan memory against library size --");
    println!("  250 files      {:>8.1} MiB", mib(small));
    println!("  1000 files     {:>8.1} MiB", mib(large));
    println!(
        "  growth         {:>8.2}x for 4x the files",
        large as f64 / small.max(1) as f64
    );

    // Generous, deliberately. What is being falsified is linear growth -- a stage that
    // accumulates would show 4x here. Page-cache and allocator noise between two runs in one
    // process is worth a good deal more slack than the margin to 2x.
    assert!(
        large < small.max(1024 * 1024) * 2,
        "scanning 4x the files cost {:.1} MiB against {:.1} MiB: memory is tracking file count",
        mib(large),
        mib(small)
    );
}

// ---------------------------------------------------------------------------------------
// Phase 4 -- the embedding pipeline
// ---------------------------------------------------------------------------------------

/// **These do not measure CLAP.** They measure everything around it.
///
/// The graph they run is `tests/fixtures/session/frames_major.onnx` -- ~130 KB of
/// deterministic nonsense with the real export's input signature. Until the export in Phase
/// 3 has actually been run, that is the only ONNX file this repository can produce a number
/// from, and the number it produces is an *upper bound* on end-to-end throughput: everything
/// except the matmuls, measured honestly, with the model-shaped hole clearly labelled.
///
/// Phase 4's exit criterion is >= 60 samples/s with real CLAP. Set [`MODEL_PATH_ENV`] to a
/// real export and these become that measurement; leave it unset and they are the plumbing
/// bound.
const MODEL_PATH_ENV: &str = "AUDIOBANK_MODEL_PATH";

/// Whether the session in hand is the real model rather than the fixture graph.
fn is_real_model() -> bool {
    std::env::var_os(MODEL_PATH_ENV).is_some()
}

/// A label for the graph a measurement ran against, printed next to every number.
fn graph_label() -> &'static str {
    if is_real_model() {
        "REAL MODEL"
    } else {
        "fixture graph, not CLAP"
    }
}
fn fixture_session() -> Option<Arc<ModelSession>> {
    // `AUDIOBANK_MODEL_PATH` is the same escape hatch `tests/parity.rs` uses, and pointing
    // it at the real export is what turns every number below from an upper bound into the
    // measurement Phase 4's exit criterion actually asks for. Each benchmark prints which
    // graph it ran, so a figure can never be read without knowing which one produced it.
    let path = match std::env::var_os(MODEL_PATH_ENV) {
        Some(path) => std::path::PathBuf::from(path),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/session/frames_major.onnx"),
    };
    match ModelSession::open(&path) {
        Ok(session) => Some(Arc::new(session)),
        Err(e) => {
            println!("SKIPPED: no fixture session ({e})");
            None
        }
    }
}

/// Builds a library and scans it with a session attached, returning what it cost.
fn embed_library(files: usize, batch: BatchConfig, session: &Arc<ModelSession>) -> EmbedRun {
    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();

    for i in 0..files {
        // Sub-second one-shots, which is what a drum library actually is. The long-window
        // cost is measured separately by `analyze_a_full_window`.
        write_wav(
            &library
                .path()
                .join(format!("bank{:02}/hit_{i:05}.wav", i % 32)),
            WavFormat::Pcm16,
            48_000,
            1,
            &sine(80.0 + i as f32 * 1.3, 0.7, 0.8, 48_000),
        );
    }

    let db = Database::open(data.path(), EMBEDDING_DIM).unwrap();
    let root = db
        .writer()
        .add_root(library.path().to_str().unwrap(), None)
        .unwrap();

    let cancel = CancellationToken::new();
    let mut options = ScanOptions::new(&cancel)
        .with_session(Arc::clone(session))
        .with_batch(batch);

    let before = rss_bytes();
    let started = Instant::now();
    let report = scan_root_with(&db, root, &mut options).unwrap();
    let elapsed = started.elapsed();
    let peak = rss_bytes();

    assert_eq!(
        report.embedded as usize, files,
        "not every file was embedded"
    );
    db.shutdown();

    EmbedRun {
        elapsed,
        rss: peak.saturating_sub(before),
        inferred: report.inferred,
        batches: report.inference_batches,
        files,
    }
}

#[derive(Debug)]
struct EmbedRun {
    elapsed: std::time::Duration,
    rss: u64,
    inferred: u64,
    batches: u64,
    files: usize,
}

impl EmbedRun {
    fn per_second(&self) -> f64 {
        self.files as f64 / self.elapsed.as_secs_f64()
    }
}

/// End-to-end throughput of the five-stage pipeline, and the peak RSS it costs.
///
/// The RSS assertion is the one that carries over to the real model unchanged: memory is
/// bounded by queue depth times payload size (`overview.md` §3), and neither of those
/// depends on how long a `run()` takes. The throughput figure does.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn embed_a_synthetic_library() {
    const FILES: usize = 400;

    let Some(session) = fixture_session() else {
        return;
    };
    println!("\n-- five-stage scan, {FILES} files --");
    println!("  provider       {:>8}", session.provider().as_str());

    let run = embed_library(FILES, BatchConfig::default(), &session);

    println!("  wall clock     {:>8} ms", run.elapsed.as_millis());
    println!("  throughput     {:>8.0} samples/s", run.per_second());
    println!(
        "  batches        {:>8} run() calls for {} spectrograms",
        run.batches, run.inferred
    );
    println!("  peak RSS       {:>8.1} MiB", mib(run.rss));
    println!("  ({})", graph_label());

    // Against the real model this is Phase 4's exit criterion. Against the fixture graph it
    // is a regression guard on the plumbing: a pipeline that cannot clear a few hundred
    // samples/s around a near-free model will not clear 60 around a real one.
    let floor = if is_real_model() { 60.0 } else { 200.0 };
    assert!(
        run.per_second() >= floor,
        "{:.0} samples/s is below the {floor:.0} floor for this graph",
        run.per_second()
    );
    assert!(
        mib(run.rss) < 800.0,
        "peak RSS {:.1} MiB exceeds the §7 budget",
        mib(run.rss)
    );
}

/// The batch-size tuning `overview.md` §3.4 defers to this phase.
///
/// It matters more than a normal tuning knob because inference is serialized behind one
/// mutex (see `model::session`): the batch size is what decides how much of the pipeline is
/// inside that critical section. A sweep against the fixture graph shows the *fixed*
/// per-`run()` overhead -- session dispatch, tensor construction, extraction -- which is
/// the part batching exists to amortize and the part that does not change when the graph
/// gets bigger.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn embed_batch_size_sweep() {
    const FILES: usize = 300;

    let Some(session) = fixture_session() else {
        return;
    };

    println!("\n-- batch size sweep, {FILES} files --");
    println!(
        "  {:>5}  {:>12}  {:>8}  {:>10}",
        "batch", "samples/s", "run()s", "peak MiB"
    );

    // The upper end of the sweep is not safe against the real model. HTSAT's activations
    // scale with the batch, and batch 16 alone was measured at 11 GiB resident -- 32 and 64
    // would put a 32 GB machine into swap and measure the pager rather than the model.
    let sizes: &[usize] = if is_real_model() {
        &[1, 2, 4, 8, 16]
    } else {
        &[1, 4, 8, 16, 32, 64]
    };

    let mut best = (0usize, 0.0f64);
    for &size in sizes {
        let run = embed_library(FILES, BatchConfig::of_size(size), &session);
        println!(
            "  {size:>5}  {:>12.0}  {:>8}  {:>10.1}",
            run.per_second(),
            run.batches,
            mib(run.rss)
        );
        if run.per_second() > best.1 {
            best = (size, run.per_second());
        }
    }

    println!("  best: {} at {:.0} samples/s", best.0, best.1);
    println!("  ({})", graph_label());
}

/// Dedup is worth more in this phase than in Phase 2, and this is the measurement that says
/// by how much: a duplicate skips a decode *and* an inference, and stores no bytes.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn duplicates_cost_nothing_to_embed() {
    const FILES: usize = 300;

    let Some(session) = fixture_session() else {
        return;
    };

    let data = tempfile::tempdir().unwrap();
    let library = tempfile::tempdir().unwrap();

    // Half the corpus is copies, which is not unrealistic for a folder of sample packs that
    // ship the same 909 kick under six names.
    for i in 0..FILES {
        let path = library.path().join(format!("hit_{i:05}.wav"));
        if i % 2 == 1 {
            std::fs::copy(library.path().join(format!("hit_{:05}.wav", i - 1)), &path).unwrap();
            continue;
        }
        write_wav(
            &path,
            WavFormat::Pcm16,
            48_000,
            1,
            &sine(80.0 + i as f32 * 1.3, 0.7, 0.8, 48_000),
        );
    }

    let db = Database::open(data.path(), EMBEDDING_DIM).unwrap();
    let root = db
        .writer()
        .add_root(library.path().to_str().unwrap(), None)
        .unwrap();
    let cancel = CancellationToken::new();
    let mut options = ScanOptions::new(&cancel).with_session(Arc::clone(&session));

    let started = Instant::now();
    let report = scan_root_with(&db, root, &mut options).unwrap();
    let elapsed = started.elapsed();

    let stored = db.embeddings().lock().unwrap().len_bytes();
    println!("\n-- half a library of duplicates, {FILES} files --");
    println!("  wall clock     {:>8} ms", elapsed.as_millis());
    println!(
        "  throughput     {:>8.0} samples/s",
        FILES as f64 / elapsed.as_secs_f64()
    );
    println!("  inferred       {:>8} of {FILES}", report.inferred);
    println!("  embeddings.bin {:>8.2} MiB", mib(stored));

    assert_eq!(
        report.embedded as usize, FILES,
        "a duplicate lost its vector"
    );
    assert_eq!(
        report.inferred as usize,
        FILES / 2,
        "the model ran on duplicate audio"
    );
    assert_eq!(
        stored,
        (FILES as u64 / 2) * EMBEDDING_DIM as u64 * 2,
        "duplicates stored their own copy of a vector they share"
    );
    db.shutdown();
}

/// The soak: memory has to stay flat as the corpus grows, with inference in the pipeline.
///
/// Same shape as `scan_memory_is_flat_as_the_library_grows`, and it is a separate test
/// because the embed stage adds the one queue in the pipeline that is *supposed* to be full
/// -- 64 spectrograms at 256 KB. If backpressure through it is wrong, this is where four
/// times the files costs four times the memory.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn embed_memory_is_flat_as_the_library_grows() {
    let Some(session) = fixture_session() else {
        return;
    };

    let small = embed_library(250, BatchConfig::default(), &session);
    let large = embed_library(1000, BatchConfig::default(), &session);

    println!("\n-- embed memory against library size --");
    println!("     250 files  {:>8.1} MiB", mib(small.rss));
    println!("   1000 files  {:>8.1} MiB", mib(large.rss));
    println!(
        "   ratio       {:>8.2}x for 4x the files",
        large.rss as f64 / small.rss.max(1) as f64
    );

    assert!(
        large.rss < small.rss.max(16 * 1024 * 1024) * 2,
        "4x the files cost {:.1} MiB against {:.1} MiB -- a stage is accumulating",
        mib(large.rss),
        mib(small.rss)
    );
}

// ---------------------------------------------------------------------------------------
// Phase 5 -- projection
// ---------------------------------------------------------------------------------------

/// A database holding `count` embedded samples with 512-dimensional vectors.
///
/// The vectors are L2-normalized noise around `clusters` centers, which is what a real
/// corpus looks like to a projector: mostly one diffuse blob with structure inside it. Pure
/// noise would be worse than unrealistic -- with no neighborhood structure at all, UMAP's
/// gradient descent has nothing to converge to and the timing would measure a pathology.
fn projection_corpus(count: usize, clusters: usize) -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), EMBEDDING_DIM).unwrap();
    let root = db
        .writer()
        .add_root("/Library/Audio/Samples", None)
        .unwrap();

    let mut rng = Xorshift(0x5EED_1234_9ABC_DEF0);
    for start in (0..count).step_by(CHUNK) {
        let end = (start + CHUNK).min(count);
        let rows: Vec<NewSample> = (start..end).map(|i| synthetic_sample(root, i)).collect();
        let ids = db.writer().upsert_samples(rows).unwrap();

        let vectors: Vec<Vec<f32>> = (start..end)
            .map(|i| {
                let center = i % clusters;
                let mut v: Vec<f32> = (0..EMBEDDING_DIM).map(|_| rng.next_f32()).collect();
                // A center strong enough to be a neighborhood and weak enough that the
                // clumps still touch -- a kNN graph in disconnected pieces is a different
                // benchmark (see `projection::umap`).
                v[center % EMBEDDING_DIM] += 3.0;
                v[(center * 31 + 7) % EMBEDDING_DIM] += 2.0;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect();
        let locs = {
            let mut store = db.embeddings().lock().unwrap();
            store.append_batch(&vectors).unwrap()
        };
        db.writer()
            .set_embeddings(ids.into_iter().zip(locs).collect())
            .unwrap();
    }
    db.writer().flush().unwrap();
    db.embeddings().lock().unwrap().sync().unwrap();
    (dir, db)
}

fn timed_refit(db: &Database, projector: &dyn Projector) -> (std::time::Duration, u64) {
    let cancel = CancellationToken::new();
    let rss_before = rss_bytes();
    let watcher = PeakRss::watch();
    let started = Instant::now();
    let report = refit(db, Refit::new(projector, &cancel)).unwrap();
    let elapsed = started.elapsed();
    let peak = watcher.finish().saturating_sub(rss_before);
    assert_eq!(report.sample_count, CORPUS);
    assert_eq!(report.algorithm, projector.name());
    (elapsed, peak)
}

/// **Exit criterion: a UMAP re-fit of 50,000 x 512 completes in under five minutes.**
///
/// Both projectors in one benchmark, because the interesting number is not either one
/// alone -- it is what PCA costs relative to the thing it is the fallback for. A fallback
/// that took the same five minutes would not be one.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn refit_50k_by_512() {
    let (dir, db) = projection_corpus(CORPUS, 24);
    let embeddings_bytes = std::fs::metadata(dir.path().join("embeddings.bin"))
        .unwrap()
        .len();

    let (pca_elapsed, pca_rss) = timed_refit(&db, &PcaProjector::new());
    let (umap_elapsed, umap_rss) = timed_refit(&db, &UmapProjector::default());

    let conn = db.read().unwrap();
    let points = queries::active_projection_points(&conn, 3).unwrap();
    assert_eq!(points.len(), CORPUS);
    assert!(points.iter().all(|(_, p)| p.iter().all(|v| v.is_finite())));

    println!("\n-- re-fit 50k x 512 --");
    println!("  embeddings.bin {:>8.1} MiB", mib(embeddings_bytes));
    println!(
        "  pca            {:>8.1} s   (peak RSS +{:.1} MiB)",
        pca_elapsed.as_secs_f64(),
        mib(pca_rss)
    );
    println!(
        "  umap           {:>8.1} s   (peak RSS +{:.1} MiB)",
        umap_elapsed.as_secs_f64(),
        mib(umap_rss)
    );

    assert!(
        umap_elapsed.as_secs_f64() < 300.0,
        "a 50k UMAP re-fit took {umap_elapsed:?}, target < 5 min"
    );
}

/// The other half of the §3.8 trade: what the *incremental* path costs on the same corpus.
///
/// The whole argument for incremental placement is that a small import must not pay for a
/// re-fit. That is a claim about a ratio, so it is measured as one -- against the PCA
/// number, which is the cheaper of the two re-fits and therefore the harder comparison.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn incremental_placement_of_a_small_import() {
    let (_dir, db) = projection_corpus(CORPUS, 24);

    let cancel = CancellationToken::new();
    let refit_started = Instant::now();
    refit(&db, Refit::new(&PcaProjector::new(), &cancel)).unwrap();
    let refit_elapsed = refit_started.elapsed();

    // 1% of the corpus: comfortably inside `INCREMENTAL_THRESHOLD`.
    let new = CORPUS / 100;
    let root = db
        .writer()
        .add_root("/Library/Audio/Samples", None)
        .unwrap();
    let rows: Vec<NewSample> = (CORPUS..CORPUS + new)
        .map(|i| synthetic_sample(root, i))
        .collect();
    let ids = db.writer().upsert_samples(rows).unwrap();
    let mut rng = Xorshift(0xABCD_0123_4567_89EF);
    let vectors: Vec<Vec<f32>> = (0..new)
        .map(|_| {
            let mut v: Vec<f32> = (0..EMBEDDING_DIM).map(|_| rng.next_f32()).collect();
            v[3] += 3.0;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect();
    let locs = {
        let mut store = db.embeddings().lock().unwrap();
        store.append_batch(&vectors).unwrap()
    };
    db.writer()
        .set_embeddings(ids.into_iter().zip(locs).collect())
        .unwrap();
    db.writer().flush().unwrap();

    let before: std::collections::HashMap<i64, Point3> = {
        let conn = db.read().unwrap();
        queries::active_projection_points(&conn, 3)
            .unwrap()
            .into_iter()
            .collect()
    };

    let rss_before = rss_bytes();
    let watcher = PeakRss::watch();
    let started = Instant::now();
    let report = place_incremental(&db, &cancel, 3).unwrap();
    let elapsed = started.elapsed();
    let rss = watcher.finish().saturating_sub(rss_before);

    assert_eq!(report.placed, new);

    // The claim the whole path exists for, checked rather than asserted in prose: not one
    // pre-existing coordinate changed.
    let conn = db.read().unwrap();
    let after: std::collections::HashMap<i64, Point3> =
        queries::active_projection_points(&conn, 3)
            .unwrap()
            .into_iter()
            .collect();
    assert_eq!(after.len(), CORPUS + new);
    for (id, old) in &before {
        assert_eq!(
            after[id], *old,
            "sample {id} moved during an incremental import"
        );
    }

    println!("\n-- incremental placement, {new} into 50k --");
    println!(
        "  placement      {:>8.0} ms  (peak RSS +{:.1} MiB)",
        elapsed.as_secs_f64() * 1000.0,
        mib(rss)
    );
    println!(
        "  pca re-fit     {:>8.0} ms  ({:.2}x)",
        refit_elapsed.as_secs_f64() * 1000.0,
        refit_elapsed.as_secs_f64() / elapsed.as_secs_f64().max(1e-9)
    );

    // The UMAP baseline runs *last*, deliberately. It allocates about 1.9 GB and evicts the
    // mmap pages the placement pass reads, so measuring it first would make the placement
    // number a measurement of page faults. This is also the comparison that matters: an
    // import must not cost what the re-fit it is avoiding costs, and in production a full
    // re-fit is a UMAP re-fit.
    let umap_started = Instant::now();
    let umap_report = refit(&db, Refit::new(&UmapProjector::default(), &cancel)).unwrap();
    let umap_elapsed = umap_started.elapsed();
    assert_eq!(umap_report.algorithm, "umap");

    println!(
        "  umap re-fit    {:>8.0} ms  ({:.1}x)",
        umap_elapsed.as_secs_f64() * 1000.0,
        umap_elapsed.as_secs_f64() / elapsed.as_secs_f64().max(1e-9)
    );
    assert!(
        elapsed.as_secs_f64() * 10.0 < umap_elapsed.as_secs_f64(),
        "placement took {elapsed:?} against a {umap_elapsed:?} re-fit"
    );
}

/// Exit criteria (`task.md` Phase 6): the 50,000-point cloud is one payload of **≤ 900 KB**,
/// and building it is not something the user waits on.
///
/// Also writes the payload to `target/point_cloud_50k.bin`, which is what
/// `scripts/decode_point_cloud.mjs` times the JavaScript half against -- the two halves of
/// the "< 300 ms to typed arrays" criterion measured against the *same bytes* rather than
/// against two independent fabrications of what the format is supposed to be.
#[test]
#[ignore = "benchmark; run under --profile perf"]
fn serve_a_fifty_thousand_point_cloud() {
    let data = tempfile::tempdir().unwrap();
    let db = Database::open(data.path(), EMBEDDING_DIM).unwrap();
    let root = db.writer().add_root("/library", None).unwrap();

    let mut rng = Xorshift(0xC10D);
    for chunk in (0..CORPUS).collect::<Vec<_>>().chunks(CHUNK) {
        let rows: Vec<NewSample> = chunk
            .iter()
            .map(|&i| NewSample {
                root_id: root,
                rel_path: format!("drums/{i:05}.wav"),
                filename: format!("{i:05}.wav"),
                ext: "wav".into(),
                size_bytes: 2048,
                mtime: 1_700_000_000 + i as i64,
                content_hash: None,
                duration_ms: Some(100 + i as i64),
                sample_rate: Some(48_000),
                channels: Some(1),
                status: SampleStatus::Decoded,
            })
            .collect();
        let ids = db.writer().upsert_samples(rows).unwrap();
        let features: Vec<(i64, SampleFeatures)> = ids
            .iter()
            .map(|&id| {
                (
                    id,
                    SampleFeatures {
                        spectral_centroid: Some(rng.next_f32() * 8000.0),
                        bpm: Some(120.0 + rng.next_f32() * 20.0),
                        ..Default::default()
                    },
                )
            })
            .collect();
        db.writer().set_features(features).unwrap();
    }
    db.writer().flush().unwrap();

    let conn = db.read().unwrap();
    let ids: Vec<i64> = search::sample_ids(&conn, &Default::default()).unwrap();
    drop(conn);
    assert_eq!(ids.len(), CORPUS);

    let run = db
        .writer()
        .begin_projection_run("pca", "{}", CORPUS as i64, 3)
        .unwrap();
    let points: Vec<(i64, Point3)> = ids
        .iter()
        .map(|&id| (id, [rng.next_f32(), rng.next_f32(), rng.next_f32()]))
        .collect();
    for chunk in points.chunks(CHUNK) {
        db.writer()
            .set_projection_points(run, chunk.to_vec())
            .unwrap();
    }
    db.writer().activate_projection_run(run).unwrap();
    db.writer().flush().unwrap();

    // What `get_point_cloud` does, timed as two halves: the statement, and the encode.
    let started = Instant::now();
    let conn = db.read().unwrap();
    let read = queries::active_projection_points(&conn, 3).unwrap();
    let query = started.elapsed();
    drop(conn);
    assert_eq!(read.len(), CORPUS);

    let started = Instant::now();
    let payload = audiobank_lib::ipc::binary::point_cloud(&read).unwrap();
    let encode = started.elapsed();

    // What `get_feature_column` and `query_samples` do.
    let conn = db.read().unwrap();
    let started = Instant::now();
    let column = search::feature_column(&conn, Feature::SpectralCentroid).unwrap();
    let column_query = started.elapsed();

    let started = Instant::now();
    let matches = search::sample_ids(
        &conn,
        &QueryFilter {
            features: vec![FeatureRange {
                feature: Feature::Bpm,
                min: Some(125.0),
                max: None,
            }],
            projected_only: true,
            ..Default::default()
        },
    )
    .unwrap();
    let filter = started.elapsed();
    drop(conn);
    assert_eq!(column.len(), CORPUS);

    let column_bytes = audiobank_lib::ipc::binary::feature_column(&column);
    let id_bytes = audiobank_lib::ipc::binary::id_list(&matches).unwrap();

    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("point_cloud_50k.bin");
    std::fs::write(&out, &payload).unwrap();

    println!("\n-- ipc payloads at 50k --");
    println!(
        "  point cloud      {:>9} bytes  ({:.0} KB, budget 900 KB)",
        payload.len(),
        payload.len() as f64 / 1024.0
    );
    println!(
        "    query          {:>8.1} ms\n    encode         {:>8.1} ms",
        query.as_secs_f64() * 1000.0,
        encode.as_secs_f64() * 1000.0
    );
    println!(
        "  feature column   {:>9} bytes  (query {:.1} ms)",
        column_bytes.len(),
        column_query.as_secs_f64() * 1000.0
    );
    println!(
        "  filter result    {:>9} bytes  ({} of {} ids, {:.1} ms)",
        id_bytes.len(),
        matches.len(),
        CORPUS,
        filter.as_secs_f64() * 1000.0
    );
    println!("  wrote {}", out.display());

    assert!(
        payload.len() <= 900 * 1024,
        "point cloud payload is {} bytes",
        payload.len()
    );
}
