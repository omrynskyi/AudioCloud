# Benchmarks

A performance target with no recorded measurement is a wish (`task.md`, cross-cutting rule
9). Every number below was produced by a committed benchmark, not by hand.

```sh
cargo test --manifest-path src-tauri/Cargo.toml --profile perf --test benchmarks \
    -- --ignored --nocapture --test-threads=1
```

`--profile perf` rather than `--release`: the release profile aborts on panic and the test
harness needs to unwind. `--test-threads=1` is not optional either — RSS is a property of
the process, so two benchmarks running at once measure each other.

**Reference machine:** Apple M1 Pro, 32 GB, macOS 26.3.1, rustc 1.94.1. Recorded
2026-08-21.

## Phase 1 — Data layer

| Measurement                             | Target      | Actual        |
| --------------------------------------- | ----------- | ------------- |
| Insert 50,000 sample rows               | < 2 s       | **521 ms**    |
| Insert 50,000 feature rows              | —           | 136 ms        |
| `library.db` at 50,000 samples          | —           | 19.1 MiB      |
| Append 50,000 × 512 f16 embeddings      | —           | 45 ms         |
| `embeddings.bin` at 50,000 samples      | 48.8 MiB    | 48.8 MiB      |
| `mmap` the whole matrix                 | —           | 0.011 ms      |
| Widen and verify all 50,000 rows        | —           | 36 ms         |
| Embedding round-trip                    | bit-exact   | **bit-exact** |
| RSS to map + read 100 scattered rows    | ≪ 51 MiB    | **1.6 MiB**   |
| RSS after traversing all 50,000 rows    | see below   | 48.9 MiB      |

`insert_50k_samples_and_features` — 95,889 rows/s across 103 commits, so batching is doing
what it is there for: 50 transactions for 50,000 rows instead of 50,000.

`append_and_mmap_50k_embeddings` — 1,074 MiB/s appending, and every one of the 25.6 million
stored values reads back exactly equal to `f16::from_f32(original)`.

### On the two RSS numbers

The exit criterion says resident memory must not grow by 51 MB on read. Read literally,
traversing every row of a 48.8 MiB file will always approach that: the pages are mapped and
touched. The distinction the criterion exists to protect is **mapped vs. heap-loaded** —
those 48.9 MiB are clean, file-backed pages the kernel can evict under pressure, not a
`Vec<f32>` it cannot.

The falsifiable form of the claim is the sparse read, which is what the assertion actually
guards: map the matrix, touch a hundred scattered rows, and pay **1.6 MiB**. A heap load
would pay 51 MiB (102 MiB widened to f32) regardless of how few rows the caller wanted. The
number lands where the arithmetic says it should — a row is 1 KiB, an Apple Silicon page is
16 KiB, and 100 scattered rows means 100 pages.

## Phase 2 — Discovery and decode

| Measurement                          | Target        | Actual              |
| ------------------------------------ | ------------- | ------------------- |
| Decode + DSP throughput              | ≥ 400 /s      | **1,599 samples/s** |
| Rescan of an unchanged folder        | < 5% of cold  | **2.1%**            |
| Peak RSS, 4× the files               | flat          | **0.66×**           |
| Corrupt files abort the scan         | never         | **never**           |
| Analyze one 10 s window (DSP alone)  | —             | 15.4 ms             |

`scan_a_synthetic_library` — 400 files, 89 MiB, mixed 16-bit and float WAV at 22.05 / 44.1 /
48 / 96 kHz, 40 of them byte-exact duplicates. The cold scan decodes 360 files in 225 ms and
deduplicates the other 40 without opening a decoder. The rescan reads no file contents at
all: 5 ms to walk the tree and match 400 `(mtime, size)` pairs against the stamps loaded in
one query.

The throughput figure counts **files actually decoded**, not files seen. An earlier version
of the generator derived its tone from `i % 40`, which made 60% of the corpus accidental
duplicates and reported 3,202 files/s — a measurement of the dedup table, not of the
decoder. The benchmark now asserts that the only duplicates are the intended ones.

### On the memory criterion

`scan_memory_is_flat_as_the_library_grows` is the falsifiable form of "backpressure is the
memory strategy" (`overview.md` §3): scan 250 files, then scan 1,000 in a fresh database, and
compare. Four times the files cost **0.66×** the resident memory — noise around a constant,
which is what bounded queues times bounded payloads predicts. A stage that accumulated
instead of streaming would show 4×.

### Where the DSP time goes

15.4 ms for a full 10-second window, attributed by removing one stage at a time:

| Stage                                     | Cost     |
| ----------------------------------------- | -------- |
| STFT + flux + descriptors (1,000 frames)  | 12.4 ms  |
| Chroma pass (116 frames of 8192-point)    | 3.0 ms   |
| Spectral flatness (513 `ln` per frame)    | 1.75 ms  |

The main STFT dominates, and it is not worth optimizing: those 1,000 frames are the CLAP
input contract (`overview.md` §3.3), so Phase 3's mel front-end needs them regardless. The
descriptors ride along on a transform that is already paid for, which is the arrangement §3.3
describes.

The number is also an upper bound rather than a typical cost. It is measured on a full
10-second window; a drum library is overwhelmingly sub-second one-shots, and the end-to-end
figure of 1,599 samples/s is what those actually cost.
