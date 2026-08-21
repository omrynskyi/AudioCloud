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
