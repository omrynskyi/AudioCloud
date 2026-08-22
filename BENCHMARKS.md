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

## Phase 4 — Embedding pipeline

**The headline number for this phase is not here, and cannot be.** Phase 4's exit criterion
is ≥ 60 samples/s end to end *with CLAP*, and the CLAP export (Phase 3) has not been run —
`ModelRelease::CURRENT.sha256` is still the `UNPINNED` sentinel. What the graph in these
measurements does is carry the real export's input signature and output width and compute
deterministic nonsense in between (`tests/fixtures/session/frames_major.onnx`, ~130 KB).

So every figure below is **the pipeline around a near-free model**: an upper bound on
end-to-end throughput, and the exact cost of everything that is not a matmul. It is a real
measurement of a real thing — five stages, one shared session, batching, backpressure,
`embeddings.bin` — with the model-shaped hole labelled rather than papered over.

| Measurement                                  | Target        | Actual              |
| -------------------------------------------- | ------------- | ------------------- |
| Five-stage throughput (fixture graph)        | ≥ 60 /s †     | **1,641 samples/s** |
| Peak RSS, 400-file five-stage scan           | < 800 MB      | **17.5 MiB**        |
| Peak RSS, 4× the files                       | flat          | 1.3 → 3.3 MiB       |
| `run()` calls for 400 files at batch 16      | 25            | 25                  |
| Inference runs, 50% duplicate corpus         | half          | **150 of 300**      |
| `embeddings.bin`, 50% duplicate corpus       | 150 vectors   | **0.15 MiB**        |
| Mel front-end, sparse vs. dense filterbank   | bit-identical | **bit-identical**   |

† The target is stated against real CLAP and is not claimed by this row. The row is a
regression guard: a pipeline that cannot clear a few hundred samples/s around a free model
will not clear 60 around a real one.

`embed_a_synthetic_library` — 400 sub-second one-shots, walk → decode → mel → embed →
persist, CoreML bound. `duplicates_cost_nothing_to_embed` is the measurement that says what
dedup is worth once inference is in the pipeline: half the corpus is byte-identical copies,
and they cost neither a decode, nor a `run()`, nor a byte of `embeddings.bin` — six rows can
point at one vector.

### The mel filterbank was the whole pipeline

The first run of `embed_a_synthetic_library` reported **218 samples/s**, against Phase 2's
1,599 samples/s for decode + DSP alone. The batch-size sweep was flat across 1 → 64, which
is the tell: if batch size does not matter, inference is not the bottleneck.

It was the filterbank. Applying a 64 × 513 mel bank densely to 1,001 frames is 32.9 million
f64 multiply-adds per file, and a mel filter is a triangle spanning about sixteen FFT bins —
so essentially all of that arithmetic was summing zeros. `FrontEnd` now stores each row's
nonzero span and multiplies only over it: **1,641 samples/s**, a 7.5× end-to-end gain, and
the pipeline is back to costing what decode + DSP costs.

The output is **bit-identical**, which matters because this is the parity surface
(`overview.md` risk 4). The skipped terms are `0.0 * p` for finite `p`, and adding exact
zero to an f64 accumulator changes nothing;
`mel::tests::sparse_application_is_bit_identical_to_the_dense_one` compares every one of the
64,064 values by bit pattern against the dense loop rather than by tolerance.

### On the batch size

| batch | samples/s | `run()`s | peak MiB |
| ----- | --------- | -------- | -------- |
| 1     | 1,546     | 300      | 5.3      |
| 4     | 1,583     | 75       | 13.3     |
| 8     | 1,595     | 38       | 7.1      |
| 16    | 1,594     | 19       | 4.9      |
| 32    | 1,558     | 10       | 32.1     |
| 64    | 1,550     | 5        | 51.2     |

Flat within noise, and that is the honest reading: **this sweep cannot tune the batch size**,
because the thing batching amortizes — the cost of a `run()` — is nearly zero for a 130 KB
graph. What it does establish is that the fixed per-call overhead outside the graph (tensor
construction, dispatch, extraction, the mutex) is small, that nothing in the batcher degrades
with size, and that memory grows with it as the arithmetic predicts.

The default stays at 16, the low end of `overview.md` §3.4's 16–32. Re-run this sweep against
the real model before changing it; that run is where the number gets chosen, and it is the
run that decides whether one serialized session is enough or whether the fallback in
`model/session.rs` — one session per worker — is needed.
