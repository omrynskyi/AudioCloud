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

## Phase 5 — Projection

| Measurement                              | Target      | Actual                 |
| ---------------------------------------- | ----------- | ---------------------- |
| UMAP re-fit, 50,000 × 512                | < 5 min     | **58–70 s**            |
| PCA re-fit, 50,000 × 512                 | —           | 0.8–0.9 s              |
| Peak RSS, UMAP re-fit                    | see below   | **2.95 GiB**           |
| Peak RSS, PCA re-fit                     | —           | 95 MiB                 |
| Incremental placement, 500 into 50,000   | ≪ a re-fit  | **2.7 s (24× UMAP)**   |
| Median displacement, 5% import (PCA)     | small       | < 10% of diagonal      |
| Median displacement, 5% import (UMAP)    | small       | **11–16%**             |
| ...the same UMAP fits, unaligned         | —           | 24–87%                 |

`refit_50k_by_512` — 50,000 L2-normalized 512-dimensional vectors around 24 cluster centers,
projected by both projectors over the same `embeddings.bin`. PCA is **seventy times cheaper**
than UMAP, and that ratio is what makes it a usable fallback rather than a nominal one: a
library that trips the disconnected-graph guard gets a map in under a second instead of
waiting a minute to be told no.

**Run the two Phase 5 benchmarks in separate processes.** The wall-clock figures are stable
either way, but a UMAP re-fit leaves the process's resident set somewhere the next
measurement's baseline cannot interpret.

The RSS figures here are a **sampled high-water mark**, not the before/after delta the other
phases use. `ps -o rss` reports current resident size, and for a re-fit that allocates
gigabytes and frees them the delta measures whatever the allocator had not yet returned —
the same fit reported 1.6 GiB one run and 2.9 GiB the next. `PeakRss` polls every 50 ms
instead. Everywhere else in this file the delta is honest, because those benchmarks peak at
the end.

### The 3 GB is a gap in §7, not a pass

**A 50,000-sample UMAP re-fit peaks at 2.95 GiB.** §7 budgets peak RSS during a *scan* at
800 MB and has no row for a re-fit, so this violates nothing as written — and treating that
as a pass would be reading the table instead of the machine. It is 3.7× the scan budget, on
an application that ships to 8 GB Macs, in a job the user can trigger from a menu. §7 is
missing a row and Phase 10 should add it.

Where it goes: `hnsw_rs` stores the vectors it is given, so the index alone is 102 MB of f32
plus its graph; `annembed` then builds a sparse Laplacian, runs a randomized SVD over it, and
keeps gradient state per edge — 50,000 nodes × 15 neighbours is 750,000 edges. None of that
is under AudioBank's control. The mmap discipline in `projection/umap.rs` covers the part
that is: vectors reach the index 1,024 rows at a time and no second copy of the matrix is
ever materialized, which is why PCA over the same corpus peaks at 95 MiB.

Two mitigations exist today and neither is sufficient. A full re-fit is a background job at
background QoS rather than something an import triggers (§3.8), so the exposure is
occasional; and `Refit::with_fallback` means the app has a projector that costs 95 MiB when
UMAP will not run. What Phase 10 should actually weigh is capping the corpus a single re-fit
sees, chunking the fit, or holding the index in f16 — and, either way, measuring this on an
8 GB machine rather than a 32 GB one.

### On incremental placement, and the loop order that decides it

The argument for the incremental path is partly speed and mostly the guarantee: no existing
point moves, exactly, because no existing row is written. The benchmark asserts both.

The speed half is measured against **both** re-fits rather than the flattering one.
Placement is 23× cheaper than the UMAP re-fit it exists to avoid, which is the comparison
that matters — a full re-fit means a UMAP re-fit in production. It is about three times
*more* expensive than a PCA re-fit, and that is not a defect to tune away: brute-force
placement of 500 points against 50,000 anchors is 12.8 GFLOP, and PCA's covariance pass over
the same corpus is 6.6 GFLOP. Comparable work costs comparable time. What is not
comparable is the memory: placement peaks at 69 MiB against UMAP's 2.95 GiB.

What *was* a defect, and what only the benchmark caught, is the loop order. The first
implementation ran **5.4 s** — three and a half times the PCA re-fit — because it was
parallel over newcomers, each scanning every anchor, which re-reads and re-widens all 50,000
mmap rows once per newcomer: 25 million row decodes for 500 placements. Transposed, holding
the newcomers' vectors and widening each anchor row exactly once, it is the same arithmetic
with a hundredth of the memory traffic and a sequential mmap walk per thread.

### The optimization that made it slower

| Dot product, 20,000 × 32 × 512 | Throughput   |
| ------------------------------ | ------------ |
| `iter().zip().map().sum()`     | 2.0 GFLOP/s  |
| eight independent accumulators | 1.1 GFLOP/s  |

The inner loop of the placement pass is a 512-element dot product, and the textbook thing to
do to one is split the sum across independent accumulators: floating-point addition is not
associative, so a single accumulator is a serial dependency chain the compiler is not allowed
to reassociate, and eight lanes states that the reassociation is acceptable.

It is **1.8× slower**. LLVM already vectorizes the iterator form on aarch64, and the
hand-written version replaces good autovectorized code with worse hand-rolled code. The naive
loop stayed; `lane_split_versus_serial_dot` is the A/B, kept in the tree so the idea does not
get had a second time.

### On the two displacement figures

`task.md` Phase 5 asks that a re-fit with 5% new data leave existing points "substantially in
place after alignment". Measured as the median distance a pre-existing point moves, over the
diagonal of the cloud it moved in — an absolute distance means nothing across layouts whose
scale is arbitrary and then rescaled again by Procrustes.

PCA is stable more or less by construction, because `pca.rs` canonicalizes eigenvector signs.
**UMAP is the case the criterion is actually about**, and there the aligned median is 11–16%
against 24–87% for the same fits unaligned. That 24–87% spread is itself the point: an
unaligned UMAP re-fit's orientation is a coin flip, and sometimes the coin lands close.

Which is also why the *assertion* in `alignment_can_never_make_a_umap_layout_move_further` is
not an effect size. Procrustes minimizes the sum of squared distances over the
correspondences across all similarity transforms, and the identity is one of those, so the
aligned RMS can never exceed the unaligned RMS — on any layout, on any run. Asserting on the
effect size instead failed about one run in eight, which is a test that reports the weather.
The effect size belongs here, as a measurement.

## Phase 6 — IPC bridge

| Measurement                                | Target   | Actual           |
| ------------------------------------------ | -------- | ---------------- |
| Point cloud payload, 50,000 points         | ≤ 900 KB | **800,016 B**    |
| ...the query behind it                     | —        | 7.3 ms           |
| ...encoding it                             | —        | 1.3 ms           |
| Decode to typed arrays, 50,000 points      | < 300 ms | **0.001 ms**     |
| ...interleaving XYZ (Phase 7's, for scale) | —        | 2.0 ms           |
| Feature column payload, 50,000 values      | —        | 200,016 B        |
| ...the query behind it                     | —        | 6.6 ms           |
| Filter result, 18,604 of 50,000 ids        | —        | 74,432 B, 8.2 ms |

`serve_a_fifty_thousand_point_cloud` produces the Rust half and writes the payload to
`src-tauri/target/point_cloud_50k.bin`; `scripts/decode_point_cloud.mjs` reads that file and
times the JavaScript half. **Two languages, one criterion, the same bytes** — a hand-written
fixture on the JS side would have measured a second guess at what the format is rather than
the format.

### On the decode number

**0.001 ms is not a fast decoder. It is the absence of one**, and that is the entire argument
of `overview.md` §6.3. Decoding an `ABPC` payload is four `new Float32Array(buffer, offset,
n)` calls: no parse, no copy, no allocation beyond four view objects. The 300 ms budget exists
because the alternative — 3–4 MB of JSON, parsed into 50,000 objects, walked to build typed
arrays — genuinely costs hundreds of milliseconds and a GC spike that shows up as a stutter on
load. The measurement's job is not to celebrate a small number; it is to establish that the
number is small **for a structural reason**, so that a future change which reintroduces a copy
is visible as a regression of three orders of magnitude rather than of thirty percent.

The interleave row is the honest counterweight. Struct-of-arrays on the wire means the
frontend weaves the three planar columns into one interleaved position attribute on arrival,
and that loop is 2.0 ms — a real cost, deliberately moved to the frontend so a single column
can be fetched on its own when the map's colouring changes. It belongs to Phase 7's
`scene/buffers.ts` and is measured here only so the trade is stated in numbers.

### Where the milliseconds actually are

Every figure above is dominated by SQLite, not by the transport: 7.3 ms to read 50,000 rows of
`(sample_id, x, y, z)` through a `WITHOUT ROWID` primary key against 1.3 ms to serialize them.
That ordering is worth knowing before anyone optimizes the encoder. The filter query is the
slowest of the three at 8.2 ms, and it is doing the most work — a `LEFT JOIN` onto
`sample_features`, a range predicate, and a join through `projections` to the active run.

Absent from this table, deliberately: **end-to-end load → first frame**, which `overview.md`
§7 budgets at < 300 ms. That number spans the WebView IPC hop and Phase 7's geometry build,
neither of which exists yet, and measuring two thirds of it now would be a figure nobody could
compare against the one Phase 7 will produce. What Phase 6 owes it is the two halves above,
which sum to about 11 ms.
