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

---

## Phase 7 — 3D visualization

Everything here is measured in a real WKWebView on Apple GPU, not in a desktop browser.
`scripts/webview_eval.swift` is a WKWebView that loads a page, evaluates one expression and
prints the JSON it resolves to; `scripts/probe_webgl_caps.mjs` and
`scripts/orbit_profile.mjs` drive it. That matters more here than anywhere else in the
project: `overview.md` §5.1's entire caveat is that WebKit runs WebGL through ANGLE on Metal
and does not behave like Chrome, so a number from Chrome is not evidence about this app.

```sh
npm run probe:webgl     # the point-size cap, and the rest of the context's limits
npm run profile         # builds the harness, then the 30 s orbit and the exit criteria
```

### The first task: `ALIASED_POINT_SIZE_RANGE`

| Parameter                            | Value                    |
| ------------------------------------ | ------------------------ |
| `ALIASED_POINT_SIZE_RANGE`           | **`[1, 511]` device px** |
| ...in CSS px at dpr 2                | `[0.5, 255.5]`           |
| Context                              | WebGL 2.0, GLSL ES 3.00  |
| Unmasked vendor / renderer           | Apple Inc. / Apple GPU   |
| `EXT_disjoint_timer_query_webgl2`    | **absent**               |
| `WEBGL_lose_context`                 | present                  |

511 px is **eight times** the 64 px floor at which `overview.md` §5.1 says to abandon
`THREE.Points` for `InstancedMesh` quads. The sprite path stands, and the architectural
question the roadmap put first in the phase is closed on the first day rather than
discovered in Phase 10. `scene/caps.ts` still performs the same query at startup and the
shader clamps to whatever it returns, because this is one Mac.

The two extension rows are not incidental. **No GPU timer** means WebKit will not tell a page
how long the GPU spent on a frame, so `overview.md` §7's "`WebGLRenderer.info` + Chrome
DevTools trace" has no in-engine equivalent and the frame-time number below has to be built
out of frame intervals instead. **`WEBGL_lose_context` present** means the context-loss
criterion is testable for real rather than by inspection.

### The 30-second scripted orbit

50,000 synthetic points in the real `ABPC` wire format, clustered the way UMAP output
clusters, in a 1440×900 window at dpr 2 — a 2880×1800 drawing buffer.

| Measurement                                | Target        | Actual                       |
| ------------------------------------------ | ------------- | ---------------------------- |
| Frame interval, p50                        | —             | **16.00 ms** = the display   |
| Frame interval, p99                        | < 16.6 ms     | **33 ms** — see below        |
| CPU inside `render()`, p99                 | —             | **2 ms**                     |
| Draw calls per frame                       | 1             | **1** (50,000 points)        |
| Idle frames, 3 s after the orbit           | 0             | **0**                        |
| Decode + interleave + geometry build       | —             | **~1 ms**                    |
| Page start → first frame                   | < 300 ms      | **91–115 ms**                |
| Hover pick, p50 / p99                      | —             | **2 ms / 9 ms**              |
| Pick agreement with a CPU model            | correct       | **200 / 200**                |
| Context loss → drawing again, same buffer  | no refetch    | **yes**                      |

And the controls, measured in the same page, in the same run, for the same wall time:

| Control                                        | p50      | p95   | p99       |
| ---------------------------------------------- | -------- | ----- | --------- |
| `requestAnimationFrame` on a blank page         | 16 ms    | 20 ms | **23 ms** |
| The same orbit loop with the cloud hidden       | 16 ms    | 24 ms | **37 ms** |
| The same orbit loop drawing 50,000 points       | 16 ms    | 24 ms | **33 ms** |

### On the frame-time row, which is the one that fails

The criterion is stated as an absolute: p99 under 16.6 ms. As measured it is 33 ms, and the
honest reading of that number is not "the renderer is twice too slow".

**An empty page in this engine has a p99 of 23 ms.** No WebGL context, no scene, no
JavaScript beyond an `requestAnimationFrame` loop that increments a counter. About one frame
in a hundred misses its vsync deadline for reasons entirely outside this project, and the
criterion's threshold is below that floor — no renderer, however fast, can meet 16.6 ms p99
where the empty case is 23. The third control settles what the scene itself costs: the same
loop with `points.visible = false` measures p99 **37 ms**, which is *worse* than the run
drawing all 50,000 points. Two samples of the same noise. Drawing the cloud is not
detectable in this measurement.

What can be said positively, and is:

- **p50 is 16 ms, exactly the display interval**, in every configuration. The common case is
  the display's own rate, which is what "60 fps orbit" means.
- **CPU time inside `render()` is 2 ms at p99**, so the main thread is nowhere near the
  budget and cross-cutting rule 1 holds with room to spare.
- **One draw call for 50,000 points**, which is the structural claim §5.1 makes and the
  reason the sprite path was chosen.
- Point count does not move any of it: at **200** points the p99 was 24 ms, the same tail.

What would actually settle the criterion is a GPU-side measurement, and WebKit does not
expose one. That is Phase 10's Instruments pass against the real app, and it is the right
place for it: `task.md` already assigns the Instruments work there, and Metal System Trace
sees the frame the way the driver does rather than the way a page's timer can.

### On the pick check, and the bug it found

"GPU pick returns the correct sample under a dense cluster" is not a number, so it was made
into one. `src/profile/checks.ts` reimplements the sizing chain from `points.vert.glsl` in
JavaScript — projection, depth fade, LOD shrink, the clamp to the device's point-size range,
the sub-pixel discard — and asks which point that model says should win on a given pixel.
Two implementations of one rule, in two languages, compared on 200 pixels each carrying a
stack of four to six overlapping sprites.

It found a real bug immediately: 181 of 200 disagreements, because `Picker` converted CSS
coordinates to device pixels with `Math.round`, and `Math.round(n + 0.5)` is `n + 1`. Every
pick was landing one pixel off — invisible by eye in a cluster, wrong every single time.

The first version of the reference was wrong too, and more interestingly. It asked which
point's *centre* lay on the cursor's pixel. The GPU was right and the model was not: points
are sprites several pixels across, so the point under the cursor is the one whose sprite
*covers* that pixel, and among those, the one nearest the camera. That is what the depth test
in `pick.frag.glsl` produces and what a person means by clicking on something.

Current agreement is 200/200 — 171 exact, 29 where the GPU returned a different point that
also covers the cursor and is no further from the camera, which is the model declining to
have an opinion about a sprite within three quarters of a pixel of its edge. Zero behind,
zero uncovered, zero missed.

### On the measurement window, which took longer than the renderer

macOS suspends `requestAnimationFrame` for a window it considers occluded, and it will
consider a window occluded for sitting behind a full-screen terminal. In that state the page
reports `document.visibilityState === "hidden"`, runs **zero** frames, and the harness hangs
— indistinguishable, from the outside, from a bug in the scene. Several confusing hours were
spent on the scene before the page was asked what it thought its own visibility was.

`scripts/webview_eval.swift` now joins all Spaces, activates, and disables WebKit's window
occlusion detection outright through SPI. That last one is unacceptable in a shipped app and
is fine in a developer measurement tool that is never bundled; the alternative is a benchmark
whose ability to run depends on which window happens to be in front. It also warns on stderr
if the system reports the window occluded anyway, so a throttled number cannot be recorded as
a real one.

## Phase 8 — Audio preview engine

| Measurement                                       | Target        | Actual        |
| -------------------------------------------------- | ------------- | ------------- |
| `play_pcm` → first audible sample                  | < 50 ms       | **~16–17 ms** |
| Audio-thread allocations                           | exactly 0     | **enforced**  |
| Rapid retrigger storm (20× in 300 ms)               | no panic      | **no panic**  |
| `stop()` mid-clip                                   | no panic      | **no panic**  |

Measured against the real default output device via `cargo test --test audio_hardware --
--ignored`, which is not run in CI — headless runners have no audio device, and this is the
one Phase 8 measurement that genuinely needs one. `tests/audio_hardware.rs` opens
[`Engine`], plays a synthetic sine burst and a real decoded WAV alike, and prints
`play_pcm -> first audible sample: 16.98ms` (and, on a second run, 16.12ms) on the machine
this was developed on.

**What the number is, precisely.** Elapsed time from `Engine::play_pcm` being called to the
real-time callback's first non-silent sample, read off `Transport::first_audible_nanos`. This
is the half of the hover-to-audible budget the audio engine owns: queueing, the attack ramp,
and the ring hand-off. It excludes the IPC round trip from a pointer event to the
`play_sample` command arriving, and it excludes the frontend's 120 ms hover debounce
(`scene/PointCloud.tsx`), because neither is observable from a Rust test — `overview.md` §7's
"hover → audio, < 50 ms" is stated as the whole chain, and 17 ms leaves comfortable room for
the rest of it. There is no in-process way to measure the full chain the way Phase 7's
`scripts/webview_eval.swift` measures a real WKWebView frame; that would need an instrumented
build with a real pointer event, which is a Phase 10 Instruments-pass question, not a unit
test.

**Zero allocations is not measured here — it is enforced**, the same distinction Phase 5's
2.95 GiB re-fit finding draws between a measured number and a budgeted one, except this one
actually holds: `audio::guard::GuardedAlloc`, installed as `#[global_allocator]` in debug
builds, panics the instant `alloc`/`dealloc`/`realloc` runs while an `AudioThreadGuard` is
active. `tests/audio_guard.rs` proves the panic actually fires — not merely that the flag
toggles — by installing its own copy of the allocator in a dedicated integration test binary
and triggering a real allocation from inside a real guard.

### On the retrigger design, and why it needed no buffer clearing from outside the audio thread

`overview.md` §2 forbids the audio thread from locking, same as it forbids allocating.
Retriggering — a fast hover sweep, or a click while a clip is already playing — needed a way
to discard whatever the previous clip had queued without either side taking a lock. The
answer is a generation counter (`Transport::generation`) rather than an explicit clear
command: [`Engine::play_pcm`] bumps it, the real-time callback notices the mismatch on its
very next call and calls `RingConsumer::clear()` itself — which it may, because it is that
ring half's sole owner — and a decode-ahead task mid-push notices the same mismatch and stops
writing. Nobody ever reaches across the SPSC boundary to touch the other side's half.
`audio::engine::tests::a_new_generation_clears_the_previous_ones_stale_audio` is the
unit-level proof, over a plain `HeapRb` with no `cpal` stream involved; `rapid_retriggers_do_not_panic_or_deadlock`
above is the same property under real hardware and real scheduling.
