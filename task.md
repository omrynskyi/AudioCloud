# AudioCloud — Implementation Roadmap

Companion to `overview.md`. That document is the architecture; this one is the order of
operations.

**How to read this.** Phases are sequential by default — each one produces something that
runs, and each one's exit criteria are falsifiable. Do not start a phase whose predecessor
has not met its exit criteria; the criteria exist to stop "mostly working" from
compounding into a system nobody can debug.

**Parallelism.** If more than one person is working, Phase 7's shader and scene scaffolding
can proceed against synthetic point data in parallel with Phases 1–2, and Phase 9's shell
can be built against mocked IPC in parallel with Phases 4–6. Nothing else parallelizes
cleanly. Solo, read it as strictly sequential.

**Pinning.** Every version below was verified on crates.io at the time of writing. Pin
exact versions and commit `Cargo.lock`.

---

## Phase 0 — Toolchain & Scaffolding

*Goal: an empty window that builds and launches on both architectures.*

- [x] Install Rust stable; add both targets:
      `rustup target add aarch64-apple-darwin x86_64-apple-darwin`
- [x] Install Xcode Command Line Tools; verify `lipo`, `codesign`, `xcrun notarytool`
- [x] Install Node LTS (build-time only — see `overview.md` §1); pin in `.nvmrc`
- [x] Install the Tauri v2 CLI (`cargo install tauri-cli --version "^2"`)
- [x] Scaffold: `cargo create-tauri-app` → React + TypeScript + Vite
- [x] Add Tailwind, configure `content` globs for `src/**/*.{ts,tsx}`
- [x] Add `three`, `@react-three/fiber`, `@react-three/drei`, `zustand`
- [x] Establish `src-tauri/` module skeleton per `overview.md` §9 — empty modules with
      `//!` doc comments stating each one's responsibility
- [x] `.gitignore`: `target/`, `node_modules/`, `dist/`, `*.onnx`, `*.dmg`,
      `src-tauri/gen/`, and the local app-data dir if it is ever symlinked in
- [x] `rustfmt.toml`, `clippy.toml`; ESLint + Prettier for the frontend
- [x] `cargo tauri dev` launches a window
- [x] `cargo tauri build --target aarch64-apple-darwin` and
      `--target x86_64-apple-darwin` both succeed (unsigned is fine here)
- [x] CI skeleton: fmt, clippy, `cargo test`, `tsc --noEmit`, both target builds

**Exit criteria:** an empty window renders via `cargo tauri dev`; both single-arch release
builds succeed in CI; `cargo clippy -- -D warnings` is clean.

---

## Phase 1 — Data Layer First

*Goal: somewhere for ingest to land, built before the thing that produces the data.*

Building this first is deliberate. A pipeline written before its sink grows an ad-hoc
in-memory store that has to be torn out later.

- [x] Add `rusqlite` (0.40, `bundled` feature), `r2d2`, `r2d2_sqlite` (0.35),
      `refinery` (0.9)
- [x] Resolve and create the app data dir (`~/Library/Application Support/<bundle-id>/`);
      create on first run with correct permissions
- [x] Write `V1__initial.sql` — the full DDL from `overview.md` §4.1, including the FTS5
      virtual table and the partial unique index on `projection_runs(is_active)`
- [x] Wire `refinery` to run migrations at startup, before any other connection opens
- [x] Pragma application (`overview.md` §4.3) via an `r2d2` connection customizer, so
      **every** pooled connection gets `foreign_keys = ON` — it is per-connection
- [x] Implement the single writer thread: `mpsc` command enum, oneshot replies, batched
      transactions flushing at 1000 rows **or** 250 ms, whichever first
- [x] Implement the read pool (4 read-only connections)
- [x] Implement `db/embeddings.rs`: append-only `embeddings.bin`, f16 storage via `half`,
      `(offset, len)` return, `memmap2` whole-matrix read, `compact()` maintenance path
- [x] Unit tests: migration idempotency, FK enforcement actually rejects an orphan,
      writer batching commits on both the count and the timeout trigger
- [x] Benchmark: insert 50,000 synthetic sample rows + features
- [x] Benchmark: append 50,000 × 512 f16 embeddings, mmap them back, verify round-trip
      values and that resident memory does not grow by 51 MB on read

**Exit criteria:** 50,000 synthetic rows insert in **< 2 s**; the embedding file
round-trips bit-exactly; `PRAGMA foreign_key_check` is clean; migrations run twice with no
effect the second time.

---

## Phase 2 — Discovery & Decode

*Goal: point the app at a real sample folder and get real rows, with no ML involved.*

- [x] Add `ignore` (0.4), `blake3` (1.8), `symphonia` (0.6, all relevant codec features),
      `rubato` (5.0), `rayon` (1.12), `realfft` (3.5), `ebur128` (0.1)
- [x] `pipeline/walk.rs`: `ignore::WalkBuilder::build_parallel()`, extension allowlist,
      symlink-loop guard, size sanity bounds
      > Note: **not `jwalk`** — deprecated upstream ("Use `dua-core` instead").
        See `overview.md` §3.1.
- [x] `(path, mtime, size)` fast-skip against existing rows before any hashing
- [x] `blake3` content hashing, with head+tail+length sampling for files > 64 MB
- [x] Dedup: link a matching `content_hash` to the existing embedding rather than
      re-processing
- [x] `pipeline/decode.rs`: `symphonia` probe + streaming decode, **hard stop at 10 s of
      48 kHz output**, mono downmix during decode, `rubato` resample
- [x] Buffer pool for decode output — no per-sample allocation of the 1.92 MB buffer
- [x] `pipeline/features.rs`: `realfft` frames, plus peak / RMS / LUFS / centroid /
      flatness / ZCR / onset density / BPM / key from the same pass
- [x] Decode-failure quarantine: `status = 'decode_failed'` + error string, scan continues
- [x] Wire the bounded channels between walk → decode → features → writer, with the queue
      depths from `overview.md` §3
- [x] Temporary dev-only command to trigger a scan and dump counts (real IPC is Phase 6)
- [ ] Benchmark on a real library of ≥ 5,000 files across mixed formats
      > Open. The committed numbers in `BENCHMARKS.md` come from a 400-file **synthetic**
        corpus of mixed formats and sample rates. Needs a run against a real library.

**Exit criteria:** a real folder scans to the database with correct metadata and DSP
features; corrupt and unsupported files are quarantined without aborting the scan; a
re-scan of an unchanged folder completes in **< 5%** of the original time; decode+DSP
throughput **≥ 400 samples/s**; peak RSS during scan stays flat as the file count grows.

---

## Phase 3 — Model Provisioning

*Goal: a verified CLAP ONNX model on disk and a warm `ort` session — with proof the mel
front-end matches the graph.*

- [x] Write `scripts/export_clap_onnx.py` — **one-time, offline, developer-only.** Loads
      LAION CLAP, exports the **audio tower only** (the text tower is dead weight), fixed
      or dynamic batch axis, opset pinned. Document exact package versions used.
      > Written, with versions pinned in its docstring and a `--recipes-only` mode that
        needs neither torch nor the checkpoint. Batch is dynamic; the mel and frame axes
        are fixed, so a graph that would accept the wrong front-end cannot load at all.
- [ ] Run the export once. Record the SHA-256. Publish the `.onnx` as a release asset.
      Do not commit a 200 MB binary to git.
      > **Open, and it is the phase's blocker.** Needs a machine with `torch` and
        `laion-clap`. Until it is run, `ModelRelease::CURRENT.sha256` is the `UNPINNED`
        sentinel and `Downloader::ensure` refuses to open a socket — fail-closed, so there
        is no state in which 200 MB arrives unverifiable.
- [ ] **Record reference outputs**: run 20 diverse fixture wavs through Python CLAP and
      commit the resulting embeddings as a test fixture. This is the parity oracle.
      > Half done. The 20 fixtures exist as committed *recipes* rather than 19 MB of wavs
        (`src-tauri/tests/fixtures/clap/fixtures.json`), generated identically in Python
        and Rust, each carrying a waveform probe so a drift between the two generators
        reports itself rather than arriving disguised as a parity failure. The embeddings
        wait on the export above.
- [x] Add `reqwest` (0.13, streaming), `sha2` (0.11), `ort` (**pinned exactly**
      `2.0.0-rc.13`)
- [x] `model/download.rs`: resumable streaming download with `Range`, incremental SHA-256
      over the stream, atomic rename from `.partial` only after verification, `fsync`
      before rename
      > Plus a `fsync` of the *directory* after the rename: without it the rename is
        metadata that a power loss can lose even though the data was durable. Resume
        re-hashes the existing partial from disk, since a `Sha256` state cannot cross a
        process restart.
- [x] Distinct handling for: no network / hash mismatch / disk full / interrupted-resumable
      > Six variants, not four: `Http` (a 404 means the asset moved and no retry fixes it),
        `Cancelled`, and `ReleaseNotPinned` also have distinct recoveries. `is_resumable()`
        is what the first-run screen will branch on. A hash mismatch **deletes** the
        partial — resuming into bytes that are already wrong would wedge the app in a loop.
- [x] `model/session.rs`: session init with CoreML EP, **verified** CPU fallback; log which
      EP actually bound; warmup run on a zero tensor to pay first-inference cost off the
      user's critical path
      > The warmup *is* the verification: a session is not returned until a real `run()` has
        succeeded on it, so CoreML registering and then rejecting the graph on first
        inference falls back at init instead of halfway into a 50,000-file scan.
        `AUDIOCLOUD_FORCE_CPU=1` forces the CPU path without a rebuild.
- [x] Lazy init — session construction must not be on the cold-start path
      (`overview.md` §7)
      > `LazySession`, `.manage()`d during setup; construction is two path joins and an
        empty cell, asserted by a test. A failed attempt is deliberately not cached: init
        fails for reasons that get fixed while the app is running.
- [ ] **Parity gate:** the mel front-end from Phase 2 feeding this session must reproduce
      the committed reference embeddings at **cosine similarity > 0.999**. Commit this as
      a regression test, not a one-time check.
      > The test is committed (`src-tauri/tests/parity.rs`) and skips with a printed reason
        until the oracle exists. `the_parity_gate_is_not_silently_disabled` fails the build
        if the release is ever pinned without a committed oracle, so the skip cannot become
        permanent by accident.
      > The front-end itself was not Phase 2's — `features.rs` left the filterbank to this
        phase — and is now `pipeline/mel.rs`. Pending the real oracle it is cross-checked
        against an independent NumPy implementation of `librosa.filters.mel` and the whole
        STFT chain: the filterbank agrees to 5e-8, and the log-mel to 2e-5 dB for every bin
        within 40 dB of the peak. That is not the gate, and it is not nothing.
- [x] Isolate every `ort` type behind `model/session.rs` so an rc upgrade touches one file
- [x] *(added)* Synthetic ONNX fixtures so the session path is testable without the real
      model — `scripts/make_session_fixtures.py`, `src-tauri/tests/session.rs`
      > Three ~130 KB graphs with the real export's input signature: frames-major,
        mels-major, and one with a dynamic mel axis that must be rejected. Proves session
        init, CoreML binding, the warmup, layout detection and its transpose, batching, and
        extraction — everything except parity, which needs the real model.

**Exit criteria:** model downloads, verifies, survives a kill -9 mid-download and resumes;
session initializes on both CoreML and forced-CPU; **the parity test passes at > 0.999**.

> If parity fails, stop. Do not proceed to Phase 4. Every embedding produced by a
> mismatched front-end is silently wrong and the map will look completely plausible.

**Status: not met.** Resume-after-interruption is tested against a server that truncates
mid-body, and the resumed request asks for the exact byte offset — but against a synthetic
release, not the real asset, and the parity test has nothing to compare against. The gate
is blocked on the export, and nothing downstream should start until it is green.

**Four findings from this phase that change later ones:**

1. **`ort` sessions cannot run concurrently.** `overview.md` §3.4 prescribes `Arc<Session>`
   across `rayon` workers on the grounds that ONNX Runtime is thread-safe for concurrent
   `run()`. It is not; `ort` 2.0.0-rc.13 takes `&mut self` precisely because earlier
   versions that allowed it "often saw crashes and memory corruption". The session is still
   built once and shared as an `Arc`, with inference serialized behind a `Mutex` — which is
   affordable only because §3.4 also prescribes batching. **Phase 4's batch-size tuning is
   now load-bearing rather than an optimization**, and if the throughput target is missed
   the fix is one session per worker thread, not concurrent `run()`.
2. **CoreML does not compute in f32.** Against an f64 NumPy reference, the CoreML provider
   differs by up to ~1.5e-5 per component where CPU differs by ~1e-7; CoreML and CPU differ
   from each other by ~5e-5. That is fp16 accumulation on the ANE. It does not threaten the
   parity gate's 0.999 cosine floor, but **Phase 5 must not assume a tolerance tighter than
   this**, and a re-fit that runs on CoreML is not bit-reproducible against one that runs on
   CPU.
3. **ONNX Runtime links statically.** `otool -L` on the built binary shows
   `CoreML.framework` and no `libonnxruntime`. **Phase 11's §8.1 problem is smaller than
   written**: both approaches it proposes are about merging two ONNX Runtime *dylibs*, and
   there is no dylib to merge. The `lipo -info` sweep over the bundle stays; there is simply
   less in it.

---

## Phase 4 — Embedding Pipeline

*Goal: 50,000 samples embedded end-to-end, with a recorded throughput number.*

- [x] `pipeline/embed.rs`: batching accumulator (start at 16, tune) with a flush timeout so
      the tail of a scan does not stall on an unfillable batch
      > The accumulator talks to an `Embed` trait, not to `ort`. That keeps cross-cutting
        rule 7's blast radius at one file and makes the batcher's own behaviour testable
        against a counting fake — including the property a real session makes nearly
        unobservable, that row 3's vector reaches row 3 and not row 5.
- [x] `Arc<Session>` shared across `rayon` workers — one session, never one per thread
      > One session, one embed thread. A second thread would only queue on the same mutex;
        see Phase 3's finding 1. The parallelism is upstream, in decode and mel.
- [x] L2-normalize outputs immediately on receipt
- [x] Wire the full five-stage pipeline: walk → decode → mel → embed → persist
      > Four threads and the `rayon` pool, three queues rather than four: **decode and mel
        share a stage**, so what crosses the next channel is the 256 KB spectrogram and not
        the 1.92 MB window. `overview.md` §3 draws them apart and then says decode output
        should be "handed over as mel frames wherever possible"; this is that, and it is why
        peak RSS for a 400-file scan is 17.5 MiB rather than a queue depth times 1.92 MB.
- [x] `pipeline/progress.rs`: atomic counters per stage + a single 100 ms ticker
      (`overview.md` §6.5). **No per-file events.**
      > The ticker drops snapshots that are identical to the last one, so an idle stage costs
        the sink nothing, and it guarantees a terminal snapshot on drop — a UI cannot be left
        at 99%. Phase 6's `Channel` is a sink like any other; nothing on the producer side
        changes when it arrives.
- [x] `CancellationToken` threaded through every stage; verify a cancel mid-scan unwinds
      cleanly, keeps partial results, and marks the `scan_runs` row `cancelled`
      > An in-flight `run()` is not interruptible — ONNX Runtime offers no such thing — so a
        cancelled scan finishes the batch it is holding and stops. Bounded by the batch size,
        which is the difference between cancellable and killable.
- [x] Resume: a re-run skips rows already `embedded`
      > This needed a change in the *walk* stage, not the embed stage:
        `SampleStatus::is_processed` counted `decoded` as finished, so a resumed scan would
        have fast-skipped exactly the rows that still owed a vector. It is now
        `is_complete(embedding_required)`, and a scan with no session — the state before the
        200 MB download finishes — is a first-class configuration rather than a failure.
- [x] Tune batch size and queue depths against the throughput target
      > Swept 1 → 64 and **the sweep cannot decide it**, because what batching amortizes is
        nearly free for a 130 KB fixture graph. The default stays at 16. What the sweep did
        find is that the front-end, not inference, was the pipeline: applying the mel bank
        densely spent 32.9M f64 multiply-adds per file summing zeros, and restricting it to
        each filter's nonzero span is a 7.5× end-to-end gain that is **bit-identical** on the
        parity surface. See `BENCHMARKS.md`.
- [ ] **Risk 5 evaluation (`overview.md` §10):** run a real one-shot drum library, not the
      demo corpus, and inspect nearest neighbors by hand. Do kicks retrieve kicks?
      > **Open, and blocked on the same export as Phase 3.** The instrument is built and
        committed: `dev_neighbors` ranks every stored vector against one sample by exact
        cosine over the mmap, brute force on purpose so that an index's recall does not sit
        between the question and the answer. Point the app at a drum library, scan, and ask
        it for a kick. There is nothing to inspect until the vectors mean something.
- [ ] If neighborhoods are poor: implement the duration-weighted CLAP + normalized-DSP
      blend before projection, behind a setting
      > Deliberately not built. It is conditional on an evaluation that cannot run, and a
        blend tuned against embeddings from a graph that computes nonsense would be a
        weighting chosen by coin flip. The DSP descriptors it needs are already in
        `sample_features` from Phase 2, which is the part that had to be done early.
- [ ] Soak test: full 50k scan, memory profiled throughout
      > Partially. `embed_memory_is_flat_as_the_library_grows` is the falsifiable half — 4×
        the files at 2.5× of 1.3 MiB, which is noise around a constant — and it runs the
        stage whose queue is *supposed* to be full. A 50k soak against a graph that is not
        the model would measure the harness; it is Phase 10's Instruments pass, with the real
        model, that closes this.

**Exit criteria:** 50,000 samples embedded end-to-end with throughput **≥ 60 samples/s**
recorded in the repo; peak RSS **< 800 MB**; cancellation is clean and resume works; a
hand-inspected neighbor check on real percussive samples is documented with a
keep-or-blend decision.

**Status: structurally complete, not met.** Every stage, every counter, every dedup and
resume path is built and tested end to end against a real ONNX Runtime — `tests/embed.rs` is
eleven integration tests over the five-stage pipeline, and the peak-RSS and cancellation
criteria are met. The throughput criterion and the Risk 5 decision are **blocked on Phase 3's
export**, and deliberately not faked: the number in `BENCHMARKS.md` is labelled as the
pipeline around a near-free model, which is an upper bound and not the claim the exit
criterion asks for.

**Two findings from this phase that change later ones:**

1. **A vector can be shared by more than one row, and Phase 10 should know it.** A duplicate
   file stores a *reference* to its twin's bytes rather than a second copy, so two rows can
   carry the same `emb_offset`. Nothing downstream is harmed — each row still gets its own
   coordinates — but `EmbeddingStore::compact` copies per sample, so compaction *un-shares*
   them and a compacted file can be larger than the one it replaced. On a library that is
   40% duplicates that is not a rounding error.
2. **`upsert_samples` now clears `emb_offset`/`emb_len`.** A row only reaches that statement
   because its file changed, so the vector those offsets point at describes audio the file no
   longer contains. Left in place it survives as a stale-but-plausible embedding that every
   duplicate of the *new* content would inherit. Phase 5's re-fit reads `all_embedding_locs`
   and must keep assuming that a row with offsets is a row whose offsets are current.

---

## Phase 5 — Projection

*Goal: stable 3D coordinates for the whole corpus.*

- [x] Define the `Projector` trait (`overview.md` §3.7)
      > It takes an `EmbeddingSet` — a borrowed view over the mmap plus the
        `(sample_id, location)` list — rather than an owned matrix, so §4.2's "never
        heap-load 102 MB" is a property every implementation gets for free instead of one
        each has to remember. What is deliberately *not* on the trait is persistence, the
        single-active invariant, the swap, and alignment: those are identical for every
        algorithm and live in `projection/refit.rs`.
- [x] `projection/pca.rs` first — `nalgebra` (0.35) truncated SVD. Fast, deterministic,
      and it unblocks Phase 7 with real coordinates immediately.
      > It is a truncated SVD, computed as the top-3 eigenvectors of the 512 × 512
        covariance — the same three vectors, reached the way that does not require a
        `DMatrix` of the whole corpus. `nalgebra` has no truncated SVD, and `SVD::new` over
        50,000 × 512 is exactly the heap load the previous bullet exists to avoid.
        **Eigenvector signs are canonicalized** (largest-magnitude component made positive),
        which is what makes two PCA fits over the same data byte-identical rather than
        mirrored, and what makes a PCA re-fit land almost on top of its predecessor before
        Procrustes is even asked.
- [x] Persist to `projection_runs` + `projections`; enforce the single-active invariant
      > Four writer commands, three of them `is_durable` so they get the writer's
        `BEGIN IMMEDIATE`…`COMMIT` to themselves. Activation also **prunes the runs it
        supersedes**, restricted to runs with a `completed_at`, so a concurrent shadow run
        cannot be deleted out from under the job building it.
- [x] Add `annembed` (**pin `0.1.6`**) and `hnsw_rs` (0.3). **Vendor `annembed` into the
      repo** — `overview.md` §10 risk 2.
      > `src-tauri/vendor/annembed/`, 416 KB, upstream's published source with **no change
        to any `.rs` file**. Only the manifest is edited, and only to remove things that
        cost build time and buy nothing: the binaries and examples (which Cargo never builds
        for a dependency, but whose `clap`/`bincode`/`byteorder`/`num_cpus` dependencies it
        did), the `python` feature, and the `cdylib`. The LAPACK backend is
        `macos-accelerate` — `lax` is a hard dependency of the crate and its symbols have to
        resolve against something, and on the only platform AudioCloud ships to that
        something is already installed.
- [x] `projection/umap.rs`: HNSW kNN graph → `annembed` → 3D; expose `n_neighbors`,
      `min_dist`, cosine metric
      > `n_neighbors` and cosine are real; **`min_dist` is a documented stand-in**.
        `annembed` does not implement Python UMAP's kernel — it derives its embedded scale
        from each point's local scale, modulated by `scale_rho`, and has no `min_dist` at
        all. The parameter is kept because it is the knob a user reaches for and mapped
        monotonically onto `scale_rho`, with the default (0.1) landing exactly on
        `annembed`'s own default (1.0). The numbers do not transfer from a Python notebook,
        and the `params_json` records the derived value alongside the asked-for one.
- [x] Read the embedding matrix via `mmap`, not a heap load
      > True of everything AudioCloud owns, and **not** true of the HNSW index: `hnsw_rs`
        stores the vectors it is given, so a 50k × 512 index is ~102 MB for the duration of
        a UMAP fit and there is no version of that which is not. What the mmap buys is
        everything around it — rows reach the index 1024 at a time and the block is dropped
        before the next is read, so peak is the index plus a chunk rather than the index
        plus a second copy of the corpus.
- [x] `projection/procrustes.rs`: SVD alignment — center, cross-covariance, `R = V Uᵀ`,
      **allow reflection**, uniform scale
      > No determinant correction, and `a_reflected_layout_is_recovered_rather_than_left_mirrored`
        is the test that fails if someone adds the textbook Kabsch one back.
- [x] Incremental placement path for small imports (< ~2% of corpus): kNN barycenter in
      existing 3D space, existing points never move
      > "Never move" is exact rather than approximate: rows for existing samples are not
        written at all, and the test asserts float equality rather than a tolerance.
        Neighbors are found by **exact brute force over the mmap, not by an HNSW index** —
        §3.8 says "via the existing HNSW index" and there is no such thing, because the
        index a re-fit builds is never persisted anywhere in the design. At 2% of 50,000 the
        exact answer costs about a second, once, on an import, and needs no index to keep in
        sync. Weights are similarities shifted into `[0, 2]`, because cosine runs to −1 and a
        negative weight pushes a point *out* of its neighbors' hull. Placements get a
        deterministic sub-pixel jitter keyed by `sample_id`, because Phase 4's dedup means
        two rows can share one vector exactly and would otherwise share one pixel forever.
- [x] Full re-fit as a cancellable background job at low priority, writing to a shadow run
      > Cancellable at every boundary, and **not** inside `Embedder::embed()`, which is one
        opaque call exactly like Phase 3's `Session::run()`. A cancelled or failed job
        deletes its shadow run and leaves the active map untouched, which is what makes
        minutes of work safe to abandon.
- [x] Atomic swap: `BEGIN IMMEDIATE`, flip `is_active`, commit — readers never see a
      half-swapped map
      > Clear, set, prune, in that order — the order is forced by the schema, since
        `idx_projection_active` is a partial unique index and setting the new flag first is a
        constraint violation rather than a momentary inconsistency. That it *fails* rather
        than *corrupts* is the reason the index is there.
        `a_concurrent_reader_never_sees_a_half_swapped_map` runs a reader through four
        consecutive swaps and asserts it only ever sees a whole map.
- [x] Test: re-fit with 5% new samples, verify median displacement of pre-existing points
      after Procrustes is small relative to the cloud's bounding box
      > Under both projectors. PCA is < 10% of the cloud diagonal; UMAP is **11–16%**
        against 24–87% unaligned. The alignment claim is measured on *one* fit rather than
        two — align a layout by hand and compare it to itself — because two stochastic
        descents differ by more than the alignment does.
- [x] Benchmark the 50k × 512 re-fit
      > See `BENCHMARKS.md`. Both projectors, plus what the incremental path costs on the
        same corpus, which is the ratio the whole §3.8 trade rests on.

**Exit criteria:** 3D coordinates exist for the full corpus under both projectors; UMAP
re-fit of 50k × 512 completes in **< 5 min**; a re-fit with 5% new data leaves existing
points substantially in place after alignment; the atomic swap holds under a concurrent
reader.

**Status: met.** `tests/projection.rs` is fourteen integration tests over a real database
and a real `embeddings.bin`, one per criterion plus the failure modes that make the criteria
safe.

**Three findings from this phase that change later ones:**

1. **`annembed` panics on a disconnected kNN graph, and the release profile aborts on
   panic.** Its initialization runs a diffusion map, and on a graph in separate components
   that decomposition degenerates into a constant-or-NaN initial embedding; `set_data_box`
   then divides by its own zero maximum and trips a bare `assert!`. A corpus of six tight,
   well-separated clumps reproduces it better than half the time, and a library of five
   hundred near-identical 909 kicks next to a folder of vocal loops is exactly that shape.
   Because `panic = "abort"` is set for release, `catch_unwind` is not available as a
   backstop — the defence has to be to never create the condition. `umap.rs` counts
   components by union-find before calling `embed()` and refuses the graph, and
   `Refit::with_fallback` turns the refusal into a PCA layout. **Phase 9's re-fit UI must
   surface that the fallback fired**, because a user who asked for UMAP and got PCA is owed
   the sentence. The named future improvement is to escalate `n_neighbors` and retry before
   giving up; it is not built because the effective value would then differ from the
   recorded one, and a `projection_runs` row that misreports its own parameters is worse
   than a plainer map.
2. **A projection run is recorded under the projector that actually produced it**, not the
   one that was asked for. This is why the fallback is a field on the job rather than a
   `Projector` that wraps two: a wrapper would have to answer `name()` before knowing which
   one ran, and a `'umap'` row over a PCA layout is a lie that outlives the session.
   Phase 6's `get_point_cloud` should pass `projection_runs.algorithm` through to the
   frontend for the same reason.
3. **Read the active run and its points in one statement.** The swap deletes the superseded
   run inside the same transaction that activates the new one, so a caller that issues two
   queries can read the old run id and then find nothing under it. `active_projection_points`
   does the join in one statement, which is atomic against the swap under WAL;
   Phase 6's `get_point_cloud` must call that rather than composing two reads.
4. **A 50k UMAP re-fit peaks at 2.95 GiB, and `overview.md` §7 has no row for it.** The
   table budgets peak RSS during a *scan* at 800 MB and says nothing about a projection, so
   this violates nothing as written — which is a gap in the table, not a pass. It is 3.7×
   the scan budget, on an app that ships to 8 GB Macs, in a job the user can start from a
   menu. Almost all of it is `hnsw_rs`'s copy of the vectors plus `annembed`'s Laplacian,
   SVD workspace and per-edge gradient state, none of which is under our control; PCA over
   the same corpus peaks at 95 MiB, which is what the mmap discipline buys where it applies.
   **Phase 10 should add the row and measure it on an 8 GB machine**, and weigh capping the
   corpus a single re-fit sees or holding the index in f16. See `BENCHMARKS.md`.

---

## Phase 6 — IPC Bridge

*Goal: the frontend can pull the real 50,000-point cloud as one ArrayBuffer.*

- [x] Implement the JSON command surface from `overview.md` §6.1
- [x] `AppError` (§6.7) with `thiserror` + tagged serde
- [x] `get_point_cloud` as `ipc::Response` raw bytes: magic + version + count + 16-byte
      aligned header, struct-of-arrays body
- [x] `get_feature_column` and `query_samples` on the same binary transport
- [x] `protocol/peaks.rs`: register the `abpeaks://` URI scheme; generate and cache
      waveform peak summaries; immutable cache headers
- [x] `Channel<ScanProgress>` and `Channel<RefitProgress>`, throttled to ≤ 10 Hz, with a
      guaranteed terminal event on complete / cancel / fail
- [x] `ts-rs` (12.x) derives on every IPC type; generated output to `src/bindings/`
- [x] CI check: fail the build if committed bindings differ from generated
- [x] `src/ipc/binary.ts`: ArrayBuffer decoders, alignment assertions, magic/version check
      with a clear error on mismatch
- [x] `capabilities/main.json` with the minimum permission set; confirm the `fs` plugin is
      **not** granted
- [x] Integration test: 50k point cloud round-trips and the payload is ≤ 900 KB

**Exit criteria:** the frontend fetches the real 50k point cloud in one call, payload
**≤ 900 KB**, decoded into typed arrays in **< 300 ms**; progress events arrive at ≤ 10 Hz
during a live scan; TypeScript types are generated and CI-enforced.

**Deviations, each deliberate:**

1. **`start_refit` returns a job id, not a `runId`.** `overview.md` §3.8 creates the shadow
   `projection_runs` row only once there are coordinates to write, which is minutes into a
   50,000-point UMAP fit; a command that returned the run id would have to block for the
   length of the job. The run id arrives in the terminal `RefitEvent`, which is where the
   frontend wants it anyway — it identifies the map now on screen, and before the swap there
   is no such map.
2. **`cancel_refit` and `cancel_download` are new commands.** §6.1 lists only `cancel_scan`.
   Cross-cutting rule 6 says every long operation is cancellable, and a minutes-long re-fit
   and a 200 MB transfer are both long operations. `set_root_enabled` is new for the same
   kind of reason: `library_roots.enabled` has been in the schema since Phase 1 with nothing
   able to set it.
3. **`AppError::InvalidArgument` is a variant §6.7 does not list.** A well-typed argument
   can still carry an unusable value — a blank tag, an empty collection name — and `NotFound`
   is about an id the database does not have. Collapsing them produces "Not found: a
   collection needs a name". The variant names the offending field so the UI highlights an
   input instead of raising a dialog. It is also what lets the command layer refuse a bad
   argument *before* the writer sees it, which matters because a durable command commits its
   transaction even when it fails: `set_tag` on an unknown sample id would otherwise leave
   behind a `tags` row the user never finished making.
4. **`AppError::Unavailable` is a variant §6.7 does not list.** `play_sample` and
   `stop_playback` are in §6.1 and their engine is Phase 8. Omitting them would desynchronize
   the type surface from §6.1 — which `task.md`'s own parallelism note relies on, since
   Phase 9's shell is meant to be buildable against these bindings — and returning `Internal`
   would tell the user something went wrong when nothing did. The variant should have no
   constructors left once Phase 8 lands.
5. **`abpeaks://` URLs carry a `?v=<updatedAt>` cache-buster.** §6.4's `immutable` header is
   right for "sample 1234 at revision N" and wrong for "sample 1234", because a rescanned
   file is different audio under the same id. The full URL is the cache key, so the query
   string is what makes the immutable promise true; `src/ipc/peaks.ts` is the only place it
   is built.
6. **The waveform summary covers the decoded window, not the file.** `decode` stops at
   `WINDOW_SECONDS`, so a four-minute loop summarizes to its first ten seconds. The `ABPK`
   header's reserved word carries `coveredMs` so the frontend can mark where the summary
   stops rather than drawing ten seconds edge to edge under a label reading "4:07".
7. **`get_similar` is exact brute force, not the Phase 5 HNSW index.** That index is built
   inside a re-fit, over a corpus snapshot, tuned for embedding quality rather than query
   latency, and does not outlive the job. One pass over 50,000 × 512 f16 is 51 MB of
   sequential page-cache reads; a second persistent index would put an approximate answer's
   recall between the question and the truth, and would be a cache with a coherence problem.
8. **`commands/dev.rs` is deleted.** It said Phase 6 would replace it, and it has.
   `dev_session_info` was the only command with no equivalent on the real surface; forcing
   the lazy session is now reachable through a scan, and `get_model_status` reports whether
   it is built without causing it.

**Measured:** payload 800,016 B against 900 KB; decode to typed arrays 0.001 ms against
300 ms (it is four typed-array views over the received buffer — no parse, no copy). See
[`BENCHMARKS.md`](BENCHMARKS.md).

---

## Phase 7 — 3D Visualization

*Goal: 60 fps orbit over 50,000 real points.*

- [x] **First task, before anything else:** query `ALIASED_POINT_SIZE_RANGE` in WKWebView
      and record the actual cap. If it is too low for the intended look, switch to the
      `InstancedMesh` quad path now — not in Phase 10. (`overview.md` §5.1)
      > **`[1, 511]` device pixels**, measured in a real WKWebView by
      > `scripts/probe_webgl_caps.mjs`. Eight times the 64 px floor §5.1 sets, so the
      > `THREE.Points` path stands and the quad path is not needed.
- [x] `<Canvas frameloop="demand" dpr={[1,2]} gl={{antialias:false}}>`
- [x] `scene/buffers.ts`: build one `BufferGeometry` from the ArrayBuffer; interleave the
      planar XYZ into the position attribute; `DynamicDrawUsage` on color and size only
- [x] `points.vert.glsl` / `points.frag.glsl`: distance-attenuated `gl_PointSize` clamped
      to the queried range, soft `smoothstep` circular sprite, depth fade, additive blend
      with `depthWrite: false`
- [x] Orbit / pan / zoom controls; `invalidate()` on interaction only
- [x] `scene/picking.ts`: ID-color pass to an offscreen target, 1-px `readPixels`;
      **on-demand only** — click always, hover throttled to ~20 Hz, never during a drag
      > A 21-px window rather than a literal 1 px. A point whose *centre* falls outside the
      > viewport is clipped before rasterization, so a 1×1 pick can only ever hit a point
      > centred on that one pixel however large its sprite is. See `scene/picking.ts`.
- [x] Hover highlight and selection ring via direct attribute mutation + `needsUpdate` —
      **no React state per point**
      > Two uniforms rather than attribute mutation. Same rule, less work: hover is one
      > point out of 50,000, and re-uploading a 200 KB attribute twenty times a second to
      > change one of them is what the rule was written against. See `scene/materials.ts`.
- [x] Color-by-feature: fetch the feature column as raw bytes, map to color in JS, write
      the typed array once
- [x] Filter dimming: filtered-out points shrink and desaturate rather than disappearing,
      so the shape of the corpus stays legible
- [x] Vertex-shader LOD; fragment discard below a size threshold
- [x] `webglcontextlost` handler that rebuilds from the cached ArrayBuffer without refetch
- [x] Frame-time profiling over a 30 s scripted orbit

**Exit criteria:** sustained **< 16.6 ms p99** frame time orbiting 50,000 points on Apple
Silicon; idle canvas renders **0 frames/s**; GPU pick returns the correct sample under a
dense cluster; context loss recovers without a refetch.

**Three of the four are met and measured** (`npm run profile`, recorded in
[`BENCHMARKS.md`](BENCHMARKS.md)): the idle canvas renders zero frames over three seconds,
the GPU pick agrees with an independent CPU model on 200 of 200 pixels each carrying a stack
of overlapping sprites, and a forced context loss recovers and draws again from the same
`ArrayBuffer` object it was holding before.

**The frame-time criterion is not met as stated, and the reason is the measurement rather
than the scene.** WebKit's `requestAnimationFrame` has a p99 of **23 ms over a blank page**
on this machine — no WebGL context, no scene, nothing to draw — against a 16 ms median. The
criterion's absolute figure is below the floor of the environment it is being measured in,
so it cannot be met there by any renderer. What the 50,000-point orbit measures is p50 =
16 ms, exactly the display interval, and a p99 statistically indistinguishable from the same
loop with the cloud hidden. Phase 10's Instruments pass is where a real GPU-side number
comes from; WebKit exposes no `EXT_disjoint_timer_query_webgl2`, so in-page there is nothing
better to be had.

### Findings

1. **The point-size cap was never the risk `overview.md` §5.1 thought it was.** The document
   warns the maximum `gl_PointSize` "can be as low as 64 px" under ANGLE-on-Metal and makes
   the whole `Points`-versus-`InstancedMesh` decision conditional on it. WKWebView on Apple
   GPU reports `[1, 511]`. The check still runs at startup in `scene/caps.ts` and the shader
   clamps to whatever it finds, because a cap measured on one Mac is not a cap promised on
   another — but the architecture question is closed.

2. **Measuring in WebKit needed its own tooling, and it earned its place.** `overview.md`
   §7 specifies "Chrome DevTools trace", and Chrome is not the engine this ships in.
   `scripts/webview_eval.swift` is a 200-line WKWebView harness that loads a page, evaluates
   an expression and prints the JSON it resolves to; both Phase 7 measurements run through
   it. Two things it turned up that a Chrome measurement would have hidden: the missing GPU
   timer extension, and the rAF floor above.

3. **`aId` is the cloud index, not the sample id.** A `float` attribute is exact to 2^24;
   sample ids are `i64` from SQLite and grow with every rescan of an edited library. Picking
   by id would have started returning a neighbouring sample somewhere past sixteen million
   rows, silently. Indexing also makes decoding a pick a subscript rather than a search.

4. **The pick check found a real bug, because it was written against an independent model.**
   `Picker` converted CSS coordinates to device pixels with `Math.round`, which sends the
   exact centre of pixel *n* to pixel *n + 1* — every pick landed one pixel off, which is
   invisible by eye in a cluster and wrong every single time. It surfaced as 181 of 200
   disagreements with the CPU reference. The first version of that reference was also wrong,
   in a more interesting way: it asked which point's *centre* was on the cursor's pixel, when
   what a pick returns — and should return — is the nearest point whose *sprite covers* it.

5. **Coordinates are not normalized on arrival, and must not be.** Phase 5 goes to real
   trouble to keep a layout stable across re-fits, and Procrustes preserves scale. Rescaling
   the cloud into a unit cube on load would mean that adding 5,000 samples moves every point
   on screen by two percent even though the alignment worked perfectly. The buffer keeps the
   numbers the core sent and the camera adapts; see `scene/framing.ts`.

---

## Phase 8 — Audio Preview Engine

*Goal: hover a point, hear it, with no glitches.*

- [x] Add `cpal` (0.18) and `ringbuf` (0.5)
- [x] `audio/engine.rs`: output stream on the default device; handle device change and
      sample-rate change without panicking
      > `cpal` 0.18's `ErrorKind::DeviceChanged` (the stream followed the new default device
      > on its own) and `ErrorKind::DeviceNotAvailable`/`StreamInvalidated` (it needs a new
      > one) are distinct, so the error callback only rebuilds for the second kind. A
      > rebuild bumps the shared generation and clears `want_playing`, which is what stops a
      > decode-ahead task mid-push on the *old* ring from writing frames sized for a channel
      > count the new one may not share.
- [x] `audio/ring.rs`: lock-free SPSC ring; the callback **only** copies and applies gain
- [x] **Debug-only guard allocator that panics if the audio thread allocates.** This is
      how the zero-allocation target becomes enforced rather than aspirational.
      > `audio/guard.rs`, installed in `main.rs` behind `#[cfg(debug_assertions)]`. The one
      > sharp edge: panicking *from inside* `GlobalAlloc::alloc` boxes the panic payload,
      > which is itself an allocation, which would re-enter the same check and abort the
      > process with "thread panicked while processing panic" instead of a catchable panic.
      > `trip()` clears the thread-local flag before calling `panic!`, so only the violation
      > itself is checked and the unwind machinery's own bookkeeping is not. `tests/
      > audio_guard.rs` is a separate integration test binary — a global allocator can only
      > be installed once per binary — that proves the panic actually fires.
- [x] Decode-ahead task on `tokio` fills the ring; small LRU cache of recently decoded
      samples
      > `audio::PcmCache`, 24 entries, keyed by `(sample_id, format_epoch)` so a device
      > rebuild invalidates every cached buffer rather than replaying a rate that no longer
      > matches the stream.
- [x] Hover-to-audition with a ~120 ms debounce so a fast cursor sweep does not machine-gun
      the decoder
      > In the frontend (`scene/PointCloud.tsx`), not the core: `overview.md` §6.1 fixes
      > `play_sample` at `(sampleId, gain)`, with nothing on the wire to distinguish a hover
      > from a click, so there is no server-side hover concept to debounce. Every call is
      > safe to make as fast as the frontend likes regardless — see the next line.
- [x] Click-to-play with a short attack/release envelope — retriggering must not click
      > Retriggering is a generation counter, not a buffer clear reaching across threads:
      > `Engine::play_pcm` bumps `Transport::generation`; the real-time callback notices the
      > change on its next call and clears its ring consumer itself (it is that half's sole
      > owner), starting a fresh 5 ms attack; a decode-ahead task mid-push notices the same
      > mismatch and stops. Nobody locks and nobody reaches across the SPSC boundary.
- [x] Master gain; optional loop for one-shots
      > Master gain: clamped to `[0.0, 2.0]` in `Engine::play_pcm`, applied in the callback
      > alongside the envelope. **Loop is not built.** `overview.md` §6.1's `play_sample`
      > takes no loop flag and there is no settings surface yet to host a persistent
      > "loop one-shots" toggle — that is Phase 9's. Adding a third command-line argument to
      > a command whose bindings Phase 6 already froze was judged worse than leaving one
      > sub-bullet open with the reason on record.
- [x] `audio/peaks.rs`: waveform peak generation, served over `abpeaks://` (Phase 6)
      > Already built in Phase 6, alongside the `abpeaks://` scheme itself — see that
      > phase's notes. Unchanged here.
- [ ] Waveform display in the inspector with a playhead
      > **Not built. There is no inspector.** `panels/` — the whole of Phase 9's Application
      > Shell — does not exist yet; `src/App.tsx` is still the Phase 7 placeholder shell.
      > A playhead needs a panel to draw it in, so this is Phase 9's to finish, using
      > `src/ipc/peaks.ts` (already wired) for the waveform and `play_sample`'s generation
      > id — exposed nowhere on the wire today — for playhead position, which is a real gap
      > Phase 9 will need to close, most likely with a `Channel<PlaybackProgress>` alongside
      > the scan and re-fit progress channels.
- [x] Latency measurement: pointer event timestamp → first non-zero sample in the callback
      > **Measures command-received → first audible sample, not pointer-event → audible.**
      > The command surface Phase 6 froze carries no pointer timestamp, and the frontend's
      > hover debounce and the IPC round trip both happen before `AudioPlayer::play` is ever
      > called — neither is observable from the Rust side of the boundary. What is measured
      > is the half of the budget this module owns: `Transport::started_at_nanos` (written
      > by `play_pcm`) to `Transport::first_audible_nanos` (written once, by the real-time
      > callback, on the first frame it actually pops real data for). Logged per playback via
      > `tracing`; `tests/audio_hardware.rs` (real hardware, `--ignored` by default — see
      > below) reads it directly and asserts a generous bound. See `BENCHMARKS.md`.

**Exit criteria:** hover-to-audible **< 50 ms**; zero allocations on the audio thread under
the guard allocator; no audible glitch when retriggering rapidly or when the output device
changes mid-playback.

**Status: met for what a Rust test can observe; the full pointer-to-speaker chain needs
Phase 9's UI to measure.** `play_pcm` → first audible sample is **~16–17 ms** against real
hardware (`BENCHMARKS.md`), comfortably inside the 50 ms budget with room for the IPC hop and
the frontend's own debounce that this measurement cannot see. Zero allocations is enforced,
not merely measured — `tests/audio_guard.rs` proves the guard allocator's panic actually
fires from a real allocation inside a real guard, on its own dedicated global allocator.
Retrigger safety is proven twice: `audio::engine`'s unit tests exercise the envelope's state
machine (`Attack`/`Sustain`/`Release`/`Idle`) against a plain `HeapRb` with no `cpal` stream
anywhere near it, and `tests/audio_hardware.rs`'s `rapid_retriggers_do_not_panic_or_deadlock`
repeats the same property against a real stream and real OS scheduling. **Device-change
handling is implemented and unit-testable in isolation (`ErrorKind` routing, generation bump
on rebuild) but not exercised end-to-end** — that needs an audio interface physically
unplugged mid-playback, which is a manual verification this environment cannot automate; it
is a reasonable Phase 10 hardening-pass item alongside the soak tests already assigned there.

**Two things this phase leaves for Phase 9, both noted above and worth repeating in one
place:** the waveform-with-playhead display has no panel to live in yet, and a playhead needs
a way to know playback position that the current wire format does not carry. Both are Phase
9 UI work, not Phase 8 engine work, and neither blocks anything downstream — `play_sample`
and `stop_playback` are fully functional today, driven from the point cloud's own hover and
click handling as a stand-in for the inspector that does not exist yet.

---

## Phase 9 — Application Shell & UX

*Goal: the parts around the canvas that make it a tool rather than a demo.*

- [x] Tailwind layout: canvas center, collapsible inspector right, library/filters left
- [x] First-run flow: welcome → add library root → model download with progress → scan
- [x] Library management: add / remove / rescan / enable-disable roots; show per-root counts
- [x] Scan progress UI driven by the throttled `Channel`; cancel button that actually
      cancels
- [x] Search over FTS5 with debounced input; results both listed and highlighted in 3D
- [x] Filter panel: duration, BPM, key, loudness, spectral centroid, tags, format
- [x] Inspector: filename, path, format, duration, DSP features, waveform, tags,
      nearest neighbors (`get_similar`), reveal-in-Finder
- [x] Tagging: create, assign, bulk-assign to a selection, color per tag
- [x] Collections: create from selection, reorder, export the file list
- [x] Settings: projection params + re-fit trigger, audio device and gain, model status and
      re-download, data dir with reveal, reset-database
- [x] Empty, loading, and error states for every panel — including "model not downloaded,"
      "no roots added," "scan failed," and "0 results"
- [x] Keyboard: space to audition, arrows to step neighbors, `/` to focus search, escape to
      clear selection
- [x] Drag a sample out of the app into a DAW from the map, search, scrub history,
      collections, the inspector, or similar-sound results. The native drag session is owned
      server-side and resolves the file from a sample id, so its path never crosses from an
      untrusted WebView argument. Hold Command to switch a map drag into pan mode.

**Exit criteria:** a new user can go from first launch to auditioning a sample from their
own library without reading documentation; every `AppError` variant has a rendered state
with a recovery action; no panel can reach a blank screen with no explanation.

---

## Phase 10 — Hardening & Performance

*Goal: make the numbers in `overview.md` §7 true and keep them true.*

- [ ] Instruments pass: allocations, leaks, time profiler on a full 50k scan
- [ ] Verify every row of the §7 budget table; record actuals in the repo next to targets
- [ ] Soak test: 250,000-sample library. Document where it degrades — it will — and whether
      the degradation is the DB, the projection, or the GPU
- [ ] Fuzz the decode path with truncated, zero-byte, and deliberately malformed audio
- [ ] Test unplugging an external drive mid-scan; test a root that disappears between
      scans; verify `status = 'missing'` and that tags survive
- [ ] Concurrency: scan while orbiting while auditioning while re-fitting — the whole
      point of the thread model in `overview.md` §2 is that this is boring
- [ ] `tracing` structured logging to a rotating file; log level in settings; a
      "reveal logs" action
- [ ] Crash reporting with symbolication
- [ ] If the picking `readPixels` stall shows in a trace: move to a WebGL2
      `PIXEL_PACK_BUFFER` async read (`overview.md` §5.3)
- [ ] `embeddings.bin` compaction path exercised after bulk deletion
- [ ] Database backup before every migration; verify restore

**Exit criteria:** every §7 target is met or has a documented, accepted variance; a 30 min
soak with concurrent scan + orbit + audition shows no memory growth and no glitch; no
crash across the malformed-input fuzz corpus.

---

## Phase 11 — Native Build & Distribution

*Goal: a signed, notarized universal `.dmg` that launches on a machine that has never seen
a developer tool.*

- [ ] **Resolve the `ort` universal-binary problem** (`overview.md` §8.1). Its own task
      because it is the known sharp edge:
  - [ ] Try approach 1 — build `aarch64-apple-darwin` and `x86_64-apple-darwin`
        separately, `lipo -create` the executables and the ONNX Runtime dylibs, assemble
        the `.app` around the fat products
  - [ ] If that is unworkable, try approach 2 — pre-`lipo` a fat ONNX Runtime dylib and
        set `ORT_LIB_LOCATION`
  - [ ] `lipo -info` on **every** binary and dylib in the finished bundle
  - [ ] Document whichever approach won, in the repo, with the exact commands
- [ ] `entitlements.plist`: hardened runtime, `allow-jit`,
      `disable-library-validation`; App Sandbox off
- [ ] Sign **inner to outer** — every nested dylib and framework first, with
      `--options runtime --timestamp`, then the bundle
- [ ] `xcrun notarytool submit --wait`; resolve every rejection (they are usually an
      unsigned nested binary)
- [ ] `xcrun stapler staple` the `.dmg` so first launch works offline
- [ ] `.dmg` bundling with a background image and an Applications symlink
- [ ] `tauri-plugin-updater`: generate the keypair, store the private key as a CI secret,
      publish `latest.json`, test an actual update from the previous version
- [ ] Version the model independently of the app, so a model revision is a download prompt
      rather than a full app update
- [ ] CI: tag → build both arches → merge → sign → notarize → staple → publish
- [ ] **Clean-machine install test on real hardware of both architectures.** Rosetta on an
      Apple Silicon Mac does not prove the Intel slice works.
- [ ] `spctl -a -vvv -t install` passes on a machine that has never run Xcode

**Exit criteria:** a notarized universal `.dmg` installs and launches on a clean Apple
Silicon Mac **and** a clean Intel Mac; `spctl` passes on both; an in-app update from the
prior version succeeds; `lipo -info` shows both slices for every binary in the bundle.

---

## Cross-cutting rules

These hold in every phase.

1. **No blocking work on the main thread.** Ever. It freezes the window.
2. **One SQLite writer.** If a second write connection appears anywhere, that is a bug,
   not an optimization.
3. **No per-file IPC events.** Coalesce to ≤ 10 Hz. (`overview.md` §6.5)
4. **No per-point React state.** Mutate typed arrays and set `needsUpdate`.
5. **No allocation on the audio thread.** Enforced by the guard allocator in debug builds.
6. **Every long operation is cancellable** and reports progress on the same throttle.
7. **Pin exact versions** for `ort` and `annembed`; commit `Cargo.lock`; vendor `annembed`.
8. **Errors are typed and recoverable.** A new failure mode means a new `AppError` variant
   and a new rendered state, not a `.unwrap()` and not a string.
9. **Benchmarks are committed with their results.** A performance target with no recorded
   measurement is a wish.
