# AudioBank — Implementation Roadmap

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

- [ ] Install Rust stable; add both targets:
      `rustup target add aarch64-apple-darwin x86_64-apple-darwin`
- [ ] Install Xcode Command Line Tools; verify `lipo`, `codesign`, `xcrun notarytool`
- [ ] Install Node LTS (build-time only — see `overview.md` §1); pin in `.nvmrc`
- [ ] Install the Tauri v2 CLI (`cargo install tauri-cli --version "^2"`)
- [ ] Scaffold: `cargo create-tauri-app` → React + TypeScript + Vite
- [ ] Add Tailwind, configure `content` globs for `src/**/*.{ts,tsx}`
- [ ] Add `three`, `@react-three/fiber`, `@react-three/drei`, `zustand`
- [ ] Establish `src-tauri/` module skeleton per `overview.md` §9 — empty modules with
      `//!` doc comments stating each one's responsibility
- [ ] `.gitignore`: `target/`, `node_modules/`, `dist/`, `*.onnx`, `*.dmg`,
      `src-tauri/gen/`, and the local app-data dir if it is ever symlinked in
- [ ] `rustfmt.toml`, `clippy.toml`; ESLint + Prettier for the frontend
- [ ] `cargo tauri dev` launches a window
- [ ] `cargo tauri build --target aarch64-apple-darwin` and
      `--target x86_64-apple-darwin` both succeed (unsigned is fine here)
- [ ] CI skeleton: fmt, clippy, `cargo test`, `tsc --noEmit`, both target builds

**Exit criteria:** an empty window renders via `cargo tauri dev`; both single-arch release
builds succeed in CI; `cargo clippy -- -D warnings` is clean.

---

## Phase 1 — Data Layer First

*Goal: somewhere for ingest to land, built before the thing that produces the data.*

Building this first is deliberate. A pipeline written before its sink grows an ad-hoc
in-memory store that has to be torn out later.

- [ ] Add `rusqlite` (0.40, `bundled` feature), `r2d2`, `r2d2_sqlite` (0.35),
      `refinery` (0.9)
- [ ] Resolve and create the app data dir (`~/Library/Application Support/<bundle-id>/`);
      create on first run with correct permissions
- [ ] Write `V1__initial.sql` — the full DDL from `overview.md` §4.1, including the FTS5
      virtual table and the partial unique index on `projection_runs(is_active)`
- [ ] Wire `refinery` to run migrations at startup, before any other connection opens
- [ ] Pragma application (`overview.md` §4.3) via an `r2d2` connection customizer, so
      **every** pooled connection gets `foreign_keys = ON` — it is per-connection
- [ ] Implement the single writer thread: `mpsc` command enum, oneshot replies, batched
      transactions flushing at 1000 rows **or** 250 ms, whichever first
- [ ] Implement the read pool (4 read-only connections)
- [ ] Implement `db/embeddings.rs`: append-only `embeddings.bin`, f16 storage via `half`,
      `(offset, len)` return, `memmap2` whole-matrix read, `compact()` maintenance path
- [ ] Unit tests: migration idempotency, FK enforcement actually rejects an orphan,
      writer batching commits on both the count and the timeout trigger
- [ ] Benchmark: insert 50,000 synthetic sample rows + features
- [ ] Benchmark: append 50,000 × 512 f16 embeddings, mmap them back, verify round-trip
      values and that resident memory does not grow by 51 MB on read

**Exit criteria:** 50,000 synthetic rows insert in **< 2 s**; the embedding file
round-trips bit-exactly; `PRAGMA foreign_key_check` is clean; migrations run twice with no
effect the second time.

---

## Phase 2 — Discovery & Decode

*Goal: point the app at a real sample folder and get real rows, with no ML involved.*

- [ ] Add `ignore` (0.4), `blake3` (1.8), `symphonia` (0.6, all relevant codec features),
      `rubato` (5.0), `rayon` (1.12), `realfft` (3.5), `ebur128` (0.1)
- [ ] `pipeline/walk.rs`: `ignore::WalkBuilder::build_parallel()`, extension allowlist,
      symlink-loop guard, size sanity bounds
      > Note: **not `jwalk`** — deprecated upstream ("Use `dua-core` instead").
        See `overview.md` §3.1.
- [ ] `(path, mtime, size)` fast-skip against existing rows before any hashing
- [ ] `blake3` content hashing, with head+tail+length sampling for files > 64 MB
- [ ] Dedup: link a matching `content_hash` to the existing embedding rather than
      re-processing
- [ ] `pipeline/decode.rs`: `symphonia` probe + streaming decode, **hard stop at 10 s of
      48 kHz output**, mono downmix during decode, `rubato` resample
- [ ] Buffer pool for decode output — no per-sample allocation of the 1.92 MB buffer
- [ ] `pipeline/features.rs`: `realfft` frames, plus peak / RMS / LUFS / centroid /
      flatness / ZCR / onset density / BPM / key from the same pass
- [ ] Decode-failure quarantine: `status = 'decode_failed'` + error string, scan continues
- [ ] Wire the bounded channels between walk → decode → features → writer, with the queue
      depths from `overview.md` §3
- [ ] Temporary dev-only command to trigger a scan and dump counts (real IPC is Phase 6)
- [ ] Benchmark on a real library of ≥ 5,000 files across mixed formats

**Exit criteria:** a real folder scans to the database with correct metadata and DSP
features; corrupt and unsupported files are quarantined without aborting the scan; a
re-scan of an unchanged folder completes in **< 5%** of the original time; decode+DSP
throughput **≥ 400 samples/s**; peak RSS during scan stays flat as the file count grows.

---

## Phase 3 — Model Provisioning

*Goal: a verified CLAP ONNX model on disk and a warm `ort` session — with proof the mel
front-end matches the graph.*

- [ ] Write `scripts/export_clap_onnx.py` — **one-time, offline, developer-only.** Loads
      LAION CLAP, exports the **audio tower only** (the text tower is dead weight), fixed
      or dynamic batch axis, opset pinned. Document exact package versions used.
- [ ] Run the export once. Record the SHA-256. Publish the `.onnx` as a release asset.
      Do not commit a 200 MB binary to git.
- [ ] **Record reference outputs**: run 20 diverse fixture wavs through Python CLAP and
      commit the resulting embeddings as a test fixture. This is the parity oracle.
- [ ] Add `reqwest` (0.13, streaming), `sha2` (0.11), `ort` (**pinned exactly**
      `2.0.0-rc.13`)
- [ ] `model/download.rs`: resumable streaming download with `Range`, incremental SHA-256
      over the stream, atomic rename from `.partial` only after verification, `fsync`
      before rename
- [ ] Distinct handling for: no network / hash mismatch / disk full / interrupted-resumable
- [ ] `model/session.rs`: session init with CoreML EP, **verified** CPU fallback; log which
      EP actually bound; warmup run on a zero tensor to pay first-inference cost off the
      user's critical path
- [ ] Lazy init — session construction must not be on the cold-start path
      (`overview.md` §7)
- [ ] **Parity gate:** the mel front-end from Phase 2 feeding this session must reproduce
      the committed reference embeddings at **cosine similarity > 0.999**. Commit this as
      a regression test, not a one-time check.
- [ ] Isolate every `ort` type behind `model/session.rs` so an rc upgrade touches one file

**Exit criteria:** model downloads, verifies, survives a kill -9 mid-download and resumes;
session initializes on both CoreML and forced-CPU; **the parity test passes at > 0.999**.

> If parity fails, stop. Do not proceed to Phase 4. Every embedding produced by a
> mismatched front-end is silently wrong and the map will look completely plausible.

---

## Phase 4 — Embedding Pipeline

*Goal: 50,000 samples embedded end-to-end, with a recorded throughput number.*

- [ ] `pipeline/embed.rs`: batching accumulator (start at 16, tune) with a flush timeout so
      the tail of a scan does not stall on an unfillable batch
- [ ] `Arc<Session>` shared across `rayon` workers — one session, never one per thread
- [ ] L2-normalize outputs immediately on receipt
- [ ] Wire the full five-stage pipeline: walk → decode → mel → embed → persist
- [ ] `pipeline/progress.rs`: atomic counters per stage + a single 100 ms ticker
      (`overview.md` §6.5). **No per-file events.**
- [ ] `CancellationToken` threaded through every stage; verify a cancel mid-scan unwinds
      cleanly, keeps partial results, and marks the `scan_runs` row `cancelled`
- [ ] Resume: a re-run skips rows already `embedded`
- [ ] Tune batch size and queue depths against the throughput target
- [ ] **Risk 5 evaluation (`overview.md` §10):** run a real one-shot drum library, not the
      demo corpus, and inspect nearest neighbors by hand. Do kicks retrieve kicks?
- [ ] If neighborhoods are poor: implement the duration-weighted CLAP + normalized-DSP
      blend before projection, behind a setting
- [ ] Soak test: full 50k scan, memory profiled throughout

**Exit criteria:** 50,000 samples embedded end-to-end with throughput **≥ 60 samples/s**
recorded in the repo; peak RSS **< 800 MB**; cancellation is clean and resume works; a
hand-inspected neighbor check on real percussive samples is documented with a
keep-or-blend decision.

---

## Phase 5 — Projection

*Goal: stable 3D coordinates for the whole corpus.*

- [ ] Define the `Projector` trait (`overview.md` §3.7)
- [ ] `projection/pca.rs` first — `nalgebra` (0.35) truncated SVD. Fast, deterministic,
      and it unblocks Phase 7 with real coordinates immediately.
- [ ] Persist to `projection_runs` + `projections`; enforce the single-active invariant
- [ ] Add `annembed` (**pin `0.1.6`**) and `hnsw_rs` (0.3). **Vendor `annembed` into the
      repo** — `overview.md` §10 risk 2.
- [ ] `projection/umap.rs`: HNSW kNN graph → `annembed` → 3D; expose `n_neighbors`,
      `min_dist`, cosine metric
- [ ] Read the embedding matrix via `mmap`, not a heap load
- [ ] `projection/procrustes.rs`: SVD alignment — center, cross-covariance, `R = V Uᵀ`,
      **allow reflection**, uniform scale
- [ ] Incremental placement path for small imports (< ~2% of corpus): kNN barycenter in
      existing 3D space, existing points never move
- [ ] Full re-fit as a cancellable background job at low priority, writing to a shadow run
- [ ] Atomic swap: `BEGIN IMMEDIATE`, flip `is_active`, commit — readers never see a
      half-swapped map
- [ ] Test: re-fit with 5% new samples, verify median displacement of pre-existing points
      after Procrustes is small relative to the cloud's bounding box
- [ ] Benchmark the 50k × 512 re-fit

**Exit criteria:** 3D coordinates exist for the full corpus under both projectors; UMAP
re-fit of 50k × 512 completes in **< 5 min**; a re-fit with 5% new data leaves existing
points substantially in place after alignment; the atomic swap holds under a concurrent
reader.

---

## Phase 6 — IPC Bridge

*Goal: the frontend can pull the real 50,000-point cloud as one ArrayBuffer.*

- [ ] Implement the JSON command surface from `overview.md` §6.1
- [ ] `AppError` (§6.7) with `thiserror` + tagged serde
- [ ] `get_point_cloud` as `ipc::Response` raw bytes: magic + version + count + 16-byte
      aligned header, struct-of-arrays body
- [ ] `get_feature_column` and `query_samples` on the same binary transport
- [ ] `protocol/peaks.rs`: register the `abpeaks://` URI scheme; generate and cache
      waveform peak summaries; immutable cache headers
- [ ] `Channel<ScanProgress>` and `Channel<RefitProgress>`, throttled to ≤ 10 Hz, with a
      guaranteed terminal event on complete / cancel / fail
- [ ] `ts-rs` (12.x) derives on every IPC type; generated output to `src/bindings/`
- [ ] CI check: fail the build if committed bindings differ from generated
- [ ] `src/ipc/binary.ts`: ArrayBuffer decoders, alignment assertions, magic/version check
      with a clear error on mismatch
- [ ] `capabilities/main.json` with the minimum permission set; confirm the `fs` plugin is
      **not** granted
- [ ] Integration test: 50k point cloud round-trips and the payload is ≤ 900 KB

**Exit criteria:** the frontend fetches the real 50k point cloud in one call, payload
**≤ 900 KB**, decoded into typed arrays in **< 300 ms**; progress events arrive at ≤ 10 Hz
during a live scan; TypeScript types are generated and CI-enforced.

---

## Phase 7 — 3D Visualization

*Goal: 60 fps orbit over 50,000 real points.*

- [ ] **First task, before anything else:** query `ALIASED_POINT_SIZE_RANGE` in WKWebView
      and record the actual cap. If it is too low for the intended look, switch to the
      `InstancedMesh` quad path now — not in Phase 10. (`overview.md` §5.1)
- [ ] `<Canvas frameloop="demand" dpr={[1,2]} gl={{antialias:false}}>`
- [ ] `scene/buffers.ts`: build one `BufferGeometry` from the ArrayBuffer; interleave the
      planar XYZ into the position attribute; `DynamicDrawUsage` on color and size only
- [ ] `points.vert.glsl` / `points.frag.glsl`: distance-attenuated `gl_PointSize` clamped
      to the queried range, soft `smoothstep` circular sprite, depth fade, additive blend
      with `depthWrite: false`
- [ ] Orbit / pan / zoom controls; `invalidate()` on interaction only
- [ ] `scene/picking.ts`: ID-color pass to an offscreen target, 1-px `readPixels`;
      **on-demand only** — click always, hover throttled to ~20 Hz, never during a drag
- [ ] Hover highlight and selection ring via direct attribute mutation + `needsUpdate` —
      **no React state per point**
- [ ] Color-by-feature: fetch the feature column as raw bytes, map to color in JS, write
      the typed array once
- [ ] Filter dimming: filtered-out points shrink and desaturate rather than disappearing,
      so the shape of the corpus stays legible
- [ ] Vertex-shader LOD; fragment discard below a size threshold
- [ ] `webglcontextlost` handler that rebuilds from the cached ArrayBuffer without refetch
- [ ] Frame-time profiling over a 30 s scripted orbit

**Exit criteria:** sustained **< 16.6 ms p99** frame time orbiting 50,000 points on Apple
Silicon; idle canvas renders **0 frames/s**; GPU pick returns the correct sample under a
dense cluster; context loss recovers without a refetch.

---

## Phase 8 — Audio Preview Engine

*Goal: hover a point, hear it, with no glitches.*

- [ ] Add `cpal` (0.18) and `ringbuf` (0.5)
- [ ] `audio/engine.rs`: output stream on the default device; handle device change and
      sample-rate change without panicking
- [ ] `audio/ring.rs`: lock-free SPSC ring; the callback **only** copies and applies gain
- [ ] **Debug-only guard allocator that panics if the audio thread allocates.** This is
      how the zero-allocation target becomes enforced rather than aspirational.
- [ ] Decode-ahead task on `tokio` fills the ring; small LRU cache of recently decoded
      samples
- [ ] Hover-to-audition with a ~120 ms debounce so a fast cursor sweep does not machine-gun
      the decoder
- [ ] Click-to-play with a short attack/release envelope — retriggering must not click
- [ ] Master gain; optional loop for one-shots
- [ ] `audio/peaks.rs`: waveform peak generation, served over `abpeaks://` (Phase 6)
- [ ] Waveform display in the inspector with a playhead
- [ ] Latency measurement: pointer event timestamp → first non-zero sample in the callback

**Exit criteria:** hover-to-audible **< 50 ms**; zero allocations on the audio thread under
the guard allocator; no audible glitch when retriggering rapidly or when the output device
changes mid-playback.

---

## Phase 9 — Application Shell & UX

*Goal: the parts around the canvas that make it a tool rather than a demo.*

- [ ] Tailwind layout: canvas center, collapsible inspector right, library/filters left
- [ ] First-run flow: welcome → add library root → model download with progress → scan
- [ ] Library management: add / remove / rescan / enable-disable roots; show per-root counts
- [ ] Scan progress UI driven by the throttled `Channel`; cancel button that actually
      cancels
- [ ] Search over FTS5 with debounced input; results both listed and highlighted in 3D
- [ ] Filter panel: duration, BPM, key, loudness, spectral centroid, tags, format
- [ ] Inspector: filename, path, format, duration, DSP features, waveform, tags,
      nearest neighbors (`get_similar`), reveal-in-Finder
- [ ] Tagging: create, assign, bulk-assign to a selection, color per tag
- [ ] Collections: create from selection, reorder, export the file list
- [ ] Settings: projection params + re-fit trigger, audio device and gain, model status and
      re-download, data dir with reveal, reset-database
- [ ] Empty, loading, and error states for every panel — including "model not downloaded,"
      "no roots added," "scan failed," and "0 results"
- [ ] Keyboard: space to audition, arrows to step neighbors, `/` to focus search, escape to
      clear selection
- [ ] Drag a sample out of the app into a DAW (macOS file promise)

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
