# AudioBank

A spatial browser for large sample libraries. Samples are embedded with CLAP, projected to
3D, and rendered as a point cloud you can orbit and audition.

- [`overview.md`](overview.md) — the architecture.
- [`task.md`](task.md) — the implementation roadmap, phase by phase.

**Status: Phase 7 implemented, Phase 3's gate not yet passed.** The window now draws the
map, and behind it the ingest pipeline is real and complete: point it at a folder of
audio and it walks, hashes, deduplicates, decodes, analyzes, computes CLAP's log-mel
spectrogram, embeds it in batches through one shared `ort` session, and writes both the DSP
features and the vector — with progress coalesced to 10 Hz, cancellation that keeps what it
wrote, and a resume that finishes what an interrupted scan started.

Phase 5 turns those vectors into a map. Two projectors behind one trait — a deterministic
PCA and UMAP over a vendored `annembed` — reduce 512 dimensions to 3, and the layout is kept
_stable_ across re-fits: Procrustes alignment (reflection allowed) onto the previous map, a
shadow run swapped in atomically so a reader never sees half a map, and an incremental path
that places a small import without moving one existing point. A 50,000 × 512 UMAP re-fit
takes 58 s against a five-minute budget. See [Phase 5 status](#phase-5-status).

Phase 6 opens that up to the frontend. The full command surface of `overview.md` §6.1 is
implemented and reachable, errors cross the boundary as a typed discriminated union rather
than as strings, and the payloads that are the size of the library take the binary
transport: the 50,000-point cloud is **one 800 KB `ArrayBuffer`** that becomes four typed
arrays with no parse and no copy. TypeScript types are generated from the Rust structs and
CI-enforced. See [Phase 6 status](#phase-6-status).

Phase 7 draws it. 50,000 points are one `BufferGeometry` and **one draw call**, shaded by a
pair of hand-written GLSL programs; hovering and selecting a point mutate two uniforms rather
than any React state, and an idle canvas renders exactly zero frames. Picking is a GPU pass
into a 21-pixel window, and it agrees with an independently written CPU model on 200 out of
200 crowded pixels. The first thing the phase did was ask WKWebView what its maximum
`gl_PointSize` actually is — `[1, 511]`, eight times the floor at which the whole layer would
have had to be rebuilt as textured quads. See [Phase 7 status](#phase-7-status).

What is **not** done is the one thing Phase 3 is actually for. Exporting the ONNX file and
recording the reference embeddings requires running LAION-CLAP under PyTorch, which is a
one-time offline step on a machine with that toolchain; it has not been run, so
`ModelRelease::CURRENT` carries the `UNPINNED` sentinel, no `.onnx` exists to download, and
the parity gate skips. See [Phase 3 status](#phase-3-status) for exactly what remains and
what stops it from being forgotten.

That blocks two of Phase 4's claims and no others. The pipeline is tested end to end against
a real ONNX Runtime using the synthetic fixture graphs, so the plumbing is proved; what
cannot be proved is that the numbers coming out of it _mean_ anything. The
`≥ 60 samples/s` throughput criterion and the risk-5 question — does CLAP say anything useful
about a 200 ms hi-hat — both wait on the export. See [Phase 4 status](#phase-4-status).

Measured numbers live in [`BENCHMARKS.md`](BENCHMARKS.md).

**Target: Apple Silicon only.** `x86_64-apple-darwin` built clean during Phase 0 but is not
carried forward -- it is out of the CI matrix and out of Phase 11's distribution work, which
means no universal binary and no `lipo` merge step. Re-add the target to
[`ci.yml`](.github/workflows/ci.yml) if Intel support returns.

## Prerequisites

| Tool                     | Version                | Notes                                      |
| ------------------------ | ---------------------- | ------------------------------------------ |
| Rust                     | stable                 | both `*-apple-darwin` targets installed    |
| Node                     | see [`.nvmrc`](.nvmrc) | **build-time only** — see `overview.md` §1 |
| Xcode Command Line Tools | any                    | provides `lipo`, `codesign`, `notarytool`  |
| Tauri CLI                | v2                     | `cargo install tauri-cli --version "^2"`   |

```sh
rustup target add aarch64-apple-darwin
cargo install tauri-cli --version "^2"
npm ci
```

Node is a build dependency, not a runtime one. The shipped `.app` contains a Rust binary
and a bundle of static assets; there is no Node process at runtime.

## Development

```sh
cargo tauri dev          # vite + the Rust shell, hot-reloading both
npm run typecheck        # tsc --noEmit over src/ and the config files
npm run lint             # eslint
npm run format           # prettier

npm run probe:webgl      # what WKWebView's WebGL will actually do (Phase 7)
npm run profile          # the 30 s scripted orbit and Phase 7's exit criteria
```

The last two compile [`scripts/webview_eval.swift`](scripts/webview_eval.swift) on demand,
so they need the Xcode Command Line Tools and a **visible display** — `requestAnimationFrame`
does not run in a window macOS considers occluded, and a frame time measured in one would be
a fiction. `npm run profile` exits non-zero if an exit criterion fails.

The app data directory — the SQLite database, its WAL, and `embeddings.bin` — is
`~/Library/Application Support/com.audiobank.app/`, created 0700 on first run. Deleting it
is a supported reset: every row in it is derived from files on disk, apart from library
roots, tags, and collections.

### Driving it

The map renders, but there is nothing to drive it with yet — Phase 9 builds the shell. Until
then the real IPC surface is reachable
from the devtools console under `cargo tauri dev`, and it is the same surface the app will
ship with; the debug-only `dev_*` commands that stood in for it through Phases 2–5 are gone.

```js
const { invoke, Channel } = window.__TAURI__.core;

await invoke('add_library_root', { path: '/Users/you/Library/Audio/Samples' });
await invoke('list_library_roots');

// Scans are long, so they stream. The promise resolves with a scanId as soon as the run row
// exists; progress arrives on the channel at <= 10 Hz and always ends in a `finished` event.
const progress = new Channel();
progress.onmessage = (e) => console.log(e);
const scanId = await invoke('scan_library', { rootId: 1, onProgress: progress });
// await invoke('cancel_scan', { scanId });

// Then build the map. Same shape: a job id now, the run id in the terminal event.
const refit = new Channel();
refit.onmessage = (e) => console.log(e);
await invoke('start_refit', { params: { algorithm: 'umap' }, onProgress: refit });
```

`scan_library` embeds if — and only if — a model is installed. A scan with no model is not a
failure: it leaves its rows at `decoded`, and the next scan with a session finishes them,
because a library is worth indexing before a 200 MB download is. Run it twice on the same
folder — the second run should report everything skipped and finish in a fraction of the
time.

Three commands answer in raw bytes rather than JSON, which `invoke` hands back as an
`ArrayBuffer`:

```js
const cloud = await invoke('get_point_cloud'); // 800 KB at 50,000 points
const bpm = await invoke('get_feature_column', { feature: 'bpm' });
const ids = await invoke('query_samples', {
  filter: { rootIds: [], tags: [], exts: ['wav'], features: [], projectedOnly: true },
});
```

Decoding those is `src/ipc/binary.ts`, and the typed wrappers around every command are
`src/ipc/commands.ts` — nothing outside `src/ipc/` calls `invoke` directly.

`get_similar` is the instrument for [risk 5](overview.md#10-risk-register): it ranks every
stored vector against one sample by exact cosine over the memory-mapped matrix. Brute force
on purpose — an approximate index would put its own recall between the question and the
answer. It is what the "do kicks retrieve kicks" evaluation will be run through, and it will
return nonsense until the real model exists.

### Model provisioning

```js
await invoke('get_model_status'); // installed / downloadable / unpinned, and partial bytes
const dl = new Channel();
dl.onmessage = (e) => console.log(e);
await invoke('download_model', { onProgress: dl }); // resumable, verified, atomic install
// await invoke('cancel_download');
```

These do nothing useful until the model has been exported and pinned (see
[Phase 3 status](#phase-3-status)), and `get_model_status` will say so — it reports
`unpinned`, and `download_model` refuses, because there is nothing safe to fetch. That is the
correct behaviour for this state, not a bug, and it is what
[`session.rs`](src-tauri/tests/session.rs) exercises with synthetic graphs instead.

`AUDIOBANK_FORCE_CPU=1` skips CoreML, which is how the "verified CPU fallback" half of the
Phase 3 exit criteria gets exercised without a machine that lacks a Neural Engine.

Benchmarks are `#[ignore]`d, so they compile on every `cargo test` and run only on request:

```sh
cargo test --manifest-path src-tauri/Cargo.toml --profile perf --test benchmarks \
    -- --ignored --nocapture --test-threads=1
```

## Phase 3 status

The Rust side is complete and tested; the offline export is not. The split matters because
`task.md` Phase 3 says that if parity fails, stop — and "parity was never checked" must not
be able to masquerade as "parity passed".

**Done and under test.** The mel front-end
([`mel.rs`](src-tauri/src/pipeline/mel.rs)) is a transcription of what LAION-CLAP does to a
waveform before its first convolution: 1024/480 framing shared with the Phase 2 DSP pass,
`center=true` reflect padding to 1001 frames, a power spectrogram, a Slaney-scale
Slaney-normalized 64-band filterbank from 50 Hz to 14 kHz, and `10·log10` with no `top_db`
clamp. Cross-checked against an independent NumPy implementation of `librosa.filters.mel`
and of the whole STFT chain: the filterbank agrees to 5·10⁻⁸, and the log-mel agrees to
2·10⁻⁵ dB for every bin within 40 dB of the peak. (The remaining ~0.16 dB differences are
all at −97 dB, where they are f32 FFT noise against an f64 reference.)

The downloader and the session are covered by real tests, including a kill-mid-transfer that
resumes from the exact byte offset, a server that ignores `Range` and forces a clean
restart, a hash mismatch that installs nothing and discards the poisoned partial, and a
cancel that keeps what it had.

**Tested against a real ONNX Runtime.** `scripts/make_session_fixtures.py` generates three
~130 KB graphs carrying the real export's input signature and output width, and
[`session.rs`](src-tauri/tests/session.rs) runs `ort` against them: the session builds,
CoreML binds, the warmup runs, batches match one-at-a-time runs, a graph with a dynamic mel
axis is refused at init, and both providers initialize. They compute nonsense on purpose —
parity is a separate job needing the real model — but every line of `model/session.rs` has
now met the library it wraps.

One measurement worth carrying forward: **CoreML does not compute this in f32.** Against an
f64 NumPy reference the CoreML provider's components differ by up to ~1.5e-5 where the CPU
provider differs by ~1e-7, and CoreML and CPU differ from each other by ~5e-5. That is fp16
accumulation on the ANE doing what it is designed to do. It is nowhere near disturbing the
parity gate's cosine floor of 0.999, but Phase 5 should not assume tighter than the numbers
support.

**Not done.** Running `scripts/export_clap_onnx.py` — a one-time, offline, developer-only
step needing `torch` and `laion-clap` — to produce the `.onnx`, publish it as a release
asset, pin its SHA-256 in `ModelRelease::CURRENT`, and commit the twenty reference
embeddings that are the parity oracle. Until then the gate in
[`parity.rs`](src-tauri/tests/parity.rs) skips with a printed reason.

**What stops that from being forgotten.** Three interlocks, all of which run on every
`cargo test`:

1. `ModelRelease::CURRENT.sha256` is the `UNPINNED` sentinel, and `Downloader::ensure`
   refuses to open a socket without a real digest. There is no way to download 200 MB that
   cannot be verified.
2. `the_parity_gate_is_not_silently_disabled` fails the build if the release is pinned and
   the oracle is not committed. Publishing a model without recording what it produces is
   unreachable by accident.
3. `the_two_fixture_generators_agree` runs today. The twenty fixtures are recipes, generated
   in both Python and Rust rather than committed as 19 MB of wavs, and each carries a
   waveform probe — so a drift between the two generators reports itself instead of arriving
   later disguised as a parity failure.

## Phase 7 status

Three of Phase 7's four exit criteria are met and measured; the fourth fails on the
measurement rather than on the renderer. Run them yourself:

```sh
npm run probe:webgl     # ALIASED_POINT_SIZE_RANGE, in a real WKWebView
npm run profile         # the 30 s scripted orbit and every exit criterion
```

Both drive [`scripts/webview_eval.swift`](scripts/webview_eval.swift), a small WKWebView that
loads a page, evaluates one expression and prints the JSON it resolves to. That exists
because `overview.md` §5.1's entire caveat is that WebKit runs WebGL through ANGLE on Metal
and does not behave like Chrome — a 60 fps number from a desktop browser is not evidence
about the engine this ships in. Numbers in [`BENCHMARKS.md`](BENCHMARKS.md).

**The point-size cap is `[1, 511]` device pixels**, so `THREE.Points` stands and the
`InstancedMesh` quad path is not needed. `overview.md` §5.1 makes that decision conditional
on a cap that "can be as low as 64 px" and `task.md` puts the query first in the phase for
exactly that reason; it is eight times the floor. [`scene/caps.ts`](src/scene/caps.ts) runs
the same query at startup and the shader clamps to what it finds, because this is one Mac.

**One draw call, and React is not in the frame loop.** The whole library is one
`BufferGeometry`: `position` static, `aColor` and `aSize` `DynamicDrawUsage`, `aId` the
picking payload. Per-frame cost is a dozen uniform writes and one `gl.drawArrays`. Hover and
selection are two uniforms rather than attribute mutation — the same rule as
`overview.md` §5.6 and less work than the roadmap's spelling of it, since re-uploading a
200 KB attribute twenty times a second to change one point out of 50,000 is what the rule was
written against.

**`aId` is the cloud index, not the sample id.** A `float` attribute is exact to 2^24 and
sample ids are `i64` that grow with every rescan of an edited library, so picking by id would
start returning a neighbouring sample past sixteen million rows, silently and with no error
anywhere. Index zero is shifted to one so the cleared background cannot decode to a point.

**The picking window is 21 pixels, not the 1 px in §5.3.** A point whose _centre_ falls
outside the viewport is clipped before rasterization, so a literal 1×1 pick can only hit a
point centred on that exact pixel however large the sprite covering it is. A small window
searched outward from the middle gives the better semantics anyway: the nearest sprite to the
cursor, and among those, the one nearest the camera.

**The frame-time criterion is the one that fails, and the control is why.** The orbit
measures p50 = 16 ms — exactly the display interval — and p99 = 33 ms against a 16.6 ms
target. But `requestAnimationFrame` on a **blank page** in this same WebView has a p99 of
23 ms, and the same orbit loop with the cloud hidden measures 37 ms. The threshold is below
the floor of the environment it is being measured in, and drawing 50,000 points is not
detectable above that floor. WebKit exposes no `EXT_disjoint_timer_query_webgl2`, so there is
no GPU clock available to a page; a real answer needs Phase 10's Instruments pass, which
`task.md` already assigns there.

**Coordinates are not normalized on arrival.** Phase 5 works hard to keep a layout stable
across re-fits and Procrustes preserves scale; rescaling the cloud into a unit cube on load
would move every point on screen by two percent when 5,000 samples are added, even though the
alignment worked perfectly. The buffer keeps the numbers the core sent and the camera adapts
— [`scene/framing.ts`](src/scene/framing.ts).

**What is not here.** The shell. [`App.tsx`](src/App.tsx) fetches the cloud, wires the two
data props and renders a one-line readout, which is the IPC-to-scene boundary Phase 9 will
inherit; the panels, the search field and the filter UI are Phase 9's. Nothing yet writes to
`useSceneStore.setColorBy` or `setFilter`, so those paths are exercised by the harness rather
than by a user.

## Phase 6 status

Phase 6's exit criteria are met. The 50,000-point cloud is one **800,016-byte** payload
against a 900 KB budget, and it becomes four typed arrays in **0.001 ms** against a 300 ms
budget — which is not a fast decoder, it is the absence of one. Numbers in
[`BENCHMARKS.md`](BENCHMARKS.md).

Four things worth knowing before touching this code:

**`get_feature_column` carries no ids, and the order is the contract.** The column is in the
point cloud's order, and both come from the same `ORDER BY sample_id` over the active
projection, so index `i` is the same sample in both. Shipping ids alongside would add 200 KB
to a payload that is refetched every time the user changes what the map is coloured by. The
`count` in the header is what makes a mismatched pair — a re-fit landing between the two
fetches — a caught error on the frontend rather than a map coloured by somebody else's
numbers. A null cell is `NaN`, not zero: zero is a legitimate value for every column in the
schema, and "no BPM" is not "0 BPM".

**The header is sixteen bytes because `new Float32Array(buffer, offset, n)` throws on an
unaligned offset.** Sixteen is a multiple of every alignment a typed array can ask for, so
every column start is aligned by construction and later versions can add an `f64` column
without moving anything. The fourth word is reserved rather than removed for the same
reason; `abpeaks://` spends it on `coveredMs`.

**`start_refit` returns a job id, not a run id, and that is forced rather than chosen.**
`overview.md` §3.8 creates the shadow `projection_runs` row only once there are coordinates
to write — minutes into a 50,000-point UMAP fit — so a command that answered with the run id
would have to block for the length of the job. The run id arrives in the terminal event,
which is where the frontend wants it: it identifies the map that is now on screen, and before
the swap there is no such map. `task.md` Phase 6 lists this and seven other deliberate
deviations.

**`src/bindings/` is generated by `cargo test`, not by a build step.** `ts-rs` exports at
test time; [`bindings.rs`](src-tauri/tests/bindings.rs) is the generator, and CI fails the
build if what it writes differs from what is committed. Change a Rust IPC struct, run
`cargo test`, commit the diff. That test uses an explicit `ts_rs::Config` rather than
`#[ts(export)]` for one reason worth not re-discovering: ts-rs defaults `i64` to `bigint`,
and every 64-bit value in this app crosses the boundary as a JSON number that JavaScript
parses into a `number` — a `bigint` type surface would be a lie about the values behind it.

**What is not covered.** Two commands in `overview.md` §6.1 — `play_sample` and
`stop_playback` — reject with `{ kind: 'unavailable' }` until Phase 8 builds the audio
engine. Their signatures are here because the generated bindings are Phase 6's deliverable
and `task.md`'s parallelism note has Phase 9's shell built against them; a command that
lies about working would be worse than one that says which feature has not landed.

## Phase 5 status

Phase 5's exit criteria are met, and the map it produces is only as meaningful as the
vectors underneath it — which is still Phase 3's open question, not a new one.

Three things worth knowing before touching this code:

**`annembed` panics on a disconnected kNN graph, and release builds abort on panic.** Its
diffusion-map initialization degenerates on a graph in separate components, and
`set_data_box` then trips a bare `assert!`. A library of five hundred near-identical 909
kicks next to a folder of vocal loops is exactly that shape. `projection/umap.rs` counts
components before calling `embed()` and refuses the graph; `Refit::with_fallback` turns the
refusal into a PCA layout, which is what `overview.md` §3.7 means by keeping PCA as a tested
fallback rather than a theoretical one.

**A run is recorded under the projector that produced it**, never the one that was asked
for. That is why the fallback is a field on the job rather than a `Projector` wrapping two:
a wrapper would have to answer `name()` before knowing which one ran.

**Read the active run and its points in one statement.** The swap deletes the superseded run
inside the transaction that activates the new one, so two separate queries can read a run id
and then find nothing under it. `queries::active_projection_points` does the join in one
statement, and Phase 6's `get_point_cloud` must call that rather than composing two reads.

## Phase 4 status

The five-stage pipeline is built, wired, and tested end to end. Two of Phase 4's claims wait
on Phase 3's export, and they are the two that are about meaning rather than mechanism.

**Done and under test.** Walk → decode → mel → embed → persist, with the queue depths from
`overview.md` §3 and one deliberate departure from its diagram: **decode and mel share a
stage.** §3 draws them apart and then observes that a decoded window is 1.92 MB against a mel
tensor's 256 KB, so a worker decodes, analyzes and computes the spectrogram for one file and
hands its window straight back to the pool. What crosses the next channel is the spectrogram,
which is why a 400-file scan peaks at 17.5 MiB rather than at a queue depth times 1.92 MB.

[`embed.rs`](src-tauri/src/pipeline/embed.rs) is the batching accumulator: sixteen
spectrograms per `run()` with a flush timeout so the tail of a scan does not wait for a batch
that will never fill, L2-normalized on receipt, one session shared behind its own mutex.
It talks to an `Embed` trait rather than to `ort`, which keeps the rc-upgrade blast radius at
one file and lets the batcher be tested against a counting fake — including the property a
real session makes nearly unobservable, that row 3's vector reaches row 3 and not row 5.

Dedup now saves inference as well as decode: a duplicate file stores a _reference_ to its
twin's bytes, so six copies of a 909 kick cost one `run()` and one kilobyte. Duplicates that
overtake the file they copy from — which happens, because decode order is not send order —
are resolved at the end of the scan rather than guessed at.

[`embed.rs` tests](src-tauri/tests/embed.rs) run all of it against a real ONNX Runtime on the
synthetic fixture graph: every row embedded and normalized, batches rather than per-file
runs, duplicates borrowing vectors, a scan with no model leaving work that a later scan
finishes, a rescan reading nothing, a changed file losing its stale vector, quarantine
without reaching the model, a cancelled scan keeping what it wrote, and progress that is
coalesced and always terminates.

**The finding worth carrying forward: the mel filterbank was the entire pipeline.** The first
five-stage measurement came in at 218 samples/s against Phase 2's 1,599 for decode + DSP
alone, and the batch-size sweep was flat from 1 to 64 — which is the tell that inference is
not the bottleneck. Applying a 64 × 513 mel bank densely to 1,001 frames is 32.9 million f64
multiply-adds per file, and a mel filter spans about sixteen FFT bins, so nearly all of that
was summing zeros. Restricting each row to its nonzero span is **7.5× end to end** and
**bit-identical** — asserted by bit pattern, not by tolerance, because this is the parity
surface.

**Not done, and blocked on the export.** The `≥ 60 samples/s` exit criterion, which is stated
against real CLAP and is not claimed by a measurement taken against a 130 KB graph; and the
risk-5 evaluation, the hand inspection of whether kicks retrieve kicks on a real one-shot
library. The instrument for the second is built and committed (`dev_neighbors`), so that
evaluation is one command away from being possible. The duration-weighted CLAP + DSP blend
that risk 5 might call for is deliberately _not_ built: it is conditional on an evaluation
that cannot run, and a weighting tuned against nonsense embeddings would be chosen by coin
flip.

## Builds

```sh
cargo tauri build --target aarch64-apple-darwin
cargo tauri build --target x86_64-apple-darwin
```

Unsigned, single-arch. The universal-binary merge, signing, and notarization are Phase 11
— and the `ort` universal-binary problem (`overview.md` §8.1) is the known sharp edge
there, not an afterthought.

## Checks that gate a merge

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml
npm run typecheck && npm run lint && npm run format:check
```

CI runs all of these plus both target builds on every push.

## Layout

The tree follows `overview.md` §9. [`src-tauri/src/db/`](src-tauri/src/db/),
[`src-tauri/src/pipeline/`](src-tauri/src/pipeline/),
[`src-tauri/src/model/`](src-tauri/src/model/) and
[`src-tauri/src/projection/`](src-tauri/src/projection/) are implemented; every other
`src-tauri/` module still holds only a `//!` doc comment stating its responsibility and the
phase that fills it in — the skeleton is there so that later phases add code to a named place
rather than inventing structure under deadline.

[`src-tauri/vendor/annembed/`](src-tauri/vendor/annembed/) is the one third-party crate
checked into the repo, for the reason `overview.md` §10 risk 2 gives: a 0.1.x crate with one
maintainer decides the layout of the user's entire map. Its `.rs` files are upstream 0.1.6
byte for byte; only its manifest is edited, and its own `Cargo.toml` says what changed and
why.

Two invariants in `db/` are worth knowing before touching it. The pool hands out
`SQLITE_OPEN_READ_ONLY` connections and the single write connection is moved into the writer
thread at construction, so "one writer" (cross-cutting rule 2) is enforced by SQLite and by
ownership rather than by review. And the writer defers each caller's reply until after the
commit that makes the write visible — answering earlier turns "the write returned" into a
race against a pooled reader.

In `pipeline/`, the load-bearing detail is that **channel senders are what shut the pipeline
down**. The walk thread owns the only `DiscoveredFile` sender, so finishing the walk ends the
decode stage, which drops the only `Analyzed` sender, which ends the persist stage. There is
no separate shutdown protocol, and adding one would give the pipeline a second way to
terminate that the first way does not know about.

On the frontend, [`src/scene/`](src/scene/) is the whole visualization layer and
[`src/ipc/`](src/ipc/) is the only code that calls `invoke`. The two do not meet: the scene
does no IPC and the store holds no point data, so something has to fetch the bytes and hand
them over, and that is the shell in [`App.tsx`](src/App.tsx). [`src/profile/`](src/profile/)
is the Phase 7 measurement harness — it mounts the real scene over a synthetic library and is
built only by `vite build --mode profile`, never into the app.

`model/session.rs` is the **only** file in the crate permitted to name an `ort` type
(cross-cutting rule 7). `ort` is pinned to an exact release candidate whose API moves
between rcs; keeping it behind one file means an upgrade has one blast radius. What leaves
that module is `f32`, a `Provider`, and a `SessionError`.

## Notes on deviations from the roadmap

- **Tailwind v4** is CSS-first: there is no `content` array in a JS config any more, and
  no `tailwind.config.ts` in the tree. The equivalent lives in
  [`src/styles/index.css`](src/styles/index.css) as `@source "../**/*.{ts,tsx}"`.
- **TypeScript is pinned to 5.9**, not 7.x. `typescript-eslint` declares
  `typescript <6.1.0`; TS 7 would mean dropping type-aware linting.
- **`rusqlite` is 0.39, not 0.40**, and `r2d2_sqlite` 0.34, not 0.35. `refinery-core`
  0.9.2 declares `rusqlite >=0.23, <=0.39`; pairing it with 0.40 resolves two copies of
  `libsqlite3-sys`, both of which set `links = "sqlite3"`, and the link fails. 0.39 is the
  newest coherent set. Revisit when `refinery` widens the bound.
- **The FTS5 tokenizer drops `tokenchars '_-'`** from the DDL in `overview.md` §4.1. That
  option makes `_` and `-` token characters rather than separators, so
  `KICK_808_Distorted-02.wav` indexes as one token and searching `808` — the example the
  document itself gives — matches nothing. With the default separators both forms work: the
  fragment as a bareword, the compound as a quoted phrase. See the note in
  [`V1__initial.sql`](src-tauri/migrations/V1__initial.sql).
- **The project was scaffolded by hand** rather than with `cargo create-tauri-app`, which
  requires an empty directory and would not have produced the §9 layout anyway.
- **The walker does not honour `.gitignore`.** `overview.md` §3.1 picks `ignore` partly
  because it gets VCS-aware filtering for free, but a sample library that happens to sit
  inside a repository whose `.gitignore` says `*.wav` — an entirely ordinary thing for a
  repository to say — would scan to zero samples, with no error and nothing to explain it.
  Hidden-file filtering is kept, because that is what excludes `.DS_Store` and resource
  forks; every other `ignore` source is switched off in
  [`walk.rs`](src-tauri/src/pipeline/walk.rs).
- **Key estimation gets its own FFT**, against §3.3's "same pass" arrangement. A 1024-point
  window at 48 kHz has 46.9 Hz bins; a semitone at middle C is 15 Hz. Pitch is simply not
  resolvable in the CLAP-contract frames in the register where pitch lives — measured, a C
  major triad comes back as C# minor, confidently. Sharing the transform is an efficiency
  argument and not an argument for filling a column with wrong numbers, so chroma runs an
  8192-point pass over the same buffer. It costs 3 ms of the 15 ms DSP window; see
  [`BENCHMARKS.md`](BENCHMARKS.md).
- **`key_mode` is left `NULL` when only the root is knowable.** A single bass note has a
  root and no mode, and reporting a coin-flip major/minor for it would put half a library's
  bass hits on the wrong side of Phase 9's key filter. The schema makes the two columns
  separately nullable, which is what makes this expressible.
- **`ort` sessions are not concurrently runnable**, against `overview.md` §3.4, which
  prescribes an `Arc<Session>` shared across `rayon` workers on the grounds that ONNX
  Runtime is thread-safe for concurrent `run()`. It is not, and `ort` 2.0.0-rc.13 says so
  in as many words: `Session::run` takes `&mut self` because earlier versions that allowed
  concurrent inference "often saw crashes and memory corruption". So the session is still
  built once and shared as an `Arc` — the expensive thing still happens exactly once — but
  inference is serialized behind a `Mutex`. §3.4's other prescription is what makes this
  cheap: the embed stage batches 16–32 spectrograms per `run()`, so the parallelism lives in
  decode and mel, and the serialized region is one large matmul per batch. Phase 4's
  throughput number decides whether that holds; if it does not, the fix is one session per
  worker thread, not concurrent `run()`.
- **ONNX Runtime links statically**, which is worth knowing before Phase 11. `ort`'s
  `download-binaries` produces a static library, not a dylib: `otool -L` on the built binary
  shows `CoreML.framework` and no `libonnxruntime`. `overview.md` §8.1's universal-binary
  problem is stated in terms of merging two ONNX Runtime _dylibs_, and there is no dylib to
  merge — which removes the second half of both approaches it proposes. The `lipo`
  verification stays; there is simply less to verify.
- **The graph takes log-mel, not a waveform**, so `overview.md` §3.3's shared STFT actually
  saves a transform. The cost is that the front-end becomes AudioBank's responsibility
  rather than the checkpoint's, which is what the parity gate exists to police. The tensor
  layout is read off the graph at session init rather than assumed: §3.4 writes the input as
  `[B, 1, 64, T]` and HTSAT's own layout is `[B, 1, T, 64]`, and rather than pick a winner on
  paper, `model::session::layout_of` accepts either and transposes if needed — while
  rejecting any graph with a dynamic mel or frame axis, because a graph that will accept the
  wrong number of mel bins will accept the wrong number of mel bins.
- **`rust-version` moved from 1.77 to 1.88**, which is what `ignore` 0.4.33, `rubato` 5.0 and
  `rayon` 1.12 — the versions `task.md` Phase 2 pins — require. CI builds on stable.
- **Hover and selection are uniforms, not attribute mutations.** `task.md` Phase 7 asks for
  "direct attribute mutation + `needsUpdate`", and the rule behind that — no React state per
  point — is kept exactly. The mechanism is not: hover changes one point out of 50,000, and
  flagging `aColor` or `aSize` for upload re-sends a 200 KB buffer twenty times a second to
  change one of them. Two `float` uniforms compared in the vertex shader cost nothing and
  cannot drift out of sync with the picking pass, which reads the same values. Attribute
  mutation is still how colour-by and filter dimming work, because those genuinely are
  per-point.
- **The pick window is 21 device pixels, not one.** `overview.md` §5.3 describes "a 1×1
  scissored" read. Point primitives are clipped by their centre, so a 1×1 viewport can only
  return a point whose centre is on that exact pixel — the sprite covering the cursor is
  never drawn if its centre is a pixel away. Widening the window and searching outward from
  the middle also changes the semantics for the better: the nearest sprite to the cursor,
  and among those, the nearest to the camera.
- **Frame time is measured from frame intervals, not from a GPU timer or a Chrome trace.**
  `overview.md` §7 specifies "`WebGLRenderer.info` + Chrome DevTools trace". Chrome is not
  the engine this ships in, and WebKit exposes no `EXT_disjoint_timer_query_webgl2` — there
  is no GPU clock available to a page at all. What the harness reports instead is the
  interval between rendered frames, the CPU time inside `render()`, the dropped-frame count,
  and two controls that establish what the engine's own tail is. See
  [`BENCHMARKS.md`](BENCHMARKS.md) on why the absolute criterion cannot be met by any
  renderer in that environment.
- **`scripts/webview_eval.swift` uses one piece of WebKit SPI.**
  `-[WKWebView _setWindowOcclusionDetectionEnabled:]`, guarded by `responds(to:)`. macOS
  suspends `requestAnimationFrame` for a window it considers occluded — including one merely
  sitting behind a full-screen terminal — and in that state the page runs zero frames and the
  harness looks like it has hung. The alternative is a benchmark whose ability to run depends
  on which window happens to be in front. This file is a developer tool in `scripts/`; it is
  never bundled and no SPI appears anywhere in the app.
