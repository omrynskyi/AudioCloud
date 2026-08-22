# AudioBank

A spatial browser for large sample libraries. Samples are embedded with CLAP, projected to
3D, and rendered as a point cloud you can orbit and audition.

- [`overview.md`](overview.md) — the architecture.
- [`task.md`](task.md) — the implementation roadmap, phase by phase.

**Status: Phase 4 implemented, Phase 3's gate not yet passed.** An empty window builds and
launches, and behind it the ingest pipeline is real and complete: point it at a folder of
audio and it walks, hashes, deduplicates, decodes, analyzes, computes CLAP's log-mel
spectrogram, embeds it in batches through one shared `ort` session, and writes both the DSP
features and the vector — with progress coalesced to 10 Hz, cancellation that keeps what it
wrote, and a resume that finishes what an interrupted scan started.

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
```

The app data directory — the SQLite database, its WAL, and `embeddings.bin` — is
`~/Library/Application Support/com.audiobank.app/`, created 0700 on first run. Deleting it
is a supported reset: every row in it is derived from files on disk, apart from library
roots, tags, and collections.

### Running a scan

The real IPC surface is Phase 6. Until then there are three development commands, compiled
only into debug builds so they cannot quietly become the surface Phase 6 was meant to build.
From the devtools console under `cargo tauri dev`:

```js
const { invoke } = window.__TAURI__.core;

await invoke('dev_add_root', { path: '/Users/you/Library/Audio/Samples' });
await invoke('dev_list_roots');
await invoke('dev_scan'); // walks, decodes, analyzes, embeds, and returns counts
await invoke('dev_neighbors', { query: 'kick', limit: 20 }); // nearest neighbors, by cosine
```

`dev_scan` returns files seen / added / skipped / failed, how many were decoded against how
many were deduplicated against how many were embedded, and the totals in the database
afterwards. Run it twice: the second run should report everything skipped and finish in a
fraction of the time.

It embeds if — and only if — a model is installed; the `inference` field says which execution
provider bound, or why no session was used. A scan with no model is not a failure. It leaves
its rows at `decoded`, and the next scan with a session finishes them, because a library is
worth indexing before a 200 MB download is.

`dev_neighbors` is the instrument for [risk 5](overview.md#10-risk-register): it ranks every
stored vector against one sample by exact cosine over the memory-mapped matrix. Brute force
on purpose — an approximate index would put its own recall between the question and the
answer. It is what the "do kicks retrieve kicks" evaluation will be run through, and it will
return nonsense until the real model exists.

### Model provisioning

Three more debug-only commands cover Phase 3. They do nothing useful until the model has
been exported and pinned (see [Phase 3 status](#phase-3-status)), and `dev_model_status`
will say so.

```js
await invoke('dev_model_status'); // installed / downloadable / unpinned, and partial bytes
await invoke('dev_download_model'); // resumable, verified, atomic install
await invoke('dev_session_info'); // forces the lazy session; reports the bound provider
```

`AUDIOBANK_FORCE_CPU=1` skips CoreML, which is how the "verified CPU fallback" half of the
Phase 3 exit criteria gets exercised without a machine that lacks a Neural Engine.

Until the model is exported and pinned, `dev_model_status` reports `unpinned`,
`dev_download_model` refuses (there is nothing safe to fetch), and `dev_session_info` reports
no model. That is the correct behaviour for this state, not a bug — and it is what
[`session.rs`](src-tauri/tests/session.rs) exercises with synthetic graphs instead.

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
[`src-tauri/src/pipeline/`](src-tauri/src/pipeline/) and
[`src-tauri/src/model/`](src-tauri/src/model/) are implemented; every other
`src-tauri/` module still holds only a `//!` doc comment stating its responsibility and the
phase that fills it in — the skeleton is there so that later phases add code to a named place
rather than inventing structure under deadline.

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
