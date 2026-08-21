# AudioBank

A spatial browser for large sample libraries. Samples are embedded with CLAP, projected to
3D, and rendered as a point cloud you can orbit and audition.

- [`overview.md`](overview.md) — the architecture.
- [`task.md`](task.md) — the implementation roadmap, phase by phase.

**Status: Phase 2 complete.** An empty window builds and launches, and behind it the ingest
pipeline is real: point it at a folder of audio and it walks, hashes, deduplicates, decodes,
analyzes, and writes rows with DSP features. No ML is involved yet — the CLAP model and the
embeddings it produces are Phases 3 and 4, and until then `samples.status` stops at
`decoded`.

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
await invoke('dev_scan'); // walks, decodes, analyzes, and returns counts
```

`dev_scan` returns files seen / added / skipped / failed, how many were decoded against how
many were deduplicated, and the totals in the database afterwards. Run it twice: the second
run should report everything skipped and finish in a fraction of the time.

Benchmarks are `#[ignore]`d, so they compile on every `cargo test` and run only on request:

```sh
cargo test --manifest-path src-tauri/Cargo.toml --profile perf --test benchmarks \
    -- --ignored --nocapture --test-threads=1
```

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

The tree follows `overview.md` §9. [`src-tauri/src/db/`](src-tauri/src/db/) and
[`src-tauri/src/pipeline/`](src-tauri/src/pipeline/) are implemented; every other
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
- **`rust-version` moved from 1.77 to 1.88**, which is what `ignore` 0.4.33, `rubato` 5.0 and
  `rayon` 1.12 — the versions `task.md` Phase 2 pins — require. CI builds on stable.
