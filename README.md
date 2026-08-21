# AudioBank

A spatial browser for large sample libraries. Samples are embedded with CLAP, projected to
3D, and rendered as a point cloud you can orbit and audition.

- [`overview.md`](overview.md) — the architecture.
- [`task.md`](task.md) — the implementation roadmap, phase by phase.

**Status: Phase 1 complete.** An empty window builds and launches, and behind it the data
layer is real: migrated SQLite schema, one writer thread, a four-connection read pool, and
the append-only embedding file. Nothing produces data for it yet — discovery and decode are
Phase 2.

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

The tree follows `overview.md` §9. [`src-tauri/src/db/`](src-tauri/src/db/) is implemented;
every other `src-tauri/` module still holds only a `//!` doc comment stating its
responsibility and the phase that fills it in — the skeleton is there so that later phases
add code to a named place rather than inventing structure under deadline.

Two invariants in `db/` are worth knowing before touching it. The pool hands out
`SQLITE_OPEN_READ_ONLY` connections and the single write connection is moved into the writer
thread at construction, so "one writer" (cross-cutting rule 2) is enforced by SQLite and by
ownership rather than by review. And the writer defers each caller's reply until after the
commit that makes the write visible — answering earlier turns "the write returned" into a
race against a pooled reader.

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
