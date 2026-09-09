# AudioCloud

AudioCloud is a macOS app for browsing large audio-sample libraries as an interactive 3D map.
Import a folder, search and filter your samples, click a point to audition it, and explore nearby
sounds.

<!-- Add a product screenshot here when one is ready:
![AudioCloud screenshot](docs/screenshot.png)
-->

## How it works

AudioCloud processes samples locally:

1. It scans a folder, identifies audio files, and extracts basic audio features.
2. It creates a compact vector from each sample's log-mel spectrogram using AudioCloud's built-in
   spectral fingerprint. No CLAP model or model download is required.
3. It compares vectors with cosine similarity. The similarity value shown in the app is the cosine
   score between two normalized vectors, ranging from `-1` to `1` and displayed as a percentage.
4. It projects the vectors into 3D with PCA or UMAP and renders the result as a point cloud.

The vectors are deterministic: the same audio content produces the same fingerprint. Similarity is
based on spectral shape, attack, and early decay, so it is most useful for finding samples with
similar timbre and character—not for identifying the same musical role in every context.

## Requirements

- macOS 11 or later
- Apple Silicon (`aarch64-apple-darwin`)
- Rust stable
- Node.js version listed in [`.nvmrc`](.nvmrc)
- Xcode Command Line Tools

Install the Rust target and JavaScript dependencies:

```sh
rustup target add aarch64-apple-darwin
npm ci
```

## Run locally

```sh
npm run tauri -- dev
```

Useful checks:

```sh
npm run format:check
npm run lint
npm run typecheck
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml --all-features
```

Build the macOS app with:

```sh
npm run tauri -- build
```

## Data storage

AudioCloud stores its local database and generated vectors at:

```text
~/Library/Application Support/com.audiocloud.app/
```

The stored data is derived from the files in your library. Removing this directory resets the
index; your original audio files are not affected.

## Project structure

- `src/` — React, TypeScript, and Three.js frontend
- `src-tauri/src/pipeline/` — audio decoding, feature extraction, and vector generation
- `src-tauri/src/projection/` — PCA, UMAP, and incremental placement
- `src-tauri/src/db/` — SQLite metadata and the memory-mapped vector store
- `src-tauri/tests/` — Rust integration tests

## License and inspiration

AudioCloud is licensed under the [Apache License 2.0](LICENSE).

The spatial audio-browser concept was inspired by Google Creative Lab's [AI Experiments Drum
Machine](https://github.com/googlecreativelab/aiexperiments-drum-machine). That project is credited
for inspiration and is not included in this repository.
