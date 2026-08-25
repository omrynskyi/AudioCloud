//! Generates `src/bindings/` from the Rust IPC types (`overview.md` §6.8).
//!
//! **This test is the generator.** `ts-rs` exports at test time, and CI runs this and then
//! fails the build if the working tree differs from what is committed -- so a Rust struct
//! that changes shape cannot silently desynchronize the frontend. Running `cargo test` and
//! committing the diff is the intended workflow; there is no separate build step and no
//! `build.rs` writing into `src/`.
//!
//! ### Why an explicit `Config` rather than `#[ts(export)]`
//!
//! `#[ts(export)]` generates a per-type test that reads `Config::from_env()`, whose defaults
//! are wrong for this project in two ways that would only show up at runtime:
//!
//! - **`i64` and `u64` default to `bigint`.** Sample ids, timestamps and byte counts are all
//!   64-bit in Rust, and every one of them crosses the boundary as a JSON number that
//!   JavaScript parses into a `number`. Typing them as `bigint` would be a type surface that
//!   lies about the values behind it -- `sampleId * 2` would not compile against a value
//!   that is, at runtime, a `number`.
//! - **The output directory defaults to `src-tauri/bindings/`.** Overriding it through
//!   `TS_RS_EXPORT_DIR` means a `.cargo/config.toml` whose discovery depends on the working
//!   directory cargo was invoked from, and CI invokes cargo from the repository root with
//!   `--manifest-path`. One explicit `Config` in one test has no such failure mode.
//!
//! Adding a type to the IPC surface means adding it to [`export_bindings`]. A type reachable
//! from one already listed is exported with it -- `export_all` walks dependencies -- so in
//! practice only new *roots* need a line here.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ts_rs::{Config, TS};

use audiobank_lib::{
    error::AppError,
    ipc::{
        events::{DownloadEvent, RefitEvent, ScanEvent},
        types::{
            AppSettings, AudioDeviceInfo, Collection, CollectionDetail, Feature, LibraryRoot,
            ModelStatus, Neighbor, QueryFilter, RefitParams, SampleDetail, Tag,
        },
        BINDINGS_DIR,
    },
};

/// Writes every IPC type to `src/bindings/`.
#[test]
fn export_bindings() {
    let cfg = Config::new()
        // See the module note: these two values are the whole reason this test exists
        // instead of `#[ts(export)]`.
        .with_large_int("number")
        .with_out_dir(BINDINGS_DIR);

    // Roots. Everything else is reachable from one of these and is exported alongside it.
    AppError::export_all(&cfg).unwrap();
    LibraryRoot::export_all(&cfg).unwrap();
    SampleDetail::export_all(&cfg).unwrap();
    Neighbor::export_all(&cfg).unwrap();
    Tag::export_all(&cfg).unwrap();
    Collection::export_all(&cfg).unwrap();
    CollectionDetail::export_all(&cfg).unwrap();
    AudioDeviceInfo::export_all(&cfg).unwrap();
    AppSettings::export_all(&cfg).unwrap();
    Feature::export_all(&cfg).unwrap();
    QueryFilter::export_all(&cfg).unwrap();
    ModelStatus::export_all(&cfg).unwrap();
    RefitParams::export_all(&cfg).unwrap();
    ScanEvent::export_all(&cfg).unwrap();
    RefitEvent::export_all(&cfg).unwrap();
    DownloadEvent::export_all(&cfg).unwrap();
}

/// A 64-bit integer must land as `number`, not `bigint`.
///
/// The generated `.ts` files are checked into the tree and diffed by CI, so this could be
/// left to a reviewer noticing. It is asserted because the failure mode is silent: `bigint`
/// compiles, and the mismatch only appears when someone does arithmetic on a sample id.
#[test]
fn sixty_four_bit_ids_are_typed_as_numbers() {
    let cfg = Config::new().with_large_int("number");
    let ts = SampleDetail::export_to_string(&cfg).unwrap();
    assert!(ts.contains("id: number"), "{ts}");
    assert!(!ts.contains("bigint"), "{ts}");
}

/// The error surface is a discriminated union the frontend can switch on exhaustively.
#[test]
fn the_error_type_is_a_discriminated_union() {
    let ts = AppError::export_to_string(&Config::new()).unwrap();
    assert!(ts.contains(r#"{ "kind": "modelMissing" }"#), "{ts}");
    assert!(ts.contains(r#""kind": "internal""#), "{ts}");
}
