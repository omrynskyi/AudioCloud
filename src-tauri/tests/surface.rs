//! The command surface, invoked through Tauri (`overview.md` §6.1).
//!
//! `tests/ipc.rs` proves the byte layouts and the queries. This proves the layer above them:
//! that the commands are *reachable* — registered under the names the frontend calls, taking
//! the argument names it sends, handing back the transport it expects, and rejecting with the
//! tagged union it switches on. None of that exists until Tauri routes a request, so none of
//! it can be checked by calling the functions directly.
//!
//! It runs on `tauri::test`'s mock runtime: a real `App`, a real invoke handler, real managed
//! state, and a webview that never draws. What it does not cover is the WebView side of the
//! boundary — `ArrayBuffer` materialization and `Channel` delivery into JavaScript — which
//! has no Rust-side surface to assert on and is measured instead by
//! `scripts/decode_point_cloud.mjs` against the same bytes the encoder here produces.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use audiobank_lib::{
    audio::{peaks::PeakCache, AudioPlayer},
    commands::Jobs,
    db::{Database, NewSample, SampleStatus},
    ipc::binary,
};
use serde_json::json;
use tauri::{
    ipc::{CallbackFn, InvokeBody, InvokeResponseBody},
    test::{mock_builder, mock_context, noop_assets, INVOKE_KEY},
    webview::InvokeRequest,
    Manager, WebviewWindow,
};
use tempfile::TempDir;

const DIM: usize = 8;

/// A running app with the real command surface and a real database behind it.
fn app() -> (TempDir, WebviewWindow<tauri::test::MockRuntime>) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), DIM).unwrap();
    let app = mock_builder()
        .invoke_handler(audiobank_lib::command_handler())
        .build(mock_context(noop_assets()))
        .unwrap();
    app.manage(db);
    app.manage(Jobs::new());
    app.manage(PeakCache::new());
    app.manage(Arc::new(AudioPlayer::new()));

    let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
        .build()
        .unwrap();
    (dir, webview)
}

/// Invokes a command the way the frontend does, with camelCase argument names.
fn invoke(
    webview: &WebviewWindow<tauri::test::MockRuntime>,
    cmd: &str,
    args: serde_json::Value,
) -> Result<InvokeResponseBody, serde_json::Value> {
    tauri::test::get_ipc_response(
        webview,
        InvokeRequest {
            cmd: cmd.into(),
            callback: CallbackFn(0),
            error: CallbackFn(1),
            // `tauri://localhost`, not `http://tauri.localhost`: `Webview::is_local_url`
            // compares against the platform's own protocol URL, and on macOS that is the
            // custom scheme. Getting it wrong makes every request a *remote* origin, which
            // the ACL refuses — correctly, and confusingly, since the rejection reads as a
            // missing command.
            url: "tauri://localhost".parse().unwrap(),
            body: InvokeBody::Json(args),
            headers: Default::default(),
            invoke_key: INVOKE_KEY.to_string(),
        },
    )
}

fn json_of(body: InvokeResponseBody) -> serde_json::Value {
    match body {
        InvokeResponseBody::Json(text) => serde_json::from_str(&text).unwrap(),
        InvokeResponseBody::Raw(bytes) => panic!("expected JSON, got {} raw bytes", bytes.len()),
    }
}

fn raw_of(body: InvokeResponseBody) -> Vec<u8> {
    match body {
        InvokeResponseBody::Json(text) => panic!("expected raw bytes, got JSON: {text}"),
        InvokeResponseBody::Raw(bytes) => bytes,
    }
}

/// Seeds a projected library through the managed database and returns its sample ids.
fn seed(webview: &WebviewWindow<tauri::test::MockRuntime>, count: usize) -> Vec<i64> {
    let db = webview.state::<Database>();
    let root = db.writer().add_root("/library", None).unwrap();
    let rows: Vec<NewSample> = (0..count)
        .map(|i| NewSample {
            root_id: root,
            rel_path: format!("drums/{i:03}.wav"),
            filename: format!("{i:03}.wav"),
            ext: "wav".into(),
            size_bytes: 1024,
            mtime: 1_700_000_000 + i as i64,
            content_hash: None,
            duration_ms: Some(500),
            sample_rate: Some(48_000),
            channels: Some(1),
            status: SampleStatus::Decoded,
        })
        .collect();
    let ids = db.writer().upsert_samples(rows).unwrap();

    let run = db
        .writer()
        .begin_projection_run("pca", "{}", count as i64)
        .unwrap();
    let points: Vec<(i64, [f32; 3])> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, [i as f32, 0.0, 0.0]))
        .collect();
    db.writer().set_projection_points(run, points).unwrap();
    db.writer().activate_projection_run(run).unwrap();
    db.writer().flush().unwrap();
    ids
}

/// The whole point of transport 2: a command that answers in bytes, not in a JSON array.
#[test]
fn the_point_cloud_arrives_as_raw_bytes_through_a_real_invoke() {
    let (_dir, webview) = app();
    let ids = seed(&webview, 32);

    let bytes = raw_of(invoke(&webview, "get_point_cloud", json!({})).unwrap());
    let header = binary::header(&bytes).unwrap();
    assert_eq!(header.magic, binary::MAGIC_POINT_CLOUD);
    assert_eq!(header.count, ids.len());
    assert_eq!(bytes.len(), binary::HEADER_BYTES + ids.len() * 16);
}

/// A run with no fit colors -- every one seeded by [`seed`], which only ever calls
/// `set_projection_points` -- answers with a full-length column of NaN, not an empty one or
/// an error. That is what lets the frontend treat "no fit colors yet" identically to "this
/// point has none", one NaN check instead of two code paths.
#[test]
fn point_colors_are_nan_for_a_run_that_never_set_any() {
    let (_dir, webview) = app();
    let ids = seed(&webview, 24);

    let bytes = raw_of(invoke(&webview, "get_point_colors", json!({})).unwrap());
    let header = binary::header(&bytes).unwrap();
    assert_eq!(header.magic, binary::MAGIC_POINT_COLORS);
    assert_eq!(header.count, ids.len());
    assert_eq!(bytes.len(), binary::HEADER_BYTES + ids.len() * 12);

    let first = f32::from_le_bytes(
        bytes[binary::HEADER_BYTES..binary::HEADER_BYTES + 4]
            .try_into()
            .unwrap(),
    );
    assert!(first.is_nan());
}

/// The other two byte transports, including the `Feature` enum arriving as its camelCase
/// serde name — the exact string the generated `Feature.ts` will make the frontend send.
#[test]
fn the_other_binary_commands_take_their_arguments_as_the_bindings_spell_them() {
    let (_dir, webview) = app();
    seed(&webview, 16);

    let column = raw_of(
        invoke(
            &webview,
            "get_feature_column",
            json!({ "feature": "spectralCentroid" }),
        )
        .unwrap(),
    );
    assert_eq!(binary::header(&column).unwrap().count, 16);

    // `{}` is a filter with no constraints, which is what `NO_FILTER` spread with nothing
    // over it deserializes to.
    let ids = raw_of(
        invoke(
            &webview,
            "query_samples",
            json!({ "filter": { "rootIds": [], "tags": [], "exts": [], "features": [], "projectedOnly": false } }),
        )
        .unwrap(),
    );
    assert_eq!(binary::header(&ids).unwrap().count, 16);
}

/// A JSON command, with a camelCase argument name and a camelCase result.
#[test]
fn a_json_command_round_trips_camel_case_in_both_directions() {
    let (_dir, webview) = app();
    let ids = seed(&webview, 4);

    let detail =
        json_of(invoke(&webview, "get_sample_detail", json!({ "sampleId": ids[1] })).unwrap());
    assert_eq!(detail["id"], json!(ids[1]));
    assert_eq!(detail["filename"], json!("001.wav"));
    // Present and camelCase: this is the value `peaksUrl` puts in the cache-busting query
    // string, and a rename would silently make every waveform stale forever.
    assert!(detail["updatedAt"].is_i64());
    assert_eq!(detail["embedded"], json!(false));
}

/// **The error contract.** A failing command must reject with the tagged union, not with a
/// string — this is what `switch (error.kind)` on the frontend is standing on.
#[test]
fn a_failing_command_rejects_with_the_tagged_error_union() {
    let (_dir, webview) = app();

    let err = invoke(&webview, "get_sample_detail", json!({ "sampleId": 999 })).unwrap_err();
    assert_eq!(err["kind"], json!("notFound"));
    assert_eq!(err["detail"], json!("sample 999"));

    // `play_sample` checks the database before it ever touches an audio device -- see
    // `audio::AudioPlayer::play` -- so a stale or made-up sample id fails the same way
    // `get_sample_detail` does, deterministically, on a machine with no audio hardware at all.
    let err = invoke(
        &webview,
        "play_sample",
        json!({ "sampleId": 1, "gain": 1.0 }),
    )
    .unwrap_err();
    assert_eq!(err["kind"], json!("notFound"));
    assert_eq!(err["detail"], json!("sample 1"));
}

/// Tags are real user work: the command must commit before it answers, so the list it
/// returns is the list a reader will see.
#[test]
fn tagging_answers_with_state_a_reader_can_already_observe() {
    let (_dir, webview) = app();
    let ids = seed(&webview, 3);

    let tags = json_of(
        invoke(
            &webview,
            "set_tag",
            json!({ "sampleId": ids[0], "tagName": "kick" }),
        )
        .unwrap(),
    );
    assert_eq!(tags, json!(["kick"]));

    let listed = json_of(invoke(&webview, "list_tags", json!({})).unwrap());
    assert_eq!(listed[0]["name"], json!("kick"));
    assert_eq!(listed[0]["sampleCount"], json!(1));

    let tags = json_of(
        invoke(
            &webview,
            "unset_tag",
            json!({ "sampleId": ids[0], "tagName": "kick" }),
        )
        .unwrap(),
    );
    assert_eq!(tags, json!([]));

    // The tag itself survives at zero: a tag the user invented is still a tag they invented.
    let listed = json_of(invoke(&webview, "list_tags", json!({})).unwrap());
    assert_eq!(listed[0]["sampleCount"], json!(0));
}

/// A durable write commits even when it fails, so an argument that cannot work has to be
/// refused before the writer sees it — otherwise `set_tag` on a bad id leaves behind a tag
/// the user never finished making.
#[test]
fn a_tag_on_an_unknown_sample_creates_nothing() {
    let (_dir, webview) = app();
    seed(&webview, 2);

    let err = invoke(
        &webview,
        "set_tag",
        json!({ "sampleId": 4242, "tagName": "kick" }),
    )
    .unwrap_err();
    assert_eq!(err["kind"], json!("notFound"));

    let listed = json_of(invoke(&webview, "list_tags", json!({})).unwrap());
    assert_eq!(listed, json!([]), "a failed tag left a row behind");
}

/// A well-typed argument whose value is unusable is its own error, not "not found".
#[test]
fn a_blank_name_is_an_invalid_argument_and_says_which_field() {
    let (_dir, webview) = app();
    seed(&webview, 2);

    let err = invoke(
        &webview,
        "create_collection",
        json!({ "name": "   ", "sampleIds": [] }),
    )
    .unwrap_err();
    assert_eq!(err["kind"], json!("invalidArgument"));
    assert_eq!(err["detail"]["field"], json!("name"));

    let err = invoke(&webview, "set_tag", json!({ "sampleId": 1, "tagName": "" })).unwrap_err();
    assert_eq!(err["kind"], json!("invalidArgument"));
    assert_eq!(err["detail"]["field"], json!("tagName"));
}

/// An unprojected library is a state, not a failure: a bare header and an empty scene.
#[test]
fn an_empty_library_answers_rather_than_failing() {
    let (_dir, webview) = app();

    let bytes = raw_of(invoke(&webview, "get_point_cloud", json!({})).unwrap());
    assert_eq!(bytes.len(), binary::HEADER_BYTES);
    assert_eq!(binary::header(&bytes).unwrap().count, 0);

    let roots = json_of(invoke(&webview, "list_library_roots", json!({})).unwrap());
    assert_eq!(roots, json!([]));
}

/// Cancelling something that is not running is not an error: the frontend cancelling a scan
/// that already finished is a race it cannot avoid and does not need to hear about.
#[test]
fn cancelling_nothing_is_not_an_error() {
    let (_dir, webview) = app();
    assert!(invoke(&webview, "cancel_scan", json!({ "scanId": 1 })).is_ok());
    assert!(invoke(&webview, "cancel_refit", json!({ "jobId": 1 })).is_ok());
}

/// A root has to exist before it can be scanned, and the refusal has to be the typed one.
#[test]
fn adding_a_root_that_is_not_a_directory_is_not_found() {
    let (_dir, webview) = app();
    let err = invoke(
        &webview,
        "add_library_root",
        json!({ "path": "/definitely/not/a/folder" }),
    )
    .unwrap_err();
    assert_eq!(err["kind"], json!("notFound"));
}

/// Every command in `overview.md` §6.1 is registered under the name the frontend calls.
///
/// A missing command does not fail to compile and does not fail any other test here -- it
/// fails at runtime, in front of a user, as "command not found". This is the list, checked.
#[test]
fn every_command_in_the_surface_is_reachable() {
    let (_dir, webview) = app();

    // Deliberately wrong arguments: what is under test is whether Tauri knows the name, and
    // "command not found" is a distinguishable failure from "bad arguments".
    for cmd in [
        "add_library_root",
        "list_library_roots",
        "remove_library_root",
        "set_root_enabled",
        "scan_library",
        "cancel_scan",
        "get_point_cloud",
        "get_point_colors",
        "get_feature_column",
        "query_samples",
        "get_sample_detail",
        "get_similar",
        "set_tag",
        "unset_tag",
        "list_tags",
        "set_tag_color",
        "create_collection",
        "list_collections",
        "get_collection",
        "reorder_collection",
        "delete_collection",
        "export_collection",
        "reveal_in_finder",
        "play_sample",
        "stop_playback",
        "start_refit",
        "cancel_refit",
        "get_settings",
        "list_audio_devices",
        "set_audio_device",
        "set_gain",
        // `reveal_data_dir` and `reset_database` are deliberately not invoked here. Every
        // other command in this list either takes an argument `{}` fails to deserialize, or
        // is a harmless no-arg read -- but both of these take no JSON arguments at all, so
        // `{}` would not be "deliberately wrong," it would be a real call: `reveal_data_dir`
        // would pop a Finder window and `reset_database` would spawn a copy of this test
        // binary and exit the process. Their registration is still checked at compile time,
        // by name, in `command_handler`'s `generate_handler!` list.
    ] {
        if let Err(err) = invoke(&webview, cmd, json!({})) {
            let message = err.to_string();
            assert!(
                !message.contains("not found"),
                "`{cmd}` is not registered: {message}"
            );
        }
    }
}
