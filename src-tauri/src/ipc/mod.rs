//! The IPC wire: binary framing, event payloads, and the DTOs the JSON commands return
//! (`overview.md` §6).
//!
//! **Why this is its own module and not part of `commands/`.** `overview.md` §9 puts the
//! command handlers in `commands/` and says they should be thin -- deserialize, call a
//! domain module, map the error. A byte layout is not thin, it is a contract with a second
//! codebase, and it has to be testable without an `AppHandle`. Everything here is plain
//! functions and plain structs over plain data; `commands/` wraps the results in
//! `tauri::ipc::Response` and nothing else.
//!
//! The three transports of §6 land in three places: JSON goes through [`types`], raw bytes
//! through [`binary`], and the `abpeaks://` scheme through [`crate::protocol::peaks`].
//! Progress channels carry [`events`].

pub mod binary;
pub mod events;
pub mod types;

/// Where `ts-rs` writes the generated TypeScript, relative to `src-tauri/`.
///
/// Named once so the export test and the CI check cannot drift apart. Every IPC type
/// carries `#[ts(export_to = "<Name>.ts")]`, which is relative to this.
pub const BINDINGS_DIR: &str = "../src/bindings";
