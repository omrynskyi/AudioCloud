//! Tauri command handlers, one module per command group (`overview.md` §6.1).
//!
//! These are deliberately thin: deserialize, call into a domain module, map the error.
//! Business logic in a `#[tauri::command]` function is logic that cannot be unit-tested
//! without an `AppHandle`.
//!
//! The real command surface is Phase 6. What is here now is [`dev`], the temporary trigger
//! Phase 2 needs to point the pipeline at a folder.

pub mod dev;
