//! Tauri command handlers, one module per command group (`overview.md` §6.1).
//!
//! These are deliberately thin: deserialize, call into a domain module, map the error.
//! Business logic in a `#[tauri::command]` function is logic that cannot be unit-tested
//! without an `AppHandle`.
//!
//! Populated in Phase 6.
