//! Progress reporting: atomic counters per stage, drained by a single 100 ms ticker.
//!
//! Cross-cutting rule 3: no per-file IPC events. 50,000 files emitting individually would
//! saturate the WebView message port and stall the render loop; the counters are cheap and
//! the ticker coalesces them to <= 10 Hz (`overview.md` §6.5).
//!
//! Populated in Phase 4.
