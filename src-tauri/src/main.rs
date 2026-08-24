// Prevent an extra console window on Windows in release. Harmless on macOS.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// The debug-only guard allocator (`audio::guard`, `task.md` Phase 8, cross-cutting rule 5):
// panics if the audio thread allocates, rather than trusting the callback body never does.
// Global allocators are process-wide and can be installed exactly once, which is why this
// lives at the binary root rather than in the library crate the test targets also link.
#[cfg(debug_assertions)]
#[global_allocator]
static GUARD_ALLOC: audiobank_lib::audio::guard::GuardedAlloc =
    audiobank_lib::audio::guard::GuardedAlloc;

fn main() {
    audiobank_lib::run();
}
