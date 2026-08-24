//! The debug-only guard allocator (`task.md` Phase 8, cross-cutting rule 5).
//!
//! "No allocation on the audio thread" is aspirational until something panics when it is
//! violated. This is that something: a `#[global_allocator]` that checks a thread-local flag
//! before delegating to the system allocator, and panics instead of delegating when the flag
//! is set. [`AudioThreadGuard`] is what sets it, for exactly the duration of one `cpal` data
//! callback.
//!
//! **Installed in `main.rs`, not here, and only under `#[cfg(debug_assertions)]`.** A global
//! allocator is process-wide and can be set exactly once, so it belongs at the binary root.
//! Release builds pay nothing: `AudioThreadGuard` still toggles the flag (it is a `Cell` write,
//! immeasurably cheap), but nothing ever reads it, so the callback's shape does not change
//! between profiles -- there is exactly one code path to reason about, not a debug one and a
//! release one that quietly diverge.
//!
//! [`GuardedAlloc`] is not tested against the *installed* global allocator here: overriding
//! `#[global_allocator]` a second time inside `cargo test` would conflict with whatever the
//! test binary already uses. `tests/audio_guard.rs` is a separate integration test binary that
//! installs its own copy and proves the panic actually fires.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// Set for the lifetime of one real-time callback on the thread that owns it, and on no
    /// other thread -- `cpal`'s data callback always runs on the same OS thread for a given
    /// stream, so a thread-local is exactly the right scope: no atomic, no cross-thread
    /// visibility to reason about, just "is this call stack the one with the deadline".
    static AUDIO_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Marks the current thread as the audio thread for the lifetime of the guard.
///
/// An RAII guard rather than a manual set/clear pair so that a `?` or an early return inside
/// the callback body -- there should never be one, but "should never" is exactly what this
/// module exists to stop trusting -- cannot leave the flag stuck on and poison every
/// allocation for the rest of the process.
#[derive(Debug)]
pub struct AudioThreadGuard {
    previous: bool,
}

impl AudioThreadGuard {
    /// Enters the guarded region. Nests correctly (restores the previous value on drop rather
    /// than unconditionally clearing), though the real-time callback never nests one of these
    /// inside another.
    pub fn enter() -> Self {
        let previous = AUDIO_THREAD.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for AudioThreadGuard {
    fn drop(&mut self) {
        AUDIO_THREAD.with(|flag| flag.set(self.previous));
    }
}

/// Whether the current thread is inside an [`AudioThreadGuard`] right now.
fn on_audio_thread() -> bool {
    AUDIO_THREAD.with(Cell::get)
}

/// Reports a violation and tears the thread down.
///
/// **Clears the flag before panicking, and this is load-bearing, not cosmetic.** Unwinding a
/// panic boxes its payload, which is itself a heap allocation; triggering that while
/// `AUDIO_THREAD` is still `true` would re-enter this exact check from inside the panic
/// machinery's own bookkeeping, panic a second time before the first has finished unwinding,
/// and abort the process with "thread panicked while processing panic" instead of the message
/// below. Clearing the flag first means only the violation itself is checked -- the cleanup
/// that follows it is not held to the same rule, which is correct: it is the same accommodation
/// [`AudioThreadGuard`]'s own `Drop` gets.
#[allow(clippy::panic)]
fn trip(what: &str, bytes: usize) -> ! {
    AUDIO_THREAD.with(|flag| flag.set(false));
    panic!("{what} {bytes} bytes on the audio thread");
}

/// Delegates to [`System`], except from inside an [`AudioThreadGuard`], where it panics.
///
/// Only meaningful once installed as `#[global_allocator]` -- see the module docs for why that
/// happens in `main.rs` and only in debug builds.
#[derive(Debug, Default)]
pub struct GuardedAlloc;

/// Deliberate exception to `clippy::panic`: panicking is this type's entire purpose, not a
/// failure to handle a case. A silent allocation on the audio thread is the bug this exists to
/// turn into a loud one.
///
/// SAFETY: every method delegates to `System`, whose implementation of `GlobalAlloc` is
/// already sound; this wrapper adds a check and nothing else, so it inherits `System`'s safety
/// obligations unchanged. `allow(unsafe_code)` covers the whole impl for the same reason
/// `db::embeddings` allows it at its one `unsafe` site: this is the one file that is allowed
/// to need this, not a blanket exemption for the crate.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for GuardedAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if on_audio_thread() {
            trip("allocated", layout.size());
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if on_audio_thread() {
            trip("deallocated", layout.size());
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if on_audio_thread() {
            trip("reallocated", new_size);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if on_audio_thread() {
            trip("allocated (zeroed)", layout.size());
        }
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_is_only_set_inside_the_guard() {
        assert!(!on_audio_thread());
        {
            let _guard = AudioThreadGuard::enter();
            assert!(on_audio_thread());
        }
        assert!(!on_audio_thread());
    }

    /// A guard that returns early -- the shape of a real callback that hit an error path --
    /// still clears the flag, because it is `Drop`, not a paired call the early return could
    /// skip.
    #[test]
    fn the_flag_clears_even_on_an_early_return() {
        fn enters_and_bails() -> bool {
            let _guard = AudioThreadGuard::enter();
            if true {
                return on_audio_thread();
            }
            #[allow(unreachable_code)]
            false
        }
        assert!(enters_and_bails());
        assert!(!on_audio_thread());
    }

    /// Nested guards restore the outer state rather than clearing unconditionally -- not a
    /// shape the real callback produces, but the RAII implementation should not assume it
    /// never will.
    #[test]
    fn nested_guards_restore_rather_than_clear() {
        let outer = AudioThreadGuard::enter();
        assert!(on_audio_thread());
        {
            let inner = AudioThreadGuard::enter();
            assert!(on_audio_thread());
            drop(inner);
        }
        assert!(
            on_audio_thread(),
            "dropping the inner guard cleared the outer one's flag"
        );
        drop(outer);
        assert!(!on_audio_thread());
    }

    /// This is the property the whole module exists for, and it is deliberately *not* proven
    /// here: proving it needs `GuardedAlloc` actually installed as `#[global_allocator]`,
    /// which conflicts with the allocator `cargo test`'s own binary already uses. See
    /// `tests/audio_guard.rs`.
    #[test]
    fn see_tests_audio_guard_rs_for_the_enforcement_proof() {}
}
