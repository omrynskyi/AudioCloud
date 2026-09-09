//! Proves the guard allocator actually enforces "no allocation on the audio thread"
//! (`task.md` Phase 8, cross-cutting rule 5) rather than merely toggling a flag nobody reads.
//!
//! `audio::guard::GuardedAlloc` is only meaningful once installed as `#[global_allocator]`,
//! and a global allocator is process-wide -- it can be set exactly once per binary. The
//! library's own test binary cannot install it without conflicting with whatever allocator
//! `cargo test`'s harness already uses, so this is a separate integration test binary
//! (`tests/*.rs` files each compile to their own binary) that installs its own copy and
//! actually allocates from inside the guard.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use audiocloud_lib::audio::guard::{AudioThreadGuard, GuardedAlloc};

#[global_allocator]
static GUARD_ALLOC: GuardedAlloc = GuardedAlloc;

/// Allocating outside the guard is unaffected -- this whole test binary has been allocating
/// under `GuardedAlloc` since before `main` ran, and it has worked exactly like `System` the
/// entire time.
#[test]
fn allocation_off_the_audio_thread_is_unaffected() {
    let v: Vec<u8> = vec![1, 2, 3, 4, 5];
    assert_eq!(v.len(), 5);
    let s = String::from("no different from the system allocator");
    assert!(!s.is_empty());
}

/// The property the whole module exists for: an allocation *inside* an [`AudioThreadGuard`]
/// panics rather than silently succeeding.
#[test]
fn allocating_inside_the_guard_panics() {
    let result = std::panic::catch_unwind(|| {
        let _guard = AudioThreadGuard::enter();
        let leak: Vec<u8> = Vec::with_capacity(64);
        std::hint::black_box(leak);
    });
    assert!(result.is_err(), "an allocation inside the guard must panic");
}

/// A guard entered and dropped on one thread must not affect allocation on another -- the
/// enforcement is thread-local, matching that `cpal`'s data callback always runs on the same
/// one OS thread for a given stream and never on the caller's.
#[test]
fn the_guard_is_thread_local() {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let _guard = AudioThreadGuard::enter();
        // Allocation on *this* thread, while its own guard is active, still panics.
        let inner = std::panic::catch_unwind(|| {
            let v: Vec<u8> = Vec::with_capacity(8);
            std::hint::black_box(v);
        });
        tx.send(inner.is_err()).unwrap();
    });
    assert!(
        rx.recv().unwrap(),
        "the spawned thread's own guard should have caught its own allocation"
    );
    handle.join().unwrap();

    // Meanwhile the guard on the spawned thread never touched this one.
    let v: Vec<u8> = Vec::with_capacity(8);
    assert_eq!(v.capacity(), 8);
}

/// After the guard is dropped, allocation on the same thread works again -- the enforcement
/// is scoped to the callback's duration, not a one-way trip.
#[test]
fn allocation_resumes_once_the_guard_drops() {
    {
        let _guard = AudioThreadGuard::enter();
    }
    let v: Vec<u8> = Vec::with_capacity(16);
    assert_eq!(v.capacity(), 16);
}
