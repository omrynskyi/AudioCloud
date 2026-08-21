//! Real-time audio preview.
//!
//! Cross-cutting rule 5: **no allocation on the audio thread**, enforced by a debug-only
//! guard allocator that panics rather than by good intentions. The callback copies from a
//! lock-free ring and applies gain; everything else -- decode, cache, envelope scheduling
//! -- happens off it.
//!
//! Populated in Phase 8.

pub mod engine;
pub mod peaks;
pub mod ring;
