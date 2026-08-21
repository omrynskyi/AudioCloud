//! `cpal` output stream and device lifecycle.
//!
//! Device change and sample-rate change are normal events, not errors: unplugging an
//! interface mid-audition must not panic the audio thread or take the app with it.
//!
//! Populated in Phase 8.
