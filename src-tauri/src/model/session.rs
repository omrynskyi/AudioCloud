//! `ort` session lifecycle: execution-provider selection, warmup, and lazy init.
//!
//! **Every `ort` type is isolated behind this module** so that an rc-version upgrade
//! touches exactly one file -- `ort` is pinned to an exact release candidate and its API
//! is not yet stable.
//!
//! CoreML is attempted first with a verified CPU fallback; which EP actually bound is
//! logged, because "CoreML was requested" and "CoreML is running" are not the same claim.
//! A warmup run on a zero tensor pays the first-inference cost off the user's critical
//! path, and session construction stays off the cold-start path entirely
//! (`overview.md` §7).
//!
//! Populated in Phase 3.
