//! Audio decode: `symphonia` probe and streaming decode, `rubato` resample.
//!
//! Hard stop at 10 s of 48 kHz output -- a 20-minute stem must not decode in full to be
//! embedded. Downmix to mono happens during decode, not after. Output buffers come from a
//! pool; allocating the 1.92 MB buffer per file is the difference between flat and
//! sawtooth RSS across a scan (`overview.md` §3.6).
//!
//! Populated in Phase 2.
