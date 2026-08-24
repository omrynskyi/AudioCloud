//! Lock-free SPSC ring buffer between the decode-ahead task and the audio callback.
//!
//! A mutex here would be a priority-inversion glitch waiting for a slow scheduler. The
//! producer is a `tokio` task; the consumer is the real-time callback, which may not
//! block, allocate, or lock.
//!
//! **Interleaved device-format `f32`, not mono.** What crosses this buffer is already at the
//! output device's sample rate and channel count -- [`crate::audio::mod::to_device_format`]
//! does that conversion off the real-time path, so the callback's only job is popping frames
//! and multiplying by gain (`overview.md` §2's audio thread row: copy and apply gain, nothing
//! else).
//!
//! `ringbuf::HeapRb` is a single allocation, made once when a stream (re)builds -- never on
//! the audio thread, which only ever holds the [`RingConsumer`] half.

use ringbuf::{traits::Split, HeapRb};

pub use ringbuf::traits::{Consumer, Observer, Producer};

/// The producer half. Lives on the control side: exactly one decode-ahead task pushes into it
/// at a time (`overview.md` §2), serialized through [`crate::audio::engine::Engine`]'s
/// producer lock so the "single producer" half of SPSC is an invariant the type system
/// doesn't have to enforce on its own.
pub type RingProducer = ringbuf::HeapProd<f32>;

/// The consumer half. Moved into the `cpal` data callback once and never shared -- the whole
/// point of SPSC is that nothing else ever needs to touch it.
pub type RingConsumer = ringbuf::HeapCons<f32>;

/// Seconds of device-format audio the ring can hold.
///
/// Small on purpose: this is not the buffer that makes decode look-ahead work -- the
/// decode-ahead task decodes the whole (≤10 s) clip before it starts pushing -- it is the
/// hand-off depth between a non-real-time producer and a real-time consumer. A quarter second
/// is enough slack to absorb scheduling jitter in the pusher task without holding megabytes of
/// audio nobody is about to play yet.
pub const RING_SECONDS: f32 = 0.25;

/// Frames below which the ring is not worth building, whatever the arithmetic above says --
/// a device reporting a nonsensical sample rate should not produce a zero-capacity `HeapRb`,
/// which `ringbuf` rejects outright.
const MIN_CAPACITY: usize = 1024;

/// Builds a fresh ring sized for `sample_rate` Hz, `channels`-wide interleaved frames.
///
/// Capacity is interleaved samples, not frames: `RING_SECONDS` of audio at this format is
/// `sample_rate * RING_SECONDS` frames, and each frame is `channels` samples wide.
pub fn ring(sample_rate: u32, channels: u16) -> (RingProducer, RingConsumer) {
    let frames = (sample_rate as f32 * RING_SECONDS) as usize;
    let capacity = (frames * channels.max(1) as usize).max(MIN_CAPACITY);
    HeapRb::<f32>::new(capacity).split()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn capacity_matches_the_requested_seconds_and_width() {
        let (prod, _cons) = ring(48_000, 2);
        // `HeapRb` may round capacity up internally; it must never round down below what was
        // asked for.
        assert!(prod.capacity().get() >= (48_000.0 * RING_SECONDS) as usize * 2);
    }

    #[test]
    fn a_degenerate_rate_still_produces_a_usable_ring() {
        let (mut prod, mut cons) = ring(0, 0);
        assert!(prod.capacity().get() >= MIN_CAPACITY);
        assert!(prod.try_push(1.0).is_ok());
        assert_eq!(cons.try_pop(), Some(1.0));
    }

    #[test]
    fn what_is_pushed_is_what_is_popped_in_order() {
        let (mut prod, mut cons) = ring(48_000, 1);
        let pushed = prod.push_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(pushed, 4);

        let mut out = [0.0f32; 4];
        let popped = cons.pop_slice(&mut out);
        assert_eq!(popped, 4);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn a_full_ring_reports_a_short_push_rather_than_blocking() {
        let (mut prod, _cons) = ring(0, 1); // MIN_CAPACITY-sized ring
        let huge = vec![0.0f32; MIN_CAPACITY * 4];
        let pushed = prod.push_slice(&huge);
        assert!(pushed <= MIN_CAPACITY);
    }

    #[test]
    fn clear_drops_pending_frames_without_yielding_them() {
        let (mut prod, mut cons) = ring(48_000, 1);
        prod.push_slice(&[1.0, 2.0, 3.0]);
        assert!(!cons.is_empty());
        cons.clear();
        assert!(cons.is_empty());
        assert_eq!(cons.try_pop(), None);
    }
}
