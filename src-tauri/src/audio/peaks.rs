//! Waveform peak-summary generation for the inspector display.
//!
//! A summary is `BUCKETS` pairs of `(min, max)` over the decoded window: the two numbers a
//! waveform view actually draws, one vertical line per bucket. Sending the samples
//! themselves would be 1.92 MB per selection for a picture 600 pixels wide.
//!
//! Summaries are computed once and cached, then served over the `abpeaks://` scheme rather
//! than through `invoke` -- see [`crate::protocol::peaks`] for why bulk assets stay off the
//! command channel.
//!
//! **The summary covers the decoded window, not the file.** `decode` stops at
//! `WINDOW_SECONDS`, because that is CLAP's window and decoding past it is work whose output
//! is discarded. A four-minute loop therefore summarizes to its first ten seconds, and the
//! payload says so: [`Summary::covered_ms`] rides in the frame header so the frontend can
//! mark where the waveform stops instead of drawing ten seconds edge to edge under a label
//! reading "4:07".

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use crate::{
    db::{queries, Database},
    error::AppError,
    ipc::binary,
    pipeline::{
        decode::{DecodeError, TARGET_SAMPLE_RATE},
        BufferPool, Decoder,
    },
};

/// Buckets per summary.
///
/// 1024 is a little over a physical pixel per bucket on a 2x inspector panel, which is the
/// resolution at which a min/max envelope stops being able to lie about a transient. It is
/// also 8 KB on the wire, so a cache of them is measured in megabytes rather than in
/// decisions.
pub const BUCKETS: usize = 1024;

/// Summaries held in memory at once.
///
/// 512 x 8 KB is about 4 MB against a 300 MB idle budget (`overview.md` §7). The bound
/// matters more than the number: a user dragging through a cluster selects hundreds of
/// samples a minute, and an unbounded cache of everything they ever hovered is a leak with
/// a friendly name.
pub const CACHE_ENTRIES: usize = 512;

/// A computed waveform summary.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// `(min, max)` per bucket, in `[-1, 1]`.
    pub buckets: Vec<(f32, f32)>,
    /// Milliseconds of audio the buckets span. Below the sample's real duration whenever
    /// the decoder stopped at its window.
    pub covered_ms: u32,
}

impl Summary {
    /// The `ABPK` payload for this summary.
    pub fn encode(&self) -> Vec<u8> {
        binary::peaks(&self.buckets, self.covered_ms)
    }
}

/// Reduces mono samples to `buckets` min/max pairs.
///
/// **Min and max, not RMS or absolute peak.** A waveform drawn from `abs().max()` is
/// symmetric about zero and loses the asymmetry of any real percussive hit; RMS loses the
/// transient entirely, which is the one thing anyone is looking at when they open a kick.
///
/// Buckets are laid out by integer division of the sample count, so the last bucket absorbs
/// the remainder rather than being short. A short final bucket is a visible notch at the
/// right edge of every waveform in the app, which is a lot of ugliness to buy nothing.
pub fn summarize(samples: &[f32], buckets: usize) -> Vec<(f32, f32)> {
    if samples.is_empty() || buckets == 0 {
        return vec![(0.0, 0.0); buckets];
    }

    let mut out = Vec::with_capacity(buckets);
    for b in 0..buckets {
        let start = b * samples.len() / buckets;
        let end = if b + 1 == buckets {
            samples.len()
        } else {
            ((b + 1) * samples.len() / buckets).max(start + 1)
        };

        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &s in &samples[start..end.min(samples.len())] {
            lo = lo.min(s);
            hi = hi.max(s);
        }
        // A bucket that saw no samples at all -- possible only when `buckets` exceeds the
        // sample count -- is silence, not infinity.
        out.push(if lo.is_finite() { (lo, hi) } else { (0.0, 0.0) });
    }
    out
}

/// The bounded summary cache, plus the decoder that fills it.
///
/// One `Decoder` behind one mutex rather than one per request: a `Decoder` owns a resampler
/// per source rate and a pool holding a 1.92 MB window, and building that per selection
/// would allocate two megabytes every time the user clicks a point. Summaries are requested
/// on selection -- a handful a second at worst -- so serializing them costs nothing a person
/// can perceive, and the mutex is what keeps the buffer pool's high-water mark at one.
#[derive(Debug)]
pub struct PeakCache {
    entries: Mutex<Entries>,
    decoder: Mutex<Decoder>,
}

/// FIFO with a cap, and named for what it is.
///
/// Not an LRU: a true LRU needs a touch on every read, which means taking a write lock on
/// the hot path to reorder a queue whose eviction decisions are, at 512 entries against a
/// selection rate a human generates, indistinguishable from FIFO's.
#[derive(Debug, Default)]
struct Entries {
    by_id: HashMap<i64, Arc<Vec<u8>>>,
    order: VecDeque<i64>,
}

impl Default for PeakCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Entries::default()),
            decoder: Mutex::new(Decoder::new(BufferPool::for_decode())),
        }
    }

    /// The encoded summary for one sample, decoding the file if it is not already cached.
    ///
    /// Returns an `Arc` so the protocol handler can hand the bytes to the WebView without
    /// copying them out of the cache.
    pub fn get(&self, db: &Database, sample_id: i64) -> Result<Arc<Vec<u8>>, AppError> {
        if let Some(hit) = self.lookup(sample_id) {
            return Ok(hit);
        }

        let conn = db.read()?;
        let row = queries::sample_row(&conn, sample_id)?
            .ok_or_else(|| AppError::not_found("sample", sample_id))?;
        drop(conn);

        let path = row.absolute_path();
        let decoded = {
            let mut decoder = self
                .decoder
                .lock()
                .map_err(|_| AppError::internal("locking the peak decoder", "poisoned"))?;
            decoder
                .decode(&path, &row.ext)
                .map_err(|e| decode_error(&row.rel_path, e))?
        };

        let covered_ms =
            (decoded.samples.len() as u64 * 1000 / u64::from(TARGET_SAMPLE_RATE)) as u32;
        let summary = Summary {
            buckets: summarize(&decoded.samples, BUCKETS),
            covered_ms,
        };
        drop(decoded);

        let bytes = Arc::new(summary.encode());
        self.insert(sample_id, Arc::clone(&bytes));
        Ok(bytes)
    }

    /// Drops every cached summary. What a scan calls when it has rewritten rows, since a
    /// rescanned file is different audio at the same sample id.
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.by_id.clear();
            entries.order.clear();
        }
    }

    fn lookup(&self, sample_id: i64) -> Option<Arc<Vec<u8>>> {
        self.entries.lock().ok()?.by_id.get(&sample_id).cloned()
    }

    fn insert(&self, sample_id: i64, bytes: Arc<Vec<u8>>) {
        let Ok(mut entries) = self.entries.lock() else {
            // A poisoned cache is a cache miss forever, which is slow and correct. It is
            // not a reason to fail a request that already has its answer in hand.
            return;
        };
        if entries.by_id.insert(sample_id, bytes).is_none() {
            entries.order.push_back(sample_id);
        }
        while entries.order.len() > CACHE_ENTRIES {
            if let Some(evicted) = entries.order.pop_front() {
                entries.by_id.remove(&evicted);
            }
        }
    }
}

/// A file that will not decode is a `decode` error with its path in it, not a 500.
///
/// The inspector renders this as "this file could not be read" next to the path, which is
/// the same state the quarantine list shows -- and it is reached by the same route, because
/// a sample whose peaks will not generate is a sample whose audio will not play either.
fn decode_error(rel_path: &str, e: DecodeError) -> AppError {
    AppError::Decode {
        path: rel_path.to_string(),
        reason: e.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn a_summary_keeps_the_shape_of_the_signal() {
        // A ramp from -1 to 1: every bucket's min and max should climb monotonically.
        let n = 4096;
        let ramp: Vec<f32> = (0..n).map(|i| -1.0 + 2.0 * i as f32 / n as f32).collect();
        let buckets = summarize(&ramp, 64);

        assert_eq!(buckets.len(), 64);
        assert!(buckets[0].0 < -0.9, "first bucket starts at the bottom");
        assert!(buckets[63].1 > 0.9, "last bucket reaches the top");
        for pair in buckets.windows(2) {
            assert!(pair[0].0 <= pair[1].0);
            assert!(pair[0].1 <= pair[1].1);
        }
    }

    /// Min and max, not an absolute envelope: an asymmetric transient has to survive.
    #[test]
    fn asymmetry_survives_the_reduction() {
        let mut samples = vec![0.0f32; 1000];
        samples[10] = 1.0;
        samples[900] = -0.5;

        let buckets = summarize(&samples, 10);
        assert_eq!(buckets[0], (0.0, 1.0));
        assert_eq!(buckets[9], (-0.5, 0.0));
    }

    /// The last bucket absorbs the remainder rather than coming up short, which is what
    /// keeps the right edge of every waveform in the app from having a notch in it.
    #[test]
    fn the_bucket_count_is_exact_for_any_length() {
        for len in [0, 1, 7, 1023, 1024, 1025, 48_000] {
            let samples = vec![0.25f32; len];
            assert_eq!(summarize(&samples, BUCKETS).len(), BUCKETS);
        }
    }

    #[test]
    fn more_buckets_than_samples_is_silence_rather_than_infinity() {
        let buckets = summarize(&[0.5, -0.5], 8);
        assert_eq!(buckets.len(), 8);
        assert!(buckets
            .iter()
            .all(|(lo, hi)| lo.is_finite() && hi.is_finite()));
    }

    #[test]
    fn the_cache_evicts_rather_than_growing() {
        let cache = PeakCache::new();
        for id in 0..(CACHE_ENTRIES as i64 + 10) {
            cache.insert(id, Arc::new(vec![0u8; 4]));
        }
        let entries = cache.entries.lock().unwrap();
        assert_eq!(entries.by_id.len(), CACHE_ENTRIES);
        assert_eq!(entries.order.len(), CACHE_ENTRIES);
        // FIFO: the oldest ids went first.
        assert!(!entries.by_id.contains_key(&0));
        assert!(entries.by_id.contains_key(&(CACHE_ENTRIES as i64 + 9)));
    }
}
