//! Times the audio-preview **cold** path against a real library -- the row lookup and the
//! decode-plus-resample behind every cache miss -- so that "hover feels slow" can be answered
//! with a distribution instead of a guess. Not part of the app; run against a **copy** of a
//! real data directory (never the live one -- `Database::open` migrates, read-write):
//!
//! ```text
//! cp -r "$HOME/Library/Application Support/com.audiobank.app" /tmp/audiobank-copy
//! cargo run --release --example preview_latency -- /tmp/audiobank-copy
//! ```
//!
//! This produced the decode numbers in `BENCHMARKS.md`'s Phase 8 section. What it deliberately
//! does *not* measure is the warm path (a `PcmCache` hit, which is a `HashMap` lookup) or the
//! engine's own play-to-audible time -- `tests/audio_hardware.rs` owns that half, against real
//! hardware, because it is the half that needs a device.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::{path::PathBuf, time::Instant};

use audiobank_lib::{
    db::{queries, Database},
    pipeline::{decode::TARGET_SAMPLE_RATE, BufferPool, Decoder},
};

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: <data dir>"));
    let db = Database::open(&dir, audiobank_lib::EMBEDDING_DIM).expect("open db");
    let conn = db.read().unwrap();
    let ids: Vec<i64> = conn
        .prepare("select id from samples order by id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    drop(conn);

    let mut decoder = Decoder::new(BufferPool::for_decode());
    let mut db_us = Vec::new();
    let mut dec_ms = Vec::new();
    let n = ids.len().min(200);
    for &id in ids.iter().take(n) {
        let t0 = Instant::now();
        let conn = db.read().unwrap();
        let row = queries::sample_row(&conn, id).unwrap().unwrap();
        drop(conn);
        db_us.push(t0.elapsed().as_secs_f64() * 1000.0);

        let path = row.absolute_path();
        let t1 = Instant::now();
        match decoder.decode(&path, &row.ext) {
            Ok(d) => {
                let ms = t1.elapsed().as_secs_f64() * 1000.0;
                dec_ms.push((ms, d.samples.len(), row.sample_rate.unwrap_or(0)));
            }
            Err(e) => println!("decode failed {}: {e}", row.rel_path),
        }
    }

    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 - 1.0) * p) as usize]
    };
    let d: Vec<f64> = dec_ms.iter().map(|x| x.0).collect();
    println!("samples measured: {}", dec_ms.len());
    println!(
        "db row lookup  ms: p50 {:.3}  p95 {:.3}  max {:.3}",
        pct(&mut db_us.clone(), 0.5),
        pct(&mut db_us.clone(), 0.95),
        pct(&mut db_us.clone(), 1.0)
    );
    println!(
        "decode+resample ms: p50 {:.2}  p90 {:.2}  p95 {:.2}  max {:.2}  mean {:.2}",
        pct(&mut d.clone(), 0.5),
        pct(&mut d.clone(), 0.9),
        pct(&mut d.clone(), 0.95),
        pct(&mut d.clone(), 1.0),
        d.iter().sum::<f64>() / d.len() as f64
    );
    let total_secs: f64 = dec_ms
        .iter()
        .map(|x| x.1 as f64 / TARGET_SAMPLE_RATE as f64)
        .sum();
    println!(
        "audio decoded: {:.1}s over {:.1}ms -> {:.0}x realtime",
        total_secs,
        d.iter().sum::<f64>(),
        total_secs * 1000.0 / d.iter().sum::<f64>()
    );
    let mut at44: Vec<f64> = dec_ms
        .iter()
        .filter(|x| x.2 == 44100)
        .map(|x| x.0)
        .collect();
    let mut at48: Vec<f64> = dec_ms
        .iter()
        .filter(|x| x.2 == 48000)
        .map(|x| x.0)
        .collect();
    if !at44.is_empty() {
        println!(
            "  44.1k (resampled) p50 {:.2}  n={}",
            pct(&mut at44, 0.5),
            at44.len()
        );
    }
    if !at48.is_empty() {
        println!(
            "  48k (no resample) p50 {:.2}  n={}",
            pct(&mut at48, 0.5),
            at48.len()
        );
    }
}
