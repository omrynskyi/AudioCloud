//! Measures how `UmapParams` fields affect on-screen neighbor fidelity against a real
//! library, rather than a synthetic corpus. Not part of the app; run against a **copy** of a
//! real data directory (never the live one -- this opens it read-write):
//!
//! ```text
//! cp -r "$(...)/com.audiobank.app" /tmp/audiobank-copy
//! cargo run --release --example retune_experiment -- /tmp/audiobank-copy
//! ```
//!
//! This produced the numbers in `UmapParams::min_dist`, `UmapParams::sharpness`, and
//! `TARGET_DIM`'s doc comments (an earlier version of this file also settled the
//! native-2D-vs-flatten-3D-to-2D question those cite -- see git history if that needs
//! re-litigating). The current version compares tuned UMAP against `TsneProjector`, both
//! over the same real CLAP embeddings, at a few `TsneParams`. Re-run it if the corpus grows
//! or changes shape enough that a past conclusion stops being the one the data recommends.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use audiobank_lib::{
    db::{queries, Database, EmbeddingLoc},
    pipeline::CancellationToken,
    projection::{
        EmbeddingSet, Point3, Projector, TsneParams, TsneProjector, UmapParams, UmapProjector,
    },
};

const DIM: usize = 512;

fn main() {
    let dir = std::env::args().nth(1).expect("path to a copied data dir");
    let db = Database::open(&dir, DIM).expect("open db");

    let conn = db.read().expect("read conn");
    let rows: Vec<(i64, EmbeddingLoc)> = queries::all_embedding_locs(&conn).expect("locs");
    drop(conn);
    println!("rows: {}", rows.len());

    let matrix = {
        let store = db.embeddings().lock().unwrap();
        store.matrix().expect("matrix")
    };

    // Distinct vectors only, exactly like `refit::group_by_vector`.
    let mut seen = std::collections::HashMap::new();
    let mut distinct: Vec<(i64, EmbeddingLoc)> = Vec::new();
    for (id, loc) in &rows {
        seen.entry((loc.offset, loc.dims)).or_insert_with(|| {
            distinct.push((*id, *loc));
            true
        });
    }
    println!("distinct vectors: {}", distinct.len());

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(distinct.len());
    for (_, loc) in &distinct {
        vectors.push(matrix.row(*loc).expect("row"));
    }
    let n = vectors.len();

    // True cosine similarity (vectors are L2-normalized on ingest) between every pair.
    let mut sim = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s: f32 = vectors[i].iter().zip(&vectors[j]).map(|(a, b)| a * b).sum();
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }

    let top_k = 15usize;
    let mut top_sim: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        idx.sort_by(|&a, &b| sim[i * n + b].total_cmp(&sim[i * n + a]));
        idx.truncate(top_k);
        top_sim.push(idx);
    }

    let hit_rate = |points: &[Point3], spatial_k: usize| -> f64 {
        let mut hits = 0usize;
        let mut total = 0usize;
        for i in 0..n {
            let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            idx.sort_by(|&a, &b| dist(points[i], points[a]).total_cmp(&dist(points[i], points[b])));
            let top_sim_set: std::collections::HashSet<usize> =
                top_sim[i].iter().copied().collect();
            for &j in idx.iter().take(spatial_k) {
                total += 1;
                if top_sim_set.contains(&j) {
                    hits += 1;
                }
            }
        }
        hits as f64 / total as f64
    };

    let data = EmbeddingSet::new(&matrix, &distinct);
    let cancel = CancellationToken::new();

    // Both `annembed` and `bhtsne` are stochastic (random init, gradient sampling), so one
    // run per config is not trustworthy when candidates are within a few points of each
    // other -- repeat each and report the range, not a single number that might just be a
    // lucky seed.
    const REPEATS: usize = 5;
    let run = |label: &str, projector: &dyn Projector| {
        let mut h5 = Vec::with_capacity(REPEATS);
        let mut h1 = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            match projector.fit_transform(&data, &cancel) {
                Ok(points) => {
                    h5.push(hit_rate(&points, 5));
                    h1.push(hit_rate(&points, 1));
                }
                Err(e) => {
                    println!("{label:<45}  FAILED: {e}");
                    return;
                }
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let fmt = |v: &[f64]| {
            format!(
                "mean={:.1}% [{:.1}-{:.1}]",
                mean(v) * 100.0,
                min(v) * 100.0,
                max(v) * 100.0
            )
        };
        println!("{label:<45}  hit@5: {}   hit@1: {}", fmt(&h5), fmt(&h1));
    };

    println!("-- CLAP embeddings: UMAP (tuned) vs Barnes-Hut t-SNE, {REPEATS} runs each --");
    run(
        "UMAP (current tuned defaults)",
        &UmapProjector::new(UmapParams::default()),
    );
    for perplexity in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0] {
        run(
            &format!("t-SNE (perplexity={perplexity}, theta=0.5)"),
            &TsneProjector::new(TsneParams {
                perplexity,
                ..Default::default()
            }),
        );
    }
}

fn dist(a: Point3, b: Point3) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
