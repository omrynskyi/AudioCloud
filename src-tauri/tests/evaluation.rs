//! Risk 5, measured rather than eyeballed (`overview.md` §10, `task.md` Phase 4).
//!
//! The open question is whether CLAP says anything useful about a 200 ms hi-hat. It is
//! outside the model's training distribution by a wide margin -- CLAP learned from
//! ten-second natural-audio and music clips paired with text -- and a sample library is
//! mostly quarter-second one-shots. `task.md` asks for a hand inspection: do kicks retrieve
//! kicks?
//!
//! A hand inspection is the right instinct and a weak instrument. A drum library names its
//! files `Kick - Frost.wav`, so the ground truth is sitting in the filenames, and that turns
//! the question into **precision@k**: of a kick's ten nearest neighbors, how many are kicks?
//! One number per category, comparable between embedders, reproducible by whoever doubts it.
//!
//! ```sh
//! AUDIOBANK_EVAL_LIBRARY=~/Documents/Music/sample\ packs \
//! AUDIOBANK_MODEL_PATH=../build/model/clap_audio.onnx \
//! AUDIOBANK_FORCE_CPU=1 \
//!   cargo test --profile perf --test evaluation -- --ignored --nocapture --test-threads=1
//! ```
//!
//! **What this does not measure.** Whether the *map* is good. Retrieval precision on a
//! labelled category is a proxy: it rewards an embedder for separating kicks from hats and
//! says nothing about whether the kicks are arranged interestingly among themselves. It is
//! the falsifiable half of a question whose other half is taste.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use audiobank_lib::{
    db::{queries, Database},
    model::session::ModelSession,
    pipeline::{
        dsp_embed::{DspEmbedder, DSP_EMBEDDING_DIM},
        mel::Padding,
        scan_root_with, BatchConfig, CancellationToken, Embed, ScanOptions,
    },
};

/// The library to evaluate against. Without it every test here skips: this needs a real
/// drum library and there is no way to synthesize one that would answer the question.
const LIBRARY_ENV: &str = "AUDIOBANK_EVAL_LIBRARY";
const MODEL_PATH_ENV: &str = "AUDIOBANK_MODEL_PATH";

/// Neighbors considered per query.
const K: usize = 10;

/// Categories below this many members are dropped: precision@10 over a category with three
/// members is noise with a decimal point.
const MIN_CATEGORY: usize = 8;

/// Filename keywords to category, in priority order.
///
/// Order is load-bearing -- `openhat` has to be tested before `hat`, and `808` before
/// anything, or the labels silently collapse into each other.
const LABELS: &[(&str, &str)] = &[
    ("808", "808"),
    ("openhat", "openhat"),
    ("open hat", "openhat"),
    ("hihat", "hihat"),
    ("hi-hat", "hihat"),
    ("hat", "hihat"),
    ("kick", "kick"),
    ("snare", "snare"),
    ("clap", "clap"),
    ("snap", "snap"),
    ("rim", "rim"),
    ("tom", "tom"),
    ("crash", "crash"),
    ("cymbal", "crash"),
    ("perc", "perc"),
    ("vox", "vox"),
    ("vocal", "vox"),
    ("bass", "bass"),
    ("synth", "melodic"),
    ("keys", "melodic"),
    ("pad", "melodic"),
    ("pluck", "melodic"),
    ("fx", "fx"),
];

/// The category a filename declares, if any.
fn label_of(rel_path: &str) -> Option<&'static str> {
    let name = rel_path
        .rsplit('/')
        .next()
        .unwrap_or(rel_path)
        .to_lowercase();
    LABELS
        .iter()
        .find(|(needle, _)| name.contains(needle))
        .map(|(_, label)| *label)
}

fn library() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(LIBRARY_ENV)?);
    if !path.is_dir() {
        println!(
            "SKIPPED: {LIBRARY_ENV} is not a directory: {}",
            path.display()
        );
        return None;
    }
    Some(path)
}

fn clap_session() -> Option<Arc<ModelSession>> {
    let path = PathBuf::from(std::env::var_os(MODEL_PATH_ENV)?);
    match ModelSession::open(&path) {
        Ok(session) => Some(Arc::new(session)),
        Err(e) => {
            println!("SKIPPED: could not open the model: {e}");
            None
        }
    }
}

/// One labelled, embedded sample.
struct Point {
    rel_path: String,
    label: &'static str,
    vector: Vec<f32>,
}

/// Scans `library` with `embedder` and returns every labelled sample's vector.
fn embed_library(
    library: &std::path::Path,
    embedder: Arc<dyn Embed>,
    padding: Padding,
    dim: usize,
) -> Vec<Point> {
    let data = tempfile::tempdir().unwrap();
    let db = Database::open(data.path(), dim).unwrap();
    let root = db
        .writer()
        .add_root(library.to_str().unwrap(), None)
        .unwrap();

    let cancel = CancellationToken::new();
    let started = std::time::Instant::now();
    let mut options = ScanOptions::new(&cancel)
        .with_embedder(embedder)
        .with_padding(padding)
        // Batch 1: the sweep showed throughput is flat in batch size against the real model
        // and only memory moves, so this is the setting that fits the budget.
        .with_batch(BatchConfig::of_size(1));
    let report = scan_root_with(&db, root, &mut options).unwrap();
    let elapsed = started.elapsed();

    println!(
        "  scanned {} files, embedded {}, in {:.1} s ({:.0} samples/s)",
        report.counts.files_seen,
        report.embedded,
        elapsed.as_secs_f64(),
        report.embedded as f64 / elapsed.as_secs_f64()
    );

    let conn = db.read().unwrap();
    let samples = queries::embedded_samples(&conn).unwrap();
    drop(conn);

    let store = db.embeddings().lock().unwrap();
    let mut points = Vec::new();
    for sample in samples {
        let Some(label) = label_of(&sample.rel_path) else {
            continue;
        };
        let Ok(vector) = store.read(sample.loc) else {
            continue;
        };
        points.push(Point {
            rel_path: sample.rel_path,
            label,
            vector,
        });
    }
    drop(store);
    db.shutdown();
    points
}

/// Cosine similarity. Every stored vector is L2-normalized, so this is a dot product.
fn similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Precision@K per category, plus the overall mean.
fn report_precision(points: &[Point]) -> f64 {
    let mut per_category: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for p in points {
        *counts.entry(p.label).or_default() += 1;
    }

    for (i, query) in points.iter().enumerate() {
        if counts[query.label] < MIN_CATEGORY {
            continue;
        }
        let mut scored: Vec<(f32, &str)> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, other)| (similarity(&query.vector, &other.vector), other.label))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        let hits = scored
            .iter()
            .take(K)
            .filter(|(_, label)| *label == query.label)
            .count();
        per_category
            .entry(query.label)
            .or_default()
            .push(hits as f64 / K as f64);
    }

    println!(
        "\n  {:<10} {:>7} {:>14} {:>10}",
        "category", "n", "precision@10", "chance"
    );
    let total = points.len() as f64;
    let mut weighted = 0.0;
    let mut n_total = 0usize;
    for (label, scores) in &per_category {
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let chance = (counts[label] - 1) as f64 / (total - 1.0);
        println!(
            "  {label:<10} {:>7} {:>13.1}% {:>9.1}%",
            counts[label],
            mean * 100.0,
            chance * 100.0
        );
        weighted += mean * scores.len() as f64;
        n_total += scores.len();
    }
    let overall = weighted / n_total as f64;
    println!(
        "  {:<10} {:>7} {:>13.1}%",
        "OVERALL",
        n_total,
        overall * 100.0
    );
    overall
}

/// Overall precision@K with no per-category table, for sweeping.
fn quiet_precision(points: &[Point]) -> f64 {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for p in points {
        *counts.entry(p.label).or_default() += 1;
    }

    let (mut total, mut n) = (0.0f64, 0usize);
    for (i, query) in points.iter().enumerate() {
        if counts[query.label] < MIN_CATEGORY {
            continue;
        }
        let mut scored: Vec<(f32, &str)> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, other)| (similarity(&query.vector, &other.vector), other.label))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let hits = scored
            .iter()
            .take(K)
            .filter(|(_, label)| *label == query.label)
            .count();
        total += hits as f64 / K as f64;
        n += 1;
    }
    total / n as f64
}

/// Prints the nearest neighbors of a few queries, because a number is not an inspection.
fn show_examples(points: &[Point], queries: &[&str]) {
    for needle in queries {
        let Some((i, query)) = points
            .iter()
            .enumerate()
            .find(|(_, p)| p.rel_path.to_lowercase().contains(needle))
        else {
            continue;
        };
        println!("\n  {} [{}]", query.rel_path, query.label);
        let mut scored: Vec<(f32, &Point)> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, other)| (similarity(&query.vector, &other.vector), other))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (score, neighbor) in scored.iter().take(6) {
            let name = neighbor.rel_path.rsplit('/').next().unwrap_or("");
            println!("      {score:.4}  {:<10} {name}", neighbor.label);
        }
    }
}

#[test]
#[ignore = "needs a real drum library; run under --profile perf"]
fn clap_neighbors_on_a_real_drum_library() {
    let (Some(library), Some(session)) = (library(), clap_session()) else {
        println!("SKIPPED: set {LIBRARY_ENV} and {MODEL_PATH_ENV}");
        return;
    };

    println!("\n== CLAP ==");
    println!("  provider {}", session.provider().as_str());
    let dim = session.embedding_dim();
    // RepeatPad: CLAP's own, and what the parity gate is stated against.
    let points = embed_library(&library, session, Padding::RepeatPad, dim);
    println!("  {} labelled samples", points.len());

    let overall = report_precision(&points);
    show_examples(&points, &["kick", "hihat", "snare", "808"]);

    println!("\n  CLAP precision@{K}: {:.1}%", overall * 100.0);
}

#[test]
#[ignore = "needs a real drum library; run under --profile perf"]
fn dsp_neighbors_on_a_real_drum_library() {
    let Some(library) = library() else {
        println!("SKIPPED: set {LIBRARY_ENV}");
        return;
    };

    println!("\n== DSP (no model) ==");
    // ZeroPad, because this embedder measures the envelope and repeat-padding a 200 ms kick
    // twenty-five times manufactures a decay the file does not have. `mel.rs` kept this
    // variant for exactly this evaluation.
    let points = embed_library(
        &library,
        Arc::new(DspEmbedder::new()),
        Padding::ZeroPad,
        DSP_EMBEDDING_DIM,
    );
    println!("  {} labelled samples", points.len());

    let overall = report_precision(&points);
    show_examples(&points, &["kick", "hihat", "snare", "808"]);

    println!("\n  DSP precision@{K}: {:.1}%", overall * 100.0);
}

/// Joins two embeddings of the same library by path, concatenating each pair into one
/// vector with `clap_weight` on the CLAP half.
///
/// Both halves arrive L2-normalized, so weighting then concatenating then re-normalizing is
/// a genuine interpolation between the two metrics: at weight 1.0 the cosine is exactly
/// CLAP's, at 0.0 exactly the DSP embedder's, and in between it is the blend
/// `overview.md` risk 5 proposes as the mitigation.
fn blend(clap: &[Point], dsp: &[Point], clap_weight: f32) -> Vec<Point> {
    let index: BTreeMap<&str, &Point> = dsp.iter().map(|p| (p.rel_path.as_str(), p)).collect();
    let mut out = Vec::with_capacity(clap.len());

    for c in clap {
        let Some(d) = index.get(c.rel_path.as_str()) else {
            continue;
        };
        let mut vector = Vec::with_capacity(c.vector.len() + d.vector.len());
        vector.extend(c.vector.iter().map(|x| x * clap_weight));
        vector.extend(d.vector.iter().map(|x| x * (1.0 - clap_weight)));

        let norm = vector
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt() as f32;
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
        out.push(Point {
            rel_path: c.rel_path.clone(),
            label: c.label,
            vector,
        });
    }
    out
}

/// The keep-or-blend decision, as a curve rather than an opinion.
///
/// `task.md` Phase 4 asks for a documented keep-or-blend decision on real percussive
/// samples. This is that decision's evidence: precision@10 across the whole interpolation
/// between the two metrics, so the answer is a number with a shape rather than a preference.
#[test]
#[ignore = "needs a real drum library; run under --profile perf"]
fn blending_clap_with_dsp_descriptors() {
    let (Some(library), Some(session)) = (library(), clap_session()) else {
        println!("SKIPPED: set {LIBRARY_ENV} and {MODEL_PATH_ENV}");
        return;
    };

    println!("\n== blend sweep ==");
    let dim = session.embedding_dim();
    let clap = embed_library(&library, session, Padding::RepeatPad, dim);
    let dsp = embed_library(
        &library,
        Arc::new(DspEmbedder::new()),
        Padding::ZeroPad,
        DSP_EMBEDDING_DIM,
    );

    println!("\n  {:>12} {:>14}", "clap weight", "precision@10");
    let mut best = (0.0f32, 0.0f64);
    let mut rows = Vec::new();
    for weight in [0.0f32, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0] {
        let blended = blend(&clap, &dsp, weight);
        let score = quiet_precision(&blended);
        println!("  {weight:>12.2} {:>13.1}%", score * 100.0);
        rows.push((weight, score));
        if score > best.1 {
            best = (weight, score);
        }
    }

    println!(
        "\n  best blend: {:.2} CLAP / {:.2} DSP at {:.1}%",
        best.0,
        1.0 - best.0,
        best.1 * 100.0
    );

    let pure_clap = rows.last().map(|r| r.1).unwrap_or(0.0);
    let pure_dsp = rows.first().map(|r| r.1).unwrap_or(0.0);
    println!(
        "  pure CLAP {:.1}%  |  pure DSP {:.1}%  |  blend gains {:+.1} points over CLAP",
        pure_clap * 100.0,
        pure_dsp * 100.0,
        (best.1 - pure_clap) * 100.0
    );

    let blended = blend(&clap, &dsp, best.0);
    report_precision(&blended);
    show_examples(&blended, &["hihat", "snare"]);
}

/// Does CLAP depend on the *content* of the padding, or only on the real audio?
///
/// A proxy for a question that otherwise needs a new export: whether the ten-second window
/// could be shortened. The model has only ever seen 1001-frame inputs filled by
/// `repeatpad`, so a shorter window is out of distribution and might be quietly
/// meaningless. This does not change the tensor -- it changes what fills it. Zero-padding
/// presents the same 1001 frames with the sample once and silence after, so the model sees
/// far less real content in the same shape.
///
/// If precision holds, CLAP is reading the sample rather than the tiling, and a shorter
/// window is worth the cost of an export. If it collapses, the window is load-bearing and
/// no amount of re-exporting will help.
///
/// **What it does not prove**: that a 301-frame graph would work. Fewer frames is a
/// different tensor, not just a differently-filled one. This is evidence, not the answer.
#[test]
#[ignore = "needs a real drum library; run under --profile perf"]
fn clap_with_an_emptier_window() {
    let (Some(library), Some(session)) = (library(), clap_session()) else {
        println!("SKIPPED: set {LIBRARY_ENV} and {MODEL_PATH_ENV}");
        return;
    };

    println!("\n== CLAP, zero-padded window ==");
    let dim = session.embedding_dim();
    let points = embed_library(&library, session, Padding::ZeroPad, dim);
    println!("  {} labelled samples", points.len());

    let overall = report_precision(&points);
    println!(
        "\n  CLAP zero-padded precision@{K}: {:.1}%  (repeat-padded was 57.5%)",
        overall * 100.0
    );
}
