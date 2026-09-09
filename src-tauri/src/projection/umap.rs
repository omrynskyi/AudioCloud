//! UMAP via `annembed` over an `hnsw_rs` kNN graph.
//!
//! `annembed` is pinned to an exact version and **vendored into the repo**
//! (`overview.md` §10, risk 2) -- it is a small crate with a single maintainer and the
//! layout of the entire map depends on it. See `src-tauri/vendor/annembed/Cargo.toml`.
//!
//! The embedding matrix is read through `mmap`, never heap-loaded: a 50k x 512 f16 matrix
//! is 51 MB that the page cache is better at managing than the allocator.
//!
//! **This file is the only one in the crate permitted to name an `annembed` or `hnsw_rs`
//! type**, on the same grounds as `model/session.rs` and `ort` (cross-cutting rule 7). An
//! upgrade, or a decision to drop UMAP entirely, has one blast radius.
//!
//! ## Three things that are not what a Python user would expect
//!
//! **1. `min_dist` is a stand-in, not an equivalence.** Python UMAP fits `a` and `b` from
//! `min_dist` and `spread` and uses them in the embedded-space kernel. `annembed`'s model
//! is different: it derives its embedded scale from the *local* scale around each point,
//! modulated by `scale_rho`, and exposes no `min_dist` at all. So the parameter is kept on
//! [`UmapParams`] because it is the knob a user reaches for, and mapped monotonically onto
//! `scale_rho` by [`UmapParams::scale_rho`]. The default (0.1) maps to `annembed`'s own
//! default (1.0), so a default fit is upstream's default fit and nothing else. Higher
//! spreads the cloud, lower tightens it -- but the numbers do not transfer from a Python
//! notebook, and pretending they did would be worse than saying so.
//!
//! **2. The HNSW index owns its own copy of the vectors.** `hnsw_rs` stores what it is
//! given, so a 50k x 512 index is ~102 MB of f32 for the duration of the fit and there is
//! no version of this that is not. What the mmap buys is everything *around* that: rows
//! arrive from the page cache a [`INSERT_CHUNK`]-sized block at a time and the block is
//! dropped before the next one is read, so AudioBank never holds a second copy of the
//! matrix alongside the index. Peak is the index plus a chunk, not the index plus the
//! corpus.
//!
//! **3. `annembed` panics on a disconnected kNN graph, so it is never handed one.**
//! Its initialization runs a diffusion map -- an SVD of the graph Laplacian -- and on a
//! graph that falls into separate components that decomposition degenerates, producing an
//! initial embedding that is constant or NaN. `set_data_box` then divides by its own zero
//! maximum and trips a bare `assert!`. This is not hypothetical and not rare: a corpus of
//! six tight, well-separated clumps reproduces it better than half the time, and a real
//! library of five hundred near-identical 909 kicks next to a folder of vocal loops is
//! exactly that shape. Because the release profile sets `panic = "abort"`, `catch_unwind`
//! is not available as a backstop -- the only defence is to not create the condition. So
//! [`connected_components`] counts them before `embed()` is called and a graph in pieces
//! comes back as [`ProjectionError::Degenerate`], which
//! [`crate::projection::Refit::with_fallback`] turns into a PCA layout. `overview.md` §3.7
//! calls PCA "the permanent fallback if `annembed` fails"; this is the failure it meant.
//!
//! **4. The gradient descent is not interruptible.** `Embedder::embed()` is one call that
//! returns when it is done, exactly like `Session::run()` in Phase 3. Cancellation is
//! checked at every boundary around it -- before the index, during the inserts, before the
//! graph, before the descent -- so a cancelled re-fit stops within a chunk during the
//! expensive-but-chunked part and within one descent otherwise. `refit` discards the
//! shadow run either way, so a cancelled fit costs time and never correctness.

use annembed::fromhnsw::kgraph::{kgraph_from_hnsw_all, KGraph};
use annembed::prelude::{Embedder, EmbedderParams};
use hnsw_rs::prelude::{DistCosine, Hnsw};

use crate::{
    pipeline::CancellationToken,
    projection::{EmbeddingSet, Point3, ProjectionError, Projector},
};

/// Target dimensionality. Two: the renderer only ever shows a flat top-down map now (the
/// orbiting 3D view was dropped).
///
/// Measured three ways against this app's own library, under the tuned params
/// (`examples/retune_experiment.rs`, 5 repeats each): a native 3D fit hits 57.3%/80.0%
/// (hit-rate@5 / hit-rate@1, see `UmapParams::sharpness`'s doc for what that means); the same
/// 3D fit flattened to 2D client-side (the previous architecture) drops to 45.5%/68.9%; a
/// native 2D fit gets 47.8%/69.1% -- narrowly but consistently ahead of flattening, with no
/// overlap in the hit@5 ranges across repeats. Fitting in 3D and then discarding an axis buys
/// nothing here: it costs the full 3D fit and still lands at or below what fitting the plane
/// directly gets for less compute, so this fits in 2D and stops there.
const TARGET_DIM: usize = 2;

/// Rows read from the mmap before they are handed to the index and dropped.
///
/// 1024 x 512 f32 is 2 MB, which is the whole transient cost of feeding a 102 MB index.
/// Larger buys nothing (the insert is already parallel within a chunk); smaller starts to
/// pay `rayon`'s fork/join per handful of points.
const INSERT_CHUNK: usize = 1024;

/// `hnsw_rs` calls `std::process::exit(1)` -- not a panic, an *exit* -- if
/// `max_nb_connection` exceeds 256. Nothing in this file may let that happen.
const MAX_HNSW_CONNECTIONS: usize = 256;

/// Below this the kNN graph is mostly the corpus and UMAP has no neighborhood structure to
/// find. Fall back to PCA rather than producing a shape that means nothing.
const MIN_SAMPLES: usize = 32;

/// What a UMAP fit was asked for. Recorded verbatim in `projection_runs.params_json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UmapParams {
    /// Neighbors per point in the kNN graph. Lower = more local clusters, higher = more
    /// global shape (`overview.md` §3.7).
    pub n_neighbors: usize,
    /// How tightly points may pack. See the module note: a monotone stand-in for
    /// `annembed`'s `scale_rho`, not Python UMAP's `min_dist`.
    ///
    /// Defaults to `0.01`, well below `annembed`'s own neutral point -- deliberately at the
    /// floor [`UmapParams::scale_rho`] clamps to. Measured against this app's own library
    /// (one-shot percussion; `retune_experiment`, five repeats per config): dropping from the
    /// neutral `0.1` to `0.01` took "is the sample nearest a hover truly one of its 15 most
    /// similar" from 53% to 68%. A looser embedded scale gives the fit more room to spread
    /// dissimilar points apart instead of packing them into the same neighborhood by default.
    pub min_dist: f32,
    /// Gradient batches. `annembed`'s default is 20; the cost of the fit is close to linear
    /// in this.
    pub n_epochs: usize,
    /// Exponent of the embedded-space kernel (`annembed`'s `b`; Python UMAP derives the same
    /// exponent from `min_dist`/`spread`). `annembed`'s own default is `1.0`.
    ///
    /// **Counter-intuitive, so it is worth stating plainly: lower is tighter here, not
    /// higher.** The instinct going in was "punish weak matches harder, so only 90%+ cosine
    /// holds a point in place" -- which reads as *raise* the exponent. Measured against this
    /// app's own library, that was backwards: raising it steadily *hurt* neighbor fidelity
    /// (hit-rate fell from 53% at `1.0` to 24% at `8.0`), and lowering it steadily helped.
    /// `0.2` combined with [`Self::min_dist`]`=0.01` took "is the sample nearest a hover truly
    /// one of its 15 most similar" from 53% to 81%, and "are its 5 closest truly in that
    /// top-15" from 42% to 58% -- both averaged over 5 stochastic re-fits with non-overlapping
    /// ranges against the `1.0` baseline, not a lucky seed. A harder falloff apparently makes
    /// `annembed`'s gradient descent commit early to a coarse near/far split and stop
    /// discriminating within "far", which is exactly the muddled middle a hover most needs
    /// ordered correctly. See `examples/retune_experiment.rs` to re-run this against a copy of
    /// a real library if the corpus changes shape enough to matter.
    pub sharpness: f64,
    /// Exponent of the edge weight in the *original* (512-dim) kNN graph (`annembed`'s
    /// `beta`). Where `sharpness` reshapes the output kernel, this reshapes the input one --
    /// how much more a rank-1 neighbor counts than a rank-15 one before the fit even starts.
    /// Left at `annembed`'s own default: the same sweep found no value that beat `1.0` by
    /// more than run-to-run noise on this corpus.
    pub input_sharpness: f64,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            min_dist: 0.01,
            n_epochs: 20,
            sharpness: 0.2,
            input_sharpness: 1.0,
        }
    }
}

/// The `min_dist` at which [`UmapParams::scale_rho`] returns `annembed`'s own default.
const NEUTRAL_MIN_DIST: f32 = 0.1;

impl UmapParams {
    /// `annembed`'s `scale_rho`, from our `min_dist`.
    ///
    /// A ratio against [`NEUTRAL_MIN_DIST`], clamped: `annembed` documents that edge weights
    /// must stay above 1e-5 or its SVD runs into numerical trouble, and an unbounded
    /// `scale_rho` from a UI slider is exactly how that gets hit. The clamp is generous
    /// enough that both ends are visibly different layouts.
    fn scale_rho(&self) -> f64 {
        let ratio = f64::from(self.min_dist.max(0.0)) / f64::from(NEUTRAL_MIN_DIST);
        ratio.clamp(0.25, 4.0)
    }

    /// Neighbors, clamped so the graph is buildable against this corpus.
    ///
    /// `kgraph_from_hnsw_all` wants `n_neighbors <= max_nb_connection`, and no point can
    /// have more neighbors than there are other points.
    fn effective_neighbors(&self, n: usize) -> usize {
        self.n_neighbors
            .clamp(2, (n - 1).min(MAX_HNSW_CONNECTIONS / 2))
    }
}

/// UMAP over an HNSW cosine kNN graph.
#[derive(Debug, Clone, Copy, Default)]
pub struct UmapProjector {
    params: UmapParams,
}

impl UmapProjector {
    pub fn new(params: UmapParams) -> Self {
        Self { params }
    }

    pub fn params(&self) -> UmapParams {
        self.params
    }
}

impl Projector for UmapProjector {
    fn fit_transform(
        &self,
        data: &EmbeddingSet<'_>,
        cancel: &CancellationToken,
    ) -> Result<Vec<Point3>, ProjectionError> {
        let n = data.len();
        if n < MIN_SAMPLES {
            return Err(ProjectionError::TooFewSamples {
                algorithm: self.name(),
                have: n,
                need: MIN_SAMPLES,
            });
        }
        data.check_uniform_dimensions()?;
        EmbeddingSet::check_cancelled(cancel)?;

        let neighbors = self.params.effective_neighbors(n);
        let hnsw = build_index(data, neighbors, cancel)?;

        EmbeddingSet::check_cancelled(cancel)?;
        let kgraph: KGraph<f32> = kgraph_from_hnsw_all(&hnsw, neighbors)
            .map_err(|e| ProjectionError::Umap(format!("building the kNN graph: {e}")))?;
        if kgraph.get_nb_nodes() != n {
            // Every vector was inserted with a contiguous data id, so a node count that
            // disagrees means `get_embedded_reindexed` -- which indexes by data id -- is
            // about to write outside the rows it allocated, or leave rows at the origin.
            return Err(ProjectionError::Umap(format!(
                "the kNN graph has {} nodes for {n} vectors",
                kgraph.get_nb_nodes()
            )));
        }

        let components = connected_components(&kgraph);
        if components > 1 {
            return Err(ProjectionError::Degenerate(format!(
                "the {neighbors}-nearest-neighbour graph falls into {components} disconnected \
                 components, which annembed's diffusion-map initialization cannot embed; \
                 raise n_neighbors or use PCA"
            )));
        }

        let mut params = EmbedderParams::default();
        params.set_dim(TARGET_DIM);
        params.scale_rho = self.params.scale_rho();
        params.nb_grad_batch = self.params.n_epochs.max(1);
        params.b = self.params.sharpness;
        params.beta = self.params.input_sharpness;

        EmbeddingSet::check_cancelled(cancel)?;
        let mut embedder = Embedder::new(&kgraph, params);
        // `embed()`'s error type is `usize`, with no documented meaning beyond "not Ok".
        // It is reported rather than swallowed so a failure is at least attributable.
        embedder
            .embed()
            .map_err(|code| ProjectionError::Umap(format!("annembed returned error {code}")))?;
        EmbeddingSet::check_cancelled(cancel)?;

        // Reindexed: `annembed` permutes nodes internally, and this undoes the permutation
        // back to the data ids we inserted -- which are row indices. Getting this wrong is
        // the "one week bug" its own source comments about, and it would surface as a map
        // that looks perfectly plausible and is wired to the wrong files.
        let embedded = embedder.get_embedded_reindexed();
        let (rows, cols) = embedded.dim();
        if rows != n || cols != TARGET_DIM {
            return Err(ProjectionError::Umap(format!(
                "annembed produced a {rows} x {cols} layout for {n} vectors"
            )));
        }

        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let point = [embedded[[i, 0]], embedded[[i, 1]], 0.0];
            if point.iter().any(|v| !v.is_finite()) {
                return Err(ProjectionError::Umap(format!(
                    "row {i} of the layout is not finite: {point:?}"
                )));
            }
            points.push(point);
        }
        Ok(points)
    }

    fn name(&self) -> &'static str {
        "umap"
    }

    fn params_json(&self) -> String {
        format!(
            r#"{{"n_neighbors":{},"min_dist":{},"n_epochs":{},"metric":"cosine","scale_rho":{},"sharpness":{},"input_sharpness":{}}}"#,
            self.params.n_neighbors,
            self.params.min_dist,
            self.params.n_epochs,
            self.params.scale_rho(),
            self.params.sharpness,
            self.params.input_sharpness,
        )
    }
}

/// Builds the HNSW index, streaming rows out of the mmap a chunk at a time.
///
/// Data ids are row indices, `0..n`, contiguous -- which is what
/// `Embedder::get_embedded_reindexed` requires in order to undo `annembed`'s internal
/// permutation.
fn build_index<'a>(
    data: &EmbeddingSet<'_>,
    neighbors: usize,
    cancel: &CancellationToken,
) -> Result<Hnsw<'a, f32, DistCosine>, ProjectionError> {
    let n = data.len();

    // `max_nb_connection` must be at least the neighbor count the kNN graph will ask for,
    // or `kgraph_from_hnsw_all` silently returns short neighbor lists and the layout is
    // built on a graph nobody asked for. Doubling it is upstream's own guidance and keeps
    // the graph's mean degree at the asked-for number rather than below it.
    let max_connections = (neighbors * 2).clamp(2, MAX_HNSW_CONNECTIONS);
    // Layer count: log2(n) is the depth the structure actually uses; 16 is `hnsw_rs`'s own
    // ceiling.
    let layers = (n.ilog2() as usize).clamp(4, 16);
    let ef_construction = (max_connections * 2).max(64);

    let hnsw =
        Hnsw::<f32, DistCosine>::new(max_connections, n, layers, ef_construction, DistCosine {});
    // Without this, pruning during construction leaves points with fewer neighbors than
    // asked for, and `kgraph_from_hnsw_all` logs a warning nobody reads and builds a
    // deficient graph.
    let mut hnsw = hnsw;
    hnsw.set_keeping_pruned(true);

    let dim = data.dim();
    let mut chunk: Vec<Vec<f32>> = Vec::with_capacity(INSERT_CHUNK);
    let mut row = Vec::with_capacity(dim);

    for start in (0..n).step_by(INSERT_CHUNK) {
        EmbeddingSet::check_cancelled(cancel)?;
        let end = (start + INSERT_CHUNK).min(n);
        chunk.clear();
        for i in start..end {
            data.row_into(i, &mut row)?;
            chunk.push(row.clone());
        }
        let refs: Vec<(&Vec<f32>, usize)> = chunk.iter().zip(start..end).collect();
        hnsw.parallel_insert(&refs);
    }

    Ok(hnsw)
}

/// Number of connected components in the kNN graph, by union-find over its edge lists.
///
/// Treated as undirected: `kgraph_from_hnsw_all` stores each node's out-edges, and "a can
/// reach b" is what matters for whether the Laplacian has one zero eigenvalue or several.
/// A node with no edges at all is its own component, which is the case that matters most --
/// it is also the one a diffusion map handles worst.
///
/// Union by size with full path compression: near-linear, and the graph has 50,000 nodes and
/// 750,000 edges at the corpus size this is stated against.
fn connected_components(kgraph: &KGraph<f32>) -> usize {
    let n = kgraph.get_nb_nodes();
    if n == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let mut size = vec![1usize; n];

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    let mut components = n;
    for (node, edges) in kgraph.get_neighbours().iter().enumerate() {
        for edge in edges {
            let neighbour = edge.node;
            if neighbour >= n {
                continue;
            }
            let (a, b) = (find(&mut parent, node), find(&mut parent, neighbour));
            if a == b {
                continue;
            }
            let (big, small) = if size[a] >= size[b] { (a, b) } else { (b, a) };
            parent[small] = big;
            size[big] += size[small];
            components -= 1;
        }
    }
    components
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::projection::{
        distance,
        test_support::{clustered, projected, try_projected},
        BoundingBox,
    };

    /// The property UMAP is bought for, and the one PCA is worse at: points that are
    /// neighbors in 512 dimensions are neighbors in three.
    ///
    /// Stated as a ratio rather than an absolute distance because the layout has no
    /// canonical scale. Six clusters, 240 points; a same-cluster pair must land closer than
    /// a different-cluster pair by a wide margin.
    #[test]
    fn neighborhood_structure_survives_the_projection() {
        let vectors = clustered(240, 32, 6, 99);
        let points = projected(&UmapProjector::default(), &vectors);
        assert_eq!(points.len(), 240);

        // `clustered` assigns cluster `i % 6`, so 0, 6, 12 ... share one and 0, 1 do not.
        let mut same = Vec::new();
        let mut other = Vec::new();
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let d = distance(points[i], points[j]);
                if i % 6 == j % 6 {
                    same.push(d);
                } else {
                    other.push(d);
                }
            }
        }
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (within, between) = (mean(&same), mean(&other));
        assert!(
            between > within * 2.0,
            "within-cluster {within}, between-cluster {between}: the structure was lost"
        );
    }

    /// A layout is only useful if it has extent. A fit that collapsed every point onto one
    /// spot would pass a neighborhood test trivially and be worthless.
    #[test]
    fn the_layout_has_extent_and_is_finite() {
        let vectors = clustered(120, 16, 4, 7);
        let points = projected(&UmapProjector::default(), &vectors);

        let bbox = BoundingBox::of(&points).unwrap();
        assert!(bbox.diagonal() > 1e-3, "the cloud collapsed to a point");
        assert!(points.iter().all(|p| p.iter().all(|v| v.is_finite())));
    }

    /// `TARGET_DIM` is 2 -- the renderer only ever shows a flat map -- so `annembed` is asked
    /// for two columns and the third coordinate must be padded to exactly `0.0`, not left at
    /// whatever a stray write would leave it. `scene/buffers.ts` renders these coordinates
    /// with no client-side flatten step any more, so this is the only thing standing between
    /// a real fit and a point lifted off the plane.
    #[test]
    fn the_layout_is_genuinely_flat() {
        let vectors = clustered(120, 16, 4, 11);
        let points = projected(&UmapProjector::default(), &vectors);

        for p in &points {
            assert_eq!(
                p[2], 0.0,
                "a 2D fit must not write a third coordinate: {p:?}"
            );
        }
    }

    /// Cancellation is checked at the boundaries; a token already tripped must stop before
    /// the index is built, not after the descent.
    #[test]
    fn a_cancelled_fit_stops_rather_than_finishing() {
        let vectors = clustered(120, 16, 4, 3);
        let (_dir, store, rows) = crate::projection::test_support::store_of(&vectors);
        let matrix = store.matrix().unwrap();
        let data = EmbeddingSet::new(&matrix, &rows);

        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = UmapProjector::default()
            .fit_transform(&data, &cancel)
            .unwrap_err();

        assert!(matches!(err, ProjectionError::Cancelled), "got {err:?}");
    }

    /// A corpus too small for a kNN graph must say so by name rather than producing a
    /// layout built on a graph that is nearly complete.
    #[test]
    fn a_corpus_below_the_graph_floor_is_refused() {
        let vectors = clustered(10, 16, 2, 1);
        let err = try_projected(&UmapProjector::default(), &vectors).unwrap_err();
        assert!(matches!(
            err,
            ProjectionError::TooFewSamples {
                algorithm: "umap",
                have: 10,
                need: MIN_SAMPLES
            }
        ));
    }

    /// The `min_dist` mapping: the default is upstream's default, and the ends are clamped
    /// where `annembed`'s weight-range constraint says they must be.
    #[test]
    fn min_dist_maps_onto_scale_rho_monotonically_and_within_bounds() {
        let at = |min_dist: f32| {
            UmapParams {
                min_dist,
                ..Default::default()
            }
            .scale_rho()
        };

        assert!(
            (at(NEUTRAL_MIN_DIST) - 1.0).abs() < 1e-12,
            "the default must be annembed's"
        );
        assert!(at(0.05) < at(0.1) && at(0.1) < at(0.2));
        assert_eq!(at(0.0), 0.25);
        assert_eq!(at(1000.0), 4.0);
    }

    /// `hnsw_rs` exits the *process* on a connection count above 256, so the clamp is not a
    /// nicety. A user typing 400 into a neighbors field must get a layout, not a dead app.
    #[test]
    fn an_absurd_neighbor_count_is_clamped_below_the_hnsw_ceiling() {
        let params = UmapParams {
            n_neighbors: 4000,
            ..Default::default()
        };
        let neighbors = params.effective_neighbors(50_000);
        assert!(neighbors * 2 <= MAX_HNSW_CONNECTIONS);
        // And a corpus smaller than the asked-for neighborhood clamps to the corpus.
        assert_eq!(
            UmapParams {
                n_neighbors: 100,
                ..Default::default()
            }
            .effective_neighbors(40),
            39
        );
    }

    /// The recorded parameters have to describe the fit that actually ran, including the
    /// derived `scale_rho` -- otherwise a run fitted under one default cannot be told apart
    /// from one fitted under the next.
    #[test]
    fn the_recorded_parameters_include_what_was_derived() {
        let json = UmapProjector::default().params_json();
        assert!(json.contains(r#""n_neighbors":15"#), "{json}");
        assert!(json.contains(r#""metric":"cosine""#), "{json}");
        // The tuned defaults: min_dist=0.01 clamps scale_rho to its floor, and sharpness=0.2
        // is the measured-best output-kernel exponent -- see `UmapParams::sharpness`'s doc.
        assert!(json.contains(r#""scale_rho":0.25"#), "{json}");
        assert!(json.contains(r#""sharpness":0.2"#), "{json}");
    }
}
