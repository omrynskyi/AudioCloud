//! The full re-fit job: read, project, align, write a shadow run, swap it in.
//!
//! Everything here is the same for every [`Projector`], which is why none of it is on the
//! trait. A projector turns vectors into coordinates; this decides what those coordinates
//! mean for a database that already has a map the user is looking at.
//!
//! **The shape of the job** (`overview.md` §3.8):
//!
//! 1. Read every embedded sample's location, in file order, and map `embeddings.bin`.
//! 2. Project. This is the long part, and the only part whose cost depends on the algorithm.
//! 3. Align the new layout onto the active one by Procrustes over the samples present in
//!    both -- rotation, reflection, uniform scale -- and apply the transform to *all* of the
//!    new layout, new points included.
//! 4. Write the coordinates into a **shadow** `projection_runs` row: created, not active,
//!    `completed_at` still NULL.
//! 5. Swap: one `BEGIN IMMEDIATE`, clear the old flag, set the new one, commit. A reader
//!    sees the whole old map or the whole new one and never a mixture.
//!
//! **A cancelled or failed re-fit changes nothing.** The shadow run is deleted and the
//! active one is untouched, which is what makes step 2 -- minutes of work at low priority --
//! safe to abandon at any moment. That is the whole reason the shadow row exists rather
//! than writing into the live run and hoping.
//!
//! **Rows that share a vector are projected once.** Phase 4's dedup stores one copy of a
//! duplicate file's embedding and points both rows at it, so `all_embedding_locs` can return
//! two rows with the same `emb_offset`. Handing both to a projector would be wrong twice
//! over: it pays for the same vector repeatedly, and it feeds an exactly-zero distance into
//! a kNN graph, which is the degenerate input `annembed` handles worst. So the fit runs over
//! *distinct vectors* and the coordinate is then fanned out to every row that shares one,
//! separated by the same deterministic sub-pixel jitter the incremental path uses -- because
//! two rows on one pixel are one unclickable dot rather than two samples. On a library that
//! is 40% duplicates this is also 40% off the cost of every re-fit.

use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{
        atomic::{AtomicU64, AtomicU8, Ordering},
        Arc,
    },
    time::Instant,
};

use crate::{
    db::{queries, Database, EmbeddingLoc},
    pipeline::{progress::Ticker, CancellationToken},
    projection::{
        distance, procrustes, Alignment, BoundingBox, EmbeddingSet, Point3, ProjectionError,
        Projector,
    },
};

/// Coordinates handed to the writer per command.
///
/// Matched to `db::writer::BATCH_ROWS` so one command fills exactly one of the writer's
/// transactions -- the same reasoning as `pipeline::PERSIST_CHUNK`, for the same reason.
const WRITE_CHUNK: usize = 1000;

/// New samples below this fraction of the corpus get incremental placement instead of a
/// re-fit (`overview.md` §3.8).
///
/// "~2%" in the design document. The point of the threshold is not the number: it is that
/// importing twelve files must not rotate a library, and that the drift incremental
/// placement accumulates has to be paid off eventually.
pub const INCREMENTAL_THRESHOLD: f64 = 0.02;

/// Which path an import should take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// Nothing new to place; the active map already covers every embedded sample.
    UpToDate,
    /// Place `new` points into the existing layout without moving anything.
    Incremental { new: usize, total: usize },
    /// Re-fit everything. Either the import is large, or there is no active map yet.
    Full { new: usize, total: usize },
}

/// Decides between incremental placement and a full re-fit.
///
/// Reads two counts, not two corpora: this is called after every scan and must not be a
/// reason to touch `embeddings.bin`.
pub fn plan(db: &Database) -> Result<Plan, ProjectionError> {
    let conn = db.read()?;
    let total = queries::count_embedded_samples(&conn)? as usize;
    let Some(active) = queries::active_projection_run(&conn)? else {
        return Ok(Plan::Full { new: total, total });
    };
    let placed = queries::count_projection_points(&conn, active.id)? as usize;
    let new = total.saturating_sub(placed);

    Ok(if new == 0 {
        Plan::UpToDate
    } else if total > 0 && (new as f64 / total as f64) < INCREMENTAL_THRESHOLD {
        Plan::Incremental { new, total }
    } else {
        Plan::Full { new, total }
    })
}

/// Which stage a re-fit is in, for a label the UI can show.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum RefitPhase {
    /// Reading locations and mapping the matrix.
    Reading = 0,
    /// Inside [`Projector::fit_transform`]. The long one, and the opaque one.
    Fitting = 1,
    /// Procrustes against the active layout.
    Aligning = 2,
    /// Writing the shadow run's coordinates.
    Writing = 3,
    /// The swap, and then done.
    Swapping = 4,
    Done = 5,
}

impl RefitPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            RefitPhase::Reading => "reading",
            RefitPhase::Fitting => "fitting",
            RefitPhase::Aligning => "aligning",
            RefitPhase::Writing => "writing",
            RefitPhase::Swapping => "swapping",
            RefitPhase::Done => "done",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => RefitPhase::Fitting,
            2 => RefitPhase::Aligning,
            3 => RefitPhase::Writing,
            4 => RefitPhase::Swapping,
            5 => RefitPhase::Done,
            _ => RefitPhase::Reading,
        }
    }
}

/// Counters for one re-fit.
///
/// `Relaxed` throughout, on the same grounds as [`crate::pipeline::ScanProgress`]: these
/// publish no other memory and a reader that is one point behind has a progress bar that is
/// one point behind.
#[derive(Debug, Default)]
pub struct RefitProgress {
    phase: AtomicU8,
    /// Samples in the re-fit. Known once the locations are read.
    pub samples: AtomicU64,
    /// Coordinates committed to the shadow run.
    pub written: AtomicU64,
}

impl RefitProgress {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances the reported phase, never rewinding it.
    pub fn enter(&self, phase: RefitPhase) {
        self.phase.fetch_max(phase as u8, Ordering::Relaxed);
    }

    pub fn phase(&self) -> RefitPhase {
        RefitPhase::from_u8(self.phase.load(Ordering::Relaxed))
    }

    pub fn snapshot(&self) -> RefitSnapshot {
        RefitSnapshot {
            phase: self.phase(),
            samples: self.samples.load(Ordering::Relaxed),
            written: self.written.load(Ordering::Relaxed),
        }
    }
}

/// What one tick of a re-fit reports.
///
/// `ts-rs` derives and the `Channel<RefitProgress>` that carries this are Phase 6; the shape
/// is chosen so that phase adds derives rather than a translation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefitSnapshot {
    pub phase: RefitPhase,
    pub samples: u64,
    pub written: u64,
}

impl RefitSnapshot {
    /// Whether this is the last snapshot of the job.
    pub fn is_terminal(&self) -> bool {
        self.phase == RefitPhase::Done
    }
}

/// How one re-fit should run. A struct rather than four positional arguments, matching
/// [`crate::pipeline::ScanOptions`].
/// A second, independent fit that colors points -- see [`Refit::with_color_fit`].
#[allow(clippy::type_complexity)]
pub type ColorFit<'a> = &'a (dyn Fn(&EmbeddingSet<'_>, &CancellationToken) -> Result<Vec<[f32; 3]>, ProjectionError>
         + Sync);

pub struct Refit<'a> {
    projector: &'a dyn Projector,
    fallback: Option<&'a dyn Projector>,
    color_fit: Option<ColorFit<'a>>,
    cancel: &'a CancellationToken,
    align: bool,
    progress: Arc<RefitProgress>,
    #[allow(clippy::type_complexity)]
    sink: Option<Box<dyn FnMut(RefitSnapshot) + Send + 'static>>,
}

impl std::fmt::Debug for Refit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Refit")
            .field("projector", &self.projector.name())
            .field("fallback", &self.fallback.map(Projector::name))
            .field("color_fit", &self.color_fit.is_some())
            .field("cancelled", &self.cancel.is_cancelled())
            .field("align", &self.align)
            .field("ticking", &self.sink.is_some())
            .finish()
    }
}

impl<'a> Refit<'a> {
    pub fn new(projector: &'a dyn Projector, cancel: &'a CancellationToken) -> Self {
        Self {
            projector,
            fallback: None,
            color_fit: None,
            cancel,
            align: true,
            progress: Arc::new(RefitProgress::new()),
            sink: None,
        }
    }

    /// A second projector to try if the first one fails.
    ///
    /// `overview.md` §3.7 asks for PCA as "the permanent fallback if `annembed` fails or is
    /// abandoned ... as a tested fallback rather than a theoretical one", and this is the
    /// mechanism. It fires on any failure except cancellation: a cancelled job is a user
    /// decision, and retrying it under a different algorithm would be the app arguing.
    ///
    /// **The run is recorded under whichever projector actually produced the coordinates.**
    /// That is why this is a field on the job rather than a `Projector` that wraps two --
    /// a wrapper would have to answer `name()` before knowing which one ran, and a UMAP run
    /// row over a PCA layout is a lie that survives in the database.
    pub fn with_fallback(mut self, fallback: &'a dyn Projector) -> Self {
        self.fallback = Some(fallback);
        self
    }

    /// A second, independent fit that colors points instead of placing them.
    ///
    /// A plain closure rather than a `Projector` method: color is not a variant of the
    /// position fit, it answers a different question of the same data (`tsne`'s module doc),
    /// and today exactly one algorithm produces one at all. Forcing PCA and UMAP to answer
    /// `fit_color` too, just to say "no", would be a method every [`Projector`] carries for
    /// one caller's sake.
    pub fn with_color_fit(mut self, f: ColorFit<'a>) -> Self {
        self.color_fit = Some(f);
        self
    }

    /// Turns Procrustes alignment off.
    ///
    /// For measuring how much a projector moves on its own -- which is the number that says
    /// whether alignment is earning its place. Never for production: an unaligned UMAP
    /// re-fit is exactly the teleporting map `overview.md` §3.8 is about.
    pub fn without_alignment(mut self) -> Self {
        self.align = false;
        self
    }

    /// Starts a 100 ms ticker over this job's counters.
    pub fn with_progress<F>(mut self, sink: F) -> Self
    where
        F: FnMut(RefitSnapshot) + Send + 'static,
    {
        self.sink = Some(Box::new(sink));
        self
    }

    /// Shares the counter set, so a caller can watch while the job runs.
    pub fn progress(&self) -> Arc<RefitProgress> {
        Arc::clone(&self.progress)
    }
}

/// What one re-fit did.
#[derive(Debug, Clone, PartialEq)]
pub struct RefitReport {
    /// The `projection_runs` row that is now active.
    pub run_id: i64,
    pub algorithm: &'static str,
    pub sample_count: usize,
    /// Vectors the projector actually saw. Below [`Self::sample_count`] by however many
    /// rows share an embedding with an earlier one -- see the module note.
    pub distinct_vectors: usize,
    /// The run this one replaced, if there was one.
    pub previous_run_id: Option<i64>,
    /// Samples present in both layouts -- what the alignment was fitted on.
    pub correspondences: usize,
    /// `det(R) < 0`: the fit chose to reflect. Expected, and allowed.
    pub reflected: bool,
    /// Uniform scale the alignment applied.
    pub scale: f32,
    /// Diagonal of the new layout's bounding box, after alignment. The denominator every
    /// displacement number here is stated against.
    pub extent: f32,
    /// Median distance a pre-existing point moved, after alignment. `None` on a first fit.
    ///
    /// **This is the number `task.md` Phase 5's stability criterion is about**, and it is
    /// reported rather than merely tested so that a real library can be asked the question
    /// the synthetic corpora only approximate.
    pub median_displacement: Option<f32>,
    pub elapsed_ms: u128,
}

impl RefitReport {
    /// Median displacement as a fraction of the cloud's own size.
    ///
    /// Absolute displacement is meaningless across layouts -- UMAP's scale is arbitrary and
    /// Procrustes then rescales it again.
    pub fn relative_displacement(&self) -> Option<f32> {
        let median = self.median_displacement?;
        (self.extent > 0.0).then(|| median / self.extent)
    }
}

/// Runs a full re-fit **at background QoS**, which is what `overview.md` §3.8 means by
/// "low priority".
///
/// A word about why this is a `rayon` pool and not a thread. On macOS, priority is a QoS
/// class and it is a property of a *thread*, so demoting the one thread that calls
/// [`refit`] would leave everything expensive -- the covariance accumulation, the HNSW
/// build, `annembed`'s gradient descent -- running at the global pool's normal priority,
/// competing with a scan and with the render loop for exactly as long as it did before.
/// Building a pool whose workers are all demoted, and calling `install`, moves the whole
/// job: nested `rayon` work picks up the current thread's pool, so `hnsw_rs` and `annembed`
/// inherit it without knowing this function exists.
///
/// The pool is built per job. That costs a handful of thread spawns against a job measured
/// in minutes, and it means an idle app is not holding a dozen parked threads for a re-fit
/// that may never be asked for.
pub fn refit_at_low_priority(
    db: &Database,
    options: Refit<'_>,
) -> Result<RefitReport, ProjectionError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("audiocloud-refit-{i}"))
        .start_handler(|_| set_background_qos())
        .build();

    match pool {
        Ok(pool) => pool.install(|| refit(db, options)),
        Err(e) => {
            // A pool that will not build is not a reason to refuse the user a map. The job
            // runs on the global pool at normal priority, which is what it would have done
            // anyway if this function did not exist.
            tracing::warn!(error = %e, "could not build the low-priority pool; running at normal priority");
            refit(db, options)
        }
    }
}

/// Lowers the calling thread to `QOS_CLASS_BACKGROUND`.
///
/// Background rather than utility: a re-fit is work nobody is waiting on, and on Apple
/// silicon this class is what parks it on the efficiency cores instead of contending with
/// the render loop for a performance one.
///
/// Apple-only, and gated rather than assumed. AudioCloud targets nothing else, but a QoS
/// class is not a portable idea and a `#[cfg]` says so where a link error would only imply
/// it. Everywhere else the re-fit simply runs at whatever priority it was given, which is
/// the pre-Phase-5 behaviour.
#[cfg(target_vendor = "apple")]
fn set_background_qos() {
    // SAFETY: `pthread_set_qos_class_self_np` takes two by-value scalars, affects only the
    // calling thread, and returns an error code rather than trapping. There is no pointer
    // and no lifetime involved; the `unsafe` is entirely because it crosses the FFI
    // boundary. A nonzero return means the thread kept its old priority, which is the same
    // outcome as not calling it.
    #[allow(unsafe_code)]
    let code =
        unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0) };
    if code != 0 {
        tracing::debug!(code, "could not lower this thread's QoS class");
    }
}

#[cfg(not(target_vendor = "apple"))]
fn set_background_qos() {}

/// Runs a full re-fit and swaps it in, at whatever priority the caller has.
///
/// Blocking and long; nothing in here touches the Tauri main thread (cross-cutting rule 1).
/// Production callers want [`refit_at_low_priority`] -- this is the form the tests and
/// benchmarks use, where a demoted thread would only make the measurement about the
/// scheduler.
pub fn refit(db: &Database, options: Refit<'_>) -> Result<RefitReport, ProjectionError> {
    let Refit {
        projector,
        fallback,
        color_fit,
        cancel,
        align,
        progress,
        sink,
    } = options;

    let started = Instant::now();
    let mut ticker =
        sink.map(|sink| Ticker::watch(Arc::clone(&progress), RefitProgress::snapshot, sink));
    // A terminal snapshot is guaranteed by `Ticker`'s `Drop`, but only if the phase is
    // right when it runs -- so every exit from here forward goes through `finish`.
    let result = run(
        db, projector, fallback, color_fit, cancel, align, &progress, started,
    );
    progress.enter(RefitPhase::Done);
    if let Some(ticker) = ticker.as_mut() {
        ticker.finish();
    }
    result
}

fn run(
    db: &Database,
    projector: &dyn Projector,
    fallback: Option<&dyn Projector>,
    color_fit: Option<ColorFit<'_>>,
    cancel: &CancellationToken,
    align: bool,
    progress: &RefitProgress,
    started: Instant,
) -> Result<RefitReport, ProjectionError> {
    progress.enter(RefitPhase::Reading);
    tracing::debug!(
        algorithm = projector.name(),
        "refit: reading embedding locations"
    );
    EmbeddingSet::check_cancelled(cancel)?;

    let conn = db.read()?;
    let rows: Vec<(i64, EmbeddingLoc)> = queries::all_embedding_locs(&conn)?;
    let previous = queries::active_projection_run(&conn)?;
    // Read regardless of `align`: the previous layout is what displacement is *measured*
    // against, and an unaligned re-fit still owes that number -- it is the number that says
    // what alignment is worth.
    let previous_points = match &previous {
        Some(run) => queries::projection_point_map(&conn, run.id)?,
        None => HashMap::new(),
    };
    drop(conn);

    progress.samples.store(rows.len() as u64, Ordering::Relaxed);
    tracing::debug!(
        samples = rows.len(),
        has_previous_layout = previous.is_some(),
        elapsed_ms = started.elapsed().as_millis(),
        "refit: locations read"
    );

    // The lock is held only long enough to create the mapping. `EmbeddingMatrix` owns its
    // `Mmap`, so the store goes back to the persist stage of any concurrent scan
    // immediately, and this job reads pages of an inode that stays intact even if a
    // compaction renames a new file over the path underneath it.
    let matrix = {
        let store = db
            .embeddings()
            .lock()
            .map_err(|_| crate::db::DbError::Poisoned("embedding store"))?;
        store.matrix()?
    };
    let (distinct, members) = group_by_vector(&rows);
    let data = EmbeddingSet::new(&matrix, &distinct);

    progress.enter(RefitPhase::Fitting);
    tracing::debug!(
        algorithm = projector.name(),
        distinct_vectors = distinct.len(),
        elapsed_ms = started.elapsed().as_millis(),
        "refit: fitting"
    );
    let (fitted, projector) = fit(projector, fallback, &data, cancel)?;
    if fitted.len() != distinct.len() {
        return Err(ProjectionError::Umap(format!(
            "{} produced {} coordinates for {} vectors",
            projector.name(),
            fitted.len(),
            distinct.len()
        )));
    }
    tracing::debug!(
        algorithm = projector.name(),
        elapsed_ms = started.elapsed().as_millis(),
        "refit: fit complete"
    );

    // Same `data` the position fit just read, while the mapping backing it is still alive --
    // an independent question asked of the same vectors, not a derivative of where they
    // landed (`tsne`'s module doc).
    let colors = match color_fit {
        Some(f) => {
            tracing::debug!(
                elapsed_ms = started.elapsed().as_millis(),
                "refit: color fitting"
            );
            let fitted_colors = f(&data, cancel)?;
            if fitted_colors.len() != distinct.len() {
                return Err(ProjectionError::Umap(format!(
                    "color fit produced {} colors for {} vectors",
                    fitted_colors.len(),
                    distinct.len()
                )));
            }
            tracing::debug!(
                elapsed_ms = started.elapsed().as_millis(),
                "refit: color fit complete"
            );
            Some(fan_out_colors(&rows, &members, &fitted_colors))
        }
        None => None,
    };
    drop(matrix);

    let mut points = fan_out(&rows, &members, &fitted);
    EmbeddingSet::check_cancelled(cancel)?;

    progress.enter(RefitPhase::Aligning);
    let (alignment, correspondences) = if align {
        align_onto_previous(&rows, &points, &previous_points)
    } else {
        (Alignment::identity(), 0)
    };
    tracing::debug!(
        correspondences,
        elapsed_ms = started.elapsed().as_millis(),
        "refit: aligned"
    );
    // Fitted on the shared points, applied to every point including the new ones: the
    // transform describes the relationship between two coordinate systems, not between two
    // sets of samples (`overview.md` §3.8, step 4).
    alignment.apply_all(&mut points);

    let median_displacement = median_displacement(&rows, &points, &previous_points);
    let extent = BoundingBox::of(&points).map_or(0.0, |b| b.diagonal());

    EmbeddingSet::check_cancelled(cancel)?;
    progress.enter(RefitPhase::Writing);
    tracing::debug!(
        elapsed_ms = started.elapsed().as_millis(),
        "refit: writing shadow run"
    );

    let run_id = db.writer().begin_projection_run(
        projector.name(),
        &projector.params_json(),
        rows.len() as i64,
    )?;

    // Everything past this point owns a shadow row that must not outlive a failure.
    let written = write_points(
        db,
        run_id,
        &rows,
        &points,
        colors.as_deref(),
        cancel,
        progress,
    );
    if let Err(e) = written {
        discard(db, run_id);
        return Err(e);
    }
    tracing::debug!(
        run_id,
        elapsed_ms = started.elapsed().as_millis(),
        "refit: shadow run written"
    );

    progress.enter(RefitPhase::Swapping);
    if let Err(e) = db.writer().activate_projection_run(run_id) {
        discard(db, run_id);
        return Err(e.into());
    }

    tracing::info!(
        run_id,
        algorithm = projector.name(),
        samples = rows.len(),
        distinct_vectors = distinct.len(),
        correspondences,
        reflected = alignment.determinant() < 0.0,
        median_displacement,
        extent,
        elapsed_ms = started.elapsed().as_millis(),
        "projection re-fit complete"
    );

    Ok(RefitReport {
        run_id,
        algorithm: projector.name(),
        sample_count: rows.len(),
        distinct_vectors: distinct.len(),
        previous_run_id: previous.map(|r| r.id),
        correspondences,
        reflected: alignment.determinant() < 0.0,
        scale: alignment.scale(),
        extent,
        median_displacement,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// Splits rows into one representative per distinct vector, plus who shares it.
///
/// Keyed by `(offset, dims)` rather than by content: two rows point at the same bytes or
/// they do not, and comparing 512 floats to discover what an integer already says would be
/// slower and no more true. Order is preserved -- `all_embedding_locs` returns file order,
/// and a projector reading the mmap forwards is why.
fn group_by_vector(rows: &[(i64, EmbeddingLoc)]) -> (Vec<(i64, EmbeddingLoc)>, Vec<Vec<usize>>) {
    let mut index: std::collections::HashMap<(u64, u32), usize> = HashMap::new();
    let mut distinct = Vec::with_capacity(rows.len());
    let mut members: Vec<Vec<usize>> = Vec::with_capacity(rows.len());

    for (row, (sample_id, loc)) in rows.iter().enumerate() {
        match index.entry((loc.offset, loc.dims)) {
            Entry::Occupied(slot) => members[*slot.get()].push(row),
            Entry::Vacant(slot) => {
                slot.insert(distinct.len());
                distinct.push((*sample_id, *loc));
                members.push(vec![row]);
            }
        }
    }
    (distinct, members)
}

/// Gives every row the coordinate of the vector it points at.
///
/// The first row holding a vector gets the fitted position exactly; the rest get it nudged,
/// so that a file and its six copies are six pickable dots rather than one. The nudge is
/// keyed by `sample_id`, so it is the same on every re-fit.
fn fan_out(rows: &[(i64, EmbeddingLoc)], members: &[Vec<usize>], fitted: &[Point3]) -> Vec<Point3> {
    let extent = BoundingBox::of(fitted).map_or(0.0, |b| b.diagonal());
    let mut points = vec![[0.0f32; 3]; rows.len()];
    for (vector, sharing) in members.iter().enumerate() {
        for (n, &row) in sharing.iter().enumerate() {
            points[row] = if n == 0 {
                fitted[vector]
            } else {
                super::incremental::jitter(fitted[vector], rows[row].0, extent)
            };
        }
    }
    points
}

/// [`fan_out`] for colors: every row sharing a vector gets its exact color, not a jittered
/// one. Jitter exists so two samples on one vector are two clickable dots; two dots one
/// shade apart would only make the shared-vector case look like a rounding error instead of
/// what it is.
fn fan_out_colors(
    rows: &[(i64, EmbeddingLoc)],
    members: &[Vec<usize>],
    fitted: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    let mut colors = vec![[0.0f32; 3]; rows.len()];
    for (vector, sharing) in members.iter().enumerate() {
        for &row in sharing {
            colors[row] = fitted[vector];
        }
    }
    colors
}

/// Runs the projector, falling back to the second one on anything but a cancellation.
///
/// Returns the projector that actually produced the coordinates, so the run row and the
/// report name the algorithm that ran rather than the one that was asked for.
fn fit<'p>(
    projector: &'p dyn Projector,
    fallback: Option<&'p dyn Projector>,
    data: &EmbeddingSet<'_>,
    cancel: &CancellationToken,
) -> Result<(Vec<Point3>, &'p dyn Projector), ProjectionError> {
    match projector.fit_transform(data, cancel) {
        Ok(points) => Ok((points, projector)),
        Err(ProjectionError::Cancelled) => Err(ProjectionError::Cancelled),
        Err(e) => {
            let Some(fallback) = fallback else {
                return Err(e);
            };
            tracing::warn!(
                projector = projector.name(),
                fallback = fallback.name(),
                error = %e,
                "the primary projector failed; falling back"
            );
            Ok((fallback.fit_transform(data, cancel)?, fallback))
        }
    }
}

/// Builds the correspondence pair lists and fits the transform.
///
/// One pass over `rows`, which is what keeps `source[i]` and `target[i]` describing the same
/// sample. Sorting two lists and zipping them would be the same code with one more way to
/// be silently wrong, and a Procrustes fit onto mismatched correspondences does not fail --
/// it returns a perfectly valid transform onto nonsense.
fn align_onto_previous(
    rows: &[(i64, EmbeddingLoc)],
    points: &[Point3],
    previous: &HashMap<i64, Point3>,
) -> (Alignment, usize) {
    if previous.is_empty() {
        return (Alignment::identity(), 0);
    }
    let mut source = Vec::new();
    let mut target = Vec::new();
    for ((sample_id, _), point) in rows.iter().zip(points.iter()) {
        if let Some(old) = previous.get(sample_id) {
            source.push(*point);
            target.push(*old);
        }
    }
    let n = source.len();
    (procrustes::fit(&source, &target), n)
}

/// Median distance a pre-existing point moved, after alignment.
///
/// The median rather than the mean: a re-fit that legitimately relocates one cluster should
/// not be reported as having moved the whole library, and the mean cannot tell those apart.
fn median_displacement(
    rows: &[(i64, EmbeddingLoc)],
    points: &[Point3],
    previous: &HashMap<i64, Point3>,
) -> Option<f32> {
    let mut moved: Vec<f32> = rows
        .iter()
        .zip(points.iter())
        .filter_map(|((id, _), new)| previous.get(id).map(|old| distance(*new, *old)))
        .collect();
    if moved.is_empty() {
        return None;
    }
    moved.sort_by(f32::total_cmp);
    Some(moved[moved.len() / 2])
}

fn write_points(
    db: &Database,
    run_id: i64,
    rows: &[(i64, EmbeddingLoc)],
    points: &[Point3],
    colors: Option<&[[f32; 3]]>,
    cancel: &CancellationToken,
    progress: &RefitProgress,
) -> Result<(), ProjectionError> {
    write_chunked(rows, points, cancel, |batch| {
        let n = batch.len() as u64;
        db.writer().set_projection_points(run_id, batch)?;
        progress.written.fetch_add(n, Ordering::Relaxed);
        Ok(())
    })?;
    if let Some(colors) = colors {
        write_chunked(rows, colors, cancel, |batch| {
            db.writer().set_projection_colors(run_id, batch)?;
            Ok(())
        })?;
    }
    db.writer().flush()?;
    Ok(())
}

/// Writes `values` (points, or colors) alongside `rows`' sample ids in chunks of
/// [`WRITE_CHUNK`], checking for cancellation once per chunk rather than once per row.
///
/// Points and colors both write this way, differing only in which writer method `write`
/// calls and whether it also has progress to report -- one shared loop instead of two
/// identical ones.
fn write_chunked<T: Copy>(
    rows: &[(i64, EmbeddingLoc)],
    values: &[T],
    cancel: &CancellationToken,
    mut write: impl FnMut(Vec<(i64, T)>) -> Result<(), ProjectionError>,
) -> Result<(), ProjectionError> {
    for chunk in rows
        .iter()
        .zip(values.iter())
        .collect::<Vec<_>>()
        .chunks(WRITE_CHUNK)
    {
        EmbeddingSet::check_cancelled(cancel)?;
        let batch: Vec<(i64, T)> = chunk.iter().map(|((id, _), v)| (*id, **v)).collect();
        write(batch)?;
    }
    Ok(())
}

/// Drops a shadow run whose job did not finish.
///
/// Logged rather than propagated: the caller already has a reason for failing, and
/// replacing it with "and also the cleanup failed" loses the one that explains what
/// happened. An orphaned inactive run costs disk and is deleted by the next successful
/// swap's prune.
fn discard(db: &Database, run_id: i64) {
    if let Err(e) = db.writer().discard_projection_run(run_id) {
        tracing::warn!(run_id, error = %e, "could not discard the abandoned projection run");
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::projection::PcaProjector;

    #[test]
    fn the_phase_never_rewinds() {
        let progress = RefitProgress::new();
        progress.enter(RefitPhase::Writing);
        progress.enter(RefitPhase::Reading);
        assert_eq!(progress.phase(), RefitPhase::Writing);
    }

    #[test]
    fn a_displacement_is_only_meaningful_next_to_the_size_of_the_cloud() {
        let report = RefitReport {
            run_id: 1,
            algorithm: "pca",
            sample_count: 10,
            distinct_vectors: 10,
            previous_run_id: None,
            correspondences: 10,
            reflected: false,
            scale: 1.0,
            extent: 20.0,
            median_displacement: Some(1.0),
            elapsed_ms: 0,
        };
        assert_eq!(report.relative_displacement(), Some(0.05));

        let flat = RefitReport {
            extent: 0.0,
            ..report.clone()
        };
        assert_eq!(flat.relative_displacement(), None);
    }

    /// A first fit has nothing to align against and must say so rather than inventing a
    /// transform from an empty correspondence set.
    #[test]
    fn a_first_fit_has_no_correspondences_and_does_not_move() {
        let rows = vec![(1i64, EmbeddingLoc { offset: 0, dims: 4 })];
        let points = vec![[3.0, 4.0, 5.0]];
        let (alignment, n) = align_onto_previous(&rows, &points, &HashMap::new());
        assert_eq!(n, 0);
        assert!(alignment.is_identity());
    }

    #[test]
    fn the_median_is_the_middle_displacement_not_the_worst_one() {
        let rows: Vec<(i64, EmbeddingLoc)> = (1..=5)
            .map(|i| (i, EmbeddingLoc { offset: 0, dims: 4 }))
            .collect();
        let points: Vec<Point3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ];
        let previous = (1..=5).map(|i| (i, [0.0, 0.0, 0.0])).collect();

        // Displacements are 0, 1, 2, 3, 100 -- the median is 2 and the mean is 21.2.
        assert_eq!(median_displacement(&rows, &points, &previous), Some(2.0));
    }

    /// Phase 4's finding 1, landing here: two rows can point at one vector, and the
    /// projector must see that vector once.
    #[test]
    fn rows_sharing_a_vector_are_projected_once() {
        let twin = EmbeddingLoc { offset: 0, dims: 4 };
        let other = EmbeddingLoc { offset: 8, dims: 4 };
        // Rows 1 and 3 are duplicates of each other; row 2 is its own file.
        let rows = vec![(10i64, twin), (11, other), (12, twin)];

        let (distinct, members) = group_by_vector(&rows);

        assert_eq!(distinct, vec![(10, twin), (11, other)]);
        assert_eq!(members, vec![vec![0, 2], vec![1]]);
    }

    /// ...and every row still gets its own coordinate, separated so a duplicate is not an
    /// unclickable dot underneath its twin.
    #[test]
    fn a_shared_vector_fans_out_to_separable_coordinates() {
        let twin = EmbeddingLoc { offset: 0, dims: 4 };
        let other = EmbeddingLoc { offset: 8, dims: 4 };
        let rows = vec![(10i64, twin), (11, other), (12, twin)];
        let (_, members) = group_by_vector(&rows);
        let fitted = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];

        let points = fan_out(&rows, &members, &fitted);

        assert_eq!(points.len(), 3);
        assert_eq!(
            points[0], fitted[0],
            "the first holder keeps the fitted position"
        );
        assert_eq!(points[1], fitted[1]);
        assert_ne!(
            points[2], points[0],
            "a duplicate landed on top of its twin"
        );
        assert!(
            distance(points[2], points[0]) < 1.0,
            "the nudge became a move: {:?}",
            points[2]
        );
        // Deterministic: the same rows fan out the same way every time.
        assert_eq!(points, fan_out(&rows, &members, &fitted));
    }

    /// The projector is named in the report, so a run built by the fallback cannot be
    /// mistaken for one built by UMAP.
    #[test]
    fn the_projector_names_itself_in_the_run() {
        assert_eq!(PcaProjector::new().name(), "pca");
        assert_eq!(crate::projection::UmapProjector::default().name(), "umap");
    }
}
