//! Tauri command handlers, one module per command group (`overview.md` §6.1).
//!
//! These are deliberately thin: deserialize, call into a domain module, map the error.
//! Business logic in a `#[tauri::command]` function is logic that cannot be unit-tested
//! without an `AppHandle`. What lives here that is *not* thin is [`Jobs`], because
//! "is a scan already running" is a question only the command layer is in a position to ask
//! -- the pipeline is a function, and a function has no opinion about being called twice.

pub mod cloud;
pub mod collections;
pub mod library;
pub mod projection;
pub mod samples;
pub mod settings;

use std::sync::{
    atomic::{AtomicI64, Ordering},
    Arc, Mutex,
};

use crate::{error::AppError, pipeline::CancellationToken};

/// The long-running jobs the app allows at most one of, and the tokens that stop them.
///
/// **Three separate slots, not one.** A model download and a library scan are unrelated and
/// a user who starts a scan before the download finishes is being sensible, not confused --
/// a scan with no model still decodes and analyzes, and the next one finishes what it
/// started. What must not overlap is two of the *same* job: two scans would have two writers
/// competing for the same `scan_runs` rows and two `ort` sessions' worth of memory, and two
/// re-fits would race to activate different shadow runs.
///
/// Cheap to clone -- it is one `Arc` -- because every command that starts a job hands a
/// clone to a blocking task that outlives the command's own borrow of managed state.
#[derive(Debug, Clone, Default)]
pub struct Jobs {
    inner: Arc<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    scan: Mutex<Option<Running>>,
    refit: Mutex<Option<Running>>,
    /// Source of job ids for work whose real database id does not exist yet. See
    /// [`Jobs::begin_refit`].
    next_job_id: AtomicI64,
}

/// One running job: what stops it, and what the frontend calls it.
#[derive(Debug, Clone)]
struct Running {
    /// `None` for a scan between the moment it is admitted and the moment its `scan_runs`
    /// row is opened -- a window measured in the time it takes the writer to run one
    /// `INSERT`. Cancellation during that window is not offered because the frontend does
    /// not have an id to cancel with yet.
    id: Option<i64>,
    cancel: CancellationToken,
}

/// A running job's slot, released when this is dropped.
///
/// An RAII guard rather than a `finally` block: the job runs on a blocking task that can
/// return early through half a dozen `?`s, and every one of them has to release the slot or
/// the app refuses to scan again until it is restarted. Dropping is the one exit path that
/// cannot be forgotten.
#[derive(Debug)]
pub struct JobSlot {
    jobs: Jobs,
    kind: JobKind,
}

#[derive(Debug, Clone, Copy)]
enum JobKind {
    Scan,
    Refit,
}

impl Drop for JobSlot {
    fn drop(&mut self) {
        let inner = &self.jobs.inner;
        match self.kind {
            JobKind::Scan => clear(&inner.scan),
            JobKind::Refit => clear(&inner.refit),
        }
    }
}

fn clear(slot: &Mutex<Option<Running>>) {
    if let Ok(mut slot) = slot.lock() {
        *slot = None;
    }
}

impl Jobs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Admits a scan, or reports the one already running.
    ///
    /// Returns the token the scan should be driven with and the slot that releases the
    /// admission when the scan ends.
    pub fn begin_scan(&self) -> Result<(CancellationToken, JobSlot), AppError> {
        let mut slot = self
            .inner
            .scan
            .lock()
            .map_err(|_| AppError::internal("locking the scan slot", "poisoned"))?;
        if let Some(running) = slot.as_ref() {
            return Err(AppError::ScanInProgress {
                scan_id: running.id.unwrap_or(0),
            });
        }

        let cancel = CancellationToken::new();
        *slot = Some(Running {
            id: None,
            cancel: cancel.clone(),
        });
        Ok((
            cancel,
            JobSlot {
                jobs: self.clone(),
                kind: JobKind::Scan,
            },
        ))
    }

    /// Records the `scan_runs` id the pipeline assigned, so `cancel_scan` can match on it.
    pub fn attach_scan_id(&self, scan_id: i64) {
        if let Ok(mut slot) = self.inner.scan.lock() {
            if let Some(running) = slot.as_mut() {
                running.id = Some(scan_id);
            }
        }
    }

    /// Signals the running scan to stop, if `scan_id` is the one running.
    ///
    /// A mismatched id is `Ok`, not an error: the frontend cancelling a scan that already
    /// finished on its own is a race it cannot avoid and does not need to hear about.
    pub fn cancel_scan(&self, scan_id: i64) -> Result<(), AppError> {
        let slot = self
            .inner
            .scan
            .lock()
            .map_err(|_| AppError::internal("locking the scan slot", "poisoned"))?;
        if let Some(running) = slot.as_ref() {
            if running.id == Some(scan_id) {
                running.cancel.cancel();
            }
        }
        Ok(())
    }

    /// Admits a re-fit, handing back the job id the frontend will cancel it with.
    ///
    /// **The id is app-assigned, not a `projection_runs` id**, and that is forced by
    /// `overview.md` §3.8 rather than chosen: the shadow run row is not created until the
    /// coordinates exist, which is minutes into a 50,000-point UMAP fit. A command that
    /// returned the run id would have to block until then, which is precisely what starting
    /// a background job is for. The real run id arrives in the terminal `RefitEvent`.
    pub fn begin_refit(&self) -> Result<(i64, CancellationToken, JobSlot), AppError> {
        let mut slot = self
            .inner
            .refit
            .lock()
            .map_err(|_| AppError::internal("locking the re-fit slot", "poisoned"))?;
        if let Some(running) = slot.as_ref() {
            return Err(AppError::RefitInProgress {
                job_id: running.id.unwrap_or(0),
            });
        }

        let job_id = self.inner.next_job_id.fetch_add(1, Ordering::Relaxed) + 1;
        let cancel = CancellationToken::new();
        *slot = Some(Running {
            id: Some(job_id),
            cancel: cancel.clone(),
        });
        Ok((
            job_id,
            cancel,
            JobSlot {
                jobs: self.clone(),
                kind: JobKind::Refit,
            },
        ))
    }

    /// Signals the running re-fit to stop, if `job_id` is the one running.
    pub fn cancel_refit(&self, job_id: i64) -> Result<(), AppError> {
        let slot = self
            .inner
            .refit
            .lock()
            .map_err(|_| AppError::internal("locking the re-fit slot", "poisoned"))?;
        if let Some(running) = slot.as_ref() {
            if running.id == Some(job_id) {
                running.cancel.cancel();
            }
        }
        Ok(())
    }

    /// Whether a scan or re-fit is running.
    ///
    /// What `settings::reset_database` refuses to run over: deleting the database out from
    /// under a job that is mid-write would not just fail that job, it would race the delete
    /// itself against whatever the writer thread is doing.
    pub fn any_running(&self) -> bool {
        let scan = self.inner.scan.lock().map(|s| s.is_some()).unwrap_or(true);
        let refit = self.inner.refit.lock().map(|s| s.is_some()).unwrap_or(true);
        scan || refit
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn a_second_scan_is_refused_and_says_which_one_is_running() {
        let jobs = Jobs::new();
        let (_cancel, slot) = jobs.begin_scan().unwrap();
        jobs.attach_scan_id(42);

        match jobs.begin_scan() {
            Err(AppError::ScanInProgress { scan_id }) => assert_eq!(scan_id, 42),
            other => panic!("expected ScanInProgress, got {other:?}"),
        }

        drop(slot);
        assert!(jobs.begin_scan().is_ok(), "the slot did not release");
    }

    /// The RAII guard is the whole point: a job that returns early through a `?` must not
    /// leave the app refusing to scan until it is restarted.
    #[test]
    fn a_slot_releases_even_when_the_job_ends_badly() {
        let jobs = Jobs::new();
        let run = || -> Result<(), AppError> {
            let (_cancel, _slot) = jobs.begin_scan()?;
            Err(AppError::Cancelled)
        };
        assert!(run().is_err());
        assert!(jobs.begin_scan().is_ok());
    }

    #[test]
    fn cancelling_a_scan_that_is_not_running_is_not_an_error() {
        let jobs = Jobs::new();
        assert!(jobs.cancel_scan(7).is_ok());

        let (cancel, _slot) = jobs.begin_scan().unwrap();
        jobs.attach_scan_id(7);
        jobs.cancel_scan(8).unwrap();
        assert!(!cancel.is_cancelled(), "cancelled the wrong scan");
        jobs.cancel_scan(7).unwrap();
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn refit_job_ids_are_distinct() {
        let jobs = Jobs::new();
        let (first, _, slot) = jobs.begin_refit().unwrap();
        drop(slot);
        let (second, _, _slot) = jobs.begin_refit().unwrap();
        assert_ne!(first, second);
    }
}
