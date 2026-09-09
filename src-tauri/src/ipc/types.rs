//! Transport 1: the JSON records the command surface returns (`overview.md` §6.1, §6.2).
//!
//! These are **DTOs, not domain types**, and the distinction is deliberate. `queries::
//! LibraryRoot` carries a `last_scan_id` the UI has no use for; `db::SampleFeatures` is a
//! block of `Option<f32>` with no idea which of them are null because the column is missing
//! and which because the analyzer declined to guess. A struct per query result would leak
//! schema decisions into TypeScript and make every migration a frontend change.
//!
//! Everything here derives `TS`. `tests/bindings.rs` writes them to `src/bindings/` and CI
//! fails the build when the committed files disagree, so a Rust field that changes name
//! cannot silently desynchronize the frontend (`overview.md` §6.8).

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::{
    audio,
    db::{queries, SampleFeatures},
};

/// A library root as the UI shows it.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "LibraryRoot.ts")]
pub struct LibraryRoot {
    pub id: i64,
    pub path: String,
    pub label: Option<String>,
    pub enabled: bool,
    pub added_at: i64,
    /// Rows currently under this root. The number the sidebar shows next to the folder, and
    /// the reason this is a DTO: `queries::LibraryRoot` is one table and this is two.
    pub sample_count: i64,
}

impl LibraryRoot {
    pub fn new(root: queries::LibraryRoot, sample_count: i64) -> Self {
        Self {
            id: root.id,
            path: root.path,
            label: root.label,
            enabled: root.enabled,
            added_at: root.added_at,
            sample_count,
        }
    }
}

/// The DSP descriptor block, as the inspector renders it (`overview.md` §3.3).
///
/// Every field is nullable because every field genuinely can be: a file that failed to
/// decode has none of them, and a pitchless sample has no key however well it decoded.
#[derive(Debug, Clone, Default, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "SampleFeatureBlock.ts")]
pub struct SampleFeatureBlock {
    pub peak_db: Option<f32>,
    pub rms_db: Option<f32>,
    pub lufs_integrated: Option<f32>,
    pub spectral_centroid: Option<f32>,
    pub spectral_flatness: Option<f32>,
    pub zero_crossing: Option<f32>,
    pub onset_density: Option<f32>,
    pub bpm: Option<f32>,
    pub bpm_confidence: Option<f32>,
    /// 0-11, C through B. `None` for unpitched material.
    pub key_root: Option<i32>,
    /// 0 minor, 1 major.
    pub key_mode: Option<i32>,
    pub key_confidence: Option<f32>,
}

impl From<SampleFeatures> for SampleFeatureBlock {
    fn from(f: SampleFeatures) -> Self {
        Self {
            peak_db: f.peak_db,
            rms_db: f.rms_db,
            lufs_integrated: f.lufs_integrated,
            spectral_centroid: f.spectral_centroid,
            spectral_flatness: f.spectral_flatness,
            zero_crossing: f.zero_crossing,
            onset_density: f.onset_density,
            bpm: f.bpm,
            bpm_confidence: f.bpm_confidence,
            key_root: f.key_root,
            key_mode: f.key_mode,
            key_confidence: f.key_confidence,
        }
    }
}

/// Everything the inspector shows for one sample.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "SampleDetail.ts")]
pub struct SampleDetail {
    pub id: i64,
    pub root_id: i64,
    /// Absolute path of the root, so the UI can show where a file came from without
    /// concatenating paths itself.
    pub root_path: String,
    pub rel_path: String,
    pub filename: String,
    pub ext: String,
    pub size_bytes: i64,
    /// Duration of the **whole** file, not of the ten-second analysis window.
    pub duration_ms: Option<i64>,
    pub sample_rate: Option<i64>,
    pub channels: Option<i64>,
    /// `pending` | `decoded` | `embedded` | `decode_failed` | `missing`.
    pub status: String,
    /// Why the file was quarantined, for a `decode_failed` row.
    pub error: Option<String>,
    /// Whether a vector exists, which is what decides if `get_similar` can say anything.
    pub embedded: bool,
    pub features: Option<SampleFeatureBlock>,
    pub tags: Vec<String>,
    /// Last write to this row. **The cache key for `abpeaks://`**: the peak summary of a
    /// file that was rescanned after an edit is different audio at the same sample id, and
    /// the scheme's `immutable` header means the WebView will never ask again on its own.
    /// The frontend appends this as a query string; see `src/ipc/peaks.ts`.
    pub updated_at: i64,
}

/// One row of a nearest-neighbor listing.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "Neighbor.ts")]
pub struct Neighbor {
    pub sample_id: i64,
    pub rel_path: String,
    pub filename: String,
    pub duration_ms: Option<i64>,
    /// Cosine similarity in `[-1, 1]`. Every stored vector is L2-normalized, so this is a
    /// dot product.
    pub similarity: f32,
}

/// A tag, and how many samples carry it.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "Tag.ts")]
pub struct Tag {
    pub id: i64,
    pub name: String,
    pub color: Option<String>,
    pub sample_count: i64,
}

/// A saved set of samples.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "Collection.ts")]
pub struct Collection {
    pub id: i64,
    pub name: String,
    pub created_at: i64,
    pub sample_count: i64,
}

/// One sample inside a collection, in display order (`overview.md` §6.1, Phase 9).
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "CollectionMember.ts")]
pub struct CollectionMember {
    pub sample_id: i64,
    pub rel_path: String,
    pub filename: String,
    pub duration_ms: Option<i64>,
}

impl From<queries::CollectionMemberRow> for CollectionMember {
    fn from(row: queries::CollectionMemberRow) -> Self {
        Self {
            sample_id: row.sample_id,
            rel_path: row.rel_path,
            filename: row.filename,
            duration_ms: row.duration_ms,
        }
    }
}

/// A collection with its members, in the order the user arranged them.
///
/// A DTO distinct from [`Collection`] rather than an optional field on it: the collection
/// list only ever needs a name and a count, and fetching every member of every collection to
/// answer `list_collections` would be a query nobody asked for.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "CollectionDetail.ts")]
pub struct CollectionDetail {
    pub id: i64,
    pub name: String,
    pub created_at: i64,
    pub members: Vec<CollectionMember>,
}

impl CollectionDetail {
    pub fn new(row: queries::CollectionRow, members: Vec<queries::CollectionMemberRow>) -> Self {
        Self {
            id: row.id,
            name: row.name,
            created_at: row.created_at,
            members: members.into_iter().map(CollectionMember::from).collect(),
        }
    }
}

/// One output device the Settings panel can offer.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "AudioDeviceInfo.ts")]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

impl From<audio::AudioDeviceInfo> for AudioDeviceInfo {
    fn from(d: audio::AudioDeviceInfo) -> Self {
        Self {
            name: d.name,
            is_default: d.is_default,
        }
    }
}

/// The settings that survive a restart (`task.md` Phase 9). Everything else Settings shows --
/// model status, projection params -- is either derived or lives on its own durable row
/// already, and does not belong in this key/value store.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "AppSettings.ts")]
pub struct AppSettings {
    /// `None` means "the OS default," which is also the initial state before anyone has
    /// chosen a device.
    pub audio_device: Option<String>,
    pub gain: f32,
}

pub use crate::db::search::{Feature, FeatureRange, QueryFilter};

/// Which projector a re-fit should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "Algorithm.ts")]
pub enum Algorithm {
    /// Barnes-Hut t-SNE (`projection::tsne`). The default now: measured against this app's
    /// own library, it roughly doubled on-screen neighbor accuracy over UMAP under UMAP's
    /// own tuned defaults (`TsneParams`'s doc has the numbers) -- not a marginal win, a
    /// different algorithm outright suiting this material better.
    Tsne,
    /// `annembed`'s UMAP over an HNSW graph. Kept as a selectable alternative, not the
    /// fallback -- PCA is, for both -- because a wholesale win on one library is evidence,
    /// not a proof it holds for every one.
    Umap,
    /// Deterministic, seconds rather than minutes, and always available as the fallback.
    Pca,
}

/// What `start_refit` is asked for (`overview.md` §3.7, §3.8).
#[derive(Debug, Clone, Copy, Deserialize, TS)]
#[serde(rename_all = "camelCase", default)]
#[ts(export_to = "RefitParams.ts")]
pub struct RefitParams {
    pub algorithm: Algorithm,
    /// UMAP's neighbourhood size. Ignored by PCA. `None` takes the tuned default (15).
    #[ts(optional)]
    pub n_neighbors: Option<usize>,
    /// How tightly points may pack (`annembed`'s `scale_rho`, by way of `min_dist`). Ignored
    /// by PCA. `None` takes the tuned default (0.01) -- see `UmapParams::min_dist`'s doc for
    /// where that number came from.
    #[ts(optional)]
    pub min_dist: Option<f32>,
    /// Exponent of the embedded-space kernel (`annembed`'s `b`). Ignored by PCA. `None` takes
    /// the tuned default (0.2). Lower is tighter here, not higher -- see `UmapParams::sharpness`'s
    /// doc before nudging this away from the measured-best value.
    #[ts(optional)]
    pub sharpness: Option<f64>,
    /// Exponent of the edge weight in the original 512-dim kNN graph (`annembed`'s `beta`).
    /// Ignored by PCA. `None` takes the tuned default (1.0, `annembed`'s own).
    #[ts(optional)]
    pub input_sharpness: Option<f64>,
    /// Gradient batches. Ignored by PCA. `None` takes the tuned default (20).
    #[ts(optional)]
    pub n_epochs: Option<usize>,
    /// Run a full re-fit even when the planner would have placed the new points
    /// incrementally.
    ///
    /// The planner is right almost always -- twelve new files must not rotate a library --
    /// and this exists because "almost always" is not "always": incremental placement
    /// accumulates drift, and the user is the one who can see that the map has gone
    /// crooked. `overview.md` §3.8 says a re-fit is announced rather than silent, and this
    /// is the announcement's button.
    ///
    /// **Also the only way a UMAP parameter change actually takes effect.** Incremental
    /// placement never calls the projector at all -- it barycenters new points into the
    /// existing layout -- and an unchanged library plans as `UpToDate` and refits nothing.
    /// A tuning UI that wants its sliders to do something has to set this.
    pub force_full: bool,
}

impl Default for RefitParams {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::Tsne,
            n_neighbors: None,
            min_dist: None,
            sharpness: None,
            input_sharpness: None,
            n_epochs: None,
            force_full: false,
        }
    }
}

/// What a finished re-fit produced. The terminal payload of `Channel<RefitEvent>`.
#[derive(Debug, Clone, Serialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "RefitOutcome.ts")]
pub struct RefitOutcome {
    /// The `projection_runs` row that is now active.
    pub run_id: i64,
    /// The projector that actually produced the coordinates, which is not necessarily the
    /// one that was asked for -- PCA is the permanent fallback.
    pub algorithm: String,
    pub sample_count: usize,
    /// Samples present in both the old layout and the new one; what Procrustes was fitted
    /// on.
    pub correspondences: usize,
    /// Median distance a pre-existing point moved, as a fraction of the cloud's diagonal.
    /// `None` on a first fit. **The number `overview.md` §3.8's stability claim is about**,
    /// and the reason the UI can say "the map barely moved" rather than hoping.
    pub relative_displacement: Option<f32>,
    /// True when the points were added to the existing layout rather than re-fitted, in
    /// which case nothing that was already on the map moved at all.
    pub incremental: bool,
    pub elapsed_ms: u64,
}
