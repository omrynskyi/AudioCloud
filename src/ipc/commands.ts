/**
 * Typed wrappers over every command in `overview.md` §6.1.
 *
 * One function per command, argument and return types generated from Rust into
 * `src/bindings/`. Nothing else in the app calls `invoke` directly — this file is the only
 * place a command name appears as a string, so a renamed command is one edit and a compile
 * error rather than a runtime "command not found" found by a user.
 *
 * Two things every wrapper does:
 *
 * - **Normalizes the rejection.** Tauri rejects with whatever the command serialized; these
 *   guarantee it is an `AppError` so callers can switch on `kind` exhaustively.
 * - **Says which transport it is on.** The three that return `ArrayBuffer` are decoded here
 *   through `binary.ts` rather than handing raw buffers to feature code.
 */

import { invoke as tauriInvoke, Channel } from '@tauri-apps/api/core';

import type { AppSettings } from '../bindings/AppSettings';
import type { AudioDeviceInfo } from '../bindings/AudioDeviceInfo';
import type { Collection } from '../bindings/Collection';
import type { CollectionDetail } from '../bindings/CollectionDetail';
import type { Feature } from '../bindings/Feature';
import type { FeatureRange } from '../bindings/FeatureRange';
import type { LibraryRoot } from '../bindings/LibraryRoot';
import type { Neighbor } from '../bindings/Neighbor';
import type { QueryFilter } from '../bindings/QueryFilter';
import type { RefitEvent } from '../bindings/RefitEvent';
import type { RefitParams } from '../bindings/RefitParams';
import type { SampleDetail } from '../bindings/SampleDetail';
import type { ScanEvent } from '../bindings/ScanEvent';
import type { Tag } from '../bindings/Tag';
import {
  decodeFeatureColumn,
  decodeIdList,
  decodePointCloud,
  decodePointColors,
  type PointCloud,
  type PointColors,
} from './binary';
import { IpcError, toAppError } from './errors';

export type {
  AppSettings,
  AudioDeviceInfo,
  Collection,
  CollectionDetail,
  Feature,
  FeatureRange,
  LibraryRoot,
  Neighbor,
  PointCloud,
  PointColors,
  QueryFilter,
  RefitEvent,
  RefitParams,
  SampleDetail,
  ScanEvent,
  Tag,
};

/** `invoke`, with the rejection normalized to an `IpcError` carrying a typed `AppError`. */
async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  try {
    return await tauriInvoke<T>(command, args);
  } catch (raw) {
    throw new IpcError(toAppError(raw));
  }
}

// ── Library roots ───────────────────────────────────────────────────────────────

export function addLibraryRoot(path: string, label?: string): Promise<LibraryRoot> {
  return invoke('add_library_root', { path, label: label ?? null });
}

export function listLibraryRoots(): Promise<LibraryRoot[]> {
  return invoke('list_library_roots');
}

/** Removes a root and, by cascade, its samples, features, tags and coordinates. */
export function removeLibraryRoot(rootId: number): Promise<void> {
  return invoke('remove_library_root', { rootId });
}

/** Skips a root on future scans while keeping its rows, tags and place on the map. */
export function setRootEnabled(rootId: number, enabled: boolean): Promise<LibraryRoot> {
  return invoke('set_root_enabled', { rootId, enabled });
}

// ── Scanning ────────────────────────────────────────────────────────────────────

/**
 * Starts a scan and resolves with its `scanId` as soon as the run row exists — not when the
 * scan finishes.
 *
 * `onEvent` receives coalesced progress at no more than 10 Hz and is **guaranteed** a final
 * `finished` event, whether the scan completed, was cancelled, or failed
 * (`overview.md` §6.5). Render from the event stream, not from this promise.
 */
export function scanLibrary(
  rootId: number,
  onEvent: (event: ScanEvent) => void,
): Promise<number> {
  const channel = new Channel<ScanEvent>();
  channel.onmessage = onEvent;
  return invoke('scan_library', { rootId, onProgress: channel });
}

/**
 * Asks the running scan to stop.
 *
 * Cooperative: the scan finishes the file it is holding and keeps everything it has already
 * written. A half-scanned library is still useful, and a resumed scan skips what is already
 * done.
 */
export function cancelScan(scanId: number): Promise<void> {
  return invoke('cancel_scan', { scanId });
}

// ── The binary transports ───────────────────────────────────────────────────────

/**
 * Fetches the whole active layout in one call.
 *
 * Returns the decoded views **and** the buffer they are views onto. Hold the buffer: the
 * arrays borrow it, and a `webglcontextlost` rebuild is meant to reuse it rather than
 * refetch (`task.md` Phase 7).
 */
export async function getPointCloud(): Promise<{
  cloud: PointCloud;
  buffer: ArrayBuffer;
}> {
  const buffer = await invoke<ArrayBuffer>('get_point_cloud');
  return { cloud: decodePointCloud(buffer), buffer };
}

/**
 * Fetches the active layout's fit colors, in the point cloud's order.
 *
 * Pass the cloud's `count` to have that contract checked, same as {@link getFeatureColumn}.
 * Every channel is `NaN` for a run whose algorithm never produced one (PCA, UMAP, or a
 * t-SNE run from before this existed) — a real, common state, not an error.
 */
export async function getPointColors(expectedCount?: number): Promise<PointColors> {
  const buffer = await invoke<ArrayBuffer>('get_point_colors');
  return decodePointColors(buffer, expectedCount);
}

/**
 * Fetches one scalar column over the active layout, in the point cloud's order.
 *
 * Pass the cloud's `count` to have that contract checked. A mismatch throws
 * `WireFormatError` and means a re-fit landed between the two fetches — refetch the cloud.
 */
export async function getFeatureColumn(
  feature: Feature,
  expectedCount?: number,
): Promise<Float32Array> {
  const buffer = await invoke<ArrayBuffer>('get_feature_column', { feature });
  return decodeFeatureColumn(buffer, expectedCount);
}

/** An empty filter: matches the whole library. Spread it and override what you mean. */
export const NO_FILTER: QueryFilter = {
  rootIds: [],
  tags: [],
  collectionIds: [],
  exts: [],
  features: [],
  projectedOnly: false,
};

/**
 * Sample ids matching a filter, ascending.
 *
 * Ascending because the point cloud is too, so a filter mask is one merge over two sorted
 * arrays rather than a `Set` rebuilt on every keystroke.
 */
export async function querySamples(
  filter: Partial<QueryFilter> = {},
): Promise<Uint32Array> {
  const buffer = await invoke<ArrayBuffer>('query_samples', {
    filter: { ...NO_FILTER, ...filter },
  });
  return decodeIdList(buffer);
}

// ── Samples ─────────────────────────────────────────────────────────────────────

export function getSampleDetail(sampleId: number): Promise<SampleDetail> {
  return invoke('get_sample_detail', { sampleId });
}

/** Starts a native file drag so Finder, a DAW, or any other app receives the real sample. */
export function startSampleDrag(sampleId: number): Promise<void> {
  return invoke('start_sample_drag', { sampleId });
}

/** The `k` nearest samples by cosine similarity over the stored vectors. */
export function getSimilar(sampleId: number, k: number): Promise<Neighbor[]> {
  return invoke('get_similar', { sampleId, k });
}

/** Attaches a tag, creating it on first use. Resolves with the sample's full tag list. */
export function setTag(sampleId: number, tagName: string): Promise<string[]> {
  return invoke('set_tag', { sampleId, tagName });
}

/** Removes a tag from a sample. Resolves with the sample's remaining tags. */
export function unsetTag(sampleId: number, tagName: string): Promise<string[]> {
  return invoke('unset_tag', { sampleId, tagName });
}

export function listTags(): Promise<Tag[]> {
  return invoke('list_tags');
}

/** Sets (or clears, for `null`) a tag's display color. */
export function setTagColor(tagId: number, color: string | null): Promise<Tag> {
  return invoke('set_tag_color', { tagId, color });
}

/** Renames a tag while retaining its assignments and display color. */
export function renameTag(tagId: number, name: string): Promise<Tag> {
  return invoke('rename_tag', { tagId, name });
}

/** Removes a tag from the library and from every sound that carries it. */
export function deleteTag(tagId: number): Promise<void> {
  return invoke('delete_tag', { tagId });
}

/**
 * Selects the sample's file in Finder.
 *
 * By id, not by path: the frontend has no filesystem permission at all and never learns a
 * path it could have made up (`overview.md` §2).
 */
export function revealInFinder(sampleId: number): Promise<void> {
  return invoke('reveal_in_finder', { sampleId });
}

/**
 * Starts (or retriggers) previewing a sample.
 *
 * Safe to call as fast as hover events arrive: the engine always retriggers through a fresh
 * attack/release envelope rather than clicking, so debouncing here is about not machine-gunning
 * the decoder on a fast cursor sweep, not about avoiding an audible glitch (`task.md` Phase 8).
 * `gain` is clamped to a sane range on the Rust side regardless of what is passed.
 */
export function playSample(sampleId: number, gain: number): Promise<void> {
  return invoke('play_sample', { sampleId, gain });
}

/** Stops whatever is playing. A no-op, not an error, if nothing is. */
export function stopPlayback(): Promise<void> {
  return invoke('stop_playback');
}

/**
 * Warms the decode cache for a sample the user has not asked to hear yet, so that hovering it
 * next lands on a cache hit instead of paying a cold decode. Meant for neighbors of whatever is
 * currently hovered, not for anything the user has not shown interest in near yet -- a no-op on
 * the Rust side until a device has actually been opened by a real `playSample` call.
 */
export function prefetchSample(sampleId: number): Promise<void> {
  return invoke('prefetch_sample', { sampleId });
}

// ── Projection ──────────────────────────────────────────────────────────────────

/**
 * Rebuilds the 3D layout, streaming progress.
 *
 * Resolves with a **job id**, not a run id: the `projection_runs` row does not exist until
 * the coordinates do, which is minutes into a UMAP fit (`overview.md` §3.8). The run id
 * arrives in the terminal event, which is where it is wanted anyway — it identifies the map
 * that is now on screen.
 *
 * With `forceFull` unset the planner decides: a small import is placed into the existing
 * layout without moving one existing point, and only a large one triggers a re-fit.
 */
export function startRefit(
  params: Partial<RefitParams>,
  onEvent: (event: RefitEvent) => void,
): Promise<number> {
  const channel = new Channel<RefitEvent>();
  channel.onmessage = onEvent;
  return invoke('start_refit', {
    // No `algorithm` here: omitting it lets the backend's own default apply
    // (`RefitParams::default()`, currently t-SNE) rather than this call site pinning one
    // that would silently outlive whichever algorithm the backend considers current.
    params: { forceFull: false, ...params },
    onProgress: channel,
  });
}

export function cancelRefit(jobId: number): Promise<void> {
  return invoke('cancel_refit', { jobId });
}

// ── Collections ─────────────────────────────────────────────────────────────────

export function createCollection(name: string, sampleIds: number[]): Promise<Collection> {
  return invoke('create_collection', { name, sampleIds });
}

/** Every collection, newest first. */
export function listCollections(): Promise<Collection[]> {
  return invoke('list_collections');
}

/** One collection with its members, in the order the user arranged them. */
export function getCollection(collectionId: number): Promise<CollectionDetail> {
  return invoke('get_collection', { collectionId });
}

/** Appends a selection to an existing collection; members already present are kept once. */
export function addToCollection(
  collectionId: number,
  sampleIds: number[],
): Promise<CollectionDetail> {
  return invoke('add_to_collection', { collectionId, sampleIds });
}

/** Removes a sound from a collection without affecting the sound in the library. */
export function removeFromCollection(
  collectionId: number,
  sampleId: number,
): Promise<CollectionDetail> {
  return invoke('remove_from_collection', { collectionId, sampleId });
}

/** Changes a collection name while retaining its ordered members. */
export function renameCollection(
  collectionId: number,
  name: string,
): Promise<Collection> {
  return invoke('rename_collection', { collectionId, name });
}

/**
 * Rewrites a collection's member order. `sampleIds` must be exactly the collection's current
 * membership, reordered -- the core rejects anything else rather than silently reconciling it.
 */
export function reorderCollection(
  collectionId: number,
  sampleIds: number[],
): Promise<CollectionDetail> {
  return invoke('reorder_collection', { collectionId, sampleIds });
}

/** Deletes a collection. The samples themselves are untouched. */
export function deleteCollection(collectionId: number): Promise<void> {
  return invoke('delete_collection', { collectionId });
}

/**
 * Writes a collection's absolute file paths, one per line, to `destPath`.
 *
 * `destPath` must come from the frontend's own save dialog (`@tauri-apps/plugin-dialog`'s
 * `save()`) -- a path the user picked through native OS UI, the same shape `addLibraryRoot`
 * already takes. Never construct or guess one.
 */
export function exportCollection(collectionId: number, destPath: string): Promise<void> {
  return invoke('export_collection', { collectionId, destPath });
}

// ── Settings ────────────────────────────────────────────────────────────────────

/** The settings that survive a restart. */
export function getSettings(): Promise<AppSettings> {
  return invoke('get_settings');
}

/** Every audio output device this machine can see. */
export function listAudioDevices(): Promise<AudioDeviceInfo[]> {
  return invoke('list_audio_devices');
}

/** Persists a preferred output device, or `null` for the OS default, and switches live. */
export function setAudioDevice(name: string | null): Promise<AppSettings> {
  return invoke('set_audio_device', { name });
}

/** Persists the master gain the Settings slider remembers across restarts. */
export function setGain(gain: number): Promise<AppSettings> {
  return invoke('set_gain', { gain });
}

/** Reveals the app's data directory (the database, the embedding store) in Finder. */
export function revealDataDir(): Promise<void> {
  return invoke('reveal_data_dir');
}

/**
 * Deletes the whole library and relaunches the app.
 *
 * **This call's promise may never settle.** The process exits before Tauri can serialize a
 * reply back. Fire it, then show a static "restarting…" screen rather than awaiting it for
 * the happy path.
 */
export function resetDatabase(): Promise<void> {
  return invoke('reset_database');
}
