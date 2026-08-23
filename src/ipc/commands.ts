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

import type { Collection } from '../bindings/Collection';
import type { DownloadEvent } from '../bindings/DownloadEvent';
import type { Feature } from '../bindings/Feature';
import type { LibraryRoot } from '../bindings/LibraryRoot';
import type { ModelStatus } from '../bindings/ModelStatus';
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
  type PointCloud,
} from './binary';
import { IpcError, toAppError } from './errors';

export type {
  Collection,
  DownloadEvent,
  Feature,
  LibraryRoot,
  ModelStatus,
  Neighbor,
  PointCloud,
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

export function createCollection(name: string, sampleIds: number[]): Promise<Collection> {
  return invoke('create_collection', { name, sampleIds });
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
 * Starts previewing a sample.
 *
 * **Rejects with `{ kind: 'unavailable' }` until Phase 8 builds the audio engine.** The
 * command's contract is settled, which is why it is here; render the transport controls
 * disabled on that variant rather than assuming it works.
 */
export function playSample(sampleId: number, gain: number): Promise<void> {
  return invoke('play_sample', { sampleId, gain });
}

/** Stops playback. See `playSample` on why this is not built yet. */
export function stopPlayback(): Promise<void> {
  return invoke('stop_playback');
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
    params: { algorithm: 'umap', forceFull: false, ...params },
    onProgress: channel,
  });
}

export function cancelRefit(jobId: number): Promise<void> {
  return invoke('cancel_refit', { jobId });
}

// ── Model ───────────────────────────────────────────────────────────────────────

/** Cheap enough to poll: no network, no filesystem beyond three stats, no session init. */
export function getModelStatus(): Promise<ModelStatus> {
  return invoke('get_model_status');
}

/**
 * Downloads and verifies the model, streaming progress.
 *
 * Resolves as soon as the download is admitted. The outcome — including a checksum
 * mismatch, which is its own `AppError` variant — arrives as the terminal event.
 */
export function downloadModel(onEvent: (event: DownloadEvent) => void): Promise<void> {
  const channel = new Channel<DownloadEvent>();
  channel.onmessage = onEvent;
  return invoke('download_model', { onProgress: channel });
}

/** Stops the download, keeping the partial so the next call resumes rather than restarts. */
export function cancelDownload(): Promise<void> {
  return invoke('cancel_download');
}
