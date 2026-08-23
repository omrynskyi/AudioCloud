/**
 * Fetching waveform summaries over the `abpeaks://` scheme (`overview.md` §6.4).
 *
 * Transport 3. Peaks are per-sample, requested on selection, and pure binary — routing them
 * through `invoke` would make them queue behind the point cloud and the filter query on the
 * IPC handler for a payload the WebView is perfectly able to cache itself.
 */

import { decodePeaks, type Peaks } from './binary';
import type { SampleDetail } from '../bindings/SampleDetail';

/**
 * Builds the URL for a sample's waveform.
 *
 * **`updatedAt` is not decoration.** The scheme serves `Cache-Control: immutable`, which
 * tells the WebView it never has to revalidate — true for "sample 1234 as it was at
 * revision N", false for "sample 1234", because a rescanned file is different audio under
 * the same id. The full URL is the cache key, so putting the revision in it is what makes
 * the immutable promise honest. Take the value from `SampleDetail.updatedAt`.
 */
export function peaksUrl(sampleId: number, updatedAt?: number): string {
  const base = `abpeaks://localhost/${sampleId}`;
  return updatedAt === undefined ? base : `${base}?v=${updatedAt}`;
}

/**
 * Fetches and decodes a sample's waveform summary.
 *
 * Returns `null` for a sample with no waveform — one that does not exist, or one whose audio
 * will not decode. Both are states the inspector already renders from `getSampleDetail`, and
 * neither is worth a thrown error at a call site that is drawing a picture.
 *
 * Pass an `AbortSignal` when the selection can change mid-flight, which it can: the user
 * dragging through a cluster starts one of these per point.
 */
export async function fetchPeaks(
  sample: Pick<SampleDetail, 'id' | 'updatedAt'>,
  signal?: AbortSignal,
): Promise<Peaks | null> {
  const response = await fetch(peaksUrl(sample.id, sample.updatedAt), {
    signal: signal ?? null,
  });
  if (!response.ok) return null;
  return decodePeaks(await response.arrayBuffer());
}
