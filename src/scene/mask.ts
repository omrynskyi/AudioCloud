/**
 * The filter mask: which points the active query matched.
 *
 * `query_samples` returns matching sample ids **ascending**, and `get_point_cloud` returns
 * its points ordered by sample id. Both halves of that were written for this function —
 * `src/ipc/binary.ts` calls it out as the contract `decodeIdList` exists to keep — because
 * two sorted arrays merge in one linear pass with no allocation, and the obvious
 * alternative does not.
 *
 * The obvious alternative is `new Set(matchedIds)` and a lookup per point. At 30,000
 * matches that is a hash table built from scratch on **every keystroke** in the search
 * field: roughly 1.5 MB of transient objects, a rehash partway through, and a collection
 * pause landing in the middle of the typing it was caused by. The merge below allocates one
 * `Uint8Array` — 50 KB — and can be handed the same one every time.
 */

/**
 * Marks each cloud index 1 where the query matched it and 0 where it did not.
 *
 * Both inputs must be ascending; `matchedIds` may contain ids that are not in the cloud at
 * all — `query_samples` answers over the whole library unless `projectedOnly` is set, and
 * the renderer should not have to ask for a narrower query just to build a mask.
 *
 * Pass `out` to reuse a buffer across calls.
 */
export function maskFrom(
  cloudIds: Uint32Array,
  matchedIds: Uint32Array,
  out?: Uint8Array,
): Uint8Array {
  const count = cloudIds.length;
  const mask = out && out.length === count ? out : new Uint8Array(count);
  mask.fill(0);

  let i = 0;
  let j = 0;
  while (i < count && j < matchedIds.length) {
    const id = cloudIds[i] as number;
    const matched = matchedIds[j] as number;
    if (id === matched) {
      mask[i] = 1;
      i++;
      j++;
    } else if (id < matched) {
      i++;
    } else {
      j++;
    }
  }
  return mask;
}
