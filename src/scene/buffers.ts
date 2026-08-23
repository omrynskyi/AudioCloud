/**
 * The typed arrays behind the cloud, and the only code allowed to mutate them
 * (`overview.md` §5.2).
 *
 * One `BufferGeometry` for the entire library, built once from the `ArrayBuffer` the core
 * sent. No per-point JavaScript object is ever created — not on load, not on hover, not on
 * a filter change. That is cross-cutting rule 4 in mechanical form: everything that would
 * otherwise be React state for 50,000 things is a write into a `Float32Array` followed by
 * `needsUpdate = true`.
 *
 * Three attributes, three different lifetimes:
 *
 * - `position` is **static**. It changes only when the projection is replaced, and then the
 *   whole geometry is replaced with it.
 * - `aColor` and `aSize` are `DynamicDrawUsage`. Colour-by and filtering rewrite them
 *   wholesale, several times a second while a slider moves.
 * - `aId` is static and is the picking payload.
 *
 * **`aId` is the cloud index plus one, not the sample id.** Two reasons, and the second is
 * the load-bearing one. It makes the pick result an array index, so decoding a pick is a
 * subscript rather than a search. And it keeps the value inside the range a `float`
 * attribute represents exactly: `f32` is exact to 2^24, sample ids are `i64` from SQLite and
 * grow with every rescan of an edited library, and an id that exceeded 2^24 would be picked
 * as its neighbour with no error anywhere. Index zero is shifted to one so that an all-zero
 * pixel — the cleared background — cannot decode to a real point.
 */

import {
  BufferAttribute,
  BufferGeometry,
  DynamicDrawUsage,
  Sphere,
  Vector3,
} from 'three';

import { decodePointCloud, type PointCloud } from '../ipc/binary';

/** The extent of a layout, in the projector's own units. */
export interface CloudBounds {
  min: Vector3;
  max: Vector3;
  center: Vector3;
  /** Radius of the bounding sphere about `center`. Zero for an empty or single-point cloud. */
  radius: number;
}

export interface CloudBufferOptions {
  /**
   * World-space radius of an ordinary point, as a fraction of the cloud's bounding radius.
   *
   * Relative rather than absolute because coordinates arrive in whatever units the
   * projector produced — UMAP's embedded scale is derived from the local density of the
   * data and PCA's is the data's own variance. A constant here would give one algorithm
   * pinheads and the other beach balls.
   */
  radiusFraction?: number;
  /** How much a filtered-out point shrinks. */
  filteredScale?: number;
  /** How much of its colour a filtered-out point keeps. */
  filteredSaturation?: number;
  /** How much of its brightness a filtered-out point keeps. */
  filteredBrightness?: number;
}

const DEFAULTS = {
  radiusFraction: 1 / 260,
  filteredScale: 0.42,
  filteredSaturation: 0.12,
  filteredBrightness: 0.3,
} as const;

/** Rec. 709 luma, for the desaturation a filtered-out point gets. */
const LUMA = [0.2126, 0.7152, 0.0722] as const;

/** The colour a point has before anything has asked for a different one. */
export const DEFAULT_POINT_COLOR: readonly [number, number, number] = [0.55, 0.72, 0.95];

export class CloudBuffers {
  /** The payload every array here is derived from. **Held for context loss** — see `rebuild`. */
  readonly source: ArrayBuffer;
  /** The decoded views. `ids` is ascending; `x`/`y`/`z` are planar. */
  readonly cloud: PointCloud;
  readonly count: number;
  readonly bounds: CloudBounds;
  readonly options: Required<CloudBufferOptions>;

  geometry: BufferGeometry;

  /** World radius of an unfiltered point. Derived from `bounds` and `radiusFraction`. */
  readonly pointRadius: number;

  private positions: Float32Array;
  private colors: Float32Array;
  private sizes: Float32Array;
  private idAttribute: Float32Array;

  /**
   * Colour before filter dimming.
   *
   * Kept separately so a filter change does not have to recompute the colour-by mapping and
   * a colour-by change does not have to know the filter. Two intents, composed on write.
   */
  private baseColors: Float32Array;
  /** 1 where the point matches the active filter. `null` means no filter at all. */
  private mask: Uint8Array | null = null;

  private constructor(
    source: ArrayBuffer,
    cloud: PointCloud,
    options: CloudBufferOptions,
  ) {
    this.source = source;
    this.cloud = cloud;
    this.count = cloud.count;
    this.options = { ...DEFAULTS, ...options };

    this.positions = new Float32Array(this.count * 3);
    this.bounds = interleave(cloud, this.positions);
    this.pointRadius = Math.max(this.bounds.radius, 1e-6) * this.options.radiusFraction;

    this.colors = new Float32Array(this.count * 3);
    this.baseColors = new Float32Array(this.count * 3);
    this.sizes = new Float32Array(this.count);
    this.idAttribute = new Float32Array(this.count);
    for (let i = 0; i < this.count; i++) this.idAttribute[i] = i + 1;

    // Defaults written straight into the arrays, before the geometry exists. Going through
    // `setUniformColor` here would be the tidier-looking spelling and would throw: it ends
    // in `needsUpdate` on an attribute that has not been created yet.
    for (let i = 0; i < this.count; i++) {
      this.baseColors[i * 3] = DEFAULT_POINT_COLOR[0];
      this.baseColors[i * 3 + 1] = DEFAULT_POINT_COLOR[1];
      this.baseColors[i * 3 + 2] = DEFAULT_POINT_COLOR[2];
    }
    this.colors.set(this.baseColors);
    this.sizes.fill(this.pointRadius);

    this.geometry = this.buildGeometry();
  }

  /** Decodes an `ABPC` payload and builds the geometry for it. */
  static fromBuffer(source: ArrayBuffer, options: CloudBufferOptions = {}): CloudBuffers {
    return new CloudBuffers(source, decodePointCloud(source), options);
  }

  private buildGeometry(): BufferGeometry {
    const geometry = new BufferGeometry();

    geometry.setAttribute('position', new BufferAttribute(this.positions, 3));

    const color = new BufferAttribute(this.colors, 3);
    color.setUsage(DynamicDrawUsage);
    geometry.setAttribute('aColor', color);

    const size = new BufferAttribute(this.sizes, 1);
    size.setUsage(DynamicDrawUsage);
    geometry.setAttribute('aSize', size);

    geometry.setAttribute('aId', new BufferAttribute(this.idAttribute, 1));

    // Computed here rather than left to three's lazy path, because the interleave pass
    // already walked every coordinate and the lazy path would walk them again — and because
    // a `Points` object with no bounding sphere is frustum-culled as if it were at the
    // origin, which for a cloud whose centroid is anywhere else means it vanishes.
    geometry.boundingSphere = new Sphere(this.bounds.center.clone(), this.bounds.radius);
    geometry.setDrawRange(0, this.count);
    return geometry;
  }

  /**
   * Rebuilds the geometry from the cached payload, for `webglcontextrestored`.
   *
   * The point of holding `source`: the position data never left the process, so recovering
   * from a lost context is a decode and an interleave — about two milliseconds at 50,000
   * points, measured in `BENCHMARKS.md` — and not a refetch. Colour and filter state
   * survive untouched, because they live in arrays this method reuses rather than rebuilds.
   *
   * Returns the new geometry; the caller must assign it to the `Points` object.
   */
  rebuild(): BufferGeometry {
    this.geometry.dispose();
    // Re-derive positions from the cached bytes rather than trusting the array in hand.
    // A rebuild that reused it would still work today and would silently stop being a test
    // of the cache the moment something upstream started mutating positions.
    const cloud = decodePointCloud(this.source);
    interleave(cloud, this.positions);
    this.geometry = this.buildGeometry();
    return this.geometry;
  }

  /** Flags every attribute for re-upload, without reallocating anything. */
  markAllForUpload(): void {
    for (const name of ['position', 'aColor', 'aSize', 'aId']) {
      const attribute = this.geometry.getAttribute(name);
      if (attribute) attribute.needsUpdate = true;
    }
  }

  // ── Colour ────────────────────────────────────────────────────────────────────

  /** Paints every point the same colour. The state before a colour-by is chosen. */
  setUniformColor(rgb: readonly [number, number, number]): void {
    const [r, g, b] = rgb;
    for (let i = 0; i < this.count; i++) {
      this.baseColors[i * 3] = r;
      this.baseColors[i * 3 + 1] = g;
      this.baseColors[i * 3 + 2] = b;
    }
    this.applyColors();
  }

  /**
   * Takes a caller-computed colour array as the base colour.
   *
   * `rgb` must be `count * 3` floats in the cloud's order — which is what
   * `scene/colors.ts` produces from a feature column, and what makes the colour mapping
   * testable without a GPU.
   */
  setColors(rgb: Float32Array): void {
    if (rgb.length !== this.count * 3) {
      throw new RangeError(
        `colour array has ${rgb.length} floats for ${this.count} points; expected ${this.count * 3}`,
      );
    }
    this.baseColors.set(rgb);
    this.applyColors();
  }

  // ── Filtering ─────────────────────────────────────────────────────────────────

  /**
   * Applies a filter mask over the cloud, or clears it with `null`.
   *
   * Filtered-out points **shrink and desaturate rather than disappear**. The corpus has a
   * shape, that shape is the thing the map is for, and a filter that deletes 90% of it
   * leaves the remaining points floating in a void with no way to read where they sit
   * relative to everything else.
   */
  setMask(mask: Uint8Array | null): void {
    if (mask && mask.length !== this.count) {
      throw new RangeError(`mask has ${mask.length} entries for ${this.count} points`);
    }
    this.mask = mask;
    this.applyColors();
    this.applySizes();
  }

  private applyColors(): void {
    const { filteredSaturation, filteredBrightness } = this.options;
    const { mask, baseColors, colors, count } = this;

    if (!mask) {
      colors.set(baseColors);
    } else {
      for (let i = 0; i < count; i++) {
        const o = i * 3;
        const r = baseColors[o] as number;
        const g = baseColors[o + 1] as number;
        const b = baseColors[o + 2] as number;
        if (mask[i] === 1) {
          colors[o] = r;
          colors[o + 1] = g;
          colors[o + 2] = b;
        } else {
          const luma = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b;
          colors[o] = (luma + (r - luma) * filteredSaturation) * filteredBrightness;
          colors[o + 1] = (luma + (g - luma) * filteredSaturation) * filteredBrightness;
          colors[o + 2] = (luma + (b - luma) * filteredSaturation) * filteredBrightness;
        }
      }
    }
    const attribute = this.geometry.getAttribute('aColor');
    if (attribute) attribute.needsUpdate = true;
  }

  private applySizes(): void {
    const { filteredScale } = this.options;
    const { mask, sizes, pointRadius, count } = this;
    if (!mask) {
      sizes.fill(pointRadius);
    } else {
      const dimmed = pointRadius * filteredScale;
      for (let i = 0; i < count; i++) sizes[i] = mask[i] === 1 ? pointRadius : dimmed;
    }
    const attribute = this.geometry.getAttribute('aSize');
    if (attribute) attribute.needsUpdate = true;
  }

  // ── Identity ──────────────────────────────────────────────────────────────────

  /** The sample at a cloud index, or `undefined` if the index is out of range. */
  sampleAt(index: number): number | undefined {
    return this.cloud.ids[index];
  }

  /**
   * The cloud index of a sample id, or `-1`.
   *
   * A binary search rather than a `Map`: the ids arrive ascending — `get_point_cloud`
   * orders by `sample_id` and `src/ipc/binary.ts` documents it as a contract — so the
   * lookup is seventeen comparisons at 50,000 points against a two-megabyte hash table that
   * would have to be rebuilt on every projection swap.
   */
  indexOfSample(sampleId: number): number {
    const ids = this.cloud.ids;
    let low = 0;
    let high = ids.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      const value = ids[mid] as number;
      if (value === sampleId) return mid;
      if (value < sampleId) low = mid + 1;
      else high = mid - 1;
    }
    return -1;
  }

  dispose(): void {
    this.geometry.dispose();
  }
}

/**
 * Weaves the three planar columns into one interleaved position attribute, and takes the
 * bounds in the same pass.
 *
 * This loop is the cost of the struct-of-arrays wire format, and it is a deliberate trade
 * rather than an oversight: planar columns are what let `get_feature_column` ship one
 * column on its own when the colouring changes, instead of resending the cloud. Measured at
 * 2.0 ms for 50,000 points in `BENCHMARKS.md`, against a 300 ms load-to-first-frame budget.
 *
 * The bounds come along for free here. Walking 150,000 floats a second time to find them
 * would cost as much as the interleave itself.
 */
function interleave(cloud: PointCloud, out: Float32Array): CloudBounds {
  const { count, x, y, z } = cloud;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (let i = 0; i < count; i++) {
    const px = x[i] as number;
    const py = y[i] as number;
    const pz = z[i] as number;
    out[i * 3] = px;
    out[i * 3 + 1] = py;
    out[i * 3 + 2] = pz;
    if (px < minX) minX = px;
    if (py < minY) minY = py;
    if (pz < minZ) minZ = pz;
    if (px > maxX) maxX = px;
    if (py > maxY) maxY = py;
    if (pz > maxZ) maxZ = pz;
  }

  if (count === 0) {
    const zero = new Vector3();
    return { min: zero.clone(), max: zero.clone(), center: zero.clone(), radius: 0 };
  }

  const min = new Vector3(minX, minY, minZ);
  const max = new Vector3(maxX, maxY, maxZ);
  const center = min.clone().add(max).multiplyScalar(0.5);
  // The half-diagonal, not the largest half-extent: the sphere has to contain the corners.
  return { min, max, center, radius: max.distanceTo(min) * 0.5 };
}
