/**
 * Feature column to colour array (`task.md` Phase 7, "Color-by-feature").
 *
 * The column arrives from `get_feature_column` as one `Float32Array` in the cloud's order —
 * no ids, no objects, no parse. This turns it into the `count * 3` floats `aColor` wants,
 * once, on the main thread, and hands them to `CloudBuffers.setColors`. It is pure: no
 * Three.js, no GL, no DOM, which is what makes the mapping testable and what keeps the
 * decision about *what colour means* out of the render loop.
 *
 * Three things it gets right that a naïve `(v - min) / (max - min)` does not:
 *
 * - **`NaN` is a value.** `src/ipc/binary.ts` sends a null cell as `NaN` precisely so that
 *   "no BPM" cannot be confused with "0 BPM". A missing cell gets its own colour rather
 *   than landing at the bottom of the ramp and reading as the quietest sample in the
 *   library.
 * - **The domain is clipped to percentiles, not to extremes.** One mastering-reference WAV
 *   at 40 minutes compresses every drum hit in a 50,000-sample library into the first
 *   pixel of the ramp. Clipping at the 2nd and 98th centiles keeps the ramp spent on the
 *   data instead of on the outliers, and the outliers still read correctly — as the ends.
 * - **Some features are not a line.** Key root is a circle: C and B are neighbours, and a
 *   sequential ramp draws them at opposite ends. It gets a cyclic ramp, and the bounded
 *   features get their true bounds rather than whatever this particular library happens to
 *   contain.
 */

import type { Feature } from '../bindings/Feature';
import type { PointColors } from '../ipc/binary';

/** A colour ramp: linear RGB stops at even intervals. */
export interface Ramp {
  readonly name: string;
  readonly stops: readonly (readonly [number, number, number])[];
  /** True when the last stop is adjacent to the first — a hue circle, not a gradient. */
  readonly cyclic: boolean;
}

/**
 * What a point with no value for the active feature is painted.
 *
 * Raised with the background (`scene/SceneCanvas.tsx`): blending is additive, so a colour
 * this close to the clear colour used to make a missing cell nearly invisible against the
 * new grey. Still the quietest thing the ramp can paint, but a point rather than a hole.
 */
export const MISSING_COLOR: readonly [number, number, number] = [0.3, 0.32, 0.38];

/** Cool to hot. The default for anything that reads as an amount. */
export const EMBER: Ramp = {
  name: 'ember',
  cyclic: false,
  stops: [
    // The dark ends of both this ramp and ICE are lifted off where they started: additive
    // blending over the grey background adds these values to the clear colour, so a stop
    // near the background's own luminance had no contrast left to spend.
    [0.18, 0.24, 0.48],
    [0.29, 0.35, 0.68],
    [0.55, 0.33, 0.62],
    [0.85, 0.36, 0.42],
    [0.98, 0.6, 0.27],
    [0.99, 0.9, 0.61],
  ],
};

/** Deep to bright, without the warm end. For features where "hot" would mislead. */
export const ICE: Ramp = {
  name: 'ice',
  cyclic: false,
  stops: [
    [0.13, 0.21, 0.39],
    [0.16, 0.36, 0.56],
    [0.16, 0.5, 0.63],
    [0.36, 0.72, 0.71],
    [0.72, 0.9, 0.83],
  ],
};

/**
 * A hue circle, for features that wrap.
 *
 * Six stops with the seventh equal to the first, so `mapValue` can interpolate across the
 * seam without a special case.
 */
export const WHEEL: Ramp = {
  name: 'wheel',
  cyclic: true,
  stops: [
    [0.91, 0.35, 0.4],
    [0.9, 0.62, 0.3],
    [0.68, 0.82, 0.36],
    [0.34, 0.79, 0.62],
    [0.36, 0.6, 0.9],
    [0.7, 0.44, 0.88],
    [0.91, 0.35, 0.4],
  ],
};

/** Two well-separated colours, for a feature that is a flag. */
export const BINARY: Ramp = {
  name: 'binary',
  cyclic: false,
  stops: [
    [0.36, 0.6, 0.9],
    [0.95, 0.65, 0.32],
  ],
};

/**
 * The domain a feature is read over, where the schema fixes one.
 *
 * Returning a fixed domain matters for more than correctness of the ends: it makes the
 * colouring **comparable between libraries**. A key-root ramp derived from percentiles
 * would paint C green in one library and orange in another, and a legend could not be
 * written for either.
 */
export function featureDomain(feature: Feature): readonly [number, number] | null {
  switch (feature) {
    case 'keyRoot':
      // Twelve semitones on a circle. The upper bound is 12, not 11, so that B occupies
      // the last twelfth of the wheel rather than landing back on C.
      return [0, 12];
    case 'keyMode':
      return [0, 1];
    case 'keyConfidence':
    case 'bpmConfidence':
    case 'spectralFlatness':
      return [0, 1];
    default:
      return null;
  }
}

/** The ramp a feature reads best under. */
export function featureRamp(feature: Feature): Ramp {
  switch (feature) {
    case 'keyRoot':
      return WHEEL;
    case 'keyMode':
      return BINARY;
    case 'spectralCentroid':
    case 'spectralFlatness':
    case 'zeroCrossing':
      return ICE;
    default:
      return EMBER;
  }
}

export interface ColorMapping {
  /** `count * 3` linear RGB floats, in the cloud's order. */
  colors: Float32Array;
  /** The range the ramp was stretched over, after clipping. */
  domain: readonly [number, number];
  /** How many cells had no value. */
  missing: number;
}

export interface ColorOptions {
  ramp?: Ramp;
  /** Overrides the derived domain. `featureDomain` supplies this for bounded features. */
  domain?: readonly [number, number] | null;
  /** Lower and upper centiles the derived domain is clipped to. */
  clip?: readonly [number, number];
  /** Reused across calls to avoid a 600 KB allocation per colour-by change. */
  out?: Float32Array;
}

/**
 * Maps one feature column onto colours.
 *
 * Allocation-conscious on purpose: at 50,000 points the output is 600 KB, and a colour-by
 * control that a user drags through six features should not leave 3.6 MB for the collector
 * to find. Pass `out` to write in place.
 */
export function colorsFromColumn(
  values: Float32Array,
  options: ColorOptions = {},
): ColorMapping {
  const count = values.length;
  const ramp = options.ramp ?? EMBER;
  const out = options.out ?? new Float32Array(count * 3);
  if (out.length !== count * 3) {
    throw new RangeError(`out has ${out.length} floats for ${count} values`);
  }

  const domain = options.domain ?? deriveDomain(values, options.clip);
  const [low, high] = domain;
  // A column with one distinct value — every sample the same key, or a library of one —
  // would divide by zero. Everything lands mid-ramp, which is the honest picture of a
  // feature that does not vary.
  const span = high - low;
  const scale = span > 0 ? 1 / span : 0;

  let missing = 0;
  const rgb: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < count; i++) {
    const value = values[i] as number;
    if (Number.isNaN(value)) {
      missing++;
      out[i * 3] = MISSING_COLOR[0];
      out[i * 3 + 1] = MISSING_COLOR[1];
      out[i * 3 + 2] = MISSING_COLOR[2];
      continue;
    }
    const t = span > 0 ? clamp01((value - low) * scale) : 0.5;
    sampleRamp(ramp, t, rgb);
    out[i * 3] = rgb[0];
    out[i * 3 + 1] = rgb[1];
    out[i * 3 + 2] = rgb[2];
  }

  return { colors: out, domain, missing };
}

/**
 * Turns a decoded `ABPX` payload into the `count * 3` floats `aColor` wants.
 *
 * Unlike {@link colorsFromColumn}, there is no ramp and no domain to derive: the fit already
 * produced RGB directly (`tsne::fit_color`'s doc), so this is a straight interleave, plus the
 * same NaN-is-a-value handling for a point the active run never colored. `fallback` defaults
 * to {@link MISSING_COLOR} but is meant to be overridden — a fit color stands in for the
 * *ordinary* default point color, and a caller using it that way wants an uncolored point to
 * blend in, not to read as an error the way a missing feature value does.
 */
export function colorsFromFit(
  colors: PointColors,
  options: { fallback?: readonly [number, number, number]; out?: Float32Array } = {},
): Float32Array {
  const { count, r, g, b } = colors;
  const fallback = options.fallback ?? MISSING_COLOR;
  const out = options.out ?? new Float32Array(count * 3);
  if (out.length !== count * 3) {
    throw new RangeError(`out has ${out.length} floats for ${count} points`);
  }

  for (let i = 0; i < count; i++) {
    const rv = r[i] as number;
    if (Number.isNaN(rv)) {
      out[i * 3] = fallback[0];
      out[i * 3 + 1] = fallback[1];
      out[i * 3 + 2] = fallback[2];
      continue;
    }
    out[i * 3] = rv;
    out[i * 3 + 1] = g[i] as number;
    out[i * 3 + 2] = b[i] as number;
  }
  return out;
}

/**
 * The percentile-clipped range of a column, ignoring `NaN`.
 *
 * Sorts a compacted copy. At 50,000 values that is a few milliseconds, it happens once per
 * colour-by change and never during an orbit, and the alternative — a histogram with a
 * chosen bucket count — trades a real number for an approximate one to save time nobody is
 * short of here.
 */
export function deriveDomain(
  values: Float32Array,
  clip: readonly [number, number] = [0.02, 0.98],
): readonly [number, number] {
  const finite = new Float32Array(values.length);
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    const value = values[i] as number;
    if (Number.isFinite(value)) finite[n++] = value;
  }
  if (n === 0) return [0, 1];

  const sorted = finite.subarray(0, n).slice().sort();
  const at = (q: number) =>
    sorted[Math.min(n - 1, Math.max(0, Math.round(q * (n - 1))))] as number;
  const low = at(clip[0]);
  const high = at(clip[1]);
  // Clipping can collapse the range when the column is nearly constant; fall back to the
  // true extremes before giving up and returning a degenerate domain.
  if (high > low) return [low, high];
  const trueLow = sorted[0] as number;
  const trueHigh = sorted[n - 1] as number;
  return trueHigh > trueLow ? [trueLow, trueHigh] : [trueLow, trueLow];
}

/** Samples a ramp at `t` in `[0, 1]`, writing into `out` to avoid an allocation per point. */
export function sampleRamp(
  ramp: Ramp,
  t: number,
  out: [number, number, number],
): [number, number, number] {
  const stops = ramp.stops;
  const last = stops.length - 1;
  const scaled = clamp01(t) * last;
  const index = Math.min(last - 1, Math.floor(scaled));
  const frac = scaled - index;
  const a = stops[index] as readonly [number, number, number];
  const b = stops[index + 1] as readonly [number, number, number];
  out[0] = a[0] + (b[0] - a[0]) * frac;
  out[1] = a[1] + (b[1] - a[1]) * frac;
  out[2] = a[2] + (b[2] - a[2]) * frac;
  return out;
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}
