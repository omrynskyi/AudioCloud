/**
 * Synthetic clouds, in the wire format.
 *
 * `task.md` allows Phase 7 to be built "against synthetic point data", and the frame-time
 * exit criterion needs 50,000 points on demand, deterministically, on a machine that has
 * not scanned a library. This produces them as an actual `ABPC` payload rather than as
 * loose arrays, so the harness exercises `decodePointCloud`, the interleave, the bounds
 * pass and the geometry build — the same code path the app runs, not a shortcut around it.
 * A profile of a path the app does not take is not a profile.
 *
 * The shape matters as much as the count. Points scattered uniformly in a cube would
 * understate the cost badly: what makes this scene expensive is **overdraw**, and overdraw
 * comes from clustering. UMAP output is a handful of dense blobs joined by thin filaments,
 * with the dense cores several hundred points deep along the view axis, so that is what is
 * generated here. The same structure is what makes the picking criterion — "the correct
 * sample under a dense cluster" — testable at all.
 *
 * Deterministic by seed, because a performance number from an input that changes between
 * runs is not comparable with the one before it.
 */

import {
  HEADER_BYTES,
  WIRE_VERSION,
  decodePointCloud,
  type PointCloud,
} from '../ipc/binary';

export interface SyntheticOptions {
  count?: number;
  clusters?: number;
  seed?: number;
  /** Fraction of points laid along filaments between clusters rather than inside one. */
  filamentFraction?: number;
}

const DEFAULTS = {
  count: 50_000,
  clusters: 14,
  seed: 0x5eed,
  filamentFraction: 0.18,
} as const;

/**
 * A deterministic 32-bit PRNG.
 *
 * `Math.random` cannot be seeded, and an unseeded profile input means two runs of the
 * harness measure two different scenes.
 */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller, so cluster cores are actually Gaussian rather than a uniform ball. */
function gaussian(random: () => number): number {
  const u = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/**
 * Builds an `ABPC` payload for a synthetic library.
 *
 * Returns the buffer and its decoded views, because every caller wants both and decoding
 * twice would allocate a second set of views over the same bytes.
 */
export function syntheticPointCloud(options: SyntheticOptions = {}): {
  buffer: ArrayBuffer;
  cloud: PointCloud;
} {
  const { count, clusters, seed, filamentFraction } = { ...DEFAULTS, ...options };
  const random = mulberry32(seed);

  const buffer = new ArrayBuffer(HEADER_BYTES + count * 16);
  const header = new DataView(buffer);
  header.setUint8(0, 0x41); // A
  header.setUint8(1, 0x42); // B
  header.setUint8(2, 0x50); // P
  header.setUint8(3, 0x43); // C
  header.setUint32(4, WIRE_VERSION, true);
  header.setUint32(8, count, true);
  header.setUint32(12, 0, true);

  const ids = new Uint32Array(buffer, HEADER_BYTES, count);
  const x = new Float32Array(buffer, HEADER_BYTES + count * 4, count);
  const y = new Float32Array(buffer, HEADER_BYTES + count * 8, count);
  const z = new Float32Array(buffer, HEADER_BYTES + count * 12, count);

  // Cluster centres on a shell, so the cloud reads as a body rather than a slab and an
  // orbit passes through the same range of depths from every angle.
  const centres: [number, number, number, number][] = [];
  for (let c = 0; c < clusters; c++) {
    const theta = random() * Math.PI * 2;
    const phi = Math.acos(2 * random() - 1);
    const radius = 4 + random() * 6;
    centres.push([
      radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.sin(phi) * Math.sin(theta),
      radius * Math.cos(phi),
      0.5 + random() * 1.4, // this cluster's spread
    ]);
  }

  const filamentCount = Math.floor(count * filamentFraction);
  for (let i = 0; i < count; i++) {
    // Sample ids ascending and sparse: real libraries have gaps from deletions and
    // deduplication, and code that quietly assumed `id === index` would pass against a
    // dense range and fail on a real one.
    ids[i] = i * 3 + 11;

    if (i < filamentCount) {
      const a = centres[Math.floor(random() * clusters)] as [
        number,
        number,
        number,
        number,
      ];
      const b = centres[Math.floor(random() * clusters)] as [
        number,
        number,
        number,
        number,
      ];
      const t = random();
      const jitter = 0.35;
      x[i] = a[0] + (b[0] - a[0]) * t + gaussian(random) * jitter;
      y[i] = a[1] + (b[1] - a[1]) * t + gaussian(random) * jitter;
      z[i] = a[2] + (b[2] - a[2]) * t + gaussian(random) * jitter;
    } else {
      const c = centres[Math.floor(random() * clusters)] as [
        number,
        number,
        number,
        number,
      ];
      const spread = c[3];
      x[i] = c[0] + gaussian(random) * spread;
      y[i] = c[1] + gaussian(random) * spread;
      z[i] = c[2] + gaussian(random) * spread;
    }
  }

  return { buffer, cloud: decodePointCloud(buffer) };
}

/**
 * A synthetic feature column for the cloud above, with a realistic share of nulls.
 *
 * Correlated with position, because that is what a real feature column is — the map exists
 * to put similar sounds near each other — and because a column of noise would make a
 * colour-by look broken rather than look like nothing.
 */
export function syntheticFeatureColumn(
  cloud: PointCloud,
  {
    seed = 0xc01,
    missingFraction = 0.07,
  }: { seed?: number; missingFraction?: number } = {},
): Float32Array {
  const random = mulberry32(seed);
  const values = new Float32Array(cloud.count);
  for (let i = 0; i < cloud.count; i++) {
    values[i] =
      random() < missingFraction
        ? Number.NaN
        : 60 +
          (cloud.x[i] as number) * 6 +
          (cloud.y[i] as number) * 2 +
          gaussian(random) * 4;
  }
  return values;
}
