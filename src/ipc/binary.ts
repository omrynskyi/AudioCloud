/**
 * Decoders for the raw-byte transport (`overview.md` §6.3).
 *
 * The Rust half of this format lives in `src-tauri/src/ipc/binary.rs`, and there is no way
 * to share one implementation across a process boundary — so the two are kept honest by the
 * version word and by `src-tauri/tests/ipc.rs`, which asserts on the exact byte offsets this
 * file reads.
 *
 * Every payload starts with the same sixteen-byte header:
 *
 * ```text
 * [0..4)   magic, four ASCII bytes
 * [4..8)   version, u32 LE
 * [8..12)  count, u32 LE
 * [12..16) reserved, u32 LE
 * ```
 *
 * **Sixteen bytes is what makes the views legal.** `new Float32Array(buffer, offset, n)`
 * throws a `RangeError` if `offset` is not a multiple of four; sixteen is a multiple of
 * every alignment a typed array can ask for, so every column start is aligned by
 * construction and `assertAligned` below is a guard against a future header change rather
 * than against arithmetic.
 *
 * Reading a `DataView` for the header and typed arrays for the body is deliberate: the
 * header is explicitly little-endian, and the body is host-endian, which is the same thing
 * on every machine this ships to. Doing the header through a `Uint32Array` would work today
 * and silently produce nonsense on a big-endian host; doing the body through a `DataView`
 * would mean 200,000 method calls where one view suffices.
 */

/** Wire format version. Must match `binary::VERSION` in Rust. */
export const WIRE_VERSION = 1;

/** Bytes before the first column. */
export const HEADER_BYTES = 16;

const MAGIC = {
  pointCloud: 'ABPC',
  featureColumn: 'ABFC',
  queryResult: 'ABQS',
  peaks: 'ABPK',
} as const;

/** A payload that is not what this build knows how to read. */
export class WireFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WireFormatError';
  }
}

interface Header {
  magic: string;
  version: number;
  count: number;
  reserved: number;
}

function readHeader(buffer: ArrayBuffer, expected: string): Header {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new WireFormatError(
      `expected at least ${HEADER_BYTES} bytes, got ${buffer.byteLength}`,
    );
  }

  const view = new DataView(buffer);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (magic !== expected) {
    throw new WireFormatError(
      `expected a ${expected} payload, got ${JSON.stringify(magic)}`,
    );
  }

  const version = view.getUint32(4, true);
  if (version !== WIRE_VERSION) {
    // Deliberately not "try to read it anyway". A version bump means a body layout an old
    // decoder would misread as plausible numbers, which is worse than a visible failure.
    throw new WireFormatError(
      `${expected} payload is version ${version}; this build reads version ${WIRE_VERSION}. ` +
        'The app and its core are out of step — reinstall rather than reload.',
    );
  }

  return {
    magic,
    version,
    count: view.getUint32(8, true),
    reserved: view.getUint32(12, true),
  };
}

/** Guards the invariant that makes a typed-array view constructible at all. */
function assertAligned(offset: number, name: string): void {
  if (offset % 4 !== 0) {
    throw new WireFormatError(
      `${name} starts at byte ${offset}, which is not 4-byte aligned; the header layout changed`,
    );
  }
}

function assertLength(buffer: ArrayBuffer, needed: number, what: string): void {
  if (buffer.byteLength < needed) {
    throw new WireFormatError(
      `a ${what} payload of ${buffer.byteLength} bytes is short of the ${needed} its header claims`,
    );
  }
}

/** The active layout: one id per point and three planar coordinate columns. */
export interface PointCloud {
  count: number;
  /** Sample ids, ascending — the same order as `queryIds` returns. */
  ids: Uint32Array;
  x: Float32Array;
  y: Float32Array;
  z: Float32Array;
}

/**
 * Decodes an `ABPC` payload.
 *
 * The four returned arrays are **views onto the payload**, not copies: nothing is allocated
 * beyond four view objects, which is the entire reason this transport exists. Keep the
 * `ArrayBuffer` alive as long as the views are in use — `scene/buffers.ts` caches it, which
 * is also what lets a `webglcontextlost` rebuild happen without refetching.
 */
export function decodePointCloud(buffer: ArrayBuffer): PointCloud {
  const { count } = readHeader(buffer, MAGIC.pointCloud);
  assertLength(buffer, HEADER_BYTES + count * 16, 'point cloud');

  const ids = HEADER_BYTES;
  const x = ids + count * 4;
  const y = x + count * 4;
  const z = y + count * 4;
  for (const [offset, name] of [
    [ids, 'the id column'],
    [x, 'the x column'],
    [y, 'the y column'],
    [z, 'the z column'],
  ] as const) {
    assertAligned(offset, name);
  }

  return {
    count,
    ids: new Uint32Array(buffer, ids, count),
    x: new Float32Array(buffer, x, count),
    y: new Float32Array(buffer, y, count),
    z: new Float32Array(buffer, z, count),
  };
}

/**
 * Decodes an `ABFC` payload into one `Float32Array`, in the point cloud's order.
 *
 * The column carries no ids — index `i` is the sample at index `i` of the cloud. Pass the
 * cloud's `count` to have that contract checked: a mismatch means a re-fit landed between
 * the two fetches, and the caller should refetch the cloud rather than colour it by
 * somebody else's numbers.
 *
 * A null cell arrives as `NaN`. Not zero: zero is a legitimate value for every column in
 * the schema, and "no BPM" is not "0 BPM".
 */
export function decodeFeatureColumn(
  buffer: ArrayBuffer,
  expectedCount?: number,
): Float32Array {
  const { count } = readHeader(buffer, MAGIC.featureColumn);
  assertLength(buffer, HEADER_BYTES + count * 4, 'feature column');
  if (expectedCount !== undefined && count !== expectedCount) {
    throw new WireFormatError(
      `feature column has ${count} values for a point cloud of ${expectedCount}; ` +
        'the projection changed between the two fetches',
    );
  }

  assertAligned(HEADER_BYTES, 'the value column');
  return new Float32Array(buffer, HEADER_BYTES, count);
}

/**
 * Decodes an `ABQS` payload: matching sample ids, ascending.
 *
 * Ascending is the contract the renderer's filter mask depends on — see `maskFrom` in
 * `src/scene/`, which merges this against the cloud's own ascending ids instead of building
 * a `Set` on every keystroke.
 */
export function decodeIdList(buffer: ArrayBuffer): Uint32Array {
  const { count } = readHeader(buffer, MAGIC.queryResult);
  assertLength(buffer, HEADER_BYTES + count * 4, 'query result');
  assertAligned(HEADER_BYTES, 'the id column');
  return new Uint32Array(buffer, HEADER_BYTES, count);
}

/** A waveform summary: one `(min, max)` pair per bucket. */
export interface Peaks {
  /** Interleaved `min, max, min, max, …` — two floats per bucket. */
  minMax: Float32Array;
  buckets: number;
  /**
   * Milliseconds of audio the buckets actually span.
   *
   * Below the sample's `durationMs` whenever the decoder stopped at its ten-second window.
   * Draw the waveform over this span, not over the file's length, or a four-minute loop gets
   * a picture of its first ten seconds stretched edge to edge.
   */
  coveredMs: number;
}

/** Decodes an `ABPK` payload, as fetched from `abpeaks://`. */
export function decodePeaks(buffer: ArrayBuffer): Peaks {
  const { count, reserved } = readHeader(buffer, MAGIC.peaks);
  assertLength(buffer, HEADER_BYTES + count * 8, 'peaks');
  assertAligned(HEADER_BYTES, 'the peak column');
  return {
    minMax: new Float32Array(buffer, HEADER_BYTES, count * 2),
    buckets: count,
    coveredMs: reserved,
  };
}
