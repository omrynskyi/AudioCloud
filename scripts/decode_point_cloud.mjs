/**
 * Times the JavaScript half of `overview.md` §6.3 against the Rust half's actual output.
 *
 * `task.md` Phase 6's exit criterion is that the 50,000-point cloud is "decoded into typed
 * arrays in < 300 ms". That is one number spanning two languages, and measuring it inside
 * the app would fold in the WebView's IPC hop and Phase 7's geometry build — neither of
 * which Phase 6 owns, and both of which have their own rows in `overview.md` §7. So this
 * measures exactly the part that is Phase 6's: taking the bytes the Rust encoder produced
 * and turning them into the four typed arrays the renderer will read.
 *
 * ```sh
 * cargo test --manifest-path src-tauri/Cargo.toml --profile perf --test benchmarks \
 *     -- --ignored --nocapture serve_a_fifty_thousand_point_cloud
 * node scripts/decode_point_cloud.mjs
 * ```
 *
 * The benchmark writes `src-tauri/target/point_cloud_50k.bin`; this reads it. Two halves of
 * one criterion measured against the same bytes, rather than against two independent
 * fabrications of what the format is supposed to be — which is the failure mode a
 * hand-written fixture on this side would have.
 *
 * Plain `.mjs` and no dependencies on purpose: this is a measurement, not a test, and it
 * must not need the app's toolchain to run. The decode logic is duplicated from
 * `src/ipc/binary.ts` rather than imported, which is a real cost — kept honest by the
 * assertion below that the arrays it produces match the format's stated arithmetic.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HEADER_BYTES = 16;
const WIRE_VERSION = 1;
const RUNS = 50;

const here = dirname(fileURLToPath(import.meta.url));
const payloadPath = join(here, '..', 'src-tauri', 'target', 'point_cloud_50k.bin');

let file;
try {
  file = readFileSync(payloadPath);
} catch {
  console.error(`no payload at ${payloadPath}`);
  console.error(
    'run: cargo test --manifest-path src-tauri/Cargo.toml --profile perf --test benchmarks \\\n' +
      '         -- --ignored --nocapture serve_a_fifty_thousand_point_cloud',
  );
  process.exit(1);
}

// `readFileSync` hands back a Buffer that may be a view into a larger pooled ArrayBuffer, so
// slicing to an exact standalone buffer is what makes the offsets below mean what they say.
const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);

function decodePointCloud(buf) {
  const view = new DataView(buf);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (magic !== 'ABPC')
    throw new Error(`expected an ABPC payload, got ${JSON.stringify(magic)}`);
  const version = view.getUint32(4, true);
  if (version !== WIRE_VERSION)
    throw new Error(`wire version ${version}, expected ${WIRE_VERSION}`);
  const count = view.getUint32(8, true);

  const ids = HEADER_BYTES;
  const x = ids + count * 4;
  const y = x + count * 4;
  const z = y + count * 4;
  return {
    count,
    ids: new Uint32Array(buf, ids, count),
    x: new Float32Array(buf, x, count),
    y: new Float32Array(buf, y, count),
    z: new Float32Array(buf, z, count),
  };
}

const first = decodePointCloud(buffer);
if (buffer.byteLength !== HEADER_BYTES + first.count * 16) {
  throw new Error(
    `payload length ${buffer.byteLength} disagrees with a count of ${first.count}`,
  );
}

const timings = [];
for (let i = 0; i < RUNS; i++) {
  const started = performance.now();
  const cloud = decodePointCloud(buffer);
  // Touch one element per column so a future engine cannot optimize the views away.
  const guard = cloud.ids[0] + cloud.x[0] + cloud.y[0] + cloud.z[0];
  timings.push(performance.now() - started);
  if (!Number.isFinite(guard)) throw new Error('decoded garbage');
}
timings.sort((a, b) => a - b);

// The interleave `scene/buffers.ts` will do on arrival (Phase 7), measured separately
// because it is a real cost this transport deliberately moved to the frontend — the wire
// stays struct-of-arrays so a single column can be fetched on its own.
const interleaveStarted = performance.now();
const positions = new Float32Array(first.count * 3);
for (let i = 0; i < first.count; i++) {
  positions[i * 3] = first.x[i];
  positions[i * 3 + 1] = first.y[i];
  positions[i * 3 + 2] = first.z[i];
}
const interleave = performance.now() - interleaveStarted;

const ms = (n) => `${n.toFixed(3)} ms`;
console.log(
  `\n-- decode ${first.count} points from ${buffer.byteLength} bytes (${RUNS} runs) --`,
);
console.log(`  median             ${ms(timings[Math.floor(RUNS / 2)])}`);
console.log(`  p99                ${ms(timings[Math.floor(RUNS * 0.99)])}`);
console.log(`  worst              ${ms(timings[RUNS - 1])}`);
console.log(
  `  interleave to XYZ  ${ms(interleave)}   (Phase 7's geometry build, for scale)`,
);
console.log(`  node               ${process.version}\n`);
