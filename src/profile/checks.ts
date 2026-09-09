/**
 * The falsifiable half of Phase 7's exit criteria.
 *
 * > GPU pick returns the correct sample under a dense cluster; context loss recovers
 * > without a refetch.
 *
 * Neither is a number, and both are the kind of claim that is easy to assert and hard to
 * have actually tested. So each is turned into something that can come back false.
 *
 * ### What "the correct sample" means
 *
 * The first version of this check got the question wrong in an instructive way. It assumed
 * a pick should return the point whose *centre* lands on the cursor's pixel, and 181 of 200
 * trials disagreed with the GPU. The GPU was right: points are drawn as sprites several
 * pixels across, so the point under the cursor is the one whose **sprite covers** that
 * pixel, and where several do, the one **nearest the camera**. That is what a person means
 * by clicking on something, and it is what the depth test in `pick.frag.glsl` produces.
 *
 * So the reference below reimplements the sizing chain from `points.vert.glsl` in
 * JavaScript — projection, depth fade, LOD shrink, the clamp to the device's point-size
 * range, the sub-pixel discard — and asks which point that model says should win. Two
 * implementations of one rule, in two languages, compared on hundreds of pixels chosen for
 * being crowded. A shader change that breaks the correspondence between the visible pass and
 * the pick pass shows up here as a falling score rather than as a user complaining that
 * clicks land on the wrong sample.
 *
 * Points within three quarters of a pixel of their sprite's edge are treated as ambiguous,
 * because that is genuinely below the resolution at which a CPU model and a rasterizer can
 * be expected to agree, and a check that failed on rasterization tie-breaks would be noise.
 */

import { Vector3, type PerspectiveCamera, type WebGLRenderer } from 'three';

import type { CloudBuffers } from '../scene/buffers';
import type { SharedUniforms } from '../scene/materials';
import { NO_PICK } from '../scene/picking';

/** How far from a sprite's edge the CPU model declines to have an opinion, in device px. */
const EDGE_TOLERANCE_PX = 0.75;

// ── Picking ─────────────────────────────────────────────────────────────────────

export interface PickCheckResult {
  trials: number;
  /** The GPU returned the point the CPU model says should win. */
  exact: number;
  /**
   * It returned a different point that also covers the cursor and is no further away.
   *
   * Counted as a pass: the CPU model's `strict` set excludes points sitting on their
   * sprite's edge, so this is the model conceding, not the picker being wrong.
   */
  equivalent: number;
  /** It returned a point that covers the cursor but is **behind** the winner. Wrong. */
  behind: number;
  /** It returned a point whose sprite does not cover the cursor at all. Wrong. */
  uncovered: number;
  /** It returned nothing where the model says something should have been hit. Wrong. */
  missed: number;
  /** Mean size of the covering set on the tested pixels — how dense "dense" was. */
  meanCandidates: number;
  maxCandidates: number;
  /** Wall time per pick, including the `readPixels` stall. */
  meanPickMs: number;
  maxPickMs: number;
  passed: boolean;
}

/** The projected state of every point, in device pixels. The CPU half of the comparison. */
interface Projected {
  x: Float32Array;
  y: Float32Array;
  /** Sprite radius in device pixels, or 0 for a point the shader discards. */
  radius: Float32Array;
  depth: Float32Array;
  /** False for points behind the camera, off-screen, or below the LOD floor. */
  drawn: Uint8Array;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0 || 1e-9)));
  return t * t * (3 - 2 * t);
}

/**
 * Reimplements `points.vert.glsl`'s sizing chain on the CPU.
 *
 * Every term here has a counterpart in the shader and the correspondence is the point of
 * the exercise, so they are written in the same order with the same names.
 */
function project(
  camera: PerspectiveCamera,
  buffers: CloudBuffers,
  uniforms: SharedUniforms,
  width: number,
  height: number,
): Projected {
  camera.updateMatrixWorld();
  const viewProjection = camera.projectionMatrix
    .clone()
    .multiply(camera.matrixWorldInverse);

  const count = buffers.count;
  const out: Projected = {
    x: new Float32Array(count),
    y: new Float32Array(count),
    radius: new Float32Array(count),
    depth: new Float32Array(count),
    drawn: new Uint8Array(count),
  };

  const positions = buffers.geometry.getAttribute('position');
  const sizes = buffers.geometry.getAttribute('aSize');
  const [rangeMin, rangeMax] = [
    uniforms.uPointSizeRange.value.x,
    uniforms.uPointSizeRange.value.y,
  ];
  const world = new Vector3();
  const view = new Vector3();

  for (let i = 0; i < count; i++) {
    world.set(positions.getX(i), positions.getY(i), positions.getZ(i));
    view.copy(world).applyMatrix4(camera.matrixWorldInverse);
    const depth = Math.max(-view.z, 1e-4);
    out.depth[i] = depth;
    if (-view.z <= 0) continue;

    const clip = world.clone().applyMatrix4(viewProjection);
    const deviceX = (clip.x * 0.5 + 0.5) * width;
    const deviceY = (1 - (clip.y * 0.5 + 0.5)) * height;
    out.x[i] = deviceX;
    out.y[i] = deviceY;

    const recede = smoothstep(uniforms.uFadeNear.value, uniforms.uFadeFar.value, depth);
    const radius =
      sizes.getX(i) *
      uniforms.uSizeMultiplier.value *
      (1 + (uniforms.uFarSizeScale.value - 1) * recede);
    const pixels = (radius * uniforms.uSizeScale.value) / depth;
    // The fragment stage discards on the *unclamped* size, then the rasterizer works with
    // the clamped one. Both, in that order, or the model disagrees with the shader at the
    // two sizes where it matters.
    if (pixels < uniforms.uMinPointSize.value) continue;
    out.radius[i] = Math.min(Math.max(pixels, rangeMin), rangeMax) * 0.5;
    out.drawn[i] = 1;
  }

  return out;
}

/**
 * Compares the GPU pick against the CPU model over the most crowded pixels on screen.
 *
 * Candidate cursor positions are the projected centres of drawn points, which guarantees
 * every trial lands on something; the trials kept are the ones where many sprites overlap,
 * because a pick that works on an isolated point and fails in a cluster fails exactly where
 * it matters.
 */
export async function checkPickingUnderDensity(
  renderer: WebGLRenderer,
  camera: PerspectiveCamera,
  buffers: CloudBuffers,
  uniforms: SharedUniforms,
  pickAt: (x: number, y: number) => number,
  pickWindowPx: number,
  {
    trials = 200,
    minCandidates = 4,
    seed = 7,
  }: {
    trials?: number;
    minCandidates?: number;
    seed?: number;
  } = {},
): Promise<PickCheckResult> {
  const ratio = renderer.getPixelRatio();
  const width = renderer.domElement.width;
  const height = renderer.domElement.height;
  const projected = project(camera, buffers, uniforms, width, height);

  // A point whose centre falls outside the pick window is clipped before rasterization, so
  // the GPU cannot return it however large its sprite is. The model has to obey the same
  // limit or it will insist on answers the pass is structurally unable to give — see the
  // "what the window size buys" note in `scene/picking.ts`.
  const halfWindow = (pickWindowPx - 1) / 2;

  const result: PickCheckResult = {
    trials: 0,
    exact: 0,
    equivalent: 0,
    behind: 0,
    uncovered: 0,
    missed: 0,
    meanCandidates: 0,
    maxCandidates: 0,
    meanPickMs: 0,
    maxPickMs: 0,
    passed: false,
  };

  let state = seed >>> 0;
  const nextRandom = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };

  const drawn: number[] = [];
  for (let i = 0; i < buffers.count; i++) if (projected.drawn[i]) drawn.push(i);
  if (drawn.length === 0) return result;

  // A uniform grid over the projected centres. Without it each trial scans all 50,000
  // points to find the handful within ten pixels of the cursor, and a few thousand
  // attempts at that turns a check into a hang -- which is exactly what happened the first
  // time this ran with the camera close enough for the sprites to overlap.
  // Sized so that a three-by-three neighbourhood covers the whole area a candidate can
  // occupy -- the pick window's half-width -- and no more. Larger cells were the first
  // attempt and made each trial scan thousands of points in a dense cluster, which turned
  // the check into a main-thread hang rather than a slow check.
  const CELL_PX = Math.max(4, Math.ceil(halfWindow));
  const columns = Math.ceil(width / CELL_PX) + 1;
  const grid = new Map<number, number[]>();
  for (const i of drawn) {
    const key =
      Math.floor((projected.y[i] as number) / CELL_PX) * columns +
      Math.floor((projected.x[i] as number) / CELL_PX);
    const bucket = grid.get(key);
    if (bucket) bucket.push(i);
    else grid.set(key, [i]);
  }
  // One cell either side covers the pick window's reach in each direction.
  const nearby = (px: number, py: number): number[] => {
    const cellX = Math.floor(px / CELL_PX);
    const cellY = Math.floor(py / CELL_PX);
    const found: number[] = [];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const bucket = grid.get((cellY + dy) * columns + (cellX + dx));
        if (bucket) found.push(...bucket);
      }
    }
    return found;
  };

  let candidateTotal = 0;
  let pickTotal = 0;
  let attempts = 0;
  // A wall-clock stop. This runs on the main thread with no yield points, and a harness
  // that blocks the page is reported by the runner as a bare timeout with no stack.
  const startedAt = performance.now();

  while (
    result.trials < trials &&
    attempts < trials * 25 &&
    performance.now() - startedAt < 20_000
  ) {
    attempts++;
    // Yield to the event loop periodically. This is a few hundred GPU round trips on the
    // main thread; without a yield the page stops answering and a harness that is merely
    // slow is indistinguishable from one that has hung -- which is how an afternoon gets
    // spent on a timeout with no stack in it.
    if (attempts % 64 === 0) await nextFrame();
    const anchor = drawn[Math.floor(nextRandom() * drawn.length)] as number;
    const cursorX = Math.floor(projected.x[anchor] as number);
    const cursorY = Math.floor(projected.y[anchor] as number);
    if (cursorX < 0 || cursorY < 0 || cursorX >= width || cursorY >= height) continue;
    const sampleX = cursorX + 0.5;
    const sampleY = cursorY + 0.5;

    // Everything the rasterizer could plausibly have drawn on this pixel, split by whether
    // the CPU model is confident about it.
    const strict: number[] = [];
    const loose: number[] = [];
    for (const i of nearby(sampleX, sampleY)) {
      const dx = (projected.x[i] as number) - sampleX;
      const dy = (projected.y[i] as number) - sampleY;
      if (Math.abs(dx) > halfWindow || Math.abs(dy) > halfWindow) continue;
      const distance = Math.hypot(dx, dy);
      const radius = projected.radius[i] as number;
      if (distance <= radius - EDGE_TOLERANCE_PX) strict.push(i);
      if (distance <= radius + EDGE_TOLERANCE_PX) loose.push(i);
    }
    if (strict.length < minCandidates) continue;

    let expected = strict[0] as number;
    for (const i of strict) {
      if ((projected.depth[i] as number) < (projected.depth[expected] as number))
        expected = i;
    }

    const started = performance.now();
    const picked = pickAt(sampleX / ratio, sampleY / ratio);
    const elapsed = performance.now() - started;

    result.trials++;
    candidateTotal += strict.length;
    result.maxCandidates = Math.max(result.maxCandidates, strict.length);
    pickTotal += elapsed;
    result.maxPickMs = Math.max(result.maxPickMs, elapsed);

    if (picked === NO_PICK) {
      result.missed++;
    } else if (picked === expected) {
      result.exact++;
    } else if (!loose.includes(picked)) {
      result.uncovered++;
    } else if (
      (projected.depth[picked] as number) <= (projected.depth[expected] as number)
    ) {
      result.equivalent++;
    } else {
      result.behind++;
    }
  }

  result.meanCandidates = candidateTotal / Math.max(result.trials, 1);
  result.meanPickMs = pickTotal / Math.max(result.trials, 1);
  result.passed =
    result.trials >= Math.min(trials, 20) &&
    result.behind === 0 &&
    result.uncovered === 0 &&
    result.missed === 0;
  return result;
}

// ── Context loss ────────────────────────────────────────────────────────────────

export interface ContextLossResult {
  /** Whether `WEBGL_lose_context` was available to force the loss. */
  supported: boolean;
  lostFired: boolean;
  restoredFired: boolean;
  /** Frames rendered after the restore. Zero would mean the canvas came back dead. */
  framesAfterRestore: number;
  /**
   * Successful picks, out of ten, after the restore.
   *
   * The liveness probe, and chosen over reading the visible framebuffer for a reason worth
   * recording: with `preserveDrawingBuffer: false` the back buffer is not readable outside
   * the frame that drew it, so `readPixels` on the canvas after an `await` returns cleared
   * memory whether the scene recovered or not — a check that passes and proves nothing, or
   * fails and proves nothing. The pick pass draws into a render target this code owns, and
   * a hit through it means the geometry, the attributes, the shader and the depth buffer
   * are all back on the GPU.
   */
  picksAfterRestore: number;
  /** True if the recovered geometry is still backed by the original payload object. */
  sameSourceBuffer: boolean;
  passed: boolean;
}

/**
 * Takes the context away and checks what comes back.
 *
 * `sameSourceBuffer` is the "without a refetch" half of the criterion. It compares object
 * identity against the payload captured before the loss, so it cannot be satisfied by a
 * refetch that happened to return equal bytes.
 */
export async function checkContextLossRecovery(
  renderer: WebGLRenderer,
  buffers: CloudBuffers,
  invalidate: () => void,
  simulateContextLoss: () => boolean,
  probeAlive: () => number,
): Promise<ContextLossResult> {
  const canvas = renderer.domElement;
  const sourceBefore = buffers.source;

  let lostFired = false;
  let restoredFired = false;
  const onLost = () => (lostFired = true);
  const onRestored = () => (restoredFired = true);
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  try {
    const supported = simulateContextLoss();
    if (!supported) {
      return {
        supported: false,
        lostFired: false,
        restoredFired: false,
        framesAfterRestore: 0,
        picksAfterRestore: 0,
        sameSourceBuffer: true,
        passed: false,
      };
    }

    // Restoration is asynchronous and driver-paced, and the only signal is the event being
    // counted, so this polls for it rather than guessing at a single sleep.
    for (let i = 0; i < 240 && !restoredFired; i++) await nextFrame();

    const framesBefore = renderer.info.render.frame;
    invalidate();
    for (let i = 0; i < 10; i++) await nextFrame();
    const framesAfterRestore = renderer.info.render.frame - framesBefore;

    const picksAfterRestore = probeAlive();

    const passed =
      lostFired &&
      restoredFired &&
      framesAfterRestore > 0 &&
      picksAfterRestore > 0 &&
      buffers.source === sourceBefore;

    return {
      supported: true,
      lostFired,
      restoredFired,
      framesAfterRestore,
      picksAfterRestore,
      sameSourceBuffer: buffers.source === sourceBefore,
      passed,
    };
  } finally {
    canvas.removeEventListener('webglcontextlost', onLost);
    canvas.removeEventListener('webglcontextrestored', onRestored);
  }
}

/**
 * Picks at the projected centre of ten drawn points and counts the hits.
 *
 * Used as the liveness probe above, and useful on its own: it is the smallest thing that
 * exercises the whole chain from a typed array to a decoded pixel.
 */
export function probeAliveByPicking(
  renderer: WebGLRenderer,
  camera: PerspectiveCamera,
  buffers: CloudBuffers,
  uniforms: SharedUniforms,
  pickAt: (x: number, y: number) => number,
  attempts = 10,
): number {
  const ratio = renderer.getPixelRatio();
  const width = renderer.domElement.width;
  const height = renderer.domElement.height;
  const projected = project(camera, buffers, uniforms, width, height);

  let hits = 0;
  let tested = 0;
  for (
    let i = 0;
    i < buffers.count && tested < attempts;
    i += Math.max(1, (buffers.count / 997) | 0)
  ) {
    if (!projected.drawn[i]) continue;
    const x = Math.floor(projected.x[i] as number) + 0.5;
    const y = Math.floor(projected.y[i] as number) + 0.5;
    if (x < 0 || y < 0 || x >= width || y >= height) continue;
    tested++;
    if (pickAt(x / ratio, y / ratio) !== NO_PICK) hits++;
  }
  return hits;
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}
