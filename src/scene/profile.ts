/**
 * The scripted orbit that Phase 7's exit criterion is stated against.
 *
 * > sustained **< 16.6 ms p99** frame time orbiting 50,000 points on Apple Silicon; idle
 * > canvas renders **0 frames/s**
 *
 * Both halves are measured here, over a deterministic 30-second camera path, so that the
 * number is reproducible rather than a description of how it felt to drag the mouse.
 *
 * ### What is actually being timed, and why it is not a GPU clock
 *
 * `overview.md` §7 says "`WebGLRenderer.info` + Chrome DevTools trace". Chrome is not the
 * engine this ships in, and the WebKit probe in `scripts/probe_webgl_caps.mjs` found no
 * `EXT_disjoint_timer_query_webgl2` — WebKit does not expose a GPU clock to WebGL at all.
 * So there are exactly three signals available in-page, and this reports all three because
 * no one of them is the whole answer:
 *
 * - **Frame interval.** Wall time between consecutive rendered frames, under a request for
 *   a frame on every vsync. This is the criterion. Rendering is vsync-locked, so a healthy
 *   scene sits at the display interval and p99 above it means frames were *missed* — which
 *   is the thing a 16.6 ms budget exists to prevent. It is also the only number here that
 *   includes GPU time, because a GPU that has not finished is what delays the next swap.
 * - **CPU time inside `render()`.** Everything the main thread does per frame. This is what
 *   a regression in the scene graph, the uniform updates, or a stray per-point loop moves.
 *   It should be a small fraction of the interval; if it approaches it, the main thread is
 *   the bottleneck rather than the GPU.
 * - **Dropped frames.** Intervals longer than 1.5 display periods. A blunt but unambiguous
 *   count of how often the scene missed a swap.
 *
 * ### And the pick pass
 *
 * Hover picking is a synchronous `readPixels`, which drains the pipeline, and the honest
 * version of this measurement includes it — the realistic worst case is a user sweeping the
 * cursor across a cluster while orbiting. `pickHz` drives that at the same 20 Hz the real
 * gate allows, and the cost is reported separately so the stall is visible rather than
 * smeared into the frame interval.
 */

import { Vector2, type PerspectiveCamera, type WebGLRenderer } from 'three';

import type { CloudBounds } from './buffers';
import type { FramingControls } from './framing';

export interface OrbitProfileOptions {
  durationMs?: number;
  /** Discarded before recording: shader compiles, first uploads, the JIT warming up. */
  warmupMs?: number;
  /** Full azimuth revolutions over `durationMs`. */
  revolutions?: number;
  /** How far the camera dollies in and out, as a fraction of its framing distance. */
  dollyAmplitude?: number;
  /** Elevation sweep, in degrees either side of the starting elevation. */
  elevationDegrees?: number;
  /**
   * Hover picks per second during the orbit. **Zero by default, and that is the criterion.**
   *
   * §5.3 gates hover picking off entirely while a drag is in progress, so an orbit — which
   * is a drag — issues no picks at all in the real app, and folding a synchronous
   * `readPixels` stall into the number would measure a combination that cannot occur.
   * Set it to 20 to characterize the other case: a cursor sweeping across the cloud while
   * something else is animating it.
   */
  pickHz?: number;
  /** How long to sit still afterwards while counting frames. */
  idleMs?: number;
}

const DEFAULTS = {
  durationMs: 30_000,
  warmupMs: 2_000,
  revolutions: 2,
  dollyAmplitude: 0.28,
  elevationDegrees: 26,
  pickHz: 0,
  idleMs: 3_000,
} as const;

export interface Distribution {
  count: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

export interface OrbitProfile {
  points: number;
  /** Framebuffer size the orbit was measured at, and the ratio that produced it. */
  drawingBuffer: { width: number; height: number; pixelRatio: number };
  /** The display's frame period, taken as the median interval. */
  displayIntervalMs: number;
  frameIntervalMs: Distribution;
  cpuRenderMs: Distribution;
  pickMs: Distribution;
  /** Intervals longer than 1.5 display periods. */
  droppedFrames: number;
  /** Frames rendered while nothing asked for one. The second exit criterion; must be 0. */
  idleFrames: number;
  idleMs: number;
  /**
   * From `WebGLRenderer.info`, per frame.
   *
   * Per frame rather than totalled, because `info.autoReset` is on and the counters are
   * cleared at the start of every render — a difference between two samples taken thirty
   * seconds apart is not a total, it is noise.
   */
  info: { callsPerFrame: number; pointsPerFrame: number; programs: number };
}

/** What the harness needs from the scene. A subset of `CloudHandle`, so tests can fake it. */
export interface ProfileTarget {
  renderer: WebGLRenderer;
  camera: PerspectiveCamera;
  controls: FramingControls | null;
  bounds: CloudBounds;
  invalidate(): void;
  /** Canvas-relative CSS coordinates to a cloud index. */
  pickAt(x: number, y: number): number;
  pointCount: number;
}

/**
 * Runs the orbit and resolves with the numbers.
 *
 * Deliberately not a React hook and not a component: it takes a plain object, drives a
 * `requestAnimationFrame` loop, and restores everything it patched. That makes it callable
 * from a harness page, from a devtools console against the running app, and from a test.
 */
export async function runOrbitProfile(
  target: ProfileTarget,
  options: OrbitProfileOptions = {},
): Promise<OrbitProfile> {
  const settings = { ...DEFAULTS, ...options };
  const { renderer, camera, controls, bounds } = target;

  const frameStarts: number[] = [];
  const cpuTimes: number[] = [];
  const pickTimes: number[] = [];
  let lastCalls = 0;
  let lastPoints = 0;

  // The camera path is described relative to where the scene framed it, so the profile
  // measures the view the user actually gets rather than an arbitrary distance that might
  // put the whole cloud in four pixels or fill the screen with one cluster.
  const startDistance = camera.position.distanceTo(bounds.center);
  const startElevation = Math.asin(
    Math.min(1, Math.max(-1, (camera.position.y - bounds.center.y) / startDistance)),
  );
  const startAzimuth = Math.atan2(
    camera.position.z - bounds.center.z,
    camera.position.x - bounds.center.x,
  );

  // Timing by monkey-patch rather than by watching `renderer.info`: `info` counts draws, and
  // what is wanted is when each frame *happened* and how long the main thread was inside it.
  // Restored in the `finally` below, including on a throw.
  const originalRender = renderer.render.bind(renderer);
  let recording = false;
  let inPick = false;

  renderer.render = (scene, renderCamera) => {
    if (inPick || !recording) {
      originalRender(scene, renderCamera);
      return;
    }
    const started = performance.now();
    frameStarts.push(started);
    originalRender(scene, renderCamera);
    cpuTimes.push(performance.now() - started);
    lastCalls = renderer.info.render.calls;
    lastPoints = renderer.info.render.points;
  };

  const size = renderer.getDrawingBufferSize(new Vector2());
  const canvas = renderer.domElement;

  try {
    const orbitEnd = await new Promise<number>((resolve, reject) => {
      const started = performance.now();
      let frame = 0;
      let nextPickAt = settings.pickHz > 0 ? started + settings.warmupMs : Infinity;

      const step = (now: number) => {
        const elapsed = now - started;

        if (!recording && elapsed >= settings.warmupMs) recording = true;

        if (elapsed >= settings.warmupMs + settings.durationMs) {
          resolve(now);
          return;
        }

        // One revolution per `durationMs / revolutions`, with the dolly and the elevation
        // on incommensurate periods so the path does not repeat itself and quietly measure
        // the same twenty frames sixty times.
        const t = Math.max(0, elapsed - settings.warmupMs) / settings.durationMs;
        const azimuth = startAzimuth + t * settings.revolutions * Math.PI * 2;
        const elevation =
          startElevation +
          Math.sin(t * Math.PI * 2 * 1.5) * ((settings.elevationDegrees * Math.PI) / 180);
        const distance =
          startDistance * (1 + Math.sin(t * Math.PI * 2 * 2.5) * settings.dollyAmplitude);

        const horizontal = Math.cos(elevation) * distance;
        camera.position.set(
          bounds.center.x + Math.cos(azimuth) * horizontal,
          bounds.center.y + Math.sin(elevation) * distance,
          bounds.center.z + Math.sin(azimuth) * horizontal,
        );
        camera.lookAt(bounds.center);
        controls?.update();

        if (now >= nextPickAt) {
          nextPickAt = now + 1000 / settings.pickHz;
          // A lissajous sweep across the middle of the canvas, so picks land on the cloud
          // rather than on the background where the pass costs nothing.
          const px = canvas.clientWidth * (0.5 + 0.28 * Math.sin(elapsed / 900));
          const py = canvas.clientHeight * (0.5 + 0.24 * Math.cos(elapsed / 1300));
          inPick = true;
          const pickStarted = performance.now();
          try {
            target.pickAt(px, py);
          } finally {
            inPick = false;
          }
          if (recording) pickTimes.push(performance.now() - pickStarted);
        }

        target.invalidate();
        frame++;
        requestAnimationFrame(step);
      };

      requestAnimationFrame(step);
      // A harness that hangs must not hang forever.
      setTimeout(
        () => reject(new Error(`orbit did not finish; ${frame} frames`)),
        settings.warmupMs + settings.durationMs + 15_000,
      );
    });

    const info = {
      callsPerFrame: lastCalls,
      pointsPerFrame: lastPoints,
      programs: renderer.info.programs?.length ?? 0,
    };

    // ── The idle criterion ──────────────────────────────────────────────────────
    //
    // Nothing calls `invalidate()` from here on. With `frameloop="demand"` that must mean
    // no frames at all. The settle window exists because orbit damping is a real animation
    // and is *supposed* to keep asking for frames until it converges; counting during it
    // would fail a criterion about idleness on the strength of a transition.
    recording = false;
    await wait(600);
    const framesBeforeIdle = renderer.info.render.frame;
    await wait(settings.idleMs);
    const idleFrames = renderer.info.render.frame - framesBeforeIdle;

    const intervals: number[] = [];
    for (let i = 1; i < frameStarts.length; i++) {
      intervals.push((frameStarts[i] as number) - (frameStarts[i - 1] as number));
    }
    const displayIntervalMs = median(intervals);
    const droppedFrames = intervals.filter((ms) => ms > displayIntervalMs * 1.5).length;

    void orbitEnd;
    return {
      points: target.pointCount,
      drawingBuffer: {
        width: size.x,
        height: size.y,
        pixelRatio: renderer.getPixelRatio(),
      },
      displayIntervalMs,
      frameIntervalMs: distribution(intervals),
      cpuRenderMs: distribution(cpuTimes),
      pickMs: distribution(pickTimes),
      droppedFrames,
      idleFrames,
      idleMs: settings.idleMs,
      info,
    };
  } finally {
    renderer.render = originalRender;
  }
}

/**
 * The frame cadence this engine can deliver with nothing to draw.
 *
 * The control the frame-time criterion turned out to need. WebKit's `requestAnimationFrame`
 * on this machine has a p99 of about 23 ms over an **empty page** — no WebGL context, no
 * scene, no work of any kind — against a 16 ms median. Roughly one frame in a hundred
 * misses its vsync deadline for reasons that have nothing to do with anything this project
 * renders, and without measuring that, a scene measured at 24 ms p99 looks like a scene
 * with a problem instead of a scene one millisecond above the floor.
 *
 * Measured inside the same page and the same run as the orbit, because a floor established
 * on a quiet machine says nothing about the machine the orbit ran on.
 */
export async function measureRafCadence(durationMs = 5_000): Promise<Distribution> {
  const intervals: number[] = [];
  await new Promise<void>((resolve) => {
    const started = performance.now();
    let last = started;
    const step = (now: number) => {
      intervals.push(now - last);
      last = now;
      if (now - started < durationMs) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
  // The first interval is measured from before the first callback and is not a frame time.
  intervals.shift();
  return distribution(intervals);
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] as number;
}

/** Percentiles by nearest rank, which is the definition that does not invent a value. */
export function distribution(values: readonly number[]): Distribution {
  if (values.length === 0) {
    return { count: 0, mean: 0, p50: 0, p95: 0, p99: 0, max: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) =>
    sorted[Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)] as number;
  let total = 0;
  for (const value of sorted) total += value;
  return {
    count: sorted.length,
    mean: total / sorted.length,
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: sorted[sorted.length - 1] as number,
  };
}
