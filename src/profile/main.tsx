/**
 * The Phase 7 measurement harness.
 *
 * A page that mounts the **real** scene — the same `SceneCanvas`, the same shaders, the
 * same picking pass — over a synthetic 50,000-point cloud in the real wire format, and
 * exposes one function that runs every exit criterion and returns the numbers as JSON.
 *
 * Synthetic rather than a scanned library, because `task.md` allows Phase 7 to be built
 * against synthetic point data and because the alternative is a benchmark nobody but its
 * author can reproduce. The generator in `scene/synthetic.ts` is seeded, and it clusters,
 * which is what makes the measurement pessimistic in the way that matters: overdraw is the
 * cost here, and overdraw comes from density.
 *
 * Not part of the app bundle. `vite build --mode profile` builds this page alone into
 * `dist-profile/`, and `scripts/orbit_profile.mjs` serves it to a real WKWebView — see
 * `scripts/webview_eval.swift` for why the engine has to be that one.
 */

import { useFrame } from '@react-three/fiber';
import { StrictMode, useCallback, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';

import { CloudBuffers } from '../scene/buffers';
import { frameBounds } from '../scene/framing';
import type { CloudHandle } from '../scene/PointCloud';
import {
  measureRafCadence,
  runOrbitProfile,
  type Distribution,
  type OrbitProfile,
  type OrbitProfileOptions,
} from '../scene/profile';
import { SceneCanvas } from '../scene/SceneCanvas';
import { syntheticFeatureColumn, syntheticPointCloud } from '../scene/synthetic';
import {
  checkContextLossRecovery,
  checkPickingUnderDensity,
  probeAliveByPicking,
  type ContextLossResult,
  type PickCheckResult,
} from './checks';
import '../styles/index.css';

const params = new URLSearchParams(window.location.search);
const POINT_COUNT = Number(params.get('points') ?? 50_000);

/**
 * When this module started evaluating.
 *
 * A module constant rather than a `useRef(performance.now())`, which would be an impure
 * call during render. It is also the more useful origin: what §7 budgets at 300 ms is
 * "point cloud load → first frame", and the clock for that starts when the code that will
 * draw the frame begins running, not when one component happens to mount.
 */
const PAGE_START = performance.now();

export interface HarnessReport {
  pointCount: number;
  /** Decode plus interleave plus geometry build, in isolation. */
  buildMs: number;
  /**
   * Module evaluation to the first frame the scene drew.
   *
   * The frontend half of §7's 300 ms "point cloud load → first frame" budget: React
   * mounting, the canvas coming up, the shaders compiling, the geometry uploading and the
   * first draw. It excludes the IPC hop, which does not exist on this page.
   */
  mountToFirstFrameMs: number;
  /**
   * What this engine's frame cadence is with nothing to draw at all.
   *
   * The control for the frame-time criterion. Everything the orbit's p99 exceeds this by is
   * attributable to the scene; everything below it is the measurement floor.
   */
  rafFloor: Distribution;
  /** The same orbit with the cloud hidden: the loop, the clear, and no points. */
  emptyScene: OrbitProfile;
  /** The criterion: a clean orbit, with hover picking gated off the way a drag gates it. */
  orbit: OrbitProfile;
  /**
   * The other case: something animating the camera while the cursor sweeps the cloud at the
   * full 20 Hz the hover gate allows. Shorter, because it characterizes rather than gates.
   */
  orbitWithHover: OrbitProfile;
  picking: PickCheckResult;
  contextLoss: ContextLossResult;
  userAgent: string;
}

declare global {
  interface Window {
    /** Called by `scripts/orbit_profile.mjs` through `webview-eval`. */
    audiocloudHarness?: {
      run(options?: OrbitProfileOptions): Promise<HarnessReport>;
      /**
       * What the run is currently doing.
       *
       * A harness that hangs is otherwise indistinguishable from a harness that is slow,
       * and the runner's only signal is a timeout with no stack. This is what it prints
       * instead.
       */
      stage: string;
    };
  }
}

/** Records the timestamp of the first frame R3F draws, then stops looking. */
function FirstFrame({ onFirst }: { onFirst: (at: number) => void }) {
  const fired = useRef(false);
  useFrame(() => {
    if (fired.current) return;
    fired.current = true;
    onFirst(performance.now());
  });
  return null;
}

function Harness() {
  const [{ buffer, cloud }] = useState(() => syntheticPointCloud({ count: POINT_COUNT }));
  const [featureValues] = useState(() => syntheticFeatureColumn(cloud));
  const firstFrameAt = useRef<number | null>(null);

  const onReady = useCallback(
    (handle: CloudHandle) => {
      const harness = {
        stage: 'idle',
        async run(options?: OrbitProfileOptions): Promise<HarnessReport> {
          const stage = (name: string) => {
            harness.stage = name;
            // Reported live to whatever is driving the page, so a run that stalls says
            // where. `scripts/webview_eval.swift` prints these to stderr and keeps going.
            const bridge = (
              window as unknown as {
                webkit?: {
                  messageHandlers?: { result?: { postMessage(value: unknown): void } };
                };
              }
            ).webkit?.messageHandlers?.result;
            bridge?.postMessage({
              progress: `${name} @ ${Math.round(performance.now())} ms`,
            });
          };
          stage('build');
          // Timed on its own, over the same payload, so the geometry build can be compared
          // against `BENCHMARKS.md`'s interleave figure rather than inferred from the
          // mount-to-first-frame number, which also contains React and shader compilation.
          const buildStarted = performance.now();
          const throwaway = CloudBuffers.fromBuffer(buffer);
          const buildMs = performance.now() - buildStarted;
          throwaway.dispose();

          const target = {
            renderer: handle.renderer,
            camera: handle.camera,
            // Null on purpose: the orbit drives the camera directly, and the mounted
            // `OrbitControls` already calls `update()` once per frame. Updating it twice
            // would measure the controls rather than the scene.
            controls: null,
            bounds: handle.buffers.bounds,
            invalidate: handle.invalidate,
            pickAt: handle.pickAt,
            pointCount: handle.buffers.count,
          };

          // The controls run for the same wall time as the orbit itself. A p99 over 500
          // samples and a p99 over 1,800 are not comparable numbers -- the longer run has
          // more chances to catch whatever the machine does once a minute -- and comparing
          // them was the first version's mistake.
          const controlDurationMs = options?.durationMs ?? 30_000;

          stage('raf-floor');
          const rafFloor = await measureRafCadence(controlDurationMs);

          // The cloud hidden, everything else identical: same camera path, same
          // `invalidate` per frame, same clear of the same 2880x1800 buffer. The difference
          // between this and the run below is what drawing 50,000 points costs.
          stage('empty-scene');
          handle.points.visible = false;
          const emptyScene = await runOrbitProfile(target, {
            ...options,
            durationMs: controlDurationMs,
            warmupMs: 500,
            idleMs: 300,
          });
          handle.points.visible = true;
          handle.invalidate();

          stage('orbit');
          const orbit = await runOrbitProfile(target, options);
          stage('orbit+hover');
          const orbitWithHover = await runOrbitProfile(target, {
            ...options,
            durationMs: Math.min(options?.durationMs ?? 30_000, 8_000),
            // The warmup is short here because everything is already warm; the shaders
            // compiled and the buffers uploaded during the run above.
            warmupMs: 500,
            idleMs: 500,
            pickHz: 20,
          });

          // Re-frame before the pick check, closer in than the default framing. Two
          // reasons, and the second is the one that makes the check mean anything: the
          // orbit left the camera mid-dolly and the CPU reference must be taken against a
          // settled one, and at the default framing the sprites are about two pixels
          // across, so almost no two of them overlap and "under a dense cluster" would be
          // measured on pixels holding one sprite each. Sprite size goes as 1/depth, so
          // pulling in to a little over a third of the framing distance is what a user
          // does when they want to click something inside a cluster, and it is where
          // picking has to be right.
          stage('reframe');
          frameBounds(handle.camera, null, handle.buffers.bounds, { padding: 0.55 });
          handle.invalidate();
          await nextFrame();
          await nextFrame();

          stage('picking');
          const picking = await checkPickingUnderDensity(
            handle.renderer,
            handle.camera,
            handle.buffers,
            handle.materials.shared,
            handle.pickAt,
            handle.picker.windowPx,
          );

          stage('context-loss');
          const contextLoss = await checkContextLossRecovery(
            handle.renderer,
            handle.buffers,
            handle.invalidate,
            handle.simulateContextLoss,
            () =>
              probeAliveByPicking(
                handle.renderer,
                handle.camera,
                handle.buffers,
                handle.materials.shared,
                handle.pickAt,
              ),
          );

          stage('done');
          return {
            pointCount: handle.buffers.count,
            buildMs,
            mountToFirstFrameMs: (firstFrameAt.current ?? performance.now()) - PAGE_START,
            rafFloor,
            emptyScene,
            orbit,
            orbitWithHover,
            picking,
            contextLoss,
            userAgent: navigator.userAgent,
          };
        },
      };
      window.audiocloudHarness = harness;
    },
    [buffer],
  );

  return (
    <SceneCanvas source={buffer} featureValues={featureValues} onReady={onReady}>
      <FirstFrame onFirst={(at) => (firstFrameAt.current ??= at)} />
    </SceneCanvas>
  );
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

const container = document.getElementById('root');
if (!container) throw new Error('#root missing from profile.html');
createRoot(container).render(
  <StrictMode>
    <Harness />
  </StrictMode>,
);
