/**
 * The cloud, as one component that mounts once and is driven imperatively thereafter
 * (`overview.md` §5.6).
 *
 * Read the render function and notice what is *not* there: no `map` over points, no state
 * that changes when the cursor moves, no dependency on the selection. It returns one
 * `<primitive>` and a set of controls, and it re-renders only when the payload itself is
 * replaced. Everything that happens while the user is working — hovering, selecting,
 * filtering, recolouring, orbiting — happens through the effects and subscriptions below,
 * which write into typed arrays and uniforms and then ask for one frame.
 *
 * The three inputs have three different shapes on purpose:
 *
 * - `source` is a prop, because replacing the projection replaces everything.
 * - `featureValues` and `matchedIds` are props, because they are *data* and §5.6 keeps data
 *   out of the store; but they are consumed in effects, so a new column costs one render of
 *   this component and none of anything below it.
 * - Hover and selection come from the Zustand store through an imperative subscription,
 *   never through a hook, because they change up to twenty times a second.
 */

import { OrbitControls } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import { useCallback, useEffect, useMemo, useRef, type ComponentRef } from 'react';
import {
  Points,
  Vector2,
  type PerspectiveCamera,
  type Scene,
  type WebGLRenderer,
} from 'three';

import { useSceneStore } from '../store/scene';
import { CloudBuffers, DEFAULT_POINT_COLOR, type CloudBufferOptions } from './buffers';
import { readCaps, type GlCaps } from './caps';
import { EMBER, colorsFromColumn, featureDomain, featureRamp } from './colors';
import { fadeRange, frameBounds } from './framing';
import { maskFrom } from './mask';
import {
  createCloudMaterials,
  pixelsPerWorldUnit,
  type CloudMaterials,
  type SharedUniforms,
} from './materials';
import { HoverGate, NO_PICK, Picker } from './picking';

/** What the harness in `scene/profile.ts` and the app shell get to hold. */
export interface CloudHandle {
  buffers: CloudBuffers;
  materials: CloudMaterials;
  caps: GlCaps;
  picker: Picker;
  points: Points;
  camera: PerspectiveCamera;
  /**
   * The renderer and the scene.
   *
   * Exposed because the measurement harness genuinely needs them — `scene/profile.ts` times
   * `render()` and `profile/checks.ts` reads the back buffer — and because a handle that
   * withheld them would only push the harness into reaching for `gl.domElement` and
   * reconstructing what it was already holding. Nothing in `panels/` should touch either.
   */
  renderer: WebGLRenderer;
  scene: Scene;
  // Declared as properties holding functions rather than as methods, deliberately: every
  // one of them is meant to be pulled off the handle and passed somewhere else, and a
  // method signature makes that a lint error about `this` on a function that has none.
  invalidate: () => void;
  /** Re-frames the camera on the whole cloud. */
  frameAll: () => void;
  /** Runs one pick at canvas-relative CSS coordinates. Returns a cloud index or `-1`. */
  pickAt: (x: number, y: number) => number;
  /** Forces a context loss and recovery, for the Phase 7 exit criterion. */
  simulateContextLoss: () => boolean;
}

export interface PointCloudProps {
  /** An `ABPC` payload. Held for the lifetime of the cloud — see `CloudBuffers.source`. */
  source: ArrayBuffer;
  /**
   * The active colour-by column, in the cloud's order, or `null` for the flat default.
   *
   * Fetched by the shell through `getFeatureColumn`, not here: the scene does not do IPC.
   */
  featureValues?: Float32Array | null;
  /** Matching sample ids from `query_samples`, ascending, or `null` for no filter. */
  matchedIds?: Uint32Array | null;
  options?: CloudBufferOptions;
  /** Called once the scene is live, with the imperative handle. */
  onReady?: (handle: CloudHandle) => void;
}

/** A pointer that moved less than this many CSS pixels between down and up is a click. */
const CLICK_SLOP_PX = 4;

export function PointCloud({
  source,
  featureValues = null,
  matchedIds = null,
  options,
  onReady,
}: PointCloudProps) {
  const gl = useThree((state) => state.gl);
  const scene = useThree((state) => state.scene);
  const invalidate = useThree((state) => state.invalidate);
  const camera = useThree((state) => state.camera) as PerspectiveCamera;

  const colorBy = useSceneStore((state) => state.colorBy);

  // One `getParameter` round trip, once per context. The shader clamps to whatever this
  // finds rather than to a constant; see `scene/caps.ts`.
  const caps = useMemo(() => readCaps(gl.getContext()), [gl]);
  const buffers = useMemo(
    () => CloudBuffers.fromBuffer(source, options ?? {}),
    [source, options],
  );
  const materials = useMemo(() => createCloudMaterials(caps), [caps]);
  const picker = useMemo(() => new Picker(), []);
  const hoverGate = useMemo(() => new HoverGate(20), []);

  const points = useMemo(() => {
    const object = new Points(buffers.geometry, materials.points);
    // Frustum culling off. The bounding sphere covers the entire library, so the test can
    // never cull anything and every frame would pay for it; and the one case where it
    // *would* fire -- the camera pushed inside a cluster -- is the case where culling the
    // whole cloud is exactly wrong.
    object.frustumCulled = false;
    Picker.enablePicking(object);
    return object;
  }, [buffers, materials]);

  const drawingBuffer = useRef(new Vector2());
  const fade = useRef({ near: 0, far: 1 });
  // The per-frame uniform writes below go through a ref rather than straight at the
  // memoized `materials`. A `ShaderMaterial`'s uniforms are mutable external state and
  // writing them every frame is the entire design, but a value produced by `useMemo` is a
  // render value as far as the compiler's rules are concerned, and mutating one inside
  // `useFrame` is the thing those rules exist to catch. A ref is the sanctioned escape
  // hatch for exactly this, and routing through one says out loud that these objects live
  // outside React's model.
  const shared = useRef<SharedUniforms | null>(null);
  useEffect(() => {
    shared.current = materials.shared;
  }, [materials]);
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null);
  const colorScratch = useRef<Float32Array | null>(null);
  const maskScratch = useRef<Uint8Array | null>(null);
  const pointerDownAt = useRef<{ x: number; y: number } | null>(null);

  // ── Per-frame uniforms ────────────────────────────────────────────────────────
  //
  // Twelve bytes of uniform writes per frame, and nothing else. Both terms genuinely
  // depend on the camera: point size is pixels-per-world-unit, which changes with the
  // window and the pixel ratio, and the depth fade is relative to where the camera is now,
  // which is the whole reason the far side of the cloud recedes as you orbit.
  useFrame(() => {
    const uniforms = shared.current;
    if (!uniforms) return;
    gl.getDrawingBufferSize(drawingBuffer.current);
    uniforms.uSizeScale.value = pixelsPerWorldUnit(camera.fov, drawingBuffer.current.y);
    const { near, far } = fadeRange(camera, buffers.bounds, fade.current);
    uniforms.uFadeNear.value = near;
    uniforms.uFadeFar.value = far;
  });

  // ── Framing ───────────────────────────────────────────────────────────────────

  const frameAll = useCallback(() => {
    frameBounds(camera, controlsRef.current, buffers.bounds);
    invalidate();
  }, [camera, buffers, invalidate]);

  useEffect(() => {
    frameAll();
  }, [frameAll]);

  // ── Colour ────────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!featureValues || featureValues.length !== buffers.count) {
      buffers.setUniformColor(DEFAULT_POINT_COLOR);
      invalidate();
      return;
    }
    // Reused across colour-by changes. At 50,000 points the output is 600 KB, and a user
    // clicking through six features should not leave 3.6 MB behind for the collector to
    // find in the middle of an orbit.
    if (colorScratch.current?.length !== buffers.count * 3) {
      colorScratch.current = new Float32Array(buffers.count * 3);
    }
    const mapping = colorsFromColumn(featureValues, {
      ramp: colorBy ? featureRamp(colorBy) : EMBER,
      domain: colorBy ? featureDomain(colorBy) : null,
      out: colorScratch.current,
    });
    buffers.setColors(mapping.colors);
    invalidate();
  }, [featureValues, colorBy, buffers, invalidate]);

  // ── Filter ────────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!matchedIds) {
      buffers.setMask(null);
      invalidate();
      return;
    }
    if (maskScratch.current?.length !== buffers.count) {
      maskScratch.current = new Uint8Array(buffers.count);
    }
    buffers.setMask(maskFrom(buffers.cloud.ids, matchedIds, maskScratch.current));
    invalidate();
  }, [matchedIds, buffers, invalidate]);

  // ── Point size ────────────────────────────────────────────────────────────────

  useEffect(
    () =>
      useSceneStore.subscribe(
        (state) => state.pointSizeMultiplier,
        (multiplier) => {
          materials.shared.uSizeMultiplier.value = multiplier;
          invalidate();
        },
        { fireImmediately: true },
      ),
    [materials, invalidate],
  );

  // ── Hover and selection ───────────────────────────────────────────────────────
  //
  // The §5.6 rule at its sharpest. Both are single scalars in the store, both are read
  // here by subscription rather than by hook, and both become a uniform write. No React
  // render happens in this subtree when the cursor moves across the cloud.

  useEffect(
    () =>
      useSceneStore.subscribe(
        (state) => state.hoveredSampleId,
        (sampleId) => {
          const index = sampleId === null ? -1 : buffers.indexOfSample(sampleId);
          materials.display.uHoverId.value = index + 1;
          invalidate();
        },
        { fireImmediately: true },
      ),
    [buffers, materials, invalidate],
  );

  useEffect(
    () =>
      useSceneStore.subscribe(
        (state) => state.selectedSampleId,
        (sampleId) => {
          const index = sampleId === null ? -1 : buffers.indexOfSample(sampleId);
          materials.display.uSelectedId.value = index + 1;
          invalidate();
        },
        { fireImmediately: true },
      ),
    [buffers, materials, invalidate],
  );

  // ── Picking ───────────────────────────────────────────────────────────────────

  const pickAt = useCallback(
    (x: number, y: number) => picker.pick(gl, scene, camera, materials.pick, x, y),
    [picker, gl, scene, camera, materials],
  );

  useEffect(() => {
    const canvas = gl.domElement;

    const toCanvas = (event: PointerEvent | MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };

    const onPointerDown = (event: PointerEvent) => {
      pointerDownAt.current = toCanvas(event);
    };

    const onPointerMove = (event: PointerEvent) => {
      // The gate is the whole of the §5.3 policy: never during a drag, and at most 20 Hz
      // otherwise. `readRenderTargetPixels` stalls the pipeline, and stalling it once per
      // pointer event during an orbit is how a 60 fps scene becomes a 20 fps one.
      if (!hoverGate.shouldPick()) return;
      const { x, y } = toCanvas(event);
      const index = pickAt(x, y);
      useSceneStore
        .getState()
        .hover(index === NO_PICK ? null : (buffers.sampleAt(index) ?? null));
    };

    const onPointerLeave = () => useSceneStore.getState().hover(null);

    const onClick = (event: MouseEvent) => {
      // A click that ended an orbit is not a click on a point. Without the slop test,
      // every drag that happens to end over the cloud changes the selection.
      const down = pointerDownAt.current;
      const { x, y } = toCanvas(event);
      pointerDownAt.current = null;
      if (down && Math.hypot(x - down.x, y - down.y) > CLICK_SLOP_PX) return;

      const index = pickAt(x, y);
      // Clicking empty space clears the selection, which is what makes the map feel like a
      // canvas rather than a list that happens to be drawn in 3D.
      useSceneStore
        .getState()
        .select(index === NO_PICK ? null : (buffers.sampleAt(index) ?? null));
    };

    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerleave', onPointerLeave);
    canvas.addEventListener('click', onClick);
    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      canvas.removeEventListener('click', onClick);
    };
  }, [gl, buffers, hoverGate, pickAt]);

  // ── Context loss ──────────────────────────────────────────────────────────────

  useEffect(() => {
    const canvas = gl.domElement;

    const onLost = (event: Event) => {
      // Without `preventDefault` the browser never fires `webglcontextrestored` and the
      // canvas is dead until the app is relaunched. This one line is the difference
      // between a recoverable hiccup and a support ticket.
      event.preventDefault();
      console.warn('[audiobank] WebGL context lost; waiting for restore');
    };

    const onRestored = () => {
      // No refetch. Every array the scene needs is a view onto the `ArrayBuffer` the core
      // sent, `CloudBuffers` has held it since load, and rebuilding from it is a decode
      // and an interleave. Colour and filter state live in arrays the rebuild reuses, so
      // they come back as they were rather than as defaults.
      points.geometry = buffers.rebuild();
      buffers.markAllForUpload();
      materials.points.needsUpdate = true;
      materials.pick.needsUpdate = true;
      invalidate();
      console.info('[audiobank] WebGL context restored from the cached payload');
    };

    canvas.addEventListener('webglcontextlost', onLost);
    canvas.addEventListener('webglcontextrestored', onRestored);
    return () => {
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
    };
  }, [gl, buffers, points, materials, invalidate]);

  // ── Lifetime ──────────────────────────────────────────────────────────────────

  useEffect(
    () => () => {
      buffers.dispose();
      materials.dispose();
      picker.dispose();
    },
    [buffers, materials, picker],
  );

  useEffect(() => {
    onReady?.({
      buffers,
      materials,
      caps,
      picker,
      points,
      camera,
      renderer: gl,
      scene,
      invalidate,
      frameAll,
      pickAt,
      simulateContextLoss: () => {
        const extension = gl.getContext().getExtension('WEBGL_lose_context');
        if (!extension) return false;
        extension.loseContext();
        // WebKit does not restore on its own; the app asks for it back. Deferred so the
        // `webglcontextlost` handler above runs first.
        setTimeout(() => extension.restoreContext(), 0);
        return true;
      },
    });
  }, [
    onReady,
    buffers,
    materials,
    caps,
    picker,
    points,
    camera,
    invalidate,
    frameAll,
    pickAt,
    gl,
    scene,
  ]);

  return (
    <>
      <primitive object={points} />
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.6}
        zoomSpeed={0.8}
        panSpeed={0.7}
        // `invalidate()` on interaction only: drei calls it from the controls' `change`
        // event, which is the only thing that moves the camera. An idle canvas costs zero
        // frames, which for a tool that sits open next to a DAW all day is the difference
        // between a good citizen and a fan-spinner (§5.5).
        onStart={() => hoverGate.setDragging(true)}
        onEnd={() => hoverGate.setDragging(false)}
      />
    </>
  );
}
