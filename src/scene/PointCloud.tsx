/**
 * The cloud, as one component that mounts once and is driven imperatively thereafter
 * (`overview.md` §5.6).
 *
 * Read the render function and notice what is *not* there: no `map` over points, no state
 * that changes when the cursor moves, no dependency on the selection. It returns one
 * `<primitive>` and a set of controls, and it re-renders only when the payload itself is
 * replaced. Everything that happens while the user is working — hovering, selecting,
 * filtering, recolouring, panning — happens through the effects and subscriptions below,
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
  MOUSE,
  Points,
  TOUCH,
  Vector2,
  Vector3,
  type PerspectiveCamera,
  type Scene,
  type WebGLRenderer,
} from 'three';

import { PREVIEW_GAIN } from '../audition';
import { getSimilar, playSample, stopPlayback } from '../ipc';
import type { PointColors } from '../ipc/binary';
import { useSceneStore } from '../store/scene';
import { CloudBuffers, DEFAULT_POINT_COLOR, type CloudBufferOptions } from './buffers';
import { readCaps, type GlCaps } from './caps';
import {
  EMBER,
  colorsFromColumn,
  colorsFromFit,
  featureDomain,
  featureRamp,
} from './colors';
import { frameBounds, updateClipPlanes } from './framing';
import { maskFrom } from './mask';
import {
  createCloudMaterials,
  pixelsPerWorldUnit,
  type CloudMaterials,
  type SharedUniforms,
} from './materials';
import { HoverGate, NO_PICK, Picker } from './picking';
import { prefetchQueue } from './prefetchQueue';
import { allSamplesByDistance, visibleSamples } from './visibility';

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
  /**
   * The active layout's fit colors, in the cloud's order, or `null`/absent before they've
   * loaded. Only ever the *default* — `colorBy` above always overrides it, same as it
   * overrides the flat `DEFAULT_POINT_COLOR` fallback this replaces when present.
   */
  fitColors?: PointColors | null;
  options?: CloudBufferOptions;
  /** Called once the scene is live, with the imperative handle. */
  onReady?: (handle: CloudHandle) => void;
}

/** A pointer that moved less than this many CSS pixels between down and up is a click. */
const CLICK_SLOP_PX = 4;

/**
 * Ceiling on hover picks per second. See `HoverGate` for why this is a safety valve against a
 * pointer device reporting faster than the display can matter, and not a cost budget -- the
 * ~20 Hz it replaces was spending 0-50 ms of pure dead time in front of every audition.
 */
const HOVER_PICK_HZ = 120;

/** Per-wheel-tick zoom factor; see the wheel effect below. */
const ZOOM_SPEED = 0.0015;

/**
 * Debounce before a hover fetches and highlights similar samples.
 *
 * Audition plays immediately on every hover (see below); this one stays debounced because it
 * guards a different cost -- `get_similar`'s tens-of-milliseconds linear scan, not the
 * decoder -- and a cursor sweeping across a dense cluster must not fire one scan per point it
 * crosses.
 */
const HOVER_SIMILARITY_DEBOUNCE_MS = 120;

/** How many nearest neighbors light up when a sample is hovered. */
const SIMILAR_HIGHLIGHT_K = 15;

/**
 * How many of those neighbors get their PCM warmed in the background while the cursor sits on
 * the current one. Small and sequential (see the prefetch loop below) on purpose: a point-cloud
 * browsing session tends to drift to a *nearby* point next, so warming a few is most of the
 * benefit, and warming all fifteen would queue that much decode work behind the single decoder
 * lock `audio/mod.rs` deliberately keeps -- exactly what would make the next *real* hover slower
 * instead of faster.
 */
const PREFETCH_NEIGHBOR_COUNT = 4;

/**
 * Debounce before a settled camera warms whatever is now on screen.
 *
 * Longer than the hover debounces: panning and zooming fire many `change` events in quick
 * succession while the gesture is still happening, and there is no point recomputing (and
 * re-queuing) the visible set until the camera has actually stopped moving.
 */
const VIEWPORT_PREFETCH_SETTLE_MS = 250;

/** How many on-screen samples get warmed after the camera settles. Capped well below what a
 *  fully zoomed-out view can put in frame -- see `visibility.ts`'s doc comment. */
const PREFETCH_VIEWPORT_MAX = 60;

/**
 * Fade planes comfortably beyond any real framing distance, so
 * `smoothstep(uFadeNear, uFadeFar, depth)` evaluates to `0` for every point. The map is a flat
 * layout viewed straight-on (see `frameAll` below) with no "far side" to recede — every point
 * sits at the same view-space depth by construction — so this holds the whole cloud at full
 * brightness/size rather than dimming it by whatever a 3D framing distance would have implied.
 */
const FADE_DISABLED_NEAR = 1e6;
const FADE_DISABLED_FAR = 1e6 + 1;

/** Looks straight down the world Z axis, so screen position is a linear function of world
 *  x/y with zero perspective distortion — the whole reason the map is trustworthy to browse
 *  by eye. */
const TOP_DOWN_DIRECTION = [0, 0, 1] as const;

export function PointCloud({
  source,
  featureValues = null,
  matchedIds = null,
  fitColors = null,
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
  const hoverGate = useMemo(() => new HoverGate(HOVER_PICK_HZ), []);

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
  const dollyOffset = useRef(new Vector3());
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
  const highlightScratch = useRef<Uint8Array | null>(null);
  const pointerDownAt = useRef<{ x: number; y: number } | null>(null);

  // ── Per-frame uniforms ────────────────────────────────────────────────────────
  //
  // Eight bytes of uniform writes per frame, and nothing else. Point size is pixels-per-
  // world-unit, which changes with the window and the pixel ratio; the fade planes are fixed
  // (see `FADE_DISABLED_NEAR`/`FAR`) but still written every frame alongside it rather than
  // once, so this stays a single easy-to-audit block instead of splitting uniform ownership
  // between here and a setup effect.
  useFrame(() => {
    const uniforms = shared.current;
    if (!uniforms) return;
    // Clip planes first: they gate whether anything below is visible at all, and they have
    // to track the *live* camera distance, not the one from when the cloud was last framed —
    // see `updateClipPlanes`'s doc comment for the "zoom in and the whole cloud vanishes" bug
    // this closes.
    updateClipPlanes(camera, buffers.bounds);
    gl.getDrawingBufferSize(drawingBuffer.current);
    uniforms.uSizeScale.value = pixelsPerWorldUnit(camera.fov, drawingBuffer.current.y);
    uniforms.uFadeNear.value = FADE_DISABLED_NEAR;
    uniforms.uFadeFar.value = FADE_DISABLED_FAR;
  });

  // ── Framing ───────────────────────────────────────────────────────────────────

  const frameAll = useCallback(() => {
    frameBounds(camera, controlsRef.current, buffers.bounds, {
      direction: TOP_DOWN_DIRECTION,
    });
    invalidate();
  }, [camera, buffers, invalidate]);

  useEffect(() => {
    frameAll();
  }, [frameAll]);

  // ── Colour ────────────────────────────────────────────────────────────────────

  useEffect(() => {
    // Reused across colour-by *and* fit-colour changes alike -- both write `count * 3`
    // floats, and a user clicking through six features (or a map that just finished a
    // re-fit) should not leave 3.6 MB behind for the collector to find mid-orbit.
    if (colorScratch.current?.length !== buffers.count * 3) {
      colorScratch.current = new Float32Array(buffers.count * 3);
    }

    if (featureValues && featureValues.length === buffers.count) {
      const mapping = colorsFromColumn(featureValues, {
        ramp: colorBy ? featureRamp(colorBy) : EMBER,
        domain: colorBy ? featureDomain(colorBy) : null,
        out: colorScratch.current,
      });
      buffers.setColors(mapping.colors);
      invalidate();
      return;
    }

    // No explicit colour-by: a fit colour, when the active layout has one, is the default --
    // falling back to the flat `DEFAULT_POINT_COLOR` only per-point, for whichever samples
    // the active run never colored (see `colorsFromFit`'s doc).
    if (fitColors && fitColors.count === buffers.count) {
      buffers.setColors(
        colorsFromFit(fitColors, {
          fallback: DEFAULT_POINT_COLOR,
          out: colorScratch.current,
        }),
      );
      invalidate();
      return;
    }

    buffers.setUniformColor(DEFAULT_POINT_COLOR);
    invalidate();
  }, [featureValues, fitColors, colorBy, buffers, invalidate]);

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

  // ── Audio preview ─────────────────────────────────────────────────────────────
  //
  // Not here. Picking below reports what the cursor is over by writing `hoveredSampleId`, and
  // `audition.ts` -- mounted once by the shell -- is what turns that into sound, for the map
  // and the rail's sample lists alike. `onPointerMove`'s `HOVER_PICK_HZ` gate is what bounds
  // how many auditions a sweep across the map can fire per second.

  // ── Similar-sound highlight ───────────────────────────────────────────────────
  //
  // Dims every point except the hovered sample and its nearest neighbors by embedding
  // similarity, so "what does this sound like" is a glance rather than a click into the
  // inspector. Debounced, unlike audition above -- a cursor sweeping across a dense cluster
  // must not fire one `get_similar` scan per point it crosses -- and the fetch is discarded
  // if the hover has already moved on by the time it resolves.

  useEffect(() => {
    let pending: ReturnType<typeof setTimeout> | null = null;
    const clearPending = () => {
      if (pending !== null) {
        clearTimeout(pending);
        pending = null;
      }
    };

    const unsubscribe = useSceneStore.subscribe(
      (state) => state.hoveredSampleId,
      (sampleId) => {
        clearPending();
        if (sampleId === null) {
          buffers.setHighlight(null);
          invalidate();
          return;
        }
        pending = setTimeout(() => {
          getSimilar(sampleId, SIMILAR_HIGHLIGHT_K)
            .then((neighbors) => {
              // Stale if the cursor has moved to a different sample, or off the cloud, since
              // this fetch started -- applying it now would highlight the wrong neighborhood.
              if (useSceneStore.getState().hoveredSampleId !== sampleId) return;
              if (highlightScratch.current?.length !== buffers.count) {
                highlightScratch.current = new Uint8Array(buffers.count);
              }
              const highlight = highlightScratch.current;
              highlight.fill(0);
              const focusIndex = buffers.indexOfSample(sampleId);
              if (focusIndex !== -1) highlight[focusIndex] = 1;
              for (const neighbor of neighbors) {
                const index = buffers.indexOfSample(neighbor.sampleId);
                if (index !== -1) highlight[index] = 1;
              }
              buffers.setHighlight(highlight);
              invalidate();

              // Warm the nearest few neighbors' decode cache while the cursor is here -- the
              // most likely next stop as the cursor keeps drifting through this cluster.
              // `prefetchQueue` supersedes this with whatever the *next* hover or viewport
              // settle asks for, so there is nothing to abort here even if the cursor has
              // already moved on by the time this resolves.
              prefetchQueue.replacePriority(
                neighbors
                  .slice(0, PREFETCH_NEIGHBOR_COUNT)
                  .map((neighbor) => neighbor.sampleId),
              );
            })
            .catch(() => {});
        }, HOVER_SIMILARITY_DEBOUNCE_MS);
      },
    );

    return () => {
      clearPending();
      unsubscribe();
    };
  }, [buffers, invalidate]);

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
      // Never during a drag, and at most `HOVER_PICK_HZ` otherwise -- see `HoverGate` for why
      // that is a safety valve rather than a budget, and why the old 20 Hz was the single
      // largest term in hover-to-sound.
      if (!hoverGate.shouldPick()) return;
      const { x, y } = toCanvas(event);
      const index = pickAt(x, y);
      useSceneStore
        .getState()
        .hover(index === NO_PICK ? null : (buffers.sampleAt(index) ?? null));
    };

    // No special stop here. Leaving the canvas for a search result in the rail is a hover
    // moving from one sample to another, not an intent to stop listening, and
    // `HOVER_STOP_GRACE_MS` is exactly long enough to let the row that is about to be entered
    // retrigger instead.
    const onPointerLeave = () => useSceneStore.getState().hover(null);

    const onClick = (event: MouseEvent) => {
      // A click that ended an orbit is not a click on a point. Without the slop test,
      // every drag that happens to end over the cloud changes the selection.
      const down = pointerDownAt.current;
      const { x, y } = toCanvas(event);
      pointerDownAt.current = null;
      if (down && Math.hypot(x - down.x, y - down.y) > CLICK_SLOP_PX) return;

      const index = pickAt(x, y);
      const sampleId = index === NO_PICK ? null : (buffers.sampleAt(index) ?? null);
      // Clicking empty space clears the selection, which is what makes the map feel like a
      // canvas rather than a list that happens to be drawn in 3D.
      useSceneStore.getState().select(sampleId);
      // Immediate, not debounced: a click is a deliberate request to hear this one now, not a
      // cursor passing through.
      if (sampleId !== null) {
        playSample(sampleId, PREVIEW_GAIN).catch(() => {});
      } else {
        stopPlayback().catch(() => {});
      }
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

  // ── Wheel: scroll to zoom ────────────────────────────────────────────────────
  //
  // `OrbitControls`' own wheel handling dollies too, but ties its speed to `zoomSpeed`'s
  // fixed multiplicative step; this handler keeps the tuned `ZOOM_SPEED` curve and the
  // explicit min/max clamp instead. `enableZoom={false}` below hands dolly entirely to this
  // handler, so there is exactly one place moving the camera on wheel input instead of two
  // listeners on the same element racing each other. Panning is a left-click drag, handled
  // natively by `OrbitControls` via the `mouseButtons`/`touches` remap on the JSX below.
  useEffect(() => {
    const canvas = gl.domElement;

    const onWheel = (event: WheelEvent) => {
      const controls = controlsRef.current;
      if (!controls) return;
      event.preventDefault();

      // `deltaMode` is almost always 0 (pixels) for trackpads and modern mice; the other two
      // only show up on the rare wheel that reports lines or pages, and a rough px-per-unit
      // guess is plenty since nobody is measuring precision here, only direction and feel.
      const scale =
        event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1;
      const deltaY = event.deltaY * scale;

      const offset = dollyOffset.current.copy(camera.position).sub(controls.target);
      const distance = offset.length();
      const min = controls.minDistance ?? 0;
      const max = controls.maxDistance ?? Infinity;
      const next = Math.min(max, Math.max(min, distance * Math.exp(deltaY * ZOOM_SPEED)));
      offset.setLength(next);
      camera.position.copy(controls.target).add(offset);

      controls.update();
      invalidate();
    };

    // `{ passive: false }` because `preventDefault` on a passive listener is a silent no-op —
    // without it the page (and the browser's own pinch-zoom) scrolls right along with the map.
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [gl, camera, invalidate]);

  // ── Viewport prefetch ─────────────────────────────────────────────────────────
  //
  // Warms the decode cache for whatever the camera settles on, so a hover landing on a point
  // the user just panned or zoomed to is a cache hit instead of the cold decode that used to be
  // "the first hover in a freshly-panned-to area is always slow." Runs once for the initial
  // framed view (the frame-all effect above has already positioned the camera by the time this
  // one runs, so there is a real view to warm from the very first frame) and again every time
  // panning or zooming settles.

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    let pending: ReturnType<typeof setTimeout> | null = null;
    const warm = () => {
      prefetchQueue.replacePriority(
        visibleSamples(camera, buffers.cloud, PREFETCH_VIEWPORT_MAX),
      );
    };
    const onChange = () => {
      if (pending !== null) clearTimeout(pending);
      pending = setTimeout(warm, VIEWPORT_PREFETCH_SETTLE_MS);
    };

    warm();
    controls.addEventListener('change', onChange);
    return () => {
      if (pending !== null) clearTimeout(pending);
      controls.removeEventListener('change', onChange);
    };
  }, [camera, buffers]);

  // ── Background prefetch sweep ────────────────────────────────────────────────
  //
  // Warms the *rest* of the library -- everything the viewport and neighbor prefetches above
  // are not already covering right now -- during whatever idle time exists between them.
  // Nearest-to-center first (`allSamplesByDistance`), so a library too large to finish warming
  // still spends its idle time on the samples closest to where the user started rather than an
  // arbitrary id-order prefix.
  //
  // Set once per loaded library, deliberately not re-triggered by hover or viewport settling --
  // `PrefetchQueue.setBackground` is exactly what keeps this from being wiped out and restarted
  // every time the user so much as moves the cursor. The Rust-side cache's byte budget
  // (`audio/mod.rs`'s `PCM_CACHE_BYTE_BUDGET`) is what actually bounds how much of this ever
  // stays resident -- a library too big to fit just self-limits to the most recently touched
  // ~1 GB rather than growing without bound.

  useEffect(() => {
    prefetchQueue.setBackground(
      allSamplesByDistance(buffers.cloud, buffers.bounds.center),
    );
  }, [buffers]);

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
        // Wheel input is handled entirely by the effect above (scroll to zoom); leaving
        // `enableZoom` on would give the same wheel event to two independent camera movers.
        enableZoom={false}
        panSpeed={0.7}
        // Pan/zoom only, permanently. Rotating the map out of its top-down lock would
        // reintroduce exactly the depth ambiguity a flat map exists to remove — screen-space
        // distance stops meaning anything the moment the camera tilts off the plane's normal.
        // It is also the whole reason the orbiting 3D view was dropped: a map you can spin
        // around is slower to scan and click through than one that just sits still.
        enableRotate={false}
        // Left-click drag (and single-finger touch) pans instead of rotating — rotation is
        // permanently off per the note above, so the button `OrbitControls` defaults to
        // rotate would otherwise do nothing.
        mouseButtons={{ LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN }}
        touches={{ ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_PAN }}
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
