/**
 * GPU picking: which point is under the cursor (`overview.md` §5.3).
 *
 * CPU raycasting is not an option at this size. Three's `Points` raycaster is a linear scan
 * over every point with a distance threshold — 50,000 distance tests per hover event, on the
 * main thread, at the exact moment the main thread is also running an orbit.
 *
 * Instead the cloud is re-rendered with `pick.frag.glsl`, which writes each point's index
 * into RGBA8, and one pixel is read back. Two things make that cheap enough to do while the
 * user is moving:
 *
 * - **A tiny viewport.** `Camera.setViewOffset` rescales the projection so that a small
 *   square around the cursor fills the whole render target. The vertex work is unchanged —
 *   still 50,000 points — but the fill is a few hundred pixels rather than four million,
 *   and 50,000 points through a trivial vertex shader is not what costs anything here.
 * - **On demand only.** `readRenderTargetPixels` is a synchronous `readPixels`, which
 *   drains the GPU pipeline. Once per click, at most twenty times a second on hover, and
 *   never while a drag is in progress. The async `PIXEL_PACK_BUFFER` path that removes the
 *   stall is a Phase 10 idea, and profiling should be what triggers it.
 *
 * ### What the window size buys, and what it costs
 *
 * A literal 1×1 pick would be almost unusable, and not for the reason it looks like. A point
 * whose *centre* falls outside the viewport is clipped before rasterization — the sprite
 * around it is never drawn — so a 1×1 window can only ever hit a point whose centre lands on
 * that exact pixel, no matter how large the sprite covering it is. Widening the window to a
 * few tens of pixels and searching outward from the middle changes the semantics to
 * something better than "what did the cursor touch": **the point whose centre is nearest the
 * cursor, and among those, the one nearest the camera.** That is what a person means by
 * clicking on a cluster.
 *
 * The cost is that a point further from the cursor than half the window cannot be picked,
 * even if its sprite covers the cursor. For a hover-to-audition interface that is the right
 * trade — the alternative is a large sprite in the foreground swallowing every click meant
 * for the detail behind it.
 */

import {
  Color,
  NearestFilter,
  UnsignedByteType,
  Vector2,
  WebGLRenderTarget,
  type Object3D,
  type PerspectiveCamera,
  type Scene,
  type ShaderMaterial,
  type WebGLRenderer,
} from 'three';

/**
 * The layer the pick pass renders.
 *
 * Layers rather than a second `Scene`, because an `Object3D` has one parent and the cloud
 * cannot be in two graphs at once. Enabling this layer on the cloud and setting the camera
 * to it means the pick pass draws the points and nothing else — no helpers, no gizmos, no
 * future overlay quietly writing an id-coloured rectangle over the whole target.
 */
export const PICK_LAYER = 1;

/** Nothing was under the cursor. */
export const NO_PICK = -1;

export interface PickerOptions {
  /**
   * Side of the pick window, in framebuffer pixels. Odd, so that "the middle" is a pixel
   * rather than a corner.
   */
  windowPx?: number;
}

const DEFAULT_WINDOW_PX = 21;

export class Picker {
  private readonly target: WebGLRenderTarget;
  private readonly pixels: Uint8Array;
  readonly windowPx: number;
  private readonly bufferSize = new Vector2();
  private readonly savedClearColor = new Color();

  constructor(options: PickerOptions = {}) {
    const requested = options.windowPx ?? DEFAULT_WINDOW_PX;
    this.windowPx = requested % 2 === 0 ? requested + 1 : requested;
    this.target = new WebGLRenderTarget(this.windowPx, this.windowPx, {
      // Nothing about an id may be interpolated, resolved, or filtered. Nearest sampling,
      // no multisampling, no mipmaps — a filtered id is the average of two unrelated
      // integers, which decodes to a third point that is not on screen at all.
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      type: UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
      samples: 0,
    });
    this.pixels = new Uint8Array(this.windowPx * this.windowPx * 4);
  }

  /**
   * Renders one pick pass and returns the cloud **index** under the cursor, or `NO_PICK`.
   *
   * `x` and `y` are CSS pixels relative to the canvas's top-left corner — what
   * `event.clientX - rect.left` gives. The conversion to framebuffer pixels happens here,
   * because it depends on the renderer's pixel ratio and nothing outside should have to
   * know that.
   */
  pick(
    renderer: WebGLRenderer,
    scene: Scene,
    camera: PerspectiveCamera,
    pickMaterial: ShaderMaterial,
    x: number,
    y: number,
  ): number {
    renderer.getDrawingBufferSize(this.bufferSize);
    const ratio = renderer.getPixelRatio();
    // `floor`, not `round`. A CSS coordinate names a position, and the pixel it is in is
    // the one that contains it -- rounding would map the exact centre of pixel *n* to
    // pixel *n + 1*, because `Math.round` breaks a .5 tie upward. It cost half an hour of
    // wondering why the harness's CPU reference disagreed with the GPU on one pixel in
    // every trial.
    const deviceX = Math.floor(x * ratio);
    const deviceY = Math.floor(y * ratio);
    if (
      deviceX < 0 ||
      deviceY < 0 ||
      deviceX >= this.bufferSize.x ||
      deviceY >= this.bufferSize.y
    ) {
      return NO_PICK;
    }

    const half = (this.windowPx - 1) / 2;

    // Everything mutated below is renderer-global. A pick that left any of it changed
    // would corrupt the next visible frame, and the symptom -- a black canvas, or the whole
    // scene drawn in id colours -- would look like a bug in the main pass.
    const previousTarget = renderer.getRenderTarget();
    const previousOverride = scene.overrideMaterial;
    const previousLayers = camera.layers.mask;
    const previousAutoClear = renderer.autoClear;
    renderer.getClearColor(this.savedClearColor);
    const previousClearAlpha = renderer.getClearAlpha();

    // `setViewOffset` takes a top-left origin, which is the same origin the pointer event
    // used, so no flip is needed here. `readRenderTargetPixels` below is bottom-up, which
    // is why the search treats the window as symmetric about its middle rather than
    // trusting a row index to mean "above".
    camera.setViewOffset(
      this.bufferSize.x,
      this.bufferSize.y,
      deviceX - half,
      deviceY - half,
      this.windowPx,
      this.windowPx,
    );
    camera.layers.set(PICK_LAYER);
    scene.overrideMaterial = pickMaterial;

    renderer.setRenderTarget(this.target);
    // Alpha zero as well as colour zero: `aId` is index + 1, so an all-zero pixel is the
    // one bit pattern that cannot be a point. `autoClear` does the clearing, so the colour
    // and the depth buffer are cleared by the same call that draws.
    renderer.autoClear = true;
    renderer.setClearColor(0x000000, 0);
    renderer.render(scene, camera);
    renderer.readRenderTargetPixels(
      this.target,
      0,
      0,
      this.windowPx,
      this.windowPx,
      this.pixels,
    );

    renderer.setRenderTarget(previousTarget);
    renderer.setClearColor(this.savedClearColor, previousClearAlpha);
    renderer.autoClear = previousAutoClear;
    scene.overrideMaterial = previousOverride;
    camera.layers.mask = previousLayers;
    camera.clearViewOffset();

    return this.searchOutward();
  }

  /**
   * The nearest hit to the middle of the window.
   *
   * Chebyshev rings outward from the centre pixel, stopping at the first ring that contains
   * anything. Within a ring the first hit in scan order wins, which is arbitrary between two
   * points equidistant from the cursor — and equidistant to the pixel is a tie no rule can
   * break in the user's favour.
   */
  private searchOutward(): number {
    const size = this.windowPx;
    const centre = (size - 1) / 2;
    for (let radius = 0; radius <= centre; radius++) {
      let found = NO_PICK;
      for (let row = centre - radius; row <= centre + radius; row++) {
        const onEdgeRow = row === centre - radius || row === centre + radius;
        for (let column = centre - radius; column <= centre + radius; column++) {
          // Interior pixels were covered by a smaller radius already.
          if (!onEdgeRow && column !== centre - radius && column !== centre + radius)
            continue;
          const index = decodePickPixel(this.pixels, (row * size + column) * 4);
          if (index !== NO_PICK) {
            found = index;
            break;
          }
        }
        if (found !== NO_PICK) break;
      }
      if (found !== NO_PICK) return found;
    }
    return NO_PICK;
  }

  /** Marks an object as pickable. Call once, on the cloud. */
  static enablePicking(object: Object3D): void {
    object.layers.enable(PICK_LAYER);
  }

  dispose(): void {
    this.target.dispose();
  }
}

/**
 * Decodes one RGBA8 pixel back to a cloud index, or `NO_PICK` for the background.
 *
 * Little-endian, matching the encoding in `pick.frag.glsl`. Multiplication rather than
 * shifts: `a << 24` is a *signed* 32-bit shift in JavaScript, so a hypothetical id above
 * 2^31 would come back negative. Nothing reaches that today; the arithmetic that cannot is
 * free.
 */
export function decodePickPixel(pixels: Uint8Array, offset: number): number {
  const encoded =
    (pixels[offset] as number) +
    (pixels[offset + 1] as number) * 256 +
    (pixels[offset + 2] as number) * 65536 +
    (pixels[offset + 3] as number) * 16777216;
  return encoded === 0 ? NO_PICK : encoded - 1;
}

/**
 * The hover gate: at most one pick per interval, and none at all while dragging.
 *
 * Separate from `Picker` because it is a policy, not a mechanism, and because the policy is
 * the part with a number in it that `overview.md` §5.3 fixes at "~20 Hz". Clicks do not go
 * through this — a click is a deliberate act and must never be dropped.
 */
export class HoverGate {
  private lastPickAt = 0;
  private dragging = false;
  readonly intervalMs: number;

  constructor(hz = 20) {
    this.intervalMs = 1000 / hz;
  }

  /** Called from the controls' `start` and `end` events. */
  setDragging(dragging: boolean): void {
    this.dragging = dragging;
    // Re-arm on release, so the first hover after a drag is immediate rather than waiting
    // out an interval that elapsed while nothing could be picked anyway.
    if (!dragging) this.lastPickAt = 0;
  }

  /** Whether a hover pick may run now. Consumes the budget when it says yes. */
  shouldPick(now = performance.now()): boolean {
    if (this.dragging) return false;
    if (now - this.lastPickAt < this.intervalMs) return false;
    this.lastPickAt = now;
    return true;
  }
}
