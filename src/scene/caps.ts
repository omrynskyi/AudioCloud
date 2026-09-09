/**
 * What this GPU will actually let the point path do (`overview.md` §5.1).
 *
 * `gl_PointSize` is clamped by the driver to `ALIASED_POINT_SIZE_RANGE`, and the value is
 * not guaranteed anywhere -- WebKit runs WebGL through ANGLE on Metal and the cap can in
 * principle be as low as 64 px. That number decides an architecture: below it, sprites
 * cannot be drawn large enough for the zoomed-in look and the layer has to become
 * `InstancedMesh` quads.
 *
 * **Measured, on this machine's WKWebView: `[1, 511]` device pixels.** Eight times the
 * floor, so the `THREE.Points` path stands. `scripts/probe_webgl_caps.mjs` is that
 * measurement and reruns in seconds; `BENCHMARKS.md` records it.
 *
 * This module is the runtime half of the same question, and it exists because a cap
 * measured on one Mac is not a cap promised on another. The shader clamps to whatever
 * `read()` finds here rather than to a constant, and a context that comes back under the
 * floor says so loudly instead of quietly drawing 64-pixel dots where 200 were asked for.
 */

/**
 * The cap below which `overview.md` §5.1 says to switch to the quad path.
 *
 * Not a hard failure: a build that renders small points is worth more than a build that
 * refuses to start, and the person who needs to act on this is a developer, not the user in
 * front of it.
 */
export const POINTS_PATH_FLOOR_PX = 64;

export interface GlCaps {
  /** `ALIASED_POINT_SIZE_RANGE`, in framebuffer pixels — `[min, max]`. */
  pointSizeRange: readonly [number, number];
  /** Whether the context is WebGL2. Three.js takes it wherever it exists. */
  webgl2: boolean;
  /** `UNMASKED_RENDERER_WEBGL` where the extension is present, else `RENDERER`. */
  renderer: string;
  /** True when the reported cap is too low for the sprite path to look right. */
  belowPointsFloor: boolean;
}

/**
 * Reads the caps off a live context.
 *
 * Called once, from the canvas's `onCreated`. Every value here is a `getParameter`, which
 * is a synchronous round trip to the driver — cheap once at init, and not something to put
 * anywhere near a frame.
 */
export function readCaps(gl: WebGL2RenderingContext | WebGLRenderingContext): GlCaps {
  const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array | number[];
  // A driver that reports nonsense should not produce a `clamp(x, NaN, NaN)` in the
  // shader, which silently renders nothing at all. `[1, 64]` is the pessimistic reading of
  // §5.1 and is at least drawable.
  const min = Number.isFinite(range?.[0]) ? Number(range[0]) : 1;
  const max = Number.isFinite(range?.[1]) && Number(range[1]) > 0 ? Number(range[1]) : 64;

  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = String(
    debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
  );

  const caps: GlCaps = {
    pointSizeRange: [min, max],
    webgl2:
      typeof WebGL2RenderingContext !== 'undefined' &&
      gl instanceof WebGL2RenderingContext,
    renderer,
    belowPointsFloor: max < POINTS_PATH_FLOOR_PX,
  };

  if (caps.belowPointsFloor) {
    console.warn(
      `[audiocloud] ALIASED_POINT_SIZE_RANGE is [${min}, ${max}] on "${renderer}". ` +
        `overview.md §5.1 puts the floor for the THREE.Points path at ${POINTS_PATH_FLOOR_PX} px; ` +
        'below it the layer is meant to become InstancedMesh quads. Sprites will be clamped ' +
        'and the zoomed-in view will look wrong.',
    );
  }

  return caps;
}
