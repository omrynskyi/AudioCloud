/**
 * The two materials the cloud is drawn with, and the uniforms they share.
 *
 * They are built together, by one function, for a reason that is easy to lose later: the
 * picking pass is only correct while it puts every point in exactly the same place, at
 * exactly the same size, as the pass the user is looking at. A pick that used last frame's
 * `uSizeScale`, or a slightly different LOD floor, would return the point next to the one
 * under the cursor — intermittently, in dense regions, and never in a way anyone could
 * reproduce.
 *
 * So the seven uniforms that decide geometry are not copied between the materials, they are
 * **the same objects** in both `uniforms` records. Three.js reads `uniform.value` at draw
 * time, so writing one writes both, and the two passes cannot drift apart even if somebody
 * later forgets that they must not.
 *
 * Colour handling, since it is invisible until it is wrong: this is a raw `ShaderMaterial`,
 * so Three.js injects neither the tone-mapping chunk nor the output-colour-space chunk. The
 * ramp values in `scene/colors.ts` are therefore written to the framebuffer as they are, and
 * are display-referred values chosen against this app's near-black background rather than
 * linear-light radiometry to be converted later.
 */

import {
  AdditiveBlending,
  Color,
  NoBlending,
  ShaderMaterial,
  Vector2,
  type IUniform,
} from 'three';

import type { GlCaps } from './caps';
import pickFragmentSource from './shaders/pick.frag.glsl?raw';
import pickVertexSource from './shaders/pick.vert.glsl?raw';
import pointsFragmentSource from './shaders/points.frag.glsl?raw';
import pointsVertexSource from './shaders/points.vert.glsl?raw';

/** The uniforms both passes read. Writing one of these changes what picking sees. */
export interface SharedUniforms {
  /** Framebuffer pixels per world unit at unit depth. Set from the camera every frame. */
  uSizeScale: IUniform<number>;
  /** The device's `ALIASED_POINT_SIZE_RANGE`, from `scene/caps.ts`. */
  uPointSizeRange: IUniform<Vector2>;
  uFadeNear: IUniform<number>;
  uFadeFar: IUniform<number>;
  uFarSizeScale: IUniform<number>;
  uSizeMultiplier: IUniform<number>;
  /** The LOD floor, in framebuffer pixels, below which a sprite is discarded. */
  uMinPointSize: IUniform<number>;
}

/** Uniforms only the visible pass has. */
export interface DisplayUniforms {
  uFarAlpha: IUniform<number>;
  uHoverId: IUniform<number>;
  uSelectedId: IUniform<number>;
  uEmphasisScale: IUniform<number>;
  uCoreRadius: IUniform<number>;
  uRingInner: IUniform<number>;
  uRingOuter: IUniform<number>;
  uHoverColor: IUniform<Color>;
  uSelectColor: IUniform<Color>;
}

export interface CloudMaterials {
  points: ShaderMaterial;
  pick: ShaderMaterial;
  shared: SharedUniforms;
  display: DisplayUniforms;
  dispose(): void;
}

/** Tunables that are not worth a uniform-by-uniform API. */
export interface MaterialOptions {
  /** Sprites below this many framebuffer pixels are discarded. See `points.frag.glsl`. */
  minPointSize?: number;
  /** What the far side of the cloud keeps, in opacity and in size. */
  farAlpha?: number;
  farSizeScale?: number;
  /** How much larger a hovered or selected point is drawn. */
  emphasisScale?: number;
}

const DEFAULTS = {
  // Under a pixel. High enough that a sprite the rasterizer would only alias is dropped, low
  // enough that zooming out does not thin the corpus into a different shape.
  //
  // This also sets the width of `points.frag.glsl`'s sub-pixel dimming ramp — a point is
  // dimmed toward `alpha *= 0.35` for anything under `2 * uMinPointSize`. At the framing
  // distance `framing.ts`'s `frameBounds` puts the camera by default, an ordinary point
  // renders at only about a pixel across (`radiusFraction`'s own doc comment on why this is
  // independent of library size or coordinate scale), which used to sit deep inside that
  // dimming zone — the map read as "faint," which was this constant doing exactly what its
  // comment said and being too conservative about it. Lower, so a point at its natural
  // framed size lands outside the ramp instead of at the dim end of it.
  minPointSize: 0.6,
  farAlpha: 0.22,
  farSizeScale: 0.55,
  emphasisScale: 1.9,
} as const;

export function createCloudMaterials(
  caps: GlCaps,
  options: MaterialOptions = {},
): CloudMaterials {
  const settings = { ...DEFAULTS, ...options };

  const shared: SharedUniforms = {
    uSizeScale: { value: 1 },
    uPointSizeRange: {
      value: new Vector2(caps.pointSizeRange[0], caps.pointSizeRange[1]),
    },
    uFadeNear: { value: 0 },
    uFadeFar: { value: 1 },
    uFarSizeScale: { value: settings.farSizeScale },
    uSizeMultiplier: { value: 1 },
    uMinPointSize: { value: settings.minPointSize },
  };

  const display: DisplayUniforms = {
    uFarAlpha: { value: settings.farAlpha },
    uHoverId: { value: 0 },
    uSelectedId: { value: 0 },
    uEmphasisScale: { value: settings.emphasisScale },
    // Fraction of the sprite's radius that reads as solid before `points.frag.glsl`'s
    // falloff begins. 0.35 spent most of a small sprite's few pixels on soft edge and almost
    // none on solid colour, which reads as a smudge rather than a point; 0.5 gives it an
    // actual visible core at the sizes this cloud renders at in practice.
    uCoreRadius: { value: 0.5 },
    uRingInner: { value: 0.62 },
    uRingOuter: { value: 0.92 },
    uHoverColor: { value: new Color(0.98, 0.98, 1.0) },
    uSelectColor: { value: new Color(1.0, 0.78, 0.35) },
  };

  const points = new ShaderMaterial({
    vertexShader: pointsVertexSource,
    fragmentShader: pointsFragmentSource,
    uniforms: { ...shared, ...display },
    transparent: true,
    // §5.4: additive with depth *test* on and depth *write* off. Test on, so the cloud
    // still occludes anything opaque behind it; write off, so 50,000 overlapping sprites
    // do not need sorting back-to-front every time the camera moves one degree.
    blending: AdditiveBlending,
    depthTest: true,
    depthWrite: false,
    toneMapped: false,
  });

  const pick = new ShaderMaterial({
    vertexShader: pickVertexSource,
    fragmentShader: pickFragmentSource,
    uniforms: { ...shared },
    // The exact opposite, and deliberately so: an integer id must not be blended with
    // anything, and depth write is what makes the surviving pixel the *nearest* point
    // rather than the last one in buffer order.
    blending: NoBlending,
    depthTest: true,
    depthWrite: true,
    transparent: false,
    toneMapped: false,
  });

  return {
    points,
    pick,
    shared,
    display,
    dispose() {
      points.dispose();
      pick.dispose();
    },
  };
}

/**
 * Framebuffer pixels per world unit at one unit of view depth.
 *
 * The whole of perspective point sizing, precomputed on the CPU so the shader does one
 * multiply and one divide. `gl_PointSize` is in framebuffer pixels, which is why the device
 * pixel ratio belongs in this constant and not in the shader: at dpr 2 a point must be
 * physically twice as many pixels across to look the same size.
 */
export function pixelsPerWorldUnit(
  fovDegrees: number,
  drawingBufferHeight: number,
): number {
  const fov = (fovDegrees * Math.PI) / 180;
  return drawingBufferHeight / (2 * Math.tan(fov / 2));
}
