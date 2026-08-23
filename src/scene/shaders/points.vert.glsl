// Vertex stage for the point cloud (`overview.md` §5.4).
//
// Everything that varies per point is an attribute; everything that varies per frame is a
// uniform. Nothing here is computed on the CPU per point, which is the whole reason the
// layer can hold 50,000 points at 60 fps: the per-frame cost is one draw call and about a
// dozen uniform writes.
//
// Written in GLSL ES 1.00. Three.js compiles its own materials this way and WebGL2 accepts
// it, so there is nothing to gain from `GLSL3` here and one fewer thing that can differ
// between the WebView and a desktop browser.

// The cloud index, plus one. Not the sample id -- see `scene/buffers.ts`. Zero is reserved
// so the picking pass can spell "background" as an all-zero pixel.
attribute float aId;
// sRGB-ish linear colour, written by `scene/colors.ts`. DynamicDrawUsage.
attribute vec3 aColor;
// World-space radius. Filtering shrinks rather than hides, so this carries the filter
// state too. DynamicDrawUsage.
attribute float aSize;

// Framebuffer pixels per world unit at one unit of view depth: viewportHeightPx /
// (2 tan(fov/2)). Passing the derived constant rather than fov and height keeps the
// division out of the shader and makes the orthographic case a matter of what the CPU
// passes in.
uniform float uSizeScale;
// The queried ALIASED_POINT_SIZE_RANGE. Clamping to the *device's* reported range rather
// than to a constant is the point of `scene/caps.ts`; see `overview.md` §5.1.
uniform vec2 uPointSizeRange;

// View-space depths between which the cloud fades out. Recomputed from the camera each
// frame -- two uniform writes, so the far side of the cloud recedes instead of cluttering.
uniform float uFadeNear;
uniform float uFadeFar;
// What a fully-faded point keeps. Not zero: the far side of the cloud should read as haze,
// not as absence.
uniform float uFarAlpha;
// Level of detail, in the vertex shader rather than by swapping geometry: a fully-faded
// point is also drawn this much smaller.
uniform float uFarSizeScale;

// `aId` of the hovered and selected points, or 0.0 for none.
uniform float uHoverId;
uniform float uSelectedId;
// How much larger an emphasized point is drawn.
uniform float uEmphasisScale;

// Global multiplier on every radius, so a UI slider does not have to rewrite 50,000 floats.
uniform float uSizeMultiplier;

varying vec3 vColor;
varying float vAlpha;
// 0 = ordinary, 1 = hovered, 2 = selected. A float because GLSL ES 1.00 has no flat
// qualifier; the values are integral and the interpolation across a point sprite is
// constant anyway.
varying float vEmphasis;
// The unclamped pixel diameter, so the fragment stage can discard sub-pixel sprites and
// compensate the ones just above the threshold.
varying float vPointSize;

void main() {
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * viewPosition;

  // Positive distance in front of the camera. The guard keeps a point exactly on the eye
  // plane from producing an infinite size rather than a clamped one.
  float depth = max(-viewPosition.z, 1e-4);

  // 0 at the near edge of the cloud, 1 at the far edge.
  float recede = smoothstep(uFadeNear, uFadeFar, depth);

  // Selection wins over hover: hovering the selected point should not un-mark it.
  float emphasis = 0.0;
  if (uSelectedId > 0.5 && abs(aId - uSelectedId) < 0.5) {
    emphasis = 2.0;
  } else if (uHoverId > 0.5 && abs(aId - uHoverId) < 0.5) {
    emphasis = 1.0;
  }

  float radius = aSize * uSizeMultiplier * mix(1.0, uFarSizeScale, recede);
  if (emphasis > 0.5) {
    radius *= uEmphasisScale;
  }

  float pixels = radius * uSizeScale / depth;
  vPointSize = pixels;
  gl_PointSize = clamp(pixels, uPointSizeRange.x, uPointSizeRange.y);

  vColor = aColor;
  vAlpha = mix(1.0, uFarAlpha, recede);
  vEmphasis = emphasis;
}
