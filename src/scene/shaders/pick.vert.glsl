// Vertex stage for the ID-colour picking pass (`overview.md` §5.3).
//
// Deliberately a near-copy of `points.vert.glsl` rather than a shared chunk with an #ifdef.
// The two shaders must agree about *where a point is and how big it is* and about nothing
// else, and an #ifdef that threaded colour, fade and emphasis through a pass that writes
// integers would make the one property that matters harder to check, not easier. The
// duplicated lines are the ones asserted on: `scene/picking.ts` documents the invariant and
// the sizing terms below are identical to the main pass by construction.

attribute float aId;
attribute float aSize;

uniform float uSizeScale;
uniform vec2 uPointSizeRange;
uniform float uFadeNear;
uniform float uFadeFar;
uniform float uFarSizeScale;
uniform float uSizeMultiplier;

varying float vId;
varying float vPointSize;

void main() {
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * viewPosition;

  float depth = max(-viewPosition.z, 1e-4);
  float recede = smoothstep(uFadeNear, uFadeFar, depth);

  // No emphasis term. A hovered point must not grow its own hit area, or the cursor sticks
  // to whatever it touched first and cannot be moved to a neighbour one pixel away.
  float radius = aSize * uSizeMultiplier * mix(1.0, uFarSizeScale, recede);

  float pixels = radius * uSizeScale / depth;
  vPointSize = pixels;
  gl_PointSize = clamp(pixels, uPointSizeRange.x, uPointSizeRange.y);
  vId = aId;
}
