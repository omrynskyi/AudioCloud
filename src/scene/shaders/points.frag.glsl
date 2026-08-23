// Fragment stage for the point cloud (`overview.md` §5.4).
//
// A soft circular sprite from `gl_PointCoord` with a `smoothstep` edge, rather than a
// texture lookup: cheaper, sharper at every size, and it costs no texture unit and no
// upload. Blending is additive with `depthWrite: false`, which is what gives the luminous
// cluster look without the back-to-front sorting that true alpha would demand of 50,000
// sprites every time the camera moves.

precision highp float;

varying vec3 vColor;
varying float vAlpha;
varying float vEmphasis;
varying float vPointSize;

// Sprites smaller than this many framebuffer pixels are discarded outright. The LOD floor:
// below about a pixel a sprite is aliasing, not information.
uniform float uMinPointSize;
// Where the sprite's core ends and its falloff begins, in units of the sprite radius.
uniform float uCoreRadius;
// The ring drawn around a hovered or selected point, in the same units.
uniform float uRingInner;
uniform float uRingOuter;
uniform vec3 uHoverColor;
uniform vec3 uSelectColor;

void main() {
  // The vertex stage's *unclamped* size, so this reads the LOD intent rather than the
  // hardware's floor -- a point clamped up to 1 px by the driver is still sub-pixel
  // information and still goes.
  if (vPointSize < uMinPointSize) discard;

  // gl_PointCoord is [0,1] across the sprite; recentre to [-1,1] so length() is the radius.
  vec2 offset = gl_PointCoord * 2.0 - 1.0;
  float radiusSquared = dot(offset, offset);
  if (radiusSquared > 1.0) discard;
  float radius = sqrt(radiusSquared);

  float core = 1.0 - smoothstep(uCoreRadius, 1.0, radius);
  vec3 color = vColor;
  float alpha = core;

  if (vEmphasis > 0.5) {
    // An annulus, drawn in addition to the sprite rather than instead of it, so an
    // emphasized point in a dense cluster reads as *that* point rather than as a hole.
    float ring = smoothstep(uRingInner, uRingInner + 0.06, radius) *
                 (1.0 - smoothstep(uRingOuter - 0.06, uRingOuter, radius));
    vec3 ringColor = vEmphasis > 1.5 ? uSelectColor : uHoverColor;
    color = mix(color, ringColor, clamp(ring * 1.5, 0.0, 1.0));
    alpha = clamp(alpha + ring, 0.0, 1.0);
  }

  // Fade a sprite out as it approaches the discard threshold, rather than letting it wink
  // out at full brightness. Zooming out of a 50,000-point cloud crosses that threshold for
  // thousands of points at once, and without this the corpus visibly thins in steps.
  // Squared because a shrinking sprite loses coverage by area, which is what the eye reads
  // as it getting dimmer.
  float subPixel = clamp(vPointSize / max(uMinPointSize * 2.0, 1e-3), 0.0, 1.0);
  alpha *= vAlpha * mix(0.35, 1.0, subPixel * subPixel);

  if (alpha <= 0.0) discard;

  // Additive blending multiplies by alpha on the source side, so the colour is passed
  // straight through and alpha alone carries both the sprite falloff and the depth fade.
  gl_FragColor = vec4(color, alpha);
}
