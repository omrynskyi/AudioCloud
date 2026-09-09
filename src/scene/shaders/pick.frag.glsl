// Fragment stage for the ID-colour picking pass (`overview.md` §5.3).
//
// Writes a 32-bit integer into RGBA8 and nothing else: no lighting, no blending, depth test
// and depth *write* both on, so the pixel that survives is the point nearest the camera.
// That is what makes a pick under a dense cluster correct rather than arbitrary --
// the main pass has `depthWrite: false` and draws in buffer order, which is exactly the
// wrong basis for deciding what the user meant to click.
//
// `highp` is load-bearing. `aId` is an integer in a float, and at mediump (10-bit mantissa)
// the arithmetic below starts returning a neighbour's id somewhere around 2048 points.

precision highp float;

varying float vId;
varying float vPointSize;

uniform float uMinPointSize;

void main() {
  // The same two rejections as the main pass, in the same order. A point the user cannot
  // see must not be pickable, and a sprite is a disc rather than the square the rasterizer
  // hands us -- picking the corner of an invisible square is the classic way a pick lands
  // on a point three rows behind the one under the cursor.
  if (vPointSize < uMinPointSize) discard;
  vec2 offset = gl_PointCoord * 2.0 - 1.0;
  if (dot(offset, offset) > 1.0) discard;

  // Little-endian byte order, matching `decodePickPixel` in `scene/picking.ts`.
  float id = vId;
  float r = mod(id, 256.0);
  id = floor(id / 256.0);
  float g = mod(id, 256.0);
  id = floor(id / 256.0);
  float b = mod(id, 256.0);
  id = floor(id / 256.0);
  float a = mod(id, 256.0);

  // Exact by specification, not by luck: writing to an RGBA8 target converts each float
  // with round-to-nearest of `value * 255`, so an integral byte divided by 255 comes back
  // as that byte. The half-texel nudge this kind of code often carries would *break* that
  // -- 0.5/255 is a tie, and a tie is where drivers disagree.
  gl_FragColor = vec4(r, g, b, a) / 255.0;
}
