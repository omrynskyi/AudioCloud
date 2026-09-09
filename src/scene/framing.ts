/**
 * Putting the camera where the cloud is.
 *
 * Coordinates arrive in the projector's own units and nothing normalizes them. That is
 * deliberate and it is worth being explicit about, because "just scale the cloud into a unit
 * cube on load" is the obvious thing to do and it would quietly undo Phase 5.
 *
 * `overview.md` §3.8 goes to real trouble to keep a layout **stable** across re-fits:
 * Procrustes alignment with reflection allowed, so that a re-fit lands the corpus back where
 * the user last saw it and the map they have learned still means something. Procrustes
 * preserves scale. Renormalizing the coordinates on arrival would throw that away — add
 * 5,000 samples to a 50,000-sample library, watch the extent grow by two percent, and every
 * point on screen moves, even though the alignment worked perfectly and nothing actually
 * moved.
 *
 * So the buffer keeps the numbers the core sent, and the *camera* adapts. The framing is
 * recomputed only when the cloud is replaced, which is exactly when the user expects the
 * view to change.
 */

import type { PerspectiveCamera, Vector3 } from 'three';

import type { CloudBounds } from './buffers';

/** Anything with a `target` and an `update()`: `OrbitControls`, or a test double. */
export interface FramingControls {
  target: Vector3;
  update(): void;
  minDistance?: number;
  maxDistance?: number;
}

export interface FrameOptions {
  /** Headroom around the bounding sphere. 1 fills the frame exactly. */
  padding?: number;
  /** Direction the camera looks from, as an offset from the centre. Normalized internally. */
  direction?: readonly [number, number, number];
}

const DEFAULTS = {
  padding: 1.35,
  // Slightly above and off-axis. The app itself always overrides this with a dead-on top-down
  // direction (`PointCloud.tsx`'s `TOP_DOWN_DIRECTION`) — the interactive view is a flat map,
  // not an orbit, so a tilted first impression has no one left to sell it to. This default
  // survives only for the orbit-sweep performance harness (`profile/main.tsx`), which still
  // wants a 3D camera path to measure render throughput under camera movement.
  direction: [0.62, 0.42, 1] as const,
} as const;

/**
 * Moves the camera so the whole cloud is in frame, and sets the clip planes around it.
 *
 * The distance uses whichever field of view is narrower. `PerspectiveCamera.fov` is the
 * *vertical* one, so on a wide window the vertical fit is binding and on a tall one — a
 * narrow panel, a split view — the horizontal is. Fitting only to the vertical would crop
 * the sides of the cloud on any window taller than it is wide.
 */
export function frameBounds(
  camera: PerspectiveCamera,
  controls: FramingControls | null,
  bounds: CloudBounds,
  options: FrameOptions = {},
): void {
  const { padding, direction } = { ...DEFAULTS, ...options };
  // An empty or single-point cloud still needs a camera somewhere sane.
  const radius = Math.max(bounds.radius, 1e-3);

  const verticalFov = (camera.fov * Math.PI) / 180;
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
  const distance =
    (radius * padding) / Math.sin(Math.min(verticalFov, horizontalFov) / 2);

  const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
  camera.position.set(
    bounds.center.x + (direction[0] / length) * distance,
    bounds.center.y + (direction[1] / length) * distance,
    bounds.center.z + (direction[2] / length) * distance,
  );

  // Clip planes around the *cloud*, not around the framing distance. Depth precision is
  // what decides whether the picking pass can tell two points in the same cluster apart,
  // and a default 0.1/2000 pair against a cloud ten units across spends almost the whole
  // buffer on empty space in front of it.
  //
  // Note that `padding` does not appear here, which it did in the first version and which
  // was a bug waiting for its first caller: padding moves the camera, it does not change
  // how deep the cloud is. Deriving the near plane from `distance - radius * padding`
  // happens to be safe while padding is above one and slices the front off the cloud the
  // moment somebody frames it tighter than that.
  camera.near = Math.max(distance - radius * 1.1, distance / 1000);
  // Far enough that zooming out to `maxDistance` does not clip the back of the cloud.
  camera.far = distance * 6 + radius * 1.1;
  camera.lookAt(bounds.center);
  camera.updateProjectionMatrix();

  if (controls) {
    controls.target.copy(bounds.center);
    // Stops a scroll wheel from either burying the camera inside a cluster with no way out
    // or throwing the cloud to a vanishing point.
    controls.minDistance = radius * 0.02;
    controls.maxDistance = distance * 6;
    controls.update();
  }
}

/**
 * Recomputes the camera's near/far clip planes for its **current** distance from the cloud.
 *
 * `frameBounds` sets `camera.near`/`camera.far` once, tightly around the cloud, at the
 * distance needed to fit the whole thing in frame. That is correct at that exact moment and
 * silently wrong the instant `OrbitControls` moves the camera anywhere else: zooming in
 * dollies the camera *toward* the cloud, and the near plane — fixed at the original framing
 * distance minus the radius — stays put. The moment the live distance drops below that fixed
 * near plane, every point in the cloud is behind it and the whole cloud clips at once, which
 * is exactly the "zoom in and everything disappears" bug this function exists to fix.
 *
 * The fade shader is permanently disabled now (`PointCloud.tsx`'s `FADE_DISABLED_NEAR`/`FAR`
 * — a flat map has no far side to recede), but the clip planes still have to track the live
 * camera distance: zoom is still a dolly, even with rotation locked out.
 *
 * Call every frame the camera might have moved. `updateProjectionMatrix` is only called when
 * the planes actually changed, since it is not free and this runs inside `useFrame`.
 */
export function updateClipPlanes(camera: PerspectiveCamera, bounds: CloudBounds): void {
  const distance = camera.position.distanceTo(bounds.center);
  const radius = Math.max(bounds.radius, 1e-3);
  const near = Math.max(distance - radius * 1.2, distance / 1000);
  const far = distance + radius * 1.5;
  if (camera.near !== near || camera.far !== far) {
    camera.near = near;
    camera.far = far;
    camera.updateProjectionMatrix();
  }
}
