/**
 * Which samples are currently on screen.
 *
 * The map is a flat layout viewed dead-on from a fixed top-down direction with rotation
 * permanently locked (`PointCloud.tsx`'s `enableRotate={false}`), so every point sits at the
 * same view-space depth and the camera's position is always directly above the plane the cloud
 * lives on. That is what makes the frustum-to-plane intersection below exact arithmetic instead
 * of an approximation: there is no tilt to account for.
 */

import type { PerspectiveCamera } from 'three';

import type { PointCloud } from '../ipc/binary';

/**
 * Sample ids inside the camera's current view, nearest-to-center first, capped at `maxCount`.
 *
 * Nearest-first plus a cap rather than "everything in frame": a zoomed-out view can put the
 * entire library on screen at once, and decoding all of it is not what "warm the visible area"
 * is meant to cost. Nearest-to-center is the reasonable proxy for "where the cursor is about to
 * go" when nothing else (a hover) has said otherwise yet.
 */
export function visibleSamples(
  camera: PerspectiveCamera,
  cloud: PointCloud,
  maxCount: number,
): number[] {
  const verticalFov = (camera.fov * Math.PI) / 180;
  // The plane every point lives on is `z = 0` (`buffers.ts`'s interleave comment); the camera
  // sits somewhere along `+z` from it, looking straight down, so its own `z` coordinate *is*
  // the viewing distance.
  const distance = Math.abs(camera.position.z);
  const halfHeight = distance * Math.tan(verticalFov / 2);
  const halfWidth = halfHeight * camera.aspect;
  const centerX = camera.position.x;
  const centerY = camera.position.y;
  const minX = centerX - halfWidth;
  const maxX = centerX + halfWidth;
  const minY = centerY - halfHeight;
  const maxY = centerY + halfHeight;

  const { x, y, ids, count } = cloud;
  const candidates: { id: number; distSq: number }[] = [];
  for (let i = 0; i < count; i++) {
    const px = x[i] as number;
    const py = y[i] as number;
    if (px < minX || px > maxX || py < minY || py > maxY) continue;
    const dx = px - centerX;
    const dy = py - centerY;
    candidates.push({ id: ids[i] as number, distSq: dx * dx + dy * dy });
  }
  candidates.sort((a, b) => a.distSq - b.distSq);
  return candidates.slice(0, maxCount).map((candidate) => candidate.id);
}

/**
 * Every sample id in `cloud`, nearest-to-`center`-first.
 *
 * What a whole-library background warm-up (`PointCloud.tsx`'s idle prefetch sweep) wants to
 * iterate in: no cap, because the point is to eventually cover everything, but still ordered so
 * that whatever gets interrupted first -- by a real hover, panning, anything more urgent --
 * leaves the *nearest* samples warmed rather than an arbitrary prefix of database id order.
 */
export function allSamplesByDistance(
  cloud: PointCloud,
  center: { x: number; y: number },
): number[] {
  const { x, y, ids, count } = cloud;
  const byDistance: { id: number; distSq: number }[] = [];
  for (let i = 0; i < count; i++) {
    const dx = (x[i] as number) - center.x;
    const dy = (y[i] as number) - center.y;
    byDistance.push({ id: ids[i] as number, distSq: dx * dx + dy * dy });
  }
  byDistance.sort((a, b) => a.distSq - b.distSq);
  return byDistance.map((candidate) => candidate.id);
}
