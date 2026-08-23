/**
 * The canvas itself, and the four settings on it that matter (`overview.md` §5.5).
 *
 * - **`frameloop="demand"`.** An idle canvas costs zero GPU and zero battery. This is not a
 *   micro-optimization; it is the difference between a tool you leave open next to a DAW all
 *   day and one you close because the fans are on. Frames are requested by `invalidate()` on
 *   orbit input, filter change, selection change, projection swap, and for the duration of
 *   an animated transition — and by nothing else.
 * - **`antialias: false`.** MSAA does nothing for an additive point sprite, whose edge is
 *   already a `smoothstep`, and it costs fill rate at exactly the moment 50,000 sprites are
 *   overlapping.
 * - **`dpr={[1, 2]}`.** Uncapped, a Pro Display XDR quadruples the fragment load for a
 *   difference nobody can see on a soft-edged dot.
 * - **`powerPreference: 'high-performance'`.** On a two-GPU Mac, the alternative is the
 *   integrated part, and this scene is fill-bound.
 *
 * Nothing in the scene carries a React pointer handler, which is not an oversight: R3F only
 * raycasts against objects that have one, so the absence keeps its CPU raycaster out of the
 * pointer path entirely. Picking is the GPU pass in `scene/picking.ts`, on demand.
 */

import { Canvas } from '@react-three/fiber';
import type { ReactNode } from 'react';

import { PointCloud, type PointCloudProps } from './PointCloud';

/** Matches `--color-canvas` in `src/styles/index.css`. */
const CLEAR_COLOR = 0x08090b;

export interface SceneCanvasProps extends PointCloudProps {
  /** Overlays drawn inside the canvas element but outside the WebGL scene. */
  children?: ReactNode;
}

export function SceneCanvas({ children, ...cloud }: SceneCanvasProps) {
  return (
    <Canvas
      frameloop="demand"
      dpr={[1, 2]}
      gl={{
        antialias: false,
        powerPreference: 'high-performance',
        // Opaque: there is nothing behind the canvas worth compositing against, and an
        // alpha channel is a per-pixel blend the compositor does for free-of-charge
        // nothing on a full-window canvas.
        alpha: false,
        depth: true,
        stencil: false,
        // Additive sprites are written and never read back except by the picking pass,
        // which renders into its own target. Nothing needs the previous frame.
        preserveDrawingBuffer: false,
      }}
      camera={{ fov: 50, near: 0.1, far: 1000, position: [0, 0, 10] }}
      onCreated={({ gl }) => {
        gl.setClearColor(CLEAR_COLOR, 1);
      }}
    >
      <PointCloud {...cloud} />
      {children}
    </Canvas>
  );
}
