/**
 * The waveform picture, drawn once and used everywhere a sample is shown.
 *
 * Extracted from `Inspector.tsx`, which used to own the only copy, because `HoverHud.tsx`
 * needs the same drawing and a waveform that looks like a different product depending on
 * which panel you are reading is not a small inconsistency in an app whose whole surface is
 * two panels.
 *
 * Three things it does that the inspector's version did not:
 *
 * - **It matches its own box.** The old canvas was a fixed 480x64 stretched by CSS to
 *   whatever width the panel happened to be, which is a blurry non-integer resample on
 *   every display and a squashed one on a wide panel. This measures the element and sizes
 *   the backing store to `clientWidth * devicePixelRatio`, so a bar is a bar.
 * - **It tells the truth about how much it drew.** `Peaks.coveredMs` is below the sample's
 *   duration whenever the decoder stopped at its ten-second window, and its own doc says
 *   so: *"Draw the waveform over this span, not over the file's length, or a four-minute
 *   loop gets a picture of its first ten seconds stretched edge to edge."* Given a
 *   `durationMs` this draws the covered span across its real fraction of the width and
 *   leaves the rest as baseline, so a long file reads as a long file.
 * - **It has a colour argument**, because hover and selection are different states with
 *   different colours in the scene (`scene/materials.ts`'s `uHoverColor` and
 *   `uSelectColor`) and the panels should not contradict the map.
 */

import { useEffect, useRef } from 'react';

import type { Peaks } from '../ipc';

export interface WaveformProps {
  peaks: Peaks | null;
  /**
   * The sample's full length, when known. Only used to decide how much of the width the
   * drawn span occupies; without it the peaks fill the box, which is right for a sample
   * short enough to have been decoded whole and wrong for one that was not.
   */
  durationMs?: number | null | undefined;
  /** CSS colour for the wave body. Defaults to the ordinary panel foreground. */
  color?: string;
  className?: string;
}

/** Matches the `text-neutral-400` the surrounding panels use for ordinary content. */
const DEFAULT_COLOR = '#a9b0ba';

export function Waveform({
  peaks,
  durationMs,
  color = DEFAULT_COLOR,
  className,
}: WaveformProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    function draw() {
      // Re-read through the ref rather than closing over `canvas`: this also runs from a
      // ResizeObserver callback, by which point the element may be gone.
      const el = ref.current;
      const ctx = el?.getContext('2d');
      if (!el || !ctx) return;

      const cssWidth = el.clientWidth;
      const cssHeight = el.clientHeight;
      if (cssWidth === 0 || cssHeight === 0) return;

      const dpr = window.devicePixelRatio || 1;
      const width = Math.round(cssWidth * dpr);
      const height = Math.round(cssHeight * dpr);
      // Assigning either dimension clears the canvas, so only do it when it actually
      // changed - otherwise every redraw of an unchanged size pays for a fresh backing
      // store allocation.
      if (el.width !== width || el.height !== height) {
        el.width = width;
        el.height = height;
      }
      ctx.clearRect(0, 0, width, height);

      const mid = height / 2;

      // The zero line, drawn whether or not there are peaks: it is what makes an empty box
      // read as "no waveform for this one" rather than as a panel that failed to load.
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.18;
      ctx.fillRect(0, Math.round(mid), width, Math.max(1, Math.round(dpr)));
      ctx.globalAlpha = 1;

      if (!peaks || peaks.buckets === 0) return;

      // How much of the box the decoded span is entitled to. Guarded on both sides: a
      // missing duration means "assume it is all there", and a `coveredMs` at or past the
      // duration means the same.
      const span =
        durationMs && durationMs > 0 && peaks.coveredMs > 0
          ? Math.min(1, peaks.coveredMs / durationMs)
          : 1;
      const drawWidth = width * span;
      const barWidth = drawWidth / peaks.buckets;

      ctx.fillStyle = color;
      for (let i = 0; i < peaks.buckets; i++) {
        const min = peaks.minMax[i * 2] ?? 0;
        const max = peaks.minMax[i * 2 + 1] ?? 0;
        const top = mid - max * mid;
        const bottom = mid - min * mid;
        ctx.fillRect(
          i * barWidth,
          top,
          // Never below one device pixel: at 400 buckets in a 300px box the honest bar
          // width is under half a pixel, and a sub-pixel rect is an invisible one.
          Math.max(dpr, barWidth - dpr * 0.5),
          Math.max(dpr, bottom - top),
        );
      }
    }

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [peaks, durationMs, color]);

  return <canvas ref={ref} className={className} />;
}
