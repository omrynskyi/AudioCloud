/**
 * Helpers for the progress `Channel`s (`overview.md` §6.5).
 *
 * The channels themselves are created inside `commands.ts` — a caller never constructs one.
 * What lives here is the shape of *consuming* one, which is the same for all three streams
 * and easy to get subtly wrong:
 *
 * - a stream always ends in a `finished` event, on completion, cancellation **and** failure,
 *   so a UI that only handles `progress` sits at 99% forever;
 * - events arrive at up to 10 Hz, which is a React re-render every 100 ms if each one is
 *   pushed straight into state;
 * - and the terminal event is the only place an error appears, because the command's own
 *   promise resolved as soon as the job was admitted.
 */

import type { DownloadEvent } from '../bindings/DownloadEvent';
import type { RefitEvent } from '../bindings/RefitEvent';
import type { ScanEvent } from '../bindings/ScanEvent';
import type { AppError } from './errors';

/** Any of the three streams. */
type StreamEvent = ScanEvent | RefitEvent | DownloadEvent;

/** Narrows an event to the terminal one. */
export function isFinished<E extends StreamEvent>(
  event: E,
): event is Extract<E, { event: 'finished' }> {
  return event.event === 'finished';
}

/**
 * The error a terminal event carries, if it carries one.
 *
 * All three streams spell this the same way — an `error` field that is null on success — so
 * one accessor covers them. A scan is the exception worth knowing about: it reports failure
 * through `status` as well, because a scan that was *cancelled* kept everything it wrote and
 * is not an error at all.
 */
export function finishedError(
  event: Extract<StreamEvent, { event: 'finished' }>,
): AppError | null {
  return event.error ?? null;
}

/**
 * Wraps a handler so it runs at most once per animation frame.
 *
 * **The counterpart to the Rust-side throttle.** The core already coalesces to ≤ 10 Hz, so
 * this is not about message volume — it is about what a handler *does* with them. Ten
 * `setState` calls a second is ten React renders a second during precisely the operation
 * where the window must stay responsive, and a progress bar cannot show more than one value
 * per frame anyway. Terminal events bypass the throttle: dropping one would leave the UI
 * mid-scan forever.
 *
 * Returns a disposer that cancels any pending frame, for a component unmounting mid-scan.
 */
export function throttleToFrame<E extends StreamEvent>(
  handler: (event: E) => void,
): ((event: E) => void) & { dispose: () => void } {
  let pending: E | null = null;
  let frame: number | null = null;

  const flush = () => {
    frame = null;
    const event = pending;
    pending = null;
    if (event) handler(event);
  };

  const throttled = (event: E) => {
    if (isFinished(event)) {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      pending = null;
      handler(event);
      return;
    }
    pending = event;
    frame ??= requestAnimationFrame(flush);
  };

  throttled.dispose = () => {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    pending = null;
  };

  return throttled;
}
