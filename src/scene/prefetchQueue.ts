/**
 * A best-effort queue that warms the audio decode cache for samples nobody has asked to hear
 * yet, one at a time.
 *
 * Sequential by design, not an oversight: `audio/mod.rs` puts exactly one `Mutex<Decoder>`
 * behind every decode, playback included, so firing prefetches in parallel would not decode any
 * faster -- it would just be that many requests queuing on the same lock, with the added risk
 * of a burst of them landing in front of a *real* hover's decode instead of behind it. One at a
 * time bounds how long a real request can ever be delayed by prefetching to at most a single
 * decode's worth of time.
 *
 * Two lists, not one, because "what's worth warming" comes from two different sources with two
 * different lifetimes:
 *
 * - `priority` is the hover the cursor is on now, or the viewport the camera just settled on --
 *   short-lived and urgent. `replacePriority` throws away whatever an older call queued and did
 *   not get to, since by the time it would run the user has already moved on from it.
 * - `background` is "warm the rest of the library while nothing more urgent is happening" --
 *   long-lived, set once per loaded library, and *not* reset by `replacePriority`. Every hover
 *   and viewport settle pauses it (the drain loop always prefers `priority` first) without
 *   losing its place, so a library small enough to fully warm eventually does, however much the
 *   user pokes at it in the meantime, while a real request is never delayed by more than the one
 *   prefetch already in flight.
 */

import { prefetchSample } from '../ipc';

export class PrefetchQueue {
  private priority: number[] = [];
  private background: number[] = [];
  private running = false;

  replacePriority(sampleIds: readonly number[]): void {
    this.priority = [...sampleIds];
    this.run();
  }

  setBackground(sampleIds: readonly number[]): void {
    this.background = [...sampleIds];
    this.run();
  }

  private run(): void {
    if (this.running) return;
    this.running = true;
    void (async () => {
      for (;;) {
        const sampleId = this.priority.shift() ?? this.background.shift();
        if (sampleId === undefined) break;
        await prefetchSample(sampleId).catch(() => {});
      }
      this.running = false;
    })();
  }
}

/**
 * The one queue the app shares.
 *
 * A singleton rather than an instance per component because the thing it is scheduling against
 * is a singleton: one `Mutex<Decoder>` on the prefetch lane in `audio/mod.rs`. Two queues would
 * each believe they were running one decode at a time while between them running two, which is
 * the exact contention this class exists to avoid. It also lets a surface that is not the map
 * -- `panels/Search.tsx`, whose results are the samples a user is about to hover one after
 * another -- ask for the same warming the map asks for on a viewport settle.
 */
export const prefetchQueue = new PrefetchQueue();
