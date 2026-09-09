/**
 * The readout at the bottom of the map: what the cursor is over, and what it looks like.
 *
 * The map auditions on hover (`audition.ts`), which means the cursor is already the app's
 * primary instrument and a sweep across a cluster is already the primary gesture. What was
 * missing was the other half of that loop: you could *hear* what you were pointing at
 * without being able to see which file it was. This is that half.
 *
 * Three decisions worth keeping:
 *
 * - **It never blocks the map.** The whole panel is `pointer-events-none`. It sits in the
 *   middle of the surface the cursor sweeps across, and a HUD that steals the pointer would
 *   cancel the very hover it exists to describe - you would sweep down into it and the
 *   sound would stop. It has nothing to click for the same reason; the inspector is where
 *   actions live.
 * - **It holds, then hands off.** Hover clearing does not empty it immediately: crossing
 *   the gap between two points is a `null` a few tens of milliseconds long, and reacting to
 *   those makes the panel strobe through a sweep. It waits out exactly the grace period the
 *   audio waits out ([`HOVER_STOP_GRACE_MS`]), so the picture and the sound leave together,
 *   and then falls back to whatever is *selected* rather than vanishing.
 * - **Its colour says which of those two it is showing.** Near-white for the hovered
 *   sample, amber for the selected one - the same two colours `scene/materials.ts` paints
 *   the rings on the map with. You can tell without reading it whether the bar is following
 *   your cursor or your last click.
 */

import { useEffect, useState } from 'react';

import { fetchPeaks, getSampleDetail, type Peaks, type SampleDetail } from '../ipc';
import { HOVER_STOP_GRACE_MS } from '../audition';
import { useSceneStore } from '../store/scene';
import { formatMs } from './format';
import { Waveform } from './Waveform';

/**
 * Wait after a hover lands before asking the backend about it.
 *
 * A sweep across a dense cluster changes `hoveredSampleId` as fast as the picker's 20 Hz
 * gate allows, and every one of those is a `get_sample_detail` on the same IPC handler the
 * map's own fetches use. Nobody reads a filename that was under the cursor for 60 ms, so
 * nothing is lost by waiting to see if the cursor stays. Anything already in [`cache`] skips
 * this entirely and paints on the current render.
 */
const SETTLE_MS = 90;

/** The near-white of `uHoverColor`, and the amber of `uSelectColor`. */
const HOVER_INK = '#e8eaee';
const SELECT_INK = '#ffc759';

interface Card {
  id: number;
  detail: SampleDetail;
  peaks: Peaks | null;
}

/**
 * Cards already fetched, so re-crossing a point you just left is instant.
 *
 * Module-level and shared by every mount, which is safe for the reason a cache usually is
 * not: the contents are keyed by sample id and immutable per revision (`peaksUrl` puts
 * `updatedAt` in the URL), so there is no stale-read to worry about short of a rescan, and
 * a rescan reloads the window's map anyway.
 */
const cache = new Map<number, Card>();
/** Enough for a long sweep through a cluster; small enough to never be worth thinking about. */
const CACHE_LIMIT = 256;

function remember(card: Card): void {
  if (cache.size >= CACHE_LIMIT) {
    // Oldest insertion first - `Map` iterates in insertion order, so this is FIFO rather
    // than LRU. The difference does not matter at this size and LRU would mean touching the
    // map on every render.
    const oldest = cache.keys().next();
    if (!oldest.done) cache.delete(oldest.value);
  }
  cache.set(card.id, card);
}

export function HoverHud() {
  // Read through hooks rather than an imperative subscription, unlike the scene: this
  // component's entire job is to re-render when the hover changes, and it is a leaf with
  // one canvas and two lines of text in it. `hover()` already collapses repeat writes of
  // the same id, so an orbit over one point does not re-render this at all.
  const hovered = useSceneStore((s) => s.hoveredSampleId);
  const selected = useSceneStore((s) => s.selectedSampleId);

  const held = useHeldHover(hovered);
  const showingHover = held !== null;
  const sampleId = held ?? selected;
  const card = useCard(sampleId);

  if (sampleId === null) return null;

  const ink = showingHover ? HOVER_INK : SELECT_INK;

  return (
    <div
      // Centred on the map rather than the window: the inspector takes the right edge when
      // it is open, and a HUD centred under it would sit off to one side of what the user
      // is actually looking at.
      className="pointer-events-none absolute bottom-5 left-1/2 z-10 w-[min(560px,calc(100%-2rem))] -translate-x-1/2"
    >
      <div
        className={`glass rise-in rounded-panel px-3.5 pt-2.5 pb-3 transition-opacity duration-150 ${
          showingHover ? 'opacity-100' : 'opacity-80'
        }`}
      >
        <div className="flex items-baseline justify-between gap-3">
          <p
            className="truncate text-[13px] font-medium text-neutral-100"
            title={card?.detail.relPath ?? ''}
          >
            {card?.detail.filename ?? '…'}
          </p>
          {card && (
            <p className="shrink-0 font-mono text-[11px] text-neutral-500">
              {formatMs(card.detail.durationMs)}
              <span className="mx-1.5 text-neutral-700">/</span>
              {card.detail.ext.toUpperCase()}
              {card.detail.features?.bpm != null && (
                <>
                  <span className="mx-1.5 text-neutral-700">/</span>
                  {card.detail.features.bpm.toFixed(0)} BPM
                </>
              )}
            </p>
          )}
        </div>
        <Waveform
          peaks={card?.peaks ?? null}
          durationMs={card?.detail.durationMs}
          color={ink}
          className="mt-2 h-9 w-full"
        />
      </div>
    </div>
  );
}

/**
 * The hovered id, held for [`HOVER_STOP_GRACE_MS`] past the moment hover clears.
 *
 * See the module docstring: without this, sweeping across a cluster empties the panel in
 * every gap between two points.
 */
function useHeldHover(hovered: number | null): number | null {
  // A new point has to be drawn in the same render that receives it. This is React's
  // documented "adjust state while rendering" pattern: it retries this component before
  // its children see the old held id. When hover clears, we deliberately leave `held` alone
  // until the timer below expires.
  const [renderedHover, setRenderedHover] = useState<number | null>(hovered);
  const [held, setHeld] = useState<number | null>(hovered);
  if (hovered !== renderedHover) {
    setRenderedHover(hovered);
    if (hovered !== null) setHeld(hovered);
  }

  useEffect(() => {
    if (hovered !== null) return;
    const timer = setTimeout(() => setHeld(null), HOVER_STOP_GRACE_MS);
    return () => clearTimeout(timer);
  }, [hovered]);

  return held;
}

/** The detail and waveform for a sample, from [`cache`] if it is there and the wire if not. */
function useCard(sampleId: number | null): Card | null {
  const [fetched, setFetched] = useState<Card | null>(null);

  useEffect(() => {
    if (sampleId === null || cache.has(sampleId)) return;
    let cancelled = false;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      void getSampleDetail(sampleId)
        .then(async (detail) => {
          // A sample whose audio will not decode has no waveform, and that is a state the
          // panel draws (an empty zero line) rather than an error it reports.
          const peaks = await fetchPeaks(detail, controller.signal).catch(() => null);
          if (cancelled) return;
          const card: Card = { id: sampleId, detail, peaks };
          remember(card);
          setFetched(card);
        })
        .catch(() => {
          // Nothing to show and nothing to say: the cursor has almost certainly moved on.
        });
    }, SETTLE_MS);

    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [sampleId]);

  if (sampleId === null) return null;
  // Cache first, so a point crossed a second time paints on this render rather than after
  // an effect - which is the difference between a HUD and a panel that blinks.
  return cache.get(sampleId) ?? (fetched?.id === sampleId ? fetched : null);
}
