/**
 * The short-term recall list for sounds heard while scrubbing across the map.
 *
 * A hover on a result here still auditions it, but only map picking writes the history (see
 * `store/scene.ts`'s `scrub`). That distinction means a user can revisit a sound without
 * rearranging the record of the route they just took through the cloud.
 */

import { Trash } from '@phosphor-icons/react';
import { useEffect, useMemo, useRef, useState } from 'react';

import { auditionProps } from '../audition';
import { getSampleDetail, type SampleDetail } from '../ipc';
import { prefetchQueue } from '../scene/prefetchQueue';
import {
  MAX_SCRUB_HISTORY,
  MIN_SCRUB_HISTORY_RANGE,
  useSceneStore,
} from '../store/scene';
import { formatMs } from './format';

export function ScrubHistory() {
  const history = useSceneStore((s) => s.scrubHistory);
  const range = useSceneStore((s) => s.scrubHistoryRange);
  const setRange = useSceneStore((s) => s.setScrubHistoryRange);
  const clear = useSceneStore((s) => s.clearScrubHistory);
  const visibleIds = useMemo(() => history.slice(0, range), [history, range]);
  const [loaded, setLoaded] = useState<{
    ids: number[];
    rows: SampleDetail[];
  } | null>(null);
  const detailCache = useRef(new Map<number, SampleDetail>());
  const pendingDetails = useRef(new Map<number, Promise<SampleDetail | null>>());
  const select = useSceneStore((s) => s.select);
  const selected = useSceneStore((s) => s.selectedSampleId);

  useEffect(() => {
    let current = true;

    // The list can update during an active sweep. Cache each detail request so every new map
    // sample adds at most one IPC call rather than refetching the whole visible history.
    function getCachedDetail(id: number): Promise<SampleDetail | null> {
      const cached = detailCache.current.get(id);
      if (cached) return Promise.resolve(cached);

      const pending = pendingDetails.current.get(id);
      if (pending) return pending;

      const request = getSampleDetail(id)
        .then((detail) => {
          detailCache.current.set(id, detail);
          return detail;
        })
        .catch(() => null)
        .finally(() => pendingDetails.current.delete(id));
      pendingDetails.current.set(id, request);
      return request;
    }

    if (visibleIds.length === 0) return;

    // These are the next samples the listener is likely to return to, so opening history is a
    // useful priority signal for the same small decode cache search results already warm.
    prefetchQueue.replacePriority([...new Set(visibleIds)]);
    void Promise.all(visibleIds.map(getCachedDetail)).then((details) => {
      if (current) {
        setLoaded({
          ids: visibleIds,
          rows: details.filter((detail): detail is SampleDetail => detail !== null),
        });
      }
    });

    return () => {
      current = false;
    };
  }, [visibleIds]);

  // A newly scrubbed id should show a small loading state rather than a briefly misordered
  // list from the previous history snapshot. `visibleIds` is memoized, so identity is a safe
  // marker for the exact set and order this render is waiting for.
  const rows = loaded?.ids === visibleIds ? loaded.rows : null;

  return (
    <section className="space-y-3">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-xs font-medium tracking-wide text-neutral-400 uppercase">
            Scrub history
          </h2>
          <p className="mt-0.5 text-[11px] text-neutral-500">
            {history.length === 0
              ? 'Samples you hover on the map appear here.'
              : `${Math.min(range, history.length)} of ${history.length} recent map samples`}
          </p>
        </div>
        {history.length > 0 && (
          <button
            type="button"
            onClick={clear}
            className="rounded-control flex h-7 w-7 shrink-0 items-center justify-center text-neutral-500 transition-colors hover:bg-neutral-800 hover:text-neutral-200"
            aria-label="Clear scrub history"
            title="Clear scrub history"
          >
            <Trash size={14} />
          </button>
        )}
      </header>

      <div>
        <div className="mb-2 flex items-center justify-between gap-3">
          <label htmlFor="scrub-history-range" className="text-xs text-neutral-300">
            History range
          </label>
          <output
            htmlFor="scrub-history-range"
            className="text-accent font-mono text-[11px]"
          >
            {range} samples
          </output>
        </div>
        <input
          id="scrub-history-range"
          type="range"
          min={MIN_SCRUB_HISTORY_RANGE}
          max={MAX_SCRUB_HISTORY}
          step={10}
          value={range}
          onChange={(event) => setRange(Number(event.target.value))}
          className="settings-range"
          style={
            {
              '--range-progress': `${
                ((range - MIN_SCRUB_HISTORY_RANGE) /
                  (MAX_SCRUB_HISTORY - MIN_SCRUB_HISTORY_RANGE)) *
                100
              }%`,
            } as React.CSSProperties
          }
          aria-label="Scrub history range"
        />
        <div className="settings-range-labels" aria-hidden="true">
          <span>{MIN_SCRUB_HISTORY_RANGE}</span>
          <span>{MAX_SCRUB_HISTORY}</span>
        </div>
      </div>

      {visibleIds.length === 0 ? (
        <p className="rounded-control border border-dashed border-neutral-800 px-2 py-3 text-center text-xs leading-relaxed text-neutral-500">
          Scrub across points on the map to build a listening trail.
        </p>
      ) : rows === null ? (
        <p className="px-1 py-1 text-xs text-neutral-500">Loading history…</p>
      ) : rows.length === 0 ? (
        <p className="rounded-control border border-dashed border-neutral-800 px-2 py-3 text-center text-xs leading-relaxed text-neutral-500">
          These samples are no longer available in the library.
        </p>
      ) : (
        <ol className="max-h-80 space-y-0.5 overflow-y-auto">
          {rows.map((row, index) => (
            <li key={`${row.id}-${index}`}>
              <button
                type="button"
                onClick={() => select(row.id)}
                {...auditionProps(row.id)}
                className={`rounded-control flex w-full items-baseline gap-2 px-2 py-1.5 text-left text-xs transition-colors ${
                  selected === row.id
                    ? 'bg-accent-muted text-accent'
                    : 'text-neutral-300 hover:bg-neutral-800 hover:text-neutral-100'
                }`}
                title={row.relPath}
              >
                <span className="w-4 shrink-0 font-mono text-[10px] text-neutral-600">
                  {index + 1}
                </span>
                <span className="min-w-0 flex-1 truncate">{row.filename}</span>
                <span className="shrink-0 font-mono text-[10px] opacity-55">
                  {formatMs(row.durationMs)}
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
