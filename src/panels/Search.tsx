/**
 * Debounced text search over filename and tags (`task.md` Phase 9). Writes into the shared
 * filter draft in `store/shell.ts`, which merges it with `panels/Filters.tsx`'s structured
 * constraints and pushes one `QueryFilter` to `store/scene.ts` after a pause in typing.
 *
 * `SearchResults` is the other half of the same checklist item — the map highlights matches
 * (`store/scene.ts`'s `filter` feeds `PointCloud`'s mask, unchanged since Phase 7) but that
 * alone is not "listed." `Shell.tsx` already runs the `query_samples` fetch this needs, so
 * this just renders what it found rather than issuing a second query.
 */

import { useEffect, useState } from 'react';

import { getSampleDetail, type SampleDetail } from '../ipc';
import { useSceneStore } from '../store/scene';
import { useShellStore } from '../store/shell';
import { EmptyState, LoadingState } from './StateViews';

/** Past this many matches, fetching a detail row per id stops being "a list" and starts
 * being a stampede on the read pool — the count is still shown, the rows are not. */
const LIST_CAP = 50;

export function Search() {
  const text = useShellStore((s) => s.filterDraft.text);
  const setDraft = useShellStore((s) => s.setDraft);

  return (
    <div className="relative">
      <input
        data-search-input
        type="text"
        value={text}
        onChange={(e) => setDraft({ text: e.target.value })}
        placeholder="Search filenames, tags…  (/)"
        className="w-full rounded border border-neutral-800 bg-neutral-950 px-2.5 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none"
      />
      {text && (
        <button
          type="button"
          onClick={() => setDraft({ text: '' })}
          className="absolute top-1/2 right-2 -translate-y-1/2 text-neutral-600 hover:text-neutral-300"
          aria-label="Clear search"
        >
          ×
        </button>
      )}
    </div>
  );
}

export function SearchResults({ ids }: { ids: Uint32Array | null }) {
  const [rows, setRows] = useState<SampleDetail[] | null>(null);
  const select = useSceneStore((s) => s.select);
  const filter = useSceneStore((s) => s.filter);

  useEffect(() => {
    // No reset for the "outside the cap" cases below — the render guards on `ids` directly
    // and never reads a stale `rows` for them, so there is nothing to clear.
    if (!ids || ids.length === 0 || ids.length > LIST_CAP) return;
    void Promise.all(Array.from(ids, (id) => getSampleDetail(id)))
      .then(setRows)
      .catch(() => setRows([]));
  }, [ids]);

  if (!filter || !ids) return null;

  if (ids.length === 0) {
    return <EmptyState title="0 results" detail="Nothing matches this filter." />;
  }

  if (ids.length > LIST_CAP) {
    return (
      <p className="font-mono text-[11px] text-neutral-500">
        {ids.length.toLocaleString()} results — narrow the search to list them
      </p>
    );
  }

  if (!rows) return <LoadingState label="Loading results…" />;

  return (
    <ul className="max-h-40 space-y-0.5 overflow-y-auto">
      {rows.map((row) => (
        <li key={row.id}>
          <button
            type="button"
            onClick={() => select(row.id)}
            className="w-full truncate rounded px-1 py-0.5 text-left text-xs text-neutral-400 hover:bg-neutral-900"
            title={row.relPath}
          >
            {row.filename}
          </button>
        </li>
      ))}
    </ul>
  );
}
