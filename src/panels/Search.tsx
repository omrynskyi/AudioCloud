/**
 * Text search over filename and tags, and the list of what it matched.
 *
 * Writes into the shared filter draft in `store/shell.ts`, which debounces before pushing one
 * `QueryFilter` to `store/scene.ts`. The map highlights the matches (`store/scene.ts`'s
 * `filter` feeds `PointCloud`'s mask, unchanged since Phase 7); this lists them, from the
 * `query_samples` fetch `Shell.tsx` already runs, rather than issuing a second query for the
 * same ids.
 *
 * It reads as one control now rather than a field with a list stapled under it: the shell's
 * existing glass toolbar grows beneath the field to hold the results. There is no nested
 * result surface to compete with the rest of the toolbar.
 */

import { MagnifyingGlass, X } from '@phosphor-icons/react';
import { useEffect, useState } from 'react';

import { auditionProps } from '../audition';
import { getSampleDetail, type AppError, type SampleDetail } from '../ipc';
import { prefetchQueue } from '../scene/prefetchQueue';
import { useSceneStore } from '../store/scene';
import { useShellStore } from '../store/shell';
import { TextInput } from './Chrome';
import { formatMs } from './format';

/**
 * Past this many matches, fetching a detail row per id stops being "a list" and starts being a
 * stampede on the read pool. The count is still shown, the rows are not.
 */
const LIST_CAP = 50;

/** `useShortcuts.ts` focuses this for the `/` shortcut. */
export const SEARCH_INPUT_ID = 'search-input';

export function Search({
  onActivate,
  className = '',
}: {
  /** Opens the one shared contextual area when the search field becomes relevant. */
  onActivate: () => void;
  className?: string;
}) {
  const text = useShellStore((s) => s.filterDraft.text);
  const setDraft = useShellStore((s) => s.setDraft);

  return (
    <div className={className}>
      <div className="relative">
        <MagnifyingGlass
          size={13}
          className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-neutral-500"
        />
        <TextInput
          id={SEARCH_INPUT_ID}
          value={text}
          onChange={(value) => {
            setDraft({ text: value });
            onActivate();
          }}
          onFocus={onActivate}
          placeholder="Search filenames and tags"
          ariaLabel="Search filenames and tags"
          hasLeadingIcon
        />
        {text && (
          <button
            type="button"
            onClick={() => setDraft({ text: '' })}
            className="absolute top-1/2 right-2 -translate-y-1/2 text-neutral-500 hover:text-neutral-200"
            aria-label="Clear search"
          >
            <X size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

/** The search body rendered inside the shared expanding toolbar, never as its own popover. */
export function SearchResults({
  ids,
  error,
}: {
  ids: Uint32Array | null;
  error: AppError | null;
}) {
  const [rows, setRows] = useState<SampleDetail[] | null>(null);
  const select = useSceneStore((s) => s.select);
  const selected = useSceneStore((s) => s.selectedSampleId);
  const filter = useSceneStore((s) => s.filter);
  const tags = useShellStore((s) => s.filterDraft.tags);
  const collectionIds = useShellStore((s) => s.filterDraft.collectionIds);
  const setDraft = useShellStore((s) => s.setDraft);

  useEffect(() => {
    // No reset for the "outside the cap" cases below - the render guards on `ids` directly
    // and never reads a stale `rows` for them, so there is nothing to clear.
    if (!ids || ids.length === 0 || ids.length > LIST_CAP) return;
    void Promise.all(Array.from(ids, (id) => getSampleDetail(id)))
      .then(setRows)
      .catch(() => setRows([]));
    // A listed result is a row the user is about to hover, and searching is exactly the case
    // the map's own warming misses: the matches are wherever they are in the cloud, not near
    // the camera, so the background sweep may be nowhere near them yet.
    prefetchQueue.replacePriority(Array.from(ids));
  }, [ids]);

  if (!filter) return null;

  const activeFilters =
    tags.length > 0 || collectionIds.length > 0 ? (
      <div className="mb-1.5 flex flex-wrap gap-1 border-b border-neutral-800 px-1 pb-1.5">
        {tags.map((tag) => (
          <button
            key={tag}
            type="button"
            onClick={() => setDraft({ tags: tags.filter((value) => value !== tag) })}
            className="border-accent/40 bg-accent-muted text-accent rounded-full border px-2 py-0.5 text-[10px] transition-colors hover:bg-neutral-800"
            aria-label={`Clear ${tag} tag filter`}
          >
            {tag} <span aria-hidden="true">×</span>
          </button>
        ))}
        {collectionIds.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() =>
              setDraft({ collectionIds: collectionIds.filter((value) => value !== id) })
            }
            className="border-accent/40 bg-accent-muted text-accent rounded-full border px-2 py-0.5 text-[10px] transition-colors hover:bg-neutral-800"
            aria-label="Clear collection filter"
          >
            Collection <span aria-hidden="true">×</span>
          </button>
        ))}
      </div>
    ) : null;

  if (error) {
    return (
      <div className="space-y-1">
        {activeFilters}
        <p className="px-1 py-1 text-xs text-red-400">Search failed. Try again.</p>
      </div>
    );
  }

  if (!ids) return null;

  if (ids.length === 0) {
    return (
      <div className="space-y-1">
        {activeFilters}
        <p className="px-1 py-1 text-xs text-neutral-500">Nothing matches.</p>
      </div>
    );
  }

  if (ids.length > LIST_CAP) {
    return (
      <div className="space-y-1">
        {activeFilters}
        <p className="px-1 py-1 text-xs text-neutral-500">
          <span className="font-mono text-neutral-300">
            {ids.length.toLocaleString()}
          </span>{' '}
          matches. Narrow the search to list them.
        </p>
      </div>
    );
  }

  if (!rows) {
    return (
      <div className="space-y-1">
        {activeFilters}
        <p className="px-1 py-1 font-mono text-xs text-neutral-500">Loading…</p>
      </div>
    );
  }

  return (
    <div className="max-h-80 overflow-y-auto">
      {activeFilters}
      <ul className="space-y-0.5">
        {rows.map((row) => (
          <li key={row.id}>
            <button
              type="button"
              onClick={() => select(row.id)}
              {...auditionProps(row.id)}
              className={`rounded-control flex w-full cursor-grab items-baseline justify-between gap-2 px-2 py-1 text-left text-xs transition-colors active:cursor-grabbing ${
                selected === row.id
                  ? 'bg-accent-muted text-accent'
                  : 'text-neutral-300 hover:bg-neutral-800 hover:text-neutral-100'
              }`}
              title={row.relPath}
            >
              <span className="truncate">{row.filename}</span>
              <span className="shrink-0 font-mono text-[10px] opacity-55">
                {formatMs(row.durationMs)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
