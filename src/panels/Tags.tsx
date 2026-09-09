/**
 * Tag list, creation, and bulk-assign over the current selection (`task.md` Phase 9).
 *
 * There is no standalone "create a tag" command — `setTag` creates one on first use, the same
 * way typing a new name into the assign box both makes the tag and applies it. Color is a
 * separate `setTagColor` call, since a tag's color is a property of the tag, not of any one
 * assignment.
 */

import { useEffect, useState } from 'react';

import { listTags, setTag, setTagColor, unsetTag, type Tag } from '../ipc';
import { useSceneStore } from '../store/scene';
import { EmptyState, LoadingState } from './StateViews';

/** The samples "the current selection" means for a bulk action: the multi-select if there is
 * one, otherwise whatever the inspector has targeted. */
function useSelectionIds(): number[] {
  const selectedIds = useSceneStore((s) => s.selectedIds);
  const selectedSampleId = useSceneStore((s) => s.selectedSampleId);
  if (selectedIds.size > 0) return [...selectedIds];
  return selectedSampleId !== null ? [selectedSampleId] : [];
}

export function Tags() {
  const [tags, setTags] = useState<Tag[] | null>(null);
  const [newName, setNewName] = useState('');
  const selection = useSelectionIds();

  async function refresh() {
    setTags(await listTags());
  }

  useEffect(() => {
    void listTags()
      .then(setTags)
      .catch(() => setTags([]));
  }, []);

  async function assign(name: string) {
    if (selection.length === 0 || name.trim() === '') return;
    await Promise.all(selection.map((id) => setTag(id, name.trim())));
    await refresh();
  }

  async function remove(name: string) {
    if (selection.length === 0) return;
    await Promise.all(selection.map((id) => unsetTag(id, name)));
    await refresh();
  }

  async function pickColor(tag: Tag, color: string) {
    setTags((prev) => prev?.map((t) => (t.id === tag.id ? { ...t, color } : t)) ?? prev);
    await setTagColor(tag.id, color);
  }

  if (tags === null) return <LoadingState label="Loading tags…" />;

  return (
    <section className="space-y-2">
      <h2 className="text-xs font-medium tracking-wide text-neutral-400 uppercase">
        Tags
      </h2>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void assign(newName).then(() => setNewName(''));
        }}
        className="flex gap-1.5"
      >
        <input
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder={
            selection.length > 0 ? 'New or existing tag…' : 'Select samples first'
          }
          disabled={selection.length === 0}
          className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none disabled:opacity-40"
        />
        <button
          type="submit"
          disabled={selection.length === 0 || newName.trim() === ''}
          className="rounded border border-neutral-700 px-2 text-xs text-neutral-300 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Add
        </button>
      </form>

      {tags.length === 0 ? (
        <EmptyState title="No tags yet" detail="Select samples and add one." />
      ) : (
        <ul className="space-y-1">
          {tags.map((tag) => (
            <li
              key={tag.id}
              className="flex items-center justify-between gap-2 rounded border border-neutral-800 px-2 py-1 text-xs"
            >
              <div className="flex min-w-0 items-center gap-1.5">
                <input
                  type="color"
                  value={tag.color ?? '#6b7280'}
                  onChange={(e) => void pickColor(tag, e.target.value)}
                  className="h-3.5 w-3.5 shrink-0 cursor-pointer rounded-full border-0 bg-transparent p-0"
                  aria-label={`Color for ${tag.name}`}
                />
                <span className="truncate text-neutral-200">{tag.name}</span>
                <span className="shrink-0 font-mono text-[11px] text-neutral-500">
                  {tag.sampleCount}
                </span>
              </div>
              <div className="flex shrink-0 gap-1">
                <button
                  type="button"
                  disabled={selection.length === 0}
                  onClick={() => void assign(tag.name)}
                  className="text-[11px] text-neutral-500 hover:text-neutral-300 disabled:opacity-30"
                >
                  Assign
                </button>
                <button
                  type="button"
                  disabled={selection.length === 0}
                  onClick={() => void remove(tag.name)}
                  className="text-[11px] text-neutral-500 hover:text-neutral-300 disabled:opacity-30"
                >
                  Remove
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
