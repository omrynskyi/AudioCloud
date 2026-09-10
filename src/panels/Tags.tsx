/**
 * Browse and maintain tags from the shared contextual area. The name itself is the primary
 * browse target; assignment and editing stay nearby but appear only when relevant.
 */

import { Minus, PencilSimple, Plus, Trash } from '@phosphor-icons/react';
import { useEffect, useState } from 'react';

import {
  deleteTag,
  listTags,
  renameTag,
  setTag,
  setTagColor,
  unsetTag,
  type Tag,
} from '../ipc';
import { useSceneStore } from '../store/scene';
import { useShellStore } from '../store/shell';
import { EmptyState, LoadingState, PanelButton } from './StateViews';

/** The multi-selection wins; otherwise the inspector target is the one bulk actions affect. */
function useSelectionIds(): number[] {
  const selectedIds = useSceneStore((s) => s.selectedIds);
  const selectedSampleId = useSceneStore((s) => s.selectedSampleId);
  if (selectedIds.size > 0) return [...selectedIds];
  return selectedSampleId !== null ? [selectedSampleId] : [];
}

export function Tags({ onBrowse }: { onBrowse?: () => void }) {
  const [tags, setTags] = useState<Tag[] | null>(null);
  const [newName, setNewName] = useState('');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selection = useSelectionIds();
  const setDraft = useShellStore((s) => s.setDraft);
  const activeTags = useShellStore((s) => s.filterDraft.tags);

  async function refresh() {
    setTags(await listTags());
  }

  useEffect(() => {
    let cancelled = false;
    void listTags()
      .then((next) => {
        if (!cancelled) setTags(next);
      })
      .catch(() => {
        if (!cancelled) setTags([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function assign(name: string) {
    const trimmed = name.trim();
    if (selection.length === 0 || trimmed === '') return;
    setError(null);
    try {
      await Promise.all(selection.map((id) => setTag(id, trimmed)));
      setNewName('');
      await refresh();
    } catch {
      setError('Couldn’t add that tag. Try again.');
    }
  }

  async function remove(name: string) {
    if (selection.length === 0) return;
    setError(null);
    try {
      await Promise.all(selection.map((id) => unsetTag(id, name)));
      await refresh();
    } catch {
      setError('Couldn’t remove that tag. Try again.');
    }
  }

  function browse(name: string) {
    // Entering a saved group starts a new view. Keeping a previous text query or collection
    // constraint would make a tag look unexpectedly empty and hides what the click did.
    setDraft({ text: '', tags: [name], collectionIds: [] });
    onBrowse?.();
  }

  async function pickColor(tag: Tag, color: string | null) {
    setError(null);
    try {
      const next = await setTagColor(tag.id, color);
      setTags(
        (current) => current?.map((item) => (item.id === tag.id ? next : item)) ?? null,
      );
    } catch {
      setError('Couldn’t update that tag color. Try again.');
    }
  }

  async function rename(tag: Tag) {
    const name = renameValue.trim();
    if (name === '') return;
    setError(null);
    try {
      const next = await renameTag(tag.id, name);
      setTags(
        (current) => current?.map((item) => (item.id === tag.id ? next : item)) ?? null,
      );
      if (activeTags.includes(tag.name)) {
        setDraft({
          tags: activeTags.map((item) => (item === tag.name ? next.name : item)),
        });
      }
      setEditingId(null);
    } catch {
      setError('That tag name is already in use, or couldn’t be saved.');
    }
  }

  async function deleteCurrent(tag: Tag) {
    setError(null);
    try {
      await deleteTag(tag.id);
      setTags((current) => current?.filter((item) => item.id !== tag.id) ?? null);
      if (activeTags.includes(tag.name)) {
        setDraft({ tags: activeTags.filter((item) => item !== tag.name) });
      }
      setEditingId(null);
      setConfirmDeleteId(null);
    } catch {
      setError('Couldn’t delete that tag. Try again.');
    }
  }

  if (tags === null) return <LoadingState label="Loading tags…" />;

  return (
    <section className="space-y-2">
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-xs font-medium tracking-wide text-neutral-300 uppercase">
          Tags
        </h2>
        {selection.length > 0 && (
          <span className="bg-accent-muted text-accent shrink-0 rounded-full px-2 py-0.5 font-mono text-[10px]">
            {selection.length} selected
          </span>
        )}
      </header>

      {selection.length > 0 && (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void assign(newName);
          }}
          className="flex gap-1.5"
        >
          <label className="sr-only" htmlFor="new-tag-name">
            Tag name
          </label>
          <input
            id="new-tag-name"
            type="text"
            value={newName}
            onChange={(event) => setNewName(event.target.value)}
            placeholder="Add tag"
            className="rounded-control min-w-0 flex-1 border border-neutral-800 bg-neutral-950 px-2.5 py-1.5 text-xs text-neutral-100 placeholder:text-neutral-500 focus:border-neutral-600 focus:outline-none"
          />
          <PanelButton type="submit" disabled={newName.trim() === ''}>
            <Plus size={13} aria-hidden="true" />
            <span className="sr-only">Add tag</span>
          </PanelButton>
        </form>
      )}

      {error && <p className="px-1 text-[11px] text-red-400">{error}</p>}

      {tags.length === 0 ? (
        <EmptyState title="No tags yet" />
      ) : (
        <ul className="max-h-80 space-y-1 overflow-y-auto pr-0.5">
          {tags.map((tag) => {
            const editing = editingId === tag.id;
            const confirmingDelete = confirmDeleteId === tag.id;
            const browsing = activeTags.includes(tag.name);
            return (
              <li
                key={tag.id}
                className="rounded-control border border-neutral-800 bg-neutral-950/45"
              >
                <div className="flex items-center gap-1 p-1">
                  <button
                    type="button"
                    onClick={() => browse(tag.name)}
                    className={`rounded-control flex min-w-0 flex-1 items-center gap-2 px-1.5 py-1 text-left text-xs transition-colors ${
                      browsing
                        ? 'bg-accent-muted text-accent'
                        : 'text-neutral-200 hover:bg-neutral-800 hover:text-neutral-100'
                    }`}
                    title={`View sounds tagged ${tag.name}`}
                  >
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: tag.color ?? '#868d98' }}
                      aria-hidden="true"
                    />
                    <span className="truncate">{tag.name}</span>
                    <span className="ml-auto shrink-0 font-mono text-[10px] text-neutral-500">
                      {tag.sampleCount}
                    </span>
                  </button>

                  {selection.length > 0 && (
                    <div className="flex shrink-0 items-center gap-0.5">
                      <button
                        type="button"
                        onClick={() => void assign(tag.name)}
                        className="rounded-control flex h-6 w-6 items-center justify-center text-neutral-400 transition-colors hover:bg-neutral-800 hover:text-neutral-100"
                        aria-label={`Add selection to ${tag.name}`}
                        title={`Add selection to ${tag.name}`}
                      >
                        <Plus size={13} />
                      </button>
                      <button
                        type="button"
                        onClick={() => void remove(tag.name)}
                        className="rounded-control flex h-6 w-6 items-center justify-center text-neutral-500 transition-colors hover:bg-neutral-800 hover:text-neutral-200"
                        aria-label={`Remove selection from ${tag.name}`}
                        title={`Remove selection from ${tag.name}`}
                      >
                        <Minus size={13} />
                      </button>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => {
                      setEditingId(editing ? null : tag.id);
                      setRenameValue(tag.name);
                      setConfirmDeleteId(null);
                    }}
                    className={`rounded-control flex h-6 w-6 shrink-0 items-center justify-center transition-colors ${
                      editing
                        ? 'bg-accent-muted text-accent'
                        : 'text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200'
                    }`}
                    aria-label={`Manage ${tag.name}`}
                    title={`Manage ${tag.name}`}
                    aria-expanded={editing}
                  >
                    <PencilSimple size={13} />
                  </button>
                </div>

                {editing && (
                  <div className="space-y-2 border-t border-neutral-800 p-2">
                    <div className="flex items-center justify-between gap-3">
                      <label className="flex items-center gap-2 text-[11px] text-neutral-400">
                        <input
                          type="color"
                          value={tag.color ?? '#868d98'}
                          onChange={(event) => void pickColor(tag, event.target.value)}
                          className="h-5 w-5 cursor-pointer rounded border-0 bg-transparent p-0"
                          aria-label={`Color for ${tag.name}`}
                        />
                        Tag color
                      </label>
                      {tag.color && (
                        <button
                          type="button"
                          onClick={() => void pickColor(tag, null)}
                          className="text-[10px] text-neutral-500 transition-colors hover:text-neutral-200"
                        >
                          Reset color
                        </button>
                      )}
                    </div>

                    <form
                      onSubmit={(event) => {
                        event.preventDefault();
                        void rename(tag);
                      }}
                      className="flex gap-1.5"
                    >
                      <label className="sr-only" htmlFor={`rename-tag-${tag.id}`}>
                        Rename tag
                      </label>
                      <input
                        id={`rename-tag-${tag.id}`}
                        type="text"
                        value={renameValue}
                        onChange={(event) => setRenameValue(event.target.value)}
                        className="rounded-control min-w-0 flex-1 border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-100 focus:border-neutral-500 focus:outline-none"
                      />
                      <PanelButton type="submit" disabled={renameValue.trim() === ''}>
                        Save
                      </PanelButton>
                    </form>

                    {confirmingDelete ? (
                      <div className="flex items-center justify-between gap-2 border-t border-neutral-800 pt-2">
                        <p className="min-w-0 text-[10px] text-red-300">
                          Remove this tag from {tag.sampleCount} sound
                          {tag.sampleCount === 1 ? '' : 's'}?
                        </p>
                        <div className="flex shrink-0 gap-1">
                          <PanelButton onClick={() => setConfirmDeleteId(null)}>
                            Cancel
                          </PanelButton>
                          <PanelButton
                            variant="danger"
                            onClick={() => void deleteCurrent(tag)}
                          >
                            Delete
                          </PanelButton>
                        </div>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setConfirmDeleteId(tag.id)}
                        className="flex items-center gap-1 text-[10px] text-neutral-500 transition-colors hover:text-red-300"
                      >
                        <Trash size={12} /> Delete tag
                      </button>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
