/**
 * Saved collections in the shared contextual area. Names are browse targets; sequence editing
 * and lower-frequency actions remain contained inside an expanded collection.
 */

import {
  CaretDown,
  DownloadSimple,
  Eye,
  PencilSimple,
  Plus,
  Trash,
  X,
} from '@phosphor-icons/react';
import { save as saveDialog } from '@tauri-apps/plugin-dialog';
import { useEffect, useState } from 'react';

import {
  addToCollection,
  createCollection,
  deleteCollection,
  exportCollection,
  getCollection,
  listCollections,
  removeFromCollection,
  renameCollection,
  reorderCollection,
  type Collection,
  type CollectionDetail,
} from '../ipc';
import { auditionProps } from '../audition';
import { useSceneStore } from '../store/scene';
import { useShellStore } from '../store/shell';
import { formatMs } from './format';
import { EmptyState, LoadingState, PanelButton } from './StateViews';

function useSelectionIds(): number[] {
  const selectedIds = useSceneStore((s) => s.selectedIds);
  const selectedSampleId = useSceneStore((s) => s.selectedSampleId);
  if (selectedIds.size > 0) return [...selectedIds];
  return selectedSampleId !== null ? [selectedSampleId] : [];
}

export function Collections({ onBrowse }: { onBrowse?: () => void }) {
  const [collections, setCollections] = useState<Collection[] | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);
  const [detail, setDetail] = useState<CollectionDetail | null>(null);
  const [newName, setNewName] = useState('');
  const [renameValue, setRenameValue] = useState('');
  const [renamingId, setRenamingId] = useState<number | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selection = useSelectionIds();
  const select = useSceneStore((s) => s.select);
  const setDraft = useShellStore((s) => s.setDraft);
  const activeCollectionIds = useShellStore((s) => s.filterDraft.collectionIds);

  async function refreshList() {
    setCollections(await listCollections());
  }

  useEffect(() => {
    let cancelled = false;
    void listCollections()
      .then((next) => {
        if (!cancelled) setCollections(next);
      })
      .catch(() => {
        if (!cancelled) setCollections([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (openId === null) return;
    let cancelled = false;
    void getCollection(openId)
      .then((next) => {
        if (!cancelled) setDetail(next);
      })
      .catch(() => {
        if (!cancelled) setDetail(null);
      });
    return () => {
      cancelled = true;
    };
  }, [openId]);

  async function create() {
    const name = newName.trim();
    if (name === '') return;
    setError(null);
    try {
      const next = await createCollection(name, selection);
      setNewName('');
      setOpenId(next.id);
      setDetail(null);
      await refreshList();
    } catch {
      setError('Couldn’t create that collection. Try again.');
    }
  }

  async function addSelection(collection: Collection) {
    if (selection.length === 0) return;
    setError(null);
    try {
      const next = await addToCollection(collection.id, selection);
      if (openId === collection.id) setDetail(next);
      await refreshList();
    } catch {
      setError('Couldn’t add the selection. Try again.');
    }
  }

  async function removeMember(sampleId: number) {
    if (openId === null) return;
    setError(null);
    try {
      setDetail(await removeFromCollection(openId, sampleId));
      await refreshList();
    } catch {
      setError('Couldn’t remove that sound. Try again.');
    }
  }

  async function rename(collection: Collection) {
    const name = renameValue.trim();
    if (name === '') return;
    setError(null);
    try {
      const next = await renameCollection(collection.id, name);
      setCollections(
        (current) =>
          current?.map((item) => (item.id === collection.id ? next : item)) ?? null,
      );
      if (detail?.id === collection.id) setDetail({ ...detail, name: next.name });
      setRenamingId(null);
    } catch {
      setError('Couldn’t rename that collection. Try again.');
    }
  }

  async function removeCollection(collection: Collection) {
    setError(null);
    try {
      await deleteCollection(collection.id);
      setCollections(
        (current) => current?.filter((item) => item.id !== collection.id) ?? null,
      );
      if (openId === collection.id) {
        setOpenId(null);
        setDetail(null);
      }
      if (activeCollectionIds.includes(collection.id)) {
        setDraft({
          collectionIds: activeCollectionIds.filter((id) => id !== collection.id),
        });
      }
      setConfirmDeleteId(null);
    } catch {
      setError('Couldn’t delete that collection. Try again.');
    }
  }

  async function move(from: number, to: number) {
    if (!detail || from === to) return;
    const ids = detail.members.map((member) => member.sampleId);
    const [moved] = ids.splice(from, 1);
    if (moved === undefined) return;
    ids.splice(to, 0, moved);
    setError(null);
    try {
      setDetail(await reorderCollection(detail.id, ids));
    } catch {
      setError('Couldn’t reorder this collection. Try again.');
    }
  }

  async function doExport(collection: Collection) {
    const path = await saveDialog({ defaultPath: `${collection.name}.m3u8` });
    if (typeof path !== 'string') return;
    setError(null);
    try {
      await exportCollection(collection.id, path);
    } catch {
      setError('Couldn’t export that collection. Try again.');
    }
  }

  function browse(collection: Collection) {
    setDraft({ text: '', tags: [], collectionIds: [collection.id] });
    onBrowse?.();
  }

  if (collections === null) return <LoadingState label="Loading collections…" />;

  return (
    <section className="space-y-2">
      <header className="flex items-center justify-between gap-3">
        <h2 className="text-xs font-medium tracking-wide text-neutral-300 uppercase">
          Collections
        </h2>
        {selection.length > 0 && (
          <span className="bg-accent-muted text-accent shrink-0 rounded-full px-2 py-0.5 font-mono text-[10px]">
            {selection.length} selected
          </span>
        )}
      </header>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void create();
        }}
        className="flex gap-1.5"
      >
        <label className="sr-only" htmlFor="new-collection-name">
          Collection name
        </label>
        <input
          id="new-collection-name"
          type="text"
          value={newName}
          onChange={(event) => setNewName(event.target.value)}
          placeholder="New collection"
          className="rounded-control min-w-0 flex-1 border border-neutral-800 bg-neutral-950 px-2.5 py-1.5 text-xs text-neutral-100 placeholder:text-neutral-500 focus:border-neutral-600 focus:outline-none"
        />
        <PanelButton type="submit" disabled={newName.trim() === ''}>
          <Plus size={13} aria-hidden="true" />
          <span className="sr-only">Create collection</span>
        </PanelButton>
      </form>

      {error && <p className="px-1 text-[11px] text-red-400">{error}</p>}

      {collections.length === 0 ? (
        <EmptyState title="No collections yet" />
      ) : (
        <ul className="max-h-96 space-y-1 overflow-y-auto pr-0.5">
          {collections.map((collection) => {
            const expanded = openId === collection.id;
            const currentDetail =
              expanded && detail?.id === collection.id ? detail : null;
            const renaming = renamingId === collection.id;
            const confirmingDelete = confirmDeleteId === collection.id;
            const browsing = activeCollectionIds.includes(collection.id);
            return (
              <li
                key={collection.id}
                className="rounded-control border border-neutral-800 bg-neutral-950/45"
              >
                <div className="flex items-center gap-1 p-1">
                  <button
                    type="button"
                    onClick={() => browse(collection)}
                    className={`rounded-control flex min-w-0 flex-1 items-center gap-2 px-1.5 py-1 text-left text-xs transition-colors ${
                      browsing
                        ? 'bg-accent-muted text-accent'
                        : 'text-neutral-200 hover:bg-neutral-800 hover:text-neutral-100'
                    }`}
                    title={`View sounds in ${collection.name}`}
                  >
                    <span className="truncate">{collection.name}</span>
                    <span className="ml-auto shrink-0 font-mono text-[10px] text-neutral-500">
                      {collection.sampleCount}
                    </span>
                  </button>
                  {selection.length > 0 && (
                    <button
                      type="button"
                      onClick={() => void addSelection(collection)}
                      className="rounded-control flex h-6 w-6 shrink-0 items-center justify-center text-neutral-300 transition-colors hover:bg-neutral-800 hover:text-neutral-100"
                      aria-label={`Add selection to ${collection.name}`}
                      title={`Add ${selection.length} selected sound${selection.length === 1 ? '' : 's'}`}
                    >
                      <Plus size={13} />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      setOpenId(expanded ? null : collection.id);
                      setRenamingId(null);
                      setConfirmDeleteId(null);
                    }}
                    className={`rounded-control flex h-6 w-6 shrink-0 items-center justify-center transition-colors ${
                      expanded
                        ? 'bg-accent-muted text-accent'
                        : 'text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200'
                    }`}
                    aria-label={`${expanded ? 'Close' : 'Manage'} ${collection.name}`}
                    title={`${expanded ? 'Close' : 'Manage'} ${collection.name}`}
                    aria-expanded={expanded}
                  >
                    <CaretDown size={13} className={expanded ? 'rotate-180' : ''} />
                  </button>
                </div>

                {expanded && (
                  <div className="space-y-2 border-t border-neutral-800 p-2">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        type="button"
                        onClick={() => browse(collection)}
                        className="rounded-control text-accent flex h-6 w-6 items-center justify-center transition-colors hover:bg-neutral-800 hover:text-neutral-100"
                        aria-label={`View ${collection.name}`}
                        title={`View ${collection.name}`}
                      >
                        <Eye size={13} />
                      </button>
                      <button
                        type="button"
                        onClick={() => void doExport(collection)}
                        className="rounded-control flex h-6 w-6 items-center justify-center text-neutral-500 transition-colors hover:bg-neutral-800 hover:text-neutral-200"
                        aria-label={`Export ${collection.name}`}
                        title={`Export ${collection.name}`}
                      >
                        <DownloadSimple size={13} />
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setRenameValue(collection.name);
                          setRenamingId(collection.id);
                          setConfirmDeleteId(null);
                        }}
                        className={`rounded-control flex h-6 w-6 items-center justify-center transition-colors ${
                          renaming
                            ? 'bg-accent-muted text-accent'
                            : 'text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200'
                        }`}
                        aria-label={`Rename ${collection.name}`}
                        title={`Rename ${collection.name}`}
                      >
                        <PencilSimple size={13} />
                      </button>
                    </div>

                    {renaming ? (
                      <form
                        onSubmit={(event) => {
                          event.preventDefault();
                          void rename(collection);
                        }}
                        className="flex gap-1.5"
                      >
                        <label
                          className="sr-only"
                          htmlFor={`rename-collection-${collection.id}`}
                        >
                          Rename collection
                        </label>
                        <input
                          id={`rename-collection-${collection.id}`}
                          type="text"
                          value={renameValue}
                          onChange={(event) => setRenameValue(event.target.value)}
                          className="rounded-control min-w-0 flex-1 border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-100 focus:border-neutral-500 focus:outline-none"
                        />
                        <PanelButton type="submit" disabled={renameValue.trim() === ''}>
                          Save
                        </PanelButton>
                        <PanelButton onClick={() => setRenamingId(null)}>
                          Cancel
                        </PanelButton>
                      </form>
                    ) : null}

                    {currentDetail === null ? (
                      <p className="px-1 py-2 font-mono text-[11px] text-neutral-500">
                        Loading sounds…
                      </p>
                    ) : currentDetail.members.length === 0 ? (
                      <p className="px-1 py-2 text-[11px] text-neutral-500">Empty</p>
                    ) : (
                      <ol className="space-y-0.5">
                        {currentDetail.members.map((member, index) => (
                          <li
                            key={member.sampleId}
                            onDragOver={(event) => event.preventDefault()}
                            onDrop={(event) => {
                              event.preventDefault();
                              const raw = event.dataTransfer.getData(
                                'application/x-audiocloud-collection-index',
                              );
                              const from = Number(raw);
                              if (Number.isInteger(from)) void move(from, index);
                            }}
                            {...auditionProps(member.sampleId)}
                            className="group rounded-control flex min-w-0 items-center text-neutral-400 hover:bg-neutral-800"
                          >
                            <span
                              draggable
                              onDragStart={(event) => {
                                event.stopPropagation();
                                event.dataTransfer.effectAllowed = 'move';
                                event.dataTransfer.setData(
                                  'application/x-audiocloud-collection-index',
                                  String(index),
                                );
                              }}
                              className="flex h-7 w-5 shrink-0 cursor-grab items-center justify-center font-mono text-[10px] text-neutral-600 active:cursor-grabbing"
                              title="Drag to reorder"
                              aria-label={`Reorder ${member.filename}`}
                            >
                              {index + 1}
                            </span>
                            <button
                              type="button"
                              onClick={() => select(member.sampleId)}
                              className="min-w-0 flex-1 truncate py-1 pr-1 text-left text-xs text-neutral-300"
                              title={member.relPath}
                            >
                              {member.filename}
                              {member.durationMs !== null && (
                                <span className="ml-1.5 font-mono text-[10px] text-neutral-600">
                                  {formatMs(member.durationMs)}
                                </span>
                              )}
                            </button>
                            <button
                              type="button"
                              onClick={() => void removeMember(member.sampleId)}
                              className="rounded-control mr-1 flex h-5 w-5 shrink-0 items-center justify-center text-neutral-600 opacity-0 transition-all group-hover:opacity-100 hover:bg-neutral-700 hover:text-neutral-100 focus:opacity-100"
                              aria-label={`Remove ${member.filename} from ${collection.name}`}
                              title="Remove from collection"
                            >
                              <X size={11} />
                            </button>
                          </li>
                        ))}
                      </ol>
                    )}

                    {confirmingDelete ? (
                      <div className="flex items-center justify-between gap-2 border-t border-neutral-800 pt-2">
                        <p className="min-w-0 text-[10px] text-red-300">
                          Delete this collection? Sounds stay in your library.
                        </p>
                        <div className="flex shrink-0 gap-1">
                          <PanelButton onClick={() => setConfirmDeleteId(null)}>
                            Cancel
                          </PanelButton>
                          <PanelButton
                            variant="danger"
                            onClick={() => void removeCollection(collection)}
                          >
                            Delete
                          </PanelButton>
                        </div>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => {
                          setConfirmDeleteId(collection.id);
                          setRenamingId(null);
                        }}
                        className="flex items-center gap-1 text-[10px] text-neutral-500 transition-colors hover:text-red-300"
                      >
                        <Trash size={12} /> Delete collection
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
