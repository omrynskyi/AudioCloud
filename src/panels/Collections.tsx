/**
 * Collections: create from the current selection, list, drag-to-reorder, export, delete
 * (`task.md` Phase 9).
 */

import { save as saveDialog } from '@tauri-apps/plugin-dialog';
import { useEffect, useState } from 'react';

import {
  createCollection,
  deleteCollection,
  exportCollection,
  getCollection,
  listCollections,
  reorderCollection,
  type Collection,
  type CollectionDetail,
} from '../ipc';
import { useSceneStore } from '../store/scene';
import { EmptyState, LoadingState, PanelButton } from './StateViews';

function useSelectionIds(): number[] {
  const selectedIds = useSceneStore((s) => s.selectedIds);
  const selectedSampleId = useSceneStore((s) => s.selectedSampleId);
  if (selectedIds.size > 0) return [...selectedIds];
  return selectedSampleId !== null ? [selectedSampleId] : [];
}

export function Collections() {
  const [collections, setCollections] = useState<Collection[] | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);
  const [detail, setDetail] = useState<CollectionDetail | null>(null);
  const [newName, setNewName] = useState('');
  const selection = useSelectionIds();

  async function refreshList() {
    setCollections(await listCollections());
  }

  useEffect(() => {
    void listCollections()
      .then(setCollections)
      .catch(() => setCollections([]));
  }, []);

  useEffect(() => {
    if (openId === null) return;
    void getCollection(openId)
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [openId]);

  async function create() {
    if (newName.trim() === '' || selection.length === 0) return;
    await createCollection(newName.trim(), selection);
    setNewName('');
    await refreshList();
  }

  async function remove(id: number) {
    await deleteCollection(id);
    if (openId === id) setOpenId(null);
    await refreshList();
  }

  async function move(from: number, to: number) {
    if (!detail || from === to) return;
    const ids = detail.members.map((m) => m.sampleId);
    const [moved] = ids.splice(from, 1);
    if (moved === undefined) return;
    ids.splice(to, 0, moved);
    const next = await reorderCollection(detail.id, ids);
    setDetail(next);
  }

  async function doExport(id: number, name: string) {
    const path = await saveDialog({ defaultPath: `${name}.m3u8` });
    if (typeof path === 'string') await exportCollection(id, path);
  }

  if (collections === null) return <LoadingState label="Loading collections…" />;

  return (
    <section className="space-y-2">
      <header className="flex items-center justify-between">
        <h2 className="text-xs font-medium tracking-wide text-neutral-400 uppercase">
          Collections
        </h2>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void create();
        }}
        className="flex gap-1.5"
      >
        <input
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder={selection.length > 0 ? 'New collection…' : 'Select samples first'}
          disabled={selection.length === 0}
          className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none disabled:opacity-40"
        />
        <button
          type="submit"
          disabled={selection.length === 0 || newName.trim() === ''}
          className="rounded border border-neutral-700 px-2 text-xs text-neutral-300 hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Save
        </button>
      </form>

      {collections.length === 0 ? (
        <EmptyState title="No collections yet" detail="Select samples and save one." />
      ) : (
        <ul className="space-y-1">
          {collections.map((c) => (
            <li key={c.id} className="rounded border border-neutral-800 text-xs">
              <div className="flex items-center justify-between gap-2 px-2 py-1">
                <button
                  type="button"
                  onClick={() => setOpenId(openId === c.id ? null : c.id)}
                  className="min-w-0 flex-1 truncate text-left text-neutral-200"
                >
                  {c.name}{' '}
                  <span className="font-mono text-[11px] text-neutral-500">
                    ({c.sampleCount})
                  </span>
                </button>
                <div className="flex shrink-0 gap-1">
                  <PanelButton onClick={() => void doExport(c.id, c.name)}>
                    Export
                  </PanelButton>
                  <PanelButton variant="danger" onClick={() => void remove(c.id)}>
                    Delete
                  </PanelButton>
                </div>
              </div>

              {openId === c.id && detail && detail.id === c.id && (
                <ol className="space-y-0.5 border-t border-neutral-800 px-2 py-1.5">
                  {detail.members.map((member, i) => (
                    <li
                      key={member.sampleId}
                      draggable
                      onDragStart={(e) => e.dataTransfer.setData('text/plain', String(i))}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => {
                        e.preventDefault();
                        const from = Number(e.dataTransfer.getData('text/plain'));
                        void move(from, i);
                      }}
                      className="cursor-grab truncate rounded px-1 py-0.5 text-neutral-400 hover:bg-neutral-900"
                      title={member.relPath}
                    >
                      {i + 1}. {member.filename}
                    </li>
                  ))}
                </ol>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
