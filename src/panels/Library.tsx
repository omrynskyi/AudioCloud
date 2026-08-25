/** Library roots: add / remove / rescan / enable-disable, with scan progress (Phase 9). */

import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { useEffect, useState } from 'react';

import { useLibraryStore } from '../store/library';
import { EmptyState, LoadingState, PanelButton } from './StateViews';

export function Library() {
  const roots = useLibraryStore((s) => s.roots);
  const rootsLoaded = useLibraryStore((s) => s.rootsLoaded);
  const refreshRoots = useLibraryStore((s) => s.refreshRoots);
  const addRoot = useLibraryStore((s) => s.addRoot);
  const removeRoot = useLibraryStore((s) => s.removeRoot);
  const setRootEnabled = useLibraryStore((s) => s.setRootEnabled);
  const activeScan = useLibraryStore((s) => s.activeScan);
  const lastScanOutcome = useLibraryStore((s) => s.lastScanOutcome);
  const startScan = useLibraryStore((s) => s.startScan);
  const cancelScan = useLibraryStore((s) => s.cancelScan);

  const [removing, setRemoving] = useState<number | null>(null);

  useEffect(() => {
    if (!rootsLoaded) void refreshRoots();
  }, [rootsLoaded, refreshRoots]);

  async function pickFolder() {
    const picked = await openDialog({ directory: true, multiple: false });
    if (typeof picked === 'string') await addRoot(picked);
  }

  async function confirmRemove(rootId: number) {
    await removeRoot(rootId);
    setRemoving(null);
  }

  if (!rootsLoaded) return <LoadingState label="Loading library roots…" />;

  return (
    <section className="space-y-2">
      <header className="flex items-center justify-between">
        <h2 className="text-xs font-medium tracking-wide text-neutral-400 uppercase">
          Library
        </h2>
        <PanelButton onClick={() => void pickFolder()}>+ Add</PanelButton>
      </header>

      {roots.length === 0 ? (
        <EmptyState title="No folders added" detail="Add one to start scanning." />
      ) : (
        <ul className="space-y-1">
          {roots.map((root) => {
            const scanning = activeScan?.rootId === root.id;
            return (
              <li
                key={root.id}
                className="rounded border border-neutral-800 px-2 py-1.5 text-xs"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-neutral-200" title={root.path}>
                      {root.label ?? root.path.split('/').pop()}
                    </p>
                    <p className="font-mono text-[11px] text-neutral-500">
                      {root.sampleCount.toLocaleString()} samples
                      {!root.enabled && ' · disabled'}
                    </p>
                  </div>
                  <div className="flex shrink-0 gap-1">
                    <PanelButton
                      onClick={() => void setRootEnabled(root.id, !root.enabled)}
                    >
                      {root.enabled ? 'Disable' : 'Enable'}
                    </PanelButton>
                    <PanelButton disabled={scanning} onClick={() => startScan(root.id)}>
                      Rescan
                    </PanelButton>
                    {removing === root.id ? (
                      <PanelButton
                        variant="danger"
                        onClick={() => void confirmRemove(root.id)}
                      >
                        Confirm
                      </PanelButton>
                    ) : (
                      <PanelButton variant="danger" onClick={() => setRemoving(root.id)}>
                        Remove
                      </PanelButton>
                    )}
                  </div>
                </div>

                {scanning && activeScan?.progress && (
                  <div className="mt-1.5 flex items-center justify-between gap-2">
                    <p className="font-mono text-[11px] text-neutral-500">
                      {activeScan.progress.filesDone.toLocaleString()} /{' '}
                      {activeScan.progress.filesSeen.toLocaleString()}
                    </p>
                    <button
                      type="button"
                      onClick={() => void cancelScan()}
                      className="text-[11px] text-neutral-500 underline hover:text-neutral-300"
                    >
                      Cancel
                    </button>
                  </div>
                )}
                {!scanning &&
                  lastScanOutcome?.rootId === root.id &&
                  lastScanOutcome.status === 'failed' && (
                    <p className="mt-1 text-[11px] text-red-400">Scan failed.</p>
                  )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
