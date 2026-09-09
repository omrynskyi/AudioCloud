/**
 * Welcome → add a library root → scan.
 *
 * Scanning uses AudioCloud's built-in fingerprint embedder and needs no model download.
 */

import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { useEffect, useRef, useState } from 'react';

import { IpcError, type AppError } from '../ipc';
import { useLibraryStore } from '../store/library';
import { useProjectionStore } from '../store/projection';
import { EmptyState, ErrorState, LoadingState, PanelButton } from './StateViews';

type Step = 'welcome' | 'addRoots' | 'scan' | 'buildMap';

export function FirstRun({ onComplete }: { onComplete: () => void }) {
  const [step, setStep] = useState<Step>('welcome');
  const [rootIds, setRootIds] = useState<number[]>([]);
  const [scanIndex, setScanIndex] = useState(0);
  const [error, setError] = useState<AppError | null>(null);
  const startedScanIndex = useRef<number | null>(null);

  const roots = useLibraryStore((s) => s.roots);
  const addRoot = useLibraryStore((s) => s.addRoot);
  const activeScan = useLibraryStore((s) => s.activeScan);
  const startScan = useLibraryStore((s) => s.startScan);
  const activeRefit = useProjectionStore((s) => s.activeRefit);
  const lastRefitOutcome = useProjectionStore((s) => s.lastRefitOutcome);
  const startRefit = useProjectionStore((s) => s.startRefit);

  useEffect(() => {
    if (step !== 'scan' || activeScan || scanIndex >= rootIds.length) return;
    const rootId = rootIds[scanIndex];
    if (rootId === undefined) return;
    if (startedScanIndex.current !== scanIndex) {
      startedScanIndex.current = scanIndex;
      startScan(rootId);
    }
  }, [step, rootIds, scanIndex, activeScan, startScan]);

  useEffect(() => {
    // Scan completion is an external store event. Advance the wizard from that subscription
    // rather than synchronously setting component state in a render-driven effect; the latter
    // creates a cascading render and is exactly what React's set-state-in-effect rule rejects.
    return useLibraryStore.subscribe((state, previous) => {
      const outcome = state.lastScanOutcome;
      if (outcome === previous.lastScanOutcome || step !== 'scan') return;
      const rootId = rootIds[scanIndex];
      if (!outcome || rootId === undefined || outcome.rootId !== rootId) return;

      if (outcome.status === 'failed') {
        setError(outcome.error ?? { kind: 'internal', detail: 'The scan failed.' });
        setStep('addRoots');
        return;
      }

      if (scanIndex + 1 < rootIds.length) {
        setScanIndex((index) => index + 1);
        return;
      }

      setStep('buildMap');
      startRefit({ forceFull: true });
    });
  }, [step, rootIds, scanIndex, startRefit]);

  useEffect(() => {
    if (step === 'buildMap' && !activeRefit && lastRefitOutcome) onComplete();
  }, [step, activeRefit, lastRefitOutcome, onComplete]);

  async function pickFolder() {
    setError(null);
    const picked = await openDialog({ directory: true, multiple: true });
    const paths = Array.isArray(picked)
      ? picked
      : typeof picked === 'string'
        ? [picked]
        : [];
    if (paths.length === 0) return;

    try {
      const addedIds: number[] = [];
      for (const path of paths) {
        await addRoot(path);
        const added = useLibraryStore.getState().roots.find((root) => root.path === path);
        if (added) addedIds.push(added.id);
      }
      setRootIds((ids) => [...ids, ...addedIds.filter((id) => !ids.includes(id))]);
    } catch (raw) {
      setError(
        raw instanceof IpcError ? raw.error : { kind: 'internal', detail: String(raw) },
      );
    }
  }

  function beginImport() {
    if (rootIds.length === 0) return;
    setError(null);
    setScanIndex(0);
    startedScanIndex.current = null;
    setStep('scan');
  }

  if (step === 'welcome') {
    return (
      <Centered>
        <h1 className="text-lg font-medium text-neutral-100">Welcome to AudioCloud</h1>
        <p className="max-w-sm text-sm text-neutral-400">
          Point it at a folder of samples and it builds a map you can fly through —
          similar sounds land near each other.
        </p>
        <PanelButton onClick={() => setStep('addRoots')}>Get started</PanelButton>
      </Centered>
    );
  }

  if (step === 'addRoots') {
    return (
      <Centered>
        <h1 className="text-lg font-medium text-neutral-100">Add sample folders</h1>
        <p className="max-w-sm text-sm text-neutral-400">
          Add one or more folders. Subfolders are included automatically, and AudioCloud
          will scan everything and build your map when you’re ready.
        </p>
        {rootIds.length > 0 && (
          <ul className="first-run-roots" aria-label="Folders to scan">
            {rootIds.map((id) => {
              const root = roots.find((item) => item.id === id);
              return root ? (
                <li key={id} className="first-run-root">
                  <span className="truncate" title={root.path}>
                    {root.label ?? root.path.split('/').pop()}
                  </span>
                  <span className="font-mono text-[10px] text-neutral-500">ready</span>
                </li>
              ) : null;
            })}
          </ul>
        )}
        <div className="flex gap-2">
          <PanelButton onClick={() => void pickFolder()}>
            {rootIds.length > 0 ? 'Add more folders…' : 'Choose folders…'}
          </PanelButton>
          {rootIds.length > 0 && (
            <PanelButton onClick={beginImport} className="settings-button-primary">
              Scan and build map
            </PanelButton>
          )}
        </div>
        {error && <ErrorState error={error} />}
      </Centered>
    );
  }

  if (step === 'buildMap') {
    if (lastRefitOutcome?.error) return <ErrorState error={lastRefitOutcome.error} />;
    return (
      <Centered>
        <h1 className="text-lg font-medium text-neutral-100">Building your map…</h1>
        <p className="max-w-sm text-sm text-neutral-400">
          Placing similar sounds together. This may take a moment for a large library.
        </p>
        {activeRefit ? (
          <p className="font-mono text-xs text-neutral-500">
            {activeRefit.progress?.phase ?? 'Preparing'}
            {activeRefit.progress
              ? ` · ${activeRefit.progress.samples.toLocaleString()} samples`
              : ''}
          </p>
        ) : (
          <LoadingState label="Starting the map build…" />
        )}
      </Centered>
    );
  }

  // step === 'scan'
  if (!activeScan) return <LoadingState label="Starting the scan…" />;
  const progress = activeScan.progress;
  return (
    <Centered>
      <h1 className="text-lg font-medium text-neutral-100">
        Scanning folder {Math.min(scanIndex + 1, rootIds.length)} of {rootIds.length}…
      </h1>
      {progress ? (
        <div className="w-64 text-center">
          <p className="font-mono text-xs text-neutral-500">
            {progress.filesDone.toLocaleString()} / {progress.filesSeen.toLocaleString()}{' '}
            files
          </p>
          {progress.currentPath && (
            <p className="mt-1 truncate font-mono text-[11px] text-neutral-600">
              {progress.currentPath}
            </p>
          )}
        </div>
      ) : (
        <EmptyState title="Discovering files…" />
      )}
    </Centered>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  // Draggable window background, like `App.tsx`'s loading screen and `Shell.tsx`'s header —
  // safe even with buttons/inputs among `children`, because Tauri v2 only drags on the exact
  // element clicked, not on an ancestor's attribute (verified against the pinned Tauri
  // version; this was a real behavior change from v1 worth not assuming).
  return (
    <main
      data-tauri-drag-region
      className="flex h-full w-full flex-col items-center justify-center gap-3 px-6 text-center"
    >
      {children}
    </main>
  );
}
