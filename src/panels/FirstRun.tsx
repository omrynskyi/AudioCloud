/**
 * Welcome → add a library root → scan.
 *
 * Scanning uses AudioBank's built-in fingerprint embedder and needs no model download.
 */

import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { useEffect, useState } from 'react';

import { IpcError, type AppError } from '../ipc';
import { useLibraryStore } from '../store/library';
import { EmptyState, ErrorState, LoadingState, PanelButton } from './StateViews';

type Step = 'welcome' | 'addRoot' | 'scan';

export function FirstRun({ onComplete }: { onComplete: () => void }) {
  const [step, setStep] = useState<Step>('welcome');
  const [rootId, setRootId] = useState<number | null>(null);
  const [error, setError] = useState<AppError | null>(null);

  const roots = useLibraryStore((s) => s.roots);
  const addRoot = useLibraryStore((s) => s.addRoot);
  const activeScan = useLibraryStore((s) => s.activeScan);
  const lastScanOutcome = useLibraryStore((s) => s.lastScanOutcome);
  const startScan = useLibraryStore((s) => s.startScan);

  useEffect(() => {
    if (step === 'scan' && rootId !== null && !activeScan && !lastScanOutcome) {
      startScan(rootId);
    }
  }, [step, rootId, activeScan, lastScanOutcome, startScan]);

  useEffect(() => {
    if (step === 'scan' && lastScanOutcome) onComplete();
  }, [step, lastScanOutcome, onComplete]);

  async function pickFolder() {
    setError(null);
    const picked = await openDialog({ directory: true, multiple: false });
    if (typeof picked !== 'string') return;
    try {
      await addRoot(picked);
      const added = useLibraryStore.getState().roots.find((r) => r.path === picked);
      setRootId(added?.id ?? roots[0]?.id ?? null);
      setStep('scan');
    } catch (raw) {
      setError(
        raw instanceof IpcError ? raw.error : { kind: 'internal', detail: String(raw) },
      );
    }
  }

  if (step === 'welcome') {
    return (
      <Centered>
        <h1 className="text-lg font-medium text-neutral-100">Welcome to AudioBank</h1>
        <p className="max-w-sm text-sm text-neutral-400">
          Point it at a folder of samples and it builds a map you can fly through —
          similar sounds land near each other.
        </p>
        <PanelButton onClick={() => setStep('addRoot')}>Get started</PanelButton>
      </Centered>
    );
  }

  if (step === 'addRoot') {
    return (
      <Centered>
        <h1 className="text-lg font-medium text-neutral-100">Add a folder</h1>
        <p className="max-w-sm text-sm text-neutral-400">
          Choose the folder that holds your samples. Subfolders are included
          automatically.
        </p>
        <PanelButton onClick={() => void pickFolder()}>Choose a folder…</PanelButton>
        {error && <ErrorState error={error} />}
      </Centered>
    );
  }

  // step === 'scan'
  if (!activeScan) return <LoadingState label="Starting the scan…" />;
  const progress = activeScan.progress;
  return (
    <Centered>
      <h1 className="text-lg font-medium text-neutral-100">Scanning your library…</h1>
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
