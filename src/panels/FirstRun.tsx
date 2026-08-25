/**
 * Welcome → add a library root → download the model (skippable) → scan (`task.md` Phase 9).
 *
 * A scan without a model still decodes and runs DSP analysis — `scan_library`'s own doc
 * comment says the next scan with a working session finishes what this one starts — so
 * "Skip for now" is a real, useful path through this wizard, not a dead end.
 */

import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { useEffect, useState } from 'react';

import { IpcError, type AppError } from '../ipc';
import { useLibraryStore } from '../store/library';
import { EmptyState, ErrorState, LoadingState, PanelButton } from './StateViews';

type Step = 'welcome' | 'addRoot' | 'model' | 'scan';

export function FirstRun({ onComplete }: { onComplete: () => void }) {
  const [step, setStep] = useState<Step>('welcome');
  const [rootId, setRootId] = useState<number | null>(null);
  const [error, setError] = useState<AppError | null>(null);

  const roots = useLibraryStore((s) => s.roots);
  const addRoot = useLibraryStore((s) => s.addRoot);
  const modelStatus = useLibraryStore((s) => s.modelStatus);
  const refreshModelStatus = useLibraryStore((s) => s.refreshModelStatus);
  const activeDownload = useLibraryStore((s) => s.activeDownload);
  const lastDownloadOutcome = useLibraryStore((s) => s.lastDownloadOutcome);
  const startDownload = useLibraryStore((s) => s.startDownload);
  const activeScan = useLibraryStore((s) => s.activeScan);
  const lastScanOutcome = useLibraryStore((s) => s.lastScanOutcome);
  const startScan = useLibraryStore((s) => s.startScan);

  useEffect(() => {
    if (step === 'model') void refreshModelStatus();
  }, [step, refreshModelStatus]);

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
      setStep('model');
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

  if (step === 'model') {
    if (!modelStatus) return <LoadingState label="Checking the audio model…" />;

    if (modelStatus.state === 'installed') {
      return (
        <Centered>
          <p className="text-sm text-neutral-400">
            The audio model is already installed.
          </p>
          <PanelButton onClick={() => setStep('scan')}>Continue</PanelButton>
        </Centered>
      );
    }

    if (activeDownload) {
      const p = activeDownload.progress;
      const pct = p?.total ? Math.round((p.downloaded / p.total) * 100) : null;
      return (
        <Centered>
          <h1 className="text-lg font-medium text-neutral-100">Downloading the model…</h1>
          <p className="font-mono text-xs text-neutral-500">
            {p
              ? `${(p.downloaded / 1e6).toFixed(1)} MB${pct !== null ? ` (${pct}%)` : ''}`
              : '…'}
          </p>
        </Centered>
      );
    }

    if (modelStatus.state === 'unpinned') {
      return (
        <Centered>
          <p className="max-w-sm text-sm text-neutral-400">
            This build has no published model checksum, so nothing can be downloaded yet.
          </p>
          <PanelButton onClick={() => setStep('scan')}>Continue without it</PanelButton>
        </Centered>
      );
    }

    return (
      <Centered>
        <h1 className="text-lg font-medium text-neutral-100">Download the audio model</h1>
        <p className="max-w-sm text-sm text-neutral-400">
          Needed to place samples on the map by how they sound. You can skip this and scan
          first — the next download finishes the job.
        </p>
        {lastDownloadOutcome?.error && <ErrorState error={lastDownloadOutcome.error} />}
        <div className="flex gap-2">
          <PanelButton onClick={() => startDownload()}>Download</PanelButton>
          <PanelButton onClick={() => setStep('scan')}>Skip for now</PanelButton>
        </div>
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
  return (
    <main className="flex h-full w-full flex-col items-center justify-center gap-3 px-6 text-center">
      {children}
    </main>
  );
}
