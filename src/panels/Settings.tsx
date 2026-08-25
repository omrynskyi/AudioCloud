/**
 * Settings modal: projection params + re-fit, audio device + gain, model status/re-download,
 * data directory, and reset-database (`task.md` Phase 9).
 */

import { useEffect, useState } from 'react';

import {
  getSettings,
  listAudioDevices,
  resetDatabase,
  revealDataDir,
  setAudioDevice,
  setGain,
  type AppSettings,
  type AudioDeviceInfo,
} from '../ipc';
import type { DownloadProgressEvent } from '../bindings/DownloadProgressEvent';
import { useLibraryStore } from '../store/library';
import { useProjectionStore } from '../store/projection';
import { useShellStore } from '../store/shell';
import { ErrorState, LoadingState, PanelButton } from './StateViews';

export function Settings() {
  const open = useShellStore((s) => s.settingsOpen);
  const close = useShellStore((s) => s.closeSettings);
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-20 flex items-center justify-center bg-black/60"
      onClick={close}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="max-h-[80vh] w-[420px] overflow-y-auto rounded-lg border border-neutral-800 bg-neutral-950 p-4 text-xs shadow-xl"
      >
        <header className="mb-3 flex items-center justify-between">
          <h1 className="text-sm font-medium text-neutral-100">Settings</h1>
          <button
            type="button"
            onClick={close}
            aria-label="Close settings"
            className="text-neutral-500 hover:text-neutral-200"
          >
            ×
          </button>
        </header>

        <div className="space-y-5">
          <ProjectionSection />
          <AudioSection />
          <ModelSection />
          <DataSection />
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2 border-t border-neutral-900 pt-3 first:border-t-0 first:pt-0">
      <h2 className="text-[11px] tracking-wide text-neutral-500 uppercase">{title}</h2>
      {children}
    </section>
  );
}

function ProjectionSection() {
  const activeRefit = useProjectionStore((s) => s.activeRefit);
  const lastRefitOutcome = useProjectionStore((s) => s.lastRefitOutcome);
  const startRefit = useProjectionStore((s) => s.startRefit);
  const cancelRefit = useProjectionStore((s) => s.cancelRefit);

  return (
    <Section title="Map">
      <div className="flex gap-2">
        <PanelButton
          disabled={!!activeRefit}
          onClick={() => startRefit({ forceFull: false })}
        >
          Re-fit
        </PanelButton>
        <PanelButton
          disabled={!!activeRefit}
          onClick={() => startRefit({ forceFull: true })}
        >
          Full re-fit
        </PanelButton>
        {activeRefit && (
          <PanelButton onClick={() => void cancelRefit()}>Cancel</PanelButton>
        )}
      </div>
      {activeRefit?.progress && (
        <p className="font-mono text-[11px] text-neutral-500">
          {activeRefit.progress.phase} · {activeRefit.progress.samples.toLocaleString()}{' '}
          samples
        </p>
      )}
      {!activeRefit && lastRefitOutcome?.error && (
        <ErrorState error={lastRefitOutcome.error} />
      )}
    </Section>
  );
}

function AudioSection() {
  const [devices, setDevices] = useState<AudioDeviceInfo[] | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);

  useEffect(() => {
    void listAudioDevices().then(setDevices);
    void getSettings().then(setSettings);
  }, []);

  async function pickDevice(name: string) {
    const next = await setAudioDevice(name === '' ? null : name);
    setSettings(next);
  }

  async function pickGain(gain: number) {
    setSettings((s) => (s ? { ...s, gain } : s));
    await setGain(gain);
  }

  return (
    <Section title="Audio">
      {!devices || !settings ? (
        <LoadingState label="Loading devices…" />
      ) : (
        <>
          <select
            value={settings.audioDevice ?? ''}
            onChange={(e) => void pickDevice(e.target.value)}
            className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 focus:border-neutral-600 focus:outline-none"
          >
            <option value="">System default</option>
            {devices.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
                {d.isDefault ? ' (default)' : ''}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-neutral-400">
            Gain
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={settings.gain}
              onChange={(e) => void pickGain(Number(e.target.value))}
              className="flex-1"
            />
            <span className="w-8 text-right font-mono text-[11px]">
              {settings.gain.toFixed(2)}
            </span>
          </label>
        </>
      )}
    </Section>
  );
}

function ModelSection() {
  const modelStatus = useLibraryStore((s) => s.modelStatus);
  const refreshModelStatus = useLibraryStore((s) => s.refreshModelStatus);
  const activeDownload = useLibraryStore((s) => s.activeDownload);
  const lastDownloadOutcome = useLibraryStore((s) => s.lastDownloadOutcome);
  const startDownload = useLibraryStore((s) => s.startDownload);
  const cancelDownload = useLibraryStore((s) => s.cancelDownload);

  useEffect(() => {
    void refreshModelStatus();
  }, [refreshModelStatus]);

  return (
    <Section title="Model">
      {!modelStatus ? (
        <LoadingState label="Checking…" />
      ) : (
        <>
          <div className="flex items-center justify-between">
            <p className="text-neutral-400">
              {modelStatus.state === 'installed'
                ? `Installed (${modelStatus.version})`
                : modelStatus.state === 'unpinned'
                  ? 'No published checksum for this build'
                  : 'Not installed'}
            </p>
            {modelStatus.state === 'downloadable' &&
              (activeDownload ? (
                <PanelButton onClick={() => void cancelDownload()}>Cancel</PanelButton>
              ) : (
                <PanelButton onClick={() => startDownload()}>Download</PanelButton>
              ))}
          </div>

          {activeDownload && <DownloadProgress event={activeDownload.progress} />}

          {!activeDownload && lastDownloadOutcome?.error && (
            <ErrorState error={lastDownloadOutcome.error} />
          )}
        </>
      )}
    </Section>
  );
}

function DownloadProgress({ event }: { event: DownloadProgressEvent | null }) {
  if (!event) return <p className="font-mono text-[11px] text-neutral-500">Starting…</p>;

  const mb = (event.downloaded / 1e6).toFixed(1);
  if (event.total) {
    const pct = Math.min(100, Math.round((event.downloaded / event.total) * 100));
    return (
      <div className="space-y-1">
        <div className="h-1.5 overflow-hidden rounded-full bg-neutral-800">
          <div
            className="h-full rounded-full bg-neutral-400 transition-[width]"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="font-mono text-[11px] text-neutral-500">
          {mb} / {(event.total / 1e6).toFixed(1)} MB ({pct}%)
        </p>
      </div>
    );
  }

  // No `Content-Length` from the server — a live byte count rather than a bar stuck at zero,
  // per `DownloadProgressEvent`'s own doc comment on why `total` is optional.
  return <p className="font-mono text-[11px] text-neutral-500">{mb} MB downloaded…</p>;
}

function DataSection() {
  const [confirming, setConfirming] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const [resetting, setResetting] = useState(false);

  function doReset() {
    setResetting(true);
    // The process exits before a reply can arrive on success — this call is fire-and-forget
    // by design; see `ipc/commands.ts`'s `resetDatabase` doc comment.
    void resetDatabase().catch(() => setResetting(false));
  }

  if (resetting) {
    return (
      <Section title="Data">
        <LoadingState label="Resetting… AudioBank will restart." />
      </Section>
    );
  }

  return (
    <Section title="Data">
      <PanelButton onClick={() => void revealDataDir()}>Reveal data folder</PanelButton>

      <div className="pt-2">
        {!confirming ? (
          <PanelButton variant="danger" onClick={() => setConfirming(true)}>
            Reset database…
          </PanelButton>
        ) : (
          <div className="space-y-1.5 rounded border border-red-950 p-2">
            <p className="text-red-400">
              This permanently deletes your library and embeddings. Type AUDIOBANK to
              confirm.
            </p>
            <input
              type="text"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 focus:border-neutral-600 focus:outline-none"
            />
            <div className="flex gap-2">
              <PanelButton
                variant="danger"
                disabled={confirmText !== 'AUDIOBANK'}
                onClick={() => void doReset()}
              >
                Delete everything
              </PanelButton>
              <PanelButton onClick={() => setConfirming(false)}>Cancel</PanelButton>
            </div>
          </div>
        )}
      </div>
    </Section>
  );
}
