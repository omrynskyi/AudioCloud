/**
 * Settings modal: projection params + re-fit, audio device + gain, data directory, and
 * reset-database (`task.md` Phase 9).
 */

import { CaretDown, Check, WarningCircle } from '@phosphor-icons/react';
import { useEffect, useRef, useState, type KeyboardEvent } from 'react';

import {
  getSettings,
  type AppError,
  listAudioDevices,
  resetDatabase,
  revealDataDir,
  setAudioDevice,
  setGain,
  type AppSettings,
  type AudioDeviceInfo,
} from '../ipc';
import { useProjectionStore } from '../store/projection';
import { useShellStore } from '../store/shell';
import { ErrorState, LoadingState, Modal, PanelButton } from './StateViews';

export function Settings() {
  const open = useShellStore((s) => s.settingsOpen);
  const close = useShellStore((s) => s.closeSettings);
  if (!open) return null;

  return (
    <Modal
      title="Settings"
      onClose={close}
      closeLabel="Close settings"
      panelClassName="glass rise-in max-h-[86vh] w-[500px]"
    >
      <div className="settings-content">
        <p className="settings-intro">
          Tune the map and audition playback to your workspace.
        </p>
        <ProjectionSection />
        <AudioSection />
        <DataSection />
      </div>
    </Modal>
  );
}

function Section({
  title,
  detail,
  children,
}: {
  title: string;
  detail: string;
  children: React.ReactNode;
}) {
  return (
    <section className="settings-section">
      <div className="settings-section-heading">
        <div>
          <h2 className="text-[13px] font-medium text-neutral-200">{title}</h2>
          <p className="mt-0.5 text-[11px] text-neutral-500">{detail}</p>
        </div>
      </div>
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
    <Section
      title="Map projection"
      detail="Refresh the spatial arrangement of your library."
    >
      <div className="settings-card">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs font-medium text-neutral-300">Update the map</p>
            <p className="mt-1 text-[11px] leading-relaxed text-neutral-500">
              Refresh the full t-SNE layout using your current library embeddings.
            </p>
          </div>
          <PanelButton
            disabled={!!activeRefit}
            onClick={() => startRefit({ forceFull: true })}
            className="settings-button-primary shrink-0"
          >
            Refresh map
          </PanelButton>
        </div>
        {activeRefit && (
          <div className="settings-progress" role="status">
            <span className="settings-progress-dot" aria-hidden="true" />
            <span>
              {activeRefit.progress?.phase ?? 'Preparing'} ·{' '}
              {activeRefit.progress?.samples.toLocaleString() ?? '0'} samples
            </span>
            <PanelButton onClick={() => void cancelRefit()} className="ml-auto">
              Cancel
            </PanelButton>
          </div>
        )}
      </div>
      {!activeRefit && lastRefitOutcome?.error && (
        <ProjectionError error={lastRefitOutcome.error} />
      )}
    </Section>
  );
}

/**
 * A projection failure is part of the map setup flow, not a modal-sized failure screen.
 * Keep the actionable empty-library case readable and reserve the generic error treatment
 * for failures that do not have a better recovery message.
 */
function ProjectionError({ error }: { error: AppError }) {
  if (error.kind !== 'tooFewSamples') return <ErrorState error={error} />;

  const missing = Math.max(0, error.detail.need - error.detail.have);
  const sampleWord = missing === 1 ? 'sample' : 'samples';

  return (
    <div className="settings-inline-error" role="alert">
      <WarningCircle size={18} weight="fill" aria-hidden="true" />
      <div className="min-w-0">
        <p className="text-xs font-medium text-neutral-200">
          Not enough samples to build a map
        </p>
        <p className="mt-1 text-[11px] leading-relaxed text-neutral-500">
          {error.detail.have === 0
            ? `Scan a folder first, then add at least ${error.detail.need} embedded samples.`
            : `Add ${missing} more embedded ${sampleWord}, then refresh the map.`}
        </p>
      </div>
    </div>
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
    <Section
      title="Audio output"
      detail="Choose where previews play and set their listening level."
    >
      {!devices || !settings ? (
        <LoadingState label="Loading devices…" />
      ) : (
        <div className="settings-card space-y-5">
          <SelectField
            label="Output device"
            value={settings.audioDevice ?? ''}
            onChange={(value) => void pickDevice(value)}
            options={[
              {
                value: '',
                label: 'System default',
                detail: 'Use the current macOS output',
              },
              ...devices.map((d) => ({
                value: d.name,
                label: d.name,
                detail: d.isDefault ? 'System default' : undefined,
              })),
            ]}
          />
          <div>
            <div className="mb-2 flex items-baseline justify-between gap-3">
              <div>
                <p className="text-xs font-medium text-neutral-300">Master gain</p>
                <p className="mt-1 text-[11px] text-neutral-500">
                  Applied to audition playback
                </p>
              </div>
              <output className="settings-value" htmlFor="master-gain">
                {settings.gain.toFixed(2)}×
              </output>
            </div>
            <input
              id="master-gain"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={settings.gain}
              onChange={(e) => void pickGain(Number(e.target.value))}
              className="settings-range"
              style={
                {
                  '--range-progress': `${(settings.gain / 2) * 100}%`,
                } as React.CSSProperties
              }
              aria-label="Master gain"
            />
            <div className="settings-range-labels" aria-hidden="true">
              <span>0.00×</span>
              <span>1.00×</span>
              <span>2.00×</span>
            </div>
          </div>
        </div>
      )}
    </Section>
  );
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
      <Section
        title="Library data"
        detail="Manage the files AudioCloud keeps on this Mac."
      >
        <LoadingState label="Resetting… AudioCloud will restart." />
      </Section>
    );
  }

  return (
    <Section title="Library data" detail="Manage the files AudioCloud keeps on this Mac.">
      <div className="settings-card space-y-3">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs font-medium text-neutral-300">Data folder</p>
            <p className="mt-1 text-[11px] text-neutral-500">
              Open the local library and index.
            </p>
          </div>
          <PanelButton
            onClick={() => void revealDataDir()}
            className="settings-button-secondary"
          >
            Reveal folder
          </PanelButton>
        </div>
        {!confirming ? (
          <button
            type="button"
            onClick={() => setConfirming(true)}
            className="settings-danger-link"
          >
            <span>Reset database</span>
            <span aria-hidden="true">↗</span>
          </button>
        ) : (
          <div className="settings-confirmation">
            <div className="flex items-start gap-2">
              <div className="settings-danger-mark" aria-hidden="true">
                !
              </div>
              <p className="leading-relaxed text-red-300">
                This permanently deletes your library and embeddings. Type AUDIOCLOUD to
                confirm.
              </p>
            </div>
            <input
              type="text"
              value={confirmText}
              onChange={(e) => setConfirmText(e.target.value)}
              placeholder="Type AUDIOCLOUD"
              aria-label="Type AUDIOCLOUD to confirm database reset"
              className="settings-text-input"
            />
            <div className="flex gap-2">
              <PanelButton
                variant="danger"
                disabled={confirmText !== 'AUDIOCLOUD'}
                onClick={() => void doReset()}
                className="settings-danger-button"
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

type SelectOption = { value: string; label: string; detail?: string | undefined };

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(() =>
    Math.max(
      0,
      options.findIndex((option) => option.value === value),
    ),
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const selected =
    options.find((option) => option.value === value) ??
    options[0] ??
    ({ value: '', label: 'No devices available' } satisfies SelectOption);
  const selectedIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );

  useEffect(() => {
    function closeOnOutside(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', closeOnOutside);
    return () => document.removeEventListener('mousedown', closeOnOutside);
  }, []);

  function choose(option: SelectOption) {
    onChange(option.value);
    setHighlighted(Math.max(0, options.indexOf(option)));
    setOpen(false);
    buttonRef.current?.focus();
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      setOpen(true);
      setHighlighted((current) =>
        event.key === 'ArrowDown'
          ? Math.min((open ? current : selectedIndex) + 1, options.length - 1)
          : Math.max((open ? current : selectedIndex) - 1, 0),
      );
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (open && options[highlighted]) choose(options[highlighted]);
      else {
        setHighlighted(selectedIndex);
        setOpen(true);
      }
    } else if (event.key === 'Escape') {
      setOpen(false);
    }
  }

  return (
    <div ref={rootRef} className="settings-field">
      <label className="settings-field-label" htmlFor="audio-device">
        {label}
      </label>
      <div className="settings-select-wrap">
        <button
          ref={buttonRef}
          id="audio-device"
          type="button"
          className="settings-select-trigger"
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen((current) => !current)}
          onKeyDown={onKeyDown}
        >
          <span className="min-w-0 truncate text-left">
            <span className="block truncate text-xs text-neutral-200">
              {selected.label}
            </span>
            {selected.detail && (
              <span className="mt-0.5 block truncate text-[10px] text-neutral-500">
                {selected.detail}
              </span>
            )}
          </span>
          <CaretDown
            size={14}
            className={`shrink-0 text-neutral-500 transition-transform ${open ? 'rotate-180' : ''}`}
          />
        </button>
        {open && (
          <div className="settings-select-menu rise-in" role="listbox" aria-label={label}>
            {options.map((option, index) => (
              <button
                key={option.value || 'system-default'}
                type="button"
                role="option"
                aria-selected={option.value === value}
                className={`settings-select-option ${index === highlighted ? 'is-highlighted' : ''}`}
                onMouseEnter={() => setHighlighted(index)}
                onClick={() => choose(option)}
              >
                <span className="min-w-0">
                  <span className="block truncate">{option.label}</span>
                  {option.detail && (
                    <span className="mt-0.5 block truncate text-[10px] text-neutral-500">
                      {option.detail}
                    </span>
                  )}
                </span>
                {option.value === value && (
                  <Check size={15} weight="bold" className="text-accent shrink-0" />
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
