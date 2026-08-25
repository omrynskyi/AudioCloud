/**
 * Structured filters: duration, BPM, key, loudness, spectral centroid, tags, format
 * (`task.md` Phase 9). Every numeric control writes a `FeatureRange` into the shared draft in
 * `store/shell.ts`; tags and extensions write their own array fields directly.
 */

import { useEffect, useState } from 'react';

import { listTags, type FeatureRange, type Tag } from '../ipc';
import { useShellStore } from '../store/shell';

const KEY_NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];

function numberField(
  value: number | undefined,
  onChange: (n: number | undefined) => void,
  placeholder: string,
) {
  return (
    <input
      type="number"
      value={value ?? ''}
      onChange={(e) =>
        onChange(e.target.value === '' ? undefined : Number(e.target.value))
      }
      placeholder={placeholder}
      className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none"
    />
  );
}

function RangeRow({
  label,
  feature,
  features,
  onSet,
  scale = 1,
}: {
  label: string;
  feature: FeatureRange['feature'];
  features: FeatureRange[];
  onSet: (feature: FeatureRange['feature'], min?: number, max?: number) => void;
  /** Divides the stored value for display — duration is stored in ms, shown in seconds. */
  scale?: number;
}) {
  const current = features.find((f) => f.feature === feature);
  const min = current?.min !== undefined ? current.min / scale : undefined;
  const max = current?.max !== undefined ? current.max / scale : undefined;

  return (
    <div className="space-y-1">
      <p className="text-[11px] text-neutral-500">{label}</p>
      <div className="flex items-center gap-1.5">
        {numberField(
          min,
          (n) => onSet(feature, n === undefined ? undefined : n * scale, current?.max),
          'min',
        )}
        <span className="text-neutral-700">–</span>
        {numberField(
          max,
          (n) => onSet(feature, current?.min, n === undefined ? undefined : n * scale),
          'max',
        )}
      </div>
    </div>
  );
}

export function Filters() {
  const draft = useShellStore((s) => s.filterDraft);
  const setDraft = useShellStore((s) => s.setDraft);
  const [tags, setTags] = useState<Tag[] | null>(null);
  const [extsText, setExtsText] = useState(draft.exts.join(', '));

  useEffect(() => {
    listTags()
      .then(setTags)
      .catch(() => setTags([]));
  }, []);

  function setFeatureRange(feature: FeatureRange['feature'], min?: number, max?: number) {
    const rest = draft.features.filter((f) => f.feature !== feature);
    if (min === undefined && max === undefined) {
      setDraft({ features: rest });
      return;
    }
    const range: FeatureRange = {
      feature,
      ...(min === undefined ? {} : { min }),
      ...(max === undefined ? {} : { max }),
    };
    setDraft({ features: [...rest, range] });
  }

  function toggleTag(name: string) {
    const has = draft.tags.includes(name);
    setDraft({
      tags: has ? draft.tags.filter((t) => t !== name) : [...draft.tags, name],
    });
  }

  function commitExts() {
    const exts = extsText
      .split(',')
      .map((e) => e.trim().toLowerCase().replace(/^\./, ''))
      .filter(Boolean);
    setDraft({ exts });
  }

  const keyRootRange = draft.features.find((f) => f.feature === 'keyRoot');
  const currentKey =
    keyRootRange?.min === keyRootRange?.max ? keyRootRange?.min : undefined;

  return (
    <section className="space-y-3">
      <header className="flex items-center justify-between">
        <h2 className="text-xs font-medium tracking-wide text-neutral-400 uppercase">
          Filters
        </h2>
        {(draft.features.length > 0 ||
          draft.tags.length > 0 ||
          draft.exts.length > 0) && (
          <button
            type="button"
            onClick={() => {
              setExtsText('');
              setDraft({ features: [], tags: [], exts: [] });
            }}
            className="text-[11px] text-neutral-500 underline hover:text-neutral-300"
          >
            Clear
          </button>
        )}
      </header>

      <RangeRow
        label="Duration (s)"
        feature="durationMs"
        features={draft.features}
        onSet={setFeatureRange}
        scale={1000}
      />
      <RangeRow
        label="BPM"
        feature="bpm"
        features={draft.features}
        onSet={setFeatureRange}
      />
      <RangeRow
        label="Loudness (LUFS)"
        feature="lufsIntegrated"
        features={draft.features}
        onSet={setFeatureRange}
      />
      <RangeRow
        label="Spectral centroid (Hz)"
        feature="spectralCentroid"
        features={draft.features}
        onSet={setFeatureRange}
      />

      <div className="space-y-1">
        <p className="text-[11px] text-neutral-500">Key</p>
        <select
          value={currentKey ?? ''}
          onChange={(e) => {
            const v = e.target.value === '' ? undefined : Number(e.target.value);
            setFeatureRange('keyRoot', v, v);
          }}
          className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 focus:border-neutral-600 focus:outline-none"
        >
          <option value="">Any</option>
          {KEY_NAMES.map((name, i) => (
            <option key={name} value={i}>
              {name}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-1">
        <p className="text-[11px] text-neutral-500">Format</p>
        <input
          type="text"
          value={extsText}
          onChange={(e) => setExtsText(e.target.value)}
          onBlur={commitExts}
          placeholder="wav, mp3, aiff…"
          className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none"
        />
      </div>

      {tags && tags.length > 0 && (
        <div className="space-y-1">
          <p className="text-[11px] text-neutral-500">Tags</p>
          <div className="flex flex-wrap gap-1">
            {tags.map((tag) => {
              const active = draft.tags.includes(tag.name);
              return (
                <button
                  key={tag.id}
                  type="button"
                  onClick={() => toggleTag(tag.name)}
                  className={`rounded-full border px-2 py-0.5 text-[11px] ${
                    active
                      ? 'border-neutral-400 bg-neutral-700 text-neutral-100'
                      : 'border-neutral-800 text-neutral-400 hover:border-neutral-600'
                  }`}
                >
                  {tag.name}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
