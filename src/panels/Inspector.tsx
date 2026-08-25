/**
 * The sample detail panel: metadata, waveform, DSP features, tags, nearest neighbors,
 * reveal-in-Finder, and the audition transport (`task.md` Phase 9).
 *
 * Also owns arrow-key neighbor stepping — `useShortcuts.ts` handles the shortcuts that need
 * no panel-specific data (space, `/`, escape), but "step to the next neighbor" needs the
 * neighbor list this panel already fetches to render, so it lives here rather than in a
 * second fetch of the same thing.
 */

import { useEffect, useRef, useState } from 'react';

import {
  IpcError,
  fetchPeaks,
  getSampleDetail,
  getSimilar,
  playSample,
  revealInFinder,
  setTag,
  stopPlayback,
  unsetTag,
  type AppError,
  type Neighbor,
  type Peaks,
  type SampleDetail,
} from '../ipc';
import { useSceneStore } from '../store/scene';
import { EmptyState, ErrorState, LoadingState } from './StateViews';

const NEIGHBOR_COUNT = 12;

export function Inspector() {
  const selectedSampleId = useSceneStore((s) => s.selectedSampleId);
  const select = useSceneStore((s) => s.select);

  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [error, setError] = useState<AppError | null>(null);
  const [peaks, setPeaks] = useState<Peaks | null>(null);
  const [neighbors, setNeighbors] = useState<Neighbor[]>([]);

  useEffect(() => {
    // A stale previous sample's detail/peaks/neighbors are left in state rather than
    // cleared here — nothing renders them while `selectedSampleId` is `null`, since the
    // component's own early return below covers that case, and reaching for `setState`
    // just to clear values nobody will read is what `react-hooks/set-state-in-effect`
    // flags this shape for.
    if (selectedSampleId === null) return;
    let cancelled = false;
    const controller = new AbortController();

    getSampleDetail(selectedSampleId)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
        setError(null);
        return fetchPeaks(d, controller.signal).then((p) => {
          if (!cancelled) setPeaks(p);
        });
      })
      .catch((raw) => {
        if (!cancelled) setError(toError(raw));
      });

    getSimilar(selectedSampleId, NEIGHBOR_COUNT)
      .then((list) => {
        if (!cancelled) setNeighbors(list);
      })
      .catch(() => {
        if (!cancelled) setNeighbors([]);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedSampleId]);

  // Arrow-key neighbor stepping — left/right move through `neighbors`, wrapping at the ends.
  useEffect(() => {
    if (selectedSampleId === null || neighbors.length === 0) return;
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const ids = neighbors.map((n) => n.sampleId);
      const current = ids.indexOf(selectedSampleId!);
      const delta = e.key === 'ArrowRight' ? 1 : -1;
      const next = current === -1 ? 0 : (current + delta + ids.length) % ids.length;
      const nextId = ids[next];
      if (nextId !== undefined) select(nextId);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedSampleId, neighbors, select]);

  if (selectedSampleId === null) {
    return <EmptyState title="Nothing selected" detail="Click a point to inspect it." />;
  }
  if (error) return <ErrorState error={error} />;
  if (!detail) return <LoadingState label="Loading sample…" />;

  return (
    <div className="space-y-3 text-xs">
      <header>
        <h2
          className="truncate text-sm font-medium text-neutral-100"
          title={detail.filename}
        >
          {detail.filename}
        </h2>
        <p
          className="truncate font-mono text-[11px] text-neutral-500"
          title={detail.relPath}
        >
          {detail.relPath}
        </p>
      </header>

      <Waveform peaks={peaks} />

      <Transport sampleId={detail.id} />

      <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-neutral-400">
        <Field label="Format" value={detail.ext.toUpperCase()} />
        <Field label="Duration" value={formatMs(detail.durationMs)} />
        <Field
          label="Sample rate"
          value={detail.sampleRate ? `${detail.sampleRate} Hz` : '—'}
        />
        <Field label="Channels" value={detail.channels?.toString() ?? '—'} />
        {detail.features?.bpm != null && (
          <Field label="BPM" value={detail.features.bpm.toFixed(1)} />
        )}
        {detail.features?.lufsIntegrated != null && (
          <Field
            label="Loudness"
            value={`${detail.features.lufsIntegrated.toFixed(1)} LUFS`}
          />
        )}
      </dl>

      {detail.status === 'decode_failed' && detail.error && (
        <p className="text-[11px] text-red-400">Decode failed: {detail.error}</p>
      )}

      <TagEditor
        sampleId={detail.id}
        tags={detail.tags}
        onChange={(tags) => setDetail({ ...detail, tags })}
      />

      <button
        type="button"
        onClick={() => void revealInFinder(detail.id)}
        className="text-[11px] text-neutral-500 underline hover:text-neutral-300"
      >
        Reveal in Finder
      </button>

      {neighbors.length > 0 && (
        <div className="space-y-1">
          <p className="text-[11px] tracking-wide text-neutral-500 uppercase">
            Similar (← →)
          </p>
          <ul className="space-y-0.5">
            {neighbors.map((n) => (
              <li key={n.sampleId}>
                <button
                  type="button"
                  onClick={() => select(n.sampleId)}
                  className="w-full truncate rounded px-1 py-0.5 text-left text-neutral-400 hover:bg-neutral-900"
                  title={n.relPath}
                >
                  {n.filename}
                  <span className="ml-1.5 font-mono text-[10px] text-neutral-600">
                    {(n.similarity * 100).toFixed(0)}%
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] text-neutral-600 uppercase">{label}</dt>
      <dd className="text-neutral-300">{value}</dd>
    </div>
  );
}

function Transport({ sampleId }: { sampleId: number }) {
  const [playing, setPlaying] = useState(false);
  // "Adjust state when a prop changes," during render rather than in an effect — the pattern
  // React's own docs recommend this for: https://react.dev/learn/you-might-not-need-an-effect.
  // A new selection is a fresh transport; whatever the previous sample was doing does not
  // carry over.
  const [renderedFor, setRenderedFor] = useState(sampleId);
  if (sampleId !== renderedFor) {
    setRenderedFor(sampleId);
    setPlaying(false);
  }

  async function toggle() {
    if (playing) {
      await stopPlayback();
      setPlaying(false);
    } else {
      await playSample(sampleId, 1.0);
      setPlaying(true);
    }
  }

  useEffect(() => {
    return () => {
      void stopPlayback();
    };
  }, [sampleId]);

  return (
    <button
      type="button"
      onClick={() => void toggle()}
      className="w-full rounded border border-neutral-700 py-1 text-neutral-200 hover:bg-neutral-800"
    >
      {playing ? '■ Stop' : '▶ Play (space)'}
    </button>
  );
}

function TagEditor({
  sampleId,
  tags,
  onChange,
}: {
  sampleId: number;
  tags: string[];
  onChange: (tags: string[]) => void;
}) {
  const [value, setValue] = useState('');

  async function add() {
    const name = value.trim();
    if (name === '') return;
    onChange(await setTag(sampleId, name));
    setValue('');
  }

  async function remove(name: string) {
    onChange(await unsetTag(sampleId, name));
  }

  return (
    <div className="space-y-1">
      <div className="flex flex-wrap gap-1">
        {tags.map((name) => (
          <span
            key={name}
            className="flex items-center gap-1 rounded-full border border-neutral-700 px-2 py-0.5 text-[11px] text-neutral-300"
          >
            {name}
            <button
              type="button"
              onClick={() => void remove(name)}
              aria-label={`Remove ${name}`}
              className="text-neutral-500 hover:text-neutral-200"
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void add();
        }}
      >
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Add a tag…"
          className="w-full rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 placeholder:text-neutral-600 focus:border-neutral-600 focus:outline-none"
        />
      </form>
    </div>
  );
}

function Waveform({ peaks }: { peaks: Peaks | null }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    if (!peaks || peaks.buckets === 0) return;

    ctx.fillStyle = '#525a68';
    const mid = height / 2;
    const w = width / peaks.buckets;
    for (let i = 0; i < peaks.buckets; i++) {
      const min = peaks.minMax[i * 2] ?? 0;
      const max = peaks.minMax[i * 2 + 1] ?? 0;
      const y1 = mid - max * mid;
      const y2 = mid - min * mid;
      ctx.fillRect(i * w, y1, Math.max(1, w - 0.5), Math.max(1, y2 - y1));
    }
  }, [peaks]);

  return (
    <div className="h-16 w-full overflow-hidden rounded border border-neutral-800 bg-neutral-950">
      {peaks ? (
        <canvas ref={ref} width={480} height={64} className="h-full w-full" />
      ) : (
        <div className="flex h-full items-center justify-center text-[11px] text-neutral-600">
          No waveform
        </div>
      )}
    </div>
  );
}

function formatMs(ms: number | null): string {
  if (ms === null) return '—';
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function toError(raw: unknown): AppError {
  return raw instanceof IpcError ? raw.error : { kind: 'internal', detail: String(raw) };
}
