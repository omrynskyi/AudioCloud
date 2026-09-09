/**
 * The application shell: canvas center, a compact top app bar, and contextual popovers.
 *
 * The point-cloud/feature-column/filter fetch effects below are `App.tsx`'s original ones,
 * moved here unchanged — `overview.md` §5.6's boundary (the scene does no IPC, the store
 * holds no point data) is still exactly what it was in Phase 7; this phase only adds the
 * chrome around it.
 */

import {
  ClockCounterClockwise,
  FolderOpen,
  Gear,
  Stack,
  Tag,
} from '@phosphor-icons/react';
import { useEffect, useState } from 'react';

import {
  IpcError,
  describe,
  getFeatureColumn,
  getPointCloud,
  getPointColors,
  querySamples,
  type AppError,
  type Feature,
  type PointColors,
  type QueryFilter,
} from '../ipc';
import { useAudition } from '../audition';
import { SceneCanvas } from '../scene/SceneCanvas';
import { useProjectionStore } from '../store/projection';
import { useSceneStore } from '../store/scene';
import { useShellStore } from '../store/shell';
import { Collections } from './Collections';
import { IconButton, Popover } from './Chrome';
import { ColorBy } from './ColorBy';
import { HoverHud } from './HoverHud';
import { Inspector } from './Inspector';
import { Library } from './Library';
import { Search } from './Search';
import { Settings } from './Settings';
import { ScrubHistory } from './ScrubHistory';
import { PanelButton } from './StateViews';
import { Tags } from './Tags';
import { useShortcuts } from './useShortcuts';

type Load =
  | { status: 'loading' }
  | { status: 'ready'; source: ArrayBuffer; count: number }
  /** Scanned but not projected, or not scanned at all. A real state, not a failure. */
  | { status: 'empty' }
  | { status: 'failed'; error: AppError };

export function Shell() {
  useShortcuts();
  // Mounted here, not in the scene: every surface that can name a sample -- the map, the
  // search results, a collection's members, the inspector's neighbors -- auditions through the
  // one subscription this installs. See `audition.ts`.
  useAudition();

  const [load, setLoad] = useState<Load>({ status: 'loading' });
  const [fitColors, setFitColors] = useState<PointColors | null>(null);
  const [column, setColumn] = useState<{ feature: Feature; values: Float32Array } | null>(
    null,
  );
  // `error` rather than dropping the result on the floor: a failed query used to become
  // `null`, which renders as nothing at all -- indistinguishable from "still typing" and from
  // "no matches". Carrying the failure lets `SearchResults` say so.
  const [matched, setMatched] = useState<{
    filter: Partial<QueryFilter>;
    ids: Uint32Array | null;
    error: AppError | null;
  } | null>(null);

  const colorBy = useSceneStore((state) => state.colorBy);
  const filter = useSceneStore((state) => state.filter);
  const openSettings = useShellStore((s) => s.openSettings);
  const lastRefitOutcome = useProjectionStore((s) => s.lastRefitOutcome);
  const selectedSampleId = useSceneStore((state) => state.selectedSampleId);

  const featureValues = colorBy && column?.feature === colorBy ? column.values : null;
  const current = filter && matched?.filter === filter ? matched : null;
  const matchedIds = current?.ids ?? null;

  useEffect(() => {
    let cancelled = false;
    // Refetches on mount, and again every time a re-fit finishes — `lastRefitOutcome` is a
    // fresh object each time `store/projection.ts` sets it, so this effect re-runs exactly
    // when there is a reason to believe the active projection changed. Without this
    // dependency, a re-fit that ran (even successfully) after the first fetch would leave
    // this screen showing whatever it showed before the re-fit was ever started.
    getPointCloud()
      .then(({ cloud, buffer }) => {
        if (cancelled) return;
        setLoad(
          cloud.count === 0
            ? { status: 'empty' }
            : { status: 'ready', source: buffer, count: cloud.count },
        );
      })
      .catch((raw: unknown) => {
        if (!cancelled) setLoad({ status: 'failed', error: toError(raw) });
      });
    return () => {
      cancelled = true;
    };
  }, [lastRefitOutcome]);

  useEffect(() => {
    let cancelled = false;
    // Same trigger as the point cloud fetch above, and for the same reason: a re-fit is the
    // only thing that can change which run is active, and therefore the only thing that can
    // change its fit colors. A failure here just means no fit-colored default this load --
    // `PointCloud` already falls back to the flat default when `fitColors` is `null`, so
    // there is nothing to surface as an error.
    getPointColors()
      .then((colors) => {
        if (!cancelled) setFitColors(colors);
      })
      .catch(() => {
        if (!cancelled) setFitColors(null);
      });
    return () => {
      cancelled = true;
    };
  }, [lastRefitOutcome]);

  const count = load.status === 'ready' ? load.count : 0;

  useEffect(() => {
    if (!colorBy || count === 0) return;
    let cancelled = false;
    getFeatureColumn(colorBy, count)
      .then((values) => {
        if (!cancelled) setColumn({ feature: colorBy, values });
      })
      .catch(() => {
        if (!cancelled) setColumn(null);
      });
    return () => {
      cancelled = true;
    };
  }, [colorBy, count]);

  useEffect(() => {
    if (!filter) return;
    let cancelled = false;
    querySamples(filter)
      .then((ids) => {
        if (!cancelled) setMatched({ filter, ids, error: null });
      })
      .catch((raw: unknown) => {
        if (!cancelled) setMatched({ filter, ids: null, error: toError(raw) });
      });
    return () => {
      cancelled = true;
    };
  }, [filter]);

  return (
    <div className="bg-canvas relative h-full w-full overflow-hidden">
      <main className="absolute inset-0 min-w-0">
        {load.status === 'loading' && <Centered>Loading the map…</Centered>}
        {load.status === 'empty' && <EmptyMap openSettings={openSettings} />}
        {load.status === 'failed' && <Centered>{describe(load.error)}</Centered>}
        {load.status === 'ready' && (
          <>
            <SceneCanvas
              source={load.source}
              featureValues={featureValues}
              matchedIds={matchedIds}
              fitColors={fitColors}
            />
            <Readout count={load.count} />
          </>
        )}
      </main>

      {/* Keep the native window controls in their own title bar. The tools remain a floating
          object over the map, so they are available without turning the canvas into chrome. */}
      <header
        data-tauri-drag-region
        className="title-bar absolute top-0 right-0 left-0 z-10 h-9"
      />

      <div className="glass rounded-panel absolute top-11 left-4 z-10 flex items-start gap-1 p-1">
        <Search ids={matchedIds} error={current?.error ?? null} />
        <div className="flex items-start gap-1">
          <ColorBy />
          <Popover
            label="Scrub history"
            icon={<ClockCounterClockwise />}
            align="right"
            width="w-80"
          >
            <ScrubHistory />
          </Popover>
          <Popover
            label="Library folders"
            icon={<FolderOpen />}
            align="right"
            width="w-80"
          >
            <Library />
          </Popover>
          <Popover label="Tags" icon={<Tag />} align="right" width="w-80">
            <Tags />
          </Popover>
          <Popover label="Collections" icon={<Stack />} align="right" width="w-80">
            <Collections />
          </Popover>
          <IconButton label="Settings" onClick={openSettings}>
            <Gear />
          </IconButton>
        </div>
      </div>

      {selectedSampleId !== null && (
        <aside className="glass rise-in rounded-panel absolute top-20 right-4 bottom-4 z-10 w-80 overflow-y-auto p-3">
          <Inspector />
        </aside>
      )}

      <HoverHud />

      <Settings />
    </div>
  );
}

function Readout({ count }: { count: number }) {
  const hovered = useSceneStore((state) => state.hoveredSampleId);
  const selected = useSceneStore((state) => state.selectedSampleId);
  return (
    <div className="pointer-events-none absolute bottom-3 left-4 font-mono text-xs text-neutral-500">
      {count.toLocaleString()} points
      {hovered !== null && ` · hover #${hovered}`}
      {selected !== null && ` · selected #${selected}`}
    </div>
  );
}

/**
 * `scan_library` always embeds now — every sample gets a fingerprint at scan time, no model
 * download involved (`commands::library::scan_library`'s doc) — so "no map yet" has one
 * cause: nothing has been scanned and built into a map yet.
 */
function EmptyMap({ openSettings }: { openSettings: () => void }) {
  return (
    <Centered>
      <p className="max-w-md">
        No map yet. Build it from Settings once your library has embedded samples.
      </p>
      <PanelButton onClick={openSettings}>Open Settings</PanelButton>
    </Centered>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-neutral-500">
      {children}
    </div>
  );
}

function toError(raw: unknown): AppError {
  return raw instanceof IpcError ? raw.error : { kind: 'internal', detail: String(raw) };
}
