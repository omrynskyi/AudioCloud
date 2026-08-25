/**
 * The application shell (`task.md` Phase 9): canvas center, library/filters/search/tags/
 * collections stacked in a left rail, inspector collapsible on the right.
 *
 * The point-cloud/feature-column/filter fetch effects below are `App.tsx`'s original ones,
 * moved here unchanged — `overview.md` §5.6's boundary (the scene does no IPC, the store
 * holds no point data) is still exactly what it was in Phase 7; this phase only adds the
 * chrome around it.
 */

import { useEffect, useState } from 'react';

import {
  IpcError,
  describe,
  getFeatureColumn,
  getPointCloud,
  querySamples,
  type AppError,
  type Feature,
  type ModelStatus,
  type QueryFilter,
} from '../ipc';
import { SceneCanvas } from '../scene/SceneCanvas';
import { useLibraryStore } from '../store/library';
import { useProjectionStore } from '../store/projection';
import { useSceneStore } from '../store/scene';
import { useShellStore, type ViewMode } from '../store/shell';
import { Collections } from './Collections';
import { Filters } from './Filters';
import { Inspector } from './Inspector';
import { Library } from './Library';
import { Search, SearchResults } from './Search';
import { Settings } from './Settings';
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

  const [load, setLoad] = useState<Load>({ status: 'loading' });
  const [column, setColumn] = useState<{ feature: Feature; values: Float32Array } | null>(
    null,
  );
  const [matched, setMatched] = useState<{
    filter: Partial<QueryFilter>;
    ids: Uint32Array;
  } | null>(null);

  const colorBy = useSceneStore((state) => state.colorBy);
  const filter = useSceneStore((state) => state.filter);
  const inspectorCollapsed = useShellStore((s) => s.inspectorCollapsed);
  const toggleInspector = useShellStore((s) => s.toggleInspector);
  const openSettings = useShellStore((s) => s.openSettings);
  const viewMode = useShellStore((s) => s.viewMode);
  const setViewMode = useShellStore((s) => s.setViewMode);
  const modelStatus = useLibraryStore((s) => s.modelStatus);
  const refreshModelStatus = useLibraryStore((s) => s.refreshModelStatus);
  const lastRefitOutcome = useProjectionStore((s) => s.lastRefitOutcome);

  useEffect(() => {
    void refreshModelStatus();
  }, [refreshModelStatus]);

  const featureValues = colorBy && column?.feature === colorBy ? column.values : null;
  const matchedIds = filter && matched?.filter === filter ? matched.ids : null;

  useEffect(() => {
    let cancelled = false;
    // Refetches on mount, again every time a re-fit finishes — `lastRefitOutcome` is a fresh
    // object each time `store/projection.ts` sets it, so this effect re-runs exactly when
    // there is a reason to believe the active projection changed — and again on a switcher
    // toggle. The previous mode's cloud stays on screen until the new one resolves rather
    // than flashing a loading state in between; fetching the cloud is a single indexed
    // statement plus a memcpy (`overview.md` §6.3), so the gap is not worth the flash.
    getPointCloud(viewMode === '2d' ? 2 : 3)
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
  }, [lastRefitOutcome, viewMode]);

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
        if (!cancelled) setMatched({ filter, ids });
      })
      .catch(() => {
        if (!cancelled) setMatched(null);
      });
    return () => {
      cancelled = true;
    };
  }, [filter]);

  return (
    <div className="grid h-full w-full grid-cols-[280px_1fr_320px] grid-rows-[auto_1fr]">
      {/* `data-tauri-drag-region` is what makes this draggable at all: `titleBarStyle:
          "Overlay"` (tauri.conf.json) removes the native title bar, so nothing responds to a
          window drag unless something explicitly opts in. `core:window:allow-start-dragging`
          in capabilities/main.json is the permission side of this, granted back in Phase 6
          for exactly this header and unused until now. `pl-20` clears the macOS traffic-light
          buttons, which float over the top-left corner independent of this header's own
          layout and would otherwise sit on top of the "AudioBank" label. The settings button
          stays clickable — the drag region only intercepts an actual press-and-drag, not a
          click, and doesn't need its own opt-out. */}
      <header
        data-tauri-drag-region
        className="col-span-3 flex items-center justify-between border-b border-neutral-900 py-1.5 pr-3 pl-20"
      >
        <span data-tauri-drag-region className="font-mono text-[11px] text-neutral-600">
          AudioBank
        </span>
        <div className="flex items-center gap-3">
          <ViewModeSwitch mode={viewMode} onChange={setViewMode} />
          <button
            type="button"
            onClick={openSettings}
            className="text-neutral-500 hover:text-neutral-200"
            aria-label="Settings"
          >
            ⚙
          </button>
        </div>
      </header>

      <aside className="space-y-4 overflow-y-auto border-r border-neutral-900 p-3">
        <Search />
        <SearchResults ids={matchedIds} />
        <Library />
        <Filters />
        <Tags />
        <Collections />
      </aside>

      <main className="relative min-w-0">
        {load.status === 'loading' && <Centered>Loading the map…</Centered>}
        {load.status === 'empty' && (
          <EmptyMap
            viewMode={viewMode}
            modelStatus={modelStatus}
            openSettings={openSettings}
          />
        )}
        {load.status === 'failed' && <Centered>{describe(load.error)}</Centered>}
        {load.status === 'ready' && (
          <>
            <SceneCanvas
              mode={viewMode}
              source={load.source}
              featureValues={featureValues}
              matchedIds={matchedIds}
            />
            <Readout count={load.count} />
          </>
        )}
      </main>

      <aside
        className={
          inspectorCollapsed
            ? 'overflow-hidden border-l border-neutral-900 p-1'
            : 'overflow-y-auto border-l border-neutral-900 p-3'
        }
      >
        <button
          type="button"
          onClick={toggleInspector}
          className="mb-2 text-[11px] text-neutral-500 hover:text-neutral-300"
        >
          {inspectorCollapsed ? '‹' : 'Hide ›'}
        </button>
        {!inspectorCollapsed && <Inspector />}
      </aside>

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
 * "No map yet" has three genuinely different causes, and a re-fit that fails with
 * `tooFewSamples` because nothing is embedded is a confusing thing to hit blind — the model
 * commonly is not installed yet, since `scan_library` runs happily without one and only
 * decodes and analyzes (`overview.md` §6.1's own doc comment on that command). Naming the
 * actual blocker here is cheaper than making the user find it via a failed re-fit.
 */
function EmptyMap({
  viewMode,
  modelStatus,
  openSettings,
}: {
  viewMode: ViewMode;
  modelStatus: ModelStatus | null;
  openSettings: () => void;
}) {
  const label = viewMode === '2d' ? '2D' : '3D';
  if (!modelStatus || modelStatus.state === 'installed') {
    return (
      <Centered>
        <p className="max-w-md">
          No {label} map yet. Build it from Settings once your library has embedded
          samples.
        </p>
        <PanelButton onClick={openSettings}>Open Settings</PanelButton>
      </Centered>
    );
  }
  return (
    <Centered>
      <p className="max-w-md">
        No {label} map yet — the audio model isn&rsquo;t installed, so nothing scanned so
        far has been embedded. Download it from Settings, then rescan your library and
        build the map.
      </p>
      <PanelButton onClick={openSettings}>Open Settings</PanelButton>
    </Centered>
  );
}

/** The header's "3D / 2D" segmented control. */
function ViewModeSwitch({
  mode,
  onChange,
}: {
  mode: ViewMode;
  onChange: (mode: ViewMode) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Map view"
      className="flex overflow-hidden rounded border border-neutral-800 text-[11px]"
    >
      {(['3d', '2d'] as const).map((option) => (
        <button
          key={option}
          type="button"
          role="radio"
          aria-checked={mode === option}
          onClick={() => onChange(option)}
          className={
            mode === option
              ? 'bg-neutral-200 px-2 py-0.5 font-medium text-neutral-900'
              : 'px-2 py-0.5 text-neutral-500 hover:text-neutral-200'
          }
        >
          {option.toUpperCase()}
        </button>
      ))}
    </div>
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
