/**
 * The shell, as far as Phase 7 needs one.
 *
 * `task.md` gives the real layout to Phase 9 — canvas centre, inspector right, library and
 * filters left. What is here is the wiring that phase will inherit rather than replace: the
 * boundary between IPC and the scene.
 *
 * That boundary is the point. `overview.md` §5.6 says the scene does no IPC and the store
 * holds no point data, which leaves someone to fetch the bytes and hand them over — and it
 * is the shell. Three fetches, three different triggers:
 *
 * - the point cloud, once, when the app opens;
 * - a feature column, whenever the colour-by mode changes;
 * - a filter's matching ids, whenever the filter changes.
 *
 * Each lands as a prop on `SceneCanvas` and is consumed there by an effect that writes a
 * typed array. Nothing on this path re-renders per point, and nothing on it knows what a
 * `BufferAttribute` is.
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
  type QueryFilter,
} from './ipc';
import { SceneCanvas } from './scene/SceneCanvas';
import { useSceneStore } from './store/scene';

type Load =
  | { status: 'loading' }
  | { status: 'ready'; source: ArrayBuffer; count: number }
  /** Scanned but not projected, or not scanned at all. A real state, not a failure. */
  | { status: 'empty' }
  | { status: 'failed'; error: AppError };

export default function App() {
  const [load, setLoad] = useState<Load>({ status: 'loading' });
  // Each fetch is stored **with the intent that asked for it**, and the value handed to the
  // scene is derived during render by checking that they still agree. The obvious spelling
  // -- an effect that calls `setFeatureValues(null)` when the mode is cleared -- reads
  // better and is wrong: it sets state synchronously during an effect, so React renders
  // once with the stale column and again with null, and for one frame the map is coloured
  // by a feature nobody selected.
  const [column, setColumn] = useState<{ feature: Feature; values: Float32Array } | null>(
    null,
  );
  const [matched, setMatched] = useState<{
    filter: Partial<QueryFilter>;
    ids: Uint32Array;
  } | null>(null);

  const colorBy = useSceneStore((state) => state.colorBy);
  const filter = useSceneStore((state) => state.filter);

  const featureValues = colorBy && column?.feature === colorBy ? column.values : null;
  // Reference equality, which is exactly right here: `filter` is a store value and a new
  // object only when something actually changed it.
  const matchedIds = filter && matched?.filter === filter ? matched.ids : null;

  useEffect(() => {
    let cancelled = false;
    getPointCloud()
      .then(({ cloud, buffer }) => {
        if (cancelled) return;
        // An empty payload is the honest answer before the first re-fit, and
        // `commands/cloud.rs` is explicit that it is not an error. Rendering it as one
        // would tell a user with a freshly scanned library that something broke.
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
  }, []);

  const count = load.status === 'ready' ? load.count : 0;

  useEffect(() => {
    if (!colorBy || count === 0) return;
    let cancelled = false;
    // `count` is passed so the core can check it: a mismatch means a re-fit landed between
    // the two fetches, and colouring this cloud by that column would paint every point with
    // somebody else's number.
    getFeatureColumn(colorBy, count)
      .then((values) => {
        if (!cancelled) setColumn({ feature: colorBy, values });
      })
      .catch(() => {
        // Dropping the column leaves the map flat, which is the honest rendering of "this
        // feature could not be read" and is not worth a dialog over.
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

  if (load.status === 'loading') return <Centered>Loading the map…</Centered>;
  if (load.status === 'empty') {
    return <Centered>No map yet. Add a folder, scan it, then build the map.</Centered>;
  }
  if (load.status === 'failed') return <Centered>{describe(load.error)}</Centered>;

  return (
    <main className="relative h-full w-full">
      <SceneCanvas
        source={load.source}
        featureValues={featureValues}
        matchedIds={matchedIds}
      />
      <Readout count={load.count} />
    </main>
  );
}

/**
 * The one piece of chrome Phase 7 needs: what is under the cursor.
 *
 * Subscribed through a hook rather than imperatively, unlike the scene, and that is the
 * point of the distinction in §5.6 — this is a single line of text that *should* re-render
 * when hover changes, and re-rendering it costs one text node. The scene, which would cost
 * a traversal of the whole graph, subscribes imperatively instead.
 */
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

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex h-full w-full items-center justify-center">
      <p className="max-w-md text-center text-sm text-neutral-500">{children}</p>
    </main>
  );
}

function toError(raw: unknown): AppError {
  return raw instanceof IpcError ? raw.error : { kind: 'internal', detail: String(raw) };
}
