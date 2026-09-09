/**
 * Library roots and scanning — the state `panels/Library.tsx`, `panels/FirstRun.tsx`
 * and `panels/Settings.tsx` all need a piece of.
 *
 * A plain `create` store, not `subscribeWithSelector`: nothing here is written at picker
 * frequency the way `store/scene.ts`'s hover is, so there is no imperative-subscription half
 * to support. Progress events are still throttled to a frame — `ipc/channels.ts`'s
 * `throttleToFrame` — because a scan reports at up to 10 Hz and this store's whole point is
 * to be read through a normal `useLibraryStore` hook.
 */

import { create } from 'zustand';

import {
  addLibraryRoot,
  cancelScan as ipcCancelScan,
  listLibraryRoots,
  removeLibraryRoot,
  scanLibrary,
  setRootEnabled as ipcSetRootEnabled,
  type LibraryRoot,
  type ScanEvent,
} from '../ipc';
import { finishedError, isFinished, throttleToFrame } from '../ipc/channels';
import type { ScanOutcome } from '../bindings/ScanOutcome';
import type { ScanProgress } from '../bindings/ScanProgress';

interface ActiveScan {
  rootId: number;
  scanId: number | null;
  progress: ScanProgress | null;
}

interface LibraryState {
  roots: LibraryRoot[];
  rootsLoaded: boolean;

  activeScan: ActiveScan | null;
  /** The most recent scan's terminal event, kept until the next scan starts. */
  lastScanOutcome: ScanOutcome | null;


  // Arrow-typed rather than method shorthand — see `store/scene.ts`'s note on
  // `@typescript-eslint/unbound-method`; the same selector pattern is used throughout the
  // panels that read this store.
  refreshRoots: () => Promise<void>;
  addRoot: (path: string, label?: string) => Promise<void>;
  removeRoot: (rootId: number) => Promise<void>;
  setRootEnabled: (rootId: number, enabled: boolean) => Promise<void>;

  startScan: (rootId: number) => void;
  cancelScan: () => Promise<void>;

}

export const useLibraryStore = create<LibraryState>()((set, get) => ({
  roots: [],
  rootsLoaded: false,
  activeScan: null,
  lastScanOutcome: null,

  refreshRoots: async () => {
    const roots = await listLibraryRoots();
    set({ roots, rootsLoaded: true });
  },

  addRoot: async (path, label) => {
    await addLibraryRoot(path, label);
    await get().refreshRoots();
  },

  removeRoot: async (rootId) => {
    await removeLibraryRoot(rootId);
    await get().refreshRoots();
  },

  setRootEnabled: async (rootId, enabled) => {
    await ipcSetRootEnabled(rootId, enabled);
    await get().refreshRoots();
  },

  startScan: (rootId) => {
    if (get().activeScan) return;
    set({ activeScan: { rootId, scanId: null, progress: null }, lastScanOutcome: null });

    const onEvent = throttleToFrame<ScanEvent>((event) => {
      if (isFinished(event)) {
        set({ activeScan: null, lastScanOutcome: event });
        void get().refreshRoots();
        return;
      }
      set((state) =>
        state.activeScan
          ? { activeScan: { ...state.activeScan, scanId: event.scanId, progress: event } }
          : state,
      );
    });

    scanLibrary(rootId, onEvent)
      .then((scanId) => {
        set((state) =>
          state.activeScan ? { activeScan: { ...state.activeScan, scanId } } : state,
        );
      })
      .catch(() => {
        // A rejection here means the scan never started at all (e.g. `scanInProgress`);
        // the terminal `finished` event is the channel's job to report anything that
        // happened after admission, so this just clears the optimistic `activeScan`.
        onEvent.dispose();
        set({ activeScan: null });
      });
  },

  cancelScan: async () => {
    const scan = get().activeScan;
    if (scan?.scanId != null) await ipcCancelScan(scan.scanId);
  },

}));

/** Whether the most recent scan ended with an error the frontend should surface. */
export function scanFailed(outcome: ScanOutcome | null): boolean {
  return outcome !== null && outcome.status === 'failed';
}

export { finishedError };
