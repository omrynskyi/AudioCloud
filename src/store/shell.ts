/**
 * Shell chrome — which overlay is open, whether the inspector is collapsed — plus the
 * debounced filter draft `panels/Search.tsx` and `panels/Filters.tsx` both write into.
 *
 * The draft is split from `store/scene.ts`'s `filter` on purpose: `scene.ts`'s `filter` is
 * "what the map is currently filtered by," read by `App.tsx`'s fetch effect on every change,
 * and pushing a fresh `QueryFilter` on every keystroke of a search box would refetch on every
 * keystroke. This store holds the fields as the user is still typing/adjusting them and
 * debounces before calling `useSceneStore.getState().setFilter`.
 */

import { create } from 'zustand';

import { NO_FILTER, type FeatureRange, type QueryFilter } from '../ipc';
import { useSceneStore } from './scene';

const DEBOUNCE_MS = 250;

export interface FilterDraft {
  text: string;
  rootIds: number[];
  tags: string[];
  exts: string[];
  features: FeatureRange[];
}

const EMPTY_DRAFT: FilterDraft = {
  text: '',
  rootIds: [],
  tags: [],
  exts: [],
  features: [],
};

function isEmptyDraft(draft: FilterDraft): boolean {
  return (
    draft.text.trim() === '' &&
    draft.rootIds.length === 0 &&
    draft.tags.length === 0 &&
    draft.exts.length === 0 &&
    draft.features.length === 0
  );
}

function toQueryFilter(draft: FilterDraft): Partial<QueryFilter> {
  const text = draft.text.trim();
  return {
    ...NO_FILTER,
    ...(text === '' ? {} : { text }),
    rootIds: draft.rootIds,
    tags: draft.tags,
    exts: draft.exts,
    features: draft.features,
  };
}

let debounceHandle: ReturnType<typeof setTimeout> | null = null;

export type ViewMode = '2d' | '3d';

interface ShellState {
  settingsOpen: boolean;
  inspectorCollapsed: boolean;
  filterDraft: FilterDraft;
  /**
   * Which independently-active layout the map shows — the header switcher's setting.
   *
   * Global UI chrome, not per-point scene data, which is why it lives here rather than in
   * `store/scene.ts`: that store is reserved for map coloring/filter/selection intent (see
   * its own docstring), and this is closer kin to `settingsOpen`/`inspectorCollapsed`.
   */
  viewMode: ViewMode;

  // Arrow-typed rather than method shorthand — see `store/scene.ts`'s note on
  // `@typescript-eslint/unbound-method`.
  openSettings: () => void;
  closeSettings: () => void;
  toggleInspector: () => void;
  setViewMode: (mode: ViewMode) => void;

  setDraft: (patch: Partial<FilterDraft>) => void;
  clearDraft: () => void;
}

export const useShellStore = create<ShellState>()((set, get) => ({
  settingsOpen: false,
  inspectorCollapsed: false,
  filterDraft: EMPTY_DRAFT,
  viewMode: '3d',

  openSettings: () => set({ settingsOpen: true }),
  closeSettings: () => set({ settingsOpen: false }),
  toggleInspector: () =>
    set((state) => ({ inspectorCollapsed: !state.inspectorCollapsed })),
  setViewMode: (viewMode) => set({ viewMode }),

  setDraft: (patch) => {
    const draft = { ...get().filterDraft, ...patch };
    set({ filterDraft: draft });

    if (debounceHandle) clearTimeout(debounceHandle);
    debounceHandle = setTimeout(() => {
      debounceHandle = null;
      useSceneStore
        .getState()
        .setFilter(isEmptyDraft(draft) ? null : toQueryFilter(draft));
    }, DEBOUNCE_MS);
  },

  clearDraft: () => {
    if (debounceHandle) {
      clearTimeout(debounceHandle);
      debounceHandle = null;
    }
    set({ filterDraft: EMPTY_DRAFT });
    useSceneStore.getState().setFilter(null);
  },
}));
