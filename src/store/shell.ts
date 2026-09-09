/**
 * Shell chrome - which overlay is open - plus the debounced filter draft `panels/Search.tsx`
 * and `panels/Tags.tsx` write into.
 *
 * The draft is split from `store/scene.ts`'s `filter` on purpose: `scene.ts`'s `filter` is
 * "what the map is currently filtered by," read by `Shell.tsx`'s fetch effect on every
 * change, and pushing a fresh `QueryFilter` on every keystroke of a search box would refetch
 * on every keystroke. This store holds the fields as the user is still typing/adjusting them
 * and debounces before calling `useSceneStore.getState().setFilter`.
 *
 * **The draft is two fields now, not five.** It used to mirror most of `QueryFilter` -
 * numeric feature ranges, file extensions, library roots - to back a panel of min/max boxes
 * that has since been removed. The backend still accepts all of it (`NO_FILTER` fills in the
 * rest on the way out); the *interface* only offers the two constraints anyone reaches for
 * while browsing by ear, which is a name and a tag. Carrying draft state for controls that no
 * longer exist is how a store quietly becomes fiction.
 */

import { create } from 'zustand';

import { NO_FILTER, type QueryFilter } from '../ipc';
import { useSceneStore } from './scene';

const DEBOUNCE_MS = 250;

export interface FilterDraft {
  text: string;
  tags: string[];
}

const EMPTY_DRAFT: FilterDraft = { text: '', tags: [] };

function isEmptyDraft(draft: FilterDraft): boolean {
  return draft.text.trim() === '' && draft.tags.length === 0;
}

function toQueryFilter(draft: FilterDraft): Partial<QueryFilter> {
  const text = draft.text.trim();
  return {
    ...NO_FILTER,
    ...(text === '' ? {} : { text }),
    tags: draft.tags,
  };
}

let debounceHandle: ReturnType<typeof setTimeout> | null = null;

interface ShellState {
  settingsOpen: boolean;
  filterDraft: FilterDraft;

  // Arrow-typed rather than method shorthand — see `store/scene.ts`'s note on
  // `@typescript-eslint/unbound-method`.
  openSettings: () => void;
  closeSettings: () => void;

  setDraft: (patch: Partial<FilterDraft>) => void;
  clearDraft: () => void;
}

export const useShellStore = create<ShellState>()((set, get) => ({
  settingsOpen: false,
  filterDraft: EMPTY_DRAFT,

  openSettings: () => set({ settingsOpen: true }),
  closeSettings: () => set({ settingsOpen: false }),

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
