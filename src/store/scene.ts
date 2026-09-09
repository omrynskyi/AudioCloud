/**
 * UI intent for the map (`overview.md` §5.6).
 *
 * The rule that section states, and that this file exists to make enforceable:
 * **Zustand describes intent; imperative buffer mutation executes it.** Changing the
 * colour-by mode writes one string here. A subscriber *outside* React reads it, fetches the
 * column, recomputes 150,000 floats, and sets `needsUpdate`. One component re-renders — the
 * control that was clicked — and the scene does not re-render at all.
 *
 * So there is no point data in this store and there must never be: no coordinates, no
 * colours, no per-point anything. Cross-cutting rule 4 in the roadmap says the same thing
 * from the other side. A 50,000-element array behind a `useStore` selector re-renders a
 * component tree on every hover, and that is the single most reliable way to make this
 * application feel broken.
 *
 * Hover and selection are here, and are the closest call in the file. They are scalars —
 * one id each, not one value per point — and the inspector, the library list and the
 * transport all need them, so a store is the right home. What keeps hover honest is that
 * the scene never reads it through a hook: it subscribes imperatively, below, and writes a
 * uniform.
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

import type { Feature } from '../bindings/Feature';
import type { QueryFilter } from '../bindings/QueryFilter';

/**
 * The scrub history deliberately stays small: it is a recall aid for the current map session,
 * not a permanent log. The range control can reveal any prefix of these most recent samples.
 */
export const MAX_SCRUB_HISTORY = 100;
export const MIN_SCRUB_HISTORY_RANGE = 10;

export interface SceneState {
  /** The sample the inspector is showing, or none. */
  selectedSampleId: number | null;
  /**
   * Every sample the user has multi-selected, for bulk tagging and "create a collection
   * from this selection." Still an id set, not point data — the same justification the
   * module docstring gives for hover and `selectedSampleId` living here rather than in a
   * panel-local store: a handful of ids, not one value per point.
   *
   * `selectedSampleId` and this are independent: the inspector always targets the single
   * id, and a multi-select does not have to include it (or vice versa). Panels that act on
   * "the current selection" fall back to `selectedSampleId` when this is empty.
   */
  selectedIds: Set<number>;
  /**
   * The sample under the cursor.
   *
   * Written at up to 20 Hz by the picker. Subscribe to it narrowly — a component that
   * re-renders on this re-renders twenty times a second during an orbit.
   */
  hoveredSampleId: number | null;
  /** Which feature colours the map, or `null` for the flat default. */
  colorBy: Feature | null;
  /** The active filter. `null` means the whole library, which is not the same as `{}`. */
  filter: Partial<QueryFilter> | null;
  /** A global multiplier on point size, for the size slider. */
  pointSizeMultiplier: number;
  /**
   * Most-recent-first sample ids heard while moving across the map.
   *
   * This is intentionally separate from `hoveredSampleId`: list rows and the inspector use
   * hover to audition too, but they are navigation surfaces, not a map scrub.
   */
  scrubHistory: number[];
  /** How many of the retained scrubbed samples the history panel shows. */
  scrubHistoryRange: number;

  // Typed as arrow-function properties rather than method shorthand (`select(id): void`)
  // deliberately: a method-typed member is exactly what `@typescript-eslint/unbound-method`
  // flags the moment a component does `useSceneStore((s) => s.select)` and calls the result
  // later — a completely ordinary selector pattern, and these never use `this` anyway.
  select: (sampleId: number | null) => void;
  hover: (sampleId: number | null) => void;
  /** Updates map hover and remembers a newly encountered sample in the scrub history. */
  scrub: (sampleId: number | null) => void;
  setColorBy: (feature: Feature | null) => void;
  setFilter: (filter: Partial<QueryFilter> | null) => void;
  setPointSizeMultiplier: (multiplier: number) => void;
  setScrubHistoryRange: (range: number) => void;
  clearScrubHistory: () => void;

  /** Replaces the multi-selection outright — a fresh drag-select or click with no modifier. */
  setSelectedIds: (ids: Iterable<number>) => void;
  /** Adds or removes one id from the multi-selection — a shift/cmd-click. */
  toggleSelected: (sampleId: number) => void;
  clearSelectedIds: () => void;
}

export const useSceneStore = create<SceneState>()(
  // `subscribeWithSelector` is what makes the imperative half of §5.6 possible: without it
  // `store.subscribe` fires on every change to any field, so the scene would recompute its
  // colour array because the filter panel opened.
  subscribeWithSelector((set) => ({
    selectedSampleId: null,
    selectedIds: new Set<number>(),
    hoveredSampleId: null,
    colorBy: null,
    filter: null,
    pointSizeMultiplier: 1,
    scrubHistory: [],
    scrubHistoryRange: MAX_SCRUB_HISTORY,

    select: (sampleId) => set({ selectedSampleId: sampleId }),
    hover: (sampleId) =>
      // Guarded because the picker calls this on every gated pointer move, and most of
      // those land on the same point as the last one. Without the guard every subscriber
      // to `hoveredSampleId` runs twenty times a second while the cursor sits still over
      // one sample.
      set((state) =>
        state.hoveredSampleId === sampleId ? state : { hoveredSampleId: sampleId },
      ),
    scrub: (sampleId) =>
      set((state) => {
        // The picker continues running while a pointer rests on one point. Collapsing that
        // repeat here keeps both the audition subscription and this list to one entry per
        // actual transition, rather than turning a pause into twenty history rows per second.
        if (state.hoveredSampleId === sampleId) return state;
        if (sampleId === null) return { hoveredSampleId: null };

        // Preserve the route through the cloud: returning to a sample after crossing another
        // is useful context in a listening trail. The transition guard above is what prevents
        // a stationary pointer from filling that trail with repeated entries.
        const scrubHistory = [sampleId, ...state.scrubHistory].slice(
          0,
          MAX_SCRUB_HISTORY,
        );
        return { hoveredSampleId: sampleId, scrubHistory };
      }),
    setColorBy: (colorBy) => set({ colorBy }),
    setFilter: (filter) => set({ filter }),
    setPointSizeMultiplier: (pointSizeMultiplier) => set({ pointSizeMultiplier }),
    setScrubHistoryRange: (range) =>
      set({
        scrubHistoryRange: Math.min(
          MAX_SCRUB_HISTORY,
          Math.max(MIN_SCRUB_HISTORY_RANGE, Math.round(range)),
        ),
      }),
    clearScrubHistory: () => set({ scrubHistory: [] }),

    setSelectedIds: (ids) => set({ selectedIds: new Set(ids) }),
    toggleSelected: (sampleId) =>
      set((state) => {
        const next = new Set(state.selectedIds);
        if (next.has(sampleId)) next.delete(sampleId);
        else next.add(sampleId);
        return { selectedIds: next };
      }),
    clearSelectedIds: () => set({ selectedIds: new Set() }),
  })),
);
