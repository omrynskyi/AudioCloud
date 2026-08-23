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

export interface SceneState {
  /** The sample the inspector is showing, or none. */
  selectedSampleId: number | null;
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

  select(sampleId: number | null): void;
  hover(sampleId: number | null): void;
  setColorBy(feature: Feature | null): void;
  setFilter(filter: Partial<QueryFilter> | null): void;
  setPointSizeMultiplier(multiplier: number): void;
}

export const useSceneStore = create<SceneState>()(
  // `subscribeWithSelector` is what makes the imperative half of §5.6 possible: without it
  // `store.subscribe` fires on every change to any field, so the scene would recompute its
  // colour array because the filter panel opened.
  subscribeWithSelector((set) => ({
    selectedSampleId: null,
    hoveredSampleId: null,
    colorBy: null,
    filter: null,
    pointSizeMultiplier: 1,

    select: (sampleId) => set({ selectedSampleId: sampleId }),
    hover: (sampleId) =>
      // Guarded because the picker calls this on every gated pointer move, and most of
      // those land on the same point as the last one. Without the guard every subscriber
      // to `hoveredSampleId` runs twenty times a second while the cursor sits still over
      // one sample.
      set((state) =>
        state.hoveredSampleId === sampleId ? state : { hoveredSampleId: sampleId },
      ),
    setColorBy: (colorBy) => set({ colorBy }),
    setFilter: (filter) => set({ filter }),
    setPointSizeMultiplier: (pointSizeMultiplier) => set({ pointSizeMultiplier }),
  })),
);
