/**
 * Hover-to-audition, for every surface in the app that can name a sample.
 *
 * The policy used to live inside `scene/PointCloud.tsx`, which was fine while the map was the
 * only thing you could hover a sample on. It is not fine once the left rail's search results,
 * a collection's members and the inspector's neighbor list all name files too: hovering a
 * filename should sound the same wherever the filename is, and a rule implemented once in a
 * scene component is a rule three other panels cannot reuse without copying it.
 *
 * So the map is no longer special. Every surface reports what the cursor is over by writing
 * `store/scene.ts`'s `hoveredSampleId` -- the map from its GPU pick, a list row from
 * [`auditionProps`] -- and [`useAudition`], mounted once by the shell, is the only thing that
 * turns that field into sound. Three things fall out of that for free rather than being
 * features anyone had to build: hovering a search result highlights its point on the map and
 * pulls up its neighbors, because those effects already watch the same field; moving the
 * cursor from a point to a list row retriggers instead of stopping and restarting, because
 * the transition is a single store write; and the audition survives an empty or still-loading
 * map, which the old placement could not since `PointCloud` was not mounted.
 *
 * **Top-level, not under `scene/` or `panels/`.** It is used by both and belongs to neither.
 */

import { useEffect } from 'react';

import { playSample, stopPlayback } from './ipc';
import { useSceneStore } from './store/scene';

/** A flat default until a settings panel exposes a gain control. */
export const PREVIEW_GAIN = 0.85;

/**
 * How long a hover over nothing waits before it silences what is playing.
 *
 * Zero would be the obvious answer and it is the wrong one. Scrubbing across a cluster means
 * crossing the gaps between points, and each gap is a hover of `null` a few tens of
 * milliseconds long -- so an immediate stop chops every sweep into fragments separated by
 * silence, which is the opposite of the continuous scrub the map is for. The same applies to
 * the space between two rows of a list, and to the moment the cursor crosses from the canvas
 * into the rail. A grace period longer than a gap and shorter than an intent to stop listening
 * lets one clip ring out until either the next sample retriggers the engine or the cursor has
 * genuinely settled on nothing.
 */
export const HOVER_STOP_GRACE_MS = 140;

/**
 * Subscribes the audio engine to `hoveredSampleId` for as long as the calling component is
 * mounted. Call this exactly once, from the shell.
 *
 * Every hover plays immediately, same as a click -- retriggering is always safe (see
 * `play_sample`'s doc comment on the Rust side), so there is no local "is something already
 * playing" state to keep in sync with the engine's. The engine is single-voice and retriggers
 * through a fresh envelope, so a fast sweep sounds like a rapid staccato of each sample's
 * attack, cut short by the next -- scrubbing, not a chord. What bounds how many of these a
 * sweep can fire is the source: the map's `HoverGate`, or how fast rows can pass under a
 * cursor. Stopping, unlike starting, is deferred -- see [`HOVER_STOP_GRACE_MS`].
 */
export function useAudition(): void {
  useEffect(() => {
    let pendingStop: ReturnType<typeof setTimeout> | null = null;
    const cancelStop = () => {
      if (pendingStop !== null) {
        clearTimeout(pendingStop);
        pendingStop = null;
      }
    };

    const unsubscribe = useSceneStore.subscribe(
      (state) => state.hoveredSampleId,
      (sampleId) => {
        cancelStop();
        if (sampleId === null) {
          pendingStop = setTimeout(() => {
            pendingStop = null;
            stopPlayback().catch(() => {});
          }, HOVER_STOP_GRACE_MS);
          return;
        }
        playSample(sampleId, PREVIEW_GAIN).catch(() => {});
      },
    );

    return () => {
      cancelStop();
      unsubscribe();
      // Mounted for the life of the shell, so this only runs on teardown -- but leaving a clip
      // playing into a torn-down UI is a worse default than one redundant stop.
      stopPlayback().catch(() => {});
    };
  }, []);
}

/**
 * Pointer handlers that make any element naming `sampleId` an audition target. Spread onto a
 * list row: `<button {...auditionProps(row.id)}>`.
 *
 * The leave handler clears the hover only if this element is still the one the store thinks is
 * hovered. Pointer events on siblings are supposed to arrive leave-then-enter, but nested
 * targets and synthetic events do not always oblige, and an out-of-order leave that cleared
 * unconditionally would cancel the row the cursor has already moved on to -- the sound would
 * stop a beat after arriving on a new row, intermittently and only sometimes, which is the
 * worst kind of bug to be handed.
 */
export function auditionProps(sampleId: number): {
  onPointerEnter: () => void;
  onPointerLeave: () => void;
} {
  return {
    onPointerEnter: () => useSceneStore.getState().hover(sampleId),
    onPointerLeave: () => {
      const state = useSceneStore.getState();
      if (state.hoveredSampleId === sampleId) state.hover(null);
    },
  };
}
