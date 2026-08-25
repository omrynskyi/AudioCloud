/**
 * Re-fit job state, kept outside `Settings.tsx` on purpose.
 *
 * The first version of this lived in local `useState` inside the Settings modal's
 * projection section, which loses track of an in-flight re-fit the moment the modal closes
 * — `Settings` unmounts its whole tree when hidden, so a user who closed and reopened it saw
 * a fresh "Re-fit" button with no memory of the job already running, clicked it again, and
 * got `refitInProgress` back with no explanation. Living in a store (the same fix
 * `store/library.ts` already applies to scans and downloads) means the state survives the
 * modal being closed, so reopening it shows the truth.
 */

import { create } from 'zustand';

import {
  cancelRefit as ipcCancelRefit,
  startRefit as ipcStartRefit,
  type RefitEvent,
  type RefitParams,
} from '../ipc';
import { isFinished, throttleToFrame } from '../ipc/channels';
import type { RefitFinished } from '../bindings/RefitFinished';
import type { RefitProgressEvent } from '../bindings/RefitProgressEvent';

interface ActiveRefit {
  jobId: number | null;
  progress: RefitProgressEvent | null;
}

interface ProjectionState {
  activeRefit: ActiveRefit | null;
  /** The most recent re-fit's terminal event, kept until the next one starts. */
  lastRefitOutcome: RefitFinished | null;

  startRefit: (params: Partial<RefitParams>) => void;
  cancelRefit: () => Promise<void>;
}

export const useProjectionStore = create<ProjectionState>()((set, get) => ({
  activeRefit: null,
  lastRefitOutcome: null,

  startRefit: (params) => {
    if (get().activeRefit) return;
    set({ activeRefit: { jobId: null, progress: null }, lastRefitOutcome: null });

    const onEvent = throttleToFrame<RefitEvent>((event) => {
      if (isFinished(event)) {
        set({ activeRefit: null, lastRefitOutcome: event });
        return;
      }
      set((state) =>
        state.activeRefit
          ? { activeRefit: { ...state.activeRefit, jobId: event.jobId, progress: event } }
          : state,
      );
    });

    ipcStartRefit(params, onEvent)
      .then((jobId) => {
        set((state) =>
          state.activeRefit ? { activeRefit: { ...state.activeRefit, jobId } } : state,
        );
      })
      .catch(() => {
        onEvent.dispose();
        set({ activeRefit: null });
      });
  },

  cancelRefit: async () => {
    const refit = get().activeRefit;
    if (refit?.jobId != null) await ipcCancelRefit(refit.jobId);
  },
}));
