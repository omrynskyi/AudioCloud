/**
 * Global keyboard shortcuts (`task.md` Phase 9): space auditions, `/` focuses search, escape
 * clears the selection. Arrow-key neighbor stepping is `Inspector.tsx`'s — it already holds
 * the neighbor list this would otherwise have to fetch a second time.
 *
 * Mounted once, in `Shell.tsx`.
 */

import { useEffect, useRef } from 'react';

import { playSample, stopPlayback } from '../ipc';
import { useSceneStore } from '../store/scene';

function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  return (
    !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
  );
}

export function useShortcuts() {
  const playing = useRef(false);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (isTypingTarget(e.target)) {
        // Escape still clears the selection from inside a text field — nothing else does.
        if (e.key === 'Escape') {
          (e.target as HTMLElement).blur();
        }
        return;
      }

      switch (e.key) {
        case ' ': {
          e.preventDefault();
          const { selectedSampleId, hoveredSampleId } = useSceneStore.getState();
          const target = hoveredSampleId ?? selectedSampleId;
          if (target === null) return;
          if (playing.current) {
            void stopPlayback();
            playing.current = false;
          } else {
            void playSample(target, 1.0);
            playing.current = true;
          }
          break;
        }
        case '/': {
          e.preventDefault();
          document.getElementById('search-input')?.focus();
          break;
        }
        case 'Escape': {
          useSceneStore.getState().select(null);
          useSceneStore.getState().clearSelectedIds();
          break;
        }
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
