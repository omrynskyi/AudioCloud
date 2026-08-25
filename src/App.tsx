/**
 * The top-level gate (`task.md` Phase 9): a fresh install with no library roots gets the
 * first-run wizard; everything else gets the shell. `Shell.tsx` carries what this file used
 * to do directly through Phase 7 — the point-cloud/feature-column/filter fetches — unchanged.
 */

import { useEffect, useState } from 'react';

import { FirstRun } from './panels/FirstRun';
import { Shell } from './panels/Shell';
import { useLibraryStore } from './store/library';

export default function App() {
  const rootsLoaded = useLibraryStore((s) => s.rootsLoaded);
  const roots = useLibraryStore((s) => s.roots);
  const refreshRoots = useLibraryStore((s) => s.refreshRoots);
  const [firstRunDone, setFirstRunDone] = useState(false);

  useEffect(() => {
    void refreshRoots();
  }, [refreshRoots]);

  if (!rootsLoaded) return <Centered>Loading…</Centered>;
  if (roots.length === 0 && !firstRunDone) {
    return <FirstRun onComplete={() => setFirstRunDone(true)} />;
  }
  return <Shell />;
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex h-full w-full items-center justify-center">
      <p className="max-w-md text-center text-sm text-neutral-500">{children}</p>
    </main>
  );
}
