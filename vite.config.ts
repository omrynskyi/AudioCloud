import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Tauri drives this config: the dev server must be on a fixed port it can point the
// WKWebView at, and `src-tauri/` must never be watched or the Rust rebuild loops.
//
// `--mode profile` builds one other thing: `profile.html`, the Phase 7 measurement harness
// (`src/profile/main.tsx`). It is a separate mode rather than a second Rollup input on the
// app build because it must never ship -- it mounts a synthetic 50,000-point library and
// exposes a function that drives the camera for thirty seconds.
export default defineConfig(({ mode }) => {
  const profiling = mode === 'profile';
  return {
    plugins: [react(), tailwindcss()],
    clearScreen: false,
    server: {
      port: 1420,
      strictPort: true,
      watch: { ignored: ['**/src-tauri/**'] },
    },
    build: {
      // WKWebView on the oldest macOS we target; no need to down-level further.
      target: 'safari15',
      sourcemap: true,
      ...(profiling
        ? {
            outDir: 'dist-profile',
            emptyOutDir: true,
            // Minified by the same default the app build uses, because a measurement of
            // unminified code reports a main-thread cost the shipped app does not pay.
            rollupOptions: { input: 'profile.html' },
          }
        : {}),
    },
  };
});
