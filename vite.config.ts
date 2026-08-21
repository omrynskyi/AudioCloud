import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Tauri drives this config: the dev server must be on a fixed port it can point the
// WKWebView at, and `src-tauri/` must never be watched or the Rust rebuild loops.
export default defineConfig({
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
  },
});
