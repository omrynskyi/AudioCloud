/**
 * Phase 7's first task: what is `ALIASED_POINT_SIZE_RANGE` in WKWebView, really?
 *
 * `overview.md` §5.1 chose `THREE.Points` over `InstancedMesh` and attached one condition to
 * the choice: WKWebView runs WebGL through ANGLE on Metal, the maximum `gl_PointSize` is
 * driver-dependent and "can be as low as 64 px", and if the cap is too low for the zoomed-in
 * look then the quad path is not a Phase 10 optimization but a Phase 7 rewrite. `task.md`
 * put the query first in the phase for exactly that reason.
 *
 * So it is answered first, and answered by asking the engine rather than by reading a spec:
 * a real WKWebView, the same class Tauri's `wry` wraps, on this machine's GPU.
 *
 *     node scripts/probe_webgl_caps.mjs
 *
 * The numbers are recorded in `BENCHMARKS.md`. `src/scene/caps.ts` performs the same query
 * at runtime and clamps to what it finds, because a cap measured on one Mac is not a cap
 * guaranteed on another — the probe decides the *architecture*, the runtime query keeps the
 * shader honest on hardware nobody here has.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runInWebView } from './webview_eval.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const root = join(repo, 'build', 'webgl-probe');

/** The lowest cap at which the `THREE.Points` path is still worth having. */
const POINTS_PATH_FLOOR = 64;

const PAGE = `<!doctype html>
<html><head><meta charset="utf-8"><title>webgl caps</title></head>
<body><canvas id="c" width="256" height="256"></canvas>
<script>
window.probe = function () {
  const canvas = document.getElementById('c');
  const report = { devicePixelRatio: window.devicePixelRatio };
  for (const name of ['webgl2', 'webgl']) {
    const gl = canvas.getContext(name, { antialias: false, powerPreference: 'high-performance' });
    if (!gl) { report[name] = null; continue; }
    const pointRange = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    report[name] = {
      pointSizeRange: [pointRange[0], pointRange[1]],
      lineWidthRange: Array.from(gl.getParameter(gl.ALIASED_LINE_WIDTH_RANGE)),
      version: gl.getParameter(gl.VERSION),
      shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
      vendor: info ? gl.getParameter(info.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      renderer: info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      maxRenderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
      maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
      maxVaryingVectors: gl.getParameter(gl.MAX_VARYING_VECTORS ?? gl.MAX_VARYING_COMPONENTS),
      // Reported because Phase 7's frame-time criterion is about the GPU, and whether this
      // engine will hand out a GPU clock decides whether that number can be measured
      // directly or only inferred from dropped frames.
      timerQuery: !!gl.getExtension('EXT_disjoint_timer_query_webgl2'),
      extensions: gl.getSupportedExtensions(),
    };
    // Only the context the app will actually get. Three.js takes WebGL2 wherever it exists,
    // and reporting a WebGL1 fallback's cap alongside it invites reading the wrong row.
    break;
  }
  return JSON.stringify(report);
};
</script></body></html>
`;

await mkdir(root, { recursive: true });
await writeFile(join(root, 'index.html'), PAGE);

const report = await runInWebView({ root, expr: 'window.probe()', timeout: 30 });
const gl = report.webgl2 ?? report.webgl;
if (!gl) {
  console.error('WKWebView gave no WebGL context at all.');
  process.exit(1);
}

const [min, max] = gl.pointSizeRange;
const cssMax = max / report.devicePixelRatio;

console.log(`\n-- WKWebView WebGL capabilities --`);
console.log(`  context                 ${gl.version}`);
console.log(`  shading language        ${gl.shadingLanguageVersion}`);
console.log(`  vendor / renderer       ${gl.vendor} / ${gl.renderer}`);
console.log(`  devicePixelRatio        ${report.devicePixelRatio}`);
console.log(`  ALIASED_POINT_SIZE_RANGE [${min}, ${max}]   device px`);
console.log(
  `                           [${min / report.devicePixelRatio}, ${cssMax}]   CSS px at this dpr`,
);
console.log(`  MAX_TEXTURE_SIZE        ${gl.maxTextureSize}`);
console.log(`  MAX_VERTEX_ATTRIBS      ${gl.maxVertexAttribs}`);

console.log(
  max >= POINTS_PATH_FLOOR
    ? `\n  VERDICT: THREE.Points stands. ${max} device px is ${(max / POINTS_PATH_FLOOR).toFixed(1)}x the\n` +
        `  ${POINTS_PATH_FLOOR} px floor below which overview.md 5.1 says to switch to InstancedMesh quads.\n`
    : `\n  VERDICT: switch to the InstancedMesh quad path. ${max} device px is below the\n` +
        `  ${POINTS_PATH_FLOOR} px floor, and overview.md 5.1 says that decision belongs in Phase 7,\n` +
        `  not Phase 10.\n`,
);
