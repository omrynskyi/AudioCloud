/**
 * Phase 7's frame-time exit criterion, measured.
 *
 * > sustained **< 16.6 ms p99** frame time orbiting 50,000 points on Apple Silicon; idle
 * > canvas renders **0 frames/s**; GPU pick returns the correct sample under a dense
 * > cluster; context loss recovers without a refetch.
 *
 * All four, in one run, in a real WKWebView:
 *
 *     npm run profile                    # builds the harness, then measures
 *     node scripts/orbit_profile.mjs --duration 30000 --points 50000
 *
 * The harness page is `src/profile/main.tsx`. It mounts the same scene the app mounts —
 * same shaders, same picking pass, same `frameloop="demand"` canvas — over a seeded
 * synthetic cloud in the real wire format, drives a scripted 30-second orbit, and reports.
 *
 * Why WKWebView and not a browser: `overview.md` §5.1's whole caveat is that WebKit's WebGL
 * goes through ANGLE on Metal and does not behave like Chrome's. A 60 fps number from a
 * different engine is not evidence about this one. See `scripts/webview_eval.swift`.
 */

import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeFile } from 'node:fs/promises';

import { runInWebView } from './webview_eval.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');

function flag(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && index + 1 < process.argv.length
    ? process.argv[index + 1]
    : fallback;
}

const durationMs = Number(flag('duration', 30_000));
const points = Number(flag('points', 50_000));
const jsonPath = flag('json', null);
// The default window is the size a real user's window is, because fill rate is the cost
// here and a measurement at 400x300 would be a measurement of nothing.
const width = Number(flag('width', 1440));
const height = Number(flag('height', 900));

// The harness publishes `window.audiocloudHarness` from a React effect, which lands after
// `didFinish`. Polling on animation frames rather than a fixed sleep, so a slow first
// shader compile delays the run instead of failing it.
const expression = `(async () => {
  for (let i = 0; i < 600 && !window.audiocloudHarness; i++) {
    await new Promise((r) => requestAnimationFrame(r));
  }
  if (!window.audiocloudHarness) throw new Error('the harness never came up');
  return await window.audiocloudHarness.run({ durationMs: ${durationMs} });
})()`;

const report = await runInWebView({
  root: join(repo, 'dist-profile'),
  page: `profile.html?points=${points}`,
  expr: expression,
  // The orbit itself, plus warmup, plus the idle window, plus the two checks.
  timeout: Math.ceil(durationMs / 1000) + 90,
  width,
  height,
});

const ms = (n) => `${n.toFixed(2)} ms`;
const { rafFloor, emptyScene, orbit, orbitWithHover, picking, contextLoss } = report;
const budgetMs = 16.6;

console.log(
  `\n-- Phase 7: ${report.pointCount.toLocaleString()} points, ${(durationMs / 1000).toFixed(0)} s scripted orbit --`,
);
console.log(`  engine                ${report.userAgent.split(') ').pop()}`);
console.log(
  `  drawing buffer        ${orbit.drawingBuffer.width}x${orbit.drawingBuffer.height} @ dpr ${orbit.drawingBuffer.pixelRatio}`,
);
console.log(
  `  display interval      ${ms(orbit.displayIntervalMs)}  (${(1000 / orbit.displayIntervalMs).toFixed(1)} Hz)`,
);
const row = (label, d) =>
  console.log(
    `  ${label.padEnd(21)} p50 ${ms(d.p50)}   p95 ${ms(d.p95)}   p99 ${ms(d.p99)}   max ${ms(d.max)}   n=${d.count}`,
  );

console.log('');
console.log('  -- controls --');
row('rAF floor, no draw', rafFloor);
row('empty scene', emptyScene.frameIntervalMs);
console.log(`\n  -- ${report.pointCount.toLocaleString()} points --`);
row('frame interval', orbit.frameIntervalMs);
row('cpu inside render()', orbit.cpuRenderMs);
console.log(`  dropped (>1.5 vsync)  ${orbit.droppedFrames}`);
console.log(
  `  draw calls / frame    ${orbit.info.callsPerFrame}   (${orbit.info.pointsPerFrame.toLocaleString()} points)`,
);

console.log(
  `\n  -- orbiting while hovering at 20 Hz, ${(orbitWithHover.frameIntervalMs.count / (1000 / orbitWithHover.displayIntervalMs)).toFixed(0)} s --`,
);
row('frame interval', orbitWithHover.frameIntervalMs);
row('hover pick', orbitWithHover.pickMs);
console.log(`  dropped (>1.5 vsync)  ${orbitWithHover.droppedFrames}`);

console.log(
  `\n  build from payload    ${ms(report.buildMs)}   (decode + interleave + geometry)`,
);
console.log(`  mount -> first frame  ${ms(report.mountToFirstFrameMs)}`);

console.log(
  `\n  pick vs CPU model     ${picking.exact} exact, ${picking.equivalent} equivalent, ` +
    `${picking.behind} behind, ${picking.uncovered} uncovered, ${picking.missed} missed  (of ${picking.trials})`,
);
console.log(
  `                        mean ${picking.meanCandidates.toFixed(1)} sprites over each tested pixel, max ${picking.maxCandidates}`,
);
console.log(
  `  pick latency          mean ${ms(picking.meanPickMs)}   max ${ms(picking.maxPickMs)}`,
);
console.log(
  `  context loss          lost=${contextLoss.lostFired} restored=${contextLoss.restoredFired} ` +
    `frames=${contextLoss.framesAfterRestore} picks=${contextLoss.picksAfterRestore}/10 sameBuffer=${contextLoss.sameSourceBuffer}`,
);

// The exit criteria are `task.md`'s four, and only those. The two derived figures below are
// printed as context and are deliberately *not* pass/fail rows: inventing a threshold for a
// derived quantity when the stated one fails is moving the goalposts, and the point of the
// controls is to explain a number, not to replace it.
const overFloor = orbit.frameIntervalMs.p99 - rafFloor.p99;
const overEmpty = orbit.frameIntervalMs.p99 - emptyScene.frameIntervalMs.p99;

console.log('\n  context for the frame-time row');
console.log(
  `    p50 equals the display interval (${ms(orbit.displayIntervalMs)}), so the common case is the display's own rate`,
);
console.log(
  `    p99 is ${overFloor >= 0 ? '+' : ''}${overFloor.toFixed(2)} ms over an empty page's rAF cadence in the same engine`,
);
console.log(
  `    p99 is ${overEmpty >= 0 ? '+' : ''}${overEmpty.toFixed(2)} ms over the same loop with the cloud hidden`,
);

const results = [
  [
    `frame time p99 < ${budgetMs} ms`,
    orbit.frameIntervalMs.p99 < budgetMs,
    ms(orbit.frameIntervalMs.p99),
  ],
  [
    'idle canvas renders 0 frames/s',
    orbit.idleFrames === 0,
    `${orbit.idleFrames} frames in ${orbit.idleMs} ms`,
  ],
  [
    'pick correct under a dense cluster',
    picking.passed,
    `${picking.exact + picking.equivalent}/${picking.trials} correct`,
  ],
  [
    'context loss recovers, no refetch',
    contextLoss.passed,
    contextLoss.supported ? 'recovered' : 'unsupported',
  ],
];

console.log('\n  exit criteria');
let failed = 0;
for (const [name, ok, detail] of results) {
  if (!ok) failed++;
  console.log(`    ${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(36)} ${detail}`);
}
console.log('');

if (jsonPath) {
  await writeFile(jsonPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`  wrote ${jsonPath}\n`);
}

process.exit(failed === 0 ? 0 : 1);
