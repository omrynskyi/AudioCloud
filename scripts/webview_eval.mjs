/**
 * Runs a page in WKWebView and returns what it measures.
 *
 * Compiles `scripts/webview_eval.swift` on demand, serves a directory over loopback, points
 * the webview at it, and resolves with the parsed JSON the page reported. Used by
 * `probe_webgl_caps.mjs` and `orbit_profile.mjs`, which are the two Phase 7 measurements
 * that are only meaningful in WebKit — see the Swift file's header for why.
 *
 * HTTP rather than `file://` because the harness page is an ES module bundle, and module
 * loading over `file://` is blocked by the same-origin rules. Loopback on an ephemeral port
 * costs nothing and makes the page load the way it will in the app.
 *
 * No dependencies, like the other scripts here: a measurement that needs the app's toolchain
 * to run stops being usable the moment the toolchain breaks.
 */

import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, stat, mkdir } from 'node:fs/promises';
import { extname, join, normalize, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, '..');
const SWIFT_SOURCE = join(here, 'webview_eval.swift');
const BINARY = join(repo, 'build', 'webview-eval');

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.map': 'application/json; charset=utf-8',
  '.glsl': 'text/plain; charset=utf-8',
};

async function newer(a, b) {
  try {
    const [sa, sb] = await Promise.all([stat(a), stat(b)]);
    return sa.mtimeMs > sb.mtimeMs;
  } catch {
    return true;
  }
}

/** Compiles the runner if the source has moved since the binary was built. */
export async function buildRunner() {
  if (!(await newer(SWIFT_SOURCE, BINARY))) return BINARY;
  await mkdir(dirname(BINARY), { recursive: true });
  await run('swiftc', ['-O', SWIFT_SOURCE, '-o', BINARY]);
  return BINARY;
}

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'], ...options });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (err += d));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolvePromise(out);
      else reject(new Error(`${command} exited ${code}\n${err.trim() || out.trim()}`));
    });
  });
}

/** A static file server over `root`, on an ephemeral loopback port. */
async function serve(root) {
  const base = resolve(root);
  const server = createServer((req, res) => {
    const requested = normalize(
      decodeURIComponent(new URL(req.url, 'http://x').pathname),
    );
    const path = join(
      base,
      requested.endsWith('/') ? `${requested}index.html` : requested,
    );
    if (!path.startsWith(base)) {
      res.writeHead(403).end();
      return;
    }
    readFile(path).then(
      (body) => {
        res.writeHead(200, {
          'content-type': CONTENT_TYPES[extname(path)] ?? 'application/octet-stream',
          // The harness is rebuilt between runs and a cached bundle would silently measure
          // the previous one.
          'cache-control': 'no-store',
        });
        res.end(body);
      },
      () => res.writeHead(404).end(),
    );
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  return { port, close: () => new Promise((r) => server.close(r)) };
}

/**
 * Serves `root`, opens `page` in WKWebView, evaluates `expr`, and resolves with its JSON.
 *
 * `expr` may produce a value or a promise; both are awaited on the page side.
 */
export async function runInWebView({
  root,
  page = 'index.html',
  expr,
  timeout = 120,
  width,
  height,
}) {
  const binary = await buildRunner();
  const server = await serve(root);
  try {
    const args = [
      '--url',
      `http://127.0.0.1:${server.port}/${page}`,
      '--expr',
      expr,
      '--timeout',
      String(timeout),
    ];
    if (width) args.push('--width', String(width));
    if (height) args.push('--height', String(height));
    return JSON.parse(await run(binary, args));
  } finally {
    await server.close();
  }
}
