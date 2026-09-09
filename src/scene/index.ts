/**
 * The visualization layer (`overview.md` §5).
 *
 * One canvas, one draw call, and no React between the cursor and the GPU. Mount
 * `SceneCanvas` with an `ABPC` payload; drive it through `src/store/scene.ts` and the two
 * data props. Everything else here is exported for the harnesses and for tests — nothing in
 * `panels/` should need it.
 */

export * from './buffers';
export * from './caps';
export * from './colors';
export * from './framing';
export * from './mask';
export * from './materials';
export * from './picking';
export * from './PointCloud';
export * from './SceneCanvas';
export * from './synthetic';
