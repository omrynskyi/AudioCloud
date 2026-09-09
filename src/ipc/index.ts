/**
 * The IPC boundary (`overview.md` §6).
 *
 * Three transports, one entry point. Nothing outside `src/ipc/` calls `invoke`, constructs a
 * `Channel`, or fetches an `abpeaks://` URL; everything outside it imports from here.
 */

export * from './binary';
export * from './channels';
export * from './commands';
export * from './errors';
export * from './peaks';
