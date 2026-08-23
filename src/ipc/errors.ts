/**
 * The typed error surface, as the frontend sees it (`overview.md` §6.7).
 *
 * `AppError` is generated from Rust into `src/bindings/AppError.ts` and is a discriminated
 * union on `kind`. Everything here is about getting a rejected `invoke` *into* that type:
 * Tauri rejects with whatever the command serialized, which is an `AppError` for every
 * command in this app — but a bug in the core, a panic, or a message from Tauri itself can
 * reject with a string, and code that assumed otherwise would crash while rendering an
 * error dialog.
 */

import type { AppError } from '../bindings/AppError';

export type { AppError };

/** Whether an unknown value is one of our tagged errors. */
export function isAppError(value: unknown): value is AppError {
  return (
    typeof value === 'object' &&
    value !== null &&
    'kind' in value &&
    typeof value.kind === 'string'
  );
}

/**
 * What `src/ipc/commands.ts` throws.
 *
 * A real `Error` wrapping the tagged union, rather than the union thrown bare. Three reasons,
 * all of which show up the first time something goes wrong in front of a user: a bare object
 * has no stack, React error boundaries and `console.error` render it as `[object Object]`,
 * and `instanceof Error` is what every piece of JavaScript ever written checks. The union is
 * on `.error`, and `toAppError` unwraps it, so a caller's `switch` is the same whether it
 * caught this or a raw rejection from Tauri.
 */
export class IpcError extends Error {
  readonly error: AppError;

  constructor(error: AppError) {
    super(describe(error));
    this.name = 'IpcError';
    this.error = error;
  }
}

/**
 * Normalizes anything a rejected `invoke` can produce into an `AppError`.
 *
 * The fallback is `internal`, which is the variant that means "nobody can act on this".
 * That is the honest classification for a rejection this layer does not recognize: it did
 * not come from the typed surface, so no recovery action is known to apply.
 */
export function toAppError(value: unknown): AppError {
  if (value instanceof IpcError) return value.error;
  if (isAppError(value)) return value;
  return { kind: 'internal', detail: String(value) };
}

/**
 * A short, human sentence for an error, for a toast or an inline message.
 *
 * **Not a substitute for switching on `kind`.** Every variant in the union exists because it
 * has a *different recovery* — `modelMissing` wants a download button, `scanInProgress`
 * wants a link to the running scan, `decode` wants the path — and rendering all of them as
 * one paragraph of text is exactly the thing cross-cutting rule 8 was written against. This
 * is the fallback for places where there is genuinely nothing to offer.
 */
export function describe(error: AppError): string {
  switch (error.kind) {
    case 'modelMissing':
      return 'The audio model is not installed yet.';
    case 'modelDownload':
      return error.detail.resumable
        ? `The download stopped: ${error.detail.message}. It can be resumed.`
        : `The download failed: ${error.detail.message}`;
    case 'checksumMismatch':
      return 'The downloaded model did not match its published checksum.';
    case 'modelUnpinned':
      return `This build has no published checksum for model ${error.detail.version}, so nothing can be downloaded.`;
    case 'database':
      return `The library database reported: ${error.detail}`;
    case 'decode':
      return `${error.detail.path} could not be read: ${error.detail.reason}`;
    case 'notFound':
      return `Not found: ${error.detail}`;
    case 'scanInProgress':
      return 'A scan is already running.';
    case 'refitInProgress':
      return 'The map is already being rebuilt.';
    case 'cancelled':
      return 'Cancelled.';
    case 'noProjection':
      return 'There is no map yet. Scan a folder, then build the map.';
    case 'tooFewSamples':
      return `${error.detail.operation} needs at least ${error.detail.need} embedded samples, and there are ${error.detail.have}.`;
    case 'invalidArgument':
      return `${error.detail.field}: ${error.detail.reason}`;
    case 'unavailable':
      return `${error.detail.feature} is not available in this build yet.`;
    case 'internal':
      return `Something went wrong (${error.detail}).`;
  }
}
