/**
 * Shared empty/loading/error presentational states (`task.md` Phase 9: "empty, loading, and
 * error states for every panel"). Every panel below renders one of these instead of rolling
 * its own, so "0 results" and "model not downloaded" look like the same application.
 *
 * `ErrorState`'s default text comes from `describe()` — good enough for a variant with
 * nothing else to offer. A panel that *does* have a specific recovery action (a download
 * button for `modelMissing`, a link to the running scan for `scanInProgress`) should switch
 * on `error.kind` itself and pass `action`/`children` rather than relying on this to guess —
 * exactly the distinction `ipc/errors.ts`'s `describe` docstring draws.
 */

import type { ReactNode } from 'react';

import { describe, type AppError } from '../ipc';

function Frame({
  icon,
  title,
  detail,
  action,
}: {
  icon: ReactNode;
  title: string;
  detail?: string | undefined;
  action?: ReactNode | undefined;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <div className="text-2xl opacity-50">{icon}</div>
      <p className="text-sm text-neutral-300">{title}</p>
      {detail && <p className="max-w-xs text-xs text-neutral-500">{detail}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex h-full items-center justify-center py-10">
      <p className="font-mono text-xs text-neutral-500">{label}</p>
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return <Frame icon="○" title={title} detail={detail} action={action} />;
}

export function ErrorState({
  error,
  fallback,
  action,
}: {
  error: AppError;
  /** Overrides `describe(error)` when a panel has a more specific message to show. */
  fallback?: string;
  action?: ReactNode;
}) {
  return <Frame icon="!" title={fallback ?? describe(error)} action={action} />;
}

/** Overlay + centered panel + title/close-button header shared by every modal panel. */
export function Modal({
  title,
  onClose,
  closeLabel,
  zIndexClassName = 'z-20',
  panelClassName = 'max-h-[80vh] w-[420px]',
  children,
}: {
  title: string;
  onClose: () => void;
  closeLabel: string;
  /** Stacking two modals above each other (e.g. tuning over settings) needs the top one higher. */
  zIndexClassName?: string;
  /** Tailwind sizing classes for the panel itself; each modal's content dictates its own. */
  panelClassName?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={`fixed inset-0 ${zIndexClassName} flex items-center justify-center bg-black/60`}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        className={`${panelClassName} overflow-y-auto rounded-lg border border-neutral-800 p-4 text-xs shadow-xl`}
      >
        <header className="mb-3 flex items-center justify-between">
          <h1 className="text-sm font-medium text-neutral-100">{title}</h1>
          <button
            type="button"
            onClick={onClose}
            aria-label={closeLabel}
            className="text-neutral-500 hover:text-neutral-200"
          >
            ×
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}

export function PanelButton({
  children,
  onClick,
  disabled,
  variant = 'default',
  className = '',
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: 'default' | 'danger';
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded border px-2.5 py-1 text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        variant === 'danger'
          ? 'border-red-900 text-red-400 hover:bg-red-950'
          : 'border-neutral-700 text-neutral-300 hover:bg-neutral-800'
      } ${className}`}
    >
      {children}
    </button>
  );
}
