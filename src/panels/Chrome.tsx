/**
 * The floating chrome primitives: the buttons on the map's top bar and its content.
 *
 * These exist because the shell stopped being two walled-off columns. Everything that is
 * not the map is now a `glass` surface sitting on top of it, which means "a control" and "a
 * panel a control opens" are shapes used often enough to be worth defining once - and
 * defining once is also the only way the shape rules in `styles/index.css` (one radius
 * scale, one accent) hold across five call sites.
 */

import { type FocusEventHandler, type Ref, type ReactNode } from 'react';

/**
 * A square icon button on the top bar.
 *
 * `active` is the accent's only appearance in the chrome: a control whose panel is open, or
 * whose setting is doing something. Everything else is grey, which is what makes the one
 * amber thing on screen mean something.
 */
export function IconButton({
  label,
  onClick,
  active = false,
  children,
}: {
  /** Accessible name, and the tooltip. Icons alone are not a label. */
  label: string;
  onClick: () => void;
  active?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      aria-pressed={active}
      className={`rounded-control flex h-7 w-7 items-center justify-center transition-colors active:translate-y-px ${
        active
          ? 'bg-accent-muted text-accent'
          : 'text-neutral-400 hover:bg-neutral-800 hover:text-neutral-100'
      }`}
    >
      {children}
    </button>
  );
}

/** The heading over a group inside the contextual area or the inspector. */
export function PanelHeading({ children }: { children: ReactNode }) {
  return (
    <h2 className="mb-2 text-[10px] font-medium tracking-[0.14em] text-neutral-500 uppercase">
      {children}
    </h2>
  );
}

/**
 * A row in a list of things you can pick exactly one of.
 *
 * Full-width, left-aligned, with the accent reserved for the current pick - the same
 * treatment the map gives the selected point, one level down.
 */
export function ChoiceRow({
  selected,
  onClick,
  children,
}: {
  selected: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      className={`rounded-control flex w-full items-center justify-between px-2 py-1.5 text-left text-xs transition-colors ${
        selected
          ? 'bg-accent-muted text-accent'
          : 'text-neutral-300 hover:bg-neutral-800 hover:text-neutral-100'
      }`}
    >
      {children}
    </button>
  );
}

/**
 * The one text input in the app.
 *
 * Five panels each carried their own copy of the same nine Tailwind classes, which is five
 * places for a focus ring or a placeholder colour to drift out of step - and placeholder
 * contrast is exactly the kind of thing that drifts quietly and then fails an accessibility
 * pass. The placeholder here is `neutral-500` against the `neutral-950` field rather than
 * the `neutral-600` the copies used, which clears 4.5:1.
 */
export function TextInput({
  value,
  onChange,
  placeholder,
  disabled = false,
  onBlur,
  onFocus,
  ariaLabel,
  className = '',
  inputRef,
  id,
  hasLeadingIcon = false,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  onBlur?: () => void;
  onFocus?: FocusEventHandler<HTMLInputElement>;
  ariaLabel?: string;
  className?: string;
  inputRef?: Ref<HTMLInputElement>;
  /** Set on the search field so `useShortcuts.ts` can focus it for the `/` shortcut. */
  id?: string;
  /**
   * Widens the left padding to clear an icon the caller positions over the field. A flag
   * rather than an `!important` override from the caller, so the padding is decided in one
   * place instead of depending on how Tailwind happens to order `px` against `pl`.
   */
  hasLeadingIcon?: boolean;
}) {
  return (
    <input
      ref={inputRef}
      id={id}
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onBlur}
      onFocus={onFocus}
      placeholder={placeholder}
      aria-label={ariaLabel ?? placeholder ?? ''}
      disabled={disabled}
      className={`rounded-control w-full border border-neutral-800 bg-neutral-950 py-1.5 text-xs text-neutral-100 transition-colors placeholder:text-neutral-500 focus:border-neutral-600 focus:outline-none disabled:opacity-40 ${
        hasLeadingIcon ? 'pr-2.5 pl-7' : 'px-2.5'
      } ${className}`}
    />
  );
}
