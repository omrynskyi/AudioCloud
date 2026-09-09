/** Value formatting shared by the panels, so a duration reads the same in all of them. */

/** `m:ss`, or an en-space-free placeholder for a sample whose length is unknown. */
export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '-';
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}
