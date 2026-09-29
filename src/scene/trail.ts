/**
 * The fading ring left behind on points the cursor has just left.
 *
 * Hover is one id in a uniform; the trail is the same idea widened to a small ring buffer of
 * ids plus how far through its fade each one is. Both live in uniforms so a sweep across the
 * cloud still costs no per-point CPU work: the vertex stage matches `aId` against the slots
 * (`points.vert.glsl`), and the only per-frame cost here is rewriting `TRAIL_SLOTS` floats.
 */

/** Must match the array size in `points.vert.glsl`. */
export const TRAIL_SLOTS = 16;

/** How long a left-behind ring takes to fade out completely. */
export const TRAIL_DURATION_MS = 900;

export class HoverTrail {
  /** `aId` per slot, or 0 for an empty one. Uploaded as-is. */
  readonly ids = new Float32Array(TRAIL_SLOTS);
  /** Fade progress per slot: 0 just left, 1 gone. Uploaded as-is. */
  readonly ages = new Float32Array(TRAIL_SLOTS);
  private readonly startedAt = new Float64Array(TRAIL_SLOTS);

  /** Starts (or restarts) a fade for `id`, evicting the oldest slot if all are busy. */
  push(id: number, now: number): void {
    if (id <= 0) return;
    // Reuse the point's own slot, else an empty one, else evict the oldest fade.
    let slot = -1;
    let empty = -1;
    let oldest = 0;
    for (let i = 0; i < TRAIL_SLOTS; i++) {
      if (this.ids[i] === id) {
        slot = i;
        break;
      }
      if (this.ids[i] === 0) {
        if (empty < 0) empty = i;
      } else if (this.startedAt[i]! < this.startedAt[oldest]!) {
        oldest = i;
      }
    }
    if (slot < 0) slot = empty >= 0 ? empty : oldest;
    this.ids[slot] = id;
    this.ages[slot] = 0;
    this.startedAt[slot] = now;
  }

  /** Advances every fade to `now`. Returns whether any slot is still live. */
  update(now: number): boolean {
    let live = false;
    for (let i = 0; i < TRAIL_SLOTS; i++) {
      if (this.ids[i] === 0) continue;
      const age = (now - this.startedAt[i]!) / TRAIL_DURATION_MS;
      if (age >= 1) {
        this.ids[i] = 0;
        this.ages[i] = 1;
      } else {
        this.ages[i] = age;
        live = true;
      }
    }
    return live;
  }
}
