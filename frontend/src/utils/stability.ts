/**
 * Temporal stabilisation for the live pose readout.
 *
 * The classifier runs per frame and is right roughly a third to a half of the
 * time on real-world input, so the raw label flips between poses several times
 * a second. On screen that reads as a broken app even when the underlying
 * model is behaving exactly as measured, and it also causes the coaching layer
 * to re-target a different joint mid-sentence.
 *
 * Nothing here changes what the model predicts. It only decides WHEN the
 * display is allowed to change, which is a presentation concern and belongs on
 * the client.
 */

/** Sticky label with hysteresis: a new pose must win the recent window by a
 *  clear margin before it is allowed to replace the current one.
 *
 *  A plain mode/majority filter still flickers whenever two poses are close,
 *  because the winner changes on a single frame. Requiring the challenger to
 *  beat the incumbent by `margin` votes means a genuine pose change (which
 *  wins decisively and keeps winning) still comes through fast, while a
 *  one-frame misfire cannot move the display at all. */
export class StickyLabel {
  private buf: string[] = [];
  private current: string;

  constructor(
    initial: string,
    private windowSize = 7,
    private margin = 2,
  ) {
    this.current = initial;
  }

  push(label: string): string {
    this.buf.push(label);
    if (this.buf.length > this.windowSize) this.buf.shift();

    const counts = new Map<string, number>();
    for (const l of this.buf) counts.set(l, (counts.get(l) ?? 0) + 1);

    const mine = counts.get(this.current) ?? 0;
    let bestLabel = this.current;
    let bestCount = mine;
    counts.forEach((c, l) => {
      if (l !== this.current && c > bestCount) {
        bestLabel = l;
        bestCount = c;
      }
    });
    if (bestLabel !== this.current && bestCount - mine >= this.margin) {
      this.current = bestLabel;
    }
    return this.current;
  }

  get value(): string {
    return this.current;
  }

  reset(to: string) {
    this.buf = [];
    this.current = to;
  }
}

/** Exponential moving average, for the scores and per-joint deviations.
 *
 *  `alpha` is the weight of the newest sample. 0.35 settles in roughly three
 *  frames, which is fast enough to feel live while removing the per-frame
 *  twitch that makes a percentage readout unreadable. */
export class Ema {
  private v: number | null = null;
  constructor(private alpha = 0.35) {}

  push(x: number): number {
    if (!Number.isFinite(x)) return this.v ?? 0;
    this.v = this.v === null ? x : this.alpha * x + (1 - this.alpha) * this.v;
    return this.v;
  }

  get value(): number | null {
    return this.v;
  }

  reset() {
    this.v = null;
  }
}

/** Per-joint EMA map, so the skeleton overlay stops strobing between
 *  "good" and "bad" colours on a joint sitting near its threshold. */
export class EmaMap {
  private m = new Map<string, Ema>();
  constructor(private alpha = 0.35) {}

  push(d: { [k: string]: number }): { [k: string]: number } {
    const out: { [k: string]: number } = {};
    for (const [k, v] of Object.entries(d)) {
      let e = this.m.get(k);
      if (!e) {
        e = new Ema(this.alpha);
        this.m.set(k, e);
      }
      out[k] = e.push(v);
    }
    return out;
  }

  reset() {
    this.m.clear();
  }
}
