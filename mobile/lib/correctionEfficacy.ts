/**
 * Correction-efficacy loop.
 *
 * Every automated coaching system this project surveyed is OPEN loop: detect a
 * fault, speak a cue, detect the fault again, speak the same cue again. It
 * never establishes whether the instruction it just gave actually changed the
 * body. That is why such systems feel deaf, and it is also a safety issue --
 * repeating "bend deeper" at someone whose hip simply will not go further is
 * the mechanism behind the hyperextension this project exists to prevent.
 *
 * This closes the loop. Because the backend now reports WHICH joint a cue was
 * trying to move, the client can watch that specific joint over a short
 * response window and measure the change in its deviation. That measurement
 * then drives two things:
 *
 *   1. escalation - a cue that produced no measurable change is not repeated
 *      verbatim; the attempt counter goes to the backend, which quantifies the
 *      cue and then, if it still fails, backs the user out of the shape
 *      instead of pushing harder;
 *   2. a per-user, per-joint efficacy record, so the app can eventually tell
 *      the user (and an instructor) which cues actually work for THIS body
 *      rather than which cues are generically popular.
 *
 * Measuring the targeted joint rather than the overall score matters: the
 * whole-body correctness number moves for reasons unrelated to the cue
 * (breathing sway, a different limb drifting), so improvement attributed to it
 * would be mostly noise.
 *
 * REACT NATIVE COPY: kept byte-identical to frontend/src/utils/ apart from
 * persistence. React Native has no window.localStorage, so the save/load pair
 * below is a deliberate no-op here -- the loop is fully functional in memory
 * for the session, it just does not survive an app restart. Using
 * AsyncStorage would make these methods async and fork the logic from the web
 * copy, which is not worth it for escalation state that is only meaningful
 * within a single practice session anyway.
 */

export interface EfficacyRecord {
  joint: string;
  pose: string;
  /** deviation in degrees when the cue was delivered */
  before: number;
  /** best deviation observed during the response window */
  after: number;
  /** positive = the joint moved toward correct */
  improvementDeg: number;
  worked: boolean;
  attempt: number;
  at: number;
}

/** How long the user gets to respond before the cue is judged. Short enough
 *  that the loop stays responsive, long enough to hear a sentence, process it
 *  and move -- a cue judged after 2s would fail purely because the sentence
 *  was still being spoken. */
export const RESPONSE_WINDOW_MS = 8000;

/** Degrees of improvement below which a cue counts as having done nothing.
 *  Set above the landmark jitter floor so ordinary noise cannot be mistaken
 *  for the user responding. */
export const IMPROVEMENT_THRESHOLD_DEG = 3.0;

interface PendingCue {
  joint: string;
  pose: string;
  before: number;
  attempt: number;
  deliveredAt: number;
  best: number;
}

export class CorrectionEfficacyTracker {
  private pending: PendingCue | null = null;
  /** attempts already made for a (pose, joint) that produced no improvement */
  private failures = new Map<string, number>();
  private history: EfficacyRecord[] = [];

  constructor() {
    this.load();
  }

  private key(pose: string, joint: string) {
    return `${pose}::${joint}`;
  }

  /** Attempt index to send with the next cue for this joint. */
  attemptFor(pose: string, joint: string): number {
    return this.failures.get(this.key(pose, joint)) ?? 0;
  }

  /** Record that a cue was just delivered. */
  onCueDelivered(pose: string, joint: string | null | undefined,
                 deviations: { [k: string]: number }, now: number) {
    if (!joint) {
      this.pending = null;
      return;
    }
    const before = deviations[joint];
    if (!Number.isFinite(before)) {
      this.pending = null;
      return;
    }
    this.pending = {
      joint, pose, before,
      attempt: this.attemptFor(pose, joint),
      deliveredAt: now,
      best: before,
    };
  }

  /**
   * Feed every subsequent frame's deviations in. Returns a record once the
   * response window closes, otherwise null.
   */
  onFrame(deviations: { [k: string]: number }, now: number): EfficacyRecord | null {
    const p = this.pending;
    if (!p) return null;

    const cur = deviations[p.joint];
    if (Number.isFinite(cur) && cur < p.best) p.best = cur;

    if (now - p.deliveredAt < RESPONSE_WINDOW_MS) return null;

    const improvement = p.before - p.best;
    const worked = improvement >= IMPROVEMENT_THRESHOLD_DEG;
    const k = this.key(p.pose, p.joint);
    if (worked) {
      // it landed -- forget the accumulated failures so the next cue for this
      // joint starts plain again rather than staying permanently escalated
      this.failures.delete(k);
    } else {
      this.failures.set(k, Math.min((this.failures.get(k) ?? 0) + 1, 2));
    }

    const rec: EfficacyRecord = {
      joint: p.joint,
      pose: p.pose,
      before: p.before,
      after: p.best,
      improvementDeg: improvement,
      worked,
      attempt: p.attempt,
      at: now,
    };
    this.history.push(rec);
    if (this.history.length > 200) this.history.shift();
    this.pending = null;
    this.save();
    return rec;
  }

  /** Per-joint hit rate, for the session summary. */
  summary(): { key: string; tried: number; worked: number; rate: number }[] {
    const agg = new Map<string, { tried: number; worked: number }>();
    for (const r of this.history) {
      const k = this.key(r.pose, r.joint);
      const a = agg.get(k) ?? { tried: 0, worked: 0 };
      a.tried += 1;
      if (r.worked) a.worked += 1;
      agg.set(k, a);
    }
    const out: { key: string; tried: number; worked: number; rate: number }[] = [];
    agg.forEach((v, key) => {
      out.push({ key, tried: v.tried, worked: v.worked,
                 rate: v.tried ? v.worked / v.tried : 0 });
    });
    return out.sort((a, b) => b.tried - a.tried);
  }

  reset() {
    this.pending = null;
    this.failures.clear();
  }

  // Persisted so escalation state survives a reload mid-session. Wrapped
  // because storage throws in private windows and is simply absent during
  // SSR -- neither should break the pipeline.
  private save() {
    /* no-op on React Native; see the note at the top of this file */
  }

  private load() {
    /* no-op on React Native; see the note at the top of this file */
  }
}
