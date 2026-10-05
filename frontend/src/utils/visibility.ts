/**
 * Visibility tracking with hysteresis, so a joint on the edge of visible does not flicker between "seen" and "hidden".
 * MediaPipe returns a position for EVERY landmark, including ones it cannot see (a guess) and attaches a visibility
 * score; this turns those scores into a stable "hidden" set and the list of angle features that depend on them.
 */
import { FEATURE_NAMES, FEATURE_LANDMARKS } from "./offlineCoach";

export class VisibilityTracker {
  private ema: number[] = new Array(33).fill(1);
  private hidden: boolean[] = new Array(33).fill(false);

  reset(): void {
    this.ema.fill(1);
    this.hidden.fill(false);
  }

  isHidden(i: number): boolean {
    return this.hidden[i] === true;
  }

  /** landmarks: [33][x, y, z, visibility]. Returns the angle features that rest on a hidden landmark. */
  update(landmarks: number[][]): string[] {
    for (let i = 0; i < 33; i++) {
      const v = landmarks[i]?.[3] ?? 0;
      this.ema[i] = 0.7 * this.ema[i] + 0.3 * v;
      if (!this.hidden[i] && this.ema[i] < 0.4) this.hidden[i] = true;
      else if (this.hidden[i] && this.ema[i] > 0.6) this.hidden[i] = false;
    }
    return FEATURE_NAMES.filter((f) => FEATURE_LANDMARKS[f].some((i) => this.hidden[i]));
  }
}
