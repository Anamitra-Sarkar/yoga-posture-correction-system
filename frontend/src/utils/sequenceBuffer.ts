/**
 * Time-based landmark buffer for the ST-GCN sequence model.
 *
 * WHY THIS EXISTS: the ST-GCN is trained on 60 CONSECUTIVE video frames
 * (about 2.4 s at 25 fps). The old buffer got one frame per completed API
 * cycle -- at most 2 per second, and ~0.5 per second in practice because each
 * cycle waits on a 1.2-2.2 s round trip -- so its "60 frames" spanned 30-120 s
 * and the model was fed motion 12-50x slower than anything it was trained on.
 *
 * This buffer is filled from EVERY camera result (not from the throttled API
 * loop), keeps timestamps, and returns a window resampled to the training
 * rate (25 fps, 60 frames) by linear interpolation, so it is correct whatever
 * frame rate the device actually achieves. It returns null -- and the caller
 * falls back to the per-frame classifier -- when there is not enough recent
 * history or the camera stalled inside the window.
 *
 * Joints MediaPipe reports as not visible keep their last good position,
 * which approximates the visibility-gated interpolation the training data got.
 */
export const SEQ_FRAMES = 60;
export const SEQ_FPS = 25;
export const SEQ_SPAN_MS = ((SEQ_FRAMES - 1) * 1000) / SEQ_FPS; // 2360 ms
const MAX_GAP_MS = 250;   // a stall longer than this inside the window invalidates it
const KEEP_MS = 4000;
const MIN_VISIBILITY = 0.5;

interface Sample { t: number; c: number[] }

export class SequenceBuffer {
  private track: Sample[] = [];

  clear(): void {
    this.track = [];
  }

  /** landmarks: [33][x, y, z, visibility]; t: timestamp in ms. */
  push(landmarks: number[][], t: number): void {
    if (landmarks.length !== 33) return;
    const prev = this.track.length ? this.track[this.track.length - 1].c : null;
    const c = new Array<number>(99);
    for (let j = 0; j < 33; j++) {
      const [x, y, z, v] = landmarks[j];
      const hold = prev !== null && (v ?? 0) < MIN_VISIBILITY;
      c[3 * j] = hold ? prev![3 * j] : x;
      c[3 * j + 1] = hold ? prev![3 * j + 1] : y;
      c[3 * j + 2] = hold ? prev![3 * j + 2] : z;
    }
    this.track.push({ t, c });
    const cutoff = t - KEEP_MS;
    while (this.track.length > 2 && this.track[0].t < cutoff) this.track.shift();
  }

  /** 60 x 99 window ending at the newest sample, resampled to SEQ_FPS; null if unusable. */
  window(): number[][] | null {
    const n = this.track.length;
    if (n < 2) return null;
    const end = this.track[n - 1].t;
    const start = end - SEQ_SPAN_MS;
    if (this.track[0].t > start) return null; // not enough history yet
    let i0 = 0;
    while (i0 + 1 < n && this.track[i0 + 1].t <= start) i0++;
    for (let i = i0 + 1; i < n; i++) {
      if (this.track[i].t - this.track[i - 1].t > MAX_GAP_MS) return null; // camera stalled
    }
    const out: number[][] = [];
    let k = i0;
    for (let f = 0; f < SEQ_FRAMES; f++) {
      const t = start + (f * 1000) / SEQ_FPS;
      while (k + 1 < n && this.track[k + 1].t <= t) k++;
      const a = this.track[k];
      const b = this.track[Math.min(k + 1, n - 1)];
      const span = b.t - a.t;
      const w = span > 0 ? Math.min(1, Math.max(0, (t - a.t) / span)) : 0;
      out.push(a.c.map((v, i) => v + (b.c[i] - v) * w));
    }
    return out;
  }
}
