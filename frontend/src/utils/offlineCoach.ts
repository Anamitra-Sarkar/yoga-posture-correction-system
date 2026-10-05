/**
 * On-device coach: the server's deterministic rule engine, ported line for line so the app keeps working when the
 * server is unreachable, asleep, or slow (and fully offline).
 *
 * Source of truth: backend/app/utils/rules_classifier.py, backend/app/services/correction.py (stage 1: no LLM),
 * backend/app/routers/pose.py (motion state, calibration) and backend/app/services/cascade.py (guided report).
 * The bands, templates and wording live in offlineData.ts, which is GENERATED from those Python modules
 * (backend/tools/gen_offline_data.py). backend/tools/offline_parity.py checks this file against the Python original
 * on thousands of generated cases; run it after touching either side.
 *
 * What it does NOT have: the learned MLP + gate (better pose naming, learned form score), the sequence model, the
 * occlusion recovery service and the LLM paraphrase. It is the "basic" mode, labelled as such in the UI.
 */
import type { CorrectionResponse, FrameInput, FrameResponse, MotionState, CalibrationProfile } from "../types/yoga";
import { POSE_BANDS, TEMPLATES, DEFAULT_CORRECTIONS, ESCALATION } from "./offlineData";

export const FEATURE_NAMES = [
  "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
  "hip_l", "hip_r", "knee_l", "knee_r",
  "ankle_l", "ankle_r", "trunk_l", "trunk_r",
  "neck", "hip_abduct_l", "hip_abduct_r",
];

export type Angles = { [name: string]: number };

/** Which MediaPipe landmarks each angle feature is computed from (mirrors utils/geometry.ts extractAnglesFromLandmarks). */
export const FEATURE_LANDMARKS: { [feature: string]: number[] } = {
  elbow_l: [11, 13, 15], elbow_r: [12, 14, 16],
  shoulder_l: [23, 11, 13], shoulder_r: [24, 12, 14],
  hip_l: [11, 23, 25], hip_r: [12, 24, 26],
  knee_l: [23, 25, 27], knee_r: [24, 26, 28],
  ankle_l: [25, 27, 29], ankle_r: [26, 28, 30],
  trunk_l: [11, 23, 24], trunk_r: [12, 24, 23],
  neck: [0, 11, 12, 23, 24],
  hip_abduct_l: [24, 23, 25], hip_abduct_r: [23, 24, 26],
};

/**
 * Features whose landmarks the camera cannot actually see. MediaPipe still returns a position for a hidden joint
 * (a guess), so an angle built on it can look plausible and be wrong. These are neither scored nor coached.
 * `landmarks` is [33][x, y, z, visibility].
 */
export function hiddenFeatures(landmarks: number[][], threshold = 0.5): string[] {
  if (!landmarks || landmarks.length < 31) return [];
  const out: string[] = [];
  for (const f of FEATURE_NAMES) {
    const ids = FEATURE_LANDMARKS[f];
    if (ids.some((i) => (landmarks[i]?.[3] ?? 0) < threshold)) out.push(f);
  }
  return out;
}

/** A server answer with the hidden joints removed from its per-joint deviations (so nothing is coached on a guess). */
export function maskDeviations<T extends { [k: string]: number } | null | undefined>(devs: T, hidden: string[]): T {
  if (!devs || hidden.length === 0) return devs;
  const out: { [k: string]: number } = { ...devs };
  hidden.forEach((h) => { if (h in out) out[h] = 0.0; });
  return out as T;
}
type Orientation = { torso_incline?: number | null; leg_torso_ratio?: number | null } | null | undefined;

const UNKNOWN = "transition/unknown";
const MOTION_HOLD_MAX_DEG_PER_SEC = 15.0;

const between = (v: number, lo: number, hi: number) => lo <= v && v <= hi;

/** Python's round(): halves go to the nearest EVEN integer (JS Math.round goes up). */
export function pyRound(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

export function anglesToDict(angles: number[]): Angles {
  const out: Angles = {};
  FEATURE_NAMES.forEach((n, i) => { out[n] = angles[i] ?? 0; });
  return out;
}

// ── pose classification (rules_classifier.py) ───────────────────────────────────────────────────────────────────

export function classifyPose(a: Angles, orientation?: Orientation): string {
  if (orientation) {
    const inc = orientation.torso_incline;
    const ratio = orientation.leg_torso_ratio;
    if (inc !== undefined && inc !== null && ratio !== undefined && ratio !== null) {
      const hit = classifyWithOrientation(a, Number(inc), Number(ratio));
      if (hit !== null) return hit;
    }
  }
  return classify2dOnly(a);
}

function classifyWithOrientation(a: Angles, inc: number, ratio: number): string | null {
  const hipL = a.hip_l, hipR = a.hip_r;
  const kneeL = a.knee_l, kneeR = a.knee_r;
  const shL = a.shoulder_l, shR = a.shoulder_r;
  const trL = a.trunk_l, trR = a.trunk_r;
  const neck = a.neck;

  const upright = inc < 35;
  const inverted = inc > 122;
  const horizontal = 22 <= inc && inc <= 122;
  const limbsStraight = hipL > 140 && hipR > 140 && kneeL > 140 && kneeR > 140;

  if (inverted && between(hipL, 20, 150) && between(hipR, 20, 150) && kneeL > 110 && kneeR > 110 && shL > 90 && shR > 90) {
    return "downward_dog";
  }

  if (horizontal) {
    if (limbsStraight && ratio > 0.6 && shL < 60 && shR < 60 && inc > 60) return "corpse";
    if (limbsStraight && between(shL, 45, 120) && between(shR, 45, 120)) return "plank";
    if (hipL > 85 && hipR > 85 && shL < 80 && shR < 80 && neck >= 80) {
      return kneeL > 150 && kneeR > 150 ? "upward_dog" : "cobra_pose";
    }
    if (between(hipL, 60, 150) && between(hipR, 60, 150) && between(kneeL, 60, 140) && between(kneeR, 60, 140)
        && between(shL, 40, 130) && between(shR, 40, 130)) {
      return "table_top";
    }
    if (kneeL > 130 && kneeR > 130 && (shL > 55 || shR > 55)) return "triangle";
  }

  if (upright) {
    if (limbsStraight && shL > 115 && shR > 115 && trL > 65 && trR > 65) return "upward_salute";
    if (ratio < 0.9 && kneeL < 140 && kneeR < 140 && trL >= 55 && trR >= 55) return "seated_easy_pose";
  }
  return null;
}

function classify2dOnly(a: Angles): string {
  const hipL = a.hip_l, hipR = a.hip_r;
  const kneeL = a.knee_l, kneeR = a.knee_r;
  const shoulderL = a.shoulder_l, shoulderR = a.shoulder_r;
  const trunkL = a.trunk_l, trunkR = a.trunk_r;
  const neck = a.neck;

  if (hipL > 140 && hipR > 140 && kneeL > 140 && kneeR > 140 && shoulderL < 55 && shoulderR < 55 && trunkL > 65 && trunkR > 65) return "mountain_pose";
  if (hipL > 140 && hipR > 140 && kneeL > 140 && kneeR > 140 && shoulderL > 115 && shoulderR > 115 && trunkL > 65 && trunkR > 65) return "upward_salute";
  if (between(hipL, 20, 140) && between(hipR, 20, 140) && kneeL > 110 && kneeR > 110 && shoulderL > 95 && shoulderR > 95) return "downward_dog";
  if (hipL > 140 && hipR > 140 && kneeL > 140 && kneeR > 140 && between(shoulderL, 60, 110) && between(shoulderR, 60, 110)) return "plank";
  if (hipL > 120 && hipR > 120 && kneeL > 120 && kneeR > 120 && between(shoulderL, 5, 50) && between(shoulderR, 5, 50) && neck >= 80) return "cobra_pose";
  if (hipL < 90 && hipR < 90 && kneeL < 90 && kneeR < 90 && shoulderL > 85 && shoulderR > 85) return "child_pose";
  if (between(hipL, 60, 120) && between(hipR, 60, 120) && kneeL > 135 && kneeR > 135 && trunkL >= 60 && trunkR >= 60) return "seated_staff";
  if (between(hipL, 50, 120) && between(hipR, 50, 120) && kneeL < 125 && kneeR < 125 && trunkL >= 60 && trunkR >= 60) return "seated_easy_pose";
  if (between(hipL, 75, 140) && between(hipR, 75, 140) && between(kneeL, 75, 140) && between(kneeR, 75, 140)
      && Math.abs(kneeL - kneeR) < 30 && shoulderL > 95 && shoulderR > 95) return "chair_pose";
  if ((kneeL > 150 && hipL > 165 && kneeR < 140) || (kneeR > 150 && hipR > 165 && kneeL < 140)) return "tree_pose";
  const w2Legs = (kneeL < 120 && kneeR > 130) || (kneeR < 120 && kneeL > 130);
  const w2Arms = between(shoulderL, 65, 125) && between(shoulderR, 65, 125);
  if (w2Legs && w2Arms) return "warrior_2";
  const w1Legs = (kneeL < 120 && kneeR > 130) || (kneeR < 120 && kneeL > 130);
  const w1Arms = shoulderL > 110 && shoulderR > 110;
  if (w1Legs && w1Arms) return "warrior_1";
  if ((kneeL < 120 && kneeR > 130) || (kneeR < 120 && kneeL > 130)) return "lunge_pose";
  if (hipL < 70 && hipR < 70 && kneeL > 120 && kneeR > 120) return "standing_forward_fold";
  if (between(hipL, 70, 115) && between(hipR, 70, 115) && kneeL > 130 && kneeR > 130) return "halfway_lift";
  if (between(hipL, 60, 125) && between(hipR, 60, 125) && between(kneeL, 60, 125) && between(kneeR, 60, 125)
      && between(shoulderL, 60, 125) && between(shoulderR, 60, 125)) return "table_top";
  if (hipL > 140 && hipR > 140 && kneeL > 140 && kneeR > 140) return "standing_pose";
  return UNKNOWN;
}

// ── scoring (score_pose) ─────────────────────────────────────────────────────────────────────────────────────────

export function scorePose(poseId: string, a: Angles, hidden: string[] = []): { correctness: number; deviations: Angles } {
  const deviations: Angles = {};
  FEATURE_NAMES.forEach((n) => { deviations[n] = 0.0; });
  // Joints the camera cannot see are skipped (no band check, no deviation): the score covers only what is visible.
  const bands = (POSE_BANDS[poseId] || []).filter(([name]) => hidden.indexOf(name) < 0);
  if (bands.length === 0) {
    return { correctness: poseId !== UNKNOWN ? 0.5 : 0.0, deviations };
  }
  let total = 0.0;
  for (const [name, lo, hi] of bands) {
    const val = a[name] ?? 0.0;
    const dev = val < lo ? lo - val : val > hi ? val - hi : 0.0;
    deviations[name] = dev;
    total += dev;
  }
  const meanDev = total / bands.length;
  return { correctness: Math.max(0.0, Math.min(1.0, 1.0 - meanDev / 45.0)), deviations };
}

// ── motion + personal calibration (routers/pose.py) ─────────────────────────────────────────────────────────────

export function classifyMotionState(motion: number | null | undefined, pose: string): MotionState {
  if (motion === undefined || motion === null) return "unknown";
  if (motion > MOTION_HOLD_MAX_DEG_PER_SEC) return "transitioning";
  return pose === UNKNOWN ? "unrecognized" : "holding";
}

export function applyCalibration(devs: Angles, angles: Angles, calibration: CalibrationProfile): Angles {
  const out: Angles = { ...devs };
  for (const joint of Object.keys(calibration)) {
    if (!(joint in out) || !(joint in angles)) continue;
    const rng = calibration[joint];
    const lo = rng?.min, hi = rng?.max;
    if (lo === undefined || lo === null || hi === undefined || hi === null) continue;
    if (lo <= angles[joint] && angles[joint] <= hi) out[joint] = 0.0;
  }
  return out;
}

export function correctnessFromDeviations(calibrated: Angles, universal: Angles, universalScore: number): number {
  const tracked = Object.keys(universal).filter((j) => universal[j] > 0.0);
  if (tracked.length === 0) return universalScore;
  const totalUniversal = tracked.reduce((s, j) => s + universal[j], 0);
  if (totalUniversal <= 0.0) return universalScore;
  const totalRemaining = tracked.reduce((s, j) => s + (calibrated[j] ?? 0.0), 0);
  const forgiven = 1.0 - totalRemaining / totalUniversal;
  return Math.max(0.0, Math.min(1.0, universalScore + forgiven * (1.0 - universalScore)));
}

export function guidedReport(target: string, detected: string, a: Angles) {
  const { correctness, deviations } = scorePose(target, a);
  return {
    target_pose: target,
    matches: detected === target,
    target_correctness: correctness,
    target_deviations: deviations,
    target_has_bands: (POSE_BANDS[target] || []).length > 0,
  };
}

/** The whole /analyse_frame answer, computed on the device. */
export function offlineFrame(req: FrameInput, hidden: string[] = []): FrameResponse {
  const a = anglesToDict(req.angles);
  const pose = classifyPose(a, req.orientation);
  const { correctness, deviations } = scorePose(pose, a, hidden);
  let personal: number | null = null;
  let calibrated: Angles | null = null;
  if (req.calibration && Object.keys(req.calibration).length > 0) {
    calibrated = applyCalibration(deviations, a, req.calibration);
    personal = correctnessFromDeviations(calibrated, deviations, correctness);
  }
  return {
    pose_id: pose,
    correctness_score: correctness,
    deviations,
    motion_state: classifyMotionState(req.motion, pose),
    personal_correctness_score: personal,
    calibrated_deviations: calibrated,
    candidates: null,
    cascade: { active: false, reason: "on-device" },
    guided: req.target_pose ? guidedReport(req.target_pose, pose, a) : null,
  };
}

// ── coaching text (correction.py, stage 1; the LLM paraphrase is server-only) ───────────────────────────────────

function escalate(text: string, language: string, attempt: number, deg: number): string {
  const tier = Math.min(Math.trunc(attempt), 2);
  if (tier <= 0) return text;
  const tmpl = ESCALATION[String(tier)][language] || ESCALATION[String(tier)].en;
  return tmpl.replace("{base}", text).replace("{deg}", String(pyRound(deg)));
}

export function offlineCorrection(
  poseId: string,
  deviations: Angles,
  language: string = "en",
  attempt: number = 0,
): CorrectionResponse {
  const poseTemplates = TEMPLATES[poseId] || {};
  let target: string | null = null;
  let maxDev = 0.0;
  for (const joint of Object.keys(deviations)) {
    const dev = deviations[joint];
    if (joint in poseTemplates && dev > maxDev) {
      if (dev > 10.0) {
        maxDev = dev;
        target = joint;
      }
    }
  }
  let text = target
    ? poseTemplates[target][language] || "Adjust your alignment."
    : DEFAULT_CORRECTIONS[language] || "Adjust your alignment.";

  if (target && attempt >= 2) {
    return { correction_text: escalate(text, language, attempt, maxDev), is_safe: true, target_joint: target };
  }
  if (target && attempt >= 1) text = escalate(text, language, attempt, maxDev);
  return { correction_text: text, is_safe: true, target_joint: target };
}
