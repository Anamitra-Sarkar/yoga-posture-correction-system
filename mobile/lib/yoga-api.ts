import type { Landmark } from "@/lib/pose-geometry";

const API_BASE_URL = "https://arko007-yoga-pose.hf.space/api";

export type MotionState = "holding" | "transitioning" | "unrecognized" | "unknown";

export type CalibrationProfile = {
  [joint: string]: { min: number; max: number; resting?: number };
};

export type FrameResult = {
  pose_id: string;
  correctness_score: number;
  deviations: Record<string, number>;
  /** holding | transitioning | unrecognized | unknown */
  motion_state?: MotionState;
  /** present only when a calibration profile was sent */
  personal_correctness_score?: number | null;
  calibrated_deviations?: Record<string, number> | null;
};

export type CorrectionResult = {
  correction_text: string;
  is_safe: boolean;
  /** joint the cue is trying to move; needed to measure whether it worked */
  target_joint?: string | null;
};

export type OcclusionResult = {
  fused_landmarks: number[][];
  occluded_joints_recovered: string[];
  method_used: string;
};

async function request<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`Coaching service returned ${response.status}`);
  return response.json() as Promise<T>;
}

export function recoverOcclusion(landmarks: Landmark[]) {
  return request<OcclusionResult>("/occlusion_recovery", {
    mp_landmarks: landmarks.map(({ x, y, z, visibility }) => [x, y, z, visibility]),
  });
}

/**
 * `motion` is mean absolute angular velocity in deg/s over the recent frame
 * buffer. Sending it is what lets the backend tell a genuine HOLD apart from a
 * TRANSITION between poses -- without it every unrecognised frame looks the
 * same as a moving one, which is the ambiguity the old single
 * "transition/unknown" bucket could never resolve.
 *
 * `calibration` opts into a second, personalised correctness score computed
 * server-side, so this client gets Digital Twin scoring without duplicating
 * the logic.
 */
export function analyseFrame(
  angles: number[],
  opts?: { motion?: number; calibration?: CalibrationProfile },
) {
  return request<FrameResult>("/analyse_frame", {
    angles,
    ...(opts?.motion === undefined ? {} : { motion: opts.motion }),
    ...(opts?.calibration ? { calibration: opts.calibration } : {}),
  });
}

/**
 * `attempt` is how many times this same cue has already been delivered
 * WITHOUT the targeted joint measurably improving. The backend escalates on
 * it: 0 gives the plain cue, 1 adds the measured magnitude, 2+ stops asking
 * for more and backs the user out of the shape. Repeating an identical
 * sentence at someone who did not respond is both useless and, when the cue
 * is "go deeper", the mechanism behind the injuries this app exists to avoid.
 */
export function generateCorrection(
  poseId: string,
  deviations: Record<string, number>,
  language: "en" | "hi" | "bn",
  attempt = 0,
) {
  return request<CorrectionResult>("/generate_correction", {
    pose_id: poseId,
    deviations,
    language,
    attempt,
  });
}
