/**
 * Smart Yoga Posture Correction System (Project P05)
 * TypeScript interfaces matching the FastAPI backend schemas
 */

export interface FrameInput {
  angles: number[]; // Array of 15 joint angles
  world_angles?: number[]; // Optional: angles derived from MediaPipe's poseWorldLandmarks (metric-scale 3D)
  motion?: number; // Mean absolute angular velocity (deg/s) over the recent frame buffer
  calibration?: CalibrationProfile; // Digital Twin profile, enables personalised scoring
}

/**
 * "holding"       - body is still and the pose is recognised
 * "transitioning" - actively moving between poses (don't nag mid-flow)
 * "unrecognized"  - body is still, but the posture isn't one we can name
 * "unknown"       - no motion signal was sent
 */
export type MotionState = "holding" | "transitioning" | "unrecognized" | "unknown";

export interface FrameResponse {
  pose_id: string;
  correctness_score: number; // 0.0 to 1.0, universal ideal form
  deviations: { [jointName: string]: number }; // Deviations in degrees
  motion_state?: MotionState;
  personal_correctness_score?: number | null; // present when calibration was sent
  calibrated_deviations?: { [jointName: string]: number } | null;
}

export interface SequenceInput {
  coordinates: number[][]; // Shape [60, 99] (rolling window coordinates)
}

export interface SequenceResponse {
  // Always a plain pose name or "transition/unknown" -- never a prefixed
  // label -- so this field means the same thing under either checkpoint.
  sequence_pose: string;
  confidence: number;
  requires_static_fallback: boolean;
  // Present once the transition-aware sequence checkpoint is loaded. Lets the
  // UI say WHICH transition is under way instead of only that the pose is
  // unrecognised.
  sequence_kind?: "hold" | "transition" | "unrecognized";
  transition_from?: string | null;
  transition_to?: string | null;
}

export interface CorrectionInput {
  pose_id: string;
  deviations: { [jointName: string]: number };
  language?: "en" | "hi" | "bn";
  groq_api_key?: string;
  // Times this same cue was already given without the targeted joint moving.
  // Drives backend escalation: 0 plain, 1 quantified, 2+ back off.
  attempt?: number;
}

export interface CorrectionResponse {
  correction_text: string;
  is_safe: boolean;
  // Joint the cue is trying to move; the client measures THIS joint to decide
  // whether the cue worked.
  target_joint?: string | null;
}

export interface OcclusionInput {
  mp_landmarks: number[][]; // Shape [33, 4] -> [x, y, z, visibility]
  cliff_landmarks?: number[][]; // Optional shape [33, 3] -> [x, y, z]
}

export interface OcclusionResponse {
  fused_landmarks: number[][]; // Shape [33, 4]
  occluded_joints_recovered: string[];
  method_used: string;
}

export interface CalibrationProfile {
  [jointName: string]: {
    min: number;
    max: number;
    resting: number;
  };
}
