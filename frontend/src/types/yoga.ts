/**
 * Smart Yoga Posture Correction System (Project P05)
 * TypeScript interfaces matching the FastAPI backend schemas
 */

export interface FrameInput {
  angles: number[]; // Array of 15 joint angles
}

export interface FrameResponse {
  pose_id: string;
  correctness_score: number; // 0.0 to 1.0
  deviations: { [jointName: string]: number }; // Deviations in degrees
}

export interface SequenceInput {
  coordinates: number[][]; // Shape [60, 99] (rolling window coordinates)
}

export interface SequenceResponse {
  sequence_pose: string;
  confidence: number;
  requires_static_fallback: boolean;
}

export interface CorrectionInput {
  pose_id: string;
  deviations: { [jointName: string]: number };
  language?: "en" | "hi" | "bn";
  groq_api_key?: string;
}

export interface CorrectionResponse {
  correction_text: string;
  is_safe: boolean;
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
