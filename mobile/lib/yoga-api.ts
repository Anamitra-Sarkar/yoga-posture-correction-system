import type { Landmark } from "@/lib/pose-geometry";

const API_BASE_URL = "https://arko007-yoga-pose.hf.space/api";

export type FrameResult = { pose_id: string; correctness_score: number; deviations: Record<string, number> };
export type CorrectionResult = { correction_text: string; is_safe: boolean };
export type OcclusionResult = { fused_landmarks: number[][]; occluded_joints_recovered: string[]; method_used: string };

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

export function analyseFrame(angles: number[]) {
  return request<FrameResult>("/analyse_frame", { angles });
}

export function generateCorrection(poseId: string, deviations: Record<string, number>, language: "en" | "hi" | "bn") {
  return request<CorrectionResult>("/generate_correction", { pose_id: poseId, deviations, language });
}
