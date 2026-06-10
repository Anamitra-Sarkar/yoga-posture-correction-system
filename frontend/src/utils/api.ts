import {
  FrameInput,
  FrameResponse,
  SequenceInput,
  SequenceResponse,
  CorrectionInput,
  CorrectionResponse,
  OcclusionInput,
  OcclusionResponse,
} from "../types/yoga";

// In production, point to your Hugging Face Space URL (e.g. https://arko007-yoga-posture-models.hf.space)
const API_BASE_URL = process.env.NEXT_PUBLIC_YOGA_API_URL || "http://localhost:8000/api";

export async function analyseFrame(data: FrameInput): Promise<FrameResponse> {
  const response = await fetch(`${API_BASE_URL}/analyse_frame`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`Frame analysis API error: ${response.statusText}`);
  }
  return response.json();
}

export async function analyseSequence(data: SequenceInput): Promise<SequenceResponse> {
  const response = await fetch(`${API_BASE_URL}/analyse_sequence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`Sequence analysis API error: ${response.statusText}`);
  }
  return response.json();
}

export async function generateCorrection(data: CorrectionInput): Promise<CorrectionResponse> {
  const response = await fetch(`${API_BASE_URL}/generate_correction`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`Correction API error: ${response.statusText}`);
  }
  return response.json();
}

export async function recoverOcclusion(data: OcclusionInput): Promise<OcclusionResponse> {
  const response = await fetch(`${API_BASE_URL}/occlusion_recovery`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(`Occlusion recovery API error: ${response.statusText}`);
  }
  return response.json();
}
