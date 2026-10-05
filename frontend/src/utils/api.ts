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
const getApiUrl = () => {
  if (typeof window !== "undefined" && (window as any).customApiUrl) {
    return (window as any).customApiUrl;
  }
  return process.env.NEXT_PUBLIC_YOGA_API_URL || "http://localhost:8000/api";
};

export class ApiError extends Error {
  kind: "timeout" | "network" | "http";
  status?: number;
  constructor(kind: "timeout" | "network" | "http", message: string, status?: number) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

/** POST JSON with a hard timeout: a server that is asleep or unreachable must fail fast, not hang the whole app. */
async function post<T>(path: string, data: unknown, timeoutMs: number): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const response = await fetch(`${getApiUrl()}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
      signal: ctrl.signal,
    });
    if (!response.ok) throw new ApiError("http", `${path} failed: ${response.status} ${response.statusText}`, response.status);
    return (await response.json()) as T;
  } catch (e: any) {
    if (e instanceof ApiError) throw e;
    if (e?.name === "AbortError") throw new ApiError("timeout", `${path} timed out after ${timeoutMs} ms`);
    throw new ApiError("network", `${path} could not be reached`);
  } finally {
    clearTimeout(timer);
  }
}

export const analyseFrame = (data: FrameInput) => post<FrameResponse>("analyse_frame", data, 9000);
export const analyseSequence = (data: SequenceInput) => post<SequenceResponse>("analyse_sequence", data, 9000);
export const generateCorrection = (data: CorrectionInput) => post<CorrectionResponse>("generate_correction", data, 15000);
export const recoverOcclusion = (data: OcclusionInput) => post<OcclusionResponse>("occlusion_recovery", data, 9000);
