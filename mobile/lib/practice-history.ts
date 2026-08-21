import AsyncStorage from "@react-native-async-storage/async-storage";

export type PracticeRecord = {
  poseId: string;
  openedAt: string;
  completedAt?: string;
  score?: number;
  detectedPoseId?: string;
  durationSeconds?: number;
  correction?: string;
};

export const LAST_PRACTICE_KEY = "asana-ai-last-practice-v1";

export async function recordPractice(poseId: string): Promise<PracticeRecord> {
  const record = { poseId, openedAt: new Date().toISOString() };
  await AsyncStorage.setItem(LAST_PRACTICE_KEY, JSON.stringify(record));
  return record;
}

export async function loadLastPractice(): Promise<PracticeRecord | null> {
  const stored = await AsyncStorage.getItem(LAST_PRACTICE_KEY);
  if (!stored) return null;
  try {
    const parsed = JSON.parse(stored) as Partial<PracticeRecord>;
    if (typeof parsed.poseId !== "string" || typeof parsed.openedAt !== "string") return null;
    return {
      poseId: parsed.poseId,
      openedAt: parsed.openedAt,
      completedAt: typeof parsed.completedAt === "string" ? parsed.completedAt : undefined,
      score: typeof parsed.score === "number" ? parsed.score : undefined,
      detectedPoseId: typeof parsed.detectedPoseId === "string" ? parsed.detectedPoseId : undefined,
      durationSeconds: typeof parsed.durationSeconds === "number" ? parsed.durationSeconds : undefined,
      correction: typeof parsed.correction === "string" ? parsed.correction : undefined,
    };
  } catch {
    return null;
  }
}

export async function completePractice(record: Omit<PracticeRecord, "openedAt" | "completedAt"> & { openedAt?: string }): Promise<PracticeRecord> {
  const completed = { ...record, openedAt: record.openedAt ?? new Date().toISOString(), completedAt: new Date().toISOString() };
  await AsyncStorage.setItem(LAST_PRACTICE_KEY, JSON.stringify(completed));
  return completed;
}
