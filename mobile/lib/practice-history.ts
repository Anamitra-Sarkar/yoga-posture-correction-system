import AsyncStorage from "@react-native-async-storage/async-storage";

export type PracticeRecord = {
  poseId: string;
  openedAt: string;
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
    return { poseId: parsed.poseId, openedAt: parsed.openedAt };
  } catch {
    return null;
  }
}
