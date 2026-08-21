import AsyncStorage from "@react-native-async-storage/async-storage";

export type Preferences = {
  poseId: string;
  voiceEnabled: boolean;
  language: "en" | "hi" | "bn";
};

export const PREFERENCES_KEY = "asana-ai-preferences-v1";

export const DEFAULT_PREFERENCES: Preferences = {
  poseId: "warrior_2",
  voiceEnabled: true,
  language: "en",
};

export async function loadPreferences(): Promise<Preferences> {
  const stored = await AsyncStorage.getItem(PREFERENCES_KEY);
  if (!stored) return DEFAULT_PREFERENCES;
  try {
    return { ...DEFAULT_PREFERENCES, ...JSON.parse(stored) };
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

export async function savePreferences(preferences: Preferences) {
  await AsyncStorage.setItem(PREFERENCES_KEY, JSON.stringify(preferences));
}
