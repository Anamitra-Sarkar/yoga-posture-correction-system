import { beforeEach, describe, expect, it, vi } from "vitest";

const storage = vi.hoisted(() => ({
  getItem: vi.fn(),
  setItem: vi.fn(),
}));

vi.mock("@react-native-async-storage/async-storage", () => ({
  default: storage,
}));

import { DEFAULT_PREFERENCES, loadPreferences, PREFERENCES_KEY, savePreferences } from "../lib/preferences";

describe("AsanaAI local preferences", () => {
  beforeEach(() => {
    storage.getItem.mockReset();
    storage.setItem.mockReset();
  });

  it("returns calm defaults when a user has not saved preferences", async () => {
    storage.getItem.mockResolvedValue(null);
    await expect(loadPreferences()).resolves.toEqual(DEFAULT_PREFERENCES);
  });

  it("merges an existing stored preference safely", async () => {
    storage.getItem.mockResolvedValue(JSON.stringify({ poseId: "cobra", language: "bn" }));
    await expect(loadPreferences()).resolves.toEqual({ ...DEFAULT_PREFERENCES, poseId: "cobra", language: "bn" });
  });

  it("serializes changes using the application-specific storage key", async () => {
    const preferences = { poseId: "tree", language: "hi" as const, voiceEnabled: false };
    await savePreferences(preferences);
    expect(storage.setItem).toHaveBeenCalledWith(PREFERENCES_KEY, JSON.stringify(preferences));
  });
});
