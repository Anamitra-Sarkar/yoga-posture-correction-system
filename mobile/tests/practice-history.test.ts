import { beforeEach, describe, expect, it, vi } from "vitest";

const storage = vi.hoisted(() => ({
  getItem: vi.fn(),
  setItem: vi.fn(),
}));

vi.mock("@react-native-async-storage/async-storage", () => ({ default: storage }));

import { LAST_PRACTICE_KEY, loadLastPractice, recordPractice } from "../lib/practice-history";

describe("AsanaAI practice history", () => {
  beforeEach(() => {
    storage.getItem.mockReset();
    storage.setItem.mockReset();
  });

  it("has no history until a user opens the live coach", async () => {
    storage.getItem.mockResolvedValue(null);
    await expect(loadLastPractice()).resolves.toBeNull();
  });

  it("records the real selected pose when live coaching is opened", async () => {
    const result = await recordPractice("cobra");
    expect(result.poseId).toBe("cobra");
    expect(storage.setItem).toHaveBeenCalledWith(LAST_PRACTICE_KEY, expect.stringContaining("cobra"));
  });

  it("rejects malformed stored session data instead of displaying it", async () => {
    storage.getItem.mockResolvedValue(JSON.stringify({ poseId: "cobra" }));
    await expect(loadLastPractice()).resolves.toBeNull();
  });
});
