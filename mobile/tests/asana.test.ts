import { describe, expect, it } from "vitest";

import { DEFAULT_POSE_ID, displayPoseId, getPose, serviceCopy } from "../lib/asana";

describe("AsanaAI domain helpers", () => {
  it("resolves the default target pose", () => {
    expect(getPose(DEFAULT_POSE_ID).name).toBe("Warrior II");
  });

  it("turns service identifiers into readable result labels", () => {
    expect(displayPoseId("downward_dog")).toBe("Downward Dog");
    expect(displayPoseId(undefined)).toBe("Waiting for a reading");
  });

  it("never reports unavailable service states as a fabricated success", () => {
    expect(serviceCopy("offline").title).toBe("Coaching unavailable");
    expect(serviceCopy("available").title).toBe("Coaching is live");
  });
});
