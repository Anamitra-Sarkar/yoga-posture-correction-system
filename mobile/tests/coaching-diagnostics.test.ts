import { describe, expect, it } from "vitest";

import { getCoachingDiagnostic } from "../lib/coaching-diagnostics";

describe("coaching diagnostics", () => {
  it("confirms a recognized selected target only when the score is strong", () => {
    const result = getCoachingDiagnostic("Warrior II", "warrior_2", "warrior_2", 1);
    expect(result).toMatchObject({ detectedPose: "Warrior II", isMismatch: false, safe: true });
  });

  it("flags a classifier mismatch without pretending the selected target was detected", () => {
    const result = getCoachingDiagnostic("Cobra", "cobra_pose", "lunge_pose", 0.26);
    expect(result).toMatchObject({ detectedPose: "Lunge Pose", isMismatch: true, safe: false });
    expect(result.correction).toContain("not Cobra");
  });

  it("explains transition output as an identifiable framing or hold issue", () => {
    const result = getCoachingDiagnostic("Plank", "plank", "transition/unknown", 0);
    expect(result).toMatchObject({ detectedPose: "Waiting for a reading", isMismatch: true, safe: false });
    expect(result.correction).toContain("not identifiable");
  });
});
