import { describe, expect, it } from "vitest";

import { extractAnglesFromLandmarks, landmarksFromRows } from "../lib/pose-geometry";
import { getPose, normalizePoseId, POSES } from "../lib/asana";

describe("native yoga pose parity", () => {
  it("keeps only the six browser-audited target poses", () => {
    expect(POSES.map((pose) => pose.id)).toEqual(["warrior_2", "cobra_pose", "mountain_pose", "tree_pose", "plank", "downward_dog"]);
  });

  it("normalizes existing local preferences to the backend-compatible pose IDs", () => {
    expect(normalizePoseId("cobra")).toBe("cobra_pose");
    expect(getPose("mountain").id).toBe("mountain_pose");
  });

  it("produces the 15-angle backend feature vector from a full landmark frame", () => {
    const landmarks = landmarksFromRows(Array.from({ length: 33 }, (_, index) => [index / 33, index / 34, 0, 1]));
    const angles = extractAnglesFromLandmarks(landmarks);
    expect(angles).toHaveLength(15);
    expect(angles.every((value) => Number.isFinite(value))).toBe(true);
  });
});
