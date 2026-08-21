import { displayPoseId } from "./asana";

export type CoachingDiagnostic = {
  detectedPose: string;
  isMismatch: boolean;
  safe: boolean;
  correction: string;
};

export function getCoachingDiagnostic(targetPoseName: string, targetPoseId: string, backendPoseId: string, score: number): CoachingDiagnostic {
  const isTransition = backendPoseId === "transition/unknown";
  const isMismatch = !isTransition && backendPoseId !== targetPoseId;
  const detectedPose = displayPoseId(backendPoseId);

  if (isTransition) {
    return {
      detectedPose,
      isMismatch: true,
      safe: false,
      correction: `Your body is in frame, but ${targetPoseName} is not identifiable yet. Step back, show head to feet, and hold the shape for a moment.`,
    };
  }

  if (isMismatch) {
    return {
      detectedPose,
      isMismatch: true,
      safe: false,
      correction: `The coach read ${detectedPose}, not ${targetPoseName}. Use the pose guide, then hold the target shape before checking form.`,
    };
  }

  return {
    detectedPose,
    isMismatch: false,
    safe: score >= 0.7,
    correction: score >= 0.7 ? "Alignment is on track. Keep breathing steadily." : "The target pose is in frame. Hold steady while the coach checks alignment.",
  };
}
