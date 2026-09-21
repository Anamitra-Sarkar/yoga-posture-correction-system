export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export function calculateAngle3D(a: Point3D, b: Point3D, c: Point3D): number {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
  const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);
  
  if (magBA === 0 || magBC === 0) return 0;
  
  let cosTheta = dot / (magBA * magBC);
  cosTheta = Math.max(-1, Math.min(1, cosTheta));
  
  return (Math.acos(cosTheta) * 180.0) / Math.PI;
}

// MediaPipe landmarks indices
export const SHOULDER_L = 11;
export const SHOULDER_R = 12;
export const ELBOW_L = 13;
export const ELBOW_R = 14;
export const WRIST_L = 15;
export const WRIST_R = 16;
export const HIP_L = 23;
export const HIP_R = 24;
export const KNEE_L = 25;
export const KNEE_R = 26;
export const ANKLE_L = 27;
export const ANKLE_R = 28;
export const HEEL_L = 29;
export const HEEL_R = 30;
export const NOSE = 0;

export function extractAnglesFromLandmarks(rawPts: Point3D[], zeroZ: boolean = true): number[] {
  if (rawPts.length < 31) return Array(15).fill(0);

  // MediaPipe's monocular z-depth estimate (on the default poseLandmarks) is
  // only reliable at the consistent camera distance/framing seen in demo
  // videos; on arbitrary real-world camera framing it degrades badly
  // (measured 0-3% real-world pose accuracy with z included, vs ~46% with it
  // dropped, confirmed across a 40-trial randomized threshold sweep). zeroZ
  // defaults to true to reproduce that proven 2D-only path. Pass zeroZ=false
  // for poseWorldLandmarks (metric-scale, separately calibrated 3D) so its
  // genuine depth is kept -- used as an independent second vote server-side,
  // not a replacement for the 2D path.
  const pts = rawPts.map((p) => ({ x: p.x, y: p.y, z: zeroZ ? 0 : p.z }));

  const shoulder_mid = {
    x: (pts[SHOULDER_L].x + pts[SHOULDER_R].x) / 2.0,
    y: (pts[SHOULDER_L].y + pts[SHOULDER_R].y) / 2.0,
    z: (pts[SHOULDER_L].z + pts[SHOULDER_R].z) / 2.0,
  };
  
  const hip_mid = {
    x: (pts[HIP_L].x + pts[HIP_R].x) / 2.0,
    y: (pts[HIP_L].y + pts[HIP_R].y) / 2.0,
    z: (pts[HIP_L].z + pts[HIP_R].z) / 2.0,
  };
  
  return [
    calculateAngle3D(pts[SHOULDER_L], pts[ELBOW_L], pts[WRIST_L]), // elbow_l
    calculateAngle3D(pts[SHOULDER_R], pts[ELBOW_R], pts[WRIST_R]), // elbow_r
    calculateAngle3D(pts[HIP_L], pts[SHOULDER_L], pts[ELBOW_L]), // shoulder_l
    calculateAngle3D(pts[HIP_R], pts[SHOULDER_R], pts[ELBOW_R]), // shoulder_r
    calculateAngle3D(pts[SHOULDER_L], pts[HIP_L], pts[KNEE_L]), // hip_l
    calculateAngle3D(pts[SHOULDER_R], pts[HIP_R], pts[KNEE_R]), // hip_r
    calculateAngle3D(pts[HIP_L], pts[KNEE_L], pts[ANKLE_L]), // knee_l
    calculateAngle3D(pts[HIP_R], pts[KNEE_R], pts[ANKLE_R]), // knee_r
    calculateAngle3D(pts[KNEE_L], pts[ANKLE_L], pts[HEEL_L]), // ankle_l
    calculateAngle3D(pts[KNEE_R], pts[ANKLE_R], pts[HEEL_R]), // ankle_r
    calculateAngle3D(pts[SHOULDER_L], pts[HIP_L], pts[HIP_R]), // trunk_l
    calculateAngle3D(pts[SHOULDER_R], pts[HIP_R], pts[HIP_L]), // trunk_r
    calculateAngle3D(pts[NOSE], shoulder_mid, hip_mid), // neck
    calculateAngle3D(pts[HIP_R], pts[HIP_L], pts[KNEE_L]), // hip_abduct_l
    calculateAngle3D(pts[HIP_L], pts[HIP_R], pts[KNEE_R]), // hip_abduct_r
  ];
}

/**
 * Global body orientation, which the 15 angle features cannot express.
 *
 * Every entry in FEATURE_NAMES is a RELATIVE joint angle (shoulder-hip-knee
 * and friends), so all of them are invariant to rotating the whole body.
 * Measured on the real-photo corpus, that blindness is not theoretical: the
 * rule engine called corpse "mountain_pose" 11 times out of 18, because a
 * person lying flat and a person standing upright produce nearly the same
 * 15-vector. No threshold change can fix it -- the information is simply not
 * in the feature vector.
 *
 * torsoIncline    degrees between shoulders->hips and image-down.
 *                 ~0 standing, ~90 lying or plank, ~180 inverted.
 *                 Measured medians: mountain 6, cobra 43, table_top 61,
 *                 triangle 71, plank 80, corpse 107, downward_dog 148.
 * legTorsoRatio   mean hip->ankle distance over shoulder->hip distance.
 *                 Separates cross-legged sitting (0.57) from every standing
 *                 pose (~1.2-1.4). Bounding-box aspect ratio does NOT do this
 *                 reliably: a standing person with arms down is just as tall
 *                 and narrow as someone seated facing the camera.
 *
 * Returns null on degenerate landmarks so the caller can simply omit the
 * field, which the backend treats exactly like an older client.
 */
export function computeOrientation(
  landmarks: { x: number; y: number }[],
): { torso_incline: number; leg_torso_ratio: number } | null {
  if (!landmarks || landmarks.length < 33) return null;
  const mid = (a: { x: number; y: number }, b: { x: number; y: number }) => ({
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
  });
  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.hypot(a.x - b.x, a.y - b.y);

  const shoulderMid = mid(landmarks[SHOULDER_L], landmarks[SHOULDER_R]);
  const hipMid = mid(landmarks[HIP_L], landmarks[HIP_R]);
  const torsoLen = dist(shoulderMid, hipMid);
  if (!Number.isFinite(torsoLen) || torsoLen < 1e-9) return null;

  // MediaPipe's y axis grows downward, so image-down is (0, +1) and the
  // inclination is the angle of the torso vector against it.
  const cos = (hipMid.y - shoulderMid.y) / torsoLen;
  const torsoIncline = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;

  const legLen =
    (dist(landmarks[ANKLE_L], landmarks[HIP_L]) +
      dist(landmarks[ANKLE_R], landmarks[HIP_R])) / 2;

  const ratio = legLen / torsoLen;
  if (!Number.isFinite(torsoIncline) || !Number.isFinite(ratio)) return null;
  return { torso_incline: torsoIncline, leg_torso_ratio: ratio };
}
