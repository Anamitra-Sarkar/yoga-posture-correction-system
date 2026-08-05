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
