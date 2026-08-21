export type Landmark = { x: number; y: number; z: number; visibility: number };

type Point3D = Pick<Landmark, "x" | "y" | "z">;

export const LANDMARK_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [23, 24],
  [11, 23], [12, 24], [23, 25], [25, 27], [24, 26], [26, 28],
];

export const SKELETON_NODES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

export const JOINT_NAMES = [
  "elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r",
  "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck", "hip_abduct_l", "hip_abduct_r",
];

function calculateAngle3D(a: Point3D, b: Point3D, c: Point3D): number {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magnitudeBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
  const magnitudeBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);
  if (!magnitudeBA || !magnitudeBC) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (magnitudeBA * magnitudeBC)))) * (180 / Math.PI);
}

export function extractAnglesFromLandmarks(landmarks: Landmark[]): number[] {
  if (landmarks.length < 31) return Array(15).fill(0);
  const points = landmarks.map(({ x, y }) => ({ x, y, z: 0 }));
  const shoulderMid = { x: (points[11].x + points[12].x) / 2, y: (points[11].y + points[12].y) / 2, z: 0 };
  const hipMid = { x: (points[23].x + points[24].x) / 2, y: (points[23].y + points[24].y) / 2, z: 0 };
  return [
    calculateAngle3D(points[11], points[13], points[15]), calculateAngle3D(points[12], points[14], points[16]),
    calculateAngle3D(points[23], points[11], points[13]), calculateAngle3D(points[24], points[12], points[14]),
    calculateAngle3D(points[11], points[23], points[25]), calculateAngle3D(points[12], points[24], points[26]),
    calculateAngle3D(points[23], points[25], points[27]), calculateAngle3D(points[24], points[26], points[28]),
    calculateAngle3D(points[25], points[27], points[29]), calculateAngle3D(points[26], points[28], points[30]),
    calculateAngle3D(points[11], points[23], points[24]), calculateAngle3D(points[12], points[24], points[23]),
    calculateAngle3D(points[0], shoulderMid, hipMid), calculateAngle3D(points[24], points[23], points[25]),
    calculateAngle3D(points[23], points[24], points[26]),
  ];
}

export function landmarksFromRows(rows: number[][]): Landmark[] {
  return rows.map(([x, y, z, visibility = 0]) => ({ x, y, z, visibility }));
}
