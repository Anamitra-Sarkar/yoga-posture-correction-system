const fs = require("fs");

const source = JSON.parse(fs.readFileSync("/tmp/asana-pose-fixture-results.json", "utf8"));
const base = "https://arko007-yoga-pose.hf.space/api";

function angle(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y };
  const bc = { x: c.x - b.x, y: c.y - b.y };
  const dot = ba.x * bc.x + ba.y * bc.y;
  const first = Math.hypot(ba.x, ba.y);
  const second = Math.hypot(bc.x, bc.y);
  return !first || !second ? 0 : Math.acos(Math.max(-1, Math.min(1, dot / (first * second)))) * 180 / Math.PI;
}

function extractAngles(landmarks) {
  const points = landmarks.map(({ x, y }) => ({ x, y }));
  const shoulderMid = { x: (points[11].x + points[12].x) / 2, y: (points[11].y + points[12].y) / 2 };
  const hipMid = { x: (points[23].x + points[24].x) / 2, y: (points[23].y + points[24].y) / 2 };
  return [angle(points[11], points[13], points[15]), angle(points[12], points[14], points[16]), angle(points[23], points[11], points[13]), angle(points[24], points[12], points[14]), angle(points[11], points[23], points[25]), angle(points[12], points[24], points[26]), angle(points[23], points[25], points[27]), angle(points[24], points[26], points[28]), angle(points[25], points[27], points[29]), angle(points[26], points[28], points[30]), angle(points[11], points[23], points[24]), angle(points[12], points[24], points[23]), angle(points[0], shoulderMid, hipMid), angle(points[24], points[23], points[25]), angle(points[23], points[24], points[26])];
}

async function post(path, payload) {
  const response = await fetch(`${base}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

async function run() {
  const output = [];
  for (const fixture of source) {
    if (fixture.landmarks !== 33) { output.push({ expected: fixture.expected, landmark_status: "incomplete", landmark_count: fixture.landmarks }); continue; }
    try {
      const recovery = await post("/occlusion_recovery", { mp_landmarks: fixture.coordinates });
      const fused = recovery.fused_landmarks.map(([x, y, z, visibility]) => ({ x, y, z, visibility }));
      const analysis = await post("/analyse_frame", { angles: extractAngles(fused) });
      output.push({ expected: fixture.expected, landmark_status: "complete", recovered_joints: recovery.occluded_joints_recovered, analysis });
    } catch (error) { output.push({ expected: fixture.expected, landmark_status: "complete", error: error.message }); }
  }
  fs.writeFileSync("/tmp/asana-backend-fixture-check.json", JSON.stringify(output, null, 2));
  console.log(JSON.stringify(output, null, 2));
}

run().catch((error) => { console.error(error); process.exit(1); });
