"""Parity check: frontend/src/utils/offlineCoach.ts (on-device coach) vs the Python rule engine it ports.

Generates thousands of cases (random angles, angles sitting exactly on every threshold, angles inside each pose's
bands, many orientation values, calibration profiles, motion values, correction languages/attempts), runs BOTH
implementations and requires identical results. Run from the repo root:   python3 backend/tools/offline_parity.py
Needs: python deps of the backend (numpy, requests), node, and frontend/node_modules (for tsc).
"""
import json, os, random, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.environ.pop("GROQ_API_KEY", None)  # stage-1 (template) path only: the LLM paraphrase is server-only
from app.utils.geometry import FEATURE_NAMES  # noqa: E402
from app.utils.rules_classifier import classify_pose, score_pose, _POSE_FEATURE_BANDS  # noqa: E402
from app.routers.pose import classify_motion_state, apply_calibration, correctness_from_deviations  # noqa: E402
from app.services.cascade import guided_report  # noqa: E402
from app.services.correction import generate_safe_correction, BIOMECHANICAL_TEMPLATES  # noqa: E402

rng = random.Random(20261005)
THRESH = [5, 20, 22, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 105, 110, 115, 120, 122, 125, 130, 135, 140, 150, 165, 180]
POSES = sorted(set(_POSE_FEATURE_BANDS) | {"transition/unknown", "not_a_pose"})


def rand_angles(kind):
    if kind == "uniform":
        return [rng.uniform(0, 180) for _ in FEATURE_NAMES]
    if kind == "boundary":
        return [min(180.0, max(0.0, rng.choice(THRESH) + rng.choice([0, 0, 0.001, -0.001, 0.5, -0.5, 1, -1]))) for _ in FEATURE_NAMES]
    pose = rng.choice([p for p, b in _POSE_FEATURE_BANDS.items() if b])
    a = {n: rng.uniform(0, 180) for n in FEATURE_NAMES}
    for n, lo, hi in _POSE_FEATURE_BANDS[pose]:
        a[n] = rng.uniform(max(0, lo - 8), min(180, hi + 8))
    return [a[n] for n in FEATURE_NAMES]


def rand_orientation():
    r = rng.random()
    if r < 0.3: return None
    if r < 0.4: return {}
    if r < 0.45: return {"torso_incline": rng.uniform(0, 180)}
    if r < 0.5: return {"torso_incline": None, "leg_torso_ratio": 1.0}
    if r < 0.75: return {"torso_incline": rng.uniform(0, 180), "leg_torso_ratio": rng.uniform(0.3, 1.6)}
    return {"torso_incline": rng.choice([0, 21.9, 22, 34.99, 35, 60, 60.01, 121.9, 122, 122.01, 179]), "leg_torso_ratio": rng.choice([0.3, 0.59, 0.6, 0.61, 0.89, 0.9, 1.0, 1.4])}


cases = []
for kind, n in (("uniform", 3500), ("boundary", 3500), ("band", 3000)):
    for _ in range(n):
        ang = rand_angles(kind)
        a = dict(zip(FEATURE_NAMES, ang))
        ori = rand_orientation()
        pose = classify_pose(a, ori)
        corr, devs = score_pose(pose, a)
        motion = rng.choice([None, 0.0, 3.0, 14.999, 15.0, 15.001, 40.0])
        c = {"angles": ang, "orientation": ori, "motion": motion, "pose": pose, "correctness": corr, "devs": devs,
             "motion_state": classify_motion_state(motion, pose)}
        if rng.random() < 0.5:  # a calibration profile covering a random subset of joints
            prof = {}
            for j in rng.sample(FEATURE_NAMES, rng.randint(1, 15)):
                lo = rng.uniform(0, 120); hi = lo + rng.uniform(5, 90)
                prof[j] = {"min": lo, "max": hi, "resting": (lo + hi) / 2}
            cal = apply_calibration(devs, a, prof)
            c["calibration"] = prof; c["cal_devs"] = cal; c["personal"] = correctness_from_deviations(cal, devs, corr)
        tgt = rng.choice(POSES)
        g = guided_report(tgt, pose, a); c["target"] = tgt; c["guided"] = g
        cases.append(c)

corr_cases = []
tmpl_poses = sorted(BIOMECHANICAL_TEMPLATES)
for _ in range(2500):
    pose = rng.choice(tmpl_poses + ["warrior_2", "not_a_pose"])
    joints = list(BIOMECHANICAL_TEMPLATES.get(pose, {}).keys()) + ["knee_l", "elbow_r"]
    devs = {n: 0.0 for n in FEATURE_NAMES}
    for j in rng.sample(FEATURE_NAMES, rng.randint(0, 5)) + rng.sample(joints, min(len(joints), rng.randint(0, 2))):
        devs[j] = rng.choice([0.0, 9.99, 10.0, 10.01, rng.uniform(0, 60), 12.5, 12.5, 2.5, 33.5])
    lang = rng.choice(["en", "hi", "bn", "xx"]); att = rng.choice([0, 1, 2, 3, 5])
    text, safe, tj = generate_safe_correction(pose, devs, lang, None, att)
    corr_cases.append({"pose": pose, "devs": devs, "lang": lang, "attempt": att, "text": text, "safe": safe, "joint": tj})

work = tempfile.mkdtemp(prefix="offline_parity_")
json.dump({"cases": cases, "corr": corr_cases}, open(os.path.join(work, "cases.json"), "w"), ensure_ascii=False)
fe = os.path.join(ROOT, "frontend")
subprocess.run([os.path.join(fe, "node_modules/.bin/tsc"), os.path.join(fe, "src/utils/offlineCoach.ts"), "--outDir", work, "--target", "es2019",
                "--module", "commonjs", "--skipLibCheck", "--moduleResolution", "node"], check=True)
open(os.path.join(work, "check.js"), "w").write(r'''
const { offlineFrame, offlineCorrection, classifyPose, anglesToDict } = require("./utils/offlineCoach.js");
const D = require("./cases.json"); const bad = []; const near = (a, b) => Math.abs(a - b) <= 1e-9 || (a === b);
let n = 0;
for (const c of D.cases) {
  n++;
  const req = { angles: c.angles, orientation: c.orientation === null ? undefined : c.orientation, motion: c.motion === null ? undefined : c.motion, calibration: c.calibration, target_pose: c.target };
  const r = offlineFrame(req);
  const errs = [];
  if (r.pose_id !== c.pose) errs.push(`pose ${r.pose_id} != ${c.pose}`);
  if (!near(r.correctness_score, c.correctness)) errs.push(`score ${r.correctness_score} != ${c.correctness}`);
  for (const k of Object.keys(c.devs)) if (!near(r.deviations[k], c.devs[k])) errs.push(`dev ${k}`);
  if (r.motion_state !== c.motion_state) errs.push(`motion ${r.motion_state} != ${c.motion_state}`);
  if (c.calibration) {
    if (!near(r.personal_correctness_score, c.personal)) errs.push(`personal ${r.personal_correctness_score} != ${c.personal}`);
    for (const k of Object.keys(c.cal_devs)) if (!near(r.calibrated_deviations[k], c.cal_devs[k])) errs.push(`caldev ${k}`);
  } else if (r.personal_correctness_score !== null) errs.push("personal should be null");
  const g = r.guided; if (!g || g.matches !== c.guided.matches || g.target_has_bands !== c.guided.target_has_bands || !near(g.target_correctness, c.guided.target_correctness)) errs.push("guided");
  if (errs.length) bad.push({ i: n, errs: errs.slice(0, 3) });
}
let cb = [];
for (const c of D.corr) {
  const r = offlineCorrection(c.pose, c.devs, c.lang, c.attempt);
  if (r.correction_text !== c.text || r.is_safe !== c.safe || (r.target_joint ?? null) !== (c.joint ?? null)) cb.push({ pose: c.pose, lang: c.lang, att: c.attempt, js: r.correction_text, py: c.text, jsj: r.target_joint, pyj: c.joint });
}
const poses = {}; D.cases.forEach(c => poses[c.pose] = (poses[c.pose] || 0) + 1);
console.log(JSON.stringify({ frame_cases: D.cases.length, frame_mismatches: bad.length, correction_cases: D.corr.length, correction_mismatches: cb.length, poses_covered: Object.keys(poses).length, pose_counts: poses, first_frame_mismatches: bad.slice(0, 5), first_correction_mismatches: cb.slice(0, 5) }));
''')
res = subprocess.run(["node", os.path.join(work, "check.js")], capture_output=True, text=True, cwd=work)
print(res.stdout.strip() or res.stderr.strip())
out = json.loads(res.stdout)
ok = out["frame_mismatches"] == 0 and out["correction_mismatches"] == 0
print("PARITY:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
