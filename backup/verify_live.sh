#!/usr/bin/env bash
# Smoke-test the DEPLOYED backend. No local build, no model download, no GPU.
#
# Re-written 2026-10-07: the first version (2026-09-19) pre-dated the pose cascade, so two checks passed while testing nothing.
# Every check is now an assertion (PASS/FAIL lines, non-zero exit on failure). Run it after any backend deploy.
#
#   bash backup/verify_live.sh
set -u
API="${1:-https://arko007-yoga-pose.hf.space}"
say() { printf '\n== %s\n' "$1"; }
post() { curl -s -m 120 -X POST "$API$1" -H 'Content-Type: application/json' -d "$2"; }

FAILS=0
check() { if [ "$1" = "0" ]; then echo "  PASS: $2"; else echo "  FAIL: $2"; FAILS=$((FAILS+1)); fi; }

# REAL 15-angle frames (+ body orientation), computed with backend/app/utils/geometry.py from hand-built skeletons.
# Why not made-up vectors: with the pose cascade ON (production), the gate rejects an implausible vector as "not one of
# my poses" and everything downstream (score, deviations, calibration) is then zero -- the old version of this script
# (2026-09-19) did exactly that and two of its checks passed while testing nothing.
TREE='[124.33,124.33,170.362,170.362,175.601,142.866,180.0,77.248,180.0,154.983,94.399,94.399,180.0,90.0,122.735]'
TREE_ORI='{"torso_incline":0.0,"leg_torso_ratio":1.118}'
UPRIGHT='[176.269,176.269,12.366,12.366,175.764,175.764,180.0,180.0,180.0,180.0,94.236,94.236,180.0,90.0,90.0]'   # named seated_staff; its hip deviations come from rule bands
UPRIGHT_ORI='{"torso_incline":0.0,"leg_torso_ratio":1.407}'
STILL="{\"angles\":$TREE,\"orientation\":$TREE_ORI,\"motion\":3.0}"
MOVING="{\"angles\":$TREE,\"orientation\":$TREE_ORI,\"motion\":120.0}"
OLDCLIENT="{\"angles\":$TREE}"

say "health  (expect status healthy; note /health is at the ROOT, not /api)"
curl -s -m 60 "$API/health" | tee /tmp/_h.json; echo
python3 -c "import json;assert json.load(open('/tmp/_h.json'))['status']=='healthy'"; check $? "status healthy"; rm -f /tmp/_h.json

say "analyse_frame, still Tree frame  (expect pose tree_pose, motion_state holding, cascade active)"
post /api/analyse_frame "$STILL" > /tmp/_f.json
python3 -c "
import json; d=json.load(open('/tmp/_f.json'))
print('  pose', d['pose_id'], '| score', round(d['correctness_score'],3), '| motion', d['motion_state'], '| cascade', d.get('cascade'))
assert d['pose_id']=='tree_pose' and d['motion_state']=='holding' and d['cascade']['active'] is True and d['cascade']['gated'] is False"; check $? "tree_pose, holding, cascade active and gate open"

say "Tree has no angle bands  (expect NO per-joint deviations: deviations_source none_no_bands, every deviation 0)"
python3 -c "
import json; d=json.load(open('/tmp/_f.json'))
print('  deviations_source', d['cascade'].get('deviations_source'), '| max deviation', max(d['deviations'].values()))
assert d['cascade'].get('deviations_source')=='none_no_bands' and max(d['deviations'].values())==0.0"; check $? "no invented per-joint deviations for a bandless pose"

say "analyse_frame, same angles at 120 deg/s  (expect motion_state transitioning)"
post /api/analyse_frame "$MOVING" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  motion',d['motion_state']);assert d['motion_state']=='transitioning'"; check $? "transitioning"

say "analyse_frame, old client (no motion, no calibration)  (expect motion_state unknown, personal score null)"
post /api/analyse_frame "$OLDCLIENT" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  motion',d['motion_state'],'| personal',d.get('personal_correctness_score'));assert d['motion_state']=='unknown' and d.get('personal_correctness_score') is None"; check $? "unknown, no personal score"

say "calibration  (a profile that CONTAINS a deviating joint's angle must raise the personal score; one that does not must change nothing)"
python3 - "$API" "$UPRIGHT" "$UPRIGHT_ORI" <<'PY'
import sys, json, urllib.request
api, angles, ori = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
N = ["elbow_l","elbow_r","shoulder_l","shoulder_r","hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r","trunk_l","trunk_r","neck","hip_abduct_l","hip_abduct_r"]
def call(cal=None):
    body = {"angles": angles, "orientation": ori, "motion": 0.0}
    if cal is not None: body["calibration"] = cal
    r = urllib.request.Request(api + "/api/analyse_frame", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())
base = call(); u = base["correctness_score"]; devs = {k: v for k, v in base["deviations"].items() if v > 0}
print("  pose", base["pose_id"], "| universal", round(u, 3), "| deviating joints", {k: round(v) for k, v in devs.items()})
assert devs, "no deviating joint to forgive: the frame was gated or rejected (cascade gate) -- update the fixture"
j = max(devs, key=devs.get); a = angles[N.index(j)]
def prof(cover): return {n: ({"min": round(angles[i]) - 10, "max": round(angles[i]) + 10} if n in cover else {"min": 0, "max": 1}) for i, n in enumerate(N)}
cov = call(prof({j})); non = call(prof(set()))
print("  covering %s: personal %.3f (calibrated deviation %.1f) | not covering: personal %.3f | universal unchanged: %s" % (j, cov["personal_correctness_score"], cov["calibrated_deviations"][j], non["personal_correctness_score"], abs(cov["correctness_score"] - u) < 1e-9))
assert cov["personal_correctness_score"] > u and cov["calibrated_deviations"][j] == 0.0
assert abs(non["personal_correctness_score"] - u) < 1e-9 and abs(cov["correctness_score"] - u) < 1e-9
PY
check $? "calibration forgives only joints inside the personal range; universal score never changes"

say "orientation field  (accepted, but NOT used while the pose cascade is on: it only feeds the rule engine in basic mode and the non-cascade fallback)"
post /api/analyse_frame "{\"angles\":$TREE,\"orientation\":{\"torso_incline\":100.0,\"leg_torso_ratio\":1.2}}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('  upright vs lying orientation both give pose:',d['pose_id'],'| cascade active:',d['cascade']['active']);assert d['pose_id']=='tree_pose'"; check $? "orientation ignored by the cascade (same pose); see backend/tests/test_orientation_path.py for the fallback path"

say "correction escalation (expect: plain cue, then quantified, then back off)"
for A in 0 1 2; do
  printf '  attempt %s -> ' "$A"
  post /api/generate_correction \
    "{\"pose_id\":\"warrior_2\",\"deviations\":{\"knee_l\":24.0},\"language\":\"en\",\"attempt\":$A}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['correction_text'][:88],'| joint:',d.get('target_joint'))"
done

say "analyse_sequence  (60x99; sequence_kind is 'hold': stgcn_target_v1 has no named transitions)"
python3 -c "
import json, random
random.seed(3)
base = []
for i in range(33): base += [0.5+0.01*i, 0.1+0.025*i, 0.0]
print(json.dumps({'coordinates': [[v+random.gauss(0,0.002) for v in base] for _ in range(60)]}))" \
  > /tmp/_seq.json
curl -s -m 120 -X POST "$API/api/analyse_sequence" -H 'Content-Type: application/json' \
  -d @/tmp/_seq.json; echo
rm -f /tmp/_seq.json

say "generate_correction en + hi"
post /api/generate_correction '{"pose_id":"warrior_2","deviations":{"knee_l":22.0},"language":"en"}'; echo
post /api/generate_correction '{"pose_id":"warrior_2","deviations":{"knee_l":22.0},"language":"hi"}'; echo
cat <<'NOTE'
   If the English text is exactly
     "Bend your left knee more to bring it directly over your ankle."
   the LLM paraphrase is NOT running -- that string is the Stage-1 template.
   Falling back to it is safe and correct, which is precisely why the failure
   is invisible. Check the Space logs for the "Groq correction call returned"
   warning, which now records the status and body.
NOTE

say "occlusion_recovery  (left knee at 0.05 visibility; expect only left_knee recovered)"
python3 -c "
import json
lm = [[0.5,0.5,0.0,0.9] for _ in range(33)]
lm[25] = [0.45,0.75,0.0,0.05]
print(json.dumps({'mp_landmarks': lm}))" > /tmp/_occ.json
curl -s -m 120 -X POST "$API/api/occlusion_recovery" -H 'Content-Type: application/json' \
  -d @/tmp/_occ.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('recovered:', d['occluded_joints_recovered'], '| method:', d['method_used'],
      '| landmarks:', len(d['fused_landmarks']))"
rm -f /tmp/_occ.json

printf '\n== done: %s check(s) failed\n' "$FAILS"
[ "$FAILS" = "0" ]
