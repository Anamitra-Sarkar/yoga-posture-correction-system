#!/usr/bin/env bash
# Smoke-test the DEPLOYED backend. No local build, no model download, no GPU.
#
# Every check below was run against the live Space on 2026-09-19 and passed;
# the expected results are recorded so a future run can tell a regression from
# a change. Run it after any backend deploy.
#
#   bash backup/verify_live.sh
set -u
API="${1:-https://arko007-yoga-pose.hf.space}"
say() { printf '\n== %s\n' "$1"; }
post() { curl -s -m 120 -X POST "$API$1" -H 'Content-Type: application/json' -d "$2"; }

# A plausible mountain_pose: every joint near straight.
STILL='{"angles":[175,175,15,15,175,175,175,175,90,90,170,170,160,175,175],"motion":3.0,
        "calibration":{"knee_l":{"min":150,"max":185},"knee_r":{"min":150,"max":185}}}'
MOVING='{"angles":[175,175,15,15,175,175,175,175,90,90,170,170,160,175,175],"motion":120.0}'
OLDCLIENT='{"angles":[175,175,15,15,175,175,175,175,90,90,170,170,160,175,175]}'
# Angles far enough off that calibration has something to forgive.
IMPERFECT='{"angles":[95,140,70,30,120,60,95,150,70,110,95,140,95,80,120],"motion":2.0,
            "calibration":{"knee_l":{"min":80,"max":110},"elbow_l":{"min":85,"max":100}}}'

say "health  (expect status healthy; note /health is at the ROOT, not /api)"
curl -s -m 60 "$API/health"; echo

say "analyse_frame, still + recognised  (expect motion_state holding)"
post /api/analyse_frame "$STILL" | head -c 300; echo

say "analyse_frame, same angles at 120 deg/s  (expect motion_state transitioning)"
post /api/analyse_frame "$MOVING" | head -c 300; echo

say "analyse_frame, no motion field  (expect motion_state unknown, personal score null)"
post /api/analyse_frame "$OLDCLIENT" | head -c 300; echo

say "analyse_frame with calibration  (expect personal score > universal)"
post /api/analyse_frame "$IMPERFECT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
u, p = d['correctness_score'], d.get('personal_correctness_score')
print(f\"universal={u:.3f} personal={p}\")
cal = d.get('calibrated_deviations') or {}
forgiven = [j for j, v in d['deviations'].items() if v > 0 and cal.get(j, v) == 0.0]
print('joints forgiven by calibration:', forgiven or '(none)')
assert p is None or p >= u, 'personal score must never be below universal'
print('OK')"

say "orientation: SAME 15 angles, upright then lying (expect mountain_pose then corpse)"
ORI='[180,180,20,20,180,180,180,180,180,180,100,100,160,180,180]'
post /api/analyse_frame "{\"angles\":$ORI,\"orientation\":{\"torso_incline\":5.0,\"leg_torso_ratio\":1.3}}" \
  | python3 -c "import sys,json;print('  upright ->',json.load(sys.stdin)['pose_id'])"
post /api/analyse_frame "{\"angles\":$ORI,\"orientation\":{\"torso_incline\":100.0,\"leg_torso_ratio\":1.2}}" \
  | python3 -c "import sys,json;print('  lying   ->',json.load(sys.stdin)['pose_id'])"
cat <<'NOTE'
   These two requests carry IDENTICAL joint angles. All 15 features are
   relative angles and so are blind to whole-body rotation; without the
   orientation field both read as mountain_pose. If "lying" does not say
   corpse, the orientation path is not reaching the rule engine.
NOTE

say "correction escalation (expect: plain cue, then quantified, then back off)"
for A in 0 1 2; do
  printf '  attempt %s -> ' "$A"
  post /api/generate_correction \
    "{\"pose_id\":\"warrior_2\",\"deviations\":{\"knee_l\":24.0},\"language\":\"en\",\"attempt\":$A}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['correction_text'][:88],'| joint:',d.get('target_joint'))"
done

say "analyse_sequence  (60x99; sequence_kind is 'hold' until the transition model is wired in)"
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

printf '\n== done\n'
