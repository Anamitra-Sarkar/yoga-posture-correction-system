# AsanaAI: Video-Frame Pipeline + Model Retraining Plan (v2)

## ⚠️ RESUME NOTES — read first

User asked to stop active verification and pivot to planning because they're
leaving for college and shutting the terminal down. This file is the
hand-off. **Use Kaggle for all heavy compute and downloads, never the local
machine** (standing rule this whole project).

## Context: what triggered this

While doing a full verification pass, found that a **separate, later session**
(not this one) added a second native mobile app — `mobile/` (Expo/React
Native), distinct from the Capacitor WebView app at `frontend/android/`. Both
now coexist in the repo; additive, non-conflicting, doesn't touch the web app.

That session's own real-photo testing (`mobile/AGENT_HANDOFF.md`,
`mobile/image_pipeline_verification.md`) found `cobra_pose` misclassifying as
`lunge_pose` and `plank` reading as `transition/unknown` on real reference
images, against the THEN-live backend.

**RESOLVED, not a current bug**: re-ran those exact fixture images
(`mobile/test-assets/pose-pipeline/*.jpg|webp|jpeg`) through the CURRENT
`backend/app/utils/rules_classifier.py` + `geometry.py` locally (raw
MediaPipe → `extract_angles_from_landmarks` → `classify_pose`) and got
CORRECT results: cobra.jpg → `cobra_pose`, plank.jpeg → `plank`,
warrior-ii.jpeg → `warrior_2`. Then confirmed directly against the LIVE
production endpoint with cobra.jpg's exact real extracted angles
(`{"angles":[132.8,142.3,16.9,20.2,130.7,127.5,173.6,177.6,99.2,108.0,27.7,
152.7,177.7,102.9,79.8]}` → `POST /api/analyse_frame`) — live returned
`cobra_pose`, correctness `1.0`, matching local exactly. **Conclusion: the
`mobile/` session's bug report was against a stale/pre-Phase-B backend
deployment, not a current issue.** No code investigation needed here — this
was just a timing artifact of when that other session ran its test. Safe to
move straight to the video-pipeline work below without re-litigating this.

## The actual ask: new video-frame pipeline

User wants to explore a genuinely new pose-estimation data pipeline,
distinct from the existing single-image Wikimedia Commons approach used in
Phase B this session:

1. **Source real yoga videos** (not single photos) — e.g. YouTube yoga
   tutorial channels (respect ToS/licensing — prefer Creative Commons or
   already-permitted sources; the project's existing MLP/STGCN training data
   presumably has a documented source, check `README.md` / paper's dataset
   section for what's already licensed and reuse that pipeline/source
   first rather than re-negotiating licensing from scratch).
2. **Extract raw video frames** at a reasonable sample rate (e.g. 1-5 fps,
   avoid near-duplicate frames) — this is a real "big file" / heavy-compute
   step (video download + decode), must run on **Kaggle**, never locally.
3. **Re-run MediaPipe pose extraction on every sampled frame** — both the
   2D-zeroed path and the `pose_world_landmarks` path (dual signal, same as
   Phase A1 this session), producing a much larger and more diverse labeled
   angle-feature dataset than the ~35-47 single images used so far per pose.
4. **Purpose of this larger dataset** — two possible directions, decide which
   before implementing:
   - (a) **Better rule-threshold calibration**: replace the current
     hand-tuned `_POSE_FEATURE_BANDS` in `rules_classifier.py` (derived from
     ~7-12 images per pose) with bands/thresholds fit statistically from
     hundreds of real frames per pose — same rule-based architecture, just
     far better-calibrated boundaries. Lower risk, consistent with "don't
     change approach" instruction from earlier this session.
   - (b) **Retrain the MLP** on this larger, more realistic frame dataset
     instead of whatever dataset it currently uses (documented in the paper
     as achieving ~90% validation but only ~0-6% real-world accuracy — a
     classic train/real-world distribution-mismatch problem that MORE
     REAL-WORLD-DISTRIBUTION training data could genuinely fix, unlike the
     2D→3D lifter which was a NO-GO this session for an unrelated reason).
     Higher effort, higher potential upside, and is genuine "improve the
     model" work the user explicitly asked for today ("better metrics",
     "more pose capability"). **This is likely the more valuable of the two
     — the real-world/validation accuracy GAP is the single biggest known
     weakness in the whole system, bigger than any individual pose's
     threshold tuning.**
5. **Expand pose coverage while at it** — with a real per-frame dataset,
   revisit `warrior_1` and `chair_pose` (both stayed disabled this session
   at 18.3%/~0% for lack of good real data, not necessarily unfixable) plus
   any of the 8 originally-templated poses not yet live.

## Concrete Kaggle-based next steps (in order)

1. Identify/confirm a licensable video source (check existing project
   dataset docs before sourcing anything new).
2. Kaggle notebook: download video(s) → sample frames → run MediaPipe
   (2D + world) → dump a structured dataset (per-frame angles + pose label +
   source video timestamp) as a compact artifact (CSV/Parquet, not raw
   video/images) pulled back for local inspection only in small form.
3. Decide (a) rule recalibration vs (b) MLP retrain — probably do (a) first
   as a fast, low-risk win, then evaluate whether (b) is worth the larger
   effort based on how much headroom is left after (a).
4. Same non-negotiable discipline as every phase this session: validate
   real-world accuracy with a proper held-out test set and randomized
   threshold sweep before promoting anything to `DISABLED_POSES` changes or
   overwriting a live checkpoint; report honest numbers, including negative
   results, exactly like Phase A2's 2D→3D lifter NO-GO and Phase B's
   chair_pose/warrior_1 stay-disabled calls this session.

## State of everything else at hand-off (all verified working)

- Backend hybrid classifier (MLP + 2D-rules + world-landmarks 3-way vote via
  `hybrid_classify` in `backend/app/utils/rules_classifier.py`): live poses
  are warrior_2, mountain_pose, cobra_pose, tree_pose, plank, downward_dog;
  chair_pose and warrior_1 stay disabled (honestly, low real-world accuracy).
- STGCN: live checkpoint NOT promoted to the newer `_v2` (no validation data
  existed to prove `_v2` is actually better — structurally sound, evaluated
  on Kaggle, correctly not promoted on structural checks alone).
- Frontend: Vercel deployment confirmed READY on latest commit at last check.
- Two Android apps now coexist: `frontend/android/` (Capacitor WebView,
  CI-built debug APK confirmed compiling green on GitHub Actions after
  fixing 4 real CI issues) and `mobile/` (Expo/React Native native app, own
  test suite, 16 passing per its own handoff doc, not independently
  re-verified by this session).
- Git: local and remote were in sync as of the last check (`origin/main` at
  commit `6e1d945`) after fast-forward-pulling the two `mobile/`-adding
  commits this session found and merged in.
