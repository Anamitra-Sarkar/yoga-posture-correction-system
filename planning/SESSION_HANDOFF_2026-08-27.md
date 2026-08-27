# AsanaAI Session Handoff — 2026-08-27

## ⚠️ READ THIS FIRST ON RESUME

Two things may still be running on Kaggle when you resume — check their real
status directly (`kaggle kernels status <slug>`), don't trust any prior
session's self-report:

1. **`anamitrasarkar007/asanaai-mlp-zeroz-retrain`** (private, GPU) — MLP
   retrain + real-photo validation. Was `RUNNING` as of this doc's last edit.
   Pull output with `kaggle kernels output anamitrasarkar007/asanaai-mlp-zeroz-retrain -p <dir>`
   once `COMPLETE`. It reports the new checkpoint's isolated real-world MLP
   accuracy vs the documented ~0-6% baseline. **Nothing has been promoted to
   the live HF repo (`Arko007/yoga-posture-models`) — that decision is still
   pending on this result.** The trained checkpoint uploads under a
   versioned filename (`mlp_3head_model_v2.pth`-style), not the live
   filename, so there's no accidental-overwrite risk even if you forget to
   check before it finishes.
2. **`anamitrasarkar007/asanaai-video-pipeline-dual-signal`** (or similarly
   named — check `kaggle kernels list -m --user anamitrasarkar007 --page-size 100 | grep -i video`
   if this exact slug wasn't used) — the video-frame dual-signal (2D +
   world-landmarks) extraction + bone-length-consistency correction
   pipeline, launched proactively same session per user request ("any kaggle
   cloud stuff left, start immediate"). Check status/pull output the same way.

## What actually happened this session (chronological, condensed)

1. **Verification pass requested** ("check the whole repo properly, I have
   updated many things"). Discovered a separate, later Claude Code session
   (commits `8da0df8`/`6e1d945`, 2026-08-21, not this session) had added a
   **second native Android app** at `mobile/` (Expo/React Native), distinct
   from the Capacitor WebView app this project built earlier at
   `frontend/android/`. Both coexist, additive, non-conflicting. Local clone
   was 2 commits behind `origin/main` — pulled clean.
2. **Three parallel Explore agents** surveyed backend / frontend+Capacitor /
   mobile for issues. Frontend and mobile came back clean (only minor
   hygiene notes). Backend audit found real, concrete bugs (below).
3. **User redirected mid-investigation**: "plan the model training and data
   part... solve that 3D joint problem." A Plan agent designed a video-frame
   pipeline + bone-length-consistency correction approach (full design
   preserved below, Section "Priority 4"). User then added explicit urgency
   ("work fast, going to college, get the model training/data part done
   first") — plan was finalized and approved via ExitPlanMode, then executed.

## Real bugs found and FIXED this session (all committed + pushed to `origin/main`)

1. **`warrior_1`/`warrior_2` branch-order bug** (commit `3b4babf`) — in
   `backend/app/utils/rules_classifier.py::classify_pose()`, the disabled
   `warrior_1` branch was checked *before* the live `warrior_2` branch, with
   an identical leg condition and an overlapping arm band (shoulder ∈
   (110°,125°]). Real warrior_2 attempts in that overlap were silently
   thrown away as `transition/unknown`. Fixed by reordering (warrior_2
   first); verified with synthetic-angle regression tests covering the
   overlap zone plus all 6 other live/disabled poses, zero regressions.
2. **The project's biggest known problem, root-caused**: the MLP's training
   feature pipeline (`/home/anamitra/Projects_and_Code/Scripts_and_Source/extract_features_safe.py`)
   computed angle features from MediaPipe's raw, **un-zeroed** z-coordinate,
   while the live rule engine and inference path
   (`backend/app/utils/geometry.py::extract_angles_from_landmarks`) always
   zero z first (documented reason: MediaPipe's monocular z is only
   reliable at fixed camera framing — real-world accuracy is 0-3% with z
   included vs ~46% zeroed). This train/inference feature mismatch is a
   strong candidate for the documented ~90-93% validation vs ~0-6%
   real-world MLP accuracy gap. **Fixed** (added `pts[:, 2] = 0.0` before
   angle computation, line ~101). This fix lives in
   `/home/anamitra/Projects_and_Code/Scripts_and_Source/` (NOT the git repo
   `yoga_posture_workspace/` — a separate local scripts directory that holds
   the training pipeline, not deployed/version-controlled the same way).
3. **Same warrior_1/warrior_2 bug, independently duplicated**, found in the
   training-label-generation rules at
   `/home/anamitra/Projects_and_Code/Scripts_and_Source/experiments/classify_all_movements.py`
   (lines ~176-186) — this one corrupts training *labels*, not just live
   inference. Fixed the same way (warrior_2 checked first).
4. **Repo hygiene** (commit `4243ead`): `FEATURE_NAMES` was independently
   duplicated in both `geometry.py` and `rules_classifier.py` (silent
   index-mismatch risk if ever edited out of sync) — `rules_classifier.py`
   now imports it from `geometry.py` instead. `backend/test_backend_groq.py`
   had a hardcoded personal path — now reads `GROQ_API_KEY`/
   `GROQ_API_KEY_FILE` env vars. Removed orphaned `chair_pose.jpg` (pose was
   removed from the vocabulary earlier, image never pruned). Added
   `engines.node>=22.0.0` to `frontend/package.json`, matching what CI
   already enforces for the Capacitor Android build.
5. **GitHub push auth was broken this session** (old embedded PAT in the git
   remote URL had expired) — fixed permanently via `gh auth setup-git` +
   resetting the remote to a plain HTTPS URL (no embedded token). Future
   sessions should no longer hit this; if it recurs, run
   `gh auth status` to confirm `gh`'s own token is still valid, then
   `gh auth setup-git` again.

## Findings NOT fixed (deliberately, with reasoning) — pick up if useful

- **`mediapipe`/`protobuf` fully unpinned** in `backend/requirements.txt`.
  This exact combination has broken Kaggle builds multiple times this
  project (the working fix there was always `pip install --no-deps
  mediapipe==0.10.14`, keeping the environment's stock protobuf). Did NOT
  pin this in the HF Space's requirements.txt under time pressure — no time
  to verify a pin wouldn't break the currently-working live deployment. If
  you do this, redeploy and directly test `/api/analyse_frame` live
  afterward before considering it done.
- **Root `Dockerfile` vs root `README.md` deployment-entrypoint mismatch** —
  `Dockerfile` copies only `backend/app` and runs bare `uvicorn
  app.main:app` (no Gradio UI); `README.md`'s HF Space frontmatter declares
  `sdk: gradio`, `app_file: app.py`. These describe two different running
  services. Likely the README/gradio declaration is authoritative (HF
  Spaces with `sdk: gradio` ignores a root Dockerfile), making the
  Dockerfile vestigial — but not confirmed. Reconcile before it misleads
  someone.
- **Groq model ID** `qwen/qwen3.6-27b` in
  `backend/app/services/correction.py` — I initially wrongly flagged this as
  invalid; the user correctly caught this, and I confirmed via Groq's live
  docs that it IS a real model, just a "Preview Model... not intended for
  production." A live test of `/api/generate_correction` returned text
  identical to the static fallback template — most likely because no
  `GROQ_API_KEY` was available for that specific unauthenticated test call,
  not a broken model ID. **Still unconfirmed**: whether the HF Space actually
  has `GROQ_API_KEY` set as a secret at all. If it does and it's still
  falling back, investigate preview-model access restrictions next.
- **Chair_pose vs downward_dog shadowing** in `classify_pose()` — dormant
  (chair_pose is disabled anyway), only matters if chair_pose is ever
  re-enabled without also reordering. Noted for whoever does that.
- **`hip_abduct_l`/`hip_abduct_r`** are computed but never used in
  `classify_pose()` — flagged as the exact feature that could more robustly
  distinguish `tree_pose` from `warrior`/`lunge` (true hip abduction vs a
  wide stance) instead of the current threshold-nudge approach. Good
  candidate for the Priority 4 work below.
- **`mobile/` (the other session's Expo app) contains a large amount of
  unused "Manus" app-builder template scaffolding** (`server/`, `drizzle/`,
  `oauth/`, `lib/_core/manus-runtime.ts`) — confirmed via grep not imported
  by any real app screen, dead code, adds dependency/audit surface for no
  reason. Safe to remove if anyone wants to clean up `mobile/`, but it's not
  this session's app to maintain — flag, don't touch without checking with
  whoever owns that app's continuity.

## Priority 4 — full design for the video-frame pipeline (started this session per "start Kaggle stuff immediately")

**Already exists and should be reused, not rebuilt:**
- Existing video corpus: `/home/anamitra/yoga_raw_dataset/` (12 full
  YouTube yoga videos, already downloaded).
- Existing extraction: `/home/anamitra/Projects_and_Code/Scripts_and_Source/kaggle_process.py`
  runs MediaPipe Tasks API `PoseLandmarker` in `VIDEO` mode, but **only
  reads `result.pose_landmarks[0]`, never `result.pose_world_landmarks`** —
  the entire existing training corpus has zero world-landmark data.

**The new technique (not yet tried anywhere in this project)**: bone-length
consistency correction. MediaPipe's `pose_world_landmarks` is claimed
metric-scale; a person's real skeletal segment lengths (upper-arm, forearm,
thigh, shin, etc.) are constant across a whole video even though MediaPipe's
independently-estimated per-frame z isn't. Compute each segment's length per
frame, take the **per-video median** as a self-calibration reference, and
correct/flag frames whose bone length deviates from it (rescale the distal
joint along the existing — already-reliable — 2D bone direction to match the
median length). This exploits temporal information neither prior attempt
(raw MediaPipe world-landmarks alone: 42.9%; a pretrained single-frame 2D→3D
lifter: NO-GO, 10%) could use.

**Sequenced steps** (full detail was in the Plan agent's original output,
condensed here):
1. Fork `kaggle_process.py` to capture `pose_world_landmarks` alongside
   `pose_landmarks` for all 12 existing videos.
2. Port `backend/app/utils/geometry.py::extract_angles_from_landmarks`
   verbatim into the Kaggle notebook (don't re-implement a third copy).
   Compute both 2D-zeroed and raw-world angle sets per frame.
3. Implement the bone-length-consistency correction (Section 2.2 of the
   original design — per-video median per segment, ratio-based flagging,
   rescale-along-existing-direction correction) on the raw world (x,y,z)
   columns.
4. Recreate a `sweep_2d_vs_3d.py`-equivalent (the original isn't in-repo,
   lived on the now-dead Lightning AI VM) — 40-trial randomized
   threshold-jitter sweep, `classify_pose()` unchanged, tested against a
   real held-out photo set (not frames from the same calibrated video).
5. **Go/no-go gate, same discipline as every prior experiment**: adopt only
   if it beats both 45.7% (2D-only) and 42.9% (raw world-landmarks)
   baselines. A clean negative result is expected and valid — report
   honestly either way, exactly like the 2D→3D lifter's NO-GO this project
   already has on record.
6. **Important scoping note from the original design**: this technique's
   practical payoff is most likely a **cleaner offline retraining signal**
   for the MLP (Section 3 below), not a new live inference-time vote — a
   single photo/webcam frame has no video to compute a bone-length median
   from. A live-session version (caching a per-user bone-length profile from
   the app's existing 15-second Digital Twin calibration) is a stretch goal,
   not a Phase-1 deliverable.

**What the resulting larger dataset should be used for, in order**:
(a) recalibrate `_POSE_FEATURE_BANDS` in `rules_classifier.py` statistically
from hundreds of real per-pose frames (fast, low-risk) — do this first;
(b) retrain the MLP on the corrected data (higher effort, higher upside).
Note: (b) may already be substantially addressed by this session's zero_z
fix alone — check the zero_z retrain's validation result before assuming
bone-length correction is still needed to close the real-world accuracy gap.

**Pose coverage revisit** (only after (a)/(b) above): `chair_pose` (~0%) and
`warrior_1` (18.3%) stay disabled today for lack of good real data, not
confirmed unfixability — video frames from real tutorials that explicitly
cue these poses (via `.description` sidecar timestamps) could give cleaner
ground truth than single curated photos did.

## Current live production state (all verified working as of this session)

- Backend hybrid classifier (`hybrid_classify` in
  `backend/app/utils/rules_classifier.py`): live poses are `warrior_2`,
  `mountain_pose`, `cobra_pose`, `tree_pose`, `plank`, `downward_dog`;
  `chair_pose` and `warrior_1` stay disabled (honestly, low real-world
  accuracy, re-confirmed this session's audit).
- STGCN: live checkpoint NOT promoted to the newer `_v2` (no validation data
  existed to prove `_v2` is actually better — correctly not promoted on
  structural checks alone, from an earlier session this same project).
- Frontend: Vercel deployment confirmed READY on the latest commit.
- Two Android apps coexist: `frontend/android/` (Capacitor WebView, CI
  builds a debug APK successfully on GitHub Actions) and `mobile/`
  (Expo/React Native native app, separate session's work, own test suite).
- Git: `origin/main` is the source of truth; always `git fetch`/pull before
  starting new work — this repo has gone stale locally more than once this
  project (another session's commits landing without this session knowing).
