# AsanaAI — CONFERENCE PIVOT handoff (2026-10-04)

> **Status (2026-10-05):** the next-step lists in this file are DONE (retrained models, held-out evaluation, new ST-GCN deployed, cascade live). Current state and open items: `backup/RESUME_HERE_2026-10-05.md` (CURRENT STATE). Keep this file for the reasoning and verified facts.

Read this FIRST. It supersedes the photo-domain (v3/v4) route for the conference
deliverable. The older `SESSION_HANDOFF_2026-09-21.md` + `docs/TRAINING_LESSONS.md`
still hold for the deployed app.

## 0. THE GOAL (do not drift)

Present at a **conference**. Mentor said tough poses are NOT required.
* Keep the 3 poses that worked: **mountain_pose, cobra_pose, warrior_2**.
* Add easier-pose data to the prepared CSV and boost **MLP + ST-GCN** metrics.
* **Retrain the user's ORIGINAL trainers, unmodified, on the ORIGINAL CSV**
  ("as it was goated"), with the new videos' rows appended. Do NOT swap in the
  newer v3/v4 photo-domain scripts. Architecture fixed by proposal P05:
  3-head MLP + ST-GCN.
* User, verbatim: "do all by yourself bro .. very childish lol" — do not hand
  work back to the user if any sanctioned route exists. "Do not fake .. make it
  better." "train the best ever model in existence".

## 1. Hard constraints (user's own words)

* HF account = **Arko007 only** (never bhumika).
* Files uploaded 2 months ago on `Arko007/yoga-posture-models` are NEVER to be
  replaced/changed/deleted. Publish new names only.
* Use the **old Modal account** (profile `anamitrasarslsn10ab`) — not the CLI of
  other agents. User has Modal credits: CPU for data, GPU when needed.
* No dataset/checkpoint downloads to the local PC (3.7GB RAM). Heavy work ->
  Modal / Kaggle (account anamitrasarkar007) / GitHub Codespace. Never
  `kaggle kernels output` on big kernels (use REST `log` field, see §7).
* Backups/scripts live in `<project>/backup/`, never the scratchpad.

## 2. Where everything is

| Thing | Location |
|---|---|
| ORIGINAL trainers | `~/Projects_and_Code/Scripts_and_Source/train_mlp_3head_gpu.py`, `train_stgcn_gpu.py` |
| Original pipeline | same dir: `extract_landmarks.py` -> `extract_features_safe.py` -> `compile_master_dataset.py` -> `generate_sequence_features.py` |
| Original CSV (654,488 rows, 12 videos) | local `~/yoga_raw_dataset/master_mlp_dataset_fully_classified.csv`; HF `Arko007/Yoga-650k` (dataset); Modal vol `csv/` |
| Raw videos (12, all vinyasa) | local `~/yoga_raw_dataset/*.mp4`; HF `Arko007/yoga-dataset-raw` (`yoga_raw_dataset/` and `Yoga_Dataset_Raw/`, same 12) |
| Landmarks (12) | local `~/yoga_raw_dataset/landmarks_*.npy`; Modal vol `landmarks/landmarks_<ytid>.npy` |
| ST-GCN arrays | local `stgcn_master_{feats,labels}.npy` (54,488x60x99); Modal vol `stgcn/orig12/` (regenerated, identical) |
| Kaggle notebook of the old code | `anamitrasarkar007/mlp-stgcn-training` |
| Modal workspace | `~/yoga_posture_workspace/modal/` |
| Modal volume `asanaai-data` | `csv/`, `landmarks/`, `stgcn_ref/`, `stgcn/orig12/`, `runs/<tag>/`, `meta/` |
| Modal secret | `arko007-hf-token`, env key name **`HF_TOKEN_ARKO007`** |
| Project memory | `~/.claude/projects/-home-anamitra/memory/yoga-video-pipeline-plan.md` |

Modal files: `asanaai_train.py` (stages `seed|build|mlp|stgcn`), `asanaai_holds.py`
(yt-dlp probe), `probe_clients.py`, `probe_titles.py`, `fetch_new_videos.sh`,
`landmark_map.json`. Kaggle probe: `planning/yt_probe/` (result: Kaggle IPs are YouTube-blocked too).

## 3. Verified facts (measured this session, not assumed)

1. **Feature space**: the CSV's 15 angles equal the original extractor with
   **[WRONG - SEE §10 CORRECTION]** (this line claimed z-zeroed; measured later: the 654k CSV is RAW-z).
   Any new row MUST be built with the identical recipe: MediaPipe Pose 0.10.14,
   `static_image_mode=False, model_complexity=1`, `max_width=640` (INTER_AREA),
   conf 0.5/0.5, EVERY frame, undetected -> zeros, then `interpolate_occlusions`
   (vis>=0.5), then z:=0, then the 15 angles in column order
   `elbow_l elbow_r shoulder_l shoulder_r hip_l hip_r knee_l knee_r ankle_l ankle_r trunk_l trunk_r neck hip_abduct_l hip_abduct_r`.
   Local mediapipe/numpy versions already match the Modal image.
2. **ST-GCN data**: original `generate_sequence_features.py` (SEQ 60, STRIDE 12,
   homogeneity 0.85, RAW 99 coords incl. z, labels from `imperfect_pose_label`)
   rerun on Modal gave **54,488 windows, 100% label agreement** with the array
   the old model trained on. Tail classes: table_top 6, corpse 10, halfway_lift 18,
   upward_salute 50, chair_pose 51 windows — this is what new data must fix.
3. **Kaggle notebook == local scripts minus 3 fixes** (real-kernel GPU test
   instead of `get_arch_list` string match [the notebook would fall back to CPU
   on L4], vectorised synthetic negatives, HF token fallback). Run the LOCAL ones.
4. **Old-code bug**: `RULES['tree_pose']` has duplicate dict keys (10 entries ->
   6 effective: needs left knee bent AND both hip_abduct>=115 — unsatisfiable).
   Baseline keeps it; measure a fix as a separate, labelled variant.
5. The original **validation is a random frame split** (lesson 1: 90.86% val vs
   10.5% real). Report held-out-VIDEO and real-photo numbers beside it.

## 4. Old-code essentials (so it need not be re-read)

MLP (`Yoga3HeadMLP`, input 15): 3 heads — pose CE (class weights 1/sqrt(count)),
correctness BCE, deviation SmoothL1 on 15 joints/180. `loss = pose + 1.0*correct + 1.0*dev`.
40 epochs, bs 64, AdamW lr 1e-3 wd 1e-4, ReduceLROnPlateau(0.5, patience 5),
early stop on val_loss. `generate_synthetic_negatives(fraction=0.4, seed=42)` adds
151,333 rows over the 15 RULES poses. 23 classes. Saves `mlp_3head_model_v2.pth`
+ `mlp_3head_pose_encoder_v2.npy`.
ST-GCN (`YogaSequenceLSTM`, genuine ST-GCN): bs 64, hidden 128, 120 epochs,
patience 20, dropout .2/.3/.3, CE(weight 1/sqrt, label_smoothing .1), AdamW(1e-3, wd 1e-3),
CosineAnnealing(eta_min 1e-5). Saves `stgcn_sequence_model_v2.pth` + encoder.
**BOTH scripts upload to HF on every val improvement under those `_v2` names,
which ALREADY EXIST there.** The Modal wrapper sets `HF_TOKEN` to an INVALID value
on purpose (absent would crash: the `~/yoga_retrain/.hf_token` fallback is outside
the try). Verified: upload 401s, files untouched. Publish results under NEW names.

## 5. State of runs

* **MLP baseline (original code, original CSV)**: running detached on Modal L4,
  app `ap-J7FcbSE8bgsPqMMmMhx61H`. Epochs 1-3 val pose acc 81.7 / 83.2 / 84.8%.
  Outputs land in volume `runs/mlp_baseline_origcode/` when it finishes. Logs:
  `modal app logs ap-J7FcbSE8bgsPqMMmMhx61H | grep -E "^Epoch|Early|exit code"`.
* **ST-GCN baseline (original code, regenerated-identical data)**: running detached on Modal L4,
  app `ap-yJROPF7jyfPtpj6kXWcMzy`, 22 classes. Outputs land in volume
  `runs/stgcn_baseline_origcode/`. Logs: `modal app logs ap-yJROPF7jyfPtpj6kXWcMzy | grep -E "^Epoch|Early|exit code"`.
  (Path bug fixed: `_prep_home` now reads `/data/stgcn/orig12/`.)

## 6. THE BLOCKER: getting the new videos

User supplied 13 YouTube URLs. Findings:
* **YouTube hard-blocks Modal AND Kaggle IPs** ("Sign in to confirm you're not a
  bot"; 10 yt-dlp player clients on Modal, 4 on Kaggle, all fail). Titles via the
  public oEmbed endpoint DO work from Modal (`probe_titles.py`).
* **They are follow-along routines, NOT hold videos** (Yoga With Adriene x2,
  Saurabh Bothra, Fit Tuber x2, Eleni Fit, Kassandra x2, Charlie Follows,
  SCImplify "Basic Yoga Asanas", growingannanas x2, Siddhi Yoga Hindi Day 1).
  So video-identity labels do NOT apply; they go through the ORIGINAL autolabel
  path (`compile_master_dataset.py` Case B: RandomForest trained on the 3
  hand-labelled videos' `mlp_dataset_cleaned_zero_z.csv` + `clean_labels`).
  Label quality is the ceiling; the real gain is new people/rooms/poses.
  Only `149Iac5fmoE` (SCImplify) looks one-pose-at-a-time.
* `Eml2xnoLpYE` is ALREADY in the CSV -> skip. 12 to fetch:
  v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk
  hHhxKkskHDg JHjV-wFTwSw EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow 4ZBUDd4bsyA
* **TRIED AND BLOCKED (2026-10-04): Modal, Kaggle, AND a GitHub Codespace** — every
  cloud IP gets "confirm you're not a bot" on all clients. Do not re-test them.
* Remaining routes: (a) [DONE - blocked] GitHub Codespace; (b) user's home IP via
  `modal/fetch_new_videos.sh` (480p video-only, one at a time, straight into
  Modal volume `hold_videos/`, then deletes); (c) NOT done: moving the user's
  Google cookies to a VM — it is a credential, do not do it without explicit
  say-so.

## 7. Next steps, in order

1. Resolve the video route (§6). Once videos are in volume `hold_videos/`:
2. Modal CPU job: original `extract_landmarks.py` recipe per video ->
   `landmarks_<id>.npy`; original `extract_features_safe` -> angles; verify on one
   video that the NEW pipeline reproduces an OLD video's CSV rows (re-extract
   `oUgpXY7QhpQ`, compare to CSV: tolerance for MediaPipe nondeterminism) BEFORE
   trusting new rows.
3. Label via original Case-B (RandomForest + `clean_labels`), then derive
   `imperfect_pose_label` the same way `experiments/classify_all_movements.py`
   did. Append to a COPY of the CSV (`master_mlp_dataset_plus_new.csv`); never
   edit the original.
4. Rebuild ST-GCN windows with the original script over old+new landmarks.
5. Train MLP + ST-GCN with the unmodified trainers on Modal GPU; also a
   `tree_pose`-fixed MLP variant. Evaluate: random-split val (for continuity),
   **held-out-video**, and the **frozen 103-photo set** (`planning/`).
6. Publish to HF under NEW names only (e.g. `mlp_3head_conf_v1.pth`,
   `stgcn_conf_v1.pth` + encoders). Do not touch `hf_loader.py` pointers until the
   user approves promotion.
7. Report honestly per pose, especially mountain_pose / cobra_pose / warrior_2.

## 8. Reading Kaggle logs without downloading output

`GET https://www.kaggle.com/api/v1/kernels/output?userName=anamitrasarkar007&kernelSlug=<slug>`
with basic auth from `~/.kaggle/kaggle.json`; the JSON `log` field is a JSON list of
`{"data": ...}` events. `kaggle kernels status <user>/<slug>` for state.

## 9. Still waiting on the user (unrelated to the conference)

Space `GROQ_API_KEY` is invalid (401) — rotation command in
`SESSION_HANDOFF_2026-09-21.md` §6.


## 10. UPDATE 2026-10-04 (late): videos obtained + A CORRECTION

**Videos solved via the user's Colab** (their own Colab IP is accepted by YouTube; every
datacenter IP is not -- re-tested Modal with deno + curl-cffi + bgutil PO-token: still blocked).
I drove Colab through Chrome: ran yt-dlp with the user's exact format selector, got all 12 (3.1GB),
then the USER pasted their Arko007 token into a hidden getpass prompt (I never type tokens) and the cell
uploaded to HF **`Arko007/yoga-dataset-raw/new_videos_2026-10/`** (12 mp4 + info.json + description).
`v7AYKMP6rOE` and `149Iac5fmoE` needed a looser selector (<=720p, not avc-only) -> may be AV1/VP9
(OpenCV-headless cannot decode -> extract_one now ffprobes + transcodes to h264 when needed).
Modal copied them HF->volume `hold_videos/` (sizes identical). A temporary public upload endpoint on Modal
was DENIED by the auto-mode classifier (persistent public endpoint) -> do not retry that route.

**Extraction** (`modal/asanaai_extract.py`, original `extract_landmarks.py` untouched): running, app ap-L9jOIas5uvstsGcG1Cdsc8,
12 parallel containers, outputs `landmarks_new/landmarks_<id>.npy`, `new_angles/<id>.csv` (ZERO-z angles; ignore, see below).
Stages: `verify` (done) | `fetch` (done) | `extract`.

**CORRECTION (measured, `modal/asanaai_diag2.py`, results in volume `runs/diag2.json`):**
the live 654k CSV (`master_mlp_dataset_fully_classified.csv`) = angles computed with occlusion
interpolation and **RAW z (3D angles)**: median err 6e-6, p95 0.0 on 3 videos. The z-ZEROED recipe
(current `extract_features_safe.py`, and the backend `geometry.extract_angles_from_landmarks(zero_z=True)`)
is off by a median ~14 deg. Volume landmark files are sha256-identical to the local ones.
=> The ORIGINAL model was trained on 3D angles but is SERVED 2D (zero-z) angles -- a genuine train/serve
mismatch (a prime suspect for 90% val vs ~10% real-world, beyond the random-split issue).
=> New rows appended to the old CSV for the faithful run MUST use interp + RAW z (write own angle fn; do not
call the current extract_features_safe). `mlp_dataset_cleaned_zero_z.csv` (RF labeller training data) is zero-z.
=> PLAN: train TWO variants with the unmodified trainers: (A) faithful: old CSV (raw-z) + new rows raw-z;
(B) serve-matched: ALL rows (old 12 from stored landmarks + new) recomputed ZERO-z, matching what the app feeds
the model. Report both; recommend B for the deployed app. Landmarks are recipe-independent so no re-extraction.
Also: the earlier verify (new extraction vs CSV = 14 deg) was this z mismatch, not MediaPipe noise; true
re-extraction noise vs stored landmarks (both zero-z) = median 1.0 deg, p95 17.5 (different encode of the video).

**Next:** (1) finish extraction; (2) upload `mlp_dataset_cleaned_zero_z.csv` (7.7MB) to volume; (3) label new
videos with original Case-B RF + clean_labels (zero-z angles, consistent with that CSV); derive
`imperfect_pose_label` per `experiments/classify_all_movements.py`; (4) build CSV A and CSV B; (5) rebuild ST-GCN
windows over old+new landmarks; (6) train both variants; eval random-split + held-out-video + frozen 103 photos;
(7) publish under NEW names only.

### 10b. The ORIGINAL label chain for new videos (read from the scripts, 3 stages, in order)
1. `compile_master_dataset.py` Case B: RandomForest(100, balanced) trained on `mlp_dataset_cleaned_zero_z.csv`
   (zero-z angles, 3 hand-labelled videos) -> `pose_label`, then `clean_labels(min_duration=15)`.
2. `experiments/compile_imperfect_dataset.py`: `imperfect_pose_label` = pose_label, but `transition/unknown` frames
   within 90 total degrees (L2) of a centroid of {triangle, plank, seated_forward, upward_dog, corpse} (centroids from
   the cleaned CSV) become `imperfect_<pose>`.
3. `experiments/classify_all_movements.py`: rule cascade turns remaining `transition/unknown` into mountain_pose,
   upward_salute, downward_dog, cobra_pose, child_pose, seated_staff, seated_easy_pose, tree_pose, warrior_2 (before
   warrior_1), lunge_pose, standing_forward_fold, halfway_lift, table_top, chair_pose, standing_pose. These rules were
   written against the RAW-z angle space of the master CSV -> apply them to raw-z angles for variant A.
Result column used for training: `imperfect_pose_label` of `master_mlp_dataset_fully_classified.csv`.
Upload `~/Yoga_Dataset_Raw/data/mlp_dataset_cleaned_zero_z.csv` (7.7MB) to the Modal volume for stage 1/2 (not done yet).

### 10c. State when the usage limit hit
* Extraction `ap-L9jOIas5uvstsGcG1Cdsc8` was ~4-23% through (longest video 116k frames, ETA ~80 min). It was launched with
  a non-detached `nohup modal run` from the user's machine -> if that process died the app stopped: check
  `modal app list`, and re-run `modal run --detach asanaai_extract.py --stage extract` if landmarks_new/ is incomplete.
* Baselines (detached): MLP ap-J7FcbSE8bgsPqMMmMhx61H, ST-GCN ap-yJROPF7jyfPtpj6kXWcMzy -> volume runs/*_baseline_origcode/.
* Colab tab "Untitled81.ipynb" still open with the 12 videos on its disk (not needed any more: they are on HF + Modal).
* HF `Arko007/yoga-dataset-raw/new_videos_2026-10/` is the only new thing added to HF (additive). No model repo touched.

## 11. UPDATE (later still): the data was rebuilt properly. READ THIS FIRST ON RESUME.

**User directive (verbatim intent):** make the data genuinely better, do whatever is needed (re-extraction OK), goal =
>= 6-7 poses correctly handled by BOTH the 3-head MLP and the ST-GCN; keep MLP + ST-GCN + MediaPipe (never swap);
CLIFF is inference-only (backend/app/services/occlusion.py) -- not trained, don't touch; nothing dummy; upload the
final ~1M-row CSV to HF *from Modal* (not done yet, see TODO).

**Why the old labels were not trusted:** the master CSV's imperfect_pose_label comes from a RandomForest that predicted
"transition/unknown" for ~100% of new frames (its hand-labelled set is 84% transition) + a centroid rule + an angle-rule
cascade => labels were rules on angles; a model copying them scores high without being right in the real world.
Original model on 422 real photos: 29.4% overall, 1 pose passes (>=70% recall AND precision, n>=10).

**New pipeline (all in modal/, all run on Modal; volume `asanaai-data`):**
1. `asanaai_whisper.py`  Whisper-large-v3 transcripts of all 24 videos -> `transcripts/<id>.json` (done, 24/24).
2. `asanaai_cuelabel.py` label = instructor NAMED the pose (regex incl. Sanskrit/Devanagari) within [-1s,+30s] AND the app's
   own rule engine (backend rules_classifier.classify_pose + orientation) says that pose AND steady hold (mean |d angle| < 8 deg
   over +-7 frames). Moving frames / no-pose steady frames = transition/unknown; everything ambiguous = `__ignore__`
   (excluded, neither positive nor negative). -> `cue/<id>.npz`. Done for 24/24 (JHjV + 4ZBU extraction finished later).
3. `asanaai_assemble.py` -> `csv/cue/cue_full.csv` (24 videos, ZERO-z angles = what the app feeds), MLP sets under
   `csv/cue2/<variant>/mlp_{all,fold0..4}.csv` (video positives + transition + public photos + out-of-vocab real photos as
   "none of our poses" + Commons photos fold-wise; `--mirror 1` adds a left/right twin of every row).
4. `asanaai_photos.py` 6,616 real photos from HF `rotemvahava/yoga-poses-107` + `AdityasArsenal/Yoga-pose-Data-Set`
   (PoseLandmarker heavy, >=0.55 vis, same alias table, pixel-hash dedup) incl. 3,982 out-of-vocabulary photos as negatives
   -> `photos/public_corpus.npz` (train 4,931 / held-out test 1,685).
5. `asanaai_harvest.py` Commons+Openverse harvest (original script unmodified, 1 container/class) -> `photos/harvest/*.npz`
   (search-query labels = WEAK); `asanaai_harvest_filter.py` keeps only dup-free photos where rule engine OR a model that never
   saw Commons agrees with the query -> `photos/harvest_kept.npz`. (queued, see TODO)
6. `asanaai_train2.py` unmodified original trainers on any CSV (HF upload disabled by an invalid token on purpose),
   `build_windows_cue` / `build_windows_cue2` (ST-GCN windows: original label_window; v2 = + mirror + photo-hold clips with sway
   measured from real video holds), evaluators: `eval_oof` (5-fold out-of-fold on the 422 Commons photos), `eval_public`
   (held-out public photos incl. OOV false-alarm rate), `eval_stgcn_oof` (3 held-out-VIDEO folds).
   Pass bar everywhere: n>=10 (windows>=20), recall>=0.70 AND precision>=0.70.

**Results so far (honest; the trainer's own "Val Acc" 85-95% is a random split and means nothing):**
* MLP v1 (cue video + 422 photos): Commons OOF 37.4%, 1 pose passes (seated_easy).
* MLP vA (+6.6k public photos + OOV negatives, neg_ratio 0.3): held-out PUBLIC photos 5 pass (warrior_2 .93/.89, downward_dog
  .87/.83, tree .97/.93, triangle 1.0/.71, chair .71/.71), OOV false alarm 21%; Commons OOF 39.1%, 0 pass.
  Without OOV negatives: 3 pass, false alarm 52%. neg_ratio 0.1: 4 pass.
* MLP vM (vA + mirror): public 4 pass; **Commons OOF 43.6%, 2 pass (seated_easy .70/.95, triangle .75/1.0)**; downward_dog .65/.77.
* ST-GCN v1 (22 videos, held-out-VIDEO folds): 68% overall, 2 pass (child_pose .81/.88, corpse .75/.73); most others call "transition".
* ST-GCN v2 (mirror + photo-hold clips) folds RUNNING: apps ap-tpwEGxnrfXczX1dbAUNLv5 (f0), ap-qGhZyokXbTWEPTMoReDws1 (f1),
  ap-NdAoYXsUsHmrvJs7P5YTnF (f2), ap-YJQ5GfcSTXQEpYjosl361i (all); tags `cue2_stgcn_f{0,1,2,all}`; score with
  `modal run asanaai_train2.py --stage soof --tag cue2_stgcn_f --csv-rel stgcn_cue2` when all 3 folds saved.
  (ST-GCN fold exit code 1 = harmless: trainer's final classification_report fails AFTER saving the model.)
* MLP vN (no Commons photos, mirror, 24 videos) training -> tag `cue2_mlp_vN`; a queued background chain scores it and runs the harvest filter.

**TODO, in order:**
1. Read vN scores (all 422 Commons photos fully independent) + harvest_filter_report.json.
2. Rebuild with `--harvest 1` (assemble_v2) -> 5 folds + all (+ mirror), re-score Commons OOF + public. Rebuild ST-GCN v3 windows
   from the 24-video cue_full (build_windows_cue then build_windows_cue2) -> 3 folds + all -> `soof`.
3. Pick the final recipe, train the deployable pair on everything, report per-pose pass table honestly.
4. UPLOAD to HF *from Modal* (user asked): final CSV(s) + labels + landmarks + transcripts-derived label files + README that states
   provenance (cue-verified, not rule-derived) to a NEW private dataset repo (e.g. `Arko007/Yoga-1M`); models under NEW names only
   (never the 2-month-old live files). Secret `arko007-hf-token` (key HF_TOKEN_ARKO007) exists in Modal.
5. Do not change hf_loader.py pointers without user approval.

## 12. UPDATE 2026-10-05: THE POSE CASCADE IS LIVE (read first)
**Deployed:** Space `Arko007/yoga_pose` runs commit `698f17a` (backend: `app/services/cascade.py`, `pose.py`, `hf_loader.py`, `config.py`, `tests/test_cascade.py`, docs `docs/CASCADE.md`)
with Space variable **`ENABLE_POSE_CASCADE=1`**. ROLLBACK = remove that variable (`HfApi.delete_space_variable("Arko007/yoga_pose","ENABLE_POSE_CASCADE")`), no redeploy.
Gate model = `mlp_3head_gate_v1.pth` + `mlp_3head_gate_v1_encoder.npy` added to `Arko007/yoga-posture-models` as NEW names (sha256-verified copy of `cueH_mlp_all`); nothing existing touched.
**How it decides:** live v4 MLP names the pose; the gate (today's 3-head MLP) says "one of my poses?" + gives the form score; rule engine only supplies per-joint deviations (angle bands).
**Evidence (real endpoint, real models):** held-out public photos 36.9% -> 78.9% overall, false alarms (lower=better; non-vocabulary pose photos named as one of ours) 78.3% -> 20.8%, poses at >=70% recall+precision 0 -> 7.
Live Space replay of the same 1,685 photos: 78.1% / 21.8% / 7 (16 requests lost to HF's 429 edge throttle under an 8-thread stress test; real users send ~1 req/10 s).
Do NOT quote the 60.2% frozen-103 number for the cascade (gate was trained on those 103); clean figure is 55.3% (gate scored out-of-fold) vs 37.9-40.8% before.
**Why the MLP swap alone did nothing:** `hybrid_classify` lets the rule engine win every MLP/rules disagreement (swapping in a better MLP = identical to rules-only).
**Old vs new MLP (clean tests):** 19 Jul original 23.3% on frozen-103, 75% false alarms; new 49.5%, 22.6%; live v4 60.2% but 81.5% false alarms. Correctness head: new best (85% drop when a joint is broken 45 deg vs 59% for v4). Deviation head weak in ALL models (12% top-1 vs 6.7% chance).
**BROKEN: GitHub Actions secret `HF_TOKEN`** ("Invalid username or password" at the push step of `.github/workflows/hf_sync.yml`, run 37228494222). I deployed MANUALLY (clone Space, wipe root, copy backend/* + README.md, push with the Arko007 token, temp copy deleted).
Until the secret is fixed (`gh secret set HF_TOKEN`) pushes to main will NOT reach the Space; repeat the manual method or fix the secret.
**UI work NOT done yet:** Free mode already exists in the web app (commit 751ec51 removed pose selection on purpose); Guided mode (select a pose + mismatch guard) must be restored behind a mode switch.
Backend already supports it: request `target_pose` -> response `guided{target_pose,matches,target_correctness,target_deviations,target_has_bands}`; response also has `candidates` (top-3) and `cascade{}`.
**Kaggle (account anamitrasarkar007):** kernels `asanaai-conf-{stgcn,mlp,compare..5,prod,prod2,e2e,live}`; PRIVATE dataset `asanaai-conf-creds` holds the Modal + HF tokens as files -> DELETE it when the ST-GCN finishes (`kaggle datasets delete`), and REVOKE the Modal token (user said they would).
`asanaai-conf-stgcn` (target-pose ST-GCN: 3 held-out-video folds + final, 2xT4) was still training (epoch ~50/120 on the first two folds); progress via HF `Arko007/asanaai-conference-runs/logs/live_stgcn.txt`, finished runs under `runs/cueT2_stgcn_*`.
Score it with `kernel_compare`-style harness: `T.eval_stgcn_oof(prefix="cueT2_stgcn_f", k=3, test_prefix="cueT2_f")` (asanaai_train2.py, via the modal stand-in in `planning/kaggle_transfer/shim`).
**HF repos:** `Arko007/Yoga-1M` (PUBLIC dataset, no transcripts: copyrighted speech), `Arko007/asanaai-conference-runs` (private models + evals).

### 12b. UI: FREE / GUIDED modes are LIVE on production (2026-10-05)
Commits 4d3886d + 93a5464 on main (branch `guided-mode` was verified first: tsc exit 0, Vercel preview build success, browser-tested on the preview).
FREE (default, = the old behaviour): detect whatever the user does, score THAT pose. GUIDED: choose a pose (selector restored from 751ec51^); request carries `target_pose`;
a recognised-but-different pose shows the "Wrong pose" badge + localized message (en/hi/bn) and no coaching is generated for the wrong pose; the reference panel follows the chosen pose.
Mode + chosen pose persist in localStorage (guarded). Verified in a browser: switch, target highlight, persistence across reload, readable hint (a light-on-dark hint bug was found and fixed).
NOT exercised live: the wrong-pose badge/message path (needs a camera feed); covered by tsc + code review, and the backend `guided.matches` field was verified on the live Space.
Production site: https://yoga-posture-correction-system.vercel.app (serves the new labels). Auto-deploy of the BACKEND from GitHub is still broken (Actions secret HF_TOKEN invalid).

### 12c. Deploy secret + speech (2026-10-05, later)
* GitHub Actions secret `HF_TOKEN` REPLACED (stdin from the Arko007 key file; never on a command line or in the repo; `git grep` for hf_ tokens = 0). Re-run of the failed workflow: auth + clone OK, "No changes to commit"
  (the Space already had the content), so the final PUSH step is only proven on the next real backend push.
* Speech: Hindi/Bengali TTS already existed (web `announceTTS` sets hi-IN/bn-IN + picks a voice; Expo app passes language to expo-speech). Fixed three gaps in the WEB app (commit c90d687, on main):
  (1) sentences were split only at . ! ? so long Hindi/Bengali text (danda "।") was never chunked and Chrome stopped it after ~15 s -> `frontend/src/utils/speechText.ts#splitForSpeech`;
  (2) Android reports locales as hi_IN -> normalised, bn-IN and bn-BD both accepted; (3) localized "no voice for this language" notice when the voice list HAS loaded and has no voice for the language (never on an empty list).
  Verified: helpers on real Hindi/Bengali text; 6-stage scenario in the real preview page; production bundle contains the strings. NOT verified: audible playback (the test Chrome has 0 speech voices). The Expo app was not changed or tested.
* Commit discipline: only explicit paths were committed. Still UNcommitted on purpose: backup/, modal/, planning/kaggle_transfer, planning/live_check, planning/yt_probe, planning/kaggle_per_class_audit.py.
