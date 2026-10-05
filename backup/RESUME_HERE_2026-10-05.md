# AsanaAI — RESUME HERE (written 2026-10-05 night; UPDATED 2026-10-06 morning)

## UPDATE 2026-10-05 night — UI REDESIGN IS LIVE ON PRODUCTION (read this block first)
* Production https://yoga-posture-correction-system.vercel.app now serves the new "calm studio" UI (main @ 6429f54). New files: `frontend/src/styles/app.css` (all dashboard styling, `ap-*` classes), `frontend/src/lib/fonts.ts`, `pages/_document.tsx`; `pages/index.tsx` render rewritten (logic unchanged).
* **Black camera on phones — root cause + fix:** `.camera-video-element` was a 1px, opacity-0 `<video>` (phones don't decode frames for it) and the canvas was only painted after the pose engine's first result, so any slow/failed engine = black box with no message. Now: the `<video>` is the visible picture, the canvas only draws the skeleton, the engine init is idempotent and retried from the frame loop, `pose.send` errors back off instead of killing the loop, failures show a message + Retry (no `alert()`), track `ended` and backgrounding are handled. Start no longer waits on the unused camera_utils script.
* **Layout contract:** camera pinned and never scrolls; everything else scrolls in its OWN region (fixes content sliding under the camera). ≥1320px two columns; 768-1319 camera on top; <768 bottom dock + pose drawer. Header holds down to 280px.
* **Smoothness:** the dashboard used to re-render ~30x/s (angles in React state every frame) and 1x/s (session timer). Now angles go to a ref (UI publishes 4 Hz), `SessionTimer` owns its state, pose inference runs per decoded video frame (`requestVideoFrameCallback`), canvas geometry cached, no blur over live video, self-hosted fonts via next/font.
* Free mode: tap any pose to preview its guide ("Follow my pose" returns to auto). Sidebar sections are an animated accordion with a one-line summary when collapsed.
* **Gotchas learned:** (1) Vercel gives every push its own preview URL; `gh ... commits/main/status` returns the PREVIEW status of the same SHA, so check the production deployment with `list_deployments target=production` before declaring a deploy done. (2) next/font in the pages router only emits @font-face if the font module is used in the client tree: apply the variable classes on a wrapper in `_app`, not only in `_document`. (3) The PWA service worker can show a returning visitor the old build for one load. (4) Mobile layouts can be tested in this Chrome by rendering the site in same-origin iframes of phone width (the window cannot be resized).
* NOT verified on a real phone (needs the user): live MediaPipe on a real camera, the Sequence Flow row, audible Hindi/Bengali speech, iOS fullscreen fallback.
* Not done / ideas: landing page (`/landing`) restyle + copy; optional hero image via Puter (user allowed, not needed so far); dark theme.

## UPDATE 2026-10-05 late (model/backend work)
**Done and LIVE (all verified):**
* **Decision: keep the NEW ST-GCN (`stgcn_target_v1`), retire the old ones.** Same held-out windows: new macro recall 0.82 vs old live 0.19 (docs/BENCHMARKS.md §5). Old files stay on HF untouched (rollback = delete the two Space variables below).
* Live Space `Arko007/yoga_pose`: variables `ENABLE_POSE_CASCADE=1`, `STGCN_MODEL_FILE=stgcn_target_v1.pth`, `STGCN_ENCODER_FILE=stgcn_target_v1_encoder.npy`; `GROQ_API_KEY` rotated (LLM paraphrase works again: Hindi cue now differs from the template). Parity: deployed endpoint == local run of the published file on 60/60 windows.
* Code on main (commits 2fcc1ba backend env switch, d7d3825 web fixes, 6e58b27 docs/scripts); Vercel production built; GitHub Actions -> Space sync WORKS (automatic "Sync backend from GitHub commit ..." on the Space).
* **Root cause of "ST-GCN looks broken" = the web app's frame feed (0.5-2 fps into the buffer vs 25 fps training), NOT the model or the server** (server code == trainer code, max logit diff 0.0). Fixed: `frontend/src/utils/sequenceBuffer.ts` (time-based, camera-rate, resampled to 25 fps; unit test `backup/test_sequence_buffer.js`).
* Also fixed: model confidence was used as the correctness score; the previous cycle's sequence pose was read from stale state. Now: the per-frame cascade names the pose and scores the form; the ST-GCN is a second opinion in the "Sequence Flow" row, shown only when confident (never overrides).
* Kaggle credentials dataset: deleted then RECREATED (user: keep tokens, we may train again); private.
* **Retraining: NOT needed now.** Open improvement (only if the mentor wants more): more videos/people for tree, warrior 2, plank, triangle, seated easy (held-out n is 30-90 windows for several); named transitions would need a retrain with transition classes on the new data.
* NOT verified (needs a real camera, the user will test): the live Sequence Flow row with a real webcam at 15-30 fps.

## UPDATE 2026-10-06 (earlier block; partly superseded by the one above)
* **ST-GCN finished** (all 4 runs on HF `runs/cueT2_stgcn_{f0,f1,f2,all}`). Held-out-VIDEO result: overall 82.2% (flattered by the 60%-of-windows transition class; macro recall 78.0%), **3 poses pass** (child .74/.91, corpse 1.00/.85, downward dog .83/.77); seated easy (.68/.81), tree (.87/.67), triangle (.64/.95) are near misses. A cross-fitted threshold shift gets **4** (adds seated easy, triangle; corpse drops out); nothing reaches 6. Full tables: `docs/BENCHMARKS.md` §5.
* **Conference claim, honestly:** MLP cascade = 7 poses on held-out photos (§4); ST-GCN = 3 (4 with the shifted threshold). Say exactly that.
* **Correctness probe through the live endpoint** (§8): form score 0.94 while held -> 0.25 when a LEG joint is broken 45 deg (every probe, every clip) but 0.92 when an ARM joint is broken: the score is blind to arm errors.
* **NEW finding to fix before ever enabling a sequence model:** `frontend/src/hooks/useYogaPipeline.ts` lines ~160-166 — when the sequence model is confident, the hook sets `correctness = flowConfidence` (the model's confidence, NOT form correctness). Dead code today (the live sequence model is never confident) but wrong by the user's own metric. Fix = use `frameRes.correctness_score` (already fetched in that branch). Not changed yet (needs the user's OK; it touches production web).
* **Awaiting the user's decision:** (1) swap the live sequence model to the new ST-GCN? Recommendation: NO before the conference (it knows only 8 poses + transition, 3 pass; the cascade already carries the live demo; the old model is harmless because it never reaches the UI). (2) commit the docs/scripts (currently uncommitted on purpose).
* Cleanup status: Kaggle creds dataset DELETED 2026-10-06 (§7); still open: the user revokes the Modal + HF tokens.

---
(original night-of-2026-10-05 text follows)


Read this file FIRST, then `SESSION_HANDOFF_2026-10-04_CONFERENCE.md` §11-12c for depth, `docs/BENCHMARKS.md` for the report numbers.

## 0. Goal and fixed rules (do not drift)
* Project P05 (RCC Institute of Information Technology; supervisor Mr. Sujit Chakraborty). Show at the conference that the models genuinely handle >= 6 poses with honest, held-out numbers. Mentor: other poses don't matter.
* Architecture is FIXED: 3-head ResMLP (pose + correctness + 15-joint deviation from 15 angles), real ST-GCN (graph conv over the 33-joint MediaPipe skeleton + temporal conv), MediaPipe. CLIFF = inference-only, never trained.
* **The metric the user cares about is CORRECTNESS (is the form right), not model confidence.** Report pose recall/precision AND the correctness head; keep "confidence" out of results tables.
* Arko007 HF account ONLY. Never overwrite/delete existing model files (new names only). Never download datasets/checkpoints to the local PC (3.7 GB RAM) — Kaggle/Codespace/Modal-free route only. `kaggle kernels output` is forbidden (use the REST `log` field via `planning/kaggle_transfer/klog.py`). No local npm installs/builds. Do not change `hf_loader.py` model pointers without the user's approval. Commit only explicit paths; docs/scripts below are intentionally UNCOMMITTED.

## 1. FIRST THING TOMORROW (5 minutes)
```bash
export PATH=$PATH:~/.local/bin
kaggle kernels status anamitrasarkar007/asanaai-conf-stgcn           # RUNNING / COMPLETE / ERROR
python3 - <<'PY'                                                       # what has been uploaded so far (token read from file, never printed)
from huggingface_hub import HfApi; t=open('/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token').read().strip()
fs=HfApi(token=t).list_repo_files('Arko007/asanaai-conference-runs'); print([f for f in fs if 'stgcn' in f.lower()])
PY
```
Heartbeat: HF `Arko007/asanaai-conference-runs/logs/live_stgcn.txt` (last lines). At the time of writing: folds f0/f1 at epoch ~88/120, val acc ~98% (RANDOM-split val: NOT a result, see §5).
**Expected noise:** the line `Failed to upload to Hugging Face Hub: 401` is the original trainer deliberately failing to overwrite old model names (invalid token on purpose). Ignore it. The Kaggle worker uploads each finished run to `runs/cueT2_stgcn_*` itself.
**Risk:** Kaggle sessions die at ~12 h. Finished folds are already on HF; if the kernel dies with folds unfinished, relaunch only the missing ones.

## 2. Running jobs
| What | Where | Done when |
|---|---|---|
| Target-pose ST-GCN (3 held-out-video folds f0,f1,f2 + final `all`, 2xT4) | Kaggle `anamitrasarkar007/asanaai-conf-stgcn` (v4) | status COMPLETE; models in `runs/cueT2_stgcn_f0,f1,f2,all`; then the kernel itself runs `eval_stgcn_oof(prefix="cueT2_stgcn_f", k=3, test_prefix="cueT2_f")` and uploads `evals_stgcn/` |
| Live replay (correctness probe added) | Kaggle `asanaai-conf-replay` v3 | DONE: `evals_compare/live_replay_v3.json` |
| ST-GCN operating-point analysis | Kaggle `asanaai-conf-stgcn-op` v2 | DONE: `evals_stgcn/stgcn_operating_point_crossfit_v2.json` |
Nothing else is running. Modal credits are exhausted (do not use Modal).

## 3. What to do when the ST-GCN finishes (in this order)
1. Read `evals_stgcn/*` (held-out-VIDEO accuracy per pose, windows >= 20, pass bar recall AND precision >= 0.70).
2. Fill `docs/BENCHMARKS.md` §5 last row (currently "TBD") and copy to `backup/BENCHMARKS_2026-10-05.md` (the two files are kept identical).
3. Count how many poses pass; the conference goal is >= 6 poses across MLP-cascade + ST-GCN (MLP cascade already has 7 on held-out photos).
4. ONLY with the user's approval: swap the live sequence model `stgcn_transitions_v1` for the new one (new name in `yoga-posture-models`, then the pointer in `hf_loader.py`, then manual Space deploy — see §6). Why it matters: see §4 sequence-model finding.
5. Cleanup (§7).

## 4. What is LIVE in production right now
* Backend Space `Arko007/yoga_pose` (https://arko007-yoga-pose.hf.space): commit `698f17a` + variable **`ENABLE_POSE_CASCADE=1`** (rollback = delete that variable, no redeploy: `HfApi().delete_space_variable("Arko007/yoga_pose","ENABLE_POSE_CASCADE")`).
  Cascade: live v4 MLP names the pose -> gate `mlp_3head_gate_v1` (new name, = `cueH_mlp_all`) vetoes "not one of mine" and gives the correctness score -> angle bands give per-joint deviations (tree/lunge have none -> gate deviations). `docs/CASCADE.md`.
* Web (Vercel, main): Free / Guided modes (`target_pose` -> `guided{...}`; wrong-pose badge + en/hi/bn message) and speech fixes for Hindi/Bengali (danda splitting, locale normalisation, missing-voice notice). https://yoga-posture-correction-system.vercel.app
* GitHub Actions `hf_sync.yml` secret `HF_TOKEN` was replaced; the final PUSH step is still unproven until the next real backend push. If a backend change must go out and Actions fails, deploy manually (clone the Space, wipe root, copy `backend/*` + README, push with the Arko007 token, delete the temp copy).
* NOT tested: real camera in the browser (Chrome tab is hidden here -> no webcam); audible Hindi/Bengali (no voices in the test Chrome); the Expo app. **User plans to test these by hand.**

## 5. Findings to remember (all measured; details in docs/BENCHMARKS.md)
**MLP, old vs new (clean tests only; trainer-reported 90-98% val is a leaky random split, never quote it):**
| Test | Original (Jul) | v4 (was live) | New (today) |
|---|---|---|---|
| Frozen 103 Commons photos, pose accuracy | 23.3% | 60.2% (contaminated: trained on Commons) | 49.5% (out-of-fold) |
| Held-out public photos (1,685; 1,037 are other poses) poses passing 70/70 | 0 | 1 | 3 (warrior 2, downward dog, tree) |
| ...false alarm (named an out-of-vocabulary pose as one of ours; lower = better) | 75% | 81.5% | 22.6% |
| Commons-422 out-of-fold | 26.8%, 1 pose | - | 44.8%, 2 poses (seated easy, triangle) |
| **Correctness head**: score drop when one joint is broken 45 deg | 65% | 59% | **85%** |
| Good-form score (higher better) | 0.28 | 0.79 | 0.81 |
| Deviation head top-1 (chance 6.7%) | 8% | 7% | 12% -> weak everywhere, do not claim it |
**Cascade (what is deployed):** held-out photos overall 36.9% -> 78.9%, false alarms 78.3% -> 20.8%, poses passing 0 -> 7; the live Space reproduces it (78.1% / 21.8% / 7).
Root cause of "a better MLP changed nothing": `hybrid_classify` lets the rule engine win every MLP/rules disagreement.
**Live replay (real MediaPipe on real video -> deployed Space)**, pose naming while held: 7 of 8 clips >= 0.81 (mountain 1.00/0.81, corpse 0.99, seated easy 0.99, child 1.00, tree 1.00, lunge 0.99) but **one seated-easy clip (EvMTrP8eRvM @44 s) scored 0.06 — named upward_dog throughout** (cause not investigated). Caveat: those clips are IN the gate's training data, so it is a plumbing check, not a generalisation number. Correctness probe results: BENCHMARKS §8.
**Sequence model finding:** `stgcn_transitions_v1` is wrong on most held poses (often "chair_pose") but was never confident (requires_static_fallback true on every call), so the UI never uses it today (`useYogaPipeline.ts` lines 118-121). Latent hazard: if it ever turns confident-and-wrong, the hook adopts its pose and its confidence as the correctness score (lines 160-166). Replacing it with the new ST-GCN removes that.
**Raw-z vs zero-z:** the old 654k CSV used 3D (raw-z) angles; the app serves 2D (zero-z) -> ~14 deg median train/serve mismatch. All new data is zero-z.

## 6. Open questions / decisions for the user
1. Swap in the new ST-GCN as the live sequence model? (only after §3 results).
2. Investigate seated_easy@EvMTrP8eRvM -> upward_dog (camera angle?). Mountain and cobra are weak on real photos (recall 12-30%); cobra has no training hold >= 5 s.
3. Record a few seconds of unseen people/rooms (the user, teammates) and run them through `kernel_replay` for the clean live number (replace the in-sample §8).
4. **Post-conference (user's idea, deferred): detect "just standing / not a normal pose".** Today the MLP only has `transition/unknown` (moving or no pose) and `standing_pose` is nearly unrecognised (recall 0.00 on Commons-422, n=6). Needs labelled standing data; do after the conference.
5. Mobile Expo speech unchanged/untested.

## 7. Cleanup checklist
* [x] DONE 2026-10-06: Kaggle private dataset `anamitrasarkar007/asanaai-conf-creds` deleted (the CLI has no delete; used `kagglesdk` `ApiDeleteDatasetRequest`; the API now returns 403 for it). Old kernels that list it as a source can no longer run; to run a new Kaggle kernel that needs HF access, recreate a private dataset holding only `hf_token` (re-create from `~/Downloads/API_Keys_and_Secrets/hf_token`) and delete it again afterwards.
* [ ] User revokes the Modal token and the HF token that was inside that dataset (user said they would; NOTE: the local `hf_token` file is the same token, so revoking it also cuts this machine's HF access until a new one is saved there).
* [ ] GROQ_API_KEY on the Space is invalid (LLM paraphrase serves the template) — one command in SESSION_HANDOFF_2026-09-21.md §4/§6.
* HF dataset `Arko007/Yoga-1M` is PUBLIC on purpose (no transcripts); `Arko007/asanaai-conference-runs` is PRIVATE.

## 8. Location map
* Docs: `docs/BENCHMARKS.md` (report numbers), `docs/LIVE_TEST_CLIPS.md` (steady-hold YouTube timestamps per pose for manual live tests), `docs/CASCADE.md`, `backup/SESSION_HANDOFF_2026-10-04_CONFERENCE.md`.
* Code (UNCOMMITTED): `modal/` (extract/whisper/cuelabel/assemble/train2…), `planning/kaggle_transfer/` (klog.py, shim, kaggle_worker.py, runner_*.py, `kernel_*` dirs, `code_stage/`), `planning/live_check/replay_live.py`, `planning/kaggle_per_class_audit.py`, `planning/yt_probe/`.
* Committed: backend cascade (`698f17a`), UI modes (`4d3886d`, `93a5464`), speech (`c90d687`).
* HF: `Arko007/Yoga-1M` (dataset mirror `vol/`, `code/`), `Arko007/asanaai-conference-runs` (runs/, evals_*, logs/, `evals_compare/live_replay*.json`, `backup_2026-10-05/` = copy of these docs), `Arko007/yoga-posture-models` (live models + gate), `Arko007/yoga-dataset-raw/new_videos_2026-10/`.
* Memory: `~/.claude/projects/-home-anamitra/memory/yoga-video-pipeline-plan.md`.

## 9. Gotchas learned
* Pass counts use different n thresholds per set (n>=10 photos, n>=5 frozen-103, >=20 windows ST-GCN) — say which when quoting.
* Frozen-103 numbers are contaminated for any model trained on Commons (v4, photodomain, gate `all`). Never quote them as generalisation.
* HF edge throttles ~17 req/s per IP (429) — fine for real use (~1 req/10 s).
* Kaggle: 2 concurrent GPU sessions max; logs only readable after completion except via the HF heartbeat; pip installs race -> `PIP_NO_INDEX=1`.
* `pkill -f <script>` can kill your own shell. Local Python is 3.10 (no `tomllib`). Minifiers escape non-ASCII in JS bundles when grepping production.
* Claude-in-Chrome tab is `visibilityState: hidden` -> real camera/MediaPipe pages cannot run there; use the Kaggle replay instead.
