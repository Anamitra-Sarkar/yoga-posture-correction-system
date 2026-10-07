# AsanaAI benchmarks (for the report) -- compiled 2026-10-05 (sections 9 and 10 added later the same day)

**Rule used everywhere:** a number counts only if it was measured on data the model never trained on. The trainers' own "Train/Val accuracy"
(90-98%) is a random split in which near-identical neighbouring frames, mirrored copies and repeated photos sit on both sides; it is NOT a result and is
listed only so it is never mistaken for one. **Pass bar** for a pose: recall >= 0.70 AND precision >= 0.70 with n >= 10 test photos (n >= 5 on the tiny frozen-103 set;
n >= 20 windows for the ST-GCN). **False alarm** = a photo of a pose that is NOT one of the app's poses that the model names as one of them (**lower is better**).

## 1. Data
| Item | Size / detail |
|---|---|
| Training videos | 24 (12 original + 12 new), 1,206,391 frames, MediaPipe landmarks; 15 joint angles with z zeroed (what the app feeds the model) |
| Labels | cue-verified: instructor named the pose (Whisper-large-v3 transcripts, en + hi) AND the app's rule engine agrees AND the pose is held steady. ~80k positive frames, ~421k transition, ~705k excluded as ambiguous |
| Old CSV | 654,488 rows, labels derived from rules on angles; built with RAW-z (3D) angles while the app serves ZERO-z (2D): median 14 deg mismatch (a train/serve gap) |
| Photo benchmark "Commons-422" | 422 freely licensed Wikimedia photos; **frozen test split = 103** (hash split, fixed since 2026-09-19) |
| Public photos | 6,616 from two public HF datasets: 2,634 labelled with our poses + 3,982 of poses outside our vocabulary (used as negatives); 4,931 train / **1,685 held-out test** (1,037 are "other poses") |
| Harvest | Commons/Openverse photos, cleaned (duplicates of the benchmark removed; label must agree with the rule engine or a model that never saw Commons): ~450 kept |

## 2. 3-head ResMLP -- pose head, old vs new (clean tests only)
Frozen 103 Commons photos (none of these models trained on them; the new model scored out-of-fold):
| Model | Overall | Macro recall (poses with >=5 photos) |
|---|---|---|
| 19 Jul original 3-head (the ~2-month-old model) | 23.3% | 17.5% |
| 3 Sep retrain | 13.6% | 11.1% |
| 18 Sep photo-domain v1 | 45.6% | 48.9% |
| 21 Sep v4 (was live) | **60.2%** | **64.8%** |
| **New (today), out-of-fold** | 49.5% | 58.0% |

Held-out public photos (1,685; 1,037 other-pose photos). v4 may have seen similar public images, which can only flatter it:
| Model | Poses passing | False alarms |
|---|---|---|
| 19 Jul original | 0 | 75.0% |
| 3 Sep retrain | 0 | 64.5% |
| 21 Sep v4 | 1 (warrior 2) | 81.5% |
| **New (today)** | **3** (warrior 2 .94/.92, downward dog .88/.84, tree .97/.92) | **22.6%** |

Commons-422, models that never saw any photo: original 19 Jul 26.8% overall (1 pose passes: seated easy); 3 Sep 17.3% (0); new model out-of-fold **44.8% (2: seated easy .70/.98, triangle .71/1.00)**.
Do NOT quote the 86-90% that the 18/21 Sep models score on all 422 Commons photos: they were trained on most of them.

Ablations that mattered (held-out public photos): adding real photos of OTHER poses as negatives cut false alarms from 52.5% to 21.4% and raised poses passing from 3 to 5;
mirror augmentation gave +4.5 points on Commons out-of-fold (39.1% -> 43.6%).

## 3. Correctness and deviation heads (no labelled good/bad-form data exists, so this is a property test)
648 held-out in-vocabulary photos x 5 trials: one joint broken by 45 degrees. Chance for "points at the broken joint" is 6.7%.
| Model | Score on good form | Score drops when form is broken | Deviation head's #1 joint = broken joint |
|---|---|---|---|
| 19 Jul original | 0.28 | 65% | 8.1% |
| 3 Sep retrain | 0.67 | 74.9% | 18.2% |
| 21 Sep v4 | 0.79 | 59.2% | 6.7% (chance) |
| **New (today)** | **0.81** | **85.0%** | 12.4% |
Conclusion: the correctness head reacts to bad form; the per-joint deviation head is weak in every model, so joint-level feedback uses each pose's angle bands.

## 4. Production decision path (the real `hybrid_classify`, then the real endpoint with the real models)
Finding: the rule engine wins every MLP/rules disagreement, so swapping in a better MLP changed nothing (identical to rules-only).
Held-out public photos, 1,685:
| Policy | Overall | False alarms | Poses passing |
|---|---|---|---|
| Production before (v4 + rules) | 36.9% | 78.3% | 0 |
| New model + rules | 36.9% | 78.3% | 0 |
| Rules only | 36.9% | 78.3% | 0 |
| v4 alone | 42.9% | 81.5% | 1 |
| Average of both MLPs | 65.2% | 47.7% | 5 |
| Veto lifted when rules agree | 65.9% | 42.3% | 5 |
| **Cascade: v4 names, new model vetoes "not mine"** | **78.9%** | **20.8%** | **7** (chair, corpse, downward dog, tree, triangle, upward dog, warrior 2) |
Policy chosen on a validation half only: validation 79.3% / 20.1% / 3 poses; **untouched half 78.6% / 21.6% / 4 poses** (corpse, downward dog, tree, warrior 2).
Wild Commons frozen-103: before 37.9% (2-way) / 40.8% (3-way vote); **cascade 55.3%** (gate scored out-of-fold). v4 alone scores 60.2% there: the cascade gives up ~5 photos of naming accuracy for ~60 points fewer false alarms.
(The endpoint run showing 60.2% for the cascade on frozen-103 is invalid: the deployed gate was trained on those photos.)
Through the real `/analyse_frame` endpoint with real models: switch OFF reproduces production exactly (36.9% / 78.3% / 0); switch ON 78.9% / 20.8% / 7; endpoint agreed with the offline policy on 100% of requests.
**Live Space replay** (cascade on, same 1,685 photos): 78.1% / 21.8% / 7 poses; 16 of 1,685 requests were rejected by Hugging Face's 429 rate limiter under an 8-thread stress test (real use is ~1 request per 10 s).
Engineering: 14 new unit/endpoint tests, full backend suite 65 passed / 2 skipped; server compute p50 7-9 ms, p95 8-11 ms per request, 324 MB peak memory with all three models loaded.

## 5. ST-GCN (sequence model; real ST-GCN: graph convolution over the 33-joint skeleton + temporal convolution)
Held-out-VIDEO evaluation (3 folds by video, windows of 60 frames; the model never saw those people or rooms):
| Version | Overall | Poses passing |
|---|---|---|
| v1, 22 videos, no augmentation | 68.4% | 2: child pose (.81/.88), corpse (.75/.73); seated easy near (.67/.91) |
| **Final target-pose ST-GCN (8 poses + transition/unknown, mirror + photo-hold augmentation), 3 held-out-VIDEO folds** | **82.2%** (flattered: 60% of the 8,002 windows are transition/unknown; macro recall over the 9 classes **78.0%**, over the 8 poses 76.9%) | **3**: child pose (.74/.91), corpse (1.00/.85), downward dog (.83/.77) |
Per pose, held-out videos (windows are 60 frames, stride 12, so overlapping: the effective sample is the number of held-out holds/videos, which is small; pass bar = recall AND precision >= 0.70, >= 20 windows):
| Pose | Windows | Recall | Precision | Verdict | Main confusion |
|---|---|---|---|---|---|
| child pose | 1045 | 0.74 | 0.91 | PASS | transition 235, plank 40 |
| corpse | 230 | 1.00 | 0.85 | PASS | - |
| downward dog | 682 | 0.83 | 0.77 | PASS | transition 107 |
| seated easy | 510 | 0.68 | 0.81 | near miss (recall) | transition 162 |
| tree | 87 | 0.87 | 0.67 | near miss (precision) | transition 6, seated easy 4 |
| triangle | 528 | 0.64 | 0.95 | near miss (recall) | transition 179 |
| warrior 2 | 89 | 0.62 | 0.26 | fail (158 transition windows are called warrior 2) | transition 34 |
| plank | 30 | 0.77 | 0.18 | fail (tiny n) | transition 7 |
| transition/unknown | 4801 | 0.87 | 0.85 | - | downward dog 167, warrior 2 158 |
Reading: the ST-GCN is conservative -- misses go to "transition/unknown" (it says "not sure", not another pose). Three poses clear the bar; three more are within ~0.07 of it. It does NOT reach 6 passing poses on held-out video. The MLP cascade reaches 7 on held-out photos (section 4), so the conference claim of "6+ poses" is supported by the MLP cascade; for the ST-GCN say "3 pass, 3 near-miss".
**Operating point (is the ST-GCN just too conservative?)** -- the model sends most misses to "transition/unknown"; one offset on that class's logit trades recall for precision. Done without leakage: the offset is chosen on two folds and applied only to the third (cross-fitted), kernel `asanaai-conf-stgcn-op`, `evals_stgcn/stgcn_operating_point_crossfit_v2.json`. Two objectives were tried and BOTH are reported:
| Setting | Overall | Poses passing |
|---|---|---|
| Default (argmax) | 82.2% | **3** (child pose, corpse, downward dog) |
| Cross-fitted offset, objective = macro F1 (offsets +0.25 / -0.5 / +0.25) | 81.7% | **3** (same) |
| Cross-fitted offset, objective = number of poses passing (offsets -0.25 / -1.25 / -1.5) | 81.9% | **4** (child pose .78/.89, downward dog .85/.72, triangle .75/.92, seated easy .86/.81; corpse drops to precision .67) |
| Ceiling: best single offset picked ON the scored windows (-0.5 to -1.0; OPTIMISTIC, not a result) | ~82% | 5 (adds seated easy, triangle) |
No offset anywhere in the sweep (-4 to +4) reaches 6. Honest statement for the report: **the ST-GCN reliably handles 3 poses on unseen videos, 4 with a less conservative cross-fitted threshold; tree, warrior 2 and plank have too few held-out windows (87 / 89 / 30) and low precision to claim.** The 6+ pose claim rests on the MLP cascade (section 4).
**Old vs new ST-GCN on the SAME held-out windows** (kernel `asanaai-conf-stgcn-cmp`, `evals_stgcn/stgcn_old_vs_new_compare.json`). 24 videos, 8,002 windows (60 frames, stride 12), 3 folds by video. NEW = each fold model scored only on videos it never trained on. OLD = the ST-GCNs in the live model repo. 12 of the 24 videos arrived after the old models were trained, so on those the old models are clean (2,682 windows); on all 24 the old ones may be in-sample (flatters them). Every model's output is mapped to the 8-pose vocabulary (`hold:X` -> X; anything else -> transition/unknown).
| Model | Overall | Macro recall | Macro precision | Poses passing | Production gate (answers that reach the UI): share of true-pose windows answered / precision of answers |
|---|---|---|---|---|---|
| OLD original (Jul) | 67.3% | 11.6% | 27.7% | 0 | 0% / - |
| OLD v2 | 67.8% | 13.2% | 28.0% | 0 | 1.8% / 100% (19 answers) |
| OLD LIVE `stgcn_transitions_v1` | 68.9% | 18.8% | 33.5% | 0 | 2.8% / 93.8% (32 answers) |
| **NEW `stgcn_target_v1`** | **76.9%** | **82.1%** | **75.3%** | 2 (tree, triangle; most poses have n < 30 on this subset) | **50.5% / 88.1%** (611 answers) |
On all 24 videos: OLD LIVE 67.5% overall, macro recall 20.9%, 1 pose passing (child), answers 22.8% of true-pose windows at 78.5% precision (possibly in-sample); NEW 82.2%, macro recall 76.8%, 3 passing, answers 60.8% at 86.8% precision (all held-out).
Verdict: the new model recognises held poses on unseen videos; the old live model essentially cannot (macro recall ~19-21%, and it is confident on under 3% of windows). Old `orig`/`v2` are no better. The old model's other feature -- NAMED transitions (`transition:A->B`) -- is not measured here and the new model does not have it (one generic transition/unknown class).

**Why the web app made the ST-GCN look broken -- frame rate.** The ST-GCN is trained on 60 consecutive native-rate frames (~2.4 s). The web app fed its buffer one frame per finished API cycle (at most 2/s, ~0.5/s in practice), so its "60 frames" spanned 30-120 s. Re-scoring the same held-out videos with every k-th frame (new-video subset):
| Frames per second into the buffer | Window span | NEW macro recall (poses passing) | OLD LIVE macro recall (poses passing) |
|---|---|---|---|
| 25 (training scale, k=1) | 2.4 s | **0.82** (2) | 0.19 (0) |
| 12.5 (k=2) | 4.8 s | 0.81 (2) | 0.24 (0) |
| 6.2 (k=4) | 9.6 s | 0.51 (0) | 0.60 (1) |
| 3.1 (k=8) | 19 s | 0.41 (0) | 0.62 (1) |
| 2.1 (k=12) | 29 s | 0.39 (0) | 0.48 (0) |
| 1.0 (k=25) | 60 s | n = 9 windows, not interpretable | |
The new model needs training-rate windows; the old live model was partly trained on 3 fps windows and is relatively better at slow rates, but its best (0.60) is still far below the new model at the right rate (0.82). Fixed in the web app (commit d7d3825): a time-based buffer filled from every camera frame and resampled to 25 fps.

**Implementation audit (server side is clean).** (1) Backend model + normalisation vs the trainer's: max |logit difference| = 0.0 on 64 windows. (2) Deployed `/analyse_sequence` vs a local run of the same weights: identical label AND confidence on 40/40 windows for `stgcn_transitions_v1` and 60/60 for `stgcn_target_v1`. (3) Architecture is a real ST-GCN: graph convolution over the 33-joint MediaPipe skeleton (symmetric-normalised adjacency) + 9-tap temporal convolution, 3 blocks. The weakness of the old model is generalisation, not serving; the live problems were the web app's frame feed and its use of model confidence as the form score (both fixed).
Models: HF private `Arko007/asanaai-conference-runs/runs/cueT2_stgcn_{f0,f1,f2,all}/`; eval `evals_stgcn/stgcn_target_heldout_video.json`. Trainer-reported val accuracy (98.4-98.7%) is a leaky random split -- never quote it.
Original 2-month-old training run, for reference only (random split, NOT generalisation): MLP 91.52% validation; ST-GCN 82.6% validation at epoch 55.

## 6. Original model on real photos (the starting point)
Commons-422, app recipe: 29.4% overall, 1 pose passing (seated easy .81/.81); with the CSV's own raw-z recipe 21.3%, 0 passing.

## 7. Limitations to state in the report
* Photo benchmarks are a proxy for live use. Section 8 is a replay of in-training-set videos (plumbing check); a clean live number from unseen people is still to be recorded.
* Small samples: the frozen set has 103 photos (an 11-point gap is ~11 photos); some poses have n < 10.
* Pose set: 7-8 poses clear the bar on held-out photos; on the hardest wild set only 1-2 do. Mountain and cobra are weak on real photos (recall 12-30%).
* The correctness score is validated only by the perturbation test, not by human coaches; the deviation head is not reliable.
* v4's training photos may overlap the public held-out sets (flattering v4, not the new model).

## 8. Live replay (real MediaPipe on real video -> the DEPLOYED Space; run 2026-10-05)
Harness: `planning/live_check/replay_live.py` via Kaggle kernel `asanaai-conf-replay` (Kaggle cannot reach a webcam, so recorded video stands in for one).
Each clip is decoded, run through MediaPipe, and the web app's calls are replayed in order (a frame call every ~0.25 s; a 60-frame sequence call every 3 s) against `https://arko007-yoga-pose.hf.space` with the cascade ON.
Raw results: private HF `Arko007/asanaai-conference-runs/evals_compare/live_replay_v2.json`.

| Clip (video @ start +length) | Held pose | Frame-level accuracy while held (this is what the UI shows) | What it called instead |
|---|---|---|---|
| 149Iac5fmoE @433 +14 s | mountain | **1.00** (84 calls) | - |
| 149Iac5fmoE @769 +25 s | corpse | **0.99** (143) | mountain x1 |
| v7AYKMP6rOE @925 +20 s | mountain | **0.81** (120) | seated_staff x22 |
| EvMTrP8eRvM @44 +17 s | seated easy | **0.06** (102) FAILS | upward_dog x96 |
| 4K2xTVRDJgA @566 +17 s | seated easy | **0.99** (107) | - |
| O2EY79Ys_qg @545 +49 s | child pose | **1.00** (307) | - |
| JHjV-wFTwSw @1304 +19 s | tree | **1.00** (114) | - |
| JHjV-wFTwSw @1829 +13 s | lunge | **0.99** (78) | unknown x1 |

7 of 8 clips >= 0.81. Call-weighted 88.4% (child pose's 307 calls dominate); mean over clips 85.5%. Zero failed API calls, 0 "unknown" while holding except 1 lunge frame.
Seated easy is NOT reliable live: one clip 0.99, the other 0.06 (named upward_dog throughout; viewing angle/body shape not yet investigated).
Moving stretches (no held pose, 3 clips): the system reports "transitioning" for 74% / 92% / 79% of frames and names a pose on 26% / 4% / 48% (no ground truth for moving frames; the 48% clip includes 12 corpse calls as the person lowers to the floor).

**Correctness probe (live endpoint, real MediaPipe angles from the same 8 clips; run 2026-10-05, `evals_compare/live_replay_v3.json`):**
Score the deployed `/analyse_frame` gives frames it correctly named while the instructor holds the pose, then the SAME frame with one joint broken by 45 degrees (a property test: no labelled bad-form data exists).
| Clip | Pose | Score while held | Left KNEE broken 45 deg | Left ELBOW broken 45 deg |
|---|---|---|---|---|
| 149Iac5fmoE @433 | mountain | 0.91 | 0.01 (dropped on 100% of frames) | 0.90 |
| 149Iac5fmoE @769 | corpse | 0.99 | 0.28 (100%) | 1.00 (no drop) |
| v7AYKMP6rOE @925 | mountain | 0.90 | 0.00 (100%) | 0.84 |
| EvMTrP8eRvM @44 | seated easy | 0.85 | 0.28 (100%, n=2 only) | 0.84 (no drop) |
| 4K2xTVRDJgA @566 | seated easy | 1.00 | 0.65 (100%) | 0.97 |
| O2EY79Ys_qg @545 | child | 0.97 | 0.02 (100%) | 0.86 |
| JHjV-wFTwSw @1304 | tree | 0.99 | 0.15 (100%) | 0.95 |
| JHjV-wFTwSw @1829 | lunge | 0.95 | 0.57 (100%) | 0.96 (drops on 27% only) |
Mean over the 8 clips: held 0.94 -> knee broken 0.25 -> elbow broken 0.92.
Reading: the correctness score is high for instructor-form holds and collapses when a LEG joint is broken (every probe on every clip), but it is nearly blind to a broken ARM joint (mean 0.94 -> 0.92). Arm errors are only caught, if at all, by the per-joint angle-band layer. State this in the report; do not claim whole-body form checking.
Same caveat as above: these frames are in the gate's training data (it is a regression check of the deployed path, not a generalisation number).

**Live replay with the new ST-GCN deployed** (Space variables `STGCN_MODEL_FILE=stgcn_target_v1.pth`, `evals_compare/live_replay_v4.json`; native-rate windows every 0.5 s, same 8 hold + 3 moving clips). The ST-GCN answers confidently and correctly (100% of calls) on the poses it knows: corpse, seated easy (BOTH clips, including the one where the per-frame cascade says upward_dog for 94% of frames), child pose, tree. For mountain and lunge (not in its vocabulary) it stays "transition/unknown" (the UI shows "Static Check"), and on the 3 moving clips it says transition/unknown 100% / 100% / 86% of the time (2 confident corpse calls on the clip where the person lowers to the floor). Caveat as before: these videos are in the training data of the published `all` model, so this verifies the deployed path, not generalisation (the held-out numbers are in section 5).

**Previous sequence model (`stgcn_transitions_v1`, deployed until 2026-10-05, now replaced by `stgcn_target_v1`):** its raw top answer on the held clips is mostly wrong (sequence accuracy 0.00 on 5 of 8 clips; frequent wrong answer: chair_pose) BUT it was NEVER confident:
`requires_static_fallback` was True on 100% of its calls on all 11 clips (mean confidence 0.15-0.41), so the web app (`useYogaPipeline.ts`) always took the per-frame cascade path and the user never saw the sequence model's pose. Net effect while it was deployed: the sequence model contributed nothing live.
That hazard (the UI adopting a wrong confident sequence pose and using its confidence as the form score) is removed: the per-frame cascade now always names the pose and scores the form (commit d7d3825).

**Caveat (read before quoting any number above):** all 8 clips come from the 12 new YouTube videos, and those videos are in the data the gate (`mlp_3head_gate_v1` = `cueH_mlp_all`) was trained on. The pose NAME comes from the older v4 MLP (trained on photos before these videos existed), so naming is out-of-sample, but the veto and correctness score are in-sample. This is a plumbing/regression check of the deployed system, NOT a generalisation number. The generalisation number still has to come from recorded people/rooms not in any training set (the harness above runs any clip list).

Reproduce: code and data `huggingface.co/datasets/Arko007/Yoga-1M` (public, no transcripts: copyrighted speech); checkpoints and eval JSONs `Arko007/asanaai-conference-runs` (private, `evals_compare/`);
pipeline `modal/`, `planning/kaggle_transfer/`; backend `backend/app/services/cascade.py`, `docs/CASCADE.md`.

## 9. Web-app changes after the replay: visibility-aware scoring and the on-device coach (2026-10-05)
These are plumbing / honesty changes, not new model accuracy. Nothing here changes the model numbers above.

**Visibility-aware scoring.** Audit finding: the server's CLIFF branch was never wired (nothing sent it CLIFF data, no CLIFF model is deployed), the SMPL refinement never ran,
and what actually ran was a left-right mirror: exact for symmetric stances, wrong for asymmetric ones (tree pose: the raised foot was placed on the floor, off by 0.23 of the frame
height) while still being reported as "recovered". Angle features always came from the raw landmarks, so a hidden joint could still produce a confident wrong score and cue.
Change: joints the camera cannot see (landmark visibility, smoothed with hysteresis in `frontend/src/utils/visibility.ts`) are neither scored nor coached, and the UI names what it is not checking.
The map "hidden joint -> affected angle features" was checked against the Python angle code by perturbation (`backend/tools/offline_parity.py`, 0 visibility errors).
Known limit: pose NAMING can still be influenced by a hallucinated hidden joint; only scoring and coaching are masked.

**On-device coach ("basic mode").** `frontend/src/utils/offlineCoach.ts` is a port of the backend rule engine and coaching templates; its data (`offlineData.ts`) is generated by `backend/tools/gen_offline_data.py`.
Parity against the Python original (`python3 backend/tools/offline_parity.py`, re-run 2026-10-05): **10,000 generated frame cases and 2,500 coaching-text cases across 21 poses, 0 mismatches.**
What basic mode is: rules only. No MLP, no gate, no ST-GCN, no LLM paraphrase. So its pose naming is the rule engine's, i.e. the "Rules only" row of section 4 (36.9% overall, 78.3% false alarms on the held-out photos), NOT the cascade's 78.9% / 20.8%. The app labels it as basic mode with the reason (offline / server waking / reconnecting).
The server answers whenever it can; the first answer of a session is local (no waiting) while a background probe looks for the server; API calls time out after 9 s (15 s for the LLM paraphrase).

**Cadence.** Pose and score refresh about every 1.5 s (it was 10 s, so a new pose took 10-20 s to appear); coaching text, speech and the 60-frame sequence window keep the ~10 s cycle.

**Smoothness (browser measurement, synthetic landmark stream, no real camera).** At 30 Hz both the old and the new build stay smooth (p95 0.4 ms, no long tasks). At about 3x that load the old build had p99 35 ms, max 101 ms and 10 long tasks; the new build p95 2.6 ms, p99 14-34 ms with only occasional long tasks. Per-frame handler cost: mean 0.33 ms.
Not measured: a real phone camera (the user tests this by hand).

## 10. CLIFF vs the mirror fallback for hidden joints (offline experiment, Kaggle T4, run 2026-10-05)
**Question:** if CLIFF were wired in, would it recover hidden joints better than the left-right mirror the old fallback used? **Not part of the deployed system**; this is evidence for the report and for future work.
**Setup** (`planning/kaggle_transfer/kernel_cliff_exp/runner_cliff.py`, results `Arko007/asanaai-conference-runs/evals_compare/cliff_occlusion_experiment.json`): real yoga video frames; MediaPipe on the clean frame is the reference
(only frames where every key joint is clearly visible); a grey box hides one limb; hidden joints are then recovered four ways: **raw** (MediaPipe's own guess on the occluded image), **mirror** (the old fallback),
**CLIFF** (ResNet-50 checkpoint from the public release, SMPL joints projected with the predicted camera, inference only) and **CLIFF aligned** (CLIFF fitted to the visible joints). Error = 2-D distance / torso length (NME) and the error of the affected joint angle in degrees.
70 frames, 4 poses (mountain, seated easy, downward dog, tree), 3 occlusion scenarios (left leg, right arm, both feet) = 141 cases.

| Subset (n) | Method | Median angle error | Mean angle error | Median NME | Within 0.15 torso |
|---|---|---|---|---|---|
| All (141) | raw | 17.9 deg | 38.9 deg | 0.291 | 27% |
| | mirror | 8.6 deg | 20.2 deg | **0.152** | **48.9%** |
| | CLIFF | **5.8 deg** | **16.8 deg** | 0.185 | 44.0% |
| | CLIFF aligned | 7.0 deg | 22.0 deg | 0.179 | 45.4% |
| Symmetric stances (117) | mirror | **5.2 deg** | 13.8 deg | **0.131** | **59.0%** |
| | CLIFF | 6.1 deg | **11.0 deg** | 0.173 | 47.0% |
| Asymmetric (24, tree pose only) | mirror | 20.1 deg | 51.2 deg | 0.470 | 0% |
| | CLIFF | **4.8 deg** | **44.8 deg** | **0.357** | **29.2%** |
| Left leg hidden (47) | mirror | 16.2 deg | 33.5 deg | **0.214** | 34.0% |
| | CLIFF | **6.2 deg** | **12.7 deg** | 0.270 | **46.8%** |
| Right arm hidden (47) | mirror | **4.9 deg** | **11.9 deg** | **0.092** | **72.3%** |
| | CLIFF | 15.1 deg | 30.7 deg | 0.210 | 31.9% |
| Both feet hidden (47) | mirror (= raw: nothing to copy from) | 6.1 deg | 15.1 deg | 0.199 | 40.4% |
| | CLIFF | **4.7 deg** | **6.9 deg** | **0.143** | **53.2%** |

**Reading.** Neither method dominates. The mirror is better where the body really is symmetric (mountain, arms, seated easy) and has the lowest median landmark error overall; CLIFF is better for legs, for both feet hidden,
and for the asymmetric tree pose where the mirror puts the foot in the wrong place. CLIFF median joint-angle error is 5.8 deg vs the mirror's 8.6 deg, but its median landmark error is higher, and its mean angle error is still large for tree (44.8 deg).
Cost: CLIFF res50 18.7 ms per image on a T4 GPU, **164 ms on 2 CPU threads**, and it needs the IMAGE, which the app never sends (the privacy promise is that only landmark numbers leave the device).
**Decision:** keep the deployed behaviour (decline to score what the camera cannot see, section 9); do not ship CLIFF. A live CLIFF would need server-side GPU inference and uploading camera images.
**Caveats (state them):** small sample (141 cases from 70 frames); the asymmetric subset is one pose; the reference is MediaPipe on the clean frame, not motion-capture ground truth, so errors are relative to MediaPipe;
the occluder is a synthetic grey box, not a real object; a neutral SMPL stand-in replaced the unavailable `smpl_mean_params.npz` (only seeds buffers that the checkpoint overwrites).

## 11. Pose engine on phones whose GPU driver breaks MediaPipe (measured over USB on a real phone, 2026-10-05)
**Device:** POCO M7 Pro 5G (MediaTek Dimensity 7025 Ultra, GPU PowerVR BXM-8-256, driver 25.1@6715691), Android 16, Chrome 154. Measured with `backup/phone_debug/` (adb + Chrome DevTools Protocol, only a tab the script creates).
**What fails (it is NOT "no WebGL"):** Chrome creates WebGL2, WebGL1 and OffscreenCanvas WebGL2 contexts fine. The legacy MediaPipe pose engine (the one the web app uses) then aborts inside the driver: `GlScalerCalculator ... gl_quad_renderer.cc:81 frame_unifs_[i] != -1` -> `Aborted(native code called abort())`, after which every frame throws. MediaPipe Tasks with the GPU delegate runs (16 ms/frame) but returns **no poses**. Known driver problems of this GPU family (WebGL context loss, shader issues) are reported on the [PowerVR developer forum](https://forums.imgtec.com/t/bxm-8-256-long-list-of-driver-issues/3891). The raw alert users saw ("Failed to create WebGL canvas context ...") is a `window.alert` inside MediaPipe's `pose.js` ([mediapipe issue 2381](https://github.com/google/mediapipe/issues/2381)).
**What works (same phone):**
| Engine | Result |
|---|---|
| MediaPipe Tasks, CPU delegate, IMAGE mode | 33 landmarks + 33 world landmarks, ~176 ms/frame |
| MediaPipe Tasks, CPU delegate, VIDEO mode (tracking), lite / full | **~80 ms / ~117 ms per frame** |
| ... in a Web Worker, round trip per frame (this is what the app uses) | 85-95 ms; in the running app ~8 results/s with the interface free |
| TF.js WASM BlazePose lite (last resort, needs no WebGL) | ~95 ms/frame; mean joint-angle difference vs MediaPipe Tasks 4.2-6.7 deg (worst 27 deg), missed a plank: lower fidelity |
**Worker pool (measured later the same day):** one instance 7.3 results/s, **two 19/s, three 24/s** (median latency 80-100 ms). Phones with >= 6 cores run two instances and alternate frames (late answers for older frames are dropped); in the running app on the same phone this took the skeleton from ~8-10 to ~22 updates/s.
**Fallback ladder in the app (automatic, nothing is shown to the user):** MediaPipe (GPU) -> if the renderer is the known-bad PowerVR BXM-8-256 (recognised at page load) or the engine aborts / raises its WebGL alert -> MediaPipe Tasks CPU in a worker (`frontend/public/engine/pose-worker.js`, `frontend/src/utils/cpuPose.ts`) -> if the worker cannot start (no WebGL / no worker) -> TF.js WASM. The decision is remembered per browser version for up to 30 days. A second MediaPipe module cannot share the page with the legacy one (global `Module` collision), hence the worker.
**Quality check, GPU engine vs CPU engine (desktop, same photos, lite models):** mean joint-angle difference 2.4 deg (tree), 2.5 deg (cobra), 4.8 deg (warrior 2); on a mountain-pose photo the two engines tracked different people; the CPU engine found a plank that the GPU engine missed. Same model family, not bit-identical. **Latency:** ~20 skeleton updates/s (two instances) vs ~25-30 on a good GPU; the 1.5 s coaching cadence, the scoring and the recognition are unaffected.
**Brave on the same phone (Brave 1.96, Chromium 154):** Brave reports the GPU as just "Brave" (renderer and vendor masked), so the up-front recognition of the PowerVR chip cannot work there. The reactive fallback did: MediaPipe aborted in the driver, the app switched to the CPU engine by itself (~12 s on the very first start incl. downloading the engine files), then ~21 results/s, skeleton drawn, Virabhadrasana II recognised at 98%, no error panel, no raw alert (one photo, 40 s). Later starts go straight to the CPU engine (remembered per browser version).
**Not measured:** other phone models with the same chip, Samsung Internet / Firefox; a phone with a different broken GPU; sustained battery/thermal behaviour; the ST-GCN row (needs 25 fps: it shows "Static Check" on this path). The Dimensity 700 (Mali-G57) works on the normal GPU path (user report, not measured here).

## 12. On-phone prediction test with photos (the real production path on the POCO M7 Pro, 2026-10-05)
**What was run:** `backup/phone_debug/photo_pose_test.py`: the app (production) runs on the phone with the CPU pose engine (this phone's GPU path is broken, section 11); the camera is replaced by Wikimedia Commons yoga photos (looked up by the tab itself, 2-6 photos per pose, plus the app's own reference photo where it has one); every answer of the live server (`/analyse_frame`, cascade on) is logged. A photo counts as correct when the most frequent of its 3 answers equals the expected pose; a pose passes when >= 50% of its photos are correct. Raw results: `backup/phone_debug/photo_pose_test_last.json`.
| Pose | Photos correct | Verdict | Notes |
|---|---|---|---|
| tree | 6 / 6 | pass | |
| triangle | 5 / 5 | pass | |
| warrior 2 | 3 / 3 | pass | |
| downward dog | 4 / 4 | pass | one photo: no person found |
| seated easy | 4 / 5 | pass | the miss was called child pose |
| plank | 3 / 4 | pass | one photo: no person found; forearm plank called transition |
| chair | 3 / 5 | pass | side view and one frontal photo missed |
| corpse | 2 / 4 | pass (50%) | two lying photos called transition/unknown |
| child's pose | 1 / 3 | **miss** | |
| cobra | 1 / 5 | **miss** | the app's own reference photo was called warrior 2 |
| mountain | 0 / 5 | **miss** | called seated staff / corpse / tree: known weak (section 7) |
**Result: 8 of 11 poses pass.** Of the 7 poses that passed the held-out benchmark (section 4), the 6 tested here (chair, corpse, downward dog, tree, triangle, warrior 2) all pass on the phone; seated easy and plank also pass; mountain, cobra and child's pose do not (mountain and cobra were already weak in section 7).
**Caveats (read before quoting):** still photos, not a moving person; labels come from file titles (not verified frame by frame); these are public Commons photos and the models were trained on Commons photos, so some of them may have been seen in training: this is an end-to-end check of the on-phone path (engine -> landmarks -> server -> label), not a generalisation number. A first search for "cobra" returned a helicopter (no person found, no prediction, not scored); the filter now excludes it. A live test with a person was attempted but nobody was in view, so there is no live-person accuracy yet.

## 13. Standard methods on the same data, near-duplicate audit, calibration check (2026-10-07)
Written for the paper (`paper/`). Kernels: `planning/kaggle_transfer/kernel_baselines_frame` (baselines), `kernel_baselines_frame2` (audit + wild set); results on HF `Arko007/asanaai-conference-runs/evals_compare/baselines_frame_v1.json` and `baselines_frame_v2.json`. One random seed each; the two runs differ slightly (random subsampling), so pose-pass counts near the 0.70 bar are not stable.

**Setup.** Each baseline is trained on EXACTLY the rows the new MLP trained on (`vol/csv/cue2/vH/mlp_all.csv`: 136,506 rows of 15 zero-z angles = cue-verified video frames + training photos + photos of other poses labelled transition/unknown) and scored with the same metric code as sections 2 and 4. k-NN k=15 distance-weighted; SVM RBF (<= 20k class-balanced rows); logistic regression; random forest 300 trees (class-balanced); plain MLP 256-256-128 ReLU (no residual, no batch norm).

| Method | Public held-out (1,685; 1,037 other poses): overall | false alarm | poses pass | Wild Commons-103, out-of-fold: overall |
|---|---|---|---|---|
| Production before (v4 + rules) | 36.9% | 78.3% | 0 | 37.9% |
| Logistic regression (run 1) | 29.7% | 95.8% | 1 | - |
| k-NN | 58.6% | 55.7% | 1 | 50.5% |
| SVM (RBF) | 58.4% | 56.2% | 3 | 52.4% |
| Plain MLP | 80.9% | 19.1% | 3 | 47.6% |
| **Random forest** | **85.3%** | **9.8%** | 7 (9 in run 1) | **35.0%** |
| v4 alone | 42.9% | 81.5% | 1 | 60.2% |
| New 3-head model alone | 79.6% | 22.6% | 3 | 49.5% |
| Cascade (ours) | 78.9% | 20.8% | 7 | 55.3% |

**Reading.** A random forest on the same 15 angles is BETTER than the cascade on photos from the training sources (85.3% vs 78.9%, 9.8% vs 20.8% false alarms) and the WORST learned method on the wild Commons-103 photos (35.0%; 13-25 points below the SVM, k-NN, plain MLP, new network, cascade and v4). A plain MLP matches the residual 3-head model on public photos (80.9% vs 79.6%): the residual trunk is not what buys accuracy. The paper reports this as is; the cascade is kept because the architecture is fixed by the project proposal, it provides the form/deviation heads, and it is within 7 points of the best method on both sets. A tree + network ensemble is untested future work.

**Near-duplicate audit (is the forest's lead a memorisation artefact?).** Distance (standardised angles) of each of the 1,685 test photos to its nearest TRAINING PHOTO: only 0.3% are within 0.05 and 4.4% within 0.2; quartile edges 0.474 / 0.795 / 1.264. Overall accuracy / false alarm by quartile (Q1 = closest): forest 93.1/16.2, 82.7/19.6, 81.0/9.2, 84.6/1.7; plain MLP 91.7/14.8, 78.6/28.0, 75.8/21.3, 77.7/13.2; new model alone 90.5/16.2, 77.9/32.9, 74.1/25.2, 75.8/16.3; cascade 89.1/16.2, 76.2/32.0, 73.4/23.2, 77.0/13.5. The forest's lead holds in every quartile, so it is not an exact-duplicate artefact; the public held-out photos simply share their sources with the training photos, and the ranking reverses on the wild set. The wild set has only 103 photos (gaps under ~10 points are noise).

**Not measured: sequence-model baselines** (LSTM, BiLSTM + multi-head attention, temporal CNN, window-statistics MLP, ST-GCN with identity adjacency = no skeleton graph) on the cueT2 folds. Needs a GPU (a CPU run is ~13 h per ST-GCN variant); the Kaggle weekly GPU quota was exhausted on 2026-10-07. Ready: `kernel_baselines_seq` (GPU, 10 epochs, <= 6,000 windows/class, last epoch scored, identical recipe for all models; fixed layout bug) and `kernel_baselines_seqcpu` (CPU, reduced; its ST-GCN rows will not finish).

**Calibration ("Your range of motion") verified end to end on the deployed server (hand-built skeleton, angles from `backend/app/utils/geometry.py`).** Frame named seated_staff, both hip deviations 56 deg from rule bands, universal score 0.924: no profile -> no personal fields; a profile NOT containing the angles -> personal 0.924 (unchanged); a profile containing the left hip's angle -> that deviation 0, personal 0.962; both hips -> 1.000; universal score never changes. A profile spanning 0-180 deg for every joint -> 1.000 (a poor profile can forgive everything). On a Tree frame (deviations from the learned head, see `docs/CASCADE.md`) covering knee_l + hip_l zeroed 73 and 47 deg and raised personal 0.938 -> 0.968. Repeatable with `bash backup/verify_live.sh` (calibration check). NOT validated with users or clinicians.

## 14. Sequence-model baselines under one recipe (GPU run, 2026-10-07)
Kernel `planning/kaggle_transfer/kernel_baselines_seq_gpu_arko` (Kaggle account arkosarkarhehe, T4 x2, 1915 s; token-free: public HF dataset, results printed to the log); raw numbers in `results_seq_gpu_v1.json`. Same three video folds and windows as section 5 (cueT2); ONE recipe for every model: AdamW lr 1e-3 wd 1e-4, cosine schedule, class weight 1/sqrt(n), batch 128, 10 epochs, <= 6,000 training windows per class (~20k per fold), LAST epoch scored (no test-set selection), single seed. ST-GCN variants receive the trainer's graph layout via a verified reshape wrapper.

| Model | Params | Overall | Macro recall (9 classes) | Macro recall (8 poses) | Poses pass |
|---|---|---|---|---|---|
| LSTM (2 layers) | 250,505 | 83.0% | 89.5% | 91.0% | 4 (child, corpse, seated easy, triangle) |
| BiLSTM + multi-head attention | 500,489 | **87.1%** | 89.6% | 90.1% | **5** (+ downward dog) |
| Temporal CNN (no skeleton graph) | 1,003,017 | 81.5% | 89.7% | 91.4% | 4 |
| MLP on window mean/std (no temporal modelling) | **85,769** | 85.3% | **94.4%** | **96.4%** | **5** |
| ST-GCN, identity adjacency (no skeleton graph) | 893,897 | 77.1% | 84.8% | 86.4% | 4 |
| **ST-GCN (ours)** | 893,897 | 76.0% | 86.4% | 88.6% | 4 |

**Reading.** (1) The skeleton graph helps only slightly over the graph-free ST-GCN (macro recall +1.6 points, overall -1.1); with one seed this is within noise. (2) Simpler models match or exceed the ST-GCN; an 86k-parameter MLP on window statistics is best by macro recall: held-pose recognition is largely static-shape. (3) All models over-call Warrior 2 (precision 0.24-0.30) and plank (0.10-0.40, n=30) because transition windows are labelled as poses. (4) An ST-GCN fold trains in ~290 s on a T4 versus 6-15 s for the others. (5) These absolute numbers differ from section 5 (production: overall 82.2%, macro recall 78.0%) because the recipe uses fewer windows, fixed epochs and class weights; compare within the table. State in any report: the ST-GCN is kept as a second opinion and as the base for future flow scores, but its evidence is parity, not superiority.

## 15. The ST-GCN on movement (flow), and previous vs current ST-GCN (GPU runs, 2026-10-07)
**Why.** Section 14 scored the ST-GCN on held-pose recognition (the MLP's job). The ST-GCN is meant for movement between poses. Kernels (account arkosarkarhehe, token-free): `kernel_flow_stats` (what the labels support, CPU), `kernel_flow_bench` (controlled six-model benchmark, T4 x2, 3925 s, `results_flow_bench_v1.json`), `kernel_flow_oldnew` (previous vs current ST-GCN, 886 s, `results_flow_oldnew_v1.json`).
**What the cue-verified labels support** (`kernel_flow_stats`): 589 holds (>= 25 frames) in 21 videos; 568 consecutive hold pairs, 205 between different poses. Within 150 frames: 40 transitions in 14 directional pairs; within 500 frames: 70 transitions in 33 pairs; **no directional pair occurs in >= 3 videos at any threshold**, so a named-transition benchmark on held-out videos is impossible on cue-verified labels.
**Rule-defined flow task (19 classes).** Scheme of `planning/kaggle_stgcn_transitions.py` rebuilt on the 24 videos (net angle change > 15 deg/s = transition, named by the dominant pose of the first and last third; frame pose = cue label else rule-engine pose; windows with > 15% ambiguous frames dropped -> 28,544 windows). Classes with >= 100 windows from >= 3 videos: 16 holds, transition:other, unrecognized and ONE named transition (cobra pose -> downward dog; 144 windows, 15 videos). Labels are rule-defined: the task measures recovery of rule-defined flow, not agreement with an annotator. Recipe: AdamW 1e-3, cosine, class weight 1/sqrt(n), 12 epochs, <= 3,000 windows per class, last epoch scored, one seed, 3 folds by video.
| Model | Params | Flow macro recall (19 cls, all 24 videos) | Overall | Cobra->down. dog recall | Arrow-of-time accuracy (chance 50%) |
|---|---|---|---|---|---|
| MLP on window mean/std (order-blind) | 85.0k | **58.6%** | 63.5% | **68.1%** | 50.0% (exactly, by construction) |
| BiLSTM + attention | 0.50M | 56.8% | **64.5%** | 62.5% | 50.1% |
| Temporal CNN | 1.00M | 55.6% | 57.9% | 58.3% | 49.9% |
| LSTM (2 layers) | 0.25M | 53.1% | 56.2% | 66.0% | 51.1% |
| ST-GCN, identity adjacency | 0.89M | 48.4% | 53.8% | 52.1% | 50.0% |
| ST-GCN (production architecture) | 0.89M | 47.7% | 54.6% | 50.7% | 50.0% |
**Arrow of time** (every moving window shown forward and reversed; 12,424 windows, 24 videos): at chance for every model and the training loss stays at ln 2 for all but the LSTM -> UNINFORMATIVE under this recipe (no model learned direction); it is not evidence for or against the ST-GCN.
**Previous vs current ST-GCN on the same held-out windows** (the 12 videos that arrived after the previous model was trained, 9,458 windows; clean for the previous model):
| Model | Overall | Macro recall (19 cls) | Hold classes | Cobra->down. dog recall |
|---|---|---|---|---|
| PREVIOUS live ST-GCN `stgcn_transitions_v1` (30-class checkpoint) | 23.1% | 14.2% | 14.4% | 0.0% |
| Current architecture, trained on this task per fold (held-out) | 56.7% | 53.1% | 52.1% | 78.3% |
| MLP on window mean/std (reference) | **65.6%** | **66.7%** | **65.8%** | **87.0%** |
On all 24 videos (previous model may be in-sample for 12): previous 31.1% overall / 20.9% macro; current 54.6% / 47.7%; MLP 63.5% / 58.6%. **Caveats:** the label scheme is the one the previous model was trained on (favours it); only one of its seven named transitions has test windows; 12 clean videos, overlapping windows, one seed.
**Verdict.** (1) Current ST-GCN >> previous on held poses (section 5: macro recall 82.1% vs 18.8%) AND on this movement task (53.1% vs 14.2%). (2) No task we could build shows the ST-GCN (or the skeleton graph, or frame order) beating a window-statistics MLP. (3) Movement modelling is untested for lack of labelled transitions.
**Correction of earlier notes.** The previous live checkpoint's encoder has **30 classes** (21 holds, 7 directional transitions: cobra->standing, downward_dog->standing_forward_fold, lunge->tree, lunge->warrior_2, standing_forward_fold->downward_dog, tree->lunge, warrior_2->lunge; transition:other; unrecognized), not the 24 or 25 quoted in `backend/app/services/hf_loader.py`, `backend/tests/test_sequence_labels.py` and `docs/TRAINING_LESSONS.md`; the 63.0% macro (2 held-out videos, rule-defined labels) belongs to an earlier run and must not be quoted for the live checkpoint. Genuine-ST-GCN check: hand count of the layers (3 blocks 3->64->128->256, head 256->128->9) = 893,897 parameters = the count the benchmark printed; every checkpoint loads strictly into the graph class; the first commit (2026-06-10) held a Conv1d + residual-GRU + attention model under the same class name `YogaSequenceLSTM`, replaced by the real ST-GCN on 2026-07-17 (commit 018a7ba).
