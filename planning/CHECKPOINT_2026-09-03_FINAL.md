# AsanaAI — FINAL CHECKPOINT, 2026-09-03

Written under rate-limit pressure. Everything below is **measured**, not assumed.
Where a number is weak, the weakness is stated.

---

## 1. SHIPPED TO PRODUCTION AND VERIFIED LIVE

### 1a. The project's central defect — root-caused and fixed
The training feature extractor
(`/home/anamitra/Projects_and_Code/Scripts_and_Source/extract_features_safe.py`)
computed the 15 angle features from MediaPipe's raw **un-zeroed z**, while the
deployed inference path (`backend/app/utils/geometry.py`) always zeroes z. The
MLP was trained on a feature distribution it never saw in production.
Fixed at line ~101 (`pts[:, 2] = 0.0`), dataset regenerated, model retrained.

**Result: real-world accuracy ~0-6% → 52.6%** (n=19 photos, the 6 live poses),
which also beat the 2D rule engine's 45.7%. Live now — `hf_loader.py` loads
`mlp_3head_model_v2.pth`.

*Why no metric caught it:* validation reuses the training-time feature code, so
it inherits the same convention and certifies a model that cannot reproduce its
accuracy in deployment. Only differential auditing of the two code paths finds
this class of bug.

### 1b. `warrior_1`/`warrior_2` branch-order bug
The disabled `warrior_1` branch was checked *before* the live `warrior_2`, with
identical leg conditions and overlapping arm bands (shoulder 110-125 deg), so
real warrior_2 attempts in that band were classified warrior_1 then discarded as
`transition/unknown`. Fixed in `rules_classifier.py` **and** in the training-label
generator `experiments/classify_all_movements.py` (it corrupted labels too).

### 1c. A deploy pipeline silently broken for a month
One rotated Arko007 HF token broke three things simultaneously: the Kaggle
training job's uploads (the 401s), the Modal `arko007-hf-token` secret, and the
GitHub Actions `HF_TOKEN` secret — the last had blocked **every backend deploy
since Aug 5**, including the warrior fix. All three repaired.

### 1d. Free-form redesign (the product vision) — live
- Pose selection removed; target-vs-detected mismatch guard deleted.
- **Transition detection**: `transition/unknown` conflated "moving between poses"
  with "holding something unrecognised". Mean absolute angular velocity over a
  1.5s window (threshold **15 deg/s**, sitting in the empty gap between MediaPipe
  jitter ~few deg/s and real transitions ~hundreds deg/s) splits them into
  `holding` / `transitioning` / `unrecognized`. Corrections suppressed while
  moving.
- **Dual correctness**: universal + calibration-personalised, both surfaced,
  computed server-side so the Expo client inherits it.
- Joint colouring uses all 15 model deviations, not 3 hand-authored cues.
- **All four states verified live** on the production endpoint, including
  personal 0.42 > universal ~0 with correct per-joint forgiveness.

### 1e. Paper updated to match the shipped system
`/home/anamitra/yoga research/ieee_research_paper.tex` — compiles clean, 16 pages,
no undefined refs. Replaced the removed mismatch-guard section with `sec:mismatch`
(open-set + motion-state, with equations) and `sec:dualscore`; added `sec:zerozfix`
+ Table `tab:zerozfix`; corrected stale pose-count/disabled-pose claims.
**Backed up into git** at `planning/ieee_research_paper_backup_2026-09-03.tex`
(the paper itself lives outside version control).

---

## 2. MEASURED RESULTS — including the negative ones

### 2a. MLP augmentation: **NO-GO**
Fair comparison on **real-world photos** (neither model trained on them):

| | n | macro | overall |
|---|---|---|---|
| v2 (live) | 26 | **25.8%** | 15.4% |
| v3_aug (mirroring/rotation/scale/noise + class balancing) | 26 | 24.2% | 15.4% |

Augmentation did **not** help. Do not promote `mlp_3head_model_v3_aug.pth`.

**Do NOT use the video-holdout comparison for this pair** (v2 95.4% vs v3_aug
70.8%). It is invalid: v2 was trained on all 12 videos *including* the two
"holdout" videos, while v3_aug genuinely excluded them. That gap measures the
confound, not model quality.

### 2b. ST-GCN z-handling variants: **bone-length 3D fix is a NO-GO**
Video-level holdout (videos `4ORRiN2_aVI`, `SZU7Sbgu57o`):

| variant | macro | overall |
|---|---|---|
| **rawz (current production convention)** | **54.6%** | 71.4% |
| bonecorr (the bone-length 3D fix) | 49.7% | 70.8% |
| zeroz | 44.8% | 55.8% |

Bone-length correction did **not** beat raw z. This is the third independent 3D
approach to fail on this project (after raw world-landmarks at 42.9% and the
pretrained single-frame 2D→3D lifter at 10%), which is itself a reportable
finding: monocular depth for yoga does not appear rescuable by these methods.
`relab_bonecorr` (transition-relabelled) did not report — re-run if needed.

### 2c. The honest headline about the 23-class goal
The **52.6%** figure applies to the **6 well-supported live poses**. Across the
broader vocabulary on real photos, macro is **~26%**, with most rare classes at
0% (chair_pose, chaturanga, corpse, plank, seated_forward, seated_staff all 0%).
**The "all 23 classes reliable" goal is NOT met**, and augmentation was not the
lever that gets there. Per-class n is 1-5 photos, so individual class numbers are
weak — but the overall picture is consistent and should not be overstated.

---

## 3. TWO BUGS CAUGHT IN THE TRAINING WORK ITSELF

1. **ST-GCN checkpoints are not production-loadable as trained.** They name the
   skip connection `block1.res.*`; production `sequence.py` expects
   `block1.residual.*`. Same architecture (still a genuine ST-GCN), pure attribute
   rename, but `load_state_dict` fails outright. A `.res.`→`.residual.` key remap
   is in `scratchpad/modal_eval_all.py` and is required before any deployment.
2. **A label-normalisation bug in my own first eval** — scoring against the raw
   `imperfect_pose_label` column made `imperfect_upward_dog` (n=687) auto-fail at
   0%. The pipeline strips the `imperfect_` prefix. That first v3_aug reading was
   unfairly low and was discarded, not reported.

---

## 4. THE MODAL LESSON (cost two training runs)
`modal run` creates an **ephemeral** app tied to the launching client. When both
training agents were killed by a session rate limit, **Modal killed the training
with them**; only Volume-written checkpoints survived.
**Always `modal run --detach` for long jobs.** Verify with `modal app list` —
State must not read `ephemeral` for anything expected to outlive the session.

---

## 5. ARTIFACTS (nothing deleted; all additive)
- Volume `asanaai-mlp-v3`: `mlp_3head_model_v3_aug.pth` + encoder,
  `eval_results_full.json`, `eval_photos_fair.json`, `raw/` (dataset),
  `testset/` (26 real-world photo landmark sets + manifest).
- Volume `asanaai-stgcn`: `stgcn_rawz.pth`, `stgcn_zeroz.pth`,
  `stgcn_bonecorr.pth`, `stgcn_relab_bonecorr.pth`, all feature arrays and
  encoders, `relab_*` (transition-relabelled data).
- HF `Arko007/yoga-posture-models`: `mlp_3head_model_v2.pth` (LIVE),
  `mlp_3head_model_v3_aug.pth` NOT uploaded (it lost).

---

## 6. NEXT SESSION — highest value first
1. **The weak classes need data, not augmentation.** Augmentation is now
   empirically ruled out. `child_pose`, `warrior_1`, and every 0% rare class need
   genuinely more/better real training examples, or should be honestly scoped out
   of the live vocabulary.
2. **Re-run `relab_bonecorr`** (transition-relabelled ST-GCN) — it never reported,
   and it is the one variant that tests whether teaching the ST-GCN real
   `hold:`/`transition:` classes works. That hypothesis is still open.
3. If any ST-GCN variant is ever promoted, note it changes the class vocabulary
   and therefore **breaks the `SequenceResponse` contract**
   (`sequence_pose`/`confidence`/`requires_static_fallback`) and
   `SEQUENCE_FALLBACK_THRESHOLDS` in `backend/app/routers/pose.py`. Migrate it
   properly; do not half-apply.
4. Consider updating the paper's 52.6% framing to state explicitly that it covers
   the 6 live poses, with the ~26% broader-vocabulary figure reported alongside —
   an examiner will otherwise read it as a whole-system claim.

## STANDING RULES
- **Arko007 HF account only**, never `bhumika-tewari-282006`. Token at
  `Downloads/API_Keys_and_Secrets/hf_token`; Modal secret `arko007-hf-token` →
  env `HF_TOKEN_ARKO007`.
- **Never delete existing artifacts**; new filenames only.
- **Never promote to a live filename without explicit user sign-off.**
- Heavy compute on Modal/Kaggle only — the local PC has 3.7GB RAM and is loaded.
- Local `~/.kaggle/kaggle.json` is the **`arkosarkarhehe`** account and is in
  active use by another session — do not swap it. The `anamitrasarkar007` creds
  are at `Downloads/API_Keys_and_Secrets/kaggle.json`, for use inside Modal only.
