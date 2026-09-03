# AsanaAI Session Handoff — 2026-09-03

## ⚠️ READ FIRST: the Modal lesson that cost us two training runs

Both training agents launched their Modal jobs with `modal run`, which creates an
**ephemeral** app tied to the launching client process. When both agents were
killed by a session rate limit (resets 15:30 IST), **Modal killed the training
with them**. Checkpoints survived only because they were written to Volumes.

**Always use `modal run --detach` (or `modal deploy` + `.spawn()`) for anything
long-running.** Verify with `modal app list`: the State column must not say
`ephemeral` for a job you expect to outlive your session.

## Everything that shipped to production today (verified live, not assumed)

1. **Root-caused the project's central defect.** The training feature extractor
   (`/home/anamitra/Projects_and_Code/Scripts_and_Source/extract_features_safe.py`)
   computed the 15 angle features from MediaPipe's raw **un-zeroed z**, while the
   deployed inference path always zeroes z. The MLP was trained on a feature
   distribution it never saw in production. Fixed (line ~101, `pts[:, 2] = 0.0`),
   dataset regenerated, model retrained → **real-world accuracy ~0-6% → 52.6%**
   (n=19 real photos), which also beats the 2D rule engine's 45.7%. Live via
   `hf_loader.py` now loading `mlp_3head_model_v2.pth`.
   *Why no metric caught it*: validation reuses the training-time feature code, so
   it inherits the same convention and certifies a model that can't reproduce its
   accuracy in deployment. Only differential auditing of the two code paths finds it.

2. **`warrior_1`/`warrior_2` branch-order bug.** The disabled `warrior_1` branch was
   checked before the live `warrior_2`, identical leg conditions, overlapping arm
   bands (shoulder 110-125 deg) → real warrior_2 attempts were silently discarded as
   `transition/unknown`. Fixed in `rules_classifier.py` AND in the training-label
   generator `experiments/classify_all_movements.py` (it corrupted labels too).

3. **A deploy pipeline broken for a month.** One rotated Arko007 HF token broke
   three things at once: the Kaggle training job's uploads (the 401s in the logs),
   the Modal `arko007-hf-token` secret, and the GitHub Actions `HF_TOKEN` secret —
   the last of which had silently blocked **every backend deploy since Aug 5**. All
   three repaired; Space verified rebuilt and serving current code.

4. **Free-form redesign (the user's product vision), shipped and live-verified.**
   - Pose selection removed; the target-vs-detected mismatch guard deleted entirely.
   - **Real transition detection**: `transition/unknown` conflated "moving between
     poses" with "holding something unrecognised" — opposite responses required.
     Mean absolute angular velocity over a 1.5s window (threshold 15 deg/s, sits in
     an empty gap between MediaPipe jitter ~few deg/s and real transitions
     ~hundreds deg/s) splits them into `holding` / `transitioning` / `unrecognized`.
     Corrections suppressed while moving, like a real instructor.
   - **Dual correctness**: universal + calibration-personalised scores, both
     surfaced. Moved server-side so the Expo mobile client inherits it.
   - Joint colouring now uses all 15 model deviations, not 3 hand-authored cues.
   - Asana library/reference content kept, now keyed to the DETECTED pose.
   - **All four states verified live** on the production endpoint, incl. personal
     0.42 > universal ~0 with correct per-joint forgiveness.

5. **Paper updated to match** (`/home/anamitra/yoga research/ieee_research_paper.tex`,
   compiles clean, 16 pages, no undefined refs). Replaced the now-removed
   mismatch-guard section with `sec:mismatch` (open-set practice + motion-state
   disambiguation, Eqs. for angular velocity and the 3-way state) and
   `sec:dualscore`; added `sec:zerozfix` documenting the train/inference mismatch
   with Table `tab:zerozfix`; corrected the stale disabled-pose and pose-count
   claims. **The paper is NOT in git** — it lives outside the repo, back it up.

## Results obtained today

### MLP v2 (currently live) — proper video-level holdout, no leakage
Holdout videos: `4ORRiN2_aVI`, `SZU7Sbgu57o`. **Macro 75.8%, overall 76.3%** (18 classes).

Strong: mountain_pose 99.7, upward_salute 98.6, upward_dog 97.8, table_top 96.8,
standing_forward_fold 92.9, halfway_lift 91.1, seated_easy 89.4, downward_dog 83.8.
Weak: **child_pose 22.0**, **warrior_1 46.4**, lunge 63.0, warrior_2 68.1,
standing_pose 68.9, transition/unknown 72.3, seated_staff 70.8 (n=48),
chair_pose n=2 (meaningless).

Note this is a *different metric* from the 52.6% real-world-photo number — video
frames vs arbitrary internet photos. The gap between them IS the domain shift.

### Trained but NOT YET EVALUATED (checkpoints safe on Modal volumes)
- Volume `asanaai-mlp-v3`: `mlp_3head_model_v3_aug.pth` + encoder (augmented:
  mirroring / rotation / scale / landmark noise + class balancing). **No evaluation
  was written** — the agent died first. Do not claim augmentation helped until this
  is measured against the same holdout.
- Volume `asanaai-stgcn`: **four** variants — `stgcn_rawz.pth`, `stgcn_zeroz.pth`,
  `stgcn_bonecorr.pth` (the bone-length 3D fix), and `stgcn_relab_bonecorr.pth`
  (trained on relabelled `hold:` / `transition:` classes), plus all feature arrays
  (`feats_rawz/zeroz/bonecorr.npy`, `relab_*.npy`) and encoders. **No eval results
  saved.** STGCN baseline log reached ep055/60.

## The next session's job (in priority order)

1. **Evaluate what's already trained** — this is cheap and the highest value left.
   Run detached on Modal: v3_aug vs v2 on the same video-level holdout, and the four
   STGCN variants against each other. The checkpoints and features are all sitting
   on the volumes; nothing needs regenerating.
2. The bone-length correction hypothesis (`stgcn_bonecorr`) is the genuine 3D fix and
   is *untested* — it must beat both 45.7% (2D-rules) and 42.9% (world-landmarks)
   to be adopted. An honest NO-GO is a valid, reportable outcome, exactly like the
   2D→3D lifter NO-GO already on record.
3. **`child_pose` at 22% and `warrior_1` at 46%** are the clear weak points to attack
   with the augmented data.
4. Watch for: the STGCN relabelling changes the class vocabulary to `hold:`/
   `transition:` strings, which **breaks the current `SequenceResponse` contract**
   (`sequence_pose` / `confidence` / `requires_static_fallback`) and
   `SEQUENCE_FALLBACK_THRESHOLDS` in `backend/app/routers/pose.py`. Don't half-apply
   it — either migrate the contract properly or keep the old vocabulary.

## Standing project rules (do not violate)
- **Arko007 HF account only**, never `bhumika-tewari-282006`. Token:
  `/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token`; Modal secret
  `arko007-hf-token` → env `HF_TOKEN_ARKO007`.
- **Never delete existing model artifacts** — new filenames only.
- **Never promote to a live filename without the user's explicit sign-off** (an
  attempt was correctly blocked by a safety classifier today).
- All heavy compute on Modal/Kaggle, never the local PC (3.7GB RAM, heavily loaded).
- The local `~/.kaggle/kaggle.json` is now the **`arkosarkarhehe`** account, not
  `anamitrasarkar007`; another session uses the Kaggle CLI actively, so don't swap
  it — the old account's creds are at `Downloads/API_Keys_and_Secrets/kaggle.json`
  and should be used inside Modal only.
