# Pose cascade (2026-10-05)

**What:** an optional two-stage pose decision in `/api/analyse_frame`, behind `ENABLE_POSE_CASCADE` (default **off**).
Stage 1 (gate, `mlp_3head_gate_v1.pth`, a second 3-head MLP): "is this one of my poses at all?" and the form score.
Stage 2 (the existing live MLP): names the pose. The rule engine no longer overrides either model on the pose *name*; it supplies
per-joint deviations from each pose's angle bands.

**Why:** through `hybrid_classify` the rule engine wins every MLP/rules disagreement, so swapping in a better MLP changed nothing
(identical to rules-only). Measured on the real production functions, then through the real endpoint with the real models:

| Held-out public photos (1,685; 1,037 are poses outside the vocabulary) | overall | false alarms* | poses at >=70% recall AND precision |
|---|---|---|---|
| cascade OFF (production before) | 36.9% | 78.3% | 0 |
| cascade ON | 78.9% | 20.8% | 7 (chair, corpse, downward dog, tree, triangle, upward dog, warrior 2) |

*false alarm = a photo of a pose NOT in the vocabulary that is named as one of ours. Lower is better.
Policy chosen on a validation half only; confirmed on the untouched half (78.6% / 21.6%).
Wild 103-photo Commons set (neither model trained on it; gate scored out-of-fold): cascade 55.3% vs production 37.9-40.8% (the live v4 MLP alone scores 60.2%
there: the cascade gives up ~5 photos of pose-naming accuracy to remove ~60 points of false alarms elsewhere).
Latency p50 7-9 ms, p95 8-11 ms per request in-process; ~324 MB peak RSS with all three models.

**Not validated:** the correctness/deviation heads against labelled good-vs-bad form (no such labels exist). Only a perturbation test:
breaking one joint 45 deg lowers the gate's correctness 85% of the time (the live v4: 59%); the deviation head points at the broken joint only
12% of the time (chance 6.7%), which is why deviations come from the angle bands wherever a pose has them.

**Rollout / rollback:** add the checkpoint (`mlp_3head_gate_v1.pth` + `_encoder.npy`, new names in `Arko007/yoga-posture-models`) -> deploy with the switch off
-> set the Space variable `ENABLE_POSE_CASCADE=1`. Rollback = unset it (no redeploy). If the gate cannot load, the endpoint answers with the original
path and reports the reason in `cascade.reason`.

**API (all new fields optional; the Expo app is unaffected):** response `candidates` (top-3), `cascade`, `guided`; request `target_pose` (Guided mode).
Reproduce: `planning/kaggle_transfer/kernel_e2e`, `kernel_prod2`; data and code on `Arko007/Yoga-1M`, checkpoints on `Arko007/asanaai-conference-runs`.
