# Rescued model artifacts (2026-09-19)

Modal hit its spend limit, killing all compute there. Volume *reads* still
worked, so everything was pulled down before the account became unusable.
These are the only copies outside Modal.

## The one that matters
- `stgcn_transitions_v1.pth` + `_encoder.npy` — **the transition-aware ST-GCN**,
  the best temporal model measured on this project: macro **63.0%** over 24
  classes, named directional transitions at **72.7-100%**. Key names already
  remapped (`res.` -> `residual.`) and **strict-load verified against the real
  production class**, so the backend can load it as-is. Also uploaded to HF as
  `Arko007/yoga-posture-models: stgcn_transitions_v1.pth`.

## Losing variants, kept for the record
- `stgcn_rawz.pth` (54.6% macro, 15 classes) — the old scheme that never saw a
  transition as a learnable class.
- `stgcn_bonecorr.pth` (49.7%) — the bone-length 3D correction. LOST to rawz.
- `stgcn_zeroz.pth` (44.8%).
- `mlp_mlp_3head_model_v3_aug.pth` — landmark-augmented MLP. Did NOT help
  (24.2% vs 25.8% macro on real photos).

NOTE: these four still use the `res.` naming and need the same remap before
they could ever load in production.

## Results
- `eval_results_full.json` — ST-GCN variant comparison (video-level holdout).
- `mlp_eval_photos_fair.json` — the fair MLP comparison on real photos.
- `mlp_eval_results_full.json` — MLP video-holdout (NOTE: v2-vs-v3aug there is
  NOT a valid comparison; v2 trained on the holdout videos, v3_aug did not).
