---
license: cc-by-nc-sa-4.0
library_name: pytorch
pretty_name: AsanaAI conference runs (evidence archive)
tags:
- yoga
- posture-correction
- evaluation
- reproducibility
- st-gcn
---

# 🧪 AsanaAI conference runs: evidence archive (private)

Every evaluation result, training log and intermediate checkpoint behind the numbers in the AsanaAI **paper** and **project report**. Nothing here is loaded by the live app; this repository is the audit trail. The three models that are deployed live in [`Arko007/yoga-posture-models`](https://huggingface.co/Arko007/yoga-posture-models).

> 🔒 Private. Never delete or overwrite files here: results in the paper are cited by file name.

## 🗂️ Layout

| Folder | Content |
|---|---|
| `evals_compare/` | Held-out comparisons: baselines (k-NN, SVM, random forest, plain MLP; `baselines_frame_v1/v2.json`), the CLIFF-versus-mirror occlusion experiment, the namer + gate combination (`combo_v4_new*.json`), the frozen 103-photo set, the production-path and live-replay checks, and the head property test |
| `evals_mlp/` | Out-of-fold and public-photo scoring of the MLP variants |
| `evals_stgcn/` | ST-GCN held-out-video results, the old-versus-new comparison, the operating-point (threshold) cross-fit, and the endpoint parity check |
| `logs/` | Trainer logs of the final and live MLP and ST-GCN runs. The accuracies printed in these logs come from a random frame split and are **not** results |
| `runs/` | Training outputs per variant: `cue*_mlp_*` (frame models, folds `fold0..4` and `all`), `cue*_stgcn_*` (sequence models, folds `f0..f2` and `all`), diagnostics (`diag2.json`, `diag_csv_vs_landmarks.json`) |
| `backup_2026-10-05/` | Snapshot of docs, planning scripts and handoff notes at the time of the conference runs |

## 🏷️ Which runs became production

| Production model | Run folder here |
|---|---|
| Gate MLP `mlp_3head_gate_v1` | `runs/cueH_mlp_all` |
| ST-GCN `stgcn_target_v1` | `runs/cueT2_stgcn_all` |
| Namer MLP `mlp_3head_v4_photos_x2000` | not here (trained earlier on the historical table, see [Yoga-650k](https://huggingface.co/datasets/Arko007/Yoga-650k)); published in the models repository |

All other `runs/*` folders are experiment variants or cross-validation folds. Fold models are trained without the held-out videos and are the ones used to score generalisation; the `all` model is trained on everything and cannot be scored on held-out data.

## 📏 Reading the results

* Quote only **held-out** numbers: held-out public photos (1,685), the frozen 103-photo wild set, and held-out videos (three folds split by video).
* A pose **passes** at recall and precision of at least 0.70 on enough items.
* Result files are plain JSON and are summarised, with the interpretation, in `docs/BENCHMARKS.md` of the [code repository](https://github.com/Anamitra-Sarkar/yoga-posture-correction-system).
