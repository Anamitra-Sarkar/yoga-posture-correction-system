---
license: other
license_name: research-use-derived-landmarks
pretty_name: Yoga-1M (AsanaAI)
size_categories:
- 1M<n<10M
tags:
- yoga
- pose-estimation
- mediapipe
- skeleton
- st-gcn
- joint-angles
- human-pose
configs:
- config_name: frames
  default: true
  data_files:
  - split: train
    path: vol/csv/cue/cue_full.csv
- config_name: mlp_vH
  data_files:
  - split: train
    path: vol/csv/cue2/vH/mlp_all.csv
---

<div align="center">

# Yoga-1M

### The AsanaAI skeleton dataset: 1.2 million labelled yoga frames as landmarks and joint angles

![frames](https://img.shields.io/badge/frames-1,206,391-4c6ef5) ![videos](https://img.shields.io/badge/videos-24-7950f2) ![features](https://img.shields.io/badge/joint%20angles-15-12b886) ![landmarks](https://img.shields.io/badge/landmarks-33%20(x%2Cy%2Cz%2Cvis)-fd7e14) ![privacy](https://img.shields.io/badge/images%20shared-none-success)

[App](https://yoga-posture-correction-system.vercel.app) · [Code](https://github.com/Anamitra-Sarkar/yoga-posture-correction-system) · [Trained models](https://huggingface.co/Arko007/yoga-posture-models) · [API](https://arko007-yoga-pose.hf.space)

*Final-year project P05 · Department of CSE-AIML · RCC Institute of Information Technology, Kolkata*

</div>

---

## At a glance

| | |
|---|---|
| **What it is** | MediaPipe Pose landmarks and 15 biomechanical joint angles for **1,206,391 frames** from **24 instructor-led yoga videos**, with cue-verified pose labels, plus landmark corpora from **photographs** |
| **What it trains** | A 3-head ResMLP (pose, correctness, joint deviation) and an ST-GCN (skeleton-graph temporal network) |
| **Privacy** | **No video frames and no images are included**: only landmarks and numbers. The Whisper transcripts used for labelling are not included either |
| **Label quality** | A frame is a labelled pose only if the instructor *named* it **and** the rule engine agrees **and** the pose is held steady |
| **Size** | 5.19 GB in 242 files (most of it the ST-GCN windows, which you can regenerate) |
| **Evaluation** | Held-out videos and held-out photos only. Random-split accuracy is *not* a result |

## Browse it in the viewer

The viewer above shows two tables (pick one in the config menu):

* **`frames`**: every frame of every video with its 15 angles and label (`vol/csv/cue/cue_full.csv`).
* **`mlp_vH`**: the exact table the shipped 3-head ResMLP and every baseline in the paper were trained on (`vol/csv/cue2/vH/mlp_all.csv`).

The landmark arrays, ST-GCN windows and photo corpora are NumPy files, which the viewer cannot display; load them as shown below.

## Quick start

```python
from datasets import load_dataset
frames = load_dataset("Arko007/Yoga-1M", "frames", split="train")      # all 1.2M frames
mlp    = load_dataset("Arko007/Yoga-1M", "mlp_vH", split="train")      # the shipped MLP's training table
```

```python
import numpy as np
from huggingface_hub import hf_hub_download

# 33 landmarks per frame: (frames, 33, 4) = x, y, z, visibility
lm  = np.load(hf_hub_download("Arko007/Yoga-1M", "vol/landmarks/landmarks_8ibxmzJziHU.npy", repo_type="dataset"))

# ST-GCN windows: (N, 60, 99) = 60 frames x 33 joints x (x, y, z)
X   = np.load(hf_hub_download("Arko007/Yoga-1M", "vol/stgcn/cueT2_f0/test_feats.npy",  repo_type="dataset"))
y   = np.load(hf_hub_download("Arko007/Yoga-1M", "vol/stgcn/cueT2_f0/test_labels.npy", repo_type="dataset"), allow_pickle=True)
```

## Which file do I need?

| I want to... | Use |
|---|---|
| Train or inspect the pose model on video frames | `vol/csv/cue2/vH/mlp_all.csv` (+ `mlp_fold0..4.csv` for cross-validation) |
| See every frame of every video with its label | `vol/csv/cue/cue_full.csv` |
| Train the skeleton-graph temporal model (ST-GCN) | `vol/stgcn/cueT2_*` (60-frame windows) |
| Build my own features from raw landmarks | `vol/landmarks/` and `vol/landmarks_new/` |
| Evaluate on photos | `vol/eval/photo_corpus.npz` (Commons-422), `vol/photos/public_corpus.npz` (6,616 public photos, fixed split) |
| Re-run the pipeline | `code/` |

## How the video labels were made

A frame receives a pose label only when all three conditions hold:

| # | Condition | How it is checked |
|---|---|---|
| 1 | The instructor names the pose | Whisper-large-v3 transcripts (English and Hindi; Sanskrit and Devanagari names), within [-1 s, +30 s] of the frame |
| 2 | The rule engine agrees | `classify_pose` with orientation cues puts the body in the same pose |
| 3 | The pose is held steady | Mean absolute angle change below 8 degrees over +-7 frames |

All other frames are assigned as follows:

| Frame | Label |
|---|---|
| Moving, or steady with no recognised pose | `transition/unknown` |
| Ambiguous | `__ignore__` (drop it) |

Of the 1.21M frames: **~80k** cue-verified poses · **~421k** transitions · **~705k** excluded.
No single source is trusted alone: an instructor may name a pose before or after the body gets there, and rule-engine labels alone would only teach a model to copy the rules.

<details>
<summary><b>Cue-verified positive frames per pose, and how many of the 24 videos support each pose</b></summary>

| Pose | Frames | Videos | | Pose | Frames | Videos |
|---|---:|---:|---|---|---:|---:|
| child_pose | 7,194 | 10 | | warrior_2 | 1,002 | 5 |
| downward_dog | 6,358 | 14 | | tree_pose | 735 | 2 |
| lunge_pose | 5,440 | 10 | | plank | 598 | 7 |
| seated_easy_pose | 4,016 | 8 | | cobra_pose | 473 | 8 |
| triangle | 3,947 | 9 | | upward_dog | 375 | 6 |
| standing_forward_fold | 3,606 | 12 | | table_top | 1,931 | 9 |
| mountain_pose | 2,689 | 7 | | corpse | 1,598 | 5 |
| warrior_1 | not kept | 2 | | | | |

Poses backed by only a few videos (tree_pose, warrior_1, corpse, warrior_2) are weak evidence for generalising to new people. The photo corpora supply the variety that 24 videos cannot.

</details>

## File guide and schemas

| Folder | Size | Contents |
|---|---:|---|
| `vol/csv/cue/` | 349 MB | `cue_full.csv` (all frames) and the subsampled training tables derived from it |
| `vol/csv/cue2/` | 609 MB | MLP training tables with photos added, 7 variants (use `vH`) |
| `vol/cue/` | 1.3 MB | per-video label arrays |
| `vol/landmarks/`, `vol/landmarks_new/` | 346 + 291 MB | raw landmarks, 12 original + 12 later videos |
| `vol/stgcn/` | 3.6 GB | ST-GCN windows: 3 video-split folds + all videos |
| `vol/eval/`, `vol/folds/` | 0.4 MB | Commons-422 photo benchmark and its fold assignment |
| `vol/photos/` | 8.6 MB | public photo corpus and harvested Commons/Openverse photos |
| `vol/labelsrc/` | 15.5 MB | **legacy** rule-derived tables (see cautions) |
| `code/` | 0.3 MB | the exact pipeline |

<details>
<summary><b>Video tables (<code>cue_full.csv</code>, <code>mlp_*.csv</code>)</b></summary>

Columns: `video_id, frame_num, difficulty, elbow_l, elbow_r, shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r, trunk_l, trunk_r, neck, hip_abduct_l, hip_abduct_r, pose_label, imperfect_pose_label`.

* The 15 angles are in degrees and computed with **z set to zero**, which is exactly what the deployed app feeds the model.
* Use `pose_label`. `difficulty` (always `cue` here) and `imperfect_pose_label` are legacy columns kept for compatibility with the original trainers.
* `vol/csv/cue/mlp_all.csv` and `mlp_fold0..4.csv` are the subsampled tables derived from `cue_full.csv` (21 classes, 96,492 rows).
* `vol/csv/cue2/<variant>/` adds public photos and out-of-vocabulary negatives to the video rows, in 5 folds; each folder's `assemble_report.json` has the exact recipe and class counts. **`vH` is the variant used for every reported model and baseline**: 136,506 rows, 23 classes (video positives, negatives at a 0.3 ratio, public training photos repeated twice, 2,945 out-of-vocabulary rows). `vA, vB, vC, vM, vN, vT` are assembly ablations and were not used for reported results.

</details>

<details>
<summary><b>Per-video arrays and landmarks</b></summary>

* `vol/cue/<video_id>.npz`: `labels` (final label, includes `__ignore__`), `rule` (what the rule engine said alone), `stable` (held-steady flag).
* `vol/landmarks/landmarks_<video_id>.npy` and `vol/landmarks_new/...`: float32 `(frames, 33, 4)` = MediaPipe x, y, z, visibility per landmark. The z coordinate is MediaPipe's monocular *relative* depth, not metric depth.

</details>

<details>
<summary><b>ST-GCN windows</b></summary>

* `vol/stgcn/cueT2_f0`, `_f1`, `_f2`: 3 folds split **by video** (a video never appears on both sides). Each has `stgcn_master_feats.npy` + `stgcn_master_labels.npy` (training windows) and `test_feats.npy` + `test_labels.npy` (windows from the held-out videos).
* `vol/stgcn/cueT2_all`: every video, used to train the shipped model: 39,416 windows, 9 classes (8 poses + `transition/unknown`).
* Windows are float32 `(N, 60, 99)`: 60 frames x 33 joints x (x, y, z). They overlap (stride 12 of 60), so the effective sample size is the number of holds, not the number of windows.

</details>

<details>
<summary><b>Photo corpora</b></summary>

* `vol/eval/photo_corpus.npz` (**Commons-422**): 422 freely licensed Wikimedia photos as landmarks (`landmarks`, `world`, `labels`, `split`); frozen test split of 103 photos (hash split). `vol/folds/photo_folds.json` gives the out-of-fold assignment.
* `vol/photos/public_corpus.npz`: 6,616 photos from two public Hugging Face datasets, `AdityasArsenal/Yoga-pose-Data-Set` (1,677) and `rotemvahava/yoga-poses-107` (4,939), after removing 348 duplicates. 22 labels (21 poses + `transition/unknown`, which stands for poses outside our vocabulary). Fixed split: 4,931 train / 1,685 held-out (1,037 of them "other poses"). `public_corpus_report.json` has per-class counts.
* `vol/photos/harvest_kept.npz` (563 rows) and `vol/photos/harvest/`: photos harvested from Wikimedia Commons and Openverse and filtered (the label must agree with the rule engine or with a model that never saw Commons; duplicates of the benchmark are removed; `harvest_filter_report.json` counts every drop). Each photo's licence is in `harvest/*_meta.json` (CC BY, BY-SA and BY-NC variants occur).

</details>

<details>
<summary><b>Code</b></summary>

`code/modal/` assembly, labelling and training scripts · `code/backend/` the rule engine and model classes · `code/originals/` the unmodified original trainers · `code/runner_*.py` and `code/kaggle_worker.py` the Kaggle runners.

</details>

## What NOT to use, and cautions

| | |
|---|---|
| **Do not use: Trainer-reported train/val accuracy** | It comes from a random frame split where near-identical neighbouring frames land on both sides (the ST-GCN trainer printed 98.4-98.7%). Only the held-out numbers count |
| **Do not use: `vol/labelsrc/`** (`mlp_dataset_cleaned*.csv`) | The *older rule-derived* labels that the cue-verified labels replaced; a model trained on them can at best copy the rule engine. Kept for provenance only |
| **Do not use: `__ignore__` frames** | Ambiguous: drop them, and do not treat them as negatives |
| **Caution: Raw-z angles** | Angles computed with z differ from the served zero-z angles by about 14 degrees (median per feature). Every training table here uses the zero-z recipe; do not mix in angles from another recipe |
| **Caution: Public held-out photos** | They come from the same two source datasets as the training photos (exact near-duplicates are rare, 0.3%), so they are an in-distribution test. The harder test is Commons-422 and its 103-photo test split |
| **Caution: Small sample** | 24 videos with a limited number of instructors; a few poses have 2-5 supporting videos; some per-pose test sets have fewer than 30 items |
| **Caution: Movement is not labelled** | Transitions are marked only as `transition/unknown`; named transitions (for example cobra to downward dog) are not annotated, so the data cannot support claims about movement quality |

## Evaluation protocol (what counts as a result)

Results are only: out-of-fold on the Commons-422 photos, held-out public photos (including out-of-vocabulary poses), and held-out *videos* for the ST-GCN. Pass bar per pose: **recall ≥ 0.70 and precision ≥ 0.70**, with n ≥ 10 (n ≥ 20 for video windows). Numbers and the full protocol are in `docs/BENCHMARKS.md` in the code repository.

## Licence and use

The 24 source videos are third-party instructor videos and are **not** redistributed; this dataset contains only landmarks and angles derived from them, plus landmarks derived from public photographs. The source photo datasets do not declare a licence on their dataset cards, and the harvested photos carry Creative Commons licences that include non-commercial (BY-NC) and share-alike (BY-SA) terms. Please treat the dataset as **for research and education, non-commercial use**, and credit the project when you use it. If you are a rights holder and want something removed, contact the authors through the code repository.

## Citation

Sarkar, A., Barman, D. D., Mondal, A. R., Samanta, T., and Chakraborty, S. (supervisor). *AsanaAI: A Privacy-Preserving, Occlusion-Aware and Safety-Screened Yoga Posture Coach for Commodity Hardware.* RCC Institute of Information Technology, Kolkata. Code and models: see the links at the top.
