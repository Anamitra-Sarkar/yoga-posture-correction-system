---
license: other
license_name: research-only-third-party-video-derivatives
language:
- en
pretty_name: Yoga-650k (historical master table)
size_categories:
- 100K<n<1M
task_categories:
- tabular-classification
tags:
- yoga
- pose-estimation
- mediapipe
- joint-angles
- historical
- legacy
configs:
- config_name: default
  data_files:
  - split: train
    path: master_mlp_dataset_fully_classified.csv
---

# 🧘 Yoga-650k: the historical master table

**654,488 per-frame rows of 15 joint angles from 12 instructor-led yoga videos.** This is the table the first version of the AsanaAI pose model was built from, and the table the **production "namer" MLP** (`mlp_3head_v4_photos_x2000`) was trained on, together with photo sets. It is kept public so that every number in the project report can be traced to its source.

> ⚠️ **Read before using.** This table is a historical record, not a clean benchmark. Its `pose_label` column is 99.36% `transition/unknown`, and its angles repeat for runs of consecutive frames where the original pipeline did not actually detect a person. For new work use the cue-verified successor, [**Yoga-1M**](https://huggingface.co/datasets/Arko007/Yoga-1M).

[Live app](https://yoga-posture-correction-system.vercel.app) · [Code](https://github.com/Anamitra-Sarkar/yoga-posture-correction-system) · [Models](https://huggingface.co/Arko007/yoga-posture-models) · [Yoga-1M](https://huggingface.co/datasets/Arko007/Yoga-1M) · [Forensic rebuild](https://huggingface.co/datasets/Arko007/yoga-skeleton-rebuilt)

---

## 📦 What is in this repository

| File | Size | Rows | Content |
|---|---:|---:|---|
| `master_mlp_dataset_fully_classified.csv` | 136 MB | 654,488 | One row per decoded video frame, 20 columns |

## 🗂️ Schema

| Column | Type | Meaning |
|---|---|---|
| `video_id` | string | YouTube id of the source video (12 distinct) |
| `frame_num` | int | Frame counter within the video |
| `difficulty` | string | `beginner`, `intermediate` or `advanced` (set per video) |
| `elbow_l`, `elbow_r`, `shoulder_l`, `shoulder_r`, `hip_l`, `hip_r`, `knee_l`, `knee_r`, `ankle_l`, `ankle_r`, `trunk_l`, `trunk_r`, `neck`, `hip_abduct_l`, `hip_abduct_r` | float | The **15 joint angles** in degrees (range 0 to 180), computed from MediaPipe Pose landmarks |
| `pose_label` | string | Strict label: 9 values, almost always `transition/unknown` |
| `imperfect_pose_label` | string | Rule-engine label: 29 values, including `imperfect_*` variants. **This is the column the production namer was trained on.** |

## 📊 Label distributions (measured from the file)

**`pose_label`** (strict): 650,304 rows (99.36%) are `transition/unknown`. The remaining 4,184 rows are spread over 8 classes: triangle 2,426; plank 872; seated forward 376; upward dog 334; corpse 102; child 33; chaturanga 25; warrior 2 16.

**`imperfect_pose_label`** (rule engine, 29 values). Top classes:

| Label | Rows | Share |
|---|---:|---:|
| transition/unknown | 223,903 | 34.2% |
| lunge_pose | 97,477 | 14.9% |
| seated_easy_pose | 87,325 | 13.3% |
| standing_forward_fold | 39,535 | 6.0% |
| warrior_2 | 35,518 | 5.4% |
| cobra_pose | 25,581 | 3.9% |
| warrior_1 | 23,678 | 3.6% |
| downward_dog | 21,992 | 3.4% |
| child_pose | 16,301 | 2.5% |
| imperfect_triangle | 13,035 | 2.0% |

The remaining 19 labels (tree, mountain, chair, table top, seated staff, upward salute, and their `imperfect_*` variants) each cover less than 2%.

**`difficulty`**: advanced 470,397 rows; beginner 145,680; intermediate 38,411. Rows are not independent: neighbouring frames of one video are near-duplicates.

---

## 🔬 Where this table sits in the project

```
12 YouTube videos ──► MediaPipe + rule engine ──► Yoga-650k (this table)
                                                     │
        photos (public corpora, ×2000 weight) ───────┴──► namer MLP  mlp_3head_v4_photos_x2000   (production)

24 videos ──► Whisper cue-verification ──► Yoga-1M ──► gate MLP + ST-GCN                         (production)
```

* The **namer** (names the pose) was trained on this table's `imperfect_pose_label` plus photo sets. It is one of the three production models.
* The **gate** and the **ST-GCN** were trained on the newer cue-verified data in Yoga-1M, not on this table.
* The builder script of this table is published in Yoga-1M under `code/originals/`.

## ⚠️ Known problems (do not hide these)

| Problem | Evidence | Consequence |
|---|---|---|
| Frames without a detected person are not marked | the forensic rebuild found that about 83% of rows carry MediaPipe output the original pipeline's own reports say was never detected; the first rows of the file show identical angles repeated frame after frame | A large share of rows are copies, not measurements |
| `pose_label` is almost entirely `transition/unknown` | 99.36% | Unusable as a training target on its own |
| Rule-engine labels are not verified | `imperfect_pose_label` comes from angle rules, with no check against what the instructor said | Label noise; this is why Yoga-1M adds Whisper cue verification |
| Train/serve angle mismatch | the table was built with 3-D landmark angles while the app serves 2-D angles (median gap about 14 degrees) | Models trained here need the corrections described in the report |
| Random-split accuracy is meaningless | near-identical neighbouring frames fall on both sides of a random split | Split by video, never by frame |

Full forensics: [`Arko007/yoga-skeleton-rebuilt`](https://huggingface.co/datasets/Arko007/yoga-skeleton-rebuilt) (`docs/FINDINGS_historical_dataset.md`).

## ✅ Recommended use

* Reproducing the training of the production namer, or auditing how it was made.
* Studying how label noise and duplicated frames inflate validation scores.

**Not recommended:** training a new pose classifier from `pose_label`, or reporting accuracy from a random split of this table.

## 📜 Licence and provenance

The numbers are derived from third-party YouTube instructional videos. The derived angle table is shared for **non-commercial research and education only**; the original videos remain the property of their creators (the raw files are in [`Arko007/yoga-dataset-raw`](https://huggingface.co/datasets/Arko007/yoga-dataset-raw) for provenance checking, not for redistribution).

## 📝 Citation

```bibtex
@misc{asanaai2026yoga650k,
  title  = {Yoga-650k: historical per-frame joint-angle table for AsanaAI},
  author = {Sarkar, Anamitra and Barman, Debjit Deb and Mondal, Adrija Ray and Samanta, Tanima},
  year   = {2026},
  note   = {Final-year project P05, RCC Institute of Information Technology, Kolkata. Supervisor: Sujit Chakraborty},
  url    = {https://huggingface.co/datasets/Arko007/Yoga-650k}
}
```
