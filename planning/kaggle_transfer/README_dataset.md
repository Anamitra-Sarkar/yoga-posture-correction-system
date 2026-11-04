---
license: other
license_name: research-use-derived-landmarks
pretty_name: Yoga-1M (AsanaAI)
size_categories: [1M<n<10M]
tags: [yoga, pose-estimation, mediapipe, skeleton, st-gcn, joint-angles, human-pose]
---

# Yoga-1M: AsanaAI skeleton dataset

Training and evaluation data for **AsanaAI**, a yoga posture coach (final-year project P05, Department of CSE-AIML, RCC Institute of Information Technology, Kolkata).
App: https://yoga-posture-correction-system.vercel.app · Code: https://github.com/Anamitra-Sarkar/yoga-posture-correction-system · Trained models (public): https://huggingface.co/Arko007/yoga-posture-models

**1,206,391 frames from 24 instructor-led yoga videos**, stored as MediaPipe Pose landmarks and the 15 joint angles computed from them, plus landmark corpora from photographs.
**No video frames and no images are redistributed here** (landmarks and numbers only), and the Whisper transcripts used for labelling are not included.

## Quick start: which files do I need?

| I want to... | Use |
|---|---|
| Train or inspect the pose model on video frames | `vol/csv/cue2/vH/mlp_all.csv` (the table the shipped 3-head ResMLP and all paper baselines were trained on) |
| See every frame of every video with its label | `vol/csv/cue/cue_full.csv` |
| Train the skeleton-graph temporal model (ST-GCN) | `vol/stgcn/cueT2_*` (60-frame windows) |
| Start from raw landmarks (build your own features) | `vol/landmarks/` + `vol/landmarks_new/` |
| Evaluate on photos | `vol/eval/photo_corpus.npz` (Commons-422), `vol/photos/public_corpus.npz` (6,616 public photos with a fixed train/test split) |
| Re-run the pipeline | `code/` |

## How the video labels were made

A frame is a labelled pose only if **all three** hold: (1) the instructor *named* the pose (Whisper-large-v3 transcripts in English and Hindi, Sanskrit and Devanagari names) within [-1 s, +30 s]; (2) the app's rule engine (`rules_classifier.classify_pose`, with orientation cues) agrees the body is in that pose; (3) the pose is held steady (mean absolute angle change < 8 degrees over +-7 frames).
Moving frames, and steady frames with no recognised pose, are `transition/unknown`. Every ambiguous frame is `__ignore__`: it is neither a positive nor a negative and **must be dropped** before training or scoring.
Of the 1.21M frames: ~80k are cue-verified poses, ~421k transitions, ~705k excluded.

Cue-verified positive frames from video, per pose (and in how many of the 24 videos the pose has support):

| Pose | Frames | Videos | | Pose | Frames | Videos |
|---|---:|---:|---|---|---:|---:|
| child_pose | 7,194 | 10 | | warrior_2 | 1,002 | 5 |
| downward_dog | 6,358 | 14 | | tree_pose | 735 | 2 |
| lunge_pose | 5,440 | 10 | | plank | 598 | 7 |
| seated_easy_pose | 4,016 | 8 | | cobra_pose | 473 | 8 |
| triangle | 3,947 | 9 | | upward_dog | 375 | 6 |
| standing_forward_fold | 3,606 | 12 | | warrior_1 | not kept (2 videos) | 2 |
| mountain_pose | 2,689 | 7 | | table_top | 1,931 | 9 |
| corpse | 1,598 | 5 | | | | |

Poses backed by only a few videos (tree_pose, warrior_1, corpse, warrior_2) are weak evidence for generalisation to new people. The photo corpora supply variety the 24 videos cannot.

## File guide and schemas

**`vol/csv/cue/cue_full.csv`**: all 1,206,391 frames. Columns: `video_id, frame_num, difficulty, elbow_l, elbow_r, shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r, trunk_l, trunk_r, neck, hip_abduct_l, hip_abduct_r, pose_label, imperfect_pose_label`. The 15 angles are in degrees and computed with **z set to zero**, which is exactly what the deployed app feeds the model. Use `pose_label`; `difficulty` (always `cue` here) and `imperfect_pose_label` are legacy columns kept for compatibility with the original trainers.
`vol/csv/cue/mlp_all.csv` and `mlp_fold0..4.csv` are the subsampled training tables derived from it (21 classes, 96,492 rows).

**`vol/csv/cue2/<variant>/mlp_all.csv`, `mlp_fold0..4.csv`**: MLP training tables that add public photos and out-of-vocabulary negatives to the video rows, with 5 folds (`assemble_report.json` in each folder gives the exact recipe and class counts). **`vH` is the variant used for every reported model and baseline** (136,506 rows, 23 classes: video positives, negatives at a 0.3 ratio, public training photos repeated twice, 2,945 out-of-vocabulary rows). `vA, vB, vC, vM, vN, vT` are assembly ablations and were not used for reported results.

**`vol/cue/<video_id>.npz`**: per-video frame arrays: `labels` (final label, includes `__ignore__`), `rule` (what the rule engine said alone), `stable` (held-steady flag).

**`vol/landmarks/landmarks_<video_id>.npy`, `vol/landmarks_new/...`**: float32 `(frames, 33, 4)` = MediaPipe x, y, z, visibility per landmark. `landmarks/` holds the 12 original videos, `landmarks_new/` the 12 added later. The z coordinate is MediaPipe's monocular *relative* depth, not metric depth.

**`vol/stgcn/cueT2_f0, _f1, _f2`**: 3 folds split **by video** (a video never appears on both sides). Each folder has `stgcn_master_feats.npy` + `stgcn_master_labels.npy` (training windows) and `test_feats.npy` + `test_labels.npy` (windows from the held-out videos). **`cueT2_all`** is every video, used to train the shipped model (39,416 windows, 9 classes: 8 poses + `transition/unknown`). Windows are float32 `(N, 60, 99)`: 60 frames x 33 joints x (x, y, z). Windows overlap (stride 12 of 60), so the effective sample size is the number of holds, not the number of windows.

**`vol/eval/photo_corpus.npz`** (Commons-422): 422 freely licensed Wikimedia photos as landmarks (`landmarks`, `world`, `labels`, `split`). The test split is 103 photos (frozen by hash). **`vol/folds/photo_folds.json`**: out-of-fold assignment.

**`vol/photos/public_corpus.npz`**: 6,616 photos from two public Hugging Face datasets, `AdityasArsenal/Yoga-pose-Data-Set` (1,677) and `rotemvahava/yoga-poses-107` (4,939), after removing 348 duplicates; 22 labels (21 poses + `transition/unknown`, which stands for poses outside our vocabulary). Fixed split: 4,931 train / 1,685 held-out (1,037 of them "other poses"). `public_corpus_report.json` has the per-class counts.
**`vol/photos/harvest_kept.npz`** (563 rows) and `vol/photos/harvest/`: photos harvested from Wikimedia Commons and Openverse, filtered (label must agree with the rule engine or a model that never saw Commons; duplicates of the benchmark removed; `harvest_filter_report.json` counts every drop). Each photo's licence is in `harvest/*_meta.json` (CC BY, BY-SA and BY-NC variants occur).

**`code/`**: the exact pipeline (`modal/` assembly, labelling and training scripts; `backend/` the rule engine and model classes; `originals/` the unmodified original trainers; `runner_*.py` the Kaggle runners).

## What NOT to use, and cautions

* **Trainer-reported train/val accuracy is not a result.** It comes from a random frame split where near-identical neighbouring frames land on both sides (the ST-GCN trainer printed 98.4-98.7%). Only the held-out numbers below count.
* **`vol/labelsrc/`** (`mlp_dataset_cleaned.csv`, `mlp_dataset_cleaned_zero_z.csv`) holds the *older rule-derived* labels that the cue-verified labels replaced: a model trained on them can at best copy the rule engine. Kept for provenance only; do not train on it.
* **`__ignore__` frames** are ambiguous: drop them, and do not treat them as negatives.
* **Raw-z angles.** Angles computed with z (older tables) differ from the served zero-z angles by about 14 degrees (median per feature). Every table here that is meant for training uses the zero-z recipe; do not mix in angles from another recipe.
* **Public photos share sources between train and test.** The 1,685 held-out public photos come from the same two source datasets as the training photos (exact near-duplicates are rare: 0.3% within a tiny distance), so they are an in-distribution test. The harder test is Commons-422 / its 103-photo test split.
* **Small sample.** 24 videos with a limited number of instructors; a few poses have 2-5 supporting videos; some per-pose test sets have fewer than 30 items.
* **Movement is not labelled.** Transitions are marked only as `transition/unknown`; named transitions (e.g. cobra to downward dog) are not annotated, so the data cannot support claims about movement quality.

## Evaluation protocol (what counts as a result)

Results are only: out-of-fold on the Commons-422 photos, held-out public photos (including out-of-vocabulary poses), and held-out *videos* for the ST-GCN. Pass bar per pose: recall >= 0.70 and precision >= 0.70, with n >= 10 (>= 20 for video windows). Numbers and the full protocol: `docs/BENCHMARKS.md` in the code repository.

## Licence and use

The 24 source videos are third-party instructor videos and are **not** redistributed; this dataset contains only landmarks and angles derived from them, plus landmarks derived from public photographs. The source photo datasets do not declare a licence on their dataset cards, and the harvested photos carry Creative Commons licences that include non-commercial (BY-NC) and share-alike (BY-SA) terms. Please treat the dataset as **for research and education, non-commercial use**, and credit the project when you use it. If you are a rights holder and want something removed, contact the authors via the code repository.

## Citation

Sarkar, A., Barman, D. D., Mondal, A. R., Samanta, T., and Chakraborty, S. (supervisor). *AsanaAI: A Privacy-Preserving, Occlusion-Aware and Safety-Screened Yoga Posture Coach for Commodity Hardware.* RCC Institute of Information Technology, Kolkata. Code and models: see the links at the top.
