---
license: other
pretty_name: Yoga-1M (AsanaAI)
tags: [yoga, pose-estimation, mediapipe, skeleton, st-gcn]
---
# Yoga-1M — AsanaAI skeleton dataset (RCC Institute of Information Technology, project P05)

**1,206,391 frames from 24 yoga videos**, stored as MediaPipe Pose landmarks (no video frames or images are redistributed here),
with the 15 biomechanical joint angles used by the 3-head ResMLP and the 33-joint sequences used by the ST-GCN.

## How the labels were made (and why they replace the older rule-derived labels)
A frame is a labelled pose only if ALL three hold: (1) the instructor *named* the pose (Whisper-large-v3 transcripts, English + Hindi,
Sanskrit and Devanagari names), within [-1 s, +30 s]; (2) the app's own rule engine (`rules_classifier.classify_pose` with orientation cues)
agrees the body is in that pose; (3) the pose is held steady (mean |angle change| < 8 deg over +-7 frames). Moving frames and steady
frames with no recognised pose are `transition/unknown`; every ambiguous frame is `__ignore__` (excluded, neither positive nor negative).
Of the 1.21M frames: ~80k are cue-verified poses, ~421k transitions, ~705k excluded.

## Contents (`vol/` mirrors the working volume)
`vol/csv/cue/cue_full.csv` all frames (zero-z angles = what the deployed app feeds the model) · `vol/csv/cue2/*` MLP training sets (variants) ·
`vol/cue` per-video label arrays · `vol/transcripts` Whisper transcripts · `vol/landmarks`, `vol/landmarks_new` raw landmarks ·
`vol/stgcn/cueT2_*` ST-GCN windows (train + held-out-video test) · `vol/photos` real-photo landmark corpora (public HF sets + cleaned
Commons/Openverse harvest) · `vol/eval`, `vol/folds` benchmark photos and folds · `code/` the exact pipeline and the unmodified original trainers.

## Evaluation protocol (what counts as a result)
Trainer-reported train/val accuracy is a random split and is NOT a result. Results are only: out-of-fold on 422 independent Commons photos,
held-out public photos (incl. out-of-vocabulary poses), and held-out *videos* for the ST-GCN. Pass bar per pose: recall >= 0.70 AND precision >= 0.70, n >= 10.
Models: https://huggingface.co/Arko007/asanaai-conference-runs
