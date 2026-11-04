---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: tabular-classification
language:
- en
tags:
- yoga
- posture-correction
- pose-estimation
- skeleton-action-recognition
- st-gcn
- mediapipe
- pytorch
datasets:
- Arko007/Yoga-1M
metrics:
- accuracy
- recall
model-index:
- name: AsanaAI pose cascade (namer + gate)
  results:
  - task:
      type: tabular-classification
      name: Per-frame yoga pose naming from 15 joint angles
    dataset:
      name: Held-out public photos (1,685, of which 1,037 show poses outside the vocabulary)
      type: Arko007/Yoga-1M
    metrics:
    - type: accuracy
      value: 78.9
      name: Overall accuracy
- name: AsanaAI ST-GCN (stgcn_target_v1)
  results:
  - task:
      type: video-classification
      name: Pose recognition from 60-frame skeleton windows
    dataset:
      name: Held-out videos (3 folds split by video)
      type: Arko007/Yoga-1M
    metrics:
    - type: recall
      value: 78.0
      name: Macro recall over 9 classes (fold models)
---

# AsanaAI model weights

The trained models behind **AsanaAI**, a yoga posture coach that names the pose you are in, scores your form and tells you what to adjust in English, Hindi or Bengali. Final-year project P05, Department of CSE-AIML, RCC Institute of Information Technology, Kolkata.

[Live app](https://yoga-posture-correction-system.vercel.app) · [API](https://arko007-yoga-pose.hf.space) · [Code](https://github.com/Anamitra-Sarkar/yoga-posture-correction-system) · [Training data (Yoga-1M)](https://huggingface.co/datasets/Arko007/Yoga-1M)

## Models in production

These three checkpoints are the ones the deployed API loads. Everything else in this repository is an earlier version, kept so that results can be reproduced and a deployment can be rolled back.

| Role | Weights | Encoder (class names) | Parameters | Size | Input | Output |
|---|---|---|---|---|---|---|
| **Gate**: vetoes "not one of my poses" and gives the form score | `mlp_3head_gate_v1.pth` | `mlp_3head_gate_v1_encoder.npy` | about 0.36 M | 1.5 MB | 15 joint angles | pose, correctness, 15 joint deviations |
| **Namer**: names the pose | `mlp_3head_v4_photos_x2000.pth` | `mlp_3head_v4_encoder.npy` | about 0.36 M | 1.5 MB | 15 joint angles | pose, correctness, 15 joint deviations |
| **Sequence model** (ST-GCN): second opinion on the last 2.4 s | `stgcn_target_v1.pth` | `stgcn_target_v1_encoder.npy` | 893,897 | 3.6 MB | 60 frames x 33 joints x (x, y, z) | 9 classes: 8 poses + `transition/unknown` |

**Pose MLPs (namer and gate).** Both are the same 3-head residual MLP: a 256-unit input layer (Linear, BatchNorm, GELU), two residual blocks, and three heads (pose logits, a correctness probability, 15 per-joint deviations). The 15 angles are computed on the device from MediaPipe landmarks with z set to zero.

**ST-GCN.** A real skeleton-graph network: graph convolution over the 33-joint MediaPipe skeleton (symmetric-normalised adjacency) followed by a 9-tap temporal convolution, three blocks (3 to 64 to 128 to 256 channels), then a 256 to 128 to C head. Windows are pelvis-centred and divided by hip width. The class is named `YogaSequenceLSTM` in `backend/app/models/sequence.py` for checkpoint compatibility only; it is not recurrent.

### How they work together

1. Every ~1.5 s the **namer** names the pose from the 15 angles.
2. The **gate** (a second MLP) decides whether the frame is one of our poses at all and supplies the form score; if it vetoes, the app says "not one of my poses" instead of guessing.
3. Per-joint corrections come from each pose's angle bands (17 of 19 poses have bands; tree and lunge do not).
4. About every 10 s the **ST-GCN** looks at the last 60 frames and is shown as a second opinion ("Flow check"), only when confident.

## Benchmarks

Every number below was measured on data the model never trained on. The validation accuracy printed by a trainer comes from a random split of neighbouring video frames and is not reported. A pose **passes** when recall and precision are both at least 0.70 on at least 10 held-out items (20 windows for the ST-GCN). A **false alarm** is a photo of a pose that is *not* one of the app's poses that the system names as one of them (lower is better).

### Per-frame pose cascade (namer + gate)

Held-out public photos: 1,685 photos, 1,037 of them poses outside our vocabulary.

| System | Overall accuracy | False alarms | Poses passing |
|---|---:|---:|---:|
| Previous production (v4 MLP + rule engine) | 36.9% | 78.3% | 0 |
| **Current cascade (namer + gate)** | **78.9%** | **20.8%** | **7** |

The seven poses that pass: chair, corpse, downward dog, tree, triangle, upward dog and warrior 2. The policy was chosen on a validation half only; on the untouched half the cascade scores 78.6% overall with 21.6% false alarms. Served through the live endpoint the result is 78.1% / 21.8% / 7 poses.

On the harder, 103-photo "wild" set of Wikimedia Commons photos (scored out-of-fold, none seen in training), the cascade scores **55.3%** against 37.9% for the previous production system.

| Decision policy (held-out public photos) | Overall | False alarms | Poses passing |
|---|---:|---:|---:|
| Rule engine only | 36.9% | 78.3% | 0 |
| Namer (v4) alone | 42.9% | 81.5% | 1 |
| Average of both MLPs | 65.2% | 47.7% | 5 |
| **Namer names, gate vetoes** | **78.9%** | **20.8%** | **7** |

### Form score (correctness head)

There are no human-labelled good/bad-form data, so this is a property test: 648 held-out photos x 5 trials, one joint broken by 45 degrees.

| Gate (current) | Result |
|---|---|
| Score on good form | 0.81 |
| Score drops when a joint is broken | 85.0% of trials (previous namer: 59.2%) |
| Deviation head names the broken joint first | 12.4% (chance 6.7%) |

The correctness score reacts to bad form. The per-joint deviation head is weak in every model we trained, so the app takes joint-level corrections from the angle bands, not from this head.

### ST-GCN sequence model

Held-out-video evaluation: three folds split by video, 8,002 windows of 60 frames, so the model is always scored on people and rooms it never saw. The shipped `stgcn_target_v1` was trained on all 24 videos, so it cannot be scored on held-out videos itself; the figures below come from its three fold models, which share its recipe.

Same windows for both models: the 12 videos recorded after the previous model was trained (2,682 windows), so the test is clean for both. Every output is mapped to the same 8 poses + `transition/unknown`.

| | Overall | Macro recall | Share of true-pose windows answered / precision of answers |
|---|---:|---:|---|
| Previous live model (`stgcn_transitions_v1`) | 68.9% | 18.8% | 2.8% / 93.8% |
| **Current model (`stgcn_target_v1`)** | **76.9%** | **82.1%** | **50.5% / 88.1%** |

"Answered" means the confidence reached the threshold at which the app shows the sequence result. The previous model recognised held poses on unseen videos almost never; the current model does.

On all 24 videos (three fold models, each scored only on videos it never trained on), the current model's overall accuracy is 82.2% and its macro recall over the 9 classes is 78.0%. Three poses pass the bar: child pose, corpse and downward dog. The per-pose results on all 24 videos:

| Pose | Windows | Recall | Precision | Result |
|---|---:|---:|---:|---|
| child pose | 1,045 | 0.74 | 0.91 | pass |
| corpse | 230 | 1.00 | 0.85 | pass |
| downward dog | 682 | 0.83 | 0.77 | pass |
| seated easy | 510 | 0.68 | 0.81 | near miss |
| triangle | 528 | 0.64 | 0.95 | near miss |
| tree | 87 | 0.87 | 0.67 | near miss |
| warrior 2 | 89 | 0.62 | 0.26 | fail |
| plank | 30 | 0.77 | 0.18 | fail (small n) |
| transition/unknown | 4,801 | 0.87 | 0.85 | n/a |

The model is conservative: when it misses, it answers `transition/unknown` ("not sure") instead of naming another pose. With a threshold fitted on the other folds, four poses pass.

**Frame-rate contract.** The network expects 60 consecutive frames at about 25 fps (2.4 s). Fed at 12.5 fps it keeps a macro recall of 0.81; at 6 fps or slower it falls to about 0.5 and below. The web app therefore buffers every camera frame with a timestamp and resamples to 25 fps before calling the model.

### How it compares with standard methods

Same 15 angles, same training rows, same metric code (held-out public photos, then the 103-photo wild set).

| Method | Public photos: overall | False alarms | Wild photos: overall |
|---|---:|---:|---:|
| k-NN | 58.6% | 55.7% | 50.5% |
| SVM (RBF) | 58.4% | 56.2% | 52.4% |
| Plain MLP (256-256-128) | 80.9% | 19.1% | 47.6% |
| Random forest | 85.3% | 9.8% | 35.0% |
| **Cascade (current)** | 78.9% | 20.8% | **55.3%** |

A random forest is better than the cascade on photos from the same sources as the training photos and the weakest method on the wild photos; the cascade is the only method with both a wild-photo score of at least 55% and false alarms at or below about 21%. The wild set has 103 photos, so gaps under about 10 points are within noise. For sequences, a plain MLP on the window mean and standard deviation matches or exceeds the ST-GCN on held poses (macro recall 94.4% against 86.4%), so we describe the ST-GCN as at parity with simpler models, not superior; its input also carries the depth coordinate, whose removal costs the ST-GCN the most accuracy of any model tested. Full tables are in the research paper and `docs/BENCHMARKS.md` in the code repository.

### Speed and size

Server compute per request: 7 to 9 ms median, 8 to 11 ms at the 95th percentile, with all three models loaded (324 MB peak memory, CPU only). Total weights: 6.6 MB.

## Training data

| Model | Trained on |
|---|---|
| Gate | `vol/csv/cue2/vH/mlp_all.csv` in [Yoga-1M](https://huggingface.co/datasets/Arko007/Yoga-1M): cue-verified frames from 24 instructor videos, plus photo landmarks |
| ST-GCN | `vol/stgcn/cueT2_all/` in Yoga-1M: 60-frame windows from the same 24 videos, with mirrored windows and photo-hold clips |
| Namer | An earlier 12-video table plus public and Wikimedia photo sets, with photos repeated 2,000 times; this older table is not part of Yoga-1M |

"Cue-verified" means a frame is labelled with a pose only if the instructor named it (Whisper transcript), the rule engine agrees, and the body is held steady. Details, file guide and cautions are on the dataset card.

## Usage

The models are used through the backend in the code repository (`backend/app/models/mlp.py`, `backend/app/models/sequence.py`), which also applies the input preparation: angles in the fixed order of `FEATURE_NAMES`, and for the ST-GCN pelvis centring and hip-width scaling of a 60 x 99 window.

```python
import numpy as np, torch
from huggingface_hub import hf_hub_download
from app.models.mlp import Yoga3HeadMLP          # from the code repository
from app.models.sequence import YogaSequenceLSTM  # the ST-GCN (name kept for checkpoint compatibility)

R = "Arko007/yoga-posture-models"
classes = [str(c).replace("imperfect_", "") for c in np.load(hf_hub_download(R, "mlp_3head_gate_v1_encoder.npy"), allow_pickle=True)]
gate = Yoga3HeadMLP(input_dim=15, num_poses=len(classes))
gate.load_state_dict(torch.load(hf_hub_download(R, "mlp_3head_gate_v1.pth"), map_location="cpu")); gate.eval()

seq_classes = [str(c) for c in np.load(hf_hub_download(R, "stgcn_target_v1_encoder.npy"), allow_pickle=True)]
stgcn = YogaSequenceLSTM(99, 128, 2, len(seq_classes))
stgcn.load_state_dict(torch.load(hf_hub_download(R, "stgcn_target_v1.pth"), map_location="cpu")); stgcn.eval()
```

## Limitations

* **Single seed and small test sets.** Every model was trained once; the wild photo set has 103 photos and some per-pose test sets have fewer than 30 items.
* **Photos, not live people.** The benchmarks use photographs and recorded videos; a study with live users and instructor ratings has not been run.
* **Form feedback is partly rule-based.** The correctness score has been validated only by the perturbation test above, the deviation head is not reliable, and arm errors are barely detected; joint-level corrections come from angle bands.
* **Pose coverage.** Seven poses clear the bar on held-out photos and three on held-out videos. Mountain, cobra and child's pose are weak on real phone photos.
* **Sequence model.** It recognises held poses; it does not model movement, because no labelled transition data exist. Its depth input is MediaPipe's monocular relative z, which can differ between cameras.
* **Not medical advice.** AsanaAI is a practice aid, not a physiotherapy tool.

## Earlier versions (superseded; kept for reproducibility and rollback)

| Weights | What it was | Status |
|---|---|---|
| `mlp_model.pth`, `mlp_label_encoder.npy` | First single-head ResMLP | Superseded |
| `mlp_3head_model.pth`, `mlp_3head_model_v2.pth` (+ `mlp_3head_pose_encoder*.npy`) | First 3-head MLPs trained on rule-labelled video | Superseded |
| `mlp_3head_photodomain_v1.pth` | Photo-trained, pose head only | Superseded |
| `stgcn_sequence_model.pth`, `stgcn_sequence_model_v2.pth` (+ `stgcn_label_encoder*.npy`) | Earlier sequence models | Superseded |
| `stgcn_transitions_v1.pth` | Previous live ST-GCN (30 classes: holds and named transitions) | Superseded by `stgcn_target_v1` |
| `ieee_research_paper.pdf` | Early draft of the paper | Outdated; the current paper is in the code repository |

The accuracy figures printed in the earlier version of this card (92.84%, 93.38%, 96.81%, 75.25%) came from random validation splits and are not valid measures of generalisation; they have been removed.

## Licence and citation

CC BY-NC-SA 4.0 (non-commercial, share-alike). Please credit the project.

Sarkar, A., Barman, D. D., Mondal, A. R., Samanta, T., and Chakraborty, S. (supervisor). *AsanaAI: A Privacy-Preserving, Occlusion-Aware and Safety-Screened Yoga Posture Coach for Commodity Hardware.* RCC Institute of Information Technology, Kolkata.
