---
language:
- en
license: cc-by-nc-sa-4.0
tags:
- yoga
- posture-correction
- pose-estimation
- pytorch
- resmlp
- gru
- attention
- computer-vision
- mediapipe
datasets:
- custom
metrics:
- accuracy
- f1
pipeline_tag: tabular-classification
model-index:
  - name: YogaResMLP (Single-Head Classifier)
    results:
      - task:
          type: tabular-classification
        dataset:
          name: yoga-pose-features-dataset
          type: custom
        metrics:
          - name: Validation Pose Accuracy
            type: accuracy
            value: 92.84
  - name: Yoga3HeadMLP (Multi-Output Model)
    results:
      - task:
          type: tabular-classification
        dataset:
          name: yoga-pose-features-dataset
          type: custom
        metrics:
          - name: Base Pose Identification Accuracy
            type: accuracy
            value: 93.38
          - name: Pose Correctness Accuracy
            type: accuracy
            value: 96.81
  - name: YogaSequenceLSTM (ST-GCN/Sequence Model)
    results:
      - task:
          type: sequence-classification
        dataset:
          name: yoga-pose-features-dataset
          type: custom
        metrics:
          - name: Flow Sequence Validation Accuracy
            type: accuracy
            value: 75.25
---

# Smart Yoga Posture Correction System (Project P05)
This repository hosts the model weights and label encoders for the **Smart Yoga Posture Correction System** (Final Year Project P05, RCC IIT Kolkata). 

The system leverages a multi-model cooperative framework to classify and correct yoga poses:
1. **Single-Head ResMLP Model (`mlp_model.pth`)**: A frame-level static posture classifier trained on 15 biomechanical joint angles, achieving **92.84%** validation accuracy across 29 classes.
2. **3-Head MLP Model (`mlp_3head_model.pth`)**: A multi-output static posture model predicting **Pose ID** (across 23 base classes, achieving **93.38%** pose accuracy), **Pose Correctness** (achieving **96.81%** accuracy), and **Joint Angle Deviations** (regression output) simultaneously.
3. **Sequence Flow Model (`stgcn_sequence_model.pth`)**: A hybrid 1D Temporal Convolution + Stacked Residual GRU + Self-Attention model trained on 60-frame skeleton coordinate sequences, achieving **75.25%** validation accuracy across 27 classes.

All models incorporate class-weight smoothing and normalization techniques to resolve pose imbalance and coordinate noise.

---

## Model Architectures & Training Logs

### 1. Static Pose Classifier (Single-Head ResMLP)

#### Architecture
The **ResMLP** classifier processes 15 frame-level joint angles (computed from MediaPipe Pose landmarks):
* **Input Layer**: `Linear(15 -> 256)` followed by Batch Normalization and `GELU` activation.
* **Residual blocks**: 2 stacked residual blocks. Each block consists of:
  * `Linear(256 -> 256)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.3)`
  * `Linear(256 -> 256)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.3)`
  * Residual skip connection: `x_out = x + block(x)`
* **Classification Head**: `Linear(256 -> 128)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.2)` -> `Linear(128 -> 29)`.

#### Dataset & Preprocessing
* **Dataset size**: 654,488 frames in total.
  * **Train size**: 523,590 frames
  * **Validation size**: 130,898 frames
* **Class Weights**: Smoothed using the square-root count inverse function `1.0 / sqrt(count)` to prevent minor classes (such as `transition/unknown` and `lunge_pose`) from dominating the gradients.

#### Training Performance & Curves
* **Best Validation Loss**: **0.1644** at Epoch 39.
* **Final Epoch (40/40)**:
  * **Train Loss**: 0.2238 | **Train Acc**: 90.78%
  * **Val Loss**: 0.1651 | **Val Acc**: 92.84%

Below is the training progress for selected epochs:
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| Epoch 01 | 0.6523 | 77.57% | 0.3930 | 83.94% |
| Epoch 02 | 0.4576 | 82.71% | 0.3231 | 86.79% |
| Epoch 03 | 0.4080 | 84.19% | 0.3005 | 87.17% |
| Epoch 04 | 0.3811 | 85.05% | 0.2700 | 88.29% |
| Epoch 05 | 0.3620 | 85.71% | 0.2756 | 87.24% |
| Epoch 10 | 0.3102 | 87.56% | 0.2421 | 89.24% |
| Epoch 20 | 0.2732 | 89.00% | 0.2091 | 90.78% |
| Epoch 30 | 0.2420 | 90.18% | 0.1872 | 91.57% |
| Epoch 39 | 0.2259 | 90.73% | **0.1644** | 92.66% |
| Epoch 40 | 0.2238 | 90.78% | 0.1651 | **92.84%** |

#### Static Pose Classification Report
```
                          precision    recall  f1-score   support

              chair_pose       0.56      0.94      0.70       366
              chaturanga       0.45      1.00      0.62         5
                   child       0.06      0.57      0.10         7
              child_pose       0.91      0.99      0.95      3260
              cobra_pose       0.90      0.96      0.93      5116
                  corpse       0.36      0.85      0.51        20
            downward_dog       0.90      0.95      0.92      4398
            halfway_lift       0.55      0.94      0.70       479
        imperfect_corpse       0.66      0.97      0.78       290
         imperfect_plank       0.86      0.96      0.91      1825
imperfect_seated_forward       0.87      0.99      0.92       938
      imperfect_triangle       0.87      0.96      0.91      2607
    imperfect_upward_dog       0.91      0.97      0.94      2556
              lunge_pose       0.97      0.93      0.95     19496
           mountain_pose       0.77      0.97      0.86      1233
                   plank       0.58      0.63      0.61       174
        seated_easy_pose       0.94      0.97      0.95     17465
          seated_forward       0.91      0.96      0.94        75
            seated_staff       0.80      0.94      0.86      1600
   standing_forward_fold       0.95      0.96      0.96      7907
           standing_pose       0.85      0.92      0.89      1405
               table_top       0.51      0.94      0.66       501
      transition/unknown       0.98      0.88      0.93     44781
               tree_pose       0.73      0.97      0.83      1474
                triangle       0.58      0.75      0.66       485
              upward_dog       0.42      0.60      0.49        67
           upward_salute       0.76      0.99      0.86       528
               warrior_1       0.94      0.98      0.96      4736
               warrior_2       0.88      0.95      0.91      7104

            weighted avg       0.94      0.93      0.93    130898
                accuracy                           0.93    130898
```

---

### 2. Multi-Output Posture Correction Model (3-Head MLP)

#### Architecture
The **3-Head MLP** classifier processes 15 frame-level joint angles (computed from MediaPipe Pose landmarks):
* **Shared Feature Trunk**:
  * Input layer `Linear(15 -> 256)` -> `BatchNorm1d` -> `GELU` activation.
  * 2 stacked residual blocks (`ResBlock` of size 256). Each block contains:
    * `Linear(256 -> 256)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.3)`
    * `Linear(256 -> 256)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.3)`
    * Skip connection: `x_out = x + block(x)`
* **Head 1: Pose ID (Classification)**:
  * `Linear(256 -> 128)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.2)` -> `Linear(128 -> 23)` (Softmax over 23 base posture classes).
* **Head 2: Correctness (Binary Classification)**:
  * `Linear(256 -> 64)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.2)` -> `Linear(64 -> 1)` (Binary Logit output: correct vs. imperfect/transition).
* **Head 3: Joint Deviation (Regression)**:
  * `Linear(256 -> 128)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.2)` -> `Linear(128 -> 15)` (Predicts normalized deviation values in $[0, 1]$ where 1.0 represents 180° deviation).

#### Dataset & Preprocessing
* **Dataset size**: 654,488 frames in total.
  * **Train size**: 523,590 frames
  * **Validation size**: 130,898 frames
* **Class Weights**: Smoothed using the square-root count inverse function `1.0 / sqrt(count)` to prevent major classes (such as `transition/unknown` and `lunge_pose`) from dominating the Pose ID loss gradients.
* **Loss Function**: $\mathcal{L}_{total} = \mathcal{L}_{pose} + \mathcal{L}_{correctness} + \mathcal{L}_{deviation}$ (combining Cross-Entropy, Binary Cross-Entropy with logits, and Huber SmoothL1 loss).

#### Training Performance & Curves
* **Best Validation Loss**: **0.2263** at Epoch 39/40.
* **Validation Pose Accuracy**: **93.38%**
* **Validation Correctness Accuracy**: **96.81%**

Below is the training progress for selected epochs:
| Epoch | Train Loss | Train Pose Acc | Val Loss | Val Pose Acc | Val Correctness Acc |
|---|---|---|---|---|---|
| Epoch 01 | 0.8631 | 79.04% | 0.5059 | 86.08% | 92.83% |
| Epoch 02 | 0.6321 | 83.78% | 0.4506 | 87.46% | 93.72% |
| Epoch 03 | 0.5702 | 85.20% | 0.3939 | 89.38% | 94.36% |
| Epoch 04 | 0.5321 | 86.14% | 0.3811 | 88.70% | 94.51% |
| Epoch 05 | 0.5055 | 86.71% | 0.3550 | 90.19% | 94.75% |
| Epoch 10 | 0.4389 | 88.33% | 0.3079 | 91.43% | 95.30% |
| Epoch 20 | 0.3864 | 89.72% | 0.2873 | 91.54% | 95.60% |
| Epoch 30 | 0.3597 | 90.42% | 0.2545 | 92.22% | 96.42% |
| Epoch 39 | 0.3224 | 91.36% | **0.2263** | **93.38%** | **96.81%** |
| Epoch 40 | 0.3215 | 91.37% | 0.2380 | 92.62% | 96.65% |

---

### 3. Sequence Flow Classifier (ST-GCN/GRU-Attention)

#### Architecture
The sequence classifier processes **60-frame coordinate sequences** (shape `[batch_size, 60, 99]`, representing 33 joints in 3D):
* **Coordinate Normalization**: Translates coordinate sequences to be **pelvis-centered** (using the midpoint between the left and right hip joints) and divides by hip-width. This guarantees absolute **translation and scale invariance**.
* **1D Temporal Convolution**: `Conv1d(in_channels=99, out_channels=128, kernel_size=5, padding=2)` -> `BatchNorm1d` -> `GELU` -> `Dropout(0.2)` to smooth coordinate sequence noise.
* **Stacked Residual GRU blocks**: Two bidirectional GRU blocks with hidden dimension 128. Output is projected back from 256 to 128, normalized with LayerNorm, dropped out with 30% rate, and summed with input (residual connection).
* **Self-Attention Pooling**: Learns step importance weights dynamically and returns a weighted summary vector across the 60-frame window.
* **Classification Head**: `Linear(128 -> 64)` -> `GELU` -> `Dropout(0.3)` -> `Linear(64 -> 27)`.

#### Dataset & Preprocessing
* **Total sequences**: 18,165 (60-frame windows).
  * **Train size**: 14,532 sequences.
  * **Validation size**: 3,633 sequences.
* **Training Hyperparameters**:
  * Batch Size: 64
  * Optimizer: `AdamW(lr=2e-3, weight_decay=1e-3)`
  * Target Metric: Best Validation Accuracy.

#### Training Performance & Curves
* **Best Validation Accuracy**: **75.25%** at Epoch 90.
* **Early Stopping**: Triggered at Epoch 110.

Selected epochs during training:
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| Epoch 01 | 3.7886 | 37.03% | 3.4506 | 45.09% |
| Epoch 02 | 3.4884 | 42.83% | 3.2596 | 49.55% |
| Epoch 10 | 2.9655 | 60.80% | 2.8399 | 64.35% |
| Epoch 20 | 2.7688 | 67.86% | 2.7106 | 68.98% |
| Epoch 30 | 2.6449 | 71.72% | 2.6624 | 69.83% |
| Epoch 50 | 2.5094 | 76.72% | 2.6160 | 72.20% |
| Epoch 90 | 2.3185 | 83.82% | 2.5877 | **75.25%** |
| Epoch 110 | 2.2777 | 85.25% | 2.6001 | 74.40% (Early Stopping) |

---

## Inference and Usage Guide

All model state dicts and label encoder maps can be downloaded and loaded in Python as follows:

```python
import numpy as np
import torch
import torch.nn as nn

# Load label encoders
mlp_classes = np.load("mlp_label_encoder.npy", allow_pickle=True)
mlp_3head_classes = np.load("mlp_3head_pose_encoder.npy", allow_pickle=True)
stgcn_classes = np.load("stgcn_label_encoder.npy", allow_pickle=True)

# 1. Instantiate the Single-Head ResMLP Model
mlp_model = YogaMLP(input_dim=15, num_classes=len(mlp_classes))
mlp_model.load_state_dict(torch.load("mlp_model.pth", map_location="cpu"))
mlp_model.eval()

# 2. Instantiate the 3-Head MLP Model
mlp_3head_model = Yoga3HeadMLP(input_dim=15, num_poses=len(mlp_3head_classes))
mlp_3head_model.load_state_dict(torch.load("mlp_3head_model.pth", map_location="cpu"))
mlp_3head_model.eval()

# 3. Instantiate the Sequence Model
sequence_model = YogaSequenceLSTM(input_dim=99, hidden_dim=128, num_layers=2, num_classes=len(stgcn_classes))
sequence_model.load_state_dict(torch.load("stgcn_sequence_model.pth", map_location="cpu"))
sequence_model.eval()
```

### Cooperative Prediction Protocol
For production deployment (e.g. FastAPI backend):
1. Extract frame joint coordinate sequences (shape `[N, 60, 99]`) using MediaPipe.
2. If the sequence is classified by `stgcn_sequence_model.pth` as `transition/unknown`, the backend falls back to using either the static single-head `mlp_model.pth` or the multi-output `mlp_3head_model.pth` classifier on individual frames.
3. This cooperative approach minimizes false positives, provides real-time latency optimization, and ensures smooth transition tracking while practicing.

---
**RCC Institute of Information Technology, Kolkata**  
*Department of Computer Science & Engineering*  
*Final Year Project 2026*
