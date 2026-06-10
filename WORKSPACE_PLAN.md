# Workspace Plan: Smart Yoga Posture Correction System (Project P05)
**Date**: June 10, 2026
**Workspace Location**: `/home/anamitra/yoga_posture_workspace/`

This workspace coordinates the final integration of the models, backend APIs, and pipeline modules for the final year project.

---

## 1. Project Component Mapping

| Module | Planned Implementation | Current Status |
| :--- | :--- | :--- |
| **Pose Estimation** | MediaPipe Pose Landmarker (33 landmarks) | Preprocessing scripts implemented. |
| **Occlusion Handling** | SMPL-X / CLIFF (ViT-based) for visibility < 0.5 | Planned (Phase 5). Fallback interpolation implemented. |
| **Static Classifier** | 3-Head MLP (Pose ID, Correctness, Deviation) | Trained single-head classifier (`mlp_model.pth`). |
| **Temporal Analyser** | ST-GCN / Sequence model (Smoothness, Balance) | Trained hybrid GRU-Attention model (`stgcn_sequence_model.pth`). |
| **Safety Pipeline** | Groq API + Symbolic Rules Engine + KG + Validator | Under construction. |
| **Backend API** | FastAPI containerized on Hugging Face Spaces | To be constructed. |
| **Frontend UI** | React Dashboard deployed on Vercel | To be constructed. |

---

## 2. Integration Routes

### Route A: FastAPI Backend Development
*   **Objective**: Build a complete, containerized FastAPI backend (`app.py` and `Dockerfile`) ready for deployment on Hugging Face Spaces.
*   **Tasks**:
    1.  Create `app.py` with endpoints for `/api/analyse_frame` and `/api/analyse_sequence`.
    2.  Write logic to download model weights directly from `Arko007/yoga-posture-models` during startup.
    3.  Implement coordinate pre-processing (pelvis-centering and hip-width scaling) and cooperative fallback flow.

### Route B: Refining the Static MLP to 3-Head Architecture
*   **Objective**: Refine `train_mlp_gpu.py` to implement a multi-head loss architecture, predicting Pose ID (classification), Correctness (binary classification), and Joint-Deviation (regression).
*   **Tasks**:
    1.  Update dataset columns to parse correctness flags and joint angle deviation profiles.
    2.  Rewrite `YogaMLP` to have 3 distinct output layers.
    3.  Compute composite loss: $\mathcal{L}_{total} = \alpha \mathcal{L}_{pose} + \beta \mathcal{L}_{correctness} + \gamma \mathcal{L}_{deviation}$.

### Route C: LLM Safety Pipeline & Rules Engine
*   **Objective**: Create the 3-stage validation pipeline for Groq API natural language feedback.
*   **Tasks**:
    1.  Write the symbolic physiotherapy checking engine.
    2.  Implement a local JSON-based knowledge graph to map deviations to template corrections.
    3.  Write the post-generation text scanner/validator.
