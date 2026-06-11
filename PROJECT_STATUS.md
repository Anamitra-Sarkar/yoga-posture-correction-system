# Smart Yoga Posture Correction System - Handover & Status Report
**Project ID**: P05 (RCC IIT Kolkata - Final Year Project)
**Current Date**: June 9, 2026

This document serves as a complete status and context guide for resuming work on this project, whether using this agent, another CLI, or a different AI assistant.

---

## 1. Executive Summary & Current Status
We have successfully resolved the transition frame labeling issue and completed the full 28-class classification across all 12 videos (654,488 frames):
* **Transition Reduction Success**: The baseline `transition/unknown` frames have been reduced from **93.08%** down to **34.21%** (well under the 50% target threshold).
* **Extended Posture Vocabulary**: Defined and classified 16 common postures using vector-based joint angle rules, including Mountain, Upward Salute, Standing Fold, Halfway Lift, Table Top, Chair, Child's, Seated Staff, Seated Easy/Meditation, Tree Pose, Warrior I, Warrior II, and general Lunges.
* **Master CSV Compiled**: [master_mlp_dataset_fully_classified.csv](file:///home/anamitra/yoga_raw_dataset/master_mlp_dataset_fully_classified.csv) contains the final pose labels.
* **Sequence Features Compiled**: [stgcn_master_feats.npy](file:///home/anamitra/yoga_raw_dataset/stgcn_master_feats.npy) (Shape: `[18165, 60, 99]`) and [stgcn_master_labels.npy](file:///home/anamitra/yoga_raw_dataset/stgcn_master_labels.npy) contain recompiled sequences ready for ST-GCN training.
* **Fast Validation Completed**: Sklearn classifiers trained on a 50,000 stratified sample validated that these 28 classes are highly distinct and learnable (**96% Random Forest accuracy** and **87% MLP accuracy**).

---

## 2. Preprocessing Files Produced
All datasets are located in:
📁 `/home/anamitra/yoga_raw_dataset/`

| File | Size | Description |
| :--- | :--- | :--- |
| `master_mlp_dataset_fully_classified.csv` | **138 MB** | Frame-level angles and 28-class labels for all 12 videos (654,488 rows). |
| `stgcn_master_feats.npy` | **412 MB** | 18,165 temporal sequences of skeleton coordinates (shape: `[18165, 60, 99]`). |
| `stgcn_master_labels.npy` | **1.3 MB** | Sequence labels for ST-GCN sequence training (only 34.96% transitions). |

---

## 3. Key Scripts & Codebases
All scripts are located in:
📁 `/home/anamitra/Projects_and_Code/Scripts_and_Source/`

### Core Source Folder:
* **`coordinator.py`**: Runs the entire compilation and feature generation.
* **`extract_features_safe.py`**: Computes the 15 biomechanical angles from landmark coordinates.
* **`generate_sequence_features.py`**: Groups frame landmarks into 60-frame windows (updated to use the fully classified dataset).
* **`train_mlp_gpu.py`**: Training script for the static pose MLP model on GPU.
* **`train_stgcn_gpu.py`**: Training script for the ST-GCN sequence model on GPU.

### Experiments Folder (`experiments/`):
* **`classify_all_movements.py`**: Applies the 16 biomechanical rules to auto-classify movement transitions.
* **`train_fully_classified_sklearn.py`**: Fast CPU-based scikit-learn MLP and Random Forest validator.
* **`discover_hidden_poses.py`**: Unsupervised K-Means clustering script to explore transition frames.

---

## 4. What to Do Afterwards (Next Session Steps - GPU Training)
Since local PyTorch training is not setup with GPU dependencies, training must be done on Kaggle.

> [!NOTE]
> * **Kaggle Path Compatibility**: Both [train_mlp_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_mlp_gpu.py) and [train_stgcn_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_stgcn_gpu.py) feature a smart `resolve_path` helper. They automatically locate input datasets under `/kaggle/input` or the local folder, and output files to the current working directory, preventing write/permission errors.
> * **Target Alignment**: [train_mlp_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_mlp_gpu.py) is configured to train on the 28-class fully classified dataset (`master_mlp_dataset_fully_classified.csv`) on class label `imperfect_pose_label`.
> * **Automatic Hugging Face Model Uploads**: Both scripts will automatically upload the best model weights (`.pth`) and label encoders (`.npy`) directly to your Hugging Face model repository `Arko007/yoga-posture-models` using your token. 
> * **Early Stopping & Real-time Logs**: Both scripts print metrics on **every epoch** for real-time visibility. Early stopping checks are now configured to monitor **Validation Loss** (patience of **8 epochs**) for MLP, and **Validation Accuracy** (patience of **20 epochs**) for the sequence model.
> * **Smoothed Class Weights**: Adjusted the loss function's class weights using square-root scaling (`1.0 / sqrt(count)`) to prevent minor classes from dominating the gradients, resolving the severe false-positive rate (low precision) issue.
> * **Residual Recurrent-Attention Sequence Architecture**: Upgraded [train_stgcn_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_stgcn_gpu.py) to a **hybrid Conv1D + Stacked Residual GRU + Self-Attention** model. It includes pelvis-centered scale-normalization for coordinates, a **Cosine Annealing** LR scheduler, and saves checkpoints based on best **Validation Accuracy** instead of loss. Uses a **batch size of 64** and **label smoothing of 0.1** to handle majority-voted sequence label noise. Training duration is set to **120 epochs** to allow full convergence.

1. **Upload Dataset and Scripts**:
   * Upload the preprocessed datasets: `master_mlp_dataset_fully_classified.csv`, `stgcn_master_feats.npy`, and `stgcn_master_labels.npy` to a Kaggle dataset.
   * Upload [train_mlp_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_mlp_gpu.py) and [train_stgcn_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_stgcn_gpu.py) to Kaggle (as code/notebook cells or script uploads).
2. **Train Models on Kaggle GPU**:
   * Create a Kaggle Notebook, attach the uploaded dataset, and enable GPU (T4 or P100).
   * Run [train_mlp_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_mlp_gpu.py) to train the static pose classifier.
   * Run [train_stgcn_gpu.py](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/train_stgcn_gpu.py) to train the Bi-LSTM sequence flow classifier.
3. **Verify and Deploy**:
   * Verify validation accuracies (expected ~96% for RF / 87%+ for MLP).
   * Download the generated `.pth` weights (`mlp_model.pth` and `stgcn_sequence_model.pth`) and the label encoders (`mlp_label_encoder.npy` and `stgcn_label_encoder.npy`) to integrate with your FastAPI backend.

---

## 5. Model Training Execution, Card Publishing & Metadata Definition (June 9, 2026)
Following dataset precompilation, both classifiers were executed in a Kaggle GPU (Tesla T4) environment. The training histories, metrics, and outcomes were analyzed, recorded, and published:

### Model Outcomes:
*   **YogaMLP (Static Pose Classifier)**:
    *   Trained on 654,488 frames (523,590 train / 130,898 val).
    *   Final Accuracy: **92.84%** (Weighted F1: **0.93**).
    *   Best Validation Loss: **0.1644** at Epoch 39/40.
    *   Model weights and label encoder pushed successfully to Hugging Face repository `Arko007/yoga-posture-models`.
*   **YogaSequenceLSTM (Sequence Flow Classifier)**:
    *   Trained on 18,165 sliding sequences (14,532 train / 3,633 val) with pelvis-centered scale normalization.
    *   Final Accuracy: **75.25%** (Weighted F1: **0.76**).
    *   Best Validation Accuracy: **75.25%** at Epoch 90/120. Early stopping triggered at Epoch 110.
    *   Model weights and label encoder pushed successfully to Hugging Face repository `Arko007/yoga-posture-models`.

### Repository & Documentation Activities:
*   **Hugging Face Model Card**: Generated and uploaded a detailed [README.md](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/README.md) containing the architectures, training logs, and classification reports.
*   **License Update**: Updated the frontmatter license from `mit` to `cc-by-nc-sa-4.0` and re-uploaded.
*   **Dataset Metadata Generation**: Prepared subtitle, description, tags, file listings, column descriptors, citation URLs, and a thumbnail generation prompt for the `yoga-pose-features-dataset`.
*   **Security Compliance**: Wiped sensitive API tokens from all temporary local scripts post-execution.

---

## 6. Remaining Tasks & Next Steps (What to Do Next)
To bring the project to its final deployment phase, the following tasks are scheduled:

1.  **FastAPI Backend Integration**:
    *   Download `mlp_model.pth`, `stgcn_sequence_model.pth`, and their corresponding `.npy` label encoders from `Arko007/yoga-posture-models` to the backend workspace.
    *   Implement coordinate pre-processing layers (pelvis translation and hip-scaling) directly into the API intake controller for live sequence predictions.
2.  **Dual-Model Cooperative Pipeline**:
    *   Configure the inference engine to run coordinate frames through `stgcn_sequence_model.pth`.
    *   If the predicted sequence label is `transition/unknown` or sequence confidence drops below $70\%$, fallback automatically to frame-level static prediction using `mlp_model.pth`.
3.  **Real-Time Posture Correction Logic**:
    *   Define ideal joint angle ranges for all 16 target poses (from project specifications).
    *   Compare the real-time angles computed by `extract_features_safe.py` against templates to generate corrective feedback (e.g., *"Straighten your spine by 15°"* or *"Bend your left knee more"*).
4.  **Frontend Dashboard & Video Pipeline**:
    *   Establish a WebSocket/WebRTC stream in the client UI to feed coordinates to the backend and display real-time tracking skeletons with colored correction overlays.
5.  **Project Documentation Compilation**:
    *   Prepare the final project report for P05 submission, compiling dataset statistics, model parameters, validation metrics, and confusion matrices.

---

## 7. Multi-Head Model Preparation & FastAPI Integration (June 10, 2026)
To satisfy the detailed requirements of the P05 project plan, the workspace was expanded and core pipeline codebases were constructed:

### Workspace Setup:
*   Created a new dedicated workspace folder: [yoga_posture_workspace](file:///home/anamitra/yoga_posture_workspace/).
*   Created an initial roadmap and tracking file [WORKSPACE_PLAN.md](file:///home/anamitra/yoga_posture_workspace/WORKSPACE_PLAN.md).

### Codebase Implementations:
1.  **3-Head Static MLP Training Script (`train_mlp_3head_gpu.py`)**:
    *   Created in [Scripts_and_Source](file:///home/anamitra/Projects_and_Code/Scripts_and_Source/).
    *   Implements a multi-head loss architecture predicting Pose ID (CE loss), Correctness (BCE loss), and joint angle deviations (SmoothL1 regression loss).
    *   Deviations are dynamically normalized to a $[0, 1]$ range.
    *   Includes auto-upload capability to Hugging Face and automated Kaggle environment pathing.
2.  **Production-Grade FastAPI Backend (`app.py` & `Dockerfile`)**:
    *   Created in [yoga_posture_workspace](file:///home/anamitra/yoga_posture_workspace/).
    *   **RAM Optimizations**: Bounded PyTorch CPU execution threads to run reliably on the 4GB RAM local check environment.
    *   **Startup Downloader**: Downloads the latest weights and label encoders from `Arko007/yoga-posture-models` dynamically on startup.
    *   **Dual-Model Pipeline**: `/api/analyse_sequence` runs ST-GCN and triggers fallback to `/api/analyse_frame` (MLP) if sequence confidence drops below $70\%$ or labels as transition.
    *   **Physiotherapy Rules Engine & RAG**: `/api/generate_correction` enforces template alignment matching, multilingual translation (EN, HI, BN), and Groq LLM validation scans.
    *   **Docker Containerization**: Custom Dockerfile configured with CPU-only PyTorch and system libraries for seamless HF Spaces deployment.

### 3-Head MLP Model Training Outcomes:
*   **Model Training Execution**: Successfully completed 40 epochs on Kaggle GPU.
*   **Best Validation Loss**: **0.2263** at Epoch 39.
*   **Pose Accuracy**: **93.38%** validation accuracy across 23 base classes (e.g. `chair_pose`, `chaturanga`, `child_pose`, etc.).
*   **Correctness Accuracy**: **96.81%** validation accuracy for binary classification (correct vs. imperfect/transition).
*   **Upload & Cards**: Model saved to `mlp_3head_model.pth` and uploaded to `Arko007/yoga-posture-models` along with `mlp_3head_pose_encoder.npy`. The Hugging Face repository `README.md` was updated to document the new 3-head architecture, training curves, and base class vocabulary.

### Production-Grade Modular Refactoring:
*   **Modular Backend Architecture**: Restructured the single-file API into a decoupled MVC/service-oriented project in [backend](file:///home/anamitra/yoga_posture_workspace/backend/). Split configurations, geometry utilities, model definitions, services, and routers.
*   **Next.js Frontend Integration**: Created production-level frontend elements in TypeScript inside [frontend](file:///home/anamitra/yoga_posture_workspace/frontend/):
    *   [yoga.ts](file:///home/anamitra/yoga_posture_workspace/frontend/src/types/yoga.ts): Complete strict typing interfaces matching API requests and responses.
    *   [api.ts](file:///home/anamitra/yoga_posture_workspace/frontend/src/utils/api.ts): Standardized client wrapper functions utilizing native fetch.
    *   [useYogaPipeline.ts](file:///home/anamitra/yoga_posture_workspace/frontend/src/hooks/useYogaPipeline.ts): A robust React custom hook orchestrating coordinate buffering, temporal window sliding, fallback triggers, and correction generation.

---

## 8. Frontend Interface Redesign, Bug Fixes, and API Optimization (June 10, 2026)
Today we completed the visual redesign of the dashboard, resolved typography and device-adaptive layout issues, developed a dedicated marketing landing page, and resolved critical backend CORS and frontend API throttling bugs:

### UI Redesign & Typography Optimization:
*   **Speech Toggle State**: Fixed the speech control toggle button in [index.tsx](file:///home/anamitra/yoga_posture_workspace/frontend/src/pages/index.tsx) to provide conditional active feedback using classes and accessibility attributes.
*   **Adaptive Sizing**: Capped the typography size scale in [globals.css](file:///home/anamitra/yoga_posture_workspace/frontend/src/styles/globals.css) on ultra-wide desktop screens (`min-width: 1280px`) and adjusted mobile scale overrides (`max-width: 480px`) to prevent visual layout breaking. Added `scrollbar-gutter: stable;` to stop horizontal page jitter.
*   **Toggleable Sidebar**: Fixed the sidebar to make it toggleable on both mobile and desktop screens, collapsing the main layout grid columns dynamically when closed.

### Marketing Landing Page Route:
*   **Product Landing Page**: Developed a fully responsive marketing landing page [landing.tsx](file:///home/anamitra/yoga_posture_workspace/frontend/src/pages/landing.tsx) mapped to the `/landing` route, featuring a custom SVG geometric warrior II skeleton, feature cards, and interactive hamburger navigation.

### Camera Stream & Layout Stability:
*   **Overlay & Invisible Video**: Restored Next.js `<Script>` loader tags for MediaPipe in `index.tsx` to enable webcam initialization. Absolute positioned the canvas to cover the webcam frame, set the video feed to `opacity: 0 !important; width: 1px !important;` to hide it, and removed HTML dimension properties to prevent a double stacked-frame rendering bug.
*   **Height Jitter Resolution**: Retained constant rendering sizes for the feedback status box and joint angle deviation charts even when no active pose is assumed (`activePose === "transition/unknown"`), completely resolving frame-rate jitter and UI shaking.

### API Throttling & CORS Setup:
*   **Fetch Throttling (2fps)**: Added a time gate checks using `lastApiCallTime` ref and `API_THROTTLE_MS = 500` inside `onPoseResults` to throttle pipeline requests, protecting the Hugging Face free-tier backend from rate-limiting (`429 Too Many Requests`).
*   **In-flight Cancellation**: Integrated an `AbortController` request cancellation interceptor into `window.fetch` inside `index.tsx` to immediately cancel stale in-flight requests when a new frame is evaluated.
*   **CORS Configuration**: Configured `CORSMiddleware` in [main.py](file:///home/anamitra/yoga_posture_workspace/backend/app/main.py) to enable cross-origin browser requests from the Vercel production origin and local development hosts.

### License Completion:
*   **CC License Verification**: Retained and completed the full legal clauses of the CC BY-NC-SA 4.0 license inside the repository's root [LICENSE](file:///home/anamitra/yoga_posture_workspace/LICENSE) file to protect the project's IP.

---

## 9. Modular Integration, Digital Twin Calibration, Custom Dropdowns, and Layout Optimization (June 11, 2026)
Today we completed the modular backend integration, implemented automated MediaPipe-guided digital twin calibration, built interactive camera swap features, customized custom themed language dropdown selectors, applied a global UI smoothness pass, and implemented a fullscreen Focus Mode.

### Modular Backend Refactoring & Import Fixes:
* **Groq Imports**: Resolved `NameError` crash by importing `os` in modular backend [correction.py](file:///home/anamitra/yoga_posture_workspace/backend/app/services/correction.py).
* **Package Shadowing**: Renamed the monolithic root `app.py` to `old_monolithic_app.py` to prevent import collisions in python checks.
* **Test Verification**: Modified test entry points ([test_app.py](file:///home/anamitra/yoga_posture_workspace/test_app.py), [run_full_pipeline_test.py](file:///home/anamitra/yoga_posture_workspace/run_full_pipeline_test.py)) to run against the modular package structure successfully.

### Automated Calibration & Digital Twin:
* **15-Second Guided Sequence**: Integrated a guided 15-second calibration loop on webcam start, instructing the user through voice prompts to stand comfortably and breathe deeply.
* **Twin Generation**: MediaPipe reads and processes coordinates over 15 seconds to define a customized joint comfort profile (padding boundaries by $\pm 15^\circ$), displaying the active profile in a scrollable **Digital Twin Profile** sidebar. Caches the profile for the active session and destroys it on camera exit.

### Camera Controls & Configuration:
* **Rear & Front Camera Swapping**: Added support for mobile/tablet devices by scanning device inputs via `enumerateDevices` and rendering a **Swap Camera** toggle header option only if multiple camera tracks exist.
* **API Dynamic Config**: Extracted static environment base URLs into an **API Configuration** collapsible settings panel, enabling dynamic host toggles between local development and the Hugging Face Space.
* **Adaptable Pose Charts**: Re-engineered the deviation bars in [index.tsx](file:///home/anamitra/yoga_posture_workspace/frontend/src/pages/index.tsx) to map targets dynamically matching active pose angle rules rather than hardcoding Warrior II.
* **Guidance Debounce**: Extended LLM spoken instructions interval to 10 seconds to allow relaxed posture transition times.

### Custom Dropdown & Fullscreen Focus Mode:
* **Custom Dropdown Selector**: Replaced native square HTML `<select>` dropdowns with a custom React dropdown UI featuring fade-in, slide-down animations, active hover state highlights, and custom select options.
* **Focus Mode (Fullscreen)**: Developed a fullscreen mode button that expands the canvas frame to `100vw` × `100dvh`, hides all outer app frames, displays a floating exit button, floating correctness score indicator (green/orange states), and a bottom-centered floating text caption overlay showing LLM guidance alerts. Esc-key triggers exit automatically.
* **UI Smoothness Overhaul**: Configured global 180ms CSS transitions across all cards, buttons, KPI metrics, and deviation bars. Replaced hard sidebar collapse toggles with CSS max-height transitions. Added a pulse-spin loading keyframe on sparkles and camera buttons.
* **Desktop Polish**: Adjusted camera video aspect ratio to `16/9`, polished desktop columns, and verified code typechecks without errors.
* **Speech & API Throttling**: Configured a strict 30-second (`30000`ms) debounce throttle on Groq LLM API requests to prevent server resource exhaustion.
* **Dual-Model Coordination Fix**: Fixed a critical bug in the sequence classifier path that was bypassing frame-deviation logic. Now, when the sequence model is confident, it maps pose IDs and correctness scores while still triggering a lightweight static check to fetch active joint deviations for the correction generator.
* **Real-Time Video Captions**: Refactored the caption overlay to render unconditionally on the active video stream (in both normal and fullscreen modes), acting like live YouTube captions to support deaf practitioners.
* **Multilingual TTS Coaching**: Configured real-time local success notifications (in English, Hindi, and Bengali) when joint alignments are correct. Throttled Speech Synthesis playback to prevent overlapping speech while reacting immediately to key state changes.
