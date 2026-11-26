---
title: Yoga Posture Correction System
emoji: 🧘
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.20.0"
app_file: app.py
python_version: "3.10"
pinned: false
---

# 🧘 AsanaAI: Smart Yoga Posture Correction

A yoga coach that watches your form through the camera, recognises the pose you are in, scores your alignment, and tells you what to adjust, in **English, हिन्दी or বাংলা**, on screen and aloud. Final-year project P05, Department of CSE-AIML, RCC Institute of Information Technology, Kolkata.

| | |
|---|---|
| 🌐 **App** | https://yoga-posture-correction-system.vercel.app (website: `/landing`) |
| ⚙️ **API** | https://arko007-yoga-pose.hf.space (`/docs` for the interactive reference) |
| 🤗 **Models** | [`Arko007/yoga-posture-models`](https://huggingface.co/Arko007/yoga-posture-models) |
| 📚 **Training data** | [`Arko007/Yoga-1M`](https://huggingface.co/datasets/Arko007/Yoga-1M) (cue-verified), [`Yoga-650k`](https://huggingface.co/datasets/Arko007/Yoga-650k) (historical), [`yoga-skeleton-rebuilt`](https://huggingface.co/datasets/Arko007/yoga-skeleton-rebuilt) (forensic rebuild), [`yoga-dataset-raw`](https://huggingface.co/datasets/Arko007/yoga-dataset-raw) (source videos) |
| 📄 **Documents** | Paper `paper/asanaai_research_paper.pdf` · Project report `report/asanaai_project_report.pdf` · Architecture `docs/architecture_doc/asanaai_architecture.pdf` |

🔒 **Privacy by design.** Your video never leaves your device. The browser extracts 33 body landmarks locally and sends only body-position numbers (landmark coordinates and joint angles) to the API, never images.

---

## ✨ Features

* **Pose recognition and form score** from a live camera, with per-joint cues ("straighten your left knee") coloured on a skeleton overlay.
* **Honest about what it cannot see.** Joints the camera cannot see are neither scored nor coached.
* **Not-one-of-my-poses veto.** A second model refuses to name poses outside the app's vocabulary instead of guessing.
* **Flow check.** A skeleton-graph network (ST-GCN) gives a second opinion on held poses over the last 2.4 seconds.
* **Personal range of motion.** A 15-second calibration widens the "correct" band to your own body (stored only in your browser).
* **Works on budget phones.** A pose-engine fallback ladder survives GPU drivers that crash the standard engine, and an on-device coach keeps working offline.
* **Three languages, screened wording.** Reviewed templates, optional LLM paraphrase behind a safety screen, never "push through pain".
* **Installable.** PWA, Capacitor Android wrapper and an Expo companion app.

## 🏗️ How it works

```
camera ─▶ MediaPipe Pose (in the browser) ─▶ 33 landmarks ─▶ 15 joint angles
                                                  │
        ┌─────────────────────────────────────────┴──────────────────────────────┐
        ▼ every ~1.5 s                                                              ▼ every ~10 s
  per-frame coach                                                          sequence model (ST-GCN)
  server: namer MLP names the pose, gate MLP vetoes "not one of           60-frame window at 25 fps,
  mine" and supplies the form score, angle bands give per-joint           shown as a second opinion
  deviations                                                              ("Flow check"), never overrides
  on device (offline / server asleep): the same rule engine,
  ported line for line, answers instantly ("basic mode")
        │
        ▼
  visibility-aware scoring: joints the camera cannot see are neither scored nor coached
        │
        ▼
  coaching: reviewed templates (+ optional LLM paraphrase behind safety filters) ─▶ screen + speech
```

### 🤖 Models in production

| Role | Weights (in [`yoga-posture-models`](https://huggingface.co/Arko007/yoga-posture-models)) | Trained on | Parameters |
|---|---|---|---:|
| **Namer** names the pose | `mlp_3head_v4_photos_x2000.pth` | historical 12-video table ([Yoga-650k](https://huggingface.co/datasets/Arko007/Yoga-650k)) plus photo sets | 357,671 |
| **Gate** vetoes and scores form | `mlp_3head_gate_v1.pth` | cue-verified video frames plus photos ([Yoga-1M](https://huggingface.co/datasets/Arko007/Yoga-1M), `vol/csv/cue2/vH/mlp_all.csv`) | 357,671 |
| **ST-GCN** second opinion on held poses | `stgcn_target_v1.pth` | 39,416 windows from the 24 videos, with mirrors and synthetic photo-hold clips | 893,897 |

* **3-head ResMLP** (pose, correctness, 15 joint deviations) over 15 biomechanical angles.
* **ST-GCN**: graph convolution over the 33-joint skeleton plus a 9-tap temporal convolution, on 60-frame windows. (The class is called `YogaSequenceLSTM` only for checkpoint compatibility; it is not recurrent.)
* **Pose cascade** ([`docs/CASCADE.md`](docs/CASCADE.md)) and **sequence model** ([`docs/SEQUENCE_MODEL.md`](docs/SEQUENCE_MODEL.md)): rollout flags, rollback and design notes.
* **On-device coach** (`frontend/src/utils/offlineCoach.ts`): generated from the backend rule engine and verified identical to it on 12,500 generated cases (`python3 backend/tools/offline_parity.py`).
* **Pose engine** (all on the device, chosen automatically): MediaPipe on the GPU; phones whose GPU driver breaks it (measured: PowerVR BXM-8-256, for example MediaTek Dimensity 7020/7025/930) transparently use MediaPipe Tasks on the CPU in a worker, and TensorFlow.js WASM as a last resort. Engine files are cached on the first visit so the app also starts offline. Measurements: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) section 11.
* **Not deployed:** CLIFF-based two-stream occlusion fusion exists in `backend/app/services/occlusion.py` only as an optional hook; no CLIFF model is deployed. The live system is visibility-aware instead. The offline CLIFF-versus-mirror experiment is in `docs/BENCHMARKS.md` section 10.

## 📊 Results (held-out only)

Training and validation accuracies printed by the trainers come from a random frame split and are **not** reported as results. Honest numbers (held-out videos, held-out public photos) are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Headlines:

| | Before | Now |
|---|---|---|
| Held-out public photos: pose accuracy | 36.9% | **78.9%** (pose cascade) |
| Photos of poses we do NOT cover wrongly named as ours (lower is better) | 78.3% | **20.8%** |
| Poses with recall and precision >= 0.70 on held-out photos | 0 | **7** |
| ST-GCN on held-out videos | live model: macro recall 0.19 | new model: macro recall 0.82, 3 poses pass the bar (4 with a cross-fitted threshold) |
| Live correctness check (one joint broken by 45 degrees) | - | form score 0.94 -> 0.25 for a leg joint; **arm joints are barely detected** |

Standard methods (k-NN, SVM, random forest, plain MLP) trained on the same data and scored on the same photos are in section 13 of `docs/BENCHMARKS.md`: a random forest beats the cascade on photos from the training sources (85.3% vs 78.9%) but is the weakest method on a wild photo set from another source (35.0% vs 55.3%), so no single method wins everywhere.

⚠️ Known limits are listed in section 7 of `docs/BENCHMARKS.md` and in `docs/REPO_AUDIT_2026-10-07.md`. In short: no live-person accuracy study yet; mountain, cobra and child's pose are weak; Tree and Lunge have no angle bands, so their per-joint colours are neutral; the form score is nearly blind to arm errors.

## 🗂️ Repository layout

| Path | What |
|---|---|
| `backend/` | FastAPI service (also the Hugging Face Space): models, rule engine, cascade, coaching, tests (`backend/tests`), tools (`backend/tools`) |
| `frontend/` | Next.js PWA (app at `/`, website at `/landing`) and the Capacitor Android project (`frontend/android`) |
| `mobile/` | Separate Expo (React Native) client |
| `modal/` | Data and training pipeline scripts (cue-verified relabelling, windows, training); index in `modal/README.md` |
| `planning/` | Experiments, Kaggle harnesses (`planning/kaggle_transfer`), rescued checkpoints (`planning/modal_rescue`), Hugging Face card sources (`planning/hf_cards`), archive of superseded files; index in `planning/README.md` |
| `docs/` | Benchmarks, cascade and sequence-model notes, training lessons, live-test clips, repository audit, and the architecture document (`docs/architecture_doc`) |
| `paper/` | The research paper (LaTeX + PDF), figures, verified bibliography |
| `report/` | The project report (LaTeX + PDF): design, deployment, verification, audit, operations |
| `backup/` | Resume and handoff notes (start with `backup/README.md`) and dated copies of key docs |

## 🚀 Run it locally

```bash
# backend (needs the model files from the Hugging Face repo; set HF_TOKEN if the repo is private)
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --port 7860          # API docs at /docs
cd backend && python3 -m pytest tests -q                 # backend test suite

# frontend
cd frontend && npm install && npm run dev                # http://localhost:3000  (set NEXT_PUBLIC_YOGA_API_URL)

# live smoke test of the DEPLOYED backend (assertions; see backup/verify_live.sh)
bash backup/verify_live.sh

# on-device coach parity check (Python original vs TypeScript port)
python3 backend/tools/offline_parity.py
```

## 📦 Deploy

* **Backend:** pushing to `main` with changes under `backend/` or this `README.md` syncs the Space (`.github/workflows/hf_sync.yml`). Defaults equal the benchmarked configuration (pose cascade on, `stgcn_target_v1`); roll back with the Space variables `ENABLE_POSE_CASCADE=0` and `STGCN_MODEL_FILE` / `STGCN_ENCODER_FILE` set to the `stgcn_transitions_v1*` files. `GROQ_API_KEY` (secret) enables the optional LLM paraphrase.
* **Web:** Vercel builds `main` (production) and every other branch (preview URL).
* **Android:** `.github/workflows/android_build.yml`.

## 🛡️ Safety

AsanaAI gives general guidance and is **not medical advice**. Cues are built from reviewed templates and never ask you to push further. If the optional LLM paraphrase is enabled, a post-generation screen rejects English wording such as "push", "force", "pain" or "further"; that screen is a list of English words, so Hindi and Bengali paraphrases are not covered by it (known gap, see `docs/REPO_AUDIT_2026-10-07.md`).

## 👥 Team

Anamitra Sarkar, Debjit Deb Barman, Adrija Ray Mondal and Tanima Samanta, under the supervision of Mr. Sujit Chakraborty, RCC Institute of Information Technology, Kolkata.

## 📜 Licence

See `LICENSE`. Pose reference photographs in `frontend/public/pose-images` are CC BY / CC BY-SA from Wikimedia Commons; credits are shown beside each photo in the app. Model weights and datasets on Hugging Face are for non-commercial research and education (see each card).
