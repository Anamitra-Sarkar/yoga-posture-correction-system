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

# AsanaAI — Smart Yoga Posture Correction System

A yoga coach that watches your form through the camera, recognises the pose you are in, scores your alignment, and tells you — in English, हिन्दी or বাংলা, on screen and aloud — what to adjust. Final-year project (P05), RCC Institute of Information Technology, Kolkata, Department of CSE-AIML.

* **App:** https://yoga-posture-correction-system.vercel.app  (website: `/landing`)
* **API:** https://arko007-yoga-pose.hf.space  (`/docs` for the interactive reference)
* Your video never leaves your device: the browser extracts body landmarks locally and sends only body-position numbers (landmark coordinates and joint angles) to the API, never images.

## How it works

```
camera ─▶ MediaPipe Pose (in the browser) ─▶ 33 landmarks ─▶ 15 joint angles (+ body orientation, used only by the rule engine in basic mode / the non-cascade path)
                                                  │
        ┌─────────────────────────────────────────┴──────────────────────────────┐
        ▼ every ~1.5 s                                                              ▼ every ~10 s
  per-frame coach                                                          sequence model (ST-GCN)
  server: 3-head ResMLP names the pose, a second MLP "gate"                60-frame window at 25 fps,
  screens out poses we do not know and supplies the form score,            shown as a second opinion
  angle bands give per-joint deviations                                    ("Flow check"), never overrides
  on device (offline / server asleep): the same rule engine,
  ported line for line, answers instantly ("basic mode")
        │
        ▼
  visibility-aware scoring: joints the camera cannot see are neither scored nor coached
        │
        ▼
  coaching: reviewed templates (+ optional LLM paraphrase behind safety filters) ─▶ screen + speech
```

* **3-head ResMLP** (pose · correctness · 15 joint deviations) over 15 biomechanical angles, trained on cue-verified video frames and real photos.
* **ST-GCN** (graph convolution over the 33-joint skeleton + temporal convolution) on 60-frame windows.
* **Pose cascade** (`docs/CASCADE.md`) and **sequence model** (`docs/SEQUENCE_MODEL.md`): rollout flags, rollback and design notes.
* **On-device coach** (`frontend/src/utils/offlineCoach.ts`): generated from the backend rule engine and verified identical to it on 12,500 generated cases (`python3 backend/tools/offline_parity.py`).
* **Pose engine** (all on the device, chosen automatically): MediaPipe on the GPU; phones whose GPU driver breaks it (measured: PowerVR BXM-8-256, e.g. MediaTek Dimensity 7020/7025/930) transparently use MediaPipe Tasks on the CPU in a worker, and TensorFlow.js WASM as a last resort. Engine files are cached on the first visit so the app also starts offline. Measurements: `docs/BENCHMARKS.md` section 11.
* **Not deployed:** CLIFF-based two-stream occlusion fusion exists in `backend/app/services/occlusion.py` only as an optional hook (nothing sends it CLIFF data and no CLIFF model is deployed). The live system is visibility-aware instead: it declines to score a joint it cannot see. An offline experiment comparing CLIFF with the old mirror fallback is written up in `docs/BENCHMARKS.md` section 10 (CLIFF needs the camera image, which the app never uploads).

## Results (held-out only)

Training/validation accuracies printed by the trainers come from a random frame split and are **not** reported as results. Honest numbers (held-out videos, held-out public photos) are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Headlines:

| | Before | Now |
|---|---|---|
| Held-out public photos: pose accuracy | 36.9% | 78.9% (pose cascade) |
| Photos of poses we do NOT cover wrongly named as ours (lower is better) | 78.3% | 20.8% |
| Poses with recall and precision >= 0.70 on held-out photos | 0 | 7 |
| ST-GCN on held-out videos | live model: macro recall 0.19 | new model: macro recall 0.82, 3 poses pass the bar (4 with a cross-fitted threshold) |
| Live correctness check (one joint broken by 45 degrees) | - | form score 0.94 -> 0.25 for a leg joint; **arm joints are barely detected** |

Standard methods (k-NN, SVM, random forest, plain MLP) trained on the same data and scored on the same photos are in section 13 of `docs/BENCHMARKS.md`: a random forest beats the cascade on photos from the training sources (85.3% vs 78.9%) but is the weakest method on a wild photo set from another source (35.0% vs 55.3%), so no single method wins everywhere.

Known limits are listed in section 7 of `docs/BENCHMARKS.md` and in `docs/REPO_AUDIT_2026-10-07.md` (Tree and Lunge have no angle bands, so their per-joint colours come from the learned deviation head, which is near chance).

## Repository layout

| Path | What |
|---|---|
| `backend/` | FastAPI service (also the Hugging Face Space): models, rule engine, cascade, coaching, tests (`backend/tests`), tools (`backend/tools`) |
| `frontend/` | Next.js PWA (app at `/`, website at `/landing`) and the Capacitor Android project (`frontend/android`) |
| `mobile/` | Separate Expo (React Native) client |
| `modal/` | Data and training pipeline scripts (cue-verified relabelling, windows, training); index in `modal/README.md` |
| `planning/` | Experiments, Kaggle harnesses (`planning/kaggle_transfer`), rescued checkpoints (`planning/modal_rescue`), archive of superseded files; index in `planning/README.md` |
| `docs/` | Benchmarks, cascade and sequence-model notes, training lessons, live-test clips, repository audit |
| `paper/` | The research paper for the conference (LaTeX + PDF, `paper/asanaai_research_paper.pdf`), figures, verified bibliography |
| `report/` | The project report (LaTeX + PDF, `report/asanaai_project_report.pdf`): design, deployment, verification, audit, operations |
| `backup/` | Resume/handoff notes (start with `backup/RESUME_HERE_2026-10-05.md`) and dated copies of key docs |

## Run it locally

```bash
# backend (needs the model files from the Hugging Face repo; set HF_TOKEN if the repo is private)
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --port 7860          # API docs at /docs
cd backend && python3 -m pytest tests -q                 # 72 tests

# frontend
cd frontend && npm install && npm run dev                # http://localhost:3000  (set NEXT_PUBLIC_YOGA_API_URL)

# live smoke test of the DEPLOYED backend (assertions; see backup/verify_live.sh)
bash backup/verify_live.sh

# on-device coach parity check (Python original vs TypeScript port)
python3 backend/tools/offline_parity.py
```

## Deploy

* **Backend:** pushing to `main` with changes under `backend/` syncs the Space (`.github/workflows/hf_sync.yml`). Defaults equal the benchmarked configuration (pose cascade on, `stgcn_target_v1`); roll back with the Space variables `ENABLE_POSE_CASCADE=0` and `STGCN_MODEL_FILE` / `STGCN_ENCODER_FILE` set to the `stgcn_transitions_v1*` files. `GROQ_API_KEY` (secret) enables the optional LLM paraphrase.
* **Web:** Vercel builds `main` (production) and every other branch (preview URL).
* **Android:** `.github/workflows/android_build.yml`.

## Safety

AsanaAI gives general guidance and is not medical advice. Cues are built from reviewed templates and never ask you to push further. If the optional LLM paraphrase is enabled, a post-generation screen rejects English wording such as "push", "force", "pain" or "further"; that screen is a list of English words, so Hindi and Bengali paraphrases are not covered by it (known gap, see `docs/REPO_AUDIT_2026-10-07.md`).

## Licence

See `LICENSE`. Pose reference photographs in `frontend/public/pose-images` are CC BY / CC BY-SA from Wikimedia Commons; credits are shown beside each photo in the app.
