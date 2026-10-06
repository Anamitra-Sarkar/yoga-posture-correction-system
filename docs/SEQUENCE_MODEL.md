# Sequence model (ST-GCN): role, rollout, rollback

**Live since 2026-10-05:** `stgcn_target_v1` (9 classes: child pose, corpse, downward dog, plank, seated easy, tree, triangle, warrior 2, transition/unknown) in `Arko007/yoga-posture-models`. Real ST-GCN: graph convolution over the 33-joint MediaPipe skeleton + 9-tap temporal convolution, 3 blocks, input = 60 frames x 33 joints x (x, y, z), pelvis-centred and hip-width-scaled.

## Role in the app
* The per-frame **MLP cascade** (`docs/CASCADE.md`) names the pose and scores the form (correctness head). It always runs and is never overridden.
* The **ST-GCN** looks at the last 2.4 s and acts as a second opinion in the "Sequence Flow" row. It is shown only when confident (>= 0.70, child pose 0.55); otherwise the row reads "Static Check". Model confidence is never used as a form score.

## Frame-rate contract (the thing that broke it before)
The model is trained on 60 consecutive native-rate frames (~2.4 s). The web app therefore buffers EVERY camera result with a timestamp (`frontend/src/utils/sequenceBuffer.ts`) and sends a window resampled to 25 fps. If the history is too short or the camera stalled (> 250 ms gap) no window is sent and the row shows "Static Check". A window is sent at most once per ~10 s cycle, and only while the server is answering; in on-device "basic mode" there is no sequence model and the row stays "Static Check". Feeding it 0.5-2 fps (the old behaviour) drops macro recall from 0.82 to ~0.4 (docs/BENCHMARKS.md section 5).

## Switching / rollback (no redeploy)
The backend reads `STGCN_MODEL_FILE` and `STGCN_ENCODER_FILE`. **Since 2026-10-07 the defaults are `stgcn_target_v1.pth` / `stgcn_target_v1_encoder.npy`** (the benchmarked model), so a fresh deploy equals production and no Space variable is needed. Roll back to the previous model by setting both Space variables to `stgcn_transitions_v1.pth` / `stgcn_transitions_v1_encoder.npy` (`HfApi().add_space_variable("Arko007/yoga_pose", "STGCN_MODEL_FILE", "stgcn_transitions_v1.pth")`, same for the encoder; the Space restarts); its vocabulary is still parsed (`parse_sequence_label`).

## Known limits (state them in the report)
* Knows 8 poses + "transition/unknown"; no named transitions (the older model had them but could not recognise held poses on unseen videos).
* Held-out-video results: 3 poses pass the 70/70 bar (child, corpse, downward dog), 4 with a cross-fitted less-conservative threshold; tree, warrior 2 and plank have 30-90 held-out windows and low precision.
* Evidence and tables: `docs/BENCHMARKS.md` section 5 and 8.
