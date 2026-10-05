# backup/ — durable notes index

The scratchpad gets wiped; anything worth keeping lives here. Newest first.

## Read in this order
1. `RESUME_HERE_2026-10-05.md` — **START HERE.** The "CURRENT STATE" block at the top is authoritative; older blocks below it are kept for history and are tagged where they are superseded.
2. `SESSION_HANDOFF_2026-10-04_CONFERENCE.md` — depth on the conference pivot (cue-verified relabelling, retraining the original MLP/ST-GCN on the extended data, Modal scripts in `../modal/`).
3. `../docs/BENCHMARKS.md` — every number for the report (held-out only). Sections 9-10: visibility-aware scoring, on-device coach, CLIFF experiment.
4. `../docs/TRAINING_LESSONS.md` — 18 lessons, each measured here. Read before writing another training script.

## Older (superseded, kept for the record)
* `SESSION_HANDOFF_2026-09-21.md`, `SESSION_HANDOFF_2026-09-19.md` — the photo-domain (MLP v4) route and the earlier ST-GCN attempts. The live models have since changed (see RESUME_HERE).
* `../planning/CHECKPOINT_2026-09-03_FINAL.md`, `../planning/SESSION_HANDOFF_2026-09-03.md`, `../planning/SESSION_HANDOFF_2026-08-27.md` — earlier checkpoints.

## Files
* `verify_live.sh` — smoke-tests the DEPLOYED backend (all 6 endpoints, no local build, no GPU). Run after any backend deploy: `bash backup/verify_live.sh`.
* `phone_debug/` — test the web app on a USB-connected Android phone (adb + DevTools protocol); see its README.
* `test_sequence_buffer.js` — unit test of `frontend/src/utils/sequenceBuffer.ts` (the 25 fps resampler).
* `BENCHMARKS_2026-10-05.md`, `LIVE_TEST_CLIPS_2026-10-05.md`, `SEQUENCE_MODEL_2026-10-05.md` — verbatim copies of `docs/BENCHMARKS.md`, `docs/LIVE_TEST_CLIPS.md`, `docs/SEQUENCE_MODEL.md` (keep identical: `cmp` them after editing the originals).
* `CLEANUP_2026-10-05.md` — what the repo cleanup deleted / moved / deliberately kept, and why; plus the file-by-file freshness audit.
* `run_logs_2026-10/` — Modal run logs moved out of `modal/` (untracked, git-ignored; no secrets).
* `../planning/kernel_metadata/` — exactly what each Kaggle kernel was pushed with, so a run can be reproduced without reconstructing the dataset wiring.
