# backup/ — durable notes index

The scratchpad gets wiped; anything worth keeping lives here.

* `SESSION_HANDOFF_2026-09-21.md` — **START HERE.** Current state: MLP v4 live
  at 47.8%, ST-GCN v6 and MLP v4-rerun training, the untrained-heads
  regression and its fix, and the one action that needs you (the Space's Groq
  key is invalid).
* `../docs/TRAINING_LESSONS.md` — 13 lessons, each measured here. Read before
  writing another training script.
* `SESSION_HANDOFF_2026-09-19.md` — previous. Two Kaggle kernels left
  running (ST-GCN attempt 4 vs the 63.0% bar; photo corpus v2), the one-line
  loader change awaiting sign-off, live-verification results, four real bugs
  fixed, and what was deliberately left undone.

* `verify_live.sh` — smoke-tests the DEPLOYED backend (all 6 endpoints, no
  local build, no GPU). Run after any deploy: `bash backup/verify_live.sh`.
* `../planning/kernel_metadata/` — exactly what each Kaggle kernel was pushed
  with, so a run can be reproduced without reconstructing the dataset wiring.

**One action is waiting on you:** the Space's `GROQ_API_KEY` secret is invalid
(401 on every call). See section 4 of the handoff for the one command.

Older checkpoints live in `planning/` (`CHECKPOINT_2026-09-03_FINAL.md`,
`SESSION_HANDOFF_2026-09-03.md`, `SESSION_HANDOFF_2026-08-27.md`).
