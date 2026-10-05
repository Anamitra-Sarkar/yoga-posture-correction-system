# modal/ — conference data and training pipeline (2026-10)

Scripts that produced the cue-verified dataset and the models in `docs/BENCHMARKS.md`. They are a record of how each artifact was made and are re-runnable
(Modal credits are exhausted; the same code ran on Kaggle through `planning/kaggle_transfer/`, see `runner_*.py` and `kernel_*`). Nothing in the running app imports them.

| Script | What it does |
|---|---|
| `asanaai_build.py` | builds the two training CSVs: the old 654k master + the 12 new videos (labels from the ORIGINAL 3-stage chain, RAW-z) |
| `asanaai_extract.py` | MediaPipe landmark + angle extraction for the 12 new videos with the original scripts |
| `asanaai_whisper.py` | transcribes the 24 videos (Whisper) so poses can be labelled by what the instructor says |
| `asanaai_cuelabel.py` | cue-verified labels: instructor named the pose AND the rule engine agrees AND the body is held steady |
| `asanaai_holds.py` | non-vinyasa hold-pose data pipeline (why: the original 12 videos are flows with almost no held poses) |
| `asanaai_photos.py` | real-photo corpus from two public HF datasets (PoseLandmarker HEAVY, visibility filter) |
| `asanaai_harvest.py`, `asanaai_harvest_filter.py` | Commons/Openverse photo harvest and its cleaning (dedupe vs the benchmark, label must agree with the rules) |
| `asanaai_assemble.py` | assembles the training sets (cue labels + transition negatives + photos) |
| `asanaai_train.py`, `asanaai_train2.py` | run the ORIGINAL MLP / ST-GCN trainers unmodified on the extended data; `train2` adds the honest held-out evaluation |
| `run_final_stgcn.sh`, `score_target_mlp.sh` | the command sequences for the final target-pose ST-GCN and for scoring the MLP |
| `asanaai_diag.py`, `asanaai_diag2.py` | diagnostics that found the raw-z vs zero-z train/serve gap (median 14 deg) |
| `asanaai_ytdlp_probe2.py`, `probe_clients.py`, `probe_titles.py`, `fetch_new_videos.sh` | YouTube reachability probes (Modal's datacenter IPs are blocked) and the home-IP fetch that pushed the 12 new videos into the volume |
| `landmark_map.json`, `local_lm_sha.json` | provenance: YouTube id <-> landmark file mapping and checksums (inputs of `asanaai_diag2.py`) |

Run logs of these jobs are in `backup/run_logs_2026-10/` (untracked). Lessons from this pipeline: `docs/TRAINING_LESSONS.md`.
