# Session handoff: mentor pack, Hugging Face cards, architecture PDF (2026-11-04)

Read this if you resume anything about what was given to the mentor, the Yoga-1M dataset card, the model card, or the architecture document. The paper/report work is in `SESSION_HANDOFF_2026-10-07_PAPER_REPORT.md`.

## 1. What the mentor received (already sent by the user)
All in `/home/anamitra/yoga-final-year-code/` (outside the repo):
| Item | Size | Content |
|---|---|---|
| `yoga-posture-correction-system-main.zip` | 26 MB | Code. Verified identical to `main` at 4949cf2 (559 files, no diff) |
| `yoga-1m-training-data.zip` (+ unzipped folder) | 21.7 MB | `vol/csv/cue_verified_poses_80k.csv` (80,136 frames), `vol/csv/cue2/vH/mlp_all.csv` + `assemble_report.json` (gate MLP training table, 136,506 rows), `vol/cue/*.npz` (24), `vol/photos/public_corpus.npz` (+report), `vol/eval/photo_corpus.npz`, `vol/folds/photo_folds.json`, dataset card as `README.md`, `MANIFEST.md` |
| `yoga-models.zip` (+ folder) | 6 MB | The 3 production checkpoints + encoders + model card; SHA-256 verified against HF; strict-loaded with the repo classes |
| `AsanaAI_Architecture.pdf` | 0.6 MB, 13 pp | Architecture-only design document (no accuracy results) |
Gmail limit is 25 MB: the code zip must go by Drive link or a separate email.

## 2. Verified facts (all from code, scripts or HF, not from memory)
* **Frame counts.** 1,206,391 frames = **80,136** cue-verified pose frames (15 poses, 24 videos) + **420,722** transition/unknown + **705,533** `__ignore__` (counted by streaming `cue_full.csv`; nothing stored). The report/paper/card say "~80k / ~421k / ~705k"; exact numbers are above.
* **Gate MLP** (`mlp_3head_gate_v1` = `cueH_mlp_all`): trained on `vol/csv/cue2/vH/mlp_all.csv` = 136,506 rows: 39,962 video positive rows (every 2nd frame, `VIDEO_STRIDE = 2`, 14 poses with >= 300 frames) + 11,988 random transition rows (0.3 x positives) + ~84.6k photo rows (public train photos x2, 2,945 out-of-vocabulary photos, Commons photos repeated). `__ignore__` is never used.
* **ST-GCN** (`stgcn_target_v1`): trained on `vol/stgcn/cueT2_all` = 39,416 windows = 8,002 real windows (60 frames, stride 12; windows with > 15% ignored frames dropped; transitions subsampled to 1.5 x pose windows) + their mirrors + synthetic "photo-hold" clips (photo landmarks held 60 frames with sway from real holds; also mirrored). Test windows of the fold folders are real, never augmented. 9 classes, 893,897 parameters.
* **Namer** (`mlp_3head_v4_photos_x2000`): trained on the ORIGINAL 12-video rule-labelled table `master_mlp_dataset_fully_classified.csv` + the photo sets of that time (photos x2000). That table is NOT in Yoga-1M (only its builder `code/originals/compile_master_dataset.py`); local copy `~/yoga_raw_dataset/master_mlp_dataset_fully_classified.csv` (221 MB). Gate and namer: 23 classes, 357,671 parameters each.
* **Deployed models** (checked: Space variables, backend defaults, live `/api/analyse_frame` returned `cascade.active=true`): namer v4 + gate v1 + `stgcn_target_v1`; Space variables `ENABLE_POSE_CASCADE=1`, `STGCN_MODEL_FILE=stgcn_target_v1.pth`.
* The Space is `sdk: gradio` (`backend/app.py` mounts the FastAPI app); the Dockerfile is not the deployment path.

## 3. What was changed on Hugging Face (account Arko007 only)
* **Dataset card** (`Arko007/Yoga-1M`, source `planning/kaggle_transfer/README_dataset.md`): rewritten: quick start, file guide, schemas, labelling tables, "which data trained which shipped model", "what happened to the 1.2M frames", directory map, "do not use / caution" table, licence, citation. YAML `configs:` = `frames` (`vol/csv/cue/cue_full.csv`, default) and `mlp_vH` (`vol/csv/cue2/vH/mlp_all.csv`); the viewer was "no supported data files" before and **now works** (verified 2026-11-04: preview/viewer/search/filter/statistics true).
* **Model card** (`Arko007/yoga-posture-models`, source `planning/kaggle_transfer/README_models.md`): rewritten around the three production models with held-out benchmarks, comparison with standard methods (kept honest: random forest wins on same-source photos, window-statistics MLP matches the ST-GCN), limitations, superseded versions. The OLD card (kept as `backup/MODEL_CARD_old_2026-11-04.md`) quoted leaky validation accuracies (92.84 / 93.38 / 96.81 / 75.25%) and described the sequence model as Conv + GRU + attention. No weight files were touched (23 files before and after).
* Uploads used `HfApi.upload_file` with the token at `/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token` (whoami = Arko007). The cached `~/.cache/huggingface/token` login is the **bhumika** account: never use it for this project.

## 4. Architecture document
`docs/architecture_doc/`: `asanaai_architecture.tex`, `make_figs.py` (skeleton graph and the real normalised adjacency matrix, drawn from `backend/app/models/sequence.py`), `fig_*.pdf`, `asanaai_architecture.pdf`. Reuses `report/fig_arch.tex`, `fig_ladder.tex` and the screenshots in `report/figures/`. Build: `cd docs/architecture_doc && pdflatex asanaai_architecture.tex` (x2). Every page was rendered and inspected; facts re-checked against code (angle/landmark map, visibility thresholds 0.4/0.6, bands, formula 1 - mean deviation/45, thresholds 0.70/0.55, API fields, calibration constants +-15 deg / 20 frames, parameter counts).

## 5. User preferences learned this session
* **No emojis in any document or card** ("very unprofessional"); professional tone.
* Downloads: allowed ONLY for what was asked and only small files (the user lifted "never download" for the training-data and model packs, nothing else). Stream-and-filter instead of storing large files.
* "Make no mistakes": verify against code/HF before stating; render PDFs and look at them.
* Show models as the current production versions with honest benchmarks, never claiming they beat every method.

## 6. Open items (none urgent)
* **Report gap:** the report states the 80k/421k/705k counts but not how each model used them (stride 2, 0.3 negatives, 15% window rule, 1.5 x transitions, the 39,416 augmented windows). Its "8,002 windows" is the real held-out set. Offered: a short subsection (text and table are in section 2 above and in the dataset card). Not done.
* **Exact counts** (80,136 / 420,722 / 705,533) could replace "~80k" etc. in report, paper and card. Not done.
* **Stale comments in `backend/`** (editing redeploys the Space): `app/services/correction.py` says the Hindi/Bengali lists are "NOT YET REVIEWED" (the user reviewed them); `hf_loader.py:67` and `tests/test_sequence_labels.py:4` say "24 classes"; `docs/TRAINING_LESSONS.md` has stale "24/25 classes" and "63.0% macro".
* **`ieee_research_paper.pdf`** in the HF model repo is the outdated first draft (the card says so); never delete artifacts; could upload the current paper under a new name if the user wants.
* The user will review the paper and send changes (see the 2026-10-07 handoff).
* Optional experiments listed in the 2026-10-07 handoff are still unrun.

## 7. Tool gotchas hit this session
`sleep N` followed by another command is blocked (use a background until-loop); the command classifier occasionally returned "no verdict" (retry once); multi-line raw strings in a `python3 - <<EOF` heredoc broke (write the edit script to a file); `pdftoppm -r 62` is too low to judge text overlap (use 75+ and crop); pdftotext shows sloped TikZ labels split letter by letter (an artefact, not a bug).
