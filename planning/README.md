# planning/ — experiments, evaluation harnesses, rescued artifacts

Everything here is a record of how a result or a checkpoint was produced; none of it is imported by the running app. Current state: `backup/RESUME_HERE_2026-10-05.md`. Report numbers: `docs/BENCHMARKS.md`.

## Dated notes (historical; each carries a SUPERSEDED banner)
`CHECKPOINT_2026-09-03_FINAL.md`, `SESSION_HANDOFF_2026-08-27.md`, `SESSION_HANDOFF_2026-09-03.md`, `NEXT_STEPS_*.md`, `ieee_research_paper_backup_2026-09-03.tex` (a backup of the paper draft written 2026-09-03, before the held-out benchmarks: check any number in it against `docs/BENCHMARKS.md` before reuse).

## Data collection
| Script | Purpose |
|---|---|
| `harvest_photo_corpus.py`, `harvest_photo_corpus_kaggle.py`, `harvest_photo_corpus_v2.py` | Commons/Openverse real-photo harvest (v1 -> v2 is the larger one) |
| `build_realworld_testset.py` | the real-world photo test set (replaces the 26-photo/11-class set) |
| `kaggle_extract_public_corpora.py` | MediaPipe joints from public yoga photo datasets |
| `source_more_training_video.py`, `kaggle_stgcn_new_people.py` | diverse extra video / multi-person sequences for the ST-GCN |
| `photo_corpus/` | the 422-photo benchmark (landmarks, labels, frozen split) |

## Model experiments (chronological)
| Script | Purpose / outcome |
|---|---|
| `modal_mlp_v3_aug.py`, `modal_eval_all.py`, `modal_eval_photos.py` | 2026-09-03: augmentation retrain and the fair real-photo comparison (`results_*_2026-09-03.json`) |
| `kaggle_mlp_photo_domain.py`, `kaggle_mlp_photo_v2.py`, `kaggle_mlp_photo_v3.py`, `kaggle_mlp_photo_v4.py` | training the MLP on real photos (v3 = all three heads trained; v4 = full photo corpus, was live until the cascade) |
| `kaggle_stgcn_transitions.py` | the transition-aware ST-GCN (`stgcn_transitions_v1`, the previous live sequence model) |
| `kaggle_per_class_audit.py` | which poses are learnable on the full photo corpus (chose the conference pose set) |
| `live_check/replay_live.py` | replay of the frozen photos through the live Space |
| `yt_probe/` | can Kaggle reach YouTube (metadata-only probe) |
| `kernel_metadata/` | exactly what each earlier Kaggle kernel was pushed with |

## Kaggle harness for the conference runs: `kaggle_transfer/`
`klog.py` (read a kernel's log through the REST API; never use `kaggle kernels output`), `hfcat.py` (print a private HF file to stdout without saving it), `kaggle_worker.py`, `runner_train.py`, `runner_transfer.py`,
`runner_probe.py` (environment probe), `shim/`, and one `kernel_*/` directory per run (each holds its `runner_*.py` and `kernel-metadata.json`): `kernel_stgcn*` (ST-GCN training/comparison/operating point),
`kernel_compare*`, `kernel_prod*`, `kernel_e2e`, `kernel_seq_e2e`, `kernel_live`, `kernel_replay` (live replay), `kernel_clips`, `kernel_publish`, `kernel_mlp`, `kernel_mpprobe`,
`kernel_cliff_probe` + `kernel_cliff_exp` (CLIFF vs mirror experiment, `docs/BENCHMARKS.md` section 10). `README_dataset.md` is the dataset card of `Arko007/Yoga-1M`.

## Rescued artifacts: `modal_rescue/` — NEVER DELETE
Trained checkpoints pulled out of Modal before its spend limit hit (`stgcn_transitions_v1.pth`, the 3 Sep MLP, label encoders). The three `stgcn_label_encoder_*.npy` files are byte-identical on purpose (one per z-handling variant).

## `archive/`
Superseded top-level files, see `archive/README.md`.
