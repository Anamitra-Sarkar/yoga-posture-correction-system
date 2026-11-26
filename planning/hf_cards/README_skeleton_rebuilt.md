# 🦴 Rebuilt Yoga Pose / Skeleton Dataset: preprocessing repository

Research-grade rebuild of a MediaPipe-derived yoga skeleton dataset from its
original raw sources, with full provenance, quality control, and an independent
audit. This repository reproduces the dataset **from raw video**; it does not
patch the historical CSV.

> **Headline finding.** The historical `master_mlp_dataset_fully_classified.csv`
> (654,488 rows) is **not** a filtered pose dataset. It is a *per-frame dump of
> every decoded frame* of the 12 source videos, in which ~83 % of rows carry
> MediaPipe output that the original pipeline's own report files admit was
> never actually detected. 99.36 % of its `pose_label` column is
> `transition/unknown`. Evidence and numbers are in
> `docs/FINDINGS_historical_dataset.md`.

> 📌 **Status and place in the project.** This repository is the **forensic audit** that showed the historical table ([`Yoga-650k`](https://huggingface.co/datasets/Arko007/Yoga-650k)) cannot be trusted as a pose dataset. The project then moved to cue-verified labels over all 24 source videos, published as [`Yoga-1M`](https://huggingface.co/datasets/Arko007/Yoga-1M); that is the data to use for new work. This rebuild covers the 12 first-release videos only (its YouTube step was blocked at the time; the 12 added videos are in [`yoga-dataset-raw`](https://huggingface.co/datasets/Arko007/yoga-dataset-raw)). Models: [`yoga-posture-models`](https://huggingface.co/Arko007/yoga-posture-models). App: https://yoga-posture-correction-system.vercel.app

---

## 1. 📂 What is in this repository

```
pipeline/yoga_core.py         canonical definitions: 33 landmarks, ST-GCN edges,
                              normalisation (verbatim from the notebook),
                              feature candidates, QC metrics
research/00_download_hf_videos.py   source-level dedup + verified download
research/00b_download_youtube.py   the 13 supplied YouTube sources
research/00b_audit_youtube.py       evidence of what could/could not be obtained
research/00c_source_registry.py     canonical source registry
research/01_historical_forensics.py what the old CSV actually contains
research/02_build_proxies.py        decode-once proxy generation
research/02b_smoke_mediapipe.py     MediaPipe sanity + throughput measurement
research/03_extract_landmarks.py    deterministic MediaPipe extraction + QC
research/03b_watcher.py             overlaps decode and inference
research/04_segment_and_label.py    segment manifest + rule-based pose labels
research/05_reverse_engineer_features.py  evidence-based feature definition search
research/06_sequences_and_split.py   ST-GCN tensors + source-level split
research/07_validate.py             the 12 independent audits
research/08_visual_verification.py  overlay panels + skeleton video previews
research/09_historical_comparison.py rebuilt vs historical agreement
research/10_stats_and_cards.py      dataset_statistics.json + DATASET_CARD.md
```

## 2. 🔎 Source inventory (what actually exists)

| | count | note |
|---|---|---|
| mp4 entries in the HF repo tree | 24 | the same 12 videos stored 3× each |
| **unique raw videos (sha256)** | **12** | 6.83 GB, all SHA-256 verified |
| report JSONs in the repo | 11 | 10 report `0` frames processed |
| supplied YouTube URLs | 13 | all identified via oEmbed |
| **YouTube URLs already present on HF** | **1** | `Eml2xnoLpYE` == HF `20201105_...` |
| **canonical sources after dedup** | **24** | 12 HF + 12 new YouTube |
| **sources obtainable here** | **12** | YouTube media is IP-blocked (see §7) |

`source_registry.csv` carries `source_id, source_type, source_url,
source_filename, youtube_id, hf_paths_in_repo, hf_copy_count, duration_sec,
fps, width, height, codec, sha256, sha256_verified, difficulty, status,
duplicate_of, notes` for all 25 rows.

## 3. 🛠️ Pipeline

```
00  inventory & dedup (by sha256, before any download)
00b acquire YouTube sources            00c source registry
01  forensics on the historical CSV
02  decode-once proxy  (10 fps, AR preserved, 640 long side)
03  MediaPipe Pose, 33 landmarks, model_complexity=1, deterministic
03b decode/inference overlap
04  segmentation + 15 features + rule-based labels
05  evidence-based reverse engineering of the 15 feature definitions
06  60-frame sequences + source-level split
07  twelve independent audits
08  visual verification (overlays + skeleton videos)
09  cross-validation against the historical CSV
10  statistics, dataset card, technical report
```

Run everything with:

```bash
bash run_all.sh          # idempotent; each stage skips completed work
```

## 4. 🎯 Design decisions that matter

**Proxy at 10 fps.** The corpus is 4K/1080p AV1/VP9/H.264. 4K AV1 decodes at
~10 fps on this 2-core host, i.e. ~9 h just to read every frame once. We pay
that cost exactly once into a proxy (`config.yaml: proxy`), then every later
stage reads the cheap proxy. A 60-frame ST-GCN window therefore spans **6.0 s**
of practice. Aspect ratio is preserved — squashing 16:9 into 4:3 would silently
distort every joint angle and therefore all 15 features.

**Model complexity 1, not 0.** Measured on 150 real yoga frames: the Lite model
is only 22 % faster (13.5 vs 11.1 fps) but disagrees with the full model by a
mean of 0.056 in normalised landmark units. Accuracy wins.

**Normalisation is verbatim from the notebook.** `normalize_coordinate_sequence()`
in `pipeline/yoga_core.py` is a character-for-character reproduction of
notebook cell 1 (pelvis = midpoint of landmarks 23/24, scaled by
‖hip_L − hip_R‖, `1e-5` guard). The ST-GCN edge list is likewise verbatim.

**RAW and NORMALISED are both kept.** `stgcn_features_raw.npy` is never
overwritten by normalisation.

**No frame-level train/test split.** Sequences are grouped by **source video**;
a source appears in exactly one split. Audit `11_split_leakage` asserts it.

**Non-overlapping windows.** `stride = 60` = `window`, so near-duplicate
sequences are impossible by construction rather than filtered out afterwards.

## 5. Requirements

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`numpy<2` is required by the pinned `mediapipe`/`jax` stack.
`ENVIRONMENT.txt` records the exact machine that produced the artefacts
(2× Xeon 8163, 3 GB RAM, **no GPU**, ffmpeg 5.1.9).

## 6. ⚠️ Known deviations from the original pipeline

| # | Deviation | Why | Documented in |
|---|---|---|---|
| 1 | 10 fps proxy, 6 s windows | 4K AV1 decode cost is ~9 h at native rate | `config.yaml`, dataset card |
| 2 | 12 of 24 sources | YouTube media IP-blocked from this host | `runs/run_001/youtube_acquisition_report.json` |
| 3 | Frames with no detection are `INVALID`, not filled | the historical pipeline emitted hallucinated poses for them | `docs/FINDINGS_historical_dataset.md` |
| 4 | `tree_pose` rules corrected | the notebook's `RULES['tree_pose']` defines `knee_r`/`hip_r` twice, so the first pair is silently overwritten by dict literal semantics | `docs/FINDINGS_historical_dataset.md` |
| 5 | 15 feature definitions are *reconstructed*, not assumed | the historical extraction code was never published | `runs/run_001/feature_reverse_engineering/` |

## 7. YouTube acquisition — honest status

`youtube.com` and its public **oEmbed** endpoint are reachable (all 13 titles
resolved), but **media extraction is blocked at the datacenter-IP level**:
`yt-dlp` returns *"Sign in to confirm you're not a bot"* for every player
client (`default, android, ios, tv_embedded, web_safari, mweb, android_vr`),
and no Piped/Invidious front-end returned a stream URL. The 12 genuinely new
YouTube sources are therefore recorded as `NOT_OBTAINED` and **excluded** —
they are **not** substituted with other videos. `Eml2xnoLpYE` is a duplicate of
an HF file and was deliberately not downloaded twice.
Full evidence: `runs/run_001/youtube_acquisition_report.{json,csv}`.

## 8. 🧾 Traceability

Any sample in the released dataset can be traced back to its origin:

```
sequence_metadata.csv  -> source_id, subject_id, segment_id, start_frame,
                          end_frame, start_time, end_time, label, quality
segment_manifest.csv   -> source_id, segment_id, time range, segment_type, reason
source_registry.csv    -> source_id, source_url, youtube_id, sha256, provenance
raw landmarks .npz     -> per frame: x, y, z, visibility, verdict, QC metrics
```

## 9. 📜 Licence / ethics note

The raw videos are third-party YouTube content obtained from the dataset
authors' Hugging Face mirror and are used here for academic research
provenance verification. Redistribution of the raw video files is not
authorised by this repository; the derived *skeleton/label* artefacts are the
research output.
