---
license: other
license_name: third-party-video-research-only
language:
- en
pretty_name: Yoga Dataset Raw (source videos)
size_categories:
- n<1K
tags:
- yoga
- video
- raw-data
- provenance
- computer-vision
---

# 🎬 Yoga Dataset Raw: the source videos

The **original instructor-led yoga videos** behind the AsanaAI project (24 distinct videos, about 25 GB in this repository because several are stored in more than one folder layout), with their YouTube metadata. This repository exists for **provenance**: anyone can check that a landmark, a label or a benchmark number really came from these videos.

> ⚠️ **Rights.** The videos are third-party YouTube content. They are shared here only so that research results can be audited. **Do not redistribute or use them commercially.** The derived data (angles, landmarks, labels) is published separately in [Yoga-1M](https://huggingface.co/datasets/Arko007/Yoga-1M) and [Yoga-650k](https://huggingface.co/datasets/Arko007/Yoga-650k).

[Live app](https://yoga-posture-correction-system.vercel.app) · [Code](https://github.com/Anamitra-Sarkar/yoga-posture-correction-system) · [Yoga-1M](https://huggingface.co/datasets/Arko007/Yoga-1M) · [Yoga-650k](https://huggingface.co/datasets/Arko007/Yoga-650k)

---

## 🗂️ Layout

| Folder | Content | Notes |
|---|---|---|
| `Yoga_Dataset_Raw/` | The **first 12 videos**, named by YouTube id (`<id>.mp4`), plus `<id>.info.json` and `<id>.description` metadata, per-video `*_report.json`, `mlp_dataset.csv`, a few `stgcn_*.npy` sequence files and `processed_ids.txt` | Sub-folders `data/` and `metadata/` hold copies of the same files |
| `yoga_raw_dataset/` | The **same first 12 videos**, named `<upload date>_<title>.mp4`, with metadata and reports | Duplicate layout kept from the original upload |
| `new_videos_2026-10/` | **12 additional videos** added later (id-named `.mp4`, `.info.json`, `.description`) | Together with the first 12 they make the 24-video corpus of the final models |

So the file count is higher than the video count: **24 distinct videos** in total (12 + 12). Count videos by YouTube id, not by number of `.mp4` files.

## 🎞️ The 24 videos (YouTube ids)

**First 12:** `4ORRiN2_aVI`, `7ciS93shMNQ`, `8ibxmzJziHU`, `Eml2xnoLpYE`, `HmZFwoUU3WQ`, `L-z1HLkS_-Y`, `P8uHMMmWMHQ`, `QiebZSlTw_U`, `RQMtwbhXD7A`, `SZU7Sbgu57o`, `oUgpXY7QhpQ`, `s-1vMbAgYWU`.

**Added 12:** `149Iac5fmoE`, `4K2xTVRDJgA`, `4ZBUDd4bsyA`, `6CueZ4zujMk`, `EvMTrP8eRvM`, `JHjV-wFTwSw`, `O2EY79Ys_qg`, `ZiQh8jA5tVM`, `dAqQqmaI9vY`, `hHhxKkskHDg`, `i6TzP2COtow`, `v7AYKMP6rOE`.

The first 12 are vinyasa-style flows with almost no held poses.

## 🔗 How these videos became training data

```
24 videos ──► MediaPipe Pose (33 landmarks per frame) ──► 15 joint angles
          ──► Whisper transcripts (instructor names the pose)
          ──► cue-verified labels: transcript agrees AND rule engine agrees AND the body is held steady
          ──► Yoga-1M: 1,206,391 frames = 80,136 verified pose frames
                                        + 420,722 transition/unknown
                                        + 705,533 ignored
```

## ⚠️ Cautions

* The per-video `*_report.json` files and the `mlp_dataset.csv` / `stgcn_*.npy` files here come from the first extraction pipeline. They are **not** the training data of the current models; use Yoga-1M.
* Large files: the repository is about 25 GB. Download single files by name rather than cloning.

## 📜 Licence

No open licence is granted for the videos. Metadata (`.info.json`, `.description`) is as published by the video platform. Research and audit use only.
