# Real-photo corpus (2026-09-19)

422 real photographs -> MediaPipe landmarks, 20 pose classes, harvested from
Wikimedia Commons (freely licensed, so this benchmark is redistributable and
citable). Replaces the previous 26-photo/11-class set, on which every
per-class number was noise.

- `photo_corpus.npz`: landmarks [422,33,4], world [422,33,3], labels, split
- split is a deterministic hash of the Commons title -> train 319 / test 103,
  reproducible and guaranteed disjoint
- filtered to genuinely full-body detections (mean visibility of shoulders/
  hips/knees/ankles >= 0.55) so a sample reflects the POSE, not the framing

## WHY THIS EXISTS
The MLP is a single-frame classifier trained on frames from 12 vinyasa videos
but judged on photographs. That mismatch IS the measured failure: ~94% on
held-in video frames vs 52.6% on real photos. Landmark augmentation was tried
to bridge it and did NOT work (24.2% vs 25.8% macro) -- synthetic jitter
around a narrow corpus cannot invent real photographic diversity. So: train on
the kind of data we are judged on.

## HONEST LIMITS (state these in the paper, do not paper over them)
- `halfway_lift` and `chaturanga`: **0** usable images. `upward_dog`: 1.
  `table_top`, `seated_forward`: 5. Freely-licensed full-body photographs of
  those asanas essentially do not exist on Commons. Per-class figures for
  them are not meaningful at any sample size we can reach this way.
- Only 8 of 20 classes reach n>=20.
- Keep rate is low by design (e.g. warrior_2: 206 candidates -> 18 kept). The
  visibility filter rejects crops and partial bodies. 18 clean samples beat
  100 where the label describes the framing.
