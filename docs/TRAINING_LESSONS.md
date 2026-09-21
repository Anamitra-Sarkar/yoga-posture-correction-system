# AsanaAI — training lessons

Everything here was **measured on this project**, not taken from a textbook.
Each entry says what went wrong, how it was caught, and what to do instead.
Read this before writing another training script.

---

## 1. A random train/test split over video frames is meaningless here

**What happened.** The 2026-07-19 run reported:

```
Epoch 40/40 | Train Pose Acc 87.63% | Val Pose Acc 90.86% (Corr Acc 93.96%)
```

That looked excellent. Real-world macro on photographs was **10.5%**.

**Why.** The split was a random `train_test_split` over frames drawn from only
**12 source videos**, sampled densely. Consecutive frames of a held pose are
near-duplicates, so almost the same image sat on both sides of the split. The
number measured memorisation, not generalisation.

**Rule.** Split by **source**, never by row. Hold out whole videos (or whole
people). `kaggle_mlp_photo_v3.py` holds out 3 of 12 videos; the ST-GCN scripts
hold out named videos via `HOLDOUT`.

**How to catch it.** If validation accuracy is far above real-world accuracy,
suspect the split before suspecting the model.

---

## 2. Sample count is not information count

654,488 video frames came from **12 videos ≈ 26 independent (video × pose)
observations**. Measured consequence: **448 real photographs beat 654,488
video frames** (47.5% vs 38.9% macro on an identical held-out photo set).

**Rule.** Count independent *sources*, not rows. When asked for "more data",
ask whether it adds sources or just more rows from the same ones.

---

## 3. Train on the distribution you are judged on

The MLP was trained on video frames and judged on photographs. Training on
real photos took macro from **14.1% → 35.5%** on the same test set.

Oversampling was load-bearing: 319 photos against 650k frames is 2000:1, and
without oversampling the mixed variant scored 18.4% — barely different from
video-only.

---

## 4. Relative joint angles are blind to whole-body rotation

All 15 features are relative angles (shoulder-hip-knee etc.), so they are
invariant to rotating the body. Measured: the rule engine called **corpse
"mountain_pose" 11 times out of 18** — lying flat and standing upright produce
nearly the same 15-vector.

**No threshold change can fix this.** The information is not in the vector.
Fixed by adding two global cues (`torso_incline`, `leg_torso_ratio`), which
took rule recall from 23.7% → 36.3% with no regressions.

---

## 5. Some poses are genuinely unseparable in 2D — do not fake it

Front-on `seated_staff` vs `mountain_pose`:

| cue | seated_staff | mountain_pose |
|---|---|---|
| leg/torso ratio | 1.27 | 1.33 |
| ankle drop | 1.25 | 1.26 |
| torso incline | 3° | 6° |

An early rule "detected" seated_staff at **76.7%** — purely by taking
mountain_pose's photos. Net system accuracy did not improve; the error moved.

**Rule.** When a fix raises one class and drops another by a similar amount,
it is a reshuffle. Always check the *other* classes before claiming a win.

---

## 6. Macro average punishes recovering more classes

Three attempts to improve the 63.0%-macro transition ST-GCN all regressed:

| attempt | change | classes | macro |
|---|---|---|---|
| baseline | — | 24 | **63.0%** |
| 2 | `MIN_PAIR=12` | 92 | 12.6% |
| 3 | `MIN_PAIR=50` | 38 | ~13.7% |
| 4 | label re-derivation | 49 | 12.5% @ ep00 |

Each fix recovered *more* labels, creating *more* classes, and macro averages
over classes — so better label coverage looked like catastrophic regression.

**Rule.** Impose a support floor on **every** class (holds and transitions
alike), and when class counts differ, compare per-class accuracy on shared
classes, not raw macro.

---

## 7. Train every head you deploy

The photo-domain scripts optimised only the pose head:

```python
pl, _, _ = m(Xt[b]); crit(pl, yt_[b]).backward()
```

`correctness_head` and `deviation_head` stayed at random init. Harmless for a
data experiment — **not** harmless once that checkpoint shipped, because
`hybrid_classify` serves `mlp_correctness` and `mlp_devs` directly whenever the
MLP and rules agree. Pose accuracy tripled while the per-joint coaching numbers
became noise.

The original trainer had always done it properly:
`loss = loss_pose + 1.0*loss_correct + 1.0*loss_dev`.

**Rule.** Before promoting a checkpoint, check every output the serving path
reads is actually supervised.

---

## 8. Silent fallbacks hide total outages

`/api/generate_correction` returned the safe template for months. `GROQ_API_KEY`
was set; the model id had been decommissioned (404), and the non-200 branch had
**no else and no log**. Falling back to a reviewed template is a *correct*
outcome, which is exactly why the failure was invisible.

The same bug recurred in new code days later: the Openverse harvest returned
**0 images for all 22 classes** because anonymous requests cap `page_size` at
20 and the code asked for 100 — and `_get()` swallowed the 401.

**Rule.** Whenever a fallback is a correct outcome, log the reason it fired.

---

## 9. Verify with the parameters the code actually sends

The pre-flight probe that "verified Openverse works" used `page_size=3`. The
real code used `page_size=100`, which is rejected. The probe passed; the
feature was dead.

**Rule.** A probe that does not use production parameters has verified nothing.

---

## 10. Deduplicate on content, not filename

Public yoga datasets overlap heavily — `shrutisaxena`'s `1. 1.png` and
`tr1gg3rtrash`'s `File1.png` are byte-identical (157390 bytes). Deduping by
filename would count the same photo repeatedly **and place it in both train and
test**.

**Rule.** Key images by a hash of decoded pixels. Key split assignment by a
source-independent stable id.

---

## 11. Uploading a model is not deploying it

The 63.0% transition ST-GCN sat unused on HF for days because `hf_loader.py`
still named the old file. Confirmed live: `/api/analyse_sequence` answered
`"transition/unknown"` for everything.

**Rule.** After promoting, verify a **behaviour change on a fixed input** — not
that the deploy was green.

---

## 12. Checkpoint key names must match the serving class

Checkpoints trained with `block1.res.*` cannot load into production, which
expects `block1.residual.*`. `load_state_dict` fails outright.

**Rule.** Strict-load every checkpoint into the real production class, with a
forward pass, before upload.

---

## 13. "Proven hyperparameters" are proven for a TASK, not an architecture

**What happened.** The transition ST-GCN was training for 30 epochs, no label
smoothing, `weight_decay=1e-4`. The 2026-07-19 run that produced a working
ST-GCN used 120 epochs, patience 20, `label_smoothing=0.1`,
`weight_decay=1e-3`, `eta_min=1e-5`. I adopted that config on the reasoning
that it was "already proven on this exact architecture and data".

**Result: macro 63.0% -> 31.8%.** Half.

The config was proven on the **15-class non-transition** classifier. The
transition-aware problem has 25 classes, several sitting just above the
support floor, and is judged on **macro** — which weights every class equally.
Label smoothing caps the confidence reachable on exactly those thin classes.
Nothing about "same architecture, same data" made the setting transfer.

**Worse, I changed five things at once** (epochs, patience, smoothing, weight
decay, eta_min), so the run reports only "worse" and cannot say which one did
it. Lesson 6 in this same document already warns about comparing across
different class counts, and the project's own Zenyx notes say *change ONE
thing at a time from the proven baseline*. I had written both down and still
did it.

**Rules.**
* Hyperparameters transfer across tasks only as a hypothesis to test, never as
  a justification to skip testing.
* Change one thing at a time from a working baseline, or the result is
  uninterpretable even when it is dramatic.
* An incumbent that survives a challenge is the correct outcome, not a wasted
  run — `stgcn_transitions_v1` (63.0%) stays live.

---

## Proven hyperparameters (ST-GCN, from the 2026-07-19 run)

Kept because they worked **on the 15-class non-transition task**. Measured NOT
to transfer to the 25-class transition task (see lesson 13: 63.0% -> 31.8%).
Change one at a time.

```python
batch_size    = 64
hidden_dim    = 128
epochs        = 120
patience      = 20            # early stop on val ACCURACY, not loss
dropout       = 0.2 / 0.3 / 0.3      # block1 / block2 / block3
criterion     = CrossEntropyLoss(weight=1/sqrt(counts) normalised,
                                 label_smoothing=0.1)
optimizer     = AdamW(lr=1e-3, weight_decay=1e-3)
scheduler     = CosineAnnealingLR(T_max=epochs, eta_min=1e-5)
```

MLP 3-head: `loss = loss_pose + 1.0*loss_correct + 1.0*loss_dev`,
AdamW lr=1e-3 wd=1e-4, cosine over 40 epochs.

---

## Current honest numbers

| thing | number | measured on |
|---|---|---|
| MLP pose (live) | 35.5% macro | frozen 103 real photos, 23 classes |
| MLP pose (best measured) | 47.5% macro | same set, photos-only recipe |
| old live MLP | 10.5% macro | same set |
| ST-GCN transitions (live) | 63.0% macro | 2 held-out **videos**, 24 classes |
| rule engine | 36.3% recall | 422 real photos |
| poses with correctness bands | 19 of 23 | — |

`chaturanga`, `seated_forward`, `upward_dog` have **no** correctness bands: the
corpus holds 0, 5 and 1 usable examples. Invented bands would produce confident
corrections never checked against anything.
