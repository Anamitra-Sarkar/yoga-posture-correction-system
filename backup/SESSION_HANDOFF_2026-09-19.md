# AsanaAI handoff — 2026-09-19

Terminal closed mid-session. Two Kaggle kernels were left RUNNING; they run
server-side and are unaffected. Everything below is committed and pushed to
`main`.

---

## 1. RESUME HERE: two Kaggle kernels left running

Check both first:

```bash
kaggle kernels status arkosarkarhehe/asanaai-stgcn-transitions
kaggle kernels status arkosarkarhehe/asanaai-photo-corpus-v2
```

Logs come from the API `log` field. Do NOT use `kaggle kernels output` on the
big ones (data budget).

### a) `asanaai-stgcn-transitions` (version 6) — ST-GCN attempt 4

**The bar is 63.0% macro.** That is `stgcn_relab_bonecorr` / uploaded as
`stgcn_transitions_v1.pth`: 24 classes, 8/8 transition classes learned, named
ones 72.7–100%, vs 54.6% for the old 15-class scheme.

Three previous attempts all regressed, for one consistent reason: each fix
recovered more labels, which created MORE classes (24 -> 38 -> 49), and macro
averages over classes. Attempt 3's label re-derivation was genuinely valuable
(cobra_pose 1880->2608, downward_dog 683->1345, plank 58->206, table_top
263->517) but fragmented the space at the same time. Attempt 4 keeps the richer
labels and adds a support floor: `MIN_PAIR = 90`, `MIN_SUPPORT = 100` applied
to EVERY class, holds included (earlier fixes policed only transitions).

Judge on named-transition per-class accuracy, since class counts differ.
**If it does not clear 63.0%, say so plainly and ship `stgcn_transitions_v1.pth`.**

### b) `asanaai-photo-corpus-v2` — expanded photo harvest

Commons + **Openverse** (new second pool, Flickr-sourced ordinary
practitioners), deeper paging, starved classes processed first.

When it finishes, push `planning/kaggle_mlp_photo_v2.py` as a GPU kernel with:

```json
"dataset_sources": ["arkosarkarhehe/asanaai-photo-testset",
                    "arkosarkarhehe/asanaai-stgcn-source"],
"kernel_sources": ["arkosarkarhehe/asanaai-photo-corpus-v2"]
```

It self-reports a verdict against **35.5%** and prints "do NOT promote" if the
bar is not cleared.

---

## 2. THE ONE-LINE CHANGE AWAITING SIGN-OFF

`backend/app/services/hf_loader.py` line 39 still pulls
`stgcn_sequence_model.pth` — the ORIGINAL 15-class model that has never seen a
transition. Verified live: `/api/analyse_sequence` answers
`"transition/unknown"`.

All the plumbing to swap it is merged and tested. Change the filename to
`stgcn_transitions_v1.pth` (+ `stgcn_transitions_v1_encoder.npy`) and the app
gains real transition detection. **Not done: promoting a model to live needs
explicit user sign-off.**

Why it was not just a filename swap (both are now handled, with tests):
* clients render `sequence_pose` directly, so `transition:warrior_2->plank`
  would have been shown as if it were the name of a pose;
* `SEQUENCE_FALLBACK_THRESHOLDS` is keyed on `child_pose`, so its tuned 0.55
  would have silently stopped applying to `hold:child_pose` and reverted to the
  0.70 default — a regression that does not throw, it just degrades.

---

## 3. Verified live this session (deployed Space, not local)

| check | result |
|---|---|
| `/api/analyse_frame` still + recognised | `motion_state: holding` |
| same angles, 120 deg/s | `motion_state: transitioning` |
| no `motion` field (old client) | `motion_state: unknown`, personal score `null` |
| with calibration | universal 0.866 -> personal **0.926**, `knee_l` 4.01deg -> 0 |
| `/api/generate_correction` en + hi | working (Hindi renders correctly) |
| `/api/occlusion_recovery` | recovered `left_knee` only, 33 landmarks back |
| `/api/analyse_sequence` | 200, old 15-class vocabulary |

So the free-form redesign (no pose selection, transition detection, dual
universal/personal scoring) is genuinely live and backward-compatible.

---

## 4. Real bugs found and fixed this session

**The LLM correction path had been silently dead for months.** `/api/generate_correction`
returned text byte-identical to the Stage-1 template. `GROQ_API_KEY` has been
set on the Space since 10 June, so the key was not the issue. Two defects, both
confirmed against the live Groq API:

1. `qwen/qwen3.6-27b` has been **decommissioned** — it was real when written,
   Groq now answers `404 model_not_found`. Current equivalent: `qwen/qwen3.8-27b`.
2. The payload sent a **system-only message list**, which that model rejects
   with `400 "No user query found in messages"` — so the call would have kept
   failing even with the id corrected.

Why it went unnoticed is the useful part: the non-200 branch was entirely
silent (no else, no log) and falling back to the reviewed template is a
correct, safe outcome — so a total outage looked exactly like success. Non-200s
are now logged. Timeout 5s -> 8s.

**`backend/app.py` could 500 on a mediapipe bump.** The try/except guarded only
`import mediapipe`, then called `mp.solutions.pose` on the next line — and
0.10.35 removed it. Now degrades with a message naming the installed version.

**Split-leak bug caught before it ran.** Openverse mirrors Wikimedia but returns
the full `upload.wikimedia.org` URL where our Commons search returns a width-900
thumb URL. Deduping on the raw URL would have admitted the same photo twice
under two ids and could have put it in BOTH train and test. Now deduped on a
canonical key, hashing a source-independent id. Verified: all 422 v1 images
reproduce their exact v1 split, 0 mismatches.

Also checked and found NOT to be bugs: `is_safe` (the UI already renders `false`
as "Fell back to reviewed template" — correct as designed); the `cobra_pose`
test fixture I first wrote (the classifier was right — `mountain_pose` is a
strict superset of cobra in 2D-angle space unless trunk <= 65, the documented
cost of dropping z).

---

## 5. Tests added (0 -> 35, all passing)

`backend/tests/` — `python3 -m pytest tests/ -q` from `backend/`.
No network, no model, no API key needed.

* `test_freeform_helpers.py` (15) — motion state, calibration, personal score
* `test_rules_warrior_order.py` (7) — the warrior_1/warrior_2 branch-order fix
* `test_correction_payload.py` (5) — Groq payload shape, asserted via **AST**
  so the comment documenting the old model id cannot make them pass by accident
* `test_sequence_labels.py` (8) — both label vocabularies

---

## 6. Deliberately NOT done

**Pinning `mediapipe`/`protobuf` in `backend/requirements.txt`.** Pinning is the
right end state, but the Space is running and serving correctly, the build log
returned only 29 lines mid-rebuild so the installed version could not be
established, and pushing a speculative pin to a live service ahead of a
submission is the riskier move. Do it once a completed build log can be read.

---

## 7. Known data-availability limits (honest, not fixable by retraining)

Five classes at 0%: `chair_pose`, `cobra_pose`, `plank`, `upward_dog`,
`standing_forward_fold` — all had ~10-21 training photos, `upward_dog` had 1.
`halfway_lift` and `chaturanga` had zero usable Commons photos; Openverse is
the only reason to expect different. `mountain_pose` regressed 30% -> 10%.
