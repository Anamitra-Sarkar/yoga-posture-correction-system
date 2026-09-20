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

**THIRD cause, found after deploying, needs YOUR action:** with the logging in
place the Space immediately reported the real remaining failure on every call:

```
Groq correction call returned 401, falling back to template:
{"error":{"message":"Invalid API Key","code":"invalid_api_key"}}
```

So the code fix is deployed and correct, but the `GROQ_API_KEY` **secret stored
on the Space is stale**. The key in
`~/Downloads/API_Keys_and_Secrets/groq_api.txt` is valid — it was used to
confirm both the model id and the corrected payload shape, returning HTTP 200
with a proper Hindi paraphrase.

Updating it is a change to account settings involving a credential, so it was
deliberately NOT done automatically. Do it either way:

* **Web UI:** huggingface.co/spaces/Arko007/yoga_pose/settings -> Variables and
  secrets -> edit `GROQ_API_KEY` -> paste the key from `groq_api.txt`.
* **Terminal:** prefix with `!` in Claude Code so the output lands in the
  session:

```bash
python3 -c "
from huggingface_hub import HfApi
k=open('/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt').read().strip()
t=open('/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token').read().strip()
HfApi(token=t).add_space_secret('Arko007/yoga_pose','GROQ_API_KEY',k)
print('updated; the Space will restart')"
```

Then confirm with `bash backup/verify_live.sh` — once the English correction
is no longer the exact template string, the LLM path is genuinely live.

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


---

## 8. UPDATE 2026-09-20: the real bug, found and fixed

User reported the live model was "too bad" and asked to diagnose, get more
data if needed, or ship the best already-available checkpoints. Diagnosis
first: re-measured the then-live `mlp_3head_model_v2.pth` against the app's
actual judging condition (the full 23-class vocabulary on the frozen 103-photo
test set, not the 6-pose/19-photo set the 52.6% headline came from). Real
score: **10.5% macro / 13.6% overall.** That was the actual bug -- a stale
benchmark had been masking how bad the live checkpoint really was.

**Fixed by promoting two already-built, already-verified checkpoints that were
sitting unused on HF** (no new training needed for this part):

* `mlp_3head_photodomain_v1.pth` -- measured on the identical frozen set:
  **35.5% macro / 45.6% overall**, a >3x gain. Same 23-class vocabulary and
  order as the old checkpoint (diffed both encoder files), so it was a pure
  swap, no remapping.
* `stgcn_transitions_v1.pth` -- the old `stgcn_sequence_model.pth` had never
  been shown a transition (confirmed live: always answered
  `"transition/unknown"`). This one: 63.0% macro, all 8 transition classes
  learned. Re-verified with a strict `state_dict` load into a standalone
  reimplementation of the real production class plus a forward pass
  (30 classes, finite output) before trusting last session's claim.

**Both confirmed live** via behavioural change on identical requests (the
calibration smoke test now scores 0.726 instead of 0.866; `analyse_sequence`
on the same synthetic input now returns `"chair_pose"` instead of always
`"transition/unknown"`), and via the Space's own log: "All models successfully
fetched and loaded into memory."

`backend/tests/test_mlp_checkpoint_choice.py` re-measures both MLP checkpoints
against a live HF pull on every test run, so this justification stays checkable
from the repo, not just this session. Network-marked, skips (never fails)
offline.

### Kaggle account blocker, worked around

The local CLI was authenticated as `anamitrasarkar007`, not `arkosarkarhehe`
(the account the two kernels from the previous session were running under) --
`403 Forbidden` on both. Modal, the documented route to that account, was over
its spend limit. Did not attempt to route around either restriction by
extracting or trying stored credentials for the other account.

Instead, pursued "get more data yourself" under the account that WAS
accessible, using files that already existed locally (no new downloads):

* `anamitrasarkar007/asanaai-photo-corpus-v1` -- new dataset, the existing
  422-photo v1 corpus.
* `anamitrasarkar007/asanaai-mlp-dataset-zeroz-fullyclassified` -- already
  existed from an earlier session (the same 650k-frame video CSV), reused
  rather than re-uploaded.
* `anamitrasarkar007/asanaai-photo-corpus-v2` -- the Commons+Openverse harvest
  kernel, pushed and running.
* `anamitrasarkar007/asanaai-mlp-photo-v2` -- queued to auto-push and run the
  moment the harvest kernel completes (a monitor is watching for this; if it
  is not still running when you read this, check its outcome directly).

**This is a bonus improvement attempt layered on an already-shipped, already-
verified fix** -- nothing above depended on it, and the app was already fixed
before this was kicked off. If it lands a further win, it's a strict
improvement to swap in next; if not, the two checkpoints already live are the
ones to ship.

### One pre-existing, unrelated issue noticed (not fixed, out of scope)

The Space log also shows the SMPL occlusion-recovery model failing to load
(`401 Repository Not Found` for `Arko007/smpl-models`) -- likely the same
class of stale-Space-secret issue as the Groq key. It degrades gracefully to
"Symmetric Kinematic solver (4GB RAM Optimization Fallback)", confirmed
working in `verify_live.sh`'s occlusion_recovery check. Not part of "model is
too bad" (that was about pose/correctness quality), so left alone rather than
chased under time pressure -- flagging for whenever the Groq secret gets
rotated, since it may be the same root cause.


---

## 9. UPDATE: terminal closed while the data-improvement harvest was mid-run

The core fix (section 8) is done, live, and does not depend on anything below.

`anamitrasarkar007/asanaai-photo-corpus-v2` (Commons + Openverse harvest) was
still `RUNNING` on Kaggle's servers when the terminal closed -- it keeps
running regardless, same as any Kaggle kernel. What does NOT survive is the
local watch loop that was going to auto-push the retrain kernel the moment the
harvest finished; that loop lived in the closed terminal, not on Kaggle.

**To resume:**

```bash
kaggle kernels status anamitrasarkar007/asanaai-photo-corpus-v2
```

If `COMPLETE`, push the retrain kernel manually:

```bash
cd /tmp/claude-1000/-home-anamitra/d0057ab9-c807-4879-8f1e-a3647448be7a/scratchpad/mlp_v2_a007
kaggle kernels push -p .
kaggle kernels status anamitrasarkar007/asanaai-mlp-photo-v2
```

(if that scratchpad directory is gone -- it can get wiped between sessions --
regenerate it: copy `planning/kaggle_mlp_photo_v2.py` to
`asanaai-mlp-photo-v2.py` alongside a `kernel-metadata.json` with `id:
anamitrasarkar007/asanaai-mlp-photo-v2`, `enable_gpu: true`,
`dataset_sources: ["anamitrasarkar007/asanaai-photo-corpus-v1",
"anamitrasarkar007/asanaai-mlp-dataset-zeroz-fullyclassified"]`,
`kernel_sources: ["anamitrasarkar007/asanaai-photo-corpus-v2"]`).

If `ERROR`, read the failure via the API `log` field (not `kaggle kernels
output`) before deciding whether to retry.

Once the retrain finishes, it self-reports a verdict against **35.5%** (the
checkpoint already live) on the frozen 103-photo test set, and says "do NOT
promote" if it does not clear that bar. Only wire it into `hf_loader.py` if it
does.
