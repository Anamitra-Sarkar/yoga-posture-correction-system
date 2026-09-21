# AsanaAI handoff — 2026-09-21

## THE GOAL (do not drift from this)

**Same two models, better real-world metrics.** The architecture is fixed by
the final-year proposal (P05, RCC Institute of Information Technology,
supervisor Mr. Sujit Chakraborty) and must not be swapped out:

* **3-head MLP** — pose identity, binary correctness, 15-joint deviation
  vector, from a single 15-D biomechanical frame.
* **ST-GCN** — 60-frame skeletal sequences for transition / flow analysis.

Plus, in the user's own words:

1. **More pose compatibility and correctness.** The original complaint was
   that only ~3 poses classified correctly in the real world and the rest were
   excluded. Every pose should be recognised AND get a real correctness score.
2. **Genuine metrics, never faked.** "Do not fake .. make it better." If a
   number does not clear its bar, say so and do not promote.
3. **Do not delete or overwrite existing model artifacts** — only add new
   names. (All 14+ files on HF are intact; nothing has ever been overwritten.)
4. **Arko007 HF account only.** Never the bhumika account.
5. Aim higher than a typical final-year project — the closed-loop
   correction-efficacy work is the genuinely novel contribution (see §5).

**Never**: download datasets to the local PC (3.7GB RAM), run npm installs or
builds locally, or use `kaggle kernels output` on large kernels.


Supersedes the 2026-09-19 handoff. Read `docs/TRAINING_LESSONS.md` first (13
lessons, all measured on this project).

---

## 1. RESUME HERE — jobs that were running at handoff

All on the **`anamitrasarkar007`** Kaggle account (the `arkosarkarhehe`
account is unreachable from this machine: local CLI is a different account and
Modal is over its spend limit — not routed around).

```bash
kaggle kernels status anamitrasarkar007/asanaai-mlp-v4      # bar: 47.8%
kaggle kernels status anamitrasarkar007/asanaai-stgcn-v6    # bar: 63.0%
```

Logs: use the API `log` field (see the snippet in §7). **Never**
`kaggle kernels output` on big kernels — data budget.

* **`asanaai-mlp-v4` (version 2)** — re-run against the REPAIRED harvest.
  v4 v1 trained on the old 553-photo harvest; the Openverse fix took it to
  2,032. Corpus is now v1 (422) + harvest (2,032) + public (5,655), deduped by
  decoded-pixel hash. **Promote only if it beats 47.8%** on the frozen 103.
* **`asanaai-stgcn-v6`** — adds the 17-person corpus with a **cross-person**
  holdout. Hyperparameters deliberately unchanged from the 63.0% config;
  **data is the only variable**. Promote only if it beats 63.0%.

### If v6 misses the bar

`stgcn_transitions_v1` (63.0%) stays live. That is already the case —
`hf_loader.py` has never been pointed away from it.

---

## 2. What is LIVE right now

| component | file on HF | measured |
|---|---|---|
| MLP | `mlp_3head_v4_photos_x2000.pth` | **47.8%** macro, frozen 103 photos |
| ST-GCN | `stgcn_transitions_v1.pth` | **63.0%** macro, held-out videos |

The MLP was promoted today. It is the **first checkpoint whose correctness and
deviation heads are actually trained** — see §4.

Rule engine: **36.3%** recall on 422 real photos, **17 of 22** poses with real
per-joint correctness bands, `DISABLED_POSES` empty.

**Every earlier checkpoint remains on HF untouched.** Nothing has ever been
overwritten or deleted.

---

## 3. Results this session

### MLP v4 — PROMOTED (47.8%, from 35.5%)

| variant | photo macro | held-out video | corr acc | dev MAE |
|---|---|---|---|---|
| **photos_x2000** (live) | **47.8%** | 65.9% | 91.5% | 2.19° |
| photos_x365 | 41.9% | 74.7% | 91.9% | 1.07° |
| photos_x1 | 30.8% | 84.3% | 94.1% | 0.39° |

Re-derived from the saved weights independently, not taken from the kernel's
own report. Encoder verified identical in content **and order**, so the swap
could not relabel classes.

Starved classes came alive: `cobra_pose` 0→100%, `triangle`→100%,
`corpse`→100%, `seated_staff` 0→80%, `chair_pose` 0→25%.

**Caveat, stated in the loader too:** `seated_staff` at 80% is partly the
mountain_pose collision resolved the other way — `mountain_pose` fell to 10%.
Net positive across the pair, but not a free win. Watch it.

**Trade-off:** heavier photo oversampling helps diverse imagery, costs
same-shoot video. x2000 chosen because the app faces arbitrary users in
arbitrary rooms. `mlp_3head_v4_photos_x365` is the balanced fallback.

### ST-GCN v5 — FAILED (31.8% vs 63.0%), my mistake

I swapped in the 2026-07-19 hyperparameters arguing they were "proven on this
exact architecture". They were proven on a **different task** (15-class
non-transition). On the 25-class transition problem: **63.0% → 31.8%**.

I also changed **five things at once**, so the run can only say "worse", not
which. Both mistakes were already written in my own lessons doc. Now lesson 13.

Reverted. Nothing shipped — the loader never pointed at it.

### Data — the real unlock

* **Public datasets**: 5,655 photos from 4 public yoga datasets. **848
  byte-identical duplicates** caught by hashing decoded pixels (filename
  dedup would have missed all of them AND straddled train/test).
* **Openverse fixed**: harvest 553 → **2,032**. Anonymous `page_size` caps at
  20; the code asked for 100 and got 401 on **every** request, silently. The
  probe that "verified" it used `page_size=3` — see lessons 8 and 9.
* **17-person video corpus**: 596 windows, 6 poses. Only ~1% of the 54,010
  existing windows, so it will not move macro by volume — its value is the
  **cross-person holdout**, the first honest "does this work on an unseen
  body" number.

---

## 4. The regression I shipped and then fixed (read this)

The photo-domain scripts optimised **only the pose head**:

```python
pl, _, _ = m(Xt[b]); crit(pl, yt_[b]).backward()
```

`correctness_head` and `deviation_head` stayed at **random initialisation**.
Fine for a data experiment — not fine once promoted on 2026-09-20, because
`hybrid_classify` serves `mlp_correctness` and `mlp_devs` directly whenever the
MLP and the 2D rules agree (the confident, common case). **Every per-joint
coaching number the app showed for a day was noise.**

Fixed in v4. `test_mlp_checkpoint_choice.py` now asserts the heads are not at
random init (a fresh sigmoid head barely leaves 0.5; this one spans 0–1 with
std 0.42, deviations reach 131°), so it cannot ship twice.

---

## 5. Shipped and verified live

* **Orientation features** — all 15 features are relative joint angles and so
  blind to whole-body rotation; the engine called `corpse` "mountain_pose"
  11/18 times. Two global cues (`torso_incline`, `leg_torso_ratio`) took rule
  recall **23.7% → 36.3%**. Verified live: identical 15 angles return
  `mountain_pose` at incline 5 and `corpse` at 100.
* **Correction-efficacy loop** — backend returns `target_joint`; client
  measures that joint over an 8s window and escalates: plain cue → cue +
  measured magnitude → **back off**. Escalation ends in backing off, not
  pushing harder. Verified live across all three tiers.
* **Groq fixed** — model id was decommissioned (404) AND the payload was
  system-message-only (400). Both fixed, non-200 now logged.
  **STILL BROKEN: the Space's `GROQ_API_KEY` secret is invalid (401).** See §6.
* **Display stabilisation** — sticky label with hysteresis + EMAs. Changes
  only *when* the display updates, never what the model predicts.
* **UI, all three clients** — free-form copy in 3 languages, motion-state
  badge, efficacy chip. Expo rebuilt around the detected pose.
* **APK** — public download, no GitHub login needed:
  `github.com/Anamitra-Sarkar/yoga-posture-correction-system/releases`
  Rebuild: `gh workflow run android_build.yml -f publish_release=true`

53 backend tests pass. `bash backup/verify_live.sh` smoke-tests all endpoints.

---

## 6. ONE THING NEEDS YOU

The Space's **`GROQ_API_KEY` is invalid** — 401 on every call, so the LLM
paraphrase silently falls back to the reviewed template. The key in
`~/Downloads/API_Keys_and_Secrets/groq_api.txt` works (verified: 200 with a
proper Hindi paraphrase). Not rotated automatically — it is a credential
change on your account.

```bash
python3 -c "
from huggingface_hub import HfApi
k=open('/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt').read().strip()
t=open('/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token').read().strip()
HfApi(token=t).add_space_secret('Arko007/yoga_pose','GROQ_API_KEY',k)
print('updated; the Space will restart')"
```

---

## 7. Open decisions

1. **Expel weak poses?** You said poses with bad recall "have the right to get
   expelled". Agreed in principle — a pose detected at 5% is worse than
   useless because it also steals other poses' predictions. Deliberately NOT
   done yet: five of the worst had almost no training data and may now work
   after the 10× corpus. `end_to_end_eval.py` (in scratchpad, regenerate if
   wiped) measures MLP+rules together, which is the right basis.
2. **`chaturanga`, `seated_forward`, `upward_dog` have no correctness bands** —
   the corpus had 0, 5 and 1 usable examples. The public corpus now has 271,
   268 and 51, so bands can finally be fitted honestly.
3. **Known hard limit — do not "fix" by widening bands.** Front-on
   `seated_staff` and `mountain_pose` are genuinely indistinguishable in 2D
   (leg/torso 1.27 vs 1.33, ankle-drop 1.25 vs 1.26). An early rule "detected"
   seated_staff at 76.7% purely by stealing mountain_pose's photos. Removed.

### Reading a kernel log

```python
import json, urllib.request, base64
u,k = "<user>","<key>"   # ~/.kaggle/kaggle.json
r = urllib.request.Request(
  "https://www.kaggle.com/api/v1/kernels/output?username=anamitrasarkar007&kernel_slug=<slug>",
  headers={"Authorization":"Basic "+base64.b64encode(f"{u}:{k}".encode()).decode()})
d = json.load(urllib.request.urlopen(r))
e = json.loads(d["log"])
print("\n".join(x.get("data","") for x in e if isinstance(x, dict))[-4000:])
```

### Datasets on `anamitrasarkar007` (all ready)

`asanaai-photo-corpus-v1`, `asanaai-master-mlp-dataset`,
`asanaai-stgcn-source-v2`, `asanaai-mlp-dataset-zeroz-fullyclassified`.

Kernel metadata for re-pushing any run: `planning/kernel_metadata/`.


---

## 8. Terminal was closed while jobs were running

The Kaggle kernels keep running server-side — closing the terminal does not
touch them. What does NOT survive is the local watcher loop, so nothing is
auto-chained any more; the next session must check and act manually.

```bash
kaggle kernels status anamitrasarkar007/asanaai-mlp-v4      # bar 47.8%
kaggle kernels status anamitrasarkar007/asanaai-stgcn-v6    # bar 63.0%
```

**If a run COMPLETEd, read its log (§7 snippet) and compare against the bar
before doing anything else.** Promote only on a clear win.

### To promote an MLP checkpoint

1. Download the `.pth` + encoder from the kernel output (small files, fine).
2. **Re-measure it yourself** on the frozen 103 rather than trusting the
   kernel's printout — `scratchpad/verify_v4.py` does this; regenerate it from
   the pattern in `backend/tests/test_mlp_checkpoint_choice.py` if the
   scratchpad was wiped.
3. Check the encoder matches the live one in content AND order.
4. Check the correctness/deviation heads are not at random init (std > 0.15,
   range > 0.5, deviations reaching tens of degrees).
5. Upload under a NEW filename; never overwrite.
6. Point `hf_loader.py` at it, run `pytest tests/ -q`, push.
7. After deploy, verify a BEHAVIOUR CHANGE on a fixed input — not a green
   deploy (lesson 11).

### To promote the ST-GCN

Same, plus a strict `load_state_dict` into the real production
`YogaSequenceLSTM` with a forward pass, and confirm `block1.residual.*` key
naming (lesson 12).

---

## 9. Still open (unchanged by the terminal closing)

1. **Rotate the Space's `GROQ_API_KEY`** — §6. One command, needs the user.
2. **Expel weak poses** — judged end-to-end after the v4 re-run, not on
   rule-engine recall alone.
3. **Fit bands for `chaturanga`, `seated_forward`, `upward_dog`** — they now
   have 271 / 268 / 51 photos, so honest bands are finally possible.
4. **ST-GCN beyond v6** — its real ceiling is source diversity. The 17-person
   corpus covers only 6 poses and no transitions, so the 9 named transition
   classes still rest entirely on the original 12 videos.
