"""Per-class audit on the FULL photo corpus — which poses are actually learnable?

WHY THIS BEFORE ANY RETRAIN
---------------------------
The pose set for the conference has to be chosen from evidence. The numbers
available so far come from the frozen 103-photo set, where most classes have
n=1-8: `cobra_pose` "100%" is two photos, `plank` "0%" is one. Ranking poses
on that would be guessing with a percentage sign on it.

The full corpus (v1 + Commons/Openverse harvest + four public datasets,
~8,000 photos) gives a test split of roughly 2,000 with n=50-300 for most
classes. That is enough to say which poses a 2D model can actually hold.

WHAT IT REPORTS
---------------
For every pose, on the held-out split:
  n, recall, precision, and WHAT IT IS CONFUSED WITH.

The confusion column is the point. A pose at 40% that scatters randomly is a
data problem — more examples will help. A pose at 40% that loses almost all
its mass to ONE other pose is a representational collision, and no amount of
data will fix it, because the two poses produce near-identical 15-angle
vectors (measured: front-on seated_staff vs mountain_pose differ by 0.06 in
leg/torso ratio). Those two situations call for opposite decisions, and
recall alone cannot tell them apart.

Also reports, per pose, how much of the TRAIN split it has, so a low score
can be attributed to scarcity rather than difficulty.

Loads the live checkpoint from HF rather than training anything: this is an
audit of what the deployed model can do, not a new experiment.
"""
import collections
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

OUT = "/kaggle/working"
HF_REPO = "Arko007/yoga-posture-models"
CHECKPOINT = "mlp_3head_v4_photos_x2000.pth"
ENCODER = "mlp_3head_v4_encoder.npy"


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.block(x)


class Yoga3HeadMLP(nn.Module):
    def __init__(self, input_dim, num_poses, num_joints=15):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU())
        self.res1 = ResBlock(256, 0.3)
        self.res2 = ResBlock(256, 0.3)
        self.pose_head = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
                                       nn.Dropout(0.2), nn.Linear(128, num_poses))
        self.correctness_head = nn.Sequential(nn.Linear(256, 64), nn.BatchNorm1d(64), nn.GELU(),
                                              nn.Dropout(0.2), nn.Linear(64, 1))
        self.deviation_head = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
                                            nn.Dropout(0.2), nn.Linear(128, num_joints))

    def forward(self, x):
        f = self.res2(self.res1(self.input_layer(x)))
        return self.pose_head(f), self.correctness_head(f).squeeze(-1), self.deviation_head(f)


NOSE = 0
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
HP_L, HP_R, KN_L, KN_R, AN_L, AN_R, HE_L, HE_R = 23, 24, 25, 26, 27, 28, 29, 30


def angles_from_landmarks(pts):
    p = np.asarray(pts, dtype=np.float64)[:, :3].copy()
    p[:, 2] = 0.0

    def a(x, y, z):
        ba, bc = p[x] - p[y], p[z] - p[y]
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nb == 0 or nc == 0:
            return 180.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ba, bc) / (nb * nc))))))

    sm = (p[SH_L] + p[SH_R]) / 2.0
    hm = (p[HP_L] + p[HP_R]) / 2.0
    ba, bc = p[NOSE] - sm, hm - sm
    nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    neck = 180.0 if (nb == 0 or nc == 0) else math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(ba, bc) / (nb * nc))))))
    return [a(SH_L, EL_L, WR_L), a(SH_R, EL_R, WR_R), a(HP_L, SH_L, EL_L), a(HP_R, SH_R, EL_R),
            a(SH_L, HP_L, KN_L), a(SH_R, HP_R, KN_R), a(HP_L, KN_L, AN_L), a(HP_R, KN_R, AN_R),
            a(KN_L, AN_L, HE_L), a(KN_R, AN_R, HE_R), a(SH_L, HP_L, HP_R), a(SH_R, HP_R, HP_L),
            neck, a(HP_R, HP_L, KN_L), a(HP_L, HP_R, KN_R)]


def orientation(pts):
    """The two global cues the relative joint angles cannot express. Reported
    per pose so the write-up can say WHY a pair collides, not just that it
    does."""
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    sm = (p[SH_L] + p[SH_R]) / 2.0
    hm = (p[HP_L] + p[HP_R]) / 2.0
    v = hm - sm
    n = np.linalg.norm(v)
    if n < 1e-9:
        return float("nan"), float("nan")
    inc = math.degrees(math.acos(max(-1.0, min(1.0, float(v[1] / n)))))
    leg = (np.linalg.norm(p[AN_L] - p[HP_L]) + np.linalg.norm(p[AN_R] - p[HP_R])) / 2.0
    return inc, float(leg / n)


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


def load(npz_path, manifest_path):
    z = np.load(npz_path, allow_pickle=True)
    lm, lab, sp = z["landmarks"], z["labels"].astype(str), z["split"].astype(str)
    ids = None
    if manifest_path and os.path.exists(manifest_path):
        m = json.load(open(manifest_path))
        meta = m if isinstance(m, list) else (m.get("meta") or [])
        if len(meta) == len(lab):
            ids = [str(e.get("id") or e.get("title")) for e in meta]
    if ids is None:
        ids = [f"h:{hash(lm[i].tobytes())}" for i in range(len(lab))]
    return lm, lab, sp, ids


def main():
    print("=== inputs ===", flush=True)
    for d in sorted(glob.glob("/kaggle/input/*")):
        print(" ", d, flush=True)

    sources = [
        (find("photo_corpus.npz"), find("photo_corpus_manifest.json"), "v1"),
        (find("photo_corpus_v2.npz"), find("photo_corpus_v2_manifest.json"), "harvest"),
        (find("public_corpus.npz"), find("public_corpus_manifest.json"), "public"),
    ]
    LM, LAB, SP, seen = [], [], [], set()
    for npz, man, tag in sources:
        if not npz:
            print(f"  (missing: {tag})", flush=True)
            continue
        lm, lab, sp, ids = load(npz, man)
        kept = 0
        for i in range(len(lab)):
            if ids[i] in seen:
                continue
            seen.add(ids[i])
            LM.append(lm[i]); LAB.append(lab[i]); SP.append(sp[i])
            kept += 1
        print(f"  {tag}: {len(lab)} photos, {kept} new after dedup", flush=True)

    LAB = np.array(LAB); SP = np.array(SP)
    te = SP == "test"
    print(f"\nunion {len(LAB)} photos | train {int((~te).sum())} | test {int(te.sum())}", flush=True)

    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    ck = hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT, token=tok)
    en = hf_hub_download(repo_id=HF_REPO, filename=ENCODER, token=tok)
    classes = list(np.load(en, allow_pickle=True))
    model = Yoga3HeadMLP(15, len(classes))
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.eval()
    print(f"loaded {CHECKPOINT} ({len(classes)} classes)\n", flush=True)

    idx_te = np.where(te)[0]
    X = np.array([angles_from_landmarks(LM[i]) for i in idx_te], dtype=np.float32)
    y = LAB[te]
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(X))
    pred = np.array([classes[i] for i in logits.argmax(1).numpy()])

    train_n = collections.Counter(LAB[~te].tolist())
    conf = collections.defaultdict(collections.Counter)
    for t, p in zip(y.tolist(), pred.tolist()):
        if t != p:
            conf[t][p] += 1

    rows = []
    for c in sorted(set(y.tolist())):
        m = y == c
        n = int(m.sum())
        rec = float((pred[m] == c).mean())
        pm = pred == c
        prec = float((y[pm] == c).mean()) if pm.sum() else 0.0
        top = conf[c].most_common(1)
        lost = top[0][1] / n if top else 0.0
        rows.append({"pose": c, "n_test": n, "n_train": train_n.get(c, 0),
                     "recall": rec, "precision": prec,
                     "top_confusion": top[0][0] if top else "-",
                     "top_confusion_frac": round(lost, 3)})

    rows.sort(key=lambda r: -r["recall"])
    print(f"{'pose':<24}{'n_te':>6}{'n_tr':>7}{'recall':>9}{'prec':>8}   confused with")
    for r in rows:
        print(f"  {r['pose']:<22}{r['n_test']:>6}{r['n_train']:>7}"
              f"{r['recall']*100:>8.1f}%{r['precision']*100:>7.1f}%   "
              f"{r['top_confusion']} ({r['top_confusion_frac']*100:.0f}%)")

    macro = float(np.mean([r["recall"] for r in rows]))
    print(f"\nmacro over {len(rows)} classes: {macro*100:.1f}%")

    # The decision this audit exists to inform.
    print("\n=== reading ===")
    strong = [r for r in rows if r["recall"] >= 0.60 and r["n_test"] >= 20]
    collide = [r for r in rows if r["recall"] < 0.60 and r["top_confusion_frac"] >= 0.30]
    scarce = [r for r in rows if r["n_train"] < 100]
    print(f"STRONG (recall >=60%, n_test >=20): {[r['pose'] for r in strong]}")
    print(f"  -> macro over just these: "
          f"{np.mean([r['recall'] for r in strong])*100:.1f}%" if strong else "")
    print(f"\nCOLLIDING (loses >=30% of its mass to ONE other pose -- a")
    print(f"representational limit, more data will not fix):")
    for r in collide:
        print(f"  {r['pose']} -> {r['top_confusion']} ({r['top_confusion_frac']*100:.0f}%)")
    print(f"\nSCARCE (<100 train photos -- low score may be scarcity, not difficulty):")
    print(f"  {[r['pose'] for r in scarce]}")

    # orientation stats, to explain collisions in the write-up
    print("\n=== orientation per pose (p10/p50/p90) ===")
    print(f"{'pose':<24}{'torso_incline':>22}{'leg/torso':>20}")
    for c in sorted(set(y.tolist())):
        sel = [orientation(LM[i]) for i in idx_te[y == c]]
        sel = [s for s in sel if not (math.isnan(s[0]) or math.isnan(s[1]))]
        if not sel:
            continue
        inc = np.array([s[0] for s in sel]); rat = np.array([s[1] for s in sel])
        print(f"  {c:<22}"
              f"{np.percentile(inc,10):>7.0f}{np.percentile(inc,50):>7.0f}{np.percentile(inc,90):>7.0f}"
              f"{np.percentile(rat,10):>8.2f}{np.percentile(rat,50):>6.2f}{np.percentile(rat,90):>6.2f}")

    json.dump({"rows": rows, "macro": macro}, open(f"{OUT}/per_class_audit.json", "w"), indent=1)


main()
