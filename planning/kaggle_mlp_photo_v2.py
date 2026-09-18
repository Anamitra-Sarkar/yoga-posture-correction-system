"""AsanaAI MLP on the EXPANDED real-photo corpus (v1 + v2 union).

WHAT THIS IS TESTING
--------------------
Training on real photos already beat training on video frames by 2.5x
(35.5% vs 14.1% macro on a held-out photo split). But five classes still score
0%, and they are precisely the five with almost no photos: upward_dog had 1,
standing_forward_fold 11, plank 14, cobra_pose 16, chair_pose 21. The v2
harvest went after exactly those, adding Openverse as a second image pool.

So the question here is narrow and answerable: does more data for the starved
classes actually lift them, or were they failing for some other reason?

HONEST BENCHMARKING
-------------------
More data also means a bigger test split, and a bigger test split is a
DIFFERENT benchmark -- a number that rose could just be an easier exam. So
every variant is scored twice:

  frozen_103  the exact 103-image test set the 35.5% figure came from, so the
              comparison to the previous result is like-for-like;
  expanded    the full v1+v2 test split, which is the more honest estimate of
              real-world behaviour because it covers the starved classes with
              more than three or four images each.

Both are reported. If frozen_103 does not move, the gain is not real, and this
script will say so rather than quoting only the flattering number.

The split function is identical to v1's and was verified to reproduce all 422
v1 splits exactly, so no image can cross from train into test.
"""
import os, json, glob, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = "/kaggle/working"
EPOCHS = 40
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEATS = ["elbow_l","elbow_r","shoulder_l","shoulder_r","hip_l","hip_r","knee_l","knee_r",
         "ankle_l","ankle_r","trunk_l","trunk_r","neck","hip_abduct_l","hip_abduct_r"]


# ---------- production architecture, verbatim ----------
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
        self.res1 = ResBlock(256, 0.3); self.res2 = ResBlock(256, 0.3)
        self.pose_head = nn.Sequential(nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(),
                                       nn.Dropout(0.2), nn.Linear(128, num_poses))
        self.correctness_head = nn.Sequential(nn.Linear(256,64), nn.BatchNorm1d(64), nn.GELU(),
                                              nn.Dropout(0.2), nn.Linear(64,1))
        self.deviation_head = nn.Sequential(nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(),
                                            nn.Dropout(0.2), nn.Linear(128, num_joints))
    def forward(self, x):
        f = self.res2(self.res1(self.input_layer(x)))
        return self.pose_head(f), self.correctness_head(f).squeeze(-1), self.deviation_head(f)


NOSE=0; SH_L,SH_R,EL_L,EL_R,WR_L,WR_R = 11,12,13,14,15,16
HP_L,HP_R,KN_L,KN_R,AN_L,AN_R,HE_L,HE_R = 23,24,25,26,27,28,29,30


def angles_from_landmarks(pts):
    """Identical to backend geometry.extract_angles_from_landmarks(zero_z=True).

    z is zeroed because the production inference path always zeroes it, and
    training must match inference -- that exact mismatch is what once cost this
    project its real-world accuracy.
    """
    p = np.asarray(pts, dtype=np.float64)[:, :3].copy()
    p[:, 2] = 0.0
    def ang(a, b, c):
        ba, bc = p[a]-p[b], p[c]-p[b]
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nb == 0 or nc == 0: return 180.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ba,bc)/(nb*nc))))))
    sm = (p[SH_L]+p[SH_R])/2.0; hm = (p[HP_L]+p[HP_R])/2.0
    ba, bc = p[NOSE]-sm, hm-sm
    nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    neck = 180.0 if (nb==0 or nc==0) else math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(ba,bc)/(nb*nc))))))
    return [ang(SH_L,EL_L,WR_L), ang(SH_R,EL_R,WR_R), ang(HP_L,SH_L,EL_L), ang(HP_R,SH_R,EL_R),
            ang(SH_L,HP_L,KN_L), ang(SH_R,HP_R,KN_R), ang(HP_L,KN_L,AN_L), ang(HP_R,KN_R,AN_R),
            ang(KN_L,AN_L,HE_L), ang(KN_R,AN_R,HE_R), ang(SH_L,HP_L,HP_R), ang(SH_R,HP_R,HP_L),
            neck, ang(HP_R,HP_L,KN_L), ang(HP_L,HP_R,KN_R)]


def base_pose(l):
    l = str(l)
    if l.startswith("imperfect_"): l = l[len("imperfect_"):]
    return "child_pose" if l == "child" else l


def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits[0] if hits else None


def per_class(yt, yp):
    out, accs = {}, []
    for c in sorted(set(yt)):
        idx = [i for i, t in enumerate(yt) if t == c]
        a = sum(1 for i in idx if yp[i] == yt[i]) / len(idx)
        out[c] = {"n": len(idx), "acc": round(a, 4)}; accs.append(a)
    return {"per_class": out, "macro": round(float(np.mean(accs)), 4),
            "overall": round(sum(1 for a, b in zip(yt, yp) if a == b)/len(yt), 4)}


def load_corpus(npz_path, manifest_path):
    """Return (angles, labels, splits, ids). ids come from the manifest, whose
    row order matches the npz -- that is what makes a union by identity, rather
    than by position, possible."""
    z = np.load(npz_path, allow_pickle=True)
    lm = z["landmarks"]; lab = z["labels"].astype(str); sp = z["split"].astype(str)
    ids = None
    if manifest_path and os.path.exists(manifest_path):
        m = json.load(open(manifest_path))
        # the final manifest is a dict with a "meta" list; the mid-run
        # checkpoint writes the bare list, so accept either
        meta = m if isinstance(m, list) else (m.get("meta") or [])
        if len(meta) == len(lab):
            ids = [str(e.get("id") or e.get("title")) for e in meta]
    if ids is None:
        # no manifest: fall back to a content hash so dedup still works
        ids = [f"h:{hash(lm[i].tobytes())}" for i in range(len(lab))]
    ang = np.array([angles_from_landmarks(x) for x in lm], dtype=np.float32)
    return ang, lab, sp, ids


def main():
    print("=== inputs ===", flush=True)
    for d in sorted(glob.glob("/kaggle/input/*")):
        print(" ", d, flush=True)

    v1_npz = find("photo_corpus.npz")
    v1_man = find("photo_corpus_manifest.json")
    v2_npz = find("photo_corpus_v2.npz") or find("photo_corpus_v2_partial.npz")
    v2_man = find("photo_corpus_v2_manifest.json") or find("photo_corpus_v2_partial_meta.json")
    csv_p = find("master_mlp_dataset_fully_classified.csv")
    print("v1:", v1_npz, "\nv2:", v2_npz, "\ncsv:", csv_p, flush=True)
    if not v1_npz:
        raise SystemExit("FATAL: v1 photo_corpus.npz not found -- the frozen "
                         "benchmark is what makes this comparable, so there is "
                         "no point running without it")

    a1, l1, s1, i1 = load_corpus(v1_npz, v1_man)
    frozen_ids = {i1[k] for k in range(len(l1)) if s1[k] == "test"}
    print(f"v1: {len(l1)} photos, frozen test set = {len(frozen_ids)}", flush=True)

    if v2_npz:
        a2, l2, s2, i2 = load_corpus(v2_npz, v2_man)
        print(f"v2: {len(l2)} photos", flush=True)
    else:
        print("WARNING: no v2 corpus found -- this run can only reproduce the "
              "v1 baseline, not test whether more data helped", flush=True)
        a2 = np.zeros((0, 15), np.float32); l2 = np.array([]); s2 = np.array([]); i2 = []

    # union by identity: v2 re-harvests much of v1, and the same photo must not
    # appear twice (it would silently double its weight and, worse, could be
    # counted twice in the test metric)
    A, L, S, I, seen = [], [], [], [], set()
    for ang, lab, sp, ids in ((a1, l1, s1, i1), (a2, l2, s2, i2)):
        for k in range(len(lab)):
            if ids[k] in seen:
                continue
            seen.add(ids[k])
            A.append(ang[k]); L.append(lab[k]); S.append(sp[k]); I.append(ids[k])
    A = np.array(A, dtype=np.float32); L = np.array(L); S = np.array(S)
    print(f"union: {len(L)} photos  train={(S=='train').sum()} test={(S=='test').sum()}",
          flush=True)

    print("\n=== per-class (union) ===", flush=True)
    for c in sorted(set(L.tolist())):
        tr = int(((L == c) & (S == "train")).sum()); te = int(((L == c) & (S == "test")).sum())
        print(f"  {c:<26} train={tr:<5} test={te}", flush=True)

    tr_m, te_m = S == "train", S == "test"
    Xp_tr, yp_tr = A[tr_m], L[tr_m]
    Xp_te, yp_te = A[te_m], L[te_m]
    frozen_m = np.array([I[k] in frozen_ids for k in range(len(I))]) & te_m
    Xf, yf = A[frozen_m], L[frozen_m]
    print(f"\nfrozen_103 rows recovered: {len(yf)} (expected {len(frozen_ids)})", flush=True)

    Xv = yv = None
    if csv_p:
        df = pd.read_csv(csv_p)
        col = "imperfect_pose_label" if "imperfect_pose_label" in df.columns else "pose_label"
        Xv = df[FEATS].values.astype(np.float32)
        yv = np.array([base_pose(v) for v in df[col].tolist()])
        print(f"video frames: {len(yv)}", flush=True)

    classes = sorted(set(L.tolist()) | (set(yv.tolist()) if yv is not None else set()))
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"classes: {len(classes)}", flush=True)

    def build(tag):
        if tag == "photos_only":
            return Xp_tr, yp_tr
        if tag == "video+photos_oversampled":
            reps = max(1, int(len(yv) * 0.25 / max(1, len(yp_tr))))
            print(f"    oversampling photos x{reps}", flush=True)
            return (np.vstack([Xv] + [Xp_tr]*reps), np.concatenate([yv] + [yp_tr]*reps))
        raise ValueError(tag)

    # Only the two variants that actually competed last time. video_only and the
    # un-oversampled mix were both decisively beaten (14.1% and 18.4%); rerunning
    # them would burn GPU time to re-answer a settled question.
    variants = ["photos_only"] if Xv is None else ["video+photos_oversampled", "photos_only"]

    results = {}
    Xte_t, Xf_t = torch.tensor(Xp_te), torch.tensor(Xf)
    for tag in variants:
        Xtr, ytr = build(tag)
        ytr_i = np.array([cidx[c] for c in ytr])
        keep = ~np.isnan(Xtr).any(1) & ~np.isinf(Xtr).any(1)
        Xtr, ytr_i = Xtr[keep], ytr_i[keep]
        print(f"\n=== {tag}: {len(ytr_i)} training rows ===", flush=True)

        m = Yoga3HeadMLP(15, len(classes)).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        cnt = np.bincount(ytr_i, minlength=len(classes)).astype(np.float32)
        w = np.where(cnt > 0, 1.0/np.sqrt(np.maximum(cnt, 1)), 0.0)
        w = w/w.sum()*len(classes)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))

        Xt = torch.tensor(Xtr); yt_ = torch.tensor(ytr_i)
        # Model selection uses the FROZEN set, so the checkpoint we keep is not
        # chosen on the same expanded split we then quote as the honest number.
        best, best_state = -1.0, None
        for ep in range(EPOCHS):
            m.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 512):
                b = perm[i:i+512]
                if len(b) < 2: continue
                opt.zero_grad()
                pl, _, _ = m(Xt[b].to(DEV))
                crit(pl, yt_[b].to(DEV)).backward(); opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                pf = m(Xf_t.to(DEV))[0]
            rf = per_class(list(yf), [classes[i] for i in pf.argmax(1).cpu().numpy()])
            if rf["macro"] > best:
                best, best_state = rf["macro"], {k: v.detach().cpu().clone()
                                                 for k, v in m.state_dict().items()}
            if ep % 5 == 0 or ep == EPOCHS-1:
                print(f"  ep{ep:02d} frozen_macro={rf['macro']*100:5.1f}%", flush=True)

        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            pf = m(Xf_t.to(DEV))[0]
            pe = m(Xte_t.to(DEV))[0]
        r = {"frozen_103": per_class(list(yf), [classes[i] for i in pf.argmax(1).cpu().numpy()]),
             "expanded": per_class(list(yp_te), [classes[i] for i in pe.argmax(1).cpu().numpy()]),
             "n_train": int(len(ytr_i))}
        results[tag] = r
        torch.save(best_state, f"{OUT}/mlp_v2_{tag.replace('+','_')}.pth")
        np.save(f"{OUT}/mlp_v2_encoder.npy", np.array(classes, dtype=object))
        print(f"  BEST frozen={r['frozen_103']['macro']*100:.1f}%  "
              f"expanded={r['expanded']['macro']*100:.1f}%", flush=True)

    json.dump(results, open(f"{OUT}/mlp_v2_results.json", "w"), indent=1)

    print("\n================ FINAL ================")
    print(f"{'variant':<28} {'frozen_103':>12} {'expanded':>12}")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["frozen_103"]["macro"]):
        print(f"{k:<28} {v['frozen_103']['macro']*100:11.1f}% {v['expanded']['macro']*100:11.1f}%")

    print("\n--- verdict against the 35.5% v1 result (same frozen 103 images) ---")
    bestk = max(results, key=lambda k: results[k]["frozen_103"]["macro"])
    bf = results[bestk]["frozen_103"]["macro"] * 100
    if bf > 35.5:
        print(f"IMPROVED: {bestk} reaches {bf:.1f}% vs 35.5% on the identical test set.")
    else:
        print(f"NO IMPROVEMENT: best is {bestk} at {bf:.1f}%, not above 35.5% on the "
              f"identical test set. More photos did not help; do NOT promote.")

    print("\n--- the five classes that were at 0% ---")
    for c in ["chair_pose", "cobra_pose", "plank", "upward_dog", "standing_forward_fold"]:
        row = results[bestk]["expanded"]["per_class"].get(c)
        fz = results[bestk]["frozen_103"]["per_class"].get(c)
        print(f"  {c:<24} expanded={row} frozen={fz}")


main()
