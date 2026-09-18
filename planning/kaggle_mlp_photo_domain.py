"""AsanaAI MLP — does training on REAL PHOTOS close the domain gap?

THE QUESTION
------------
The MLP is a single-frame classifier trained on frames from 12 vinyasa videos
but judged on photographs. That mismatch is the measured failure: ~94% on
held-in video frames vs 52.6% on real photos, and ~26% macro across the
broader vocabulary.

Landmark augmentation was already tried to bridge it and did NOT work
(24.2% vs 25.8% macro on real photos). Synthetic jitter around a narrow corpus
cannot invent real photographic diversity.

So the experiment is simple and decisive: train on the kind of data we are
judged on, and measure on a held-out split of exactly that data.

VARIANTS (identical architecture and schedule, only the training data differs)
  A video_only    - frames from the 12 videos. Reproduces current behaviour.
  B video+photos  - frames plus the 319-photo training split.
  C photos_only   - the 319-photo training split alone.
  D video+photos_oversampled - as B, but photos repeated so they are not
                    drowned by ~650k video rows (319 photos vs 650k frames is
                    a 2000:1 imbalance; without this, B is barely different
                    from A and the experiment would answer nothing).

ALL are scored on the SAME held-out 103-photo test split, which no variant
trains on. That split is a deterministic hash of the Commons image title, so
it is reproducible and cannot leak.

Architecture is production's Yoga3HeadMLP verbatim -- this is a data
experiment, not an architecture change.
"""
import os, json, glob, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = "/kaggle/working"
EPOCHS = 25
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

# ---------- the 15 production angle features ----------
NOSE=0; SH_L,SH_R,EL_L,EL_R,WR_L,WR_R = 11,12,13,14,15,16
HP_L,HP_R,KN_L,KN_R,AN_L,AN_R,HE_L,HE_R = 23,24,25,26,27,28,29,30

def angles_from_landmarks(pts):
    """Identical to backend geometry.extract_angles_from_landmarks(zero_z=True).

    z is zeroed because MediaPipe's monocular depth is only reliable at the
    fixed framing of the training video; the production inference path always
    zeroes it, and training must match inference or we recreate exactly the
    mismatch bug that cost this project its accuracy once already.
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
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return hits[0] if hits else None


def per_class(yt, yp):
    out, accs = {}, []
    for c in sorted(set(yt)):
        idx = [i for i, t in enumerate(yt) if t == c]
        a = sum(1 for i in idx if yp[i] == yt[i]) / len(idx)
        out[c] = {"n": len(idx), "acc": round(a, 4)}; accs.append(a)
    return {"per_class": out, "macro": round(float(np.mean(accs)), 4),
            "overall": round(sum(1 for a, b in zip(yt, yp) if a == b)/len(yt), 4)}


def main():
    print("=== inputs ===", flush=True)
    for d in sorted(glob.glob("/kaggle/input/*")):
        print(" ", d, flush=True)

    npz_p = find("photo_corpus.npz")
    csv_p = find("master_mlp_dataset_fully_classified.csv")
    print("photo corpus:", npz_p, "\nvideo csv   :", csv_p, flush=True)
    if not npz_p:
        raise SystemExit("FATAL: photo_corpus.npz not found")

    z = np.load(npz_p, allow_pickle=True)
    lm, plab, psplit = z["landmarks"], z["labels"].astype(str), z["split"].astype(str)
    pang = np.array([angles_from_landmarks(x) for x in lm], dtype=np.float32)
    tr_m, te_m = psplit == "train", psplit == "test"
    print(f"photos: {len(plab)}  train={tr_m.sum()} test={te_m.sum()}", flush=True)

    Xp_tr, yp_tr = pang[tr_m], plab[tr_m]
    Xp_te, yp_te = pang[te_m], plab[te_m]

    Xv = yv = None
    if csv_p:
        df = pd.read_csv(csv_p)
        col = "imperfect_pose_label" if "imperfect_pose_label" in df.columns else "pose_label"
        Xv = df[FEATS].values.astype(np.float32)
        yv = np.array([base_pose(v) for v in df[col].tolist()])
        print(f"video frames: {len(yv)}", flush=True)

    # a variant may only predict classes it has seen; score against the union
    classes = sorted(set(plab.tolist()) | (set(yv.tolist()) if yv is not None else set()))
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"classes: {len(classes)}", flush=True)

    def build(tag):
        if tag == "video_only":
            return Xv, yv
        if tag == "photos_only":
            return Xp_tr, yp_tr
        if tag == "video+photos":
            return np.vstack([Xv, Xp_tr]), np.concatenate([yv, yp_tr])
        if tag == "video+photos_oversampled":
            # 319 photos against ~650k frames is 2000:1; without oversampling the
            # photos contribute almost no gradient and the variant would be
            # indistinguishable from video_only, answering nothing.
            reps = max(1, int(len(yv) * 0.25 / max(1, len(yp_tr))))
            print(f"    oversampling photos x{reps}", flush=True)
            return (np.vstack([Xv] + [Xp_tr]*reps),
                    np.concatenate([yv] + [yp_tr]*reps))
        raise ValueError(tag)

    variants = ["photos_only"] if Xv is None else [
        "video_only", "video+photos", "video+photos_oversampled", "photos_only"]

    results = {}
    Xte = torch.tensor(Xp_te)
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
                pl, _, _ = m(Xte.to(DEV))
            pred = [classes[i] for i in pl.argmax(1).cpu().numpy()]
            r = per_class(list(yp_te), pred)
            if r["macro"] > best:
                best, best_state = r["macro"], {k: v.detach().cpu().clone()
                                                for k, v in m.state_dict().items()}
            if ep % 5 == 0 or ep == EPOCHS-1:
                print(f"  ep{ep:02d} macro={r['macro']*100:5.1f}% overall={r['overall']*100:5.1f}%", flush=True)

        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            pl, _, _ = m(Xte.to(DEV))
        pred = [classes[i] for i in pl.argmax(1).cpu().numpy()]
        results[tag] = per_class(list(yp_te), pred)
        results[tag]["n_train"] = int(len(ytr_i))
        torch.save(best_state, f"{OUT}/mlp_{tag.replace('+','_')}.pth")
        np.save(f"{OUT}/mlp_encoder.npy", np.array(classes, dtype=object))
        print(f"  BEST macro={results[tag]['macro']*100:.1f}% overall={results[tag]['overall']*100:.1f}%", flush=True)

    json.dump(results, open(f"{OUT}/mlp_domain_results.json", "w"), indent=1)
    print("\n================ FINAL (held-out real photos) ================")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["macro"]):
        print(f"{k:<28} macro={v['macro']*100:5.1f}%  overall={v['overall']*100:5.1f}%  (train n={v['n_train']})")

main()
