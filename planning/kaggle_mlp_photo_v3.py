"""AsanaAI MLP v3 -- all THREE heads trained, on the expanded photo corpus.

WHY THIS EXISTS (a regression I introduced, stated plainly)
-----------------------------------------------------------
The photo-domain experiments (kaggle_mlp_photo_domain.py and
kaggle_mlp_photo_v2.py) train ONLY the pose head:

    pl, _, _ = m(Xt[b]); crit(pl, yt_[b]).backward()

`correctness_head` and `deviation_head` receive no loss at all, so they stay
at random initialisation. That was fine while those scripts were pure
data-domain EXPERIMENTS measuring pose accuracy. It stopped being fine the
moment one of their checkpoints was promoted to production, which is what
happened on 2026-09-20 with mlp_3head_photodomain_v1.

The backend consumes all three heads. In `hybrid_classify`, whenever the MLP
and the 2D rule engine AGREE on the pose -- the confident, common case --
the app uses `mlp_correctness` and `mlp_devs` directly. So pose accuracy went
up 3x while the per-joint coaching numbers underneath it became random.
The original `train_mlp_3head_gpu.py` had always trained all three
(`loss = loss_pose + 1.0 * loss_correct + 1.0 * loss_dev`); the photo scripts
silently dropped two of them.

WHAT THIS FIXES
---------------
Restores the 3-head loss while keeping the photo-domain data recipe that
actually works. Both target types are derivable for photos too, which is the
key point -- no new labelling needed:

  * deviations come from the biomechanical RULES bands given the angles and
    the pose label, both of which photos have;
  * correctness comes from the `imperfect_` label prefix. Photos have no
    imperfect variants, so the correctness head would collapse to
    always-correct if trained on photos alone -- which is exactly why the
    correctness loss is MASKED to rows that carry a real correct/incorrect
    distinction (the video rows). Pose and deviation losses still see
    everything.

WHICH RECIPE
------------
Measured on the identical frozen 103-photo set, pose-head-only:
  photos_only               47.5% macro   (448 training rows)
  video+photos_oversampled  38.9% macro   (818,008 training rows)
  [previously promoted]     35.5% macro
So photos win decisively for POSE -- 448 real photos beat 654,488 video
frames. But the correctness head needs video's imperfect examples to be
anything but degenerate. Rather than guess the trade-off, this trains several
photo-weighting ratios with the full 3-head loss and reports pose macro,
correctness accuracy AND deviation MAE for each, so the choice is made on
measurements instead of argument.
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

# Copied verbatim from train_mlp_3head_gpu.py so the deviation targets this
# model learns are the SAME bands the rule engine and the live scoring path
# use. If these drift apart, the model's deviations and the app's rule-based
# deviations start meaning different things.
RULES = {
    'mountain_pose': {
        'hip_l': (140, 180), 'hip_r': (140, 180),
        'knee_l': (140, 180), 'knee_r': (140, 180),
        'shoulder_l': (0, 55), 'shoulder_r': (0, 55),
        'trunk_l': (65, 180), 'trunk_r': (65, 180)
    },
    'upward_salute': {
        'hip_l': (140, 180), 'hip_r': (140, 180),
        'knee_l': (140, 180), 'knee_r': (140, 180),
        'shoulder_l': (115, 180), 'shoulder_r': (115, 180),
        'trunk_l': (65, 180), 'trunk_r': (65, 180)
    },
    'downward_dog': {
        'hip_l': (45, 130), 'hip_r': (45, 130),
        'knee_l': (110, 180), 'knee_r': (110, 180),
        'shoulder_l': (95, 180), 'shoulder_r': (95, 180)
    },
    'cobra_pose': {
        'hip_l': (120, 180), 'hip_r': (120, 180),
        'knee_l': (120, 180), 'knee_r': (120, 180),
        'shoulder_l': (45, 135), 'shoulder_r': (45, 135),
        'neck': (80, 180)
    },
    'child_pose': {
        'hip_l': (0, 90), 'hip_r': (0, 90),
        'knee_l': (0, 90), 'knee_r': (0, 90),
        'shoulder_l': (85, 180), 'shoulder_r': (85, 180)
    },
    'seated_staff': {
        'hip_l': (60, 120), 'hip_r': (60, 120),
        'knee_l': (135, 180), 'knee_r': (135, 180),
        'trunk_l': (60, 180), 'trunk_r': (60, 180)
    },
    'seated_easy_pose': {
        'hip_l': (50, 120), 'hip_r': (50, 120),
        'knee_l': (0, 125), 'knee_r': (0, 125),
        'trunk_l': (60, 180), 'trunk_r': (60, 180)
    },
    'tree_pose': {
        'knee_l': (140, 180), 'hip_l': (140, 180),
        'knee_r': (0, 120), 'hip_r': (70, 135),
        'hip_abduct_r': (115, 180),
        'knee_r': (140, 180), 'hip_r': (140, 180),
        'knee_l': (0, 120), 'hip_l': (70, 135),
        'hip_abduct_l': (115, 180)
    },
    'warrior_1': {
        'knee_l': (0, 120), 'knee_r': (130, 180),
        'shoulder_l': (110, 180), 'shoulder_r': (110, 180)
    },
    'warrior_2': {
        'knee_l': (0, 120), 'knee_r': (130, 180),
        'shoulder_l': (65, 125), 'shoulder_r': (65, 125)
    },
    'lunge_pose': {
        'knee_l': (0, 120), 'knee_r': (130, 180)
    },
    'standing_forward_fold': {
        'hip_l': (0, 70), 'hip_r': (0, 70),
        'knee_l': (120, 180), 'knee_r': (120, 180)
    },
    'halfway_lift': {
        'hip_l': (70, 115), 'hip_r': (70, 115),
        'knee_l': (130, 180), 'knee_r': (130, 180)
    },
    'table_top': {
        'hip_l': (60, 125), 'hip_r': (60, 125),
        'knee_l': (60, 125), 'knee_r': (60, 125),
        'shoulder_l': (60, 125), 'shoulder_r': (60, 125)
    },
    'chair_pose': {
        'hip_l': (75, 140), 'hip_r': (75, 140),
        'knee_l': (75, 140), 'knee_r': (75, 140),
        'shoulder_l': (95, 180), 'shoulder_r': (95, 180)
    }
}

def calculate_deviations(angles_row, base_pose):
    """Verbatim port of train_mlp_3head_gpu.calculate_deviations.

    Degrees outside the pose's biomechanical band, per joint, 0 when inside.
    Needs only angles + pose label -- which is why photos can supervise this
    head despite carrying no hand-authored deviation ground truth.
    """
    devs = np.zeros(len(FEATS), dtype=np.float32)
    if base_pose in ("transition/unknown", "corpse"):
        return devs
    pose_rules = RULES.get(base_pose, {})
    for idx, col in enumerate(FEATS):
        if col in pose_rules:
            low, high = pose_rules[col]
            val = float(angles_row[idx])
            if val < low:
                devs[idx] = low - val
            elif val > high:
                devs[idx] = val - high
    return devs


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
    """Matches geometry.extract_angles_from_landmarks(zero_z=True) exactly."""
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
    z = np.load(npz_path, allow_pickle=True)
    lm = z["landmarks"]; lab = z["labels"].astype(str); sp = z["split"].astype(str)
    ids = None
    if manifest_path and os.path.exists(manifest_path):
        m = json.load(open(manifest_path))
        meta = m if isinstance(m, list) else (m.get("meta") or [])
        if len(meta) == len(lab):
            ids = [str(e.get("id") or e.get("title")) for e in meta]
    if ids is None:
        ids = [f"h:{hash(lm[i].tobytes())}" for i in range(len(lab))]
    ang = np.array([angles_from_landmarks(x) for x in lm], dtype=np.float32)
    return ang, lab, sp, ids


def main():
    print("=== inputs ===", flush=True)
    for d in sorted(glob.glob("/kaggle/input/*")):
        print(" ", d, flush=True)

    v1_npz = find("photo_corpus.npz"); v1_man = find("photo_corpus_manifest.json")
    v2_npz = find("photo_corpus_v2.npz") or find("photo_corpus_v2_partial.npz")
    v2_man = find("photo_corpus_v2_manifest.json") or find("photo_corpus_v2_partial_meta.json")
    csv_p = find("master_mlp_dataset_fully_classified.csv")
    if not v1_npz:
        raise SystemExit("FATAL: v1 photo_corpus.npz not found -- it defines the frozen benchmark")
    if not csv_p:
        raise SystemExit("FATAL: video CSV not found -- the correctness head needs its "
                         "imperfect_ examples; without them that head cannot be trained")

    a1, l1, s1, i1 = load_corpus(v1_npz, v1_man)
    frozen_ids = {i1[k] for k in range(len(l1)) if s1[k] == "test"}
    print(f"v1: {len(l1)} photos, frozen test = {len(frozen_ids)}", flush=True)
    if v2_npz:
        a2, l2, s2, i2 = load_corpus(v2_npz, v2_man)
        print(f"v2: {len(l2)} photos", flush=True)
    else:
        a2 = np.zeros((0,15), np.float32); l2 = np.array([]); s2 = np.array([]); i2 = []

    A, L, S, I, seen = [], [], [], [], set()
    for ang, lab, sp, ids in ((a1,l1,s1,i1), (a2,l2,s2,i2)):
        for k in range(len(lab)):
            if ids[k] in seen: continue
            seen.add(ids[k])
            A.append(ang[k]); L.append(lab[k]); S.append(sp[k]); I.append(ids[k])
    A = np.array(A, dtype=np.float32); L = np.array(L); S = np.array(S)
    print(f"union: {len(L)} photos  train={(S=='train').sum()} test={(S=='test').sum()}", flush=True)

    tr_m, te_m = S == "train", S == "test"
    Xp_tr, yp_tr = A[tr_m], L[tr_m]
    Xp_te, yp_te = A[te_m], L[te_m]
    frozen_m = np.array([I[k] in frozen_ids for k in range(len(I))]) & te_m
    Xf, yf = A[frozen_m], L[frozen_m]
    print(f"frozen rows recovered: {len(yf)} / {len(frozen_ids)}", flush=True)

    df = pd.read_csv(csv_p)
    col = "imperfect_pose_label" if "imperfect_pose_label" in df.columns else "pose_label"
    vid = df["video_id"].astype(str).values if "video_id" in df.columns else None
    Xv = df[FEATS].values.astype(np.float32)
    raw_v = df[col].astype(str).values
    yv = np.array([base_pose(v) for v in raw_v])
    # correctness: 0 for a deliberately-broken row, 1 otherwise -- the same
    # derivation the original 3-head trainer used
    cv = np.array([0.0 if (r.startswith("imperfect_") or r == "transition/unknown") else 1.0
                   for r in raw_v], dtype=np.float32)
    print(f"video frames: {len(yv)}  ({int((cv==0).sum())} imperfect)", flush=True)

    # IN-DOMAIN VALIDATION MUST BE GROUPED BY SOURCE VIDEO.
    # The 2026-07-19 run reported 90.86% val pose accuracy from a random
    # train_test_split over frames. There are only 12 source videos and frames
    # are densely sampled, so consecutive frames of a held pose are
    # near-duplicates: a random split puts almost the same image on both sides
    # and the number measures memorisation. That is how 90.9% validation
    # coexisted with ~10.5% on real photographs. Holding out whole videos is
    # what makes the in-domain number mean anything.
    vid_val_mask = None
    if vid is not None:
        uniq = sorted(set(vid.tolist()))
        rs = np.random.RandomState(0)
        holdout = set(rs.choice(uniq, max(1, len(uniq) // 4), replace=False).tolist())
        vid_val_mask = np.array([v in holdout for v in vid])
        print(f"grouped split: {len(holdout)}/{len(uniq)} videos held out "
              f"({int(vid_val_mask.sum())} frames)", flush=True)
    else:
        print("WARNING: no video_id column; the in-domain split cannot be "
              "grouped and its accuracy will be leak-inflated", flush=True)

    classes = sorted(set(L.tolist()) | set(yv.tolist()))
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"classes: {len(classes)}", flush=True)

    def devs_for(X, y):
        return np.stack([calculate_deviations(X[i], y[i]) for i in range(len(y))]).astype(np.float32) / 180.0

    print("deriving deviation targets...", flush=True)
    Dv = devs_for(Xv, yv)
    Dp = devs_for(Xp_tr, yp_tr)
    # photos carry no imperfect/correct distinction, so their correctness loss
    # is masked off rather than being invented as always-1 (which would teach
    # the head to answer "correct" unconditionally)
    cp = np.ones(len(yp_tr), dtype=np.float32)
    mv = np.ones(len(yv), dtype=np.float32)
    mp = np.zeros(len(yp_tr), dtype=np.float32)

    VARIANTS = {
        "photos_x1":   1,
        "photos_x365": None,   # matches the old 25%-of-video ratio
        "photos_x2000": 2000,  # photo-dominant, approximating photos_only for
                               # the pose head while video still supervises
                               # correctness
    }

    results = {}
    Xte_t, Xf_t = torch.tensor(Xp_te), torch.tensor(Xf)

    for tag, reps in VARIANTS.items():
        if reps is None:
            reps = max(1, int(len(yv) * 0.25 / max(1, len(yp_tr))))
        tr_v = ~vid_val_mask if vid_val_mask is not None else np.ones(len(yv), bool)
        Xtr = np.vstack([Xv[tr_v]] + [Xp_tr]*reps)
        ytr = np.concatenate([yv[tr_v]] + [yp_tr]*reps)
        Dtr = np.vstack([Dv[tr_v]] + [Dp]*reps)
        Ctr = np.concatenate([cv[tr_v]] + [cp]*reps)
        Mtr = np.concatenate([mv[tr_v]] + [mp]*reps)
        ytr_i = np.array([cidx[c] for c in ytr])
        keep = ~np.isnan(Xtr).any(1) & ~np.isinf(Xtr).any(1)
        Xtr, ytr_i, Dtr, Ctr, Mtr = Xtr[keep], ytr_i[keep], Dtr[keep], Ctr[keep], Mtr[keep]
        print(f"\n=== {tag}: reps={reps}, {len(ytr_i)} rows "
              f"({int(Mtr.sum())} with correctness labels) ===", flush=True)

        m = Yoga3HeadMLP(15, len(classes)).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        cnt = np.bincount(ytr_i, minlength=len(classes)).astype(np.float32)
        w = np.where(cnt > 0, 1.0/np.sqrt(np.maximum(cnt, 1)), 0.0); w = w/w.sum()*len(classes)
        pose_crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))
        corr_crit = nn.BCEWithLogitsLoss(reduction="none")
        dev_crit = nn.SmoothL1Loss()

        Xt = torch.tensor(Xtr); yt_ = torch.tensor(ytr_i)
        Dt = torch.tensor(Dtr); Ct = torch.tensor(Ctr); Mt = torch.tensor(Mtr)
        best, best_state = -1.0, None
        for ep in range(EPOCHS):
            m.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), 512):
                b = perm[i:i+512]
                if len(b) < 2: continue
                opt.zero_grad()
                pl, cl, dp = m(Xt[b].to(DEV))
                loss_pose = pose_crit(pl, yt_[b].to(DEV))
                msk = Mt[b].to(DEV)
                raw_c = corr_crit(cl, Ct[b].to(DEV))
                loss_corr = (raw_c * msk).sum() / msk.sum().clamp(min=1.0)
                loss_dev = dev_crit(dp, Dt[b].to(DEV))
                (loss_pose + 1.0*loss_corr + 1.0*loss_dev).backward()
                opt.step()
            sched.step()
            m.eval()
            with torch.no_grad():
                pf, _, _ = m(Xf_t.to(DEV))
            rf = per_class(list(yf), [classes[i] for i in pf.argmax(1).cpu().numpy()])
            if rf["macro"] > best:
                best, best_state = rf["macro"], {k: v.detach().cpu().clone()
                                                 for k, v in m.state_dict().items()}
            if ep % 10 == 0 or ep == EPOCHS-1:
                print(f"  ep{ep:02d} frozen_macro={rf['macro']*100:5.1f}%", flush=True)

        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            pf, cf, df_ = m(Xf_t.to(DEV))
            pe, _, _ = m(Xte_t.to(DEV))
            # correctness/deviation sanity on a VIDEO holdout, since that is
            # where real correct/incorrect labels exist
            pool = np.where(vid_val_mask)[0] if vid_val_mask is not None \
                else np.arange(len(Xv))
            n = min(20000, len(pool))
            sel = np.random.RandomState(0).choice(pool, n, replace=False)
            pv, cvl, dvp = m(torch.tensor(Xv[sel]).to(DEV))
        corr_pred = (torch.sigmoid(cvl).cpu().numpy() > 0.5).astype(np.float32)
        corr_acc = float((corr_pred == cv[sel]).mean())
        dev_mae = float(np.abs(dvp.cpu().numpy() - Dv[sel]).mean() * 180.0)

        r = {"frozen_103": per_class(list(yf), [classes[i] for i in pf.argmax(1).cpu().numpy()]),
             "expanded": per_class(list(yp_te), [classes[i] for i in pe.argmax(1).cpu().numpy()]),
             "correctness_acc_heldout_video": round(corr_acc, 4),
             "deviation_mae_deg_heldout_video": round(dev_mae, 3),
             "pose_acc_heldout_video": round(float(
                 (pv.argmax(1).cpu().numpy() ==
                  np.array([cidx[c] for c in yv[sel]])).mean()), 4),
             "n_train": int(len(ytr_i))}
        results[tag] = r
        torch.save(best_state, f"{OUT}/mlp_v3_{tag}.pth")
        np.save(f"{OUT}/mlp_v3_encoder.npy", np.array(classes, dtype=object))
        print(f"  BEST frozen={r['frozen_103']['macro']*100:.1f}%  "
              f"heldout_vid_pose={r['pose_acc_heldout_video']*100:.1f}%  "
              f"expanded={r['expanded']['macro']*100:.1f}%  "
              f"corr_acc={corr_acc*100:.1f}%  dev_MAE={dev_mae:.2f}deg", flush=True)

    json.dump(results, open(f"{OUT}/mlp_v3_results.json","w"), indent=1)
    print("\n================ FINAL (all three heads trained) ================")
    print("in-domain numbers below are on HELD-OUT VIDEOS, so they are not")
    print("comparable to the 90.86% the 2026-07-19 run reported from a random")
    print("frame split -- that one had near-duplicate frames on both sides.\n")
    print(f"{'variant':<16}{'photo macro':>13}{'heldvid pose':>14}"
          f"{'corr acc':>10}{'dev MAE':>9}")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["frozen_103"]["macro"]):
        print(f"{k:<16}{v['frozen_103']['macro']*100:12.1f}%"
              f"{v['pose_acc_heldout_video']*100:13.1f}%"
              f"{v['correctness_acc_heldout_video']*100:9.1f}%"
              f"{v['deviation_mae_deg_heldout_video']:8.2f}d")

    bestk = max(results, key=lambda k: results[k]["frozen_103"]["macro"])
    bf = results[bestk]["frozen_103"]["macro"]*100
    print(f"\nbest pose variant: {bestk} at {bf:.1f}%")
    print("bar to beat = 35.5% (mlp_3head_photodomain_v1, currently live, but "
          "with UNTRAINED correctness/deviation heads)")
    if bf > 35.5:
        print(f"PROMOTE {bestk}: better pose AND real correctness/deviation heads.")
    else:
        print("Pose did not clear the bar. Note the live checkpoint's "
              "correctness/deviation heads are random, so a small pose "
              "regression may still be worth trading for working heads -- "
              "decide explicitly, do not auto-promote.")


main()
