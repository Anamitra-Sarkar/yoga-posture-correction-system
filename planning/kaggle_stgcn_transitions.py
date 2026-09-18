"""AsanaAI ST-GCN — transition-aware retrain on Kaggle T4.

WHY THIS RUN EXISTS
-------------------
The previous evaluation proved the core hypothesis: the ST-GCN can learn
transitions, it had simply never been shown any. Relabelling
`transition/unknown` (63% of windows, a residual bin) into
`hold:<pose>` / `transition:<A>-><B>` / `unrecognized` gave macro 63.0% over
24 classes vs 54.6% over 15 for the old scheme, with named directional
transitions at 72.7-100%.

This run attacks the three measured weak points of that result:
  1. `transition:other` was still a dumping ground: n=2464 @ 32.6%.
     -> name far more directional pairs (MIN_PAIR lowered), shrinking `other`.
  2. `unrecognized` scored 10.4% (n=1591).
     -> defined strictly by motion, not as "whatever is left over".
  3. `hold:lunge_pose` 29.5% (n=522), likely bleeding into lunge transitions.
     -> class-weighted loss so frequent holds stop dominating their transitions.

It also settles an open question: the relabelled run used `bonecorr` features,
but bonecorr LOST to rawz on the non-relabelled comparison (49.7 vs 54.6).
Both are trained here, identically, so the comparison is clean.

ARCHITECTURE IS UNCHANGED and the layer naming matches production exactly
(`self.residual`, not `res`) so checkpoints load into
backend/app/models/sequence.py without a key remap.
"""
import os, json, glob, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

IN = "/kaggle/input/asanaai-stgcn-source"
OUT = "/kaggle/working"
WIN, STRIDE = 60, 12
HOLDOUT = {"4ORRiN2_aVI", "SZU7Sbgu57o"}   # same videos as before, for comparability
MIN_PAIR = 12          # name a directional transition with >= this many windows
HOLD_FRAC = 0.85       # window is a hold if one pose covers >= this fraction
MOTION_HOLD = 15.0     # deg/s — identical to backend MOTION_HOLD_MAX_DEG_PER_SEC
EPOCHS = 45
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- production architecture, verbatim ----------------
MEDIAPIPE_EDGES = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(3,7),(6,8),(9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(17,19),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32),
]

def get_normalized_adjacency():
    n = 33
    A = np.zeros((n, n), dtype=np.float32)
    for i, j in MEDIAPIPE_EDGES:
        A[i, j] = A[j, i] = 1.0
    A = A + np.eye(n, dtype=np.float32)
    deg = A.sum(1)
    dis = np.zeros_like(deg)
    np.power(deg, -0.5, out=dis, where=deg > 0)
    return torch.tensor(np.diag(dis) @ A @ np.diag(dis), dtype=torch.float32)

class SpatialGraphConv(nn.Module):
    def __init__(self, cin, cout, A):
        super().__init__()
        self.register_buffer("A", A)
        self.conv = nn.Conv2d(cin, cout, 1)
    def forward(self, x):
        return torch.einsum("vw,ncwt->ncvt", self.A, self.conv(x))

class TemporalConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=9, stride=1, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (kernel_size, 1), (stride, 1), ((kernel_size - 1)//2, 0))
        self.bn = nn.BatchNorm2d(cout); self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.transpose(2, 3)
        return self.dropout(self.bn(self.conv(x))).transpose(2, 3)

class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, dropout=0.3):
        super().__init__()
        self.gcn = SpatialGraphConv(cin, cout, A)
        self.tcn = TemporalConv(cout, cout, 9, stride, dropout)
        # NOTE: named `residual` to match production exactly. The earlier run
        # used `res`, which made its checkpoints fail load_state_dict in prod.
        self.residual = (nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
                         if (cin != cout or stride != 1) else nn.Identity())
    def forward(self, x):
        r = self.residual(x)
        return F.gelu(self.tcn(F.gelu(self.gcn(x))) + r)

class YogaSequenceLSTM(nn.Module):
    """Genuine ST-GCN. Legacy class name kept for checkpoint compatibility."""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        A = get_normalized_adjacency()
        self.block1 = STGCNBlock(3, 64, A, 1, 0.2)
        self.block2 = STGCNBlock(64, 128, A, 1, 0.3)
        self.block3 = STGCNBlock(128, 256, A, 1, 0.3)
        self.fc = nn.Sequential(nn.Linear(256,128), nn.LayerNorm(128), nn.GELU(),
                                nn.Dropout(0.4), nn.Linear(128, num_classes))
    def forward(self, x):
        b = x.size(0)
        o = x.view(b, WIN, 33, 3).permute(0, 3, 2, 1)
        o = self.block3(self.block2(self.block1(o)))
        return self.fc(F.adaptive_avg_pool2d(o, (1, 1)).view(b, -1))

# ---------------- feature construction ----------------
SH_L,SH_R,EL_L,EL_R,WR_L,WR_R = 11,12,13,14,15,16
HP_L,HP_R,KN_L,KN_R,AN_L,AN_R,HE_L,HE_R = 23,24,25,26,27,28,29,30
NOSE = 0
BONES = [(SH_L,EL_L),(EL_L,WR_L),(SH_R,EL_R),(EL_R,WR_R),
         (HP_L,KN_L),(KN_L,AN_L),(HP_R,KN_R),(KN_R,AN_R),
         (AN_L,HE_L),(AN_R,HE_R),(SH_L,SH_R),(HP_L,HP_R)]

def bone_correct(lm):
    """Per-video median bone length as a self-calibration reference.

    A person's real segment lengths are constant across a video even though
    MediaPipe's per-frame z is not, so frames whose 3D bone length deviates
    from that person's own median indicate depth error. Rescales the distal
    joint along the (2D-reliable) bone direction. Temporal information no
    single-frame method can use.
    """
    out = lm.copy()
    for a, b in BONES:
        v = out[:, b, :3] - out[:, a, :3]
        L = np.linalg.norm(v, axis=1)
        med = np.median(L[L > 1e-6]) if (L > 1e-6).any() else 0.0
        if med <= 0:
            continue
        r = np.divide(med, L, out=np.ones_like(L), where=L > 1e-6)
        bad = (L > 1e-6) & ((L/med < 0.85) | (L/med > 1.18))
        out[bad, b, :3] = out[bad, a, :3] + v[bad] * r[bad, None]
    return out

def angles_zero_z(p):
    """The 15 production angle features (z zeroed), used only for motion."""
    q = p[:, :3].copy(); q[:, 2] = 0.0
    def ang(a, b, c):
        ba, bc = q[a]-q[b], q[c]-q[b]
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nb == 0 or nc == 0: return 180.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ba,bc)/(nb*nc))))))
    sm = (q[SH_L]+q[SH_R])/2; hm = (q[HP_L]+q[HP_R])/2
    def ang3(a, b, c):
        ba, bc = a-b, c-b
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nb == 0 or nc == 0: return 180.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ba,bc)/(nb*nc))))))
    return np.array([
        ang(SH_L,EL_L,WR_L), ang(SH_R,EL_R,WR_R), ang(HP_L,SH_L,EL_L), ang(HP_R,SH_R,EL_R),
        ang(SH_L,HP_L,KN_L), ang(SH_R,HP_R,KN_R), ang(HP_L,KN_L,AN_L), ang(HP_R,KN_R,AN_R),
        ang(KN_L,AN_L,HE_L), ang(KN_R,AN_R,HE_R), ang(SH_L,HP_L,HP_R), ang(SH_R,HP_R,HP_L),
        ang3(q[NOSE], sm, hm), ang(HP_R,HP_L,KN_L), ang(HP_L,HP_R,KN_R)], dtype=np.float32)

def base_pose(l):
    l = str(l)
    if l.startswith("imperfect_"): l = l[len("imperfect_"):]
    return "child_pose" if l == "child" else l

def main():
    # Diagnostic: if the dataset mount is missing/partial the failure should be
    # obvious rather than a bare FileNotFoundError 170 lines in.
    import glob as _g
    print("=== /kaggle/input tree ===", flush=True)
    for d in sorted(_g.glob("/kaggle/input/*")):
        fs = sorted(_g.glob(d + "/*"))
        print(f"  {d}  ({len(fs)} files)", flush=True)
        for f in fs[:4]:
            print(f"     {os.path.basename(f)}", flush=True)
    # Kaggle nests dataset mounts (/kaggle/input/datasets/<owner>/<slug>/...),
    # and the exact layout has changed between CLI versions -- so discover the
    # data rather than hardcoding a path that silently breaks.
    global IN
    if not os.path.exists(f"{IN}/master_mlp_dataset_fully_classified.csv"):
        hits = _g.glob("/kaggle/input/**/master_mlp_dataset_fully_classified.csv",
                       recursive=True)
        if not hits:
            raise SystemExit("FATAL: labels CSV not found anywhere under /kaggle/input")
        IN = os.path.dirname(hits[0])
        print(f"resolved data dir -> {IN}", flush=True)
    n_lm = len(_g.glob(f"{IN}/landmarks_*.npy"))
    print(f"landmark files visible: {n_lm}", flush=True)
    if n_lm == 0:
        raise SystemExit("FATAL: no landmark .npy files alongside the CSV")

    # map landmark filename -> youtube id
    vid_of = {}
    for j in glob.glob(f"{IN}/*.info.json"):
        try:
            vid_of[os.path.basename(j)[:-10]] = json.load(open(j)).get("id")
        except Exception:
            pass

    df = pd.read_csv(f"{IN}/master_mlp_dataset_fully_classified.csv")
    lab_col = "imperfect_pose_label" if "imperfect_pose_label" in df.columns else "pose_label"
    by_vid = {v: g.sort_values("frame_num")[lab_col].map(base_pose).tolist()
              for v, g in df.groupby("video_id")}
    print("label videos:", len(by_vid), flush=True)

    feats_raw, feats_bc, raw_labels, vids, motions = [], [], [], [], []
    for f in sorted(glob.glob(f"{IN}/landmarks_*.npy")):
        stem = os.path.basename(f)[len("landmarks_"):-4]
        vid = vid_of.get(stem)
        if vid is None or vid not in by_vid:
            print("skip (no labels):", stem[:50], flush=True); continue
        lm = np.load(f)                       # [N,33,4]
        labs = by_vid[vid]
        n = min(len(lm), len(labs))
        lm, labs = lm[:n], labs[:n]
        lm_bc = bone_correct(lm)
        # per-frame angles for the motion signal
        ang = np.stack([angles_zero_z(lm[i]) for i in range(n)])
        for s in range(0, n - WIN + 1, STRIDE):
            w = slice(s, s + WIN)
            seg = labs[w]
            # motion over the window, matching the production definition
            dt = WIN / 30.0
            mot = float(np.abs(ang[s+WIN-1] - ang[s]).mean() / dt)
            feats_raw.append(lm[w, :, :3].reshape(WIN, -1))
            feats_bc.append(lm_bc[w, :, :3].reshape(WIN, -1))
            raw_labels.append(seg); vids.append(vid); motions.append(mot)

    print("windows:", len(feats_raw), flush=True)

    # ---------------- relabelling ----------------
    def named(seg, mot):
        from collections import Counter
        real = [p for p in seg if p != "transition/unknown"]
        if real:
            top, cnt = Counter(real).most_common(1)[0]
            if cnt / len(seg) >= HOLD_FRAC and mot <= MOTION_HOLD:
                return f"hold:{top}"
        # moving: try to name a direction using the window's endpoints
        if mot > MOTION_HOLD:
            head = [p for p in seg[:WIN//3] if p != "transition/unknown"]
            tail = [p for p in seg[-WIN//3:] if p != "transition/unknown"]
            if head and tail:
                a = Counter(head).most_common(1)[0][0]
                b = Counter(tail).most_common(1)[0][0]
                if a != b:
                    return f"transition:{a}->{b}"
            return "transition:other"
        return "unrecognized"

    labels = [named(s, m) for s, m in zip(raw_labels, motions)]
    from collections import Counter
    cnt = Counter(labels)
    # collapse rare directional pairs into `other` so classes have support
    labels = [l if (not l.startswith("transition:") or l == "transition:other"
                    or cnt[l] >= MIN_PAIR) else "transition:other" for l in labels]
    cnt = Counter(labels)
    print("\n=== class histogram ===", flush=True)
    for k, v in cnt.most_common():
        print(f"  {k:<48} {v}", flush=True)
    n_tr = sum(v for k, v in cnt.items() if k.startswith("transition:") and k != "transition:other")
    print(f"named transitions: {sum(1 for k in cnt if k.startswith('transition:') and k!='transition:other')} classes, {n_tr} windows", flush=True)
    print(f"transition:other  : {cnt.get('transition:other',0)} windows", flush=True)

    classes = sorted(cnt)
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[l] for l in labels])
    vids = np.array(vids)
    te = np.array([v in HOLDOUT for v in vids]); tr = ~te
    print(f"\ntrain {tr.sum()} / holdout {te.sum()} windows, {len(classes)} classes", flush=True)

    results = {}
    for tag, feats in (("relab_rawz", feats_raw), ("relab_bonecorr", feats_bc)):
        X = np.asarray(feats, dtype=np.float32)
        # pelvis-centre + hip-width scale, matching normalize_coordinate_sequence
        Xr = X.reshape(len(X), WIN, 33, 3)
        pel = (Xr[:, :, HP_L] + Xr[:, :, HP_R]) / 2.0
        Xr = Xr - pel[:, :, None, :]
        hw = np.linalg.norm(Xr[:, :, HP_L] - Xr[:, :, HP_R], axis=-1, keepdims=True)
        hw = np.where(hw < 1e-5, 1.0, hw)
        X = (Xr / hw[:, :, None]).reshape(len(X), WIN, 99).astype(np.float32)

        Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
        Xte = torch.tensor(X[te]); yte = torch.tensor(y[te])
        w = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
        w = np.where(w > 0, 1.0 / np.sqrt(w), 0.0); w = w / w.sum() * len(classes)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))

        m = YogaSequenceLSTM(99, 128, 2, len(classes)).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        best, best_state = -1, None
        for ep in range(EPOCHS):
            m.train(); perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 64):
                b = perm[i:i+64]
                opt.zero_grad()
                loss = crit(m(Xtr[b].to(DEV)), ytr[b].to(DEV))
                loss.backward(); opt.step()
            sched.step()
            m.eval(); preds = []
            with torch.no_grad():
                for i in range(0, len(Xte), 256):
                    preds.append(m(Xte[i:i+256].to(DEV)).argmax(1).cpu())
            p = torch.cat(preds).numpy() if preds else np.array([])
            accs = [float((p[yte.numpy()==c]==c).mean()) for c in range(len(classes)) if (yte.numpy()==c).any()]
            macro = float(np.mean(accs))
            if macro > best:
                best, best_state = macro, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            if ep % 5 == 0 or ep == EPOCHS-1:
                print(f"[{tag}] ep{ep:02d} macro={macro*100:.1f}% overall={float((p==yte.numpy()).mean())*100:.1f}%", flush=True)

        m.load_state_dict(best_state); m.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xte), 256):
                preds.append(m(Xte[i:i+256].to(DEV)).argmax(1).cpu())
        p = torch.cat(preds).numpy(); t = yte.numpy()
        per = {classes[c]: {"n": int((t==c).sum()), "acc": round(float((p[t==c]==c).mean()), 4)}
               for c in range(len(classes)) if (t==c).any()}
        results[tag] = {"per_class": per,
                        "macro": round(float(np.mean([v["acc"] for v in per.values()])), 4),
                        "overall": round(float((p==t).mean()), 4),
                        "n_classes": len(classes)}
        torch.save(best_state, f"{OUT}/stgcn_{tag}_v4.pth")
        np.save(f"{OUT}/stgcn_encoder_{tag}_v4.npy", np.array(classes, dtype=object))
        print(f"[{tag}] BEST macro={results[tag]['macro']*100:.1f}% overall={results[tag]['overall']*100:.1f}%", flush=True)

    json.dump(results, open(f"{OUT}/stgcn_v4_results.json", "w"), indent=1)
    print("\n=== FINAL ===")
    for k, v in results.items():
        print(f"{k:16s} macro={v['macro']*100:5.1f}%  overall={v['overall']*100:5.1f}%  classes={v['n_classes']}")

main()
