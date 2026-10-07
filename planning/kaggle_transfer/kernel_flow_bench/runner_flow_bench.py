"""FLOW benchmark for the ST-GCN (the temporal model).  Held-out VIDEOS (3 folds by video), ONE recipe for every model, the LAST epoch is scored.  Token-free (public HF dataset Arko007/Yoga-1M); results are PRINTED.

Why this exists: the earlier sequence comparison scored the ST-GCN on HELD-POSE recognition (name the pose in a still window) -- that is the MLP's job.  The ST-GCN's job is FLOW: how the body MOVES between poses.
The cue-verified labels cannot name transitions (kernel asanaai-flow-stats: no directional pose pair A->B is held in >= 3 videos), so two complementary tests are used:

TASK 1  ARROW OF TIME (label-free, no rule engine anywhere).  Every moving window of every video is shown forward (label 0) and time-reversed (label 1).  50 % is chance.
        A model that ignores frame ORDER (window mean/std MLP) is exactly 50 % by construction, so this isolates what temporal modelling buys.
TASK 2  FLOW-24 (the label scheme of the original stgcn_transitions_v1): hold:<pose> / transition:<A>-><B> / transition:other / unrecognized.  Frame pose = the cue-verified label where it exists, the
        rule engine's pose from the 15 angles otherwise; a window is a TRANSITION when its net angle change exceeds 15 deg/s, named by the dominant pose of its first and last third.
        CAVEAT (stated in the paper): these labels are rule-defined, so the task measures whether a learner recovers named flow from the raw skeleton WITHOUT the rules, not agreement with a human annotator.
        A directional-confusion rate (A->B predicted as B->A) separates models that see order from models that do not.
Models: LSTM, BiLSTM+attention, temporal CNN, window mean/std MLP, ST-GCN with identity adjacency (no skeleton graph), ST-GCN (production architecture)."""
import collections, glob, importlib.util, json, math, os, sys, time

EPOCHS = int(os.environ.get("FLOW_EPOCHS", "12")); BATCH = 128; SEED = 0
AOT_CAP = int(os.environ.get("AOT_CAP", "12000"))      # training windows per fold for the arrow-of-time task (each shown forward AND reversed)
F24_CAP = int(os.environ.get("F24_CAP", "3000"))        # training windows per class for the flow-24 task
FPS = 25.0; MOTION_HOLD = 15.0; HOLD_FRAC = 0.85; MIN_PAIR = 90; MIN_SUPPORT = 100; MIN_VIDEOS = 3; AOT_MIN_DEG = 15.0; IGN_MAX = 0.15
U, IGN = "transition/unknown", "__ignore__"
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
HP_L, HP_R, KN_L, KN_R, AN_L, AN_R, HE_L, HE_R, NOSE = 23, 24, 25, 26, 27, 28, 29, 30, 0


# ------------------------------------------------------------------ labels
def angles15(P):
    """The 15 production angle features (z zeroed), vectorised over frames: P [N,33,3] -> [N,15]."""
    import numpy as np
    q = P.astype(np.float32).copy(); q[:, :, 2] = 0.0
    def _a(ba, bc):
        nb = np.linalg.norm(ba, axis=1); nc = np.linalg.norm(bc, axis=1); d = nb * nc
        out = np.degrees(np.arccos(np.clip((ba * bc).sum(1) / np.where(d == 0, 1.0, d), -1, 1))); out[d == 0] = 180.0; return out
    def ang(a, b, c): return _a(q[:, a] - q[:, b], q[:, c] - q[:, b])
    sm = (q[:, SH_L] + q[:, SH_R]) / 2; hm = (q[:, HP_L] + q[:, HP_R]) / 2
    return np.stack([ang(SH_L, EL_L, WR_L), ang(SH_R, EL_R, WR_R), ang(HP_L, SH_L, EL_L), ang(HP_R, SH_R, EL_R), ang(SH_L, HP_L, KN_L), ang(SH_R, HP_R, KN_R),
                     ang(HP_L, KN_L, AN_L), ang(HP_R, KN_R, AN_R), ang(KN_L, AN_L, HE_L), ang(KN_R, AN_R, HE_R), ang(SH_L, HP_L, HP_R), ang(SH_R, HP_R, HP_L),
                     _a(q[:, NOSE] - sm, hm - sm), ang(HP_R, HP_L, KN_L), ang(HP_L, HP_R, KN_R)], 1).astype(np.float32)


def rule_pose(a):
    """The backend rule ordering (same as planning/kaggle_stgcn_transitions.py)."""
    hl, hr, kl, kr, sl, sr, tl, tr, nk = a[4], a[5], a[6], a[7], a[2], a[3], a[10], a[11], a[12]
    B = lambda v, lo, hi: lo <= v <= hi
    if hl > 140 and hr > 140 and kl > 140 and kr > 140 and sl < 55 and sr < 55 and tl > 65 and tr > 65: return "mountain_pose"
    if hl > 140 and hr > 140 and kl > 140 and kr > 140 and sl > 115 and sr > 115 and tl > 65 and tr > 65: return "upward_salute"
    if B(hl, 20, 140) and B(hr, 20, 140) and kl > 110 and kr > 110 and sl > 95 and sr > 95: return "downward_dog"
    if hl > 140 and hr > 140 and kl > 140 and kr > 140 and B(sl, 60, 110) and B(sr, 60, 110): return "plank"
    if hl > 120 and hr > 120 and kl > 120 and kr > 120 and B(sl, 5, 50) and B(sr, 5, 50) and nk >= 80: return "cobra_pose"
    if hl < 90 and hr < 90 and kl < 90 and kr < 90 and sl > 85 and sr > 85: return "child_pose"
    if B(hl, 60, 120) and B(hr, 60, 120) and kl > 135 and kr > 135 and tl >= 60 and tr >= 60: return "seated_staff"
    if B(hl, 50, 120) and B(hr, 50, 120) and kl < 125 and kr < 125 and tl >= 60 and tr >= 60: return "seated_easy_pose"
    if B(hl, 75, 140) and B(hr, 75, 140) and B(kl, 75, 140) and B(kr, 75, 140) and abs(kl - kr) < 30 and sl > 95 and sr > 95: return "chair_pose"
    if (kl > 150 and hl > 165 and kr < 140) or (kr > 150 and hr > 165 and kl < 140): return "tree_pose"
    legs = (kl < 120 and kr > 130) or (kr < 120 and kl > 130)
    if legs and B(sl, 65, 125) and B(sr, 65, 125): return "warrior_2"
    if legs and sl > 110 and sr > 110: return "warrior_1"
    if legs: return "lunge_pose"
    if hl < 70 and hr < 70 and kl > 120 and kr > 120: return "standing_forward_fold"
    if B(hl, 70, 115) and B(hr, 70, 115) and kl > 130 and kr > 130: return "halfway_lift"
    if B(hl, 60, 125) and B(hr, 60, 125) and B(kl, 60, 125) and B(kr, 60, 125) and B(sl, 60, 125) and B(sr, 60, 125): return "table_top"
    if hl > 140 and hr > 140 and kl > 140 and kr > 140: return "standing_pose"
    return U


def named(seg, mot_rate):
    """hold / transition:A->B / transition:other / unrecognized -- the original stgcn_transitions_v1 scheme."""
    real = [p for p in seg if p != U]
    if mot_rate > MOTION_HOLD:
        head = [p for p in seg[:len(seg) // 3] if p != U]; tail = [p for p in seg[-(len(seg) // 3):] if p != U]
        if head and tail:
            a = collections.Counter(head).most_common(1)[0][0]; b = collections.Counter(tail).most_common(1)[0][0]
            if a != b: return f"transition:{a}->{b}"
        return "transition:other"
    if real:
        top, c = collections.Counter(real).most_common(1)[0]
        if c / len(seg) >= HOLD_FRAC: return f"hold:{top}"
        if c / len(real) >= 0.60 and len(real) >= len(seg) * 0.4: return f"hold:{top}"
    return "unrecognized"


# ------------------------------------------------------------------ metrics
def flow_metrics(P, T, min_n=20):
    import numpy as np
    cls = sorted(set(T.tolist())); rows = {}
    for c in cls:
        t, pm = T == c, P == c; n = int(t.sum())
        rows[c] = {"n": n, "recall": round(float((P[t] == c).mean()), 3), "precision": round(float((T[pm] == c).mean()) if pm.sum() else 0.0, 3)}
    ok = [c for c in cls if rows[c]["n"] >= min_n]
    grp = lambda f: [rows[c]["recall"] for c in ok if f(c)]
    mean = lambda v: round(float(np.mean(v)), 3) if v else None
    trans = lambda c: c.startswith("transition:") and c != "transition:other"
    dirs = []   # share of A->B windows predicted as exactly the reverse B->A
    for c in ok:
        if trans(c):
            a, b = c[len("transition:"):].split("->"); r = f"transition:{b}->{a}"
            if r in cls: dirs.append(float((P[T == c] == r).mean()))
    return {"overall": round(float((P == T).mean()), 3), "macro_all": mean(grp(lambda c: True)), "macro_holds": mean(grp(lambda c: c.startswith("hold:"))),
            "macro_named_transitions": mean(grp(trans)), "n_named_transition_classes": len(grp(trans)), "direction_confusion": mean(dirs), "n_reversible_pairs": len(dirs), "per_class": rows}


# ------------------------------------------------------------------ models
def make_models(tr):
    import torch, torch.nn as nn
    class LSTMNet(nn.Module):
        def __init__(s, c): super().__init__(); s.rnn = nn.LSTM(99, 128, num_layers=2, batch_first=True, dropout=0.3); s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, c))
        def forward(s, x): return s.fc(s.rnn(x)[0][:, -1])
    class BiLSTMAttn(nn.Module):
        def __init__(s, c):
            super().__init__(); s.rnn = nn.LSTM(99, 128, num_layers=1, batch_first=True, bidirectional=True)
            s.att = nn.MultiheadAttention(256, 4, batch_first=True, dropout=0.1); s.ln = nn.LayerNorm(256); s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, c))
        def forward(s, x): h = s.rnn(x)[0]; a = s.att(h, h, h, need_weights=False)[0]; return s.fc(s.ln(h + a).mean(1))
    class TCN(nn.Module):
        def __init__(s, c):
            super().__init__()
            def blk(i, o): return nn.Sequential(nn.Conv1d(i, o, 9, padding=4), nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(0.2))
            s.net = nn.Sequential(blk(99, 128), blk(128, 256), blk(256, 256)); s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, c))
        def forward(s, x): return s.fc(s.net(x.transpose(1, 2)).mean(2))
    class StatMLP(nn.Module):   # order-blind: mean and std of every coordinate over the window
        def __init__(s, c): super().__init__(); s.net = nn.Sequential(nn.Linear(198, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Linear(128, c))
        def forward(s, x): return s.net(torch.cat([x.mean(1), x.std(1)], 1))
    class To4D(nn.Module):
        def __init__(s, inner): super().__init__(); s.inner = inner
        def forward(s, x): return s.inner(x.reshape(x.shape[0], 60, 33, 3).permute(0, 3, 2, 1).contiguous())
    def stgcn(c): return To4D(tr.YogaSequenceLSTM(99, 128, 2, c))
    def stgcn_noedges(c):
        m = tr.YogaSequenceLSTM(99, 128, 2, c)
        for mod in m.modules():
            if hasattr(mod, "A") and getattr(mod, "A", None) is not None and mod.A.shape == (33, 33): mod.A.copy_(torch.eye(33))
        return To4D(m)
    return {"MLP on window mean/std (order-blind)": StatMLP, "LSTM (2 layers)": LSTMNet, "BiLSTM + multi-head attention": BiLSTMAttn, "Temporal CNN (no skeleton graph)": TCN,
            "ST-GCN (identity adjacency = no skeleton graph)": stgcn_noedges, "ST-GCN (production architecture)": stgcn}


def train_seq(make, Xtr, ytr, ncls, epochs, dev, seed=SEED):
    import numpy as np, torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); t0 = time.time(); m = make(ncls).to(dev)
    w = np.bincount(ytr, minlength=ncls).astype(np.float32); w = np.where(w > 0, 1.0 / np.sqrt(w), 0.0); w = w / w.sum() * ncls
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev)); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    amp = dev == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp); X = torch.as_tensor(Xtr); Y = torch.as_tensor(ytr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X)); tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2: continue
            xb, yb = X[b].to(dev), Y[b].to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): loss = crit(m(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += float(loss) * len(b)
        sched.step()
        if ep % 3 == 0 or ep == epochs - 1: print(f"      ep{ep:02d} loss {tot / len(X):.4f} ({time.time() - t0:.0f} s)", flush=True)
    return m.eval()


def predict_seq(m, X, dev):
    import numpy as np, torch
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 256): out.append(m(torch.as_tensor(X[b:b + 256]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    T0 = time.time(); BUDGET = 7.0 * 3600
    import numpy as np, pandas as pd, torch
    from huggingface_hub import snapshot_download
    ROOT = "/kaggle/working"; DS = "Arko007/Yoga-1M"
    if not torch.cuda.is_available(): raise SystemExit("NO CUDA DEVICE: refusing to run the flow benchmark on CPU")
    try: _x = torch.randn(256, 256, device="cuda"); float((_x @ _x).sum()); print("GPU OK:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    except Exception as _e: raise SystemExit(f"GPU PRESENT BUT UNUSABLE: {_e}")
    DEV = "cuda"
    snapshot_download(DS, repo_type="dataset", local_dir=ROOT, allow_patterns=["code/originals/**", "vol/csv/cue/cue_full.csv", "vol/landmarks/**", "vol/landmarks_new/**"])
    sys.path.insert(0, f"{ROOT}/code/originals")
    spec = importlib.util.spec_from_file_location("stgcn_trainer", f"{ROOT}/code/originals/train_stgcn_gpu.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
    import generate_sequence_features as gsf
    SEQ, STRIDE = gsf.SEQ_LENGTH, gsf.STRIDE; assert SEQ == 60, SEQ
    OLD_IDS = "HmZFwoUU3WQ SZU7Sbgu57o oUgpXY7QhpQ 7ciS93shMNQ P8uHMMmWMHQ 4ORRiN2_aVI RQMtwbhXD7A 8ibxmzJziHU Eml2xnoLpYE QiebZSlTw_U L-z1HLkS_-Y s-1vMbAgYWU".split()
    RES = {"setup": {"epochs": EPOCHS, "batch": BATCH, "seed": SEED, "aot_cap_windows_per_fold": AOT_CAP, "flow24_cap_per_class": F24_CAP, "window": SEQ, "stride": STRIDE, "fps": FPS,
                     "motion_hold_deg_per_s": MOTION_HOLD, "aot_min_net_change_deg": AOT_MIN_DEG, "min_videos_per_class": MIN_VIDEOS, "note": "one recipe for all models; last epoch scored; single seed"}}
    def save(tag): json.dump(RES, open(f"{ROOT}/flow_bench.json", "w"), indent=1); print("saved", tag, flush=True)

    # ---------------------------------------------------------- windows
    df = pd.read_csv(f"{ROOT}/vol/csv/cue/cue_full.csv", usecols=["video_id", "frame_num", "imperfect_pose_label"])
    COORD, FRAME_POSE, FRAME_IGN, ANG = {}, {}, {}, {}
    for vid in df.video_id.unique():
        lp = f"{ROOT}/vol/landmarks/landmarks_{vid}.npy" if vid in OLD_IDS else f"{ROOT}/vol/landmarks_new/landmarks_{vid}.npy"
        lm = np.load(lp); lab = df[df.video_id == vid].sort_values("frame_num")["imperfect_pose_label"].astype(str).values; assert len(lab) == len(lm), (vid, len(lab), len(lm))
        COORD[vid] = lm[:, :, :3].reshape(len(lm), -1).astype(np.float32); ANG[vid] = angles15(lm[:, :, :3]); FRAME_IGN[vid] = lab == IGN
        need = (lab == U) | (lab == IGN); fp = lab.astype(object)
        for i in np.where(need)[0]: fp[i] = rule_pose(ANG[vid][i].tolist())
        FRAME_POSE[vid] = fp
    print("videos", len(COORD), "frames", sum(len(v) for v in COORD.values()), flush=True)
    W = []   # (video, start, flow-24 label, net angle change in deg)
    for vid in COORD:
        n = len(COORD[vid])
        for st in range(0, n - SEQ + 1, STRIDE):
            if FRAME_IGN[vid][st:st + SEQ].mean() > IGN_MAX: continue
            net = float(np.abs(ANG[vid][st + SEQ - 1] - ANG[vid][st]).mean()); rate = net / (SEQ / FPS)
            W.append((vid, st, named(list(FRAME_POSE[vid][st:st + SEQ]), rate), net))
    print("usable windows (<=15% ambiguous frames):", len(W), flush=True)

    # ---------------------------------------------------------- flow-24 labels (rare directional pairs -> other; classes need support in >= 3 videos)
    cnt = collections.Counter(w[2] for w in W)
    lab2 = [w[2] if (not w[2].startswith("transition:") or w[2] == "transition:other" or cnt[w[2]] >= MIN_PAIR) else "transition:other" for w in W]
    cnt2 = collections.Counter(lab2); vids_of = collections.defaultdict(set)
    for w, l in zip(W, lab2): vids_of[l].add(w[0])
    keep = {k for k, v in cnt2.items() if v >= MIN_SUPPORT and len(vids_of[k]) >= MIN_VIDEOS}
    print("\nflow-24 classes kept (windows, videos):", {k: (cnt2[k], len(vids_of[k])) for k in sorted(keep)}, flush=True)
    print("dropped (windows, videos):", {k: (cnt2[k], len(vids_of[k])) for k in sorted(cnt2) if k not in keep}, flush=True)
    RES["flow24_classes"] = {k: {"windows": cnt2[k], "videos": len(vids_of[k])} for k in sorted(keep)}
    RES["flow24_dropped"] = {k: {"windows": cnt2[k], "videos": len(vids_of[k])} for k in sorted(cnt2) if k not in keep}
    F24 = [(w[0], w[1], l) for w, l in zip(W, lab2) if l in keep]; classes = sorted(keep); cid = {c: i for i, c in enumerate(classes)}
    # folds by video, balanced on the flow-24 class load (same assignment is used for the arrow-of-time task)
    vids = sorted(COORD); load = [collections.Counter() for _ in range(3)]; FO = {}
    per_v = {v: collections.Counter(l for (vv, _, l) in F24 if vv == v) for v in vids}
    for v in sorted(vids, key=lambda v: -sum(per_v[v].values())):
        b = min(range(3), key=lambda f: sum((load[f][c] + n) ** 2 for c, n in per_v[v].items()) + 0.001 * sum(load[f].values())); FO[v] = b; load[b].update(per_v[v])
    RES["fold_of_video"] = FO; print("fold_of_video", FO, flush=True)
    mat = lambda items: np.asarray(tr.normalize_coordinate_sequence(np.stack([COORD[v][s:s + SEQ] for v, s in items]).astype(np.float32)), dtype=np.float32)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------- task data
    AOT = [(w[0], w[1]) for w in W if w[3] >= AOT_MIN_DEG]
    RES["aot_windows"] = {"n": len(AOT), "videos": len({a[0] for a in AOT}), "per_fold": [sum(1 for a in AOT if FO[a[0]] == f) for f in range(3)]}
    print("\narrow-of-time windows (net change >= %.0f deg):" % AOT_MIN_DEG, RES["aot_windows"], flush=True)
    AOT_FOLDS, F24_FOLDS = [], []
    for f in range(3):
        a_tr = [a for a in AOT if FO[a[0]] != f]; a_tr = [a_tr[i] for i in rng.permutation(len(a_tr))[:AOT_CAP]]; a_te = [a for a in AOT if FO[a[0]] == f]
        Xa, Xt = mat(a_tr), mat(a_te)
        AOT_FOLDS.append(dict(Xtr=np.concatenate([Xa, Xa[:, ::-1]]).copy(), ytr=np.r_[np.zeros(len(Xa), int), np.ones(len(Xa), int)], Xte=np.concatenate([Xt, Xt[:, ::-1]]).copy(), yte=np.r_[np.zeros(len(Xt), int), np.ones(len(Xt), int)]))
        tr_items = [(v, s, l) for v, s, l in F24 if FO[v] != f]; te_items = [(v, s, l) for v, s, l in F24 if FO[v] == f]
        sel = np.sort(np.concatenate([rng.permutation(np.where(np.array([x[2] for x in tr_items]) == c)[0])[:F24_CAP] for c in classes if any(x[2] == c for x in tr_items)]))
        F24_FOLDS.append(dict(Xtr=mat([(tr_items[i][0], tr_items[i][1]) for i in sel]), ytr=np.array([cid[tr_items[i][2]] for i in sel]), Xte=mat([(v, s) for v, s, _ in te_items]), yte=np.array([l for _, _, l in te_items])))
        print(f"fold {f}: AOT train {len(Xa)}x2 test {len(Xt)}x2 | flow-24 train {len(sel)} test {len(te_items)} (classes in test: {len(set(x[2] for x in te_items))}, in train: {len(set(tr_items[i][2] for i in sel))})", flush=True)
    assert AOT_FOLDS[0]["Xtr"].shape[1:] == (60, 99)
    MODELS = make_models(tr)
    RES["models"] = {k: int(sum(p.numel() for p in v(3).parameters())) for k, v in MODELS.items()}
    RES["arrow_of_time"] = {k: {"folds": {}} for k in MODELS}; RES["flow24"] = {k: {"folds": {}} for k in MODELS}
    for name, mk in MODELS.items():      # fast models first, the two ST-GCNs last; results saved after every model
        if time.time() - T0 > BUDGET: print("SKIPPED (time budget):", name, flush=True); RES["arrow_of_time"][name]["skipped"] = RES["flow24"][name]["skipped"] = "time budget"; continue
        # ---- task 1
        accs = []
        for f, D in enumerate(AOT_FOLDS):
            t0 = time.time(); m = train_seq(mk, D["Xtr"], D["ytr"], 2, EPOCHS, DEV); p = predict_seq(m, D["Xte"], DEV); a = float((p == D["yte"]).mean()); accs.append(a)
            RES["arrow_of_time"][name]["folds"][str(f)] = {"accuracy": round(a, 3), "n_test": int(len(p)), "seconds": round(time.time() - t0)}; print(f"   [arrow of time] fold {f} {name:<48} acc {a:.3f} ({round(time.time() - t0)} s)", flush=True); del m
        RES["arrow_of_time"][name]["mean_accuracy"] = round(float(np.mean(accs)), 3); print(f"== [arrow of time] {name:<48} mean acc {np.mean(accs):.3f}  folds {[round(a, 3) for a in accs]}", flush=True)
        # ---- task 2
        allp, allt = [], []
        for f, D in enumerate(F24_FOLDS):
            t0 = time.time(); m = train_seq(mk, D["Xtr"], D["ytr"], len(classes), EPOCHS, DEV); pred = [classes[j] for j in predict_seq(m, D["Xte"], DEV)]; allp += pred; allt += D["yte"].tolist()
            RES["flow24"][name]["folds"][str(f)] = {"overall": round(float((np.array(pred) == D["yte"]).mean()), 3), "seconds": round(time.time() - t0)}; print(f"   [flow-24] fold {f} {name:<48} overall {RES['flow24'][name]['folds'][str(f)]['overall']:.3f} ({round(time.time() - t0)} s)", flush=True); del m
        r = flow_metrics(np.array(allp), np.array(allt)); RES["flow24"][name].update(r)
        print(f"== [flow-24] {name:<48} overall {r['overall']:.3f} | macro all {r['macro_all']} | holds {r['macro_holds']} | named transitions {r['macro_named_transitions']} ({r['n_named_transition_classes']} cls) | A->B predicted as B->A {r['direction_confusion']} ({r['n_reversible_pairs']} pairs)", flush=True)
        save("model " + name)
    print("ALL DONE", round(time.time() - T0), "s", flush=True)
    print("RESULTS_JSON_BEGIN"); print(json.dumps(RES)); print("RESULTS_JSON_END", flush=True)
