"""Controlled baselines for the paper (reviewers ask: 'compared with what?').  Everything is scored on data the models never trained on.

PART A  frame level -- held-out PUBLIC photos (1,685, 1,037 of them poses outside the vocabulary), the SAME test set and metrics as the cascade / rules-only numbers.
        Classical and plain-neural baselines are trained on EXACTLY the rows the new MLP trained on (vol/csv/cue2/vH/mlp_all.csv, 15 zero-z joint angles).
PART B  sequence level -- held-out VIDEOS (3 folds by video), the SAME windows the ST-GCN was scored on (vol/stgcn/cueT2_f{0,1,2}).
        LSTM, BiLSTM + multi-head attention (the PosePilot-style recipe), temporal CNN, window-statistics MLP, ST-GCN, and ST-GCN with the skeleton graph removed
        (identity adjacency, an ablation), all with ONE recipe: AdamW 1e-3 / wd 1e-4, cosine, class weight 1/sqrt(n), fixed epochs, the LAST epoch is scored
        (no epoch picked on the test windows), a per-class cap on training windows.  Single seed (variance not measured).
THIS FILE = PART B on a GPU kernel (T4): <=6,000 windows per class, 10 epochs, models run one after the other, results saved after each model, 9.5 h time guard (expected ~30-60 min).
Results -> printed to the kernel log (RESULTS_JSON_BEGIN/END); nothing is uploaded."""
import collections, glob, importlib.util, json, os, sys, time

PART = "B"   # "A" = frame level (CPU only), "B" = sequence level (needs a GPU); the two kernels are this file with PART changed

SEQ_EPOCHS = int(os.environ.get("SEQ_EPOCHS", "10")); CAP_PER_CLASS = int(os.environ.get("CAP_PER_CLASS", "6000")); BATCH = 128   # GPU configuration (the CPU kernel uses 6 epochs / 1500 per class)
ANG = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck", "hip_abduct_l", "hip_abduct_r"]
U = "transition/unknown"


# ------------------------------------------------------------------ metrics (same definitions as runner_compare5.evaluate / runner_cmp.score)
def frame_metrics(pred, y, min_n=10, bar=0.70):
    import numpy as np
    inv = y != U
    cl = [c for c in sorted(set(y[inv])) if (y == c).sum() >= min_n]
    macro = float(np.mean([(pred[y == c] == c).mean() for c in cl])) if cl else float("nan")
    ok = [str(c) for c in cl if (pred[y == c] == c).mean() >= bar and (pred == c).sum() and (y[pred == c] == c).mean() >= bar]
    fa = float((pred[~inv] != U).mean()) if (~inv).sum() else float("nan")
    return {"overall": round(float((pred == y).mean()), 3), "macro_recall": round(macro, 3), "n_pass": len(ok), "pass": ok, "other_pose_false_alarm": round(fa, 3),
            "per_pose": {c: {"n": int((y == c).sum()), "recall": round(float((pred[y == c] == c).mean()), 3), "precision": round(float((y[pred == c] == c).mean()) if (pred == c).sum() else 0.0, 3)} for c in cl}}


def seq_metrics(P, T, min_n=20, bar=0.70):
    import numpy as np
    rows = {}
    for c in sorted(set(T.tolist())):
        t, pm = T == c, P == c; n = int(t.sum()); rec = float((P[t] == c).mean()); prec = float((T[pm] == c).mean()) if pm.sum() else 0.0
        rows[c] = {"n": n, "recall": round(rec, 3), "precision": round(prec, 3), "pass": bool(n >= min_n and rec >= bar and prec >= bar and c != U)}
    poses = [c for c in rows if c != U]; ok = [c for c in poses if rows[c]["pass"]]
    return {"overall": round(float((P == T).mean()), 3), "n_pass": len(ok), "pass": ok,
            "macro_recall_9_classes": round(float(np.mean([rows[c]["recall"] for c in rows])), 3),
            "macro_recall_poses": round(float(np.mean([rows[c]["recall"] for c in poses])), 3), "per_class": rows}


# ------------------------------------------------------------------ sequence models
def make_sequence_models(tr):
    import torch, torch.nn as nn
    class LSTMNet(nn.Module):
        def __init__(s, c):
            super().__init__(); s.rnn = nn.LSTM(99, 128, num_layers=2, batch_first=True, dropout=0.3); s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, c))
        def forward(s, x): return s.fc(s.rnn(x)[0][:, -1])
    class BiLSTMAttn(nn.Module):
        def __init__(s, c):
            super().__init__(); s.rnn = nn.LSTM(99, 128, num_layers=1, batch_first=True, bidirectional=True)
            s.att = nn.MultiheadAttention(256, 4, batch_first=True, dropout=0.1); s.ln = nn.LayerNorm(256); s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, c))
        def forward(s, x):
            h = s.rnn(x)[0]; a = s.att(h, h, h, need_weights=False)[0]; return s.fc(s.ln(h + a).mean(1))
    class TCN(nn.Module):
        def __init__(s, c):
            super().__init__()
            def blk(i, o): return nn.Sequential(nn.Conv1d(i, o, 9, padding=4), nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(0.2))
            s.net = nn.Sequential(blk(99, 128), blk(128, 256), blk(256, 256)); s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, c))
        def forward(s, x): return s.fc(s.net(x.transpose(1, 2)).mean(2))
    class StatMLP(nn.Module):   # no temporal modelling: mean and std of every coordinate over the window
        def __init__(s, c):
            super().__init__(); s.net = nn.Sequential(nn.Linear(198, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Linear(128, c))
        def forward(s, x): return s.net(torch.cat([x.mean(1), x.std(1)], 1))
    class To4D(nn.Module):   # the trainer's ST-GCN takes the graph layout [N, 3, 33, 60]; every baseline here gets [N, 60, 99] (frames x 33 joints x xyz)
        def __init__(s, inner): super().__init__(); s.inner = inner
        def forward(s, x): return s.inner(x.reshape(x.shape[0], 60, 33, 3).permute(0, 3, 2, 1).contiguous())
    def stgcn(c): return To4D(tr.YogaSequenceLSTM(99, 128, 2, c))
    def stgcn_noedges(c):
        m = tr.YogaSequenceLSTM(99, 128, 2, c)
        for mod in m.modules():
            if hasattr(mod, "A") and getattr(mod, "A", None) is not None and mod.A.shape == (33, 33): mod.A.copy_(torch.eye(33))
        return To4D(m)
    return {"LSTM (2 layers)": LSTMNet, "BiLSTM + multi-head attention": BiLSTMAttn, "Temporal CNN (no skeleton graph)": TCN, "MLP on window mean/std": StatMLP,
            "ST-GCN (identity adjacency = no skeleton graph)": stgcn_noedges, "ST-GCN (ours)": stgcn}


def train_seq(make, Xtr, ytr, ncls, epochs, dev, seed=0):
    import numpy as np, torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); T_START = time.time()
    m = make(ncls).to(dev)
    w = np.bincount(ytr, minlength=ncls).astype(np.float32); w = np.where(w > 0, 1.0 / np.sqrt(w), 0.0); w = w / w.sum() * ncls
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    amp = dev == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp)
    X = torch.as_tensor(Xtr); Y = torch.as_tensor(ytr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X)); tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2: continue
            xb, yb = X[b].to(dev), Y[b].to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): loss = crit(m(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += float(loss) * len(b)
        sched.step()
        print(f"      ep{ep:02d} train loss {tot / len(X):.4f} ({time.time() - T_START:.0f} s since model start)", flush=True)
    return m.eval()


def predict_seq(m, X, dev):
    import numpy as np, torch
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 256): out.append(m(torch.as_tensor(X[b:b + 256]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    T0 = time.time(); BUDGET = 9.5 * 3600
    # TOKENLESS variant (runs under a different Kaggle account that cannot see the private creds dataset): the HF dataset
    # Arko007/Yoga-1M is public, nothing is uploaded, and the results are PRINTED to the log between RESULTS_JSON markers.
    from huggingface_hub import snapshot_download
    import numpy as np, torch
    ROOT = "/kaggle/working"; DS = "Arko007/Yoga-1M"
    # FAIL FAST: this kernel is only meaningful on a working GPU (the ST-GCN costs ~10 GFLOP per window; on CPU it would take ~13 h per variant).
    if not torch.cuda.is_available(): raise SystemExit("NO CUDA DEVICE: refusing to run the sequence baselines on CPU")
    try:
        _x = torch.randn(256, 256, device="cuda"); float((_x @ _x).sum()); print("GPU OK:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), "torch", torch.__version__, flush=True)
    except Exception as _e:
        raise SystemExit(f"GPU PRESENT BUT UNUSABLE ({torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}, torch {torch.__version__}): {_e}")
    DEV = "cuda"
    snapshot_download(DS, repo_type="dataset", local_dir=ROOT, allow_patterns=["code/**", "vol/stgcn/cueT2_f[012]/**"])
    sys.path.insert(0, f"{ROOT}/code/originals")
    spec = importlib.util.spec_from_file_location("stgcn_trainer", f"{ROOT}/code/originals/train_stgcn_gpu.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
    rng = np.random.default_rng(0)
    RES = {"setup": {"seq_epochs": SEQ_EPOCHS, "cap_per_class": CAP_PER_CLASS, "batch": BATCH, "seed": 0, "device": DEV, "note": "GPU budget; all models share one recipe"}}
    def save(tag):
        json.dump(RES, open(f"{ROOT}/baselines_seq_gpu.json", "w"), indent=1)
        print("saved", tag, flush=True)
    FOLDS = []
    for f in range(3):
        d = f"{ROOT}/vol/stgcn/cueT2_f{f}"
        ytr = np.load(f"{d}/stgcn_master_labels.npy", allow_pickle=True).astype(str); Xm = np.load(f"{d}/stgcn_master_feats.npy", mmap_mode="r")
        keep = np.sort(np.concatenate([rng.permutation(np.where(ytr == c)[0])[:CAP_PER_CLASS] for c in sorted(set(ytr.tolist()))]))
        classes = sorted(set(ytr.tolist())); cid = {c: i for i, c in enumerate(classes)}
        Xtr = np.asarray(tr.normalize_coordinate_sequence(np.asarray(Xm[keep], dtype=np.float32)), dtype=np.float32); ytri = np.array([cid[c] for c in ytr[keep]])
        assert Xtr.ndim == 3 and Xtr.shape[1:] == (60, 99), Xtr.shape
        if f == 0:   # the wrapper's reshape must reproduce the trainer's own graph layout exactly
            ds4 = tr.SequenceDataset(Xtr[:8], np.zeros(8, dtype=np.int64)).X; mine = torch.tensor(Xtr[:8]).reshape(8, 60, 33, 3).permute(0, 3, 2, 1)
            assert tuple(ds4.shape) == (8, 3, 33, 60) and torch.allclose(ds4.float(), mine.float(), atol=1e-5), ("layout mismatch", tuple(ds4.shape))
            print("layout check OK: trainer graph layout == reshape/permute of [N,60,99]", flush=True)
        Xte = np.asarray(tr.normalize_coordinate_sequence(np.load(f"{d}/test_feats.npy")), dtype=np.float32); assert Xte.shape[1:] == (60, 99), Xte.shape; yte = np.load(f"{d}/test_labels.npy", allow_pickle=True).astype(str)
        FOLDS.append(dict(Xtr=Xtr, ytri=ytri, Xte=Xte, yte=yte, classes=classes)); print(f"fold {f}: train windows {len(keep)} (of {len(ytr)}), classes {classes}, test windows {len(yte)}", flush=True)
    MODELS = make_sequence_models(tr)
    RES["sequence"] = {k: {"params": int(sum(p.numel() for p in v(9).parameters())), "folds": {}} for k, v in MODELS.items()}
    Tt = np.concatenate([D["yte"] for D in FOLDS])
    for name, mk in MODELS.items():
        if time.time() - T0 > BUDGET: RES["sequence"][name]["skipped"] = "time budget"; print("SKIPPED (time budget):", name, flush=True); save("skip " + name); continue
        allp = []
        for f, D in enumerate(FOLDS):
            t0 = time.time(); m = train_seq(mk, D["Xtr"], D["ytri"], len(D["classes"]), SEQ_EPOCHS, DEV); pred = [D["classes"][j] for j in predict_seq(m, D["Xte"], DEV)]
            allp += pred; RES["sequence"][name]["folds"][str(f)] = {"overall": round(float((np.array(pred) == D["yte"]).mean()), 3), "seconds": round(time.time() - t0)}
            print(f"   fold {f} {name:<48} overall {RES['sequence'][name]['folds'][str(f)]['overall']:.3f} ({round(time.time() - t0)} s)", flush=True); del m
        r = seq_metrics(np.array(allp), Tt); RES["sequence"][name].update(r)
        print(f"== {name:<48} params {RES['sequence'][name]['params']:>8} | overall {r['overall']:.3f} | macro recall (9 cls) {r['macro_recall_9_classes']:.3f} (8 poses {r['macro_recall_poses']:.3f}) | pass {r['n_pass']} {r['pass']}", flush=True)
        print("     " + " ".join(f"{c.split('_')[0][:7]}:{x['recall']:.2f}/{x['precision']:.2f}" for c, x in r["per_class"].items()), flush=True)
        save("model " + name)
    print("ALL DONE", round(time.time() - T0), "s", flush=True)
    print("RESULTS_JSON_BEGIN"); print(json.dumps(RES)); print("RESULTS_JSON_END", flush=True)
