"""Controlled baselines for the paper (reviewers ask: 'compared with what?').  Everything is scored on data the models never trained on.

PART A  frame level -- held-out PUBLIC photos (1,685, 1,037 of them poses outside the vocabulary), the SAME test set and metrics as the cascade / rules-only numbers.
        Classical and plain-neural baselines are trained on EXACTLY the rows the new MLP trained on (vol/csv/cue2/vH/mlp_all.csv, 15 zero-z joint angles).
PART B  sequence level -- held-out VIDEOS (3 folds by video), the SAME windows the ST-GCN was scored on (vol/stgcn/cueT2_f{0,1,2}).
        LSTM, BiLSTM + multi-head attention (the PosePilot-style recipe), temporal CNN, window-statistics MLP, ST-GCN, and ST-GCN with the skeleton graph removed
        (identity adjacency, an ablation), all with ONE recipe: AdamW 1e-3 / wd 1e-4, cosine, class weight 1/sqrt(n), fixed epochs, the LAST epoch is scored
        (no epoch picked on the test windows), a per-class cap on training windows.  Single seed (variance not measured).
Results -> HF Arko007/asanaai-conference-runs/evals_compare/baselines_frame_v1.json (PART A) / baselines_seq_v1.json (PART B) (NEW names; nothing existing is touched)."""
import collections, glob, importlib.util, json, os, sys, time

PART = "A"   # "A" = frame level (CPU only), "B" = sequence level (needs a GPU); the two kernels are this file with PART changed

SEQ_EPOCHS = int(os.environ.get("SEQ_EPOCHS", "10")); CAP_PER_CLASS = int(os.environ.get("CAP_PER_CLASS", "6000")); BATCH = 128
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
    def stgcn(c): return tr.YogaSequenceLSTM(99, 128, 2, c)
    def stgcn_noedges(c):
        m = tr.YogaSequenceLSTM(99, 128, 2, c)
        for mod in m.modules():
            if hasattr(mod, "A") and getattr(mod, "A", None) is not None and mod.A.shape == (33, 33): mod.A.copy_(torch.eye(33))
        return m
    return {"LSTM (2 layers)": LSTMNet, "BiLSTM + multi-head attention": BiLSTMAttn, "Temporal CNN (no skeleton graph)": TCN, "MLP on window mean/std": StatMLP,
            "ST-GCN (identity adjacency = no skeleton graph)": stgcn_noedges, "ST-GCN (ours)": stgcn}


def train_seq(make, Xtr, ytr, ncls, epochs, dev, seed=0):
    import numpy as np, torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
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
        if ep % 3 == 0 or ep == epochs - 1: print(f"      ep{ep:02d} train loss {tot / len(X):.4f}", flush=True)
    return m.eval()


def predict_seq(m, X, dev):
    import numpy as np, torch
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 256): out.append(m(torch.as_tensor(X[b:b + 256]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    def cred(name):
        hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
        return open(hits[0]).read().strip()
    HF = cred("hf_token")
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    import numpy as np, pandas as pd, torch
    api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
    ROOT = "/kaggle/working"; DS, MD = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs"
    DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device", DEV, flush=True)
    snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/**"] + (["vol/csv/cue2/vH/mlp_all.csv", "vol/photos/public_corpus.npz"] if PART == "A" else ["vol/stgcn/cueT2_f[012]/**"]))
    sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal"); os.makedirs("/models_src", exist_ok=True)
    import shutil
    for f in os.listdir(f"{ROOT}/code/models_src"): shutil.copy(f"{ROOT}/code/models_src/{f}", f"/models_src/{f}")
    import asanaai_train2 as T
    rng = np.random.default_rng(0)
    RES = {"setup": {"seq_epochs": SEQ_EPOCHS, "cap_per_class": CAP_PER_CLASS, "batch": BATCH, "seed": 0, "device": DEV}}

    def save(tag):
        json.dump(RES, open(f"{ROOT}/baselines.json", "w"), indent=1)
        api.upload_file(path_or_fileobj=f"{ROOT}/baselines.json", path_in_repo=f"evals_compare/baselines_{'frame' if PART == 'A' else 'seq'}_v1.json", repo_id=MD, repo_type="model"); print("uploaded", tag, flush=True)

    if PART == "A":
    # ============================================================ PART A: frame level
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        df = pd.read_csv(f"{ROOT}/vol/csv/cue2/vH/mlp_all.csv", usecols=ANG + ["imperfect_pose_label"])
        ytr_all = df.imperfect_pose_label.astype(str).str.replace("imperfect_", "", regex=False).values; Xtr_all = df[ANG].values.astype(np.float32); del df
        print("frame training rows", len(ytr_all), dict(collections.Counter(ytr_all.tolist()).most_common(12)), flush=True)
        pc = np.load(f"{ROOT}/vol/photos/public_corpus.npz", allow_pickle=True); te = pc["split"] == "test"
        yp = pc["labels"][te].astype(str); Xp = np.array([T._angles(l, True) for l in pc["landmarks"][te]], dtype=np.float32)
        print("held-out public photos", len(yp), "other-pose", int((yp == U).sum()), flush=True)
        def capped(cap):
            idx = np.concatenate([rng.permutation(np.where(ytr_all == c)[0])[:cap] for c in sorted(set(ytr_all.tolist()))]); return idx
        sc = StandardScaler().fit(Xtr_all); Z = sc.transform(Xtr_all); Zp = sc.transform(Xp)
        classes_in_test = sorted(set(yp.tolist()))
        zoo = {"k-NN (k=15, distance-weighted)": (lambda: KNeighborsClassifier(15, weights="distance", n_jobs=-1), 200000),
               "Logistic regression": (lambda: LogisticRegression(max_iter=400, class_weight="balanced"), 100000),
               "SVM (RBF)": (lambda: SVC(C=3.0, gamma="scale", class_weight="balanced"), 20000),
               "Random forest (300 trees)": (lambda: RandomForestClassifier(300, n_jobs=-1, class_weight="balanced_subsample", random_state=0), 300000),
               "Plain MLP (256-256-128, ReLU)": (lambda: MLPClassifier((256, 256, 128), max_iter=30, early_stopping=False, random_state=0), 200000)}
        RES["frame"] = {}
        for name, (mk, cap_rows) in zoo.items():
            t0 = time.time(); per = max(1, cap_rows // len(set(ytr_all.tolist()))); idx = capped(per)
            if len(idx) > cap_rows: idx = rng.permutation(idx)[:cap_rows]
            m = mk().fit(Z[idx], ytr_all[idx]); pred = m.predict(Zp).astype(str)
            r = frame_metrics(pred, yp); r["train_rows"] = int(len(idx)); r["seconds"] = round(time.time() - t0)
            RES["frame"][name] = r
            print(f"[frame] {name:<34} rows {len(idx):>7} | overall {r['overall']:.3f} | macro recall {r['macro_recall']:.3f} | pass {r['n_pass']} {r['pass']} | other-pose false alarm {r['other_pose_false_alarm']:.3f} | {r['seconds']} s", flush=True)
            save("frame:" + name)
        # reference rows measured earlier on the same 1,685 photos (same metric code): read back, not recomputed
        try:
            ref = json.load(open(hf_hub_download(MD, "evals_compare/combo_v4_new_bysource.json", repo_type="model", token=HF))); RES["frame_reference_from_earlier_run"] = ref.get("public")
            print("reference (earlier run, same photos):", json.dumps(ref.get("public"))[:700], flush=True)
        except Exception as e: print("no reference file:", e, flush=True)
        save("frame done")


    if PART == "B":
    # ============================================================ PART B: sequence level
        spec = importlib.util.spec_from_file_location("stgcn_trainer", f"{ROOT}/code/originals/train_stgcn_gpu.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
        MODELS = make_sequence_models(tr)
        RES["sequence"] = {k: {"params": int(sum(p.numel() for p in v(9).parameters())), "folds": {}} for k, v in MODELS.items()}
        allP = {k: [] for k in MODELS}; allT = []
        for f in range(3):
            d = f"{ROOT}/vol/stgcn/cueT2_f{f}"
            ytr = np.load(f"{d}/stgcn_master_labels.npy", allow_pickle=True).astype(str); Xm = np.load(f"{d}/stgcn_master_feats.npy", mmap_mode="r")
            keep = np.sort(np.concatenate([rng.permutation(np.where(ytr == c)[0])[:CAP_PER_CLASS] for c in sorted(set(ytr.tolist()))]))
            classes = sorted(set(ytr.tolist())); cid = {c: i for i, c in enumerate(classes)}
            Xtr = tr.SequenceDataset(tr.normalize_coordinate_sequence(np.asarray(Xm[keep], dtype=np.float32)), np.zeros(len(keep), dtype=np.int64)).X.numpy(); ytri = np.array([cid[c] for c in ytr[keep]])
            Xte = tr.normalize_coordinate_sequence(np.load(f"{d}/test_feats.npy"))
            Xte = tr.SequenceDataset(Xte, np.zeros(len(Xte), dtype=np.int64)).X.numpy(); yte = np.load(f"{d}/test_labels.npy", allow_pickle=True).astype(str)
            print(f"\n=== fold {f}: train windows {len(keep)} (of {len(ytr)}), classes {classes}, test windows {len(yte)}", flush=True)
            allT += yte.tolist()
            for name, mk in MODELS.items():
                t0 = time.time(); m = train_seq(mk, Xtr, ytri, len(classes), SEQ_EPOCHS, DEV); pi = predict_seq(m, Xte, DEV); pred = [classes[j] for j in pi]
                allP[name] += pred; RES["sequence"][name]["folds"][str(f)] = {"overall": round(float((np.array(pred) == yte).mean()), 3), "seconds": round(time.time() - t0)}
                print(f"   fold {f} {name:<48} overall {RES['sequence'][name]['folds'][str(f)]['overall']:.3f} ({round(time.time() - t0)} s)", flush=True)
                del m; torch.cuda.empty_cache() if DEV == "cuda" else None
            save(f"seq fold {f}")
        Tt = np.array(allT)
        print("\n================ SEQUENCE BASELINES, held-out videos (3 folds pooled; pass = n>=20 windows & recall AND precision >= 0.70) ================", flush=True)
        for name in MODELS:
            r = seq_metrics(np.array(allP[name]), Tt); RES["sequence"][name].update(r)
            print(f"{name:<48} params {RES['sequence'][name]['params']:>8} | overall {r['overall']:.3f} | macro recall (9 cls) {r['macro_recall_9_classes']:.3f} (8 poses {r['macro_recall_poses']:.3f}) | pass {r['n_pass']} {r['pass']}", flush=True)
            print("     " + " ".join(f"{c.split('_')[0][:7]}:{x['recall']:.2f}/{x['precision']:.2f}" for c, x in r["per_class"].items()), flush=True)
        save("ALL DONE")
