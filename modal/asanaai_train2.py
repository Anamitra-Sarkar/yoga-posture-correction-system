"""Train the ORIGINAL (unmodified) MLP + ST-GCN trainers on the extended dataset, then evaluate honestly.

* train_mlp2(csv_rel, tag)      original train_mlp_3head_gpu.py on /data/<csv_rel> (HF upload disabled: see asanaai_train.py)
* build_stgcn2(csv_rel, tag)    original generate_sequence_features.py over old+new landmarks
* train_stgcn2(feats_tag, tag)  original train_stgcn_gpu.py on those windows
* eval_photos(tag)              per-pose precision/recall on the 422 REAL photos (never seen in training),
                                under BOTH angle recipes (raw-z = how the CSV was built, zero-z = what the app feeds)
"""
import json
import os
import subprocess
import sys

import modal

SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
HOME = "/home/anamitra"
DATA = f"{HOME}/yoga_raw_dataset"
app = modal.App("asanaai-train2")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("pandas", "numpy<2", "scikit-learn", "huggingface_hub")
    .add_local_file(f"{SRC}/train_mlp_3head_gpu.py", "/opt/orig/train_mlp_3head_gpu.py", copy=True)
    .add_local_file(f"{SRC}/train_stgcn_gpu.py", "/opt/orig/train_stgcn_gpu.py", copy=True)
    .add_local_file(f"{SRC}/generate_sequence_features.py", "/opt/orig/generate_sequence_features.py", copy=True)
    .add_local_dir("/home/anamitra/yoga_posture_workspace/backend/app/models", remote_path="/models_src", copy=True)
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
secret = modal.Secret.from_name("arko007-hf-token")
CSV_NAME = "master_mlp_dataset_fully_classified.csv"


def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def _run(script, tag, outputs):
    env = {**os.environ, "HF_TOKEN": "upload-disabled-by-wrapper", "PYTHONUNBUFFERED": "1"}
    print(f"\n{'='*70}\nRUN {script} tag={tag}\n{'='*70}", flush=True)
    r = subprocess.run([sys.executable, script], env=env, cwd=HOME)
    print(f"\nexit code {r.returncode}", flush=True)
    outdir = f"{VOL}/runs/{tag}"
    os.makedirs(outdir, exist_ok=True)
    import shutil
    saved = []
    for f in outputs:
        if os.path.exists(f"{HOME}/{f}"):
            shutil.copy(f"{HOME}/{f}", f"{outdir}/{f}")
            saved.append((f, os.path.getsize(f"{HOME}/{f}")))
    vol.commit()
    print("saved:", saved, flush=True)
    return {"exit": r.returncode, "saved": saved}


@app.function(image=image, volumes={VOL: vol}, secrets=[secret], gpu="L4", timeout=4 * 3600, memory=32768)
def train_mlp2(csv_rel: str, tag: str):
    vol.reload()
    os.makedirs(DATA, exist_ok=True)
    _link(f"{VOL}/{csv_rel}", f"{DATA}/{CSV_NAME}")
    print("CSV:", csv_rel, f"{os.path.getsize(f'{VOL}/{csv_rel}')/1e6:.0f} MB", flush=True)
    return _run("/opt/orig/train_mlp_3head_gpu.py", tag,
                ["mlp_3head_model_v2.pth", "mlp_3head_pose_encoder_v2.npy"])


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=49152, timeout=3 * 3600)
def build_stgcn2(csv_rel: str, tag: str):
    import glob
    import numpy as np
    vol.reload()
    os.makedirs(DATA, exist_ok=True)
    for f in glob.glob(f"{VOL}/landmarks/landmarks_*.npy") + glob.glob(f"{VOL}/landmarks_new/landmarks_*.npy"):
        _link(f, f"{DATA}/{os.path.basename(f)}")
    _link(f"{VOL}/{csv_rel}", f"{DATA}/{CSV_NAME}")
    r = subprocess.run([sys.executable, "/opt/orig/generate_sequence_features.py"], cwd=HOME,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print("exit", r.returncode, flush=True)
    if r.returncode:
        return {"exit": r.returncode}
    import shutil
    out = f"{VOL}/stgcn/{tag}"
    os.makedirs(out, exist_ok=True)
    for f in ("stgcn_master_feats.npy", "stgcn_master_labels.npy"):
        shutil.copy(f"{DATA}/{f}", f"{out}/{f}")
    lab = np.load(f"{out}/stgcn_master_labels.npy", allow_pickle=True)
    import collections
    cnt = collections.Counter(lab.tolist())
    vol.commit()
    print("windows:", len(lab), "classes:", len(cnt), dict(cnt.most_common(30)), flush=True)
    return {"windows": int(len(lab)), "classes": dict(cnt)}


@app.function(image=image, volumes={VOL: vol}, secrets=[secret], gpu="L4", timeout=6 * 3600, memory=49152)
def train_stgcn2(feats_tag: str, tag: str):
    vol.reload()
    os.makedirs(DATA, exist_ok=True)
    for f in ("stgcn_master_feats.npy", "stgcn_master_labels.npy"):
        _link(f"{VOL}/stgcn/{feats_tag}/{f}", f"{DATA}/{f}")
    return _run("/opt/orig/train_stgcn_gpu.py", tag, ["stgcn_sequence_model_v2.pth", "stgcn_label_encoder_v2.npy"])


NOSE = 0; SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
HP_L, HP_R, KN_L, KN_R, AN_L, AN_R, HE_L, HE_R = 23, 24, 25, 26, 27, 28, 29, 30


def _angles(pts, zero_z):
    import numpy as np
    p = np.asarray(pts, dtype=np.float64)[:, :3].copy()
    if zero_z:
        p[:, 2] = 0.0

    def a(x, y, z):
        ba, bc = p[x] - p[y], p[z] - p[y]
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        return 180.0 if nb == 0 or nc == 0 else float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (nb * nc), -1, 1))))
    sm, hm = (p[SH_L] + p[SH_R]) / 2, (p[HP_L] + p[HP_R]) / 2
    ba, bc = p[NOSE] - sm, hm - sm
    nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    neck = 180.0 if nb == 0 or nc == 0 else float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (nb * nc), -1, 1))))
    return [a(SH_L, EL_L, WR_L), a(SH_R, EL_R, WR_R), a(HP_L, SH_L, EL_L), a(HP_R, SH_R, EL_R),
            a(SH_L, HP_L, KN_L), a(SH_R, HP_R, KN_R), a(HP_L, KN_L, AN_L), a(HP_R, KN_R, AN_R),
            a(KN_L, AN_L, HE_L), a(KN_R, AN_R, HE_R), a(SH_L, HP_L, HP_R), a(SH_R, HP_R, HP_L),
            neck, a(HP_R, HP_L, KN_L), a(HP_L, HP_R, KN_R)]


@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def eval_photos(tag: str, min_n: int = 10, bar: float = 0.70):
    """Real-photo evaluation. The video-trained models never saw any photo, so all 422 are held out."""
    import collections
    import numpy as np
    import torch
    sys.path.insert(0, "/models_src")
    from mlp import Yoga3HeadMLP
    vol.reload()
    z = np.load(f"{VOL}/eval/photo_corpus.npz", allow_pickle=True)
    lm, y, sp = z["landmarks"], z["labels"].astype(str), z["split"].astype(str)
    cls = list(np.load(f"{VOL}/runs/{tag}/mlp_3head_pose_encoder_v2.npy", allow_pickle=True))
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls))
    m.load_state_dict(torch.load(f"{VOL}/runs/{tag}/mlp_3head_model_v2.pth", map_location="cpu"))
    m.eval()
    base = lambda c: c.replace("imperfect_", "")  # noqa: E731
    out = {"tag": tag, "n_photos": int(len(y)), "classes": cls}
    for recipe, zero_z in (("zero_z(app)", True), ("raw_z(csv)", False)):
        X = torch.tensor(np.array([_angles(l, zero_z) for l in lm], dtype=np.float32))
        with torch.no_grad():
            pred = np.array([base(cls[i]) for i in m(X)[0].argmax(1).numpy()])
        rows = {}
        for c in sorted(set(y.tolist())):
            t, pm = y == c, pred == c
            n = int(t.sum())
            rec = float((pred[t] == c).mean())
            prec = float((y[pm] == c).mean()) if pm.sum() else 0.0
            conf = collections.Counter(pred[t & (pred != c)].tolist()).most_common(2)
            rows[c] = {"n": n, "recall": round(rec, 3), "precision": round(prec, 3),
                       "pass": bool(n >= min_n and rec >= bar and prec >= bar), "confused_with": conf}
        ok = [c for c, r in rows.items() if r["pass"]]
        enough = [r["recall"] for r in rows.values() if r["n"] >= min_n]
        out[recipe] = {"per_pose": rows, "passing_poses": ok, "n_passing": len(ok),
                       "macro_recall_n>=%d" % min_n: round(float(np.mean(enough)), 3) if enough else None,
                       "overall_acc": round(float((pred == y).mean()), 3)}
        print(f"\n== {tag} | {recipe} | passing(n>={min_n}, recall&precision>={bar}): {ok}  "
              f"overall acc {out[recipe]['overall_acc']}", flush=True)
        for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"]):
            print(f"   {c:<24} n={r['n']:<4} rec={r['recall']:.2f} prec={r['precision']:.2f} "
                  f"{'PASS' if r['pass'] else '    '} {r['confused_with']}", flush=True)
    json.dump(out, open(f"{VOL}/runs/{tag}/photo_eval.json", "w"), indent=1)
    vol.commit()
    return out


@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def eval_oof(prefix: str = "cue_mlp_fold", k: int = 5, min_n: int = 10, bar: float = 0.70, name: str = "oof", targets: str = ""):
    """Out-of-fold real-photo evaluation: model <prefix><i> scores ONLY the photos of fold i, which it never saw.
    Photo angles use the app's recipe (z zeroed)."""
    import collections
    import numpy as np
    import torch
    sys.path.insert(0, "/models_src")
    from mlp import Yoga3HeadMLP
    vol.reload()
    z = np.load(f"{VOL}/eval/photo_corpus.npz", allow_pickle=True)
    lm, y = z["landmarks"], z["labels"].astype(str)
    fold = np.array(json.load(open(f"{VOL}/folds/photo_folds.json"))["fold"])
    X_all = np.array([_angles(l, True) for l in lm], dtype=np.float32)
    pred = np.empty(len(y), dtype=object)
    for i in range(k):
        tag = f"{prefix}{i}"
        cls = list(np.load(f"{VOL}/runs/{tag}/mlp_3head_pose_encoder_v2.npy", allow_pickle=True))
        m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls))
        m.load_state_dict(torch.load(f"{VOL}/runs/{tag}/mlp_3head_model_v2.pth", map_location="cpu"))
        m.eval()
        idx = np.where(fold == i)[0]
        with torch.no_grad():
            pr = m(torch.tensor(X_all[idx]))[0].argmax(1).numpy()
        for j, ii in enumerate(idx):
            pred[ii] = cls[pr[j]].replace("imperfect_", "")
    pred = pred.astype(str)
    if targets:
        tl = set(targets.split(","))
        y = np.array([l if l in tl else "transition/unknown" for l in y])
    rows = {}
    for c in sorted(set(y.tolist())):
        t, pm = y == c, pred == c
        n = int(t.sum())
        rec = float((pred[t] == c).mean())
        prec = float((y[pm] == c).mean()) if pm.sum() else 0.0
        rows[c] = {"n": n, "recall": round(rec, 3), "precision": round(prec, 3),
                   "pass": bool(n >= min_n and rec >= bar and prec >= bar and (not targets or c != "transition/unknown")),
                   "confused_with": collections.Counter(pred[t & (pred != c)].tolist()).most_common(2)}
    ok = [c for c, r in rows.items() if r["pass"]]
    enough = [r["recall"] for r in rows.values() if r["n"] >= min_n]
    out = {"name": name, "n_photos": int(len(y)), "passing_poses": ok, "n_passing": len(ok),
           "overall_acc": round(float((pred == y).mean()), 3),
           "macro_recall_n>=%d" % min_n: round(float(np.mean(enough)), 3), "per_pose": rows}
    print(f"\n== {name} OUT-OF-FOLD on {len(y)} real photos | passing (n>={min_n}, recall&precision>={bar}): "
          f"{len(ok)} -> {ok} | overall acc {out['overall_acc']}", flush=True)
    for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"]):
        print(f"   {c:<24} n={r['n']:<4} rec={r['recall']:.2f} prec={r['precision']:.2f} "
              f"{'PASS' if r['pass'] else '    '} {r['confused_with']}", flush=True)
    os.makedirs(f"{VOL}/runs/_oof", exist_ok=True)
    json.dump(out, open(f"{VOL}/runs/_oof/{name}.json", "w"), indent=1)
    vol.commit()
    return out


OLD_IDS = "HmZFwoUU3WQ SZU7Sbgu57o oUgpXY7QhpQ 7ciS93shMNQ P8uHMMmWMHQ 4ORRiN2_aVI RQMtwbhXD7A 8ibxmzJziHU Eml2xnoLpYE QiebZSlTw_U L-z1HLkS_-Y s-1vMbAgYWU".split()


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=49152, timeout=3 * 3600)
def build_windows_cue(nfold: int = 3, ignore_max: float = 0.15, trans_ratio: float = 1.5, targets: str = "", out_prefix: str = "cue"):
    """60-frame windows over the cue-labelled videos, using the ORIGINAL label_window / SEQ_LENGTH / STRIDE.
    Windows holding >ignore_max excluded frames are dropped; the rest are labelled from their non-excluded frames.
    Videos are split into nfold held-out-video folds (balanced by positive seconds) so every pose is scored on
    people/rooms the model never trained on."""
    import collections
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/opt/orig")
    import generate_sequence_features as gsf
    vol.reload()
    df = pd.read_csv(f"{VOL}/csv/cue/cue_full.csv", usecols=["video_id", "frame_num", "imperfect_pose_label"])
    rng = np.random.default_rng(7)
    X, Y, V = [], [], []
    for vid in df.video_id.unique():
        lp = f"{VOL}/landmarks/landmarks_{vid}.npy" if vid in OLD_IDS else f"{VOL}/landmarks_new/landmarks_{vid}.npy"
        lm = np.load(lp)
        coords = lm[:, :, :3].reshape(len(lm), -1)
        lab = df[df.video_id == vid].sort_values("frame_num")["imperfect_pose_label"].values
        assert len(lab) == len(coords), (vid, len(lab), len(coords))
        if targets:
            tl = set(targets.split(","))
            lab = np.array([l if (l in tl or l in ("transition/unknown", "__ignore__")) else "transition/unknown" for l in lab], dtype=object)
        for st in range(0, len(coords) - gsf.SEQ_LENGTH + 1, gsf.STRIDE):
            w = lab[st:st + gsf.SEQ_LENGTH]
            ign = w == "__ignore__"
            if ign.mean() > ignore_max:
                continue
            X.append(coords[st:st + gsf.SEQ_LENGTH]); Y.append(gsf.label_window(w[~ign])); V.append(vid)
        print(vid, "windows so far", len(Y), flush=True)
    X, Y, V = np.array(X, dtype=np.float32), np.array(Y), np.array(V)
    pos = Y != "transition/unknown"
    tr_idx = np.where(~pos)[0]
    keep_tr = rng.choice(tr_idx, min(len(tr_idx), int(trans_ratio * pos.sum())), replace=False)
    sel = np.sort(np.concatenate([np.where(pos)[0], keep_tr]))
    X, Y, V = X[sel], Y[sel], V[sel]
    # balanced held-out-video folds
    vids = sorted(set(V.tolist()))
    secs = {v: collections.Counter(Y[(V == v) & (Y != "transition/unknown")].tolist()) for v in vids}
    fold_load = [collections.Counter() for _ in range(nfold)]
    fold_of = {}
    for v in sorted(vids, key=lambda v: -sum(secs[v].values())):
        best = min(range(nfold), key=lambda f: sum((fold_load[f][c] + n) ** 2 for c, n in secs[v].items())
                   + 0.001 * sum(fold_load[f].values()))
        fold_of[v] = best
        fold_load[best].update(secs[v])
    fv = np.array([fold_of[v] for v in V])
    base = f"{VOL}/stgcn"
    for f in list(range(nfold)) + ["all"]:
        d = f"{base}/{out_prefix}_{'all' if f == 'all' else 'f' + str(f)}"
        os.makedirs(d, exist_ok=True)
        tr = np.ones(len(Y), bool) if f == "all" else fv != f
        np.save(f"{d}/stgcn_master_feats.npy", X[tr]); np.save(f"{d}/stgcn_master_labels.npy", Y[tr])
        if f != "all":
            np.save(f"{d}/test_feats.npy", X[~tr]); np.save(f"{d}/test_labels.npy", Y[~tr])
    rep = {"windows": int(len(Y)), "classes": dict(collections.Counter(Y.tolist())), "fold_of_video": fold_of,
           "fold_windows_per_class": [dict(c) for c in fold_load]}
    json.dump(rep, open(f"{base}/{out_prefix}_build_report.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(rep, indent=1), flush=True)
    return rep


@app.function(image=image, volumes={VOL: vol}, gpu="L4", timeout=3600, memory=32768)
def eval_stgcn_oof(prefix: str = "cue_stgcn_f", k: int = 3, min_n: int = 20, bar: float = 0.70, name: str = "stgcn_oof", test_prefix: str = "cue_f"):
    """Held-out-VIDEO evaluation: model <prefix><i> scores only windows from videos it never trained on."""
    import collections
    import importlib.util
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("stgcn_trainer", "/opt/orig/train_stgcn_gpu.py")
    tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
    vol.reload()
    P, T = [], []
    for i in range(k):
        cls = list(np.load(f"{VOL}/runs/{prefix}{i}/stgcn_label_encoder_v2.npy", allow_pickle=True))
        Xt = tr.normalize_coordinate_sequence(np.load(f"{VOL}/stgcn/{test_prefix}{i}/test_feats.npy"))
        yt = np.load(f"{VOL}/stgcn/{test_prefix}{i}/test_labels.npy", allow_pickle=True)
        model = tr.YogaSequenceLSTM(99, 128, 2, len(cls))
        model.load_state_dict(torch.load(f"{VOL}/runs/{prefix}{i}/stgcn_sequence_model_v2.pth", map_location="cpu"))
        model.cuda().eval()
        ds = tr.SequenceDataset(Xt, np.zeros(len(Xt), dtype=np.int64))
        pr = []
        with torch.no_grad():
            for b in range(0, len(ds), 256):
                pr.append(model(ds.X[b:b + 256].cuda()).argmax(1).cpu().numpy())
        pr = np.concatenate(pr)
        P += [cls[j] for j in pr]; T += yt.tolist()
    P, T = np.array(P), np.array(T)
    rows = {}
    for c in sorted(set(T.tolist())):
        t, pm = T == c, P == c
        n = int(t.sum())
        rec = float((P[t] == c).mean())
        prec = float((T[pm] == c).mean()) if pm.sum() else 0.0
        rows[c] = {"n_windows": n, "recall": round(rec, 3), "precision": round(prec, 3),
                   "pass": bool(n >= min_n and rec >= bar and prec >= bar and c != "transition/unknown"),
                   "confused_with": collections.Counter(P[t & (P != c)].tolist()).most_common(2)}
    ok = [c for c, r in rows.items() if r["pass"]]
    out = {"name": name, "passing_poses": ok, "n_passing": len(ok), "overall_acc": round(float((P == T).mean()), 3), "per_pose": rows}
    print(f"\n== {name} HELD-OUT-VIDEO windows | passing (n>={min_n}, recall&precision>={bar}): {len(ok)} -> {ok} | "
          f"overall acc {out['overall_acc']}", flush=True)
    for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["n_windows"]):
        print(f"   {c:<24} n={r['n_windows']:<5} rec={r['recall']:.2f} prec={r['precision']:.2f} "
              f"{'PASS' if r['pass'] else '    '} {r['confused_with']}", flush=True)
    os.makedirs(f"{VOL}/runs/_oof", exist_ok=True)
    json.dump(out, open(f"{VOL}/runs/_oof/{name}.json", "w"), indent=1)
    vol.commit()
    return out


@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def eval_public(tag: str, min_n: int = 10, bar: float = 0.70, name: str = "", targets: str = ""):
    """Held-out PUBLIC photos (hash-split test, never in any training set) incl. out-of-vocabulary negatives.
    recall on in-vocabulary poses; precision counts every test photo, so false alarms on 'other poses' hurt."""
    import collections
    import numpy as np
    import torch
    sys.path.insert(0, "/models_src")
    from mlp import Yoga3HeadMLP
    vol.reload()
    z = np.load(f"{VOL}/photos/public_corpus.npz", allow_pickle=True)
    te = z["split"] == "test"
    lm, y = z["landmarks"][te], z["labels"][te].astype(str)
    if targets:
        tl = set(targets.split(","))
        y = np.array([l if l in tl else "transition/unknown" for l in y])
    cls = list(np.load(f"{VOL}/runs/{tag}/mlp_3head_pose_encoder_v2.npy", allow_pickle=True))
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls))
    m.load_state_dict(torch.load(f"{VOL}/runs/{tag}/mlp_3head_model_v2.pth", map_location="cpu"))
    m.eval()
    X = torch.tensor(np.array([_angles(l, True) for l in lm], dtype=np.float32))
    with torch.no_grad():
        pred = np.array([cls[i].replace("imperfect_", "") for i in m(X)[0].argmax(1).numpy()])
    rows = {}
    for c in sorted(set(y.tolist()) - {"transition/unknown"}):
        t, pm = y == c, pred == c
        n = int(t.sum())
        rec = float((pred[t] == c).mean())
        prec = float((y[pm] == c).mean()) if pm.sum() else 0.0
        rows[c] = {"n": n, "recall": round(rec, 3), "precision": round(prec, 3), "pass": bool(n >= min_n and rec >= bar and prec >= bar),
                   "confused_with": collections.Counter(pred[t & (pred != c)].tolist()).most_common(2)}
    oov = y == "transition/unknown"
    out = {"tag": tag, "n_test": int(len(y)), "n_oov": int(oov.sum()),
           "oov_false_alarm_rate": round(float((pred[oov] != "transition/unknown").mean()), 3) if oov.sum() else None,
           "passing_poses": [c for c, r in rows.items() if r["pass"]], "per_pose": rows}
    out["n_passing"] = len(out["passing_poses"])
    print(f"\n== {name or tag} | PUBLIC-TEST ({len(y)} photos, {int(oov.sum())} out-of-vocab) | passing {out['n_passing']}: "
          f"{out['passing_poses']} | OOV false-alarm {out['oov_false_alarm_rate']}", flush=True)
    for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["n"]):
        print(f"   {c:<24} n={r['n']:<4} rec={r['recall']:.2f} prec={r['precision']:.2f} {'PASS' if r['pass'] else '    '} {r['confused_with']}", flush=True)
    os.makedirs(f"{VOL}/runs/_oof", exist_ok=True)
    json.dump(out, open(f"{VOL}/runs/_oof/public_{name or tag}.json", "w"), indent=1)
    vol.commit()
    return out


MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
                (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def _mirror_windows(X):
    """Left-right mirror of [N,60,99] raw MediaPipe windows: x -> 1-x and swap left/right joints."""
    import numpy as np
    J = X.reshape(len(X), X.shape[1], 33, 3).copy()
    J[..., 0] = 1.0 - J[..., 0]
    for a, b in MIRROR_PAIRS:
        J[:, :, [a, b], :] = J[:, :, [b, a], :]
    return J.reshape(len(X), X.shape[1], 99)


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=49152, timeout=3 * 3600)
def build_windows_cue2(nfold: int = 3, reps: int = 3, oov_cap: int = 1500, min_photos: int = 10, in_prefix: str = "cue", out_prefix: str = "cue2", targets: str = ""):
    """v2 training windows = v1 train windows + their mirror + photo-hold clips (+mirror). Test windows are the v1 ones
    (held-out videos, untouched). Photo-hold clip = a real photo's 33x3 landmarks held for 60 frames, plus natural sway
    taken from REAL stable video holds (residual of a random cue-verified positive window about its own mean,
    rescaled by hip width). Photo labels are human/dataset labels; out-of-vocabulary photos become 'transition/unknown'."""
    import collections
    import numpy as np
    vol.reload()
    rng = np.random.default_rng(11)
    pc = np.load(f"{VOL}/photos/public_corpus.npz", allow_pickle=True)
    cz = np.load(f"{VOL}/eval/photo_corpus.npz", allow_pickle=True)
    ph_lm = [pc["landmarks"][pc["split"] == "train"], cz["landmarks"]]
    ph_lab = [pc["labels"][pc["split"] == "train"].astype(str), cz["labels"].astype(str)]
    LM = np.concatenate(ph_lm)[:, :, :3].reshape(-1, 99).astype(np.float32)
    LAB = np.concatenate(ph_lab)
    # keep poses with enough photos; cap out-of-vocab
    cnt = collections.Counter(LAB.tolist())
    ok = np.array([(cnt[l] >= min_photos) for l in LAB])
    oov = np.where(ok & (LAB == "transition/unknown"))[0]
    if len(oov) > oov_cap:
        drop = set(rng.choice(oov, len(oov) - oov_cap, replace=False).tolist())
        ok &= np.array([i not in drop for i in range(len(LAB))])
    LM, LAB = LM[ok], LAB[ok]
    if targets:
        tl = set(targets.split(","))
        LAB = np.array([l if l in tl else "transition/unknown" for l in LAB])

    def hw(w):  # hip width of a [..,99] pose
        j = w.reshape(*w.shape[:-1], 33, 3)
        return np.linalg.norm(j[..., 23, :2] - j[..., 24, :2], axis=-1)

    rep = {}
    for f in list(range(nfold)) + ["all"]:
        src = f"{VOL}/stgcn/{in_prefix}_{'all' if f == 'all' else 'f' + str(f)}"
        X, Y = np.load(f"{src}/stgcn_master_feats.npy"), np.load(f"{src}/stgcn_master_labels.npy", allow_pickle=True)
        bank = X[Y != "transition/unknown"]
        res = bank - bank.mean(axis=1, keepdims=True)
        res_hw = np.maximum(hw(bank.mean(axis=1)), 1e-3)
        clips, clab = [], []
        for _ in range(reps):
            idx = rng.integers(0, len(bank), len(LM))
            scale = (np.maximum(hw(LM), 1e-3) / res_hw[idx])[:, None, None]
            clips.append(LM[:, None, :] + res[idx] * scale)
            clab.append(LAB)
        P = np.concatenate(clips).astype(np.float32)
        PY = np.concatenate(clab)
        Xa = np.concatenate([X, _mirror_windows(X), P, _mirror_windows(P)])
        Ya = np.concatenate([Y, Y, PY, PY])
        perm = rng.permutation(len(Ya))
        out = f"{VOL}/stgcn/{out_prefix}_{'all' if f == 'all' else 'f' + str(f)}"
        os.makedirs(out, exist_ok=True)
        if f != "all":
            import shutil
            for t in ("test_feats.npy", "test_labels.npy"):
                shutil.copy(f"{src}/{t}", f"{out}/{t}")
        np.save(f"{out}/stgcn_master_feats.npy", Xa[perm]); np.save(f"{out}/stgcn_master_labels.npy", Ya[perm])
        rep[str(f)] = {"windows": int(len(Ya)), "classes": dict(collections.Counter(Ya.tolist()).most_common(40))}
        print(f, rep[str(f)]["windows"], flush=True)
    json.dump(rep, open(f"{VOL}/stgcn/{out_prefix}_build_report.json", "w"), indent=1)
    vol.commit()
    return rep


@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def calibrate(tag_all: str, fold_prefix: str, targets: str, min_prec: float = 0.78, min_n: int = 10, bar: float = 0.70, k: int = 5):
    """Per-pose confidence thresholds (operating points). Tuned ONLY on the VAL half of the held-out public photos
    (hash parity). Reported on (a) the untouched TEST half and (b) the 422 independent Commons photos scored out-of-fold.
    Rule: pose c is emitted only if it is the arg-max AND its probability >= tau_c, else 'transition/unknown' (other)."""
    import collections
    import numpy as np
    import torch
    sys.path.insert(0, "/models_src")
    from mlp import Yoga3HeadMLP
    vol.reload()
    tl = [t for t in targets.split(",") if t]

    def load(tag):
        cls = list(np.load(f"{VOL}/runs/{tag}/mlp_3head_pose_encoder_v2.npy", allow_pickle=True))
        m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls))
        m.load_state_dict(torch.load(f"{VOL}/runs/{tag}/mlp_3head_model_v2.pth", map_location="cpu")); m.eval()
        return m, [c.replace("imperfect_", "") for c in cls]

    def probs(m, base, X):
        with torch.no_grad():
            p = torch.softmax(m(torch.tensor(X))[0], 1).numpy()
        # collapse imperfect_* columns onto their base pose
        names = sorted(set(base)); out = np.zeros((len(X), len(names)), dtype=np.float32)
        for j, b in enumerate(base):
            out[:, names.index(b)] += p[:, j]
        return out, names

    def decide(P, names, tau):
        a = P.argmax(1)
        pred = np.array([names[i] for i in a], dtype=object)
        for i, c in enumerate(pred):
            if c in tau and P[i, a[i]] < tau[c]:
                pred[i] = "transition/unknown"
        return pred

    def table(y, pred, label):
        rows, ok = {}, []
        for c in tl:
            t, pm = y == c, pred == c
            n = int(t.sum())
            if n == 0:
                continue
            rec = float((pred[t] == c).mean()); prec = float((y[pm] == c).mean()) if pm.sum() else 0.0
            ps = bool(n >= min_n and rec >= bar and prec >= bar)
            rows[c] = (n, round(rec, 3), round(prec, 3), ps)
            if ps:
                ok.append(c)
        oth = y == "transition/unknown"
        print(f"\n== {label} | passing {len(ok)}: {ok} | other-photos wrongly flagged as a pose: "
              f"{float((pred[oth] != 'transition/unknown').mean()):.3f} (n={int(oth.sum())})", flush=True)
        for c, (n, r, p_, ps) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            print(f"   {c:<22} n={n:<4} rec={r:.2f} prec={p_:.2f} {'PASS' if ps else ''}", flush=True)
        return {"passing": ok, "rows": rows}

    z = np.load(f"{VOL}/photos/public_corpus.npz", allow_pickle=True)
    te = np.where(z["split"] == "test")[0]
    ylab = np.array([l if l in tl else "transition/unknown" for l in z["labels"][te].astype(str)])
    keys = z["keys"][te].astype(str)
    is_val = np.array([int(k[:8], 16) % 2 == 0 for k in keys])
    X = np.array([_angles(l, True) for l in z["landmarks"][te]], dtype=np.float32)
    m_all, base = load(tag_all)
    P, names = probs(m_all, base, X)
    tau = {}
    for c in tl:
        if c not in names:
            continue
        j = names.index(c)
        best = None
        for t in np.arange(0.20, 0.991, 0.02):
            pred = decide(P[is_val], names, {**tau, c: t})
            yv = ylab[is_val]
            tp, pp, nn = int(((pred == c) & (yv == c)).sum()), int((pred == c).sum()), int((yv == c).sum())
            if pp == 0 or nn == 0:
                continue
            prec, rec = tp / pp, tp / nn
            if prec >= min_prec and (best is None or rec > best[1]):
                best = (float(t), rec, prec)
        tau[c] = best[0] if best else 0.5
    print("thresholds tuned on VAL half:", {k_: round(v, 2) for k_, v in tau.items()}, flush=True)
    res = {"tau": tau}
    res["public_val(tuned)"] = table(ylab[is_val], decide(P[is_val], names, tau), "PUBLIC VAL half (used for tuning)")
    res["public_test(honest)"] = table(ylab[~is_val], decide(P[~is_val], names, tau), "PUBLIC TEST half (never used for tuning)")
    # Commons out-of-fold, thresholds fixed from the public VAL half
    c = np.load(f"{VOL}/eval/photo_corpus.npz", allow_pickle=True)
    cy = np.array([l if l in tl else "transition/unknown" for l in c["labels"].astype(str)])
    fold = np.array(json.load(open(f"{VOL}/folds/photo_folds.json"))["fold"])
    Xc = np.array([_angles(l, True) for l in c["landmarks"]], dtype=np.float32)
    predc = np.empty(len(cy), dtype=object)
    for i in range(k):
        mi, bi = load(f"{fold_prefix}{i}")
        Pi, ni = probs(mi, bi, Xc[fold == i])
        predc[fold == i] = decide(Pi, ni, tau)
    res["commons_oof(honest)"] = table(cy, predc.astype(str), "COMMONS 422 photos, out-of-fold (never used for tuning)")
    json.dump(res, open(f"{VOL}/runs/{tag_all}/operating_points.json", "w"), indent=1, default=str)
    vol.commit()
    return res


@app.local_entrypoint()
def main(stage: str, csv_rel: str = "", tag: str = "", feats_tag: str = "", targets: str = "", in_prefix: str = "cue", out_prefix: str = "", test_prefix: str = "cue_f"):
    if stage == "mlp":
        print(train_mlp2.remote(csv_rel, tag))
    elif stage == "build":
        print(build_stgcn2.remote(csv_rel, tag))
    elif stage == "stgcn":
        print(train_stgcn2.remote(feats_tag, tag))
    elif stage == "eval":
        eval_photos.remote(tag)
    elif stage == "windows2":
        print(build_windows_cue2.remote(in_prefix=in_prefix, out_prefix=out_prefix or "cue2", targets=targets))
    elif stage == "windows":
        print(build_windows_cue.remote(targets=targets, out_prefix=out_prefix or "cue"))
    elif stage == "soof":
        eval_stgcn_oof.remote(prefix=tag, name=csv_rel or "stgcn_oof", test_prefix=test_prefix)
    elif stage == "calib":
        calibrate.remote(tag, csv_rel, targets)
    elif stage == "pub":
        eval_public.remote(tag, targets=targets)
    elif stage == "oof":
        eval_oof.remote(prefix=tag, name=csv_rel or "oof", targets=targets)
    else:
        raise SystemExit("stage = mlp|build|stgcn|eval")
