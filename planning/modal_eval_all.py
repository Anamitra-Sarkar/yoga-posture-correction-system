"""Evaluate every checkpoint trained today. Runs DETACHED so it survives the client.

Two independent questions:
  1. Did landmark augmentation improve the MLP?  (v3_aug vs v2)
  2. Does any z-handling variant improve the ST-GCN? (rawz / zeroz / bonecorr / relab)

Nothing is promoted. Results are written back to the volumes as JSON.
"""
import modal

app = modal.App("asanaai-eval-all")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pandas", "huggingface_hub")
    .add_local_dir("/home/anamitra/yoga_posture_workspace/backend/app/models", remote_path="/models_src")
)

mlp_vol = modal.Volume.from_name("asanaai-mlp-v3")
stgcn_vol = modal.Volume.from_name("asanaai-stgcn")

HOLDOUT = {"4ORRiN2_aVI", "SZU7Sbgu57o"}


def _per_class(y_true, y_pred, classes):
    import numpy as np
    out, accs = {}, []
    for c in sorted(set(y_true)):
        m = [i for i, t in enumerate(y_true) if t == c]
        if not m:
            continue
        a = sum(1 for i in m if y_pred[i] == y_true[i]) / len(m)
        out[c] = {"n": len(m), "acc": round(a, 4)}
        accs.append(a)
    overall = sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))
    return {"per_class": out, "macro": round(sum(accs) / max(1, len(accs)), 4),
            "overall": round(overall, 4)}


@app.function(image=image, volumes={"/mlp": mlp_vol, "/stgcn": stgcn_vol}, timeout=3600, secrets=[modal.Secret.from_name("arko007-hf-token")])
def eval_mlp():
    import sys, os, json, glob
    import numpy as np, pandas as pd, torch
    sys.path.insert(0, "/models_src")
    sys.modules.setdefault("app", type(sys)("app"))
    from mlp import Yoga3HeadMLP

    results = {}
    FEATS = ["elbow_l","elbow_r","shoulder_l","shoulder_r","hip_l","hip_r","knee_l","knee_r",
             "ankle_l","ankle_r","trunk_l","trunk_r","neck","hip_abduct_l","hip_abduct_r"]

    ckpts = {
        "v2_live": ("/mlp/mlp_3head_model_v2.pth", "/mlp/mlp_3head_pose_encoder_v3_aug.npy"),
        "v3_aug":  ("/mlp/mlp_3head_model_v3_aug.pth", "/mlp/mlp_3head_pose_encoder_v3_aug.npy"),
    }
    # v2 lives on HF, not on this volume - fetch it so the comparison is real
    try:
        from huggingface_hub import hf_hub_download
        tok = os.environ.get("HF_TOKEN_ARKO007")
        mp2 = hf_hub_download("Arko007/yoga-posture-models", "mlp_3head_model_v2.pth", token=tok)
        ep2 = hf_hub_download("Arko007/yoga-posture-models", "mlp_3head_pose_encoder.npy", token=tok)
        ckpts["v2_live"] = (mp2, ep2)
        print("fetched v2 from HF", flush=True)
    except Exception as e:
        print("could not fetch v2:", e, flush=True)

    loaded = {}
    for name, (mp, ep) in ckpts.items():
        if not os.path.exists(mp):
            results[name] = {"error": f"missing {mp}"}
            continue
        classes = list(np.load(ep, allow_pickle=True))
        m = Yoga3HeadMLP(input_dim=15, num_poses=len(classes))
        m.load_state_dict(torch.load(mp, map_location="cpu"))
        m.eval()
        loaded[name] = (m, classes)
        print(f"loaded {name}: {len(classes)} classes", flush=True)

    # ---------- (a) real-world photo testset ----------
    man_p = "/mlp/testset/manifest.json"
    if os.path.exists(man_p):
        man = json.load(open(man_p))
        print("manifest type:", type(man), "len", len(man), flush=True)
        entries = man if isinstance(man, list) else man.get("items", man.get("samples", []))
        photo = {}
        for name, (m, classes) in loaded.items():
            yt, yp = [], []
            for e in (entries or []):
                f = e.get("file") or e.get("path") or e.get("npy")
                lab = e.get("true") or e.get("label") or e.get("pose")
                if not f or not lab:
                    continue
                fp = f if f.startswith("/") else f"/mlp/testset/{os.path.basename(f)}"
                if not os.path.exists(fp):
                    continue
                ang = np.load(fp, allow_pickle=True).astype(np.float32).reshape(-1)[:15]
                if ang.shape[0] != 15:
                    continue
                with torch.no_grad():
                    pl, _, _ = m(torch.tensor([ang]))
                yt.append(lab); yp.append(classes[int(pl.argmax(1))])
            if yt:
                photo[name] = _per_class(yt, yp, classes)
                print(f"[photos] {name}: n={len(yt)} macro={photo[name]['macro']} overall={photo[name]['overall']}", flush=True)
        results["real_world_photos"] = photo
    else:
        # fall back: infer labels from filenames like warrior_2_0.npy
        files = sorted(glob.glob("/mlp/testset/*.npy"))
        photo = {}
        for name, (m, classes) in loaded.items():
            yt, yp = [], []
            for fp in files:
                base = os.path.basename(fp)[:-4]
                lab = base.rsplit("_", 1)[0]
                ang = np.load(fp, allow_pickle=True).astype(np.float32).reshape(-1)[:15]
                if ang.shape[0] != 15:
                    continue
                with torch.no_grad():
                    pl, _, _ = m(torch.tensor([ang]))
                yt.append(lab); yp.append(classes[int(pl.argmax(1))])
            if yt:
                photo[name] = _per_class(yt, yp, classes)
                print(f"[photos-fromfilename] {name}: n={len(yt)} macro={photo[name]['macro']}", flush=True)
        results["real_world_photos"] = photo

    # ---------- (b) video-level holdout from the CSV ----------
    csv_p = "/mlp/raw/master_mlp_dataset_fully_classified.csv"
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        lab_col = "imperfect_pose_label" if "imperfect_pose_label" in df.columns else "pose_label"
        hv = df[df["video_id"].isin(HOLDOUT)]
        print(f"holdout rows: {len(hv)}", flush=True)
        if len(hv):
            X = hv[FEATS].values.astype(np.float32)
            def _base(l):
                l = str(l)
                if l.startswith("imperfect_"):
                    l = l[len("imperfect_"):]
                return "child_pose" if l == "child" else ("corpse" if l == "corpse" else l)
            y = [_base(v) for v in hv[lab_col].tolist()]
            vid = {}
            for name, (m, classes) in loaded.items():
                preds = []
                with torch.no_grad():
                    for i in range(0, len(X), 8192):
                        pl, _, _ = m(torch.tensor(X[i:i+8192]))
                        preds.extend(classes[int(k)] for k in pl.argmax(1))
                vid[name] = _per_class(y, preds, classes)
                print(f"[video-holdout] {name}: macro={vid[name]['macro']} overall={vid[name]['overall']}", flush=True)
            results["video_holdout"] = vid

    with open("/mlp/eval_results_full.json", "w") as f:
        json.dump(results, f, indent=1)
    mlp_vol.commit()
    return results


@app.function(image=image, volumes={"/stgcn": stgcn_vol}, timeout=3600)
def eval_stgcn():
    import sys, os, json
    import numpy as np, torch
    sys.path.insert(0, "/models_src")
    from sequence import YogaSequenceLSTM

    results = {}
    variants = [
        ("rawz",     "/stgcn/feats_rawz.npy",      "/stgcn/stgcn_rawz.pth",      "/stgcn/stgcn_label_encoder_rawz.npy",     "/stgcn/labels_seq.npy",  "/stgcn/videos_seq.npy"),
        ("zeroz",    "/stgcn/feats_zeroz.npy",     "/stgcn/stgcn_zeroz.pth",     "/stgcn/stgcn_label_encoder_zeroz.npy",    "/stgcn/labels_seq.npy",  "/stgcn/videos_seq.npy"),
        ("bonecorr", "/stgcn/feats_bonecorr.npy",  "/stgcn/stgcn_bonecorr.pth",  "/stgcn/stgcn_label_encoder_bonecorr.npy", "/stgcn/labels_seq.npy",  "/stgcn/videos_seq.npy"),
        ("relab_bonecorr", "/stgcn/relab_feats_bonecorr.npy", "/stgcn/stgcn_relab_bonecorr.pth", "/stgcn/stgcn_relab_encoder.npy", "/stgcn/relab_labels.npy", "/stgcn/relab_videos.npy"),
    ]
    for name, fp, mp, ep, lp, vp in variants:
        missing = [p for p in (fp, mp, ep, lp, vp) if not os.path.exists(p)]
        if missing:
            results[name] = {"error": f"missing {missing}"}
            print(f"{name}: MISSING {missing}", flush=True)
            continue
        X = np.load(fp, mmap_mode="r")
        y = np.load(lp, allow_pickle=True)
        vids = np.load(vp, allow_pickle=True)
        classes = list(np.load(ep, allow_pickle=True))
        mask = np.array([str(v) in HOLDOUT for v in vids])
        print(f"{name}: X{X.shape} holdout={int(mask.sum())}/{len(mask)} classes={len(classes)}", flush=True)
        if mask.sum() == 0:
            results[name] = {"error": "no holdout windows for these video ids", "videos_seen": sorted({str(v) for v in vids})[:12]}
            continue
        m = YogaSequenceLSTM(input_dim=99, hidden_dim=128, num_layers=2, num_classes=len(classes))
        sd = torch.load(mp, map_location="cpu")
        # trained checkpoints name the skip connection `res`; the production
        # class names it `residual`. Pure attribute rename, same architecture --
        # remap so production can actually load these.
        remapped = { (k.replace(".res.", ".residual.") if ".res." in k else k): v
                     for k, v in sd.items() }
        missing, unexpected = m.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            print(f"{name}: after remap missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        m.eval()
        idx = np.where(mask)[0]
        yt, yp = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                chunk = idx[i:i+256]
                xb = torch.tensor(np.asarray(X[chunk], dtype=np.float32))
                out = m(xb)
                for k, j in enumerate(chunk):
                    yt.append(str(y[j])); yp.append(classes[int(out[k].argmax())])
        results[name] = _per_class(yt, yp, classes)
        print(f"{name}: macro={results[name]['macro']} overall={results[name]['overall']}", flush=True)

    with open("/stgcn/eval_results_full.json", "w") as f:
        json.dump(results, f, indent=1)
    stgcn_vol.commit()
    return results


@app.local_entrypoint()
def main():
    print("=== MLP ===")
    try:
        r = eval_mlp.remote()
        print(str(r)[:1500])
    except Exception as e:
        print("MLP EVAL FAILED:", type(e).__name__, str(e)[:600])
    print("=== STGCN ===")
    try:
        r2 = eval_stgcn.remote()
        print(str(r2)[:1500])
    except Exception as e:
        print("STGCN EVAL FAILED:", type(e).__name__, str(e)[:600])
