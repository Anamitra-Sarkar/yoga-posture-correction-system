"""Does combining v4 (live) + NEW beat either alone? Rules fixed IN ADVANCE (no tuning):
  GATE : answer = v4's top pose, unless NEW's top class is 'transition/unknown' (then: unknown).
  AVG  : average the two probability vectors (over base-pose classes), take the argmax.
Frozen 103 Commons photos: neither model trained on them (NEW scored out-of-fold). Public held-out photos: NEW never saw them; v4 MAY have seen
similar public images (unknown) -- flagged, and it can only flatter v4."""
import glob, json, os, shutil, sys
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import numpy as np, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/**", "vol/eval/**", "vol/folds/**", "vol/photos/public_corpus.npz"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueH_mlp_*/**"])
os.makedirs("/models_src", exist_ok=True)
for f in os.listdir(f"{ROOT}/code/models_src"): shutil.copy(f"{ROOT}/code/models_src/{f}", f"/models_src/{f}")
sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal"); sys.path.insert(0, "/models_src")
import asanaai_train2 as T
from mlp import Yoga3HeadMLP
def load(mp, ep):
    cls = [c.replace("imperfect_", "") for c in np.load(ep, allow_pickle=True)]
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls)); m.load_state_dict(torch.load(mp, map_location="cpu")); m.eval(); return m, cls
def probs(m, cls, X, names):
    with torch.no_grad(): p = torch.softmax(m(torch.tensor(X))[0], 1).numpy()
    out = np.zeros((len(X), len(names)), dtype=np.float32)
    for j, c in enumerate(cls): out[:, names.index(c)] += p[:, j]
    return out
v4m, v4c = load(hf_hub_download(LIVE, "mlp_3head_v4_photos_x2000.pth", token=HF), hf_hub_download(LIVE, "mlp_3head_v4_encoder.npy", token=HF))
new_all = [c.replace("imperfect_", "") for c in np.load(f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_pose_encoder_v2.npy", allow_pickle=True)]
names = sorted(set(v4c) | set(new_all)); U = names.index("transition/unknown")
def evaluate(X, y, Pv, Pn, tag, min_n):
    out = {}
    cands = {"v4 alone": Pv.argmax(1), "NEW alone": Pn.argmax(1), "GATE v4 unless NEW says other": np.where(Pn.argmax(1) == U, U, Pv.argmax(1)),
             "AVG of both": (Pv + Pn).argmax(1)}
    print(f"\n===== {tag} (n={len(y)}) =====")
    for k, a in cands.items():
        pred = np.array([names[i] for i in a])
        inv = y != "transition/unknown"
        cl = [c for c in sorted(set(y[inv])) if (y == c).sum() >= min_n]
        macro = float(np.mean([(pred[y == c] == c).mean() for c in cl])) if cl else float("nan")
        ok = [c for c in cl if (pred[y == c] == c).mean() >= .7 and (pred == c).sum() and (y[pred == c] == c).mean() >= .7]
        fa = float((pred[~inv] != "transition/unknown").mean()) if (~inv).sum() else float("nan")
        out[k] = {"overall": round(float((pred == y).mean()), 3), "macro_recall": round(macro, 3), "n_pass": len(ok), "pass": ok, "other_pose_false_alarm": None if np.isnan(fa) else round(fa, 3)}
        print(f"{k:<32} overall {out[k]['overall']:.3f} | macro recall {macro:.3f} | pass {len(ok)} {ok} | other-pose false alarm {out[k]['other_pose_false_alarm']}")
    return out
res = {}
# --- frozen 103 Commons photos
z = np.load(f"{ROOT}/vol/eval/photo_corpus.npz", allow_pickle=True)
lm, y, sp = z["landmarks"], z["labels"].astype(str), z["split"].astype(str)
fold = np.array(json.load(open(f"{ROOT}/vol/folds/photo_folds.json"))["fold"])
X = np.array([T._angles(l, True) for l in lm], dtype=np.float32)
Pv = probs(v4m, v4c, X, names); Pn = np.zeros_like(Pv)
for i in range(5):
    m, c = load(f"{ROOT}/vol/runs/cueH_mlp_fold{i}/mlp_3head_model_v2.pth", f"{ROOT}/vol/runs/cueH_mlp_fold{i}/mlp_3head_pose_encoder_v2.npy")
    idx = np.where(fold == i)[0]; Pn[idx] = probs(m, c, X[idx], names)
t = sp == "test"
res["frozen103"] = evaluate(X[t], y[t], Pv[t], Pn[t], "FROZEN 103 Commons photos (clean for both)", 5)
# --- held-out public photos
p = np.load(f"{ROOT}/vol/photos/public_corpus.npz", allow_pickle=True)
te = p["split"] == "test"; yp = p["labels"][te].astype(str)
Xp = np.array([T._angles(l, True) for l in p["landmarks"][te]], dtype=np.float32)
m, c = load(f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_model_v2.pth", f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_pose_encoder_v2.npy")
res["public"] = evaluate(Xp, yp, probs(v4m, v4c, Xp, names), probs(m, c, Xp, names), "HELD-OUT PUBLIC photos (v4 may have seen similar ones -> only flatters v4)", 10)
json.dump(res, open(f"{ROOT}/combo.json", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{ROOT}/combo.json", path_in_repo="evals_compare/combo_v4_new.json", repo_id=MD, repo_type="model")
