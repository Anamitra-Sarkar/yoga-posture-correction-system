"""PRODUCTION-PATH test. Drives the REAL backend functions (extract_angles_from_landmarks, compute_orientation, classify_pose, hybrid_classify)
exactly as analyse_frame does, with the same inputs the web client sends, on photos no policy was tuned on.
Policies fixed in advance:
  H0 production today : hybrid_classify(v4 outputs)
  H1 gate in MLP slot : mlp_pose := 'transition/unknown' when NEW says other, else v4 pose; then hybrid_classify
  H2 hard veto        : H0, then final := 'transition/unknown' if NEW says other
  H3 NEW alone        : hybrid_classify(NEW outputs)
  R  rules only       : classify_pose (reference)"""
import glob, json, os, shutil, sys
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import numpy as np, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/backend/**", "vol/eval/**", "vol/folds/**", "vol/photos/public_corpus.npz"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueH_mlp_*/**"])
sys.path.insert(0, f"{ROOT}/code/backend")
from app.utils.geometry import extract_angles_from_landmarks, compute_orientation, FEATURE_NAMES
from app.utils.rules_classifier import hybrid_classify, classify_pose, sanitize_pose
from app.models.mlp import Yoga3HeadMLP
def load(mp, ep):
    cls = list(np.load(ep, allow_pickle=True))
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls)); m.load_state_dict(torch.load(mp, map_location="cpu")); m.eval(); return m, cls
V4 = load(hf_hub_download(LIVE, "mlp_3head_v4_photos_x2000.pth", token=HF), hf_hub_download(LIVE, "mlp_3head_v4_encoder.npy", token=HF))
print("v4 classes:", V4[1], flush=True)
def mlp_outputs(model, angles):                       # identical to analyse_frame
    m, cls = model
    with torch.no_grad():
        pl, cl, dv = m(torch.tensor([angles], dtype=torch.float32))
    pose = cls[pl.argmax(1).item()]; corr = torch.sigmoid(cl).item()
    devs = {FEATURE_NAMES[i]: float(min(180.0, max(0.0, dv[0].numpy()[i] * 180.0))) for i in range(15)}
    return pose, corr, devs
def new_other(model, angles):                         # NEW's collapsed distribution: is 'transition/unknown' the top class?
    m, cls = model
    with torch.no_grad(): p = torch.softmax(m(torch.tensor([angles], dtype=torch.float32))[0], 1)[0].numpy()
    agg = {}
    for c, v in zip(cls, p): agg[c.replace("imperfect_", "")] = agg.get(c.replace("imperfect_", ""), 0) + float(v)
    top = max(agg, key=agg.get); return top, agg
def run(lm, world, new_models):
    """returns dict policy -> pose for one photo"""
    pts = lm[:, :3].astype(np.float64)
    ang = [float(a) for a in extract_angles_from_landmarks(pts, True)]
    ad = dict(zip(FEATURE_NAMES, ang)); ori = compute_orientation(pts)
    wd = None if world is None else dict(zip(FEATURE_NAMES, [float(a) for a in extract_angles_from_landmarks(world.astype(np.float64), False)]))
    vp, vc, vd = mlp_outputs(V4, ang)
    out = {}
    for nm, nmodel in new_models:
        ntop, _ = new_other(nmodel, ang)
        np_, nc, nd = mlp_outputs(nmodel, ang)
        out[nm] = {"H0": hybrid_classify(vp, vc, vd, ad, wd, ori)[0],
                   "H1": hybrid_classify("transition/unknown" if ntop == "transition/unknown" else vp, vc, vd, ad, wd, ori)[0],
                   "H3": hybrid_classify(np_.replace("imperfect_", ""), nc, nd, ad, wd, ori)[0],
                   "R": sanitize_pose(classify_pose(ad, ori))}
        out[nm]["H2"] = "transition/unknown" if ntop == "transition/unknown" else out[nm]["H0"]
    return out
POL = ["H0", "H1", "H2", "H3", "R"]
LAB = {"H0": "production today (v4+rules)", "H1": "gate in MLP slot", "H2": "hard veto after hybrid", "H3": "NEW alone + rules", "R": "rules only"}
def report(y, preds, title, min_n):
    print(f"\n===== {title} (n={len(y)}) =====")
    res = {}
    for k in POL:
        p = np.array(preds[k]); inv = y != "transition/unknown"
        cl = [c for c in sorted(set(y[inv])) if (y == c).sum() >= min_n]
        macro = float(np.mean([(p[y == c] == c).mean() for c in cl])) if cl else float("nan")
        ok = [str(c) for c in cl if (p[y == c] == c).mean() >= .7 and (p == c).sum() and (y[p == c] == c).mean() >= .7]
        fa = float((p[~inv] != "transition/unknown").mean()) if (~inv).sum() else None
        res[k] = {"overall": round(float((p == y).mean()), 3), "macro_recall": round(macro, 3), "pass": ok, "false_alarm": None if fa is None else round(fa, 3)}
        print(f"{LAB[k]:<28} overall {res[k]['overall']:.3f} | macro recall {macro:.3f} | pass {len(ok)} {ok} | other-pose false alarm {res[k]['false_alarm']}")
    return res
out = {}
# ---- A) frozen 103 Commons photos (with world landmarks => 3-way vote path, as the modern web client sends)
z = np.load(f"{ROOT}/vol/eval/photo_corpus.npz", allow_pickle=True)
lm, wl, y, sp = z["landmarks"], z["world"], z["labels"].astype(str), z["split"].astype(str)
fold = np.array(json.load(open(f"{ROOT}/vol/folds/photo_folds.json"))["fold"])
fm = [load(f"{ROOT}/vol/runs/cueH_mlp_fold{i}/mlp_3head_model_v2.pth", f"{ROOT}/vol/runs/cueH_mlp_fold{i}/mlp_3head_pose_encoder_v2.npy") for i in range(5)]
for use_world in (True, False):
    P = {k: [] for k in POL}; idx = np.where(sp == "test")[0]
    for i in idx:
        r = run(lm[i], wl[i] if use_world else None, [("n", fm[fold[i]])])["n"]
        for k in POL: P[k].append(r[k])
    out[f"frozen103_{'3way(world)' if use_world else '2way'}"] = report(y[idx], P, f"FROZEN 103 Commons photos, {'3-way vote with world landmarks' if use_world else '2-way (no world landmarks)'}", 5)
# ---- B) held-out public photos (no world landmarks stored => 2-way path); VAL/TEST halves by hash parity
p = np.load(f"{ROOT}/vol/photos/public_corpus.npz", allow_pickle=True)
te = np.where(p["split"] == "test")[0]; yp = p["labels"][te].astype(str); keys = p["keys"][te].astype(str)
allm = load(f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_model_v2.pth", f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_pose_encoder_v2.npy")
P = {k: [] for k in POL}
for j in te:
    r = run(p["landmarks"][j], None, [("n", allm)])["n"]
    for k in POL: P[k].append(r[k])
isval = np.array([int(k_[:8], 16) % 2 == 0 for k_ in keys])
out["public_VAL(selection half)"] = report(yp[isval], {k: np.array(v)[isval] for k, v in P.items()}, "PUBLIC held-out, VAL half (used to choose the policy)", 10)
out["public_TEST(honest half)"] = report(yp[~isval], {k: np.array(v)[~isval] for k, v in P.items()}, "PUBLIC held-out, TEST half (not used for choosing)", 10)
out["public_ALL"] = report(yp, P, "PUBLIC held-out, ALL", 10)
json.dump(out, open(f"{ROOT}/prod_path.json", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{ROOT}/prod_path.json", path_in_repo="evals_compare/production_path.json", repo_id=MD, repo_type="model")
