"""Operating-point analysis for the target-pose ST-GCN, done WITHOUT leakage (CPU, ~minutes).

The model is conservative: most misses go to 'transition/unknown'. One scalar (an offset added to that class's logit) trades recall for precision.
Choosing the scalar on the same windows it is scored on would flatter the result, so it is CROSS-FITTED: for held-out fold i the offset is chosen on the
OTHER two folds' held-out predictions (objective = macro F1 over the pose classes) and applied only to fold i; the three cross-fitted parts are then pooled
and scored exactly like eval_stgcn_oof. The single global offset (tuned on everything) is printed for reference only -- it is optimistic.
"""
import glob, importlib.util, json, os, shutil, sys
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, snapshot_download
import numpy as np, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/originals/**", "vol/stgcn/cueT2_f[012]/test_*.npy"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueT2_stgcn_f[012]/**"])
spec = importlib.util.spec_from_file_location("stgcn_trainer", f"{ROOT}/code/originals/train_stgcn_gpu.py")
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
U = "transition/unknown"
LOG, TRUE, CLS = [], [], None
for i in range(3):
    cls = list(np.load(f"{ROOT}/vol/runs/cueT2_stgcn_f{i}/stgcn_label_encoder_v2.npy", allow_pickle=True)); CLS = CLS or cls
    assert cls == CLS, "class order differs between folds"
    X = tr.normalize_coordinate_sequence(np.load(f"{ROOT}/vol/stgcn/cueT2_f{i}/test_feats.npy"))
    y = np.load(f"{ROOT}/vol/stgcn/cueT2_f{i}/test_labels.npy", allow_pickle=True).astype(str)
    m = tr.YogaSequenceLSTM(99, 128, 2, len(cls)); m.load_state_dict(torch.load(f"{ROOT}/vol/runs/cueT2_stgcn_f{i}/stgcn_sequence_model_v2.pth", map_location="cpu")); m.eval()
    ds = tr.SequenceDataset(X, np.zeros(len(X), dtype=np.int64)); out = []
    with torch.no_grad():
        for b in range(0, len(ds), 256): out.append(m(ds.X[b:b + 256]).numpy())
    LOG.append(np.concatenate(out)); TRUE.append(y); print(f"fold {i}: {len(y)} windows", flush=True)
CLS = np.array(CLS); u = int(np.where(CLS == U)[0][0])
def predict(L, d):
    L = L.copy(); L[:, u] += d; return CLS[L.argmax(1)]
def score(P, T, min_n=20, bar=0.70):
    rows = {}
    for c in sorted(set(T.tolist())):
        t, pm = T == c, P == c; n = int(t.sum()); rec = float((P[t] == c).mean()); prec = float((T[pm] == c).mean()) if pm.sum() else 0.0
        rows[c] = {"n_windows": n, "recall": round(rec, 3), "precision": round(prec, 3), "pass": bool(n >= min_n and rec >= bar and prec >= bar and c != U)}
    ok = [c for c, r in rows.items() if r["pass"]]
    poses = [c for c, r in rows.items() if c != U and r["n_windows"] >= min_n]
    f1 = [2 * rows[c]["recall"] * rows[c]["precision"] / max(1e-9, rows[c]["recall"] + rows[c]["precision"]) for c in poses]
    return {"overall": round(float((P == T).mean()), 3), "n_pass": len(ok), "pass": ok, "macro_f1_poses": round(float(np.mean(f1)), 3),
            "macro_recall_poses": round(float(np.mean([rows[c]["recall"] for c in poses])), 3), "per_pose": rows}
GRID = [round(x, 2) for x in np.arange(-4.0, 4.01, 0.25)]
cat = lambda idx, key: np.concatenate([key[j] for j in idx])
base = score(cat(range(3), [predict(LOG[j], 0.0) for j in range(3)]), cat(range(3), TRUE))
print(f"BASELINE (argmax)  overall {base['overall']}  pass {base['n_pass']} {base['pass']}  macroF1 {base['macro_f1_poses']}", flush=True)
chosen, parts = [], []
for i in range(3):
    others = [j for j in range(3) if j != i]
    best = max(GRID, key=lambda d: (score(cat(others, [predict(LOG[j], d) for j in range(3)]), cat(others, TRUE))["macro_f1_poses"], -abs(d)))
    chosen.append(best); parts.append(predict(LOG[i], best))
    print(f"fold {i}: offset chosen on the other folds = {best:+.2f}", flush=True)
cf = score(np.concatenate(parts), cat(range(3), TRUE))
print(f"CROSS-FITTED       overall {cf['overall']}  pass {cf['n_pass']} {cf['pass']}  macroF1 {cf['macro_f1_poses']}  macroRecall {cf['macro_recall_poses']}", flush=True)
for c, r in sorted(cf["per_pose"].items(), key=lambda kv: -kv[1]["n_windows"]): print(f"   {c:<22} n={r['n_windows']:<5} rec={r['recall']:.2f} prec={r['precision']:.2f} {'PASS' if r['pass'] else ''}", flush=True)
# SECOND objective, disclosed as such (two objectives were tried, report both): choose the offset by the NUMBER OF POSES PASSING on the other folds (ties -> macro F1, then smallest |offset|), apply to the held-out fold.
chosen2, parts2 = [], []
for i in range(3):
    others = [j for j in range(3) if j != i]
    def obj(d):
        r = score(cat(others, [predict(LOG[j], d) for j in range(3)]), cat(others, TRUE)); return (r["n_pass"], r["macro_f1_poses"], -abs(d))
    best = max(GRID, key=obj); chosen2.append(best); parts2.append(predict(LOG[i], best))
    print(f"[pass-count objective] fold {i}: offset chosen on the other folds = {best:+.2f}", flush=True)
cf2 = score(np.concatenate(parts2), cat(range(3), TRUE))
print(f"CROSS-FITTED (pass-count objective)  overall {cf2['overall']}  pass {cf2['n_pass']} {cf2['pass']}  macroF1 {cf2['macro_f1_poses']}", flush=True)
for c, r in sorted(cf2["per_pose"].items(), key=lambda kv: -kv[1]["n_windows"]): print(f"   {c:<22} n={r['n_windows']:<5} rec={r['recall']:.2f} prec={r['precision']:.2f} {'PASS' if r['pass'] else ''}", flush=True)
gb = max(GRID, key=lambda d: (score(cat(range(3), [predict(LOG[j], d) for j in range(3)]), cat(range(3), TRUE))["macro_f1_poses"], -abs(d)))
gl = score(cat(range(3), [predict(LOG[j], gb) for j in range(3)]), cat(range(3), TRUE))
print(f"GLOBAL offset {gb:+.2f} (tuned on the scored windows: OPTIMISTIC, reference only)  pass {gl['n_pass']} {gl['pass']}  macroF1 {gl['macro_f1_poses']}", flush=True)
sweep = {str(d): {k: v for k, v in score(cat(range(3), [predict(LOG[j], d) for j in range(3)]), cat(range(3), TRUE)).items() if k != "per_pose"} for d in GRID}
res = {"baseline": base, "cross_fitted": {"offsets_per_fold": chosen, **cf}, "cross_fitted_passcount_objective": {"offsets_per_fold": chosen2, **cf2}, "global_reference": {"offset": gb, **gl}, "sweep": sweep}
json.dump(res, open(f"{ROOT}/op.json", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{ROOT}/op.json", path_in_repo="evals_stgcn/stgcn_operating_point_crossfit_v2.json", repo_id=MD, repo_type="model")
print("uploaded", flush=True)
