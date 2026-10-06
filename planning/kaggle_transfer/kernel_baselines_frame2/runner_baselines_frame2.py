"""Follow-up to baselines-frame (v1): a random forest on the same 15 angles scored 85.2% / 9.7% false alarms / 9 poses on the 1,685 held-out public photos, above the
neural cascade (78.9% / 20.8% / 7).  Before that goes into a paper we check whether it is real:
 (1) a SECOND held-out set: the frozen Commons-103 photos, each scored by a model trained WITHOUT its fold (out-of-fold, the same protocol as the neural models);
 (2) NEAR-DUPLICATE audit of the public test set: distance of every test photo to its nearest TRAINING PHOTO (standardised 15-angle space); accuracy / false alarms per distance quartile
     for the baselines AND for the actual networks (v4, new gate model, cascade) on the SAME photos;
 (3) the per-pose recall/precision of every method.
Everything single-seed.  Results -> HF Arko007/asanaai-conference-runs/evals_compare/baselines_frame_v2.json (new name)."""
import glob, json, os, shutil, sys, time

ANG = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck", "hip_abduct_l", "hip_abduct_r"]
U = "transition/unknown"

def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
import numpy as np, pandas as pd, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/**", "vol/csv/cue2/vH/mlp_*.csv", "vol/photos/public_corpus.npz", "vol/eval/**", "vol/folds/**"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueH_mlp_all/**"])
os.makedirs("/models_src", exist_ok=True)
for f in os.listdir(f"{ROOT}/code/models_src"): shutil.copy(f"{ROOT}/code/models_src/{f}", f"/models_src/{f}")
sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal"); sys.path.insert(0, "/models_src")
import asanaai_train2 as T
from mlp import Yoga3HeadMLP
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
RES = {}
def save(tag):
    json.dump(RES, open(f"{ROOT}/baselines2.json", "w"), indent=1)
    api.upload_file(path_or_fileobj=f"{ROOT}/baselines2.json", path_in_repo="evals_compare/baselines_frame_v2.json", repo_id=MD, repo_type="model"); print("uploaded", tag, flush=True)

def metrics(pred, y, min_n):
    inv = y != U; cl = [c for c in sorted(set(y[inv])) if (y == c).sum() >= min_n]
    per = {c: {"n": int((y == c).sum()), "recall": round(float((pred[y == c] == c).mean()), 3), "precision": round(float((y[pred == c] == c).mean()) if (pred == c).sum() else 0.0, 3)} for c in cl}
    ok = [c for c in cl if per[c]["recall"] >= .7 and per[c]["precision"] >= .7 and (pred == c).sum() > 0]
    fa = float((pred[~inv] != U).mean()) if (~inv).sum() else float("nan")
    return {"n": int(len(y)), "overall": round(float((pred == y).mean()), 3), "macro_recall": round(float(np.mean([per[c]["recall"] for c in cl])), 3) if cl else None,
            "n_pass": len(ok), "pass": ok, "other_pose_false_alarm": None if np.isnan(fa) else round(fa, 3), "per_pose": per}
def line(name, m): return f"{name:<34} n={m['n']:<5} overall {m['overall']:.3f} | macro recall {m['macro_recall']} | pass {m['n_pass']} {m['pass']} | false alarm {m['other_pose_false_alarm']}"

ZOO = {"Random forest (300 trees)": lambda: RandomForestClassifier(300, n_jobs=-1, class_weight="balanced_subsample", random_state=0),
       "Plain MLP (256-256-128, ReLU)": lambda: MLPClassifier((256, 256, 128), max_iter=30, early_stopping=False, random_state=0),
       "k-NN (k=15, distance-weighted)": lambda: KNeighborsClassifier(15, weights="distance", n_jobs=-1),
       "SVM (RBF)": lambda: SVC(C=3.0, gamma="scale", class_weight="balanced")}
CAP = {"Random forest (300 trees)": 300000, "Plain MLP (256-256-128, ReLU)": 200000, "k-NN (k=15, distance-weighted)": 200000, "SVM (RBF)": 20000}
rng = np.random.default_rng(0)
def fit_predict(csv, Xq_list):
    df = pd.read_csv(csv, usecols=ANG + ["imperfect_pose_label", "difficulty"])
    y = df.imperfect_pose_label.astype(str).str.replace("imperfect_", "", regex=False).values; X = df[ANG].values.astype(np.float32); photo = (df.difficulty.astype(str) == "photo").values
    sc = StandardScaler().fit(X); Z = sc.transform(X); outs = {}
    classes = sorted(set(y.tolist()))
    for name, mk in ZOO.items():
        per = max(1, CAP[name] // len(classes)); idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:per] for c in classes])
        if len(idx) > CAP[name]: idx = rng.permutation(idx)[:CAP[name]]
        m = mk().fit(Z[idx], y[idx]); outs[name] = [m.predict(sc.transform(Xq)).astype(str) for Xq in Xq_list]
    return outs, sc, Z, photo

# ---------------------------------------------------------------- (1)+(2) held-out public photos, with the networks on the SAME photos
p = np.load(f"{ROOT}/vol/photos/public_corpus.npz", allow_pickle=True); te = p["split"] == "test"; yp = p["labels"][te].astype(str)
Xp = np.array([T._angles(l, True) for l in p["landmarks"][te]], dtype=np.float32)
print("held-out public photos", len(yp), "other-pose", int((yp == U).sum()), flush=True)
t0 = time.time(); outs, sc, Z, photo = fit_predict(f"{ROOT}/vol/csv/cue2/vH/mlp_all.csv", [Xp]); print("baselines fitted", round(time.time() - t0), "s", flush=True)
PRED = {k: v[0] for k, v in outs.items()}
def load(mp, ep):
    cls = [c.replace("imperfect_", "") for c in np.load(ep, allow_pickle=True)]
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls)); m.load_state_dict(torch.load(mp, map_location="cpu")); m.eval(); return m, cls
def probs(m, cls, X, names):
    with torch.no_grad(): pr = torch.softmax(m(torch.tensor(X))[0], 1).numpy()
    out = np.zeros((len(X), len(names)), dtype=np.float32)
    for j, c in enumerate(cls): out[:, names.index(c)] += pr[:, j]
    return out
v4m, v4c = load(hf_hub_download(LIVE, "mlp_3head_v4_photos_x2000.pth", token=HF), hf_hub_download(LIVE, "mlp_3head_v4_encoder.npy", token=HF))
nm, nc = load(f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_model_v2.pth", f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_pose_encoder_v2.npy")
names = sorted(set(v4c) | set(nc)); Ui = names.index(U); Pv = probs(v4m, v4c, Xp, names); Pn = probs(nm, nc, Xp, names)
PRED["v4 (earlier photo-trained ResMLP)"] = np.array([names[i] for i in Pv.argmax(1)])
PRED["New 3-head ResMLP (gate model) alone"] = np.array([names[i] for i in Pn.argmax(1)])
PRED["Cascade (v4 names, new model vetoes)"] = np.array([names[i] for i in np.where(Pn.argmax(1) == Ui, Ui, Pv.argmax(1))])
RES["public_overall"] = {k: metrics(v, yp, 10) for k, v in PRED.items()}
print("\n===== HELD-OUT PUBLIC PHOTOS (1,685) =====", flush=True)
for k, m in RES["public_overall"].items(): print(line(k, m), flush=True)
for k, m in RES["public_overall"].items(): print("   per-pose", k, " ".join(f"{c.split('_')[0][:8]}:{x['recall']:.2f}/{x['precision']:.2f}(n{x['n']})" for c, x in m["per_pose"].items()), flush=True)
save("public")

# near-duplicate audit
phZ = np.unique(np.round(Z[photo], 4), axis=0); nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(phZ)
d = nn.kneighbors(sc.transform(Xp))[0][:, 0]; q = np.quantile(d, [0.25, 0.5, 0.75])
RES["nn_distance"] = {"photo_rows_unique": int(len(phZ)), "quantiles_25_50_75": [round(float(x), 4) for x in q], "frac_below_0.05": round(float((d < 0.05).mean()), 3), "frac_below_0.2": round(float((d < 0.2).mean()), 3)}
print("\nnearest TRAINING-photo distance (standardised angles): quartile edges", RES["nn_distance"]["quantiles_25_50_75"], "| share <0.05:", RES["nn_distance"]["frac_below_0.05"], "| share <0.2:", RES["nn_distance"]["frac_below_0.2"], flush=True)
edges = [-1, q[0], q[1], q[2], 1e9]; RES["by_nn_quartile"] = {}
for i in range(4):
    mk = (d > edges[i]) & (d <= edges[i + 1]); RES["by_nn_quartile"][f"Q{i + 1}"] = {k: {"n": int(mk.sum()), "overall": round(float((v[mk] == yp[mk]).mean()), 3),
        "false_alarm": round(float((v[mk & (yp == U)] != U).mean()), 3) if (mk & (yp == U)).sum() else None} for k, v in PRED.items()}
    print(f"\n-- nearest-photo distance quartile Q{i + 1} (Q1 = closest to a training photo), n={int(mk.sum())}, other-pose photos {int((mk & (yp == U)).sum())}", flush=True)
    for k, r in RES["by_nn_quartile"][f"Q{i + 1}"].items(): print(f"   {k:<40} overall {r['overall']:.3f} | false alarm {r['false_alarm']}", flush=True)
save("nn audit")

# ---------------------------------------------------------------- (3) frozen Commons-103, out-of-fold (a different, wild photo source)
z = np.load(f"{ROOT}/vol/eval/photo_corpus.npz", allow_pickle=True); lm, yc, sp = z["landmarks"], z["labels"].astype(str), z["split"].astype(str)
fold = np.array(json.load(open(f"{ROOT}/vol/folds/photo_folds.json"))["fold"]); Xc = np.array([T._angles(l, True) for l in lm], dtype=np.float32)
OOF = {k: np.empty(len(yc), dtype=object) for k in ZOO}
for i in range(5):
    idx = np.where(fold == i)[0]; o, *_ = fit_predict(f"{ROOT}/vol/csv/cue2/vH/mlp_fold{i}.csv", [Xc[idx]])
    for k in ZOO: OOF[k][idx] = o[k][0]
    print("commons fold", i, "done", flush=True)
t = sp == "test"; RES["commons103_oof"] = {k: metrics(v[t].astype(str), yc[t], 5) for k, v in OOF.items()}
print("\n===== FROZEN COMMONS-103, out-of-fold (min 5 photos per pose) =====", flush=True)
for k, m in RES["commons103_oof"].items(): print(line(k, m), flush=True)
RES["commons103_reference_from_earlier_run"] = {"v4 alone": {"overall": 0.602, "macro_recall": 0.648}, "new ResMLP out-of-fold": {"overall": 0.495, "macro_recall": 0.58}, "cascade": {"overall": 0.553}}
save("ALL DONE")
