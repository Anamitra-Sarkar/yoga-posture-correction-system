"""Fair comparison on the FROZEN 103-photo Commons test split (hash-split 'test' rows) -- the only photos the 18/21 Sep photo-domain
models were NOT trained on. Every model is scored by predicting all 422, then restricting to split=='test'. NEW uses out-of-fold predictions."""
import glob, json, os, shutil, sys
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import numpy as np, torch, collections
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/**", "vol/eval/**", "vol/folds/**"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueH_mlp_fold*/**"])
os.makedirs("/models_src", exist_ok=True)
for f in os.listdir(f"{ROOT}/code/models_src"): shutil.copy(f"{ROOT}/code/models_src/{f}", f"/models_src/{f}")
sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal"); sys.path.insert(0, "/models_src")
import asanaai_train2 as T
from mlp import Yoga3HeadMLP
z = np.load(f"{ROOT}/vol/eval/photo_corpus.npz", allow_pickle=True)
lm, y, sp = z["landmarks"], z["labels"].astype(str), z["split"].astype(str)
fold = np.array(json.load(open(f"{ROOT}/vol/folds/photo_folds.json"))["fold"])
X = torch.tensor(np.array([T._angles(l, True) for l in lm], dtype=np.float32))
test = sp == "test"
print("frozen test photos:", int(test.sum()), "| classes with n>=5:", {c: int((y[test] == c).sum()) for c in sorted(set(y[test])) if (y[test] == c).sum() >= 5}, flush=True)
def load(model_path, enc_path):
    cls = [c.replace("imperfect_", "") for c in np.load(enc_path, allow_pickle=True)]
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls)); m.load_state_dict(torch.load(model_path, map_location="cpu")); m.eval()
    return m, cls
def predict(m, cls, Xs):
    with torch.no_grad(): return np.array([cls[i] for i in m(Xs)[0].argmax(1).numpy()])
OLD = [("19 Jul original 3-head", "mlp_3head_model.pth", "mlp_3head_pose_encoder.npy"),
       ("3 Sep v2", "mlp_3head_model_v2.pth", "mlp_3head_pose_encoder.npy"),
       ("18 Sep photo-domain v1", "mlp_3head_photodomain_v1.pth", "mlp_3head_photodomain_v1_encoder.npy"),
       ("21 Sep v4 (LIVE in the app)", "mlp_3head_v4_photos_x2000.pth", "mlp_3head_v4_encoder.npy")]
preds = {}
for label, mf, ef in OLD:
    m, cls = load(hf_hub_download(LIVE, mf, token=HF), hf_hub_download(LIVE, ef, token=HF)); preds[label] = predict(m, cls, X)
newp = np.empty(len(y), dtype=object)
for i in range(5):
    m, cls = load(f"{ROOT}/vol/runs/cueH_mlp_fold{i}/mlp_3head_model_v2.pth", f"{ROOT}/vol/runs/cueH_mlp_fold{i}/mlp_3head_pose_encoder_v2.npy")
    idx = np.where(fold == i)[0]; newp[idx] = predict(m, cls, X[idx])
preds["NEW (today, out-of-fold)"] = newp.astype(str)
print("\n=========== FROZEN 103-photo test split (no model below was trained on these) ===========")
rows = {}
for label, p in preds.items():
    yt, pt = y[test], p[test]
    cl = [c for c in sorted(set(yt)) if (yt == c).sum() >= 5]
    macro = float(np.mean([(pt[yt == c] == c).mean() for c in cl]))
    ok = [c for c in sorted(set(yt)) if (yt == c).sum() >= 10 and (pt[yt == c] == c).mean() >= .7 and (yt[pt == c] == c).mean() >= .7 if (pt == c).sum()]
    rows[label] = {"overall_acc": round(float((pt == yt).mean()), 3), "macro_recall_n>=5": round(macro, 3), "passing_n>=10": ok}
    print(f"{label:<32} overall {rows[label]['overall_acc']:.3f} | macro recall (classes n>=5) {macro:.3f} | pass(n>=10): {ok}")
print("\nper-class recall on the frozen test (n>=5):")
cl = [c for c in sorted(set(y[test])) if (y[test] == c).sum() >= 5]
print(f"{'pose':<22}{'n':>3} " + " ".join(f"{k[:12]:>13}" for k in preds))
for c in cl:
    t = test & (y == c)
    print(f"{c:<22}{int(t.sum()):>3} " + " ".join(f"{(p[t] == c).mean():>13.2f}" for p in preds.values()))
json.dump(rows, open(f"{ROOT}/compare2.json", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{ROOT}/compare2.json", path_in_repo="evals_compare/frozen103.json", repo_id=MD, repo_type="model")
