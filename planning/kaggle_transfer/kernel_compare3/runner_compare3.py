"""Do the correctness + deviation heads respond to wrong form? Held-out in-vocabulary PUBLIC-TEST photos (never trained on by Jul-19, Sep-3 or NEW).
For each photo, break ONE joint angle by 45 deg (random joint, 5 trials) and measure: (a) correctness score drops, (b) deviation head's #1 joint is the broken one,
(c) predicted deviation at the broken joint rises. Chance for (b) is 1/15 = 6.7%. A sanity/property test -- NOT a labelled-form accuracy (no such labels exist)."""
import glob, json, os, shutil, sys
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import numpy as np, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/**", "vol/photos/public_corpus.npz"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueH_mlp_all/**"])
os.makedirs("/models_src", exist_ok=True)
for f in os.listdir(f"{ROOT}/code/models_src"): shutil.copy(f"{ROOT}/code/models_src/{f}", f"/models_src/{f}")
sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal"); sys.path.insert(0, "/models_src")
import asanaai_train2 as T
from mlp import Yoga3HeadMLP
z = np.load(f"{ROOT}/vol/photos/public_corpus.npz", allow_pickle=True)
te = (z["split"] == "test") & (z["labels"].astype(str) != "transition/unknown")
X = np.array([T._angles(l, True) for l in z["landmarks"][te]], dtype=np.float32)
print("in-vocabulary held-out photos:", len(X), flush=True)
MODELS = [("19 Jul original", hf_hub_download(LIVE, "mlp_3head_model.pth", token=HF), hf_hub_download(LIVE, "mlp_3head_pose_encoder.npy", token=HF)),
          ("3 Sep v2", hf_hub_download(LIVE, "mlp_3head_model_v2.pth", token=HF), hf_hub_download(LIVE, "mlp_3head_pose_encoder.npy", token=HF)),
          ("21 Sep v4 (live)", hf_hub_download(LIVE, "mlp_3head_v4_photos_x2000.pth", token=HF), hf_hub_download(LIVE, "mlp_3head_v4_encoder.npy", token=HF)),
          ("NEW (today)", f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_model_v2.pth", f"{ROOT}/vol/runs/cueH_mlp_all/mlp_3head_pose_encoder_v2.npy")]
rng = np.random.default_rng(0)
J = rng.integers(0, 15, size=(5, len(X)))
res = {}
for name, mp, ep in MODELS:
    cls = list(np.load(ep, allow_pickle=True))
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls)); m.load_state_dict(torch.load(mp, map_location="cpu")); m.eval()
    def heads(A):
        with torch.no_grad():
            _, c, d = m(torch.tensor(A, dtype=torch.float32))
        return torch.sigmoid(c).numpy(), np.clip(d.numpy() * 180.0, 0, 180)
    c0, d0 = heads(X)
    drop = hit = rise = n = 0
    for t in range(5):
        A = X.copy(); idx = np.arange(len(X))
        delta = np.where(A[idx, J[t]] <= 135, 45.0, -45.0)
        A[idx, J[t]] = np.clip(A[idx, J[t]] + delta, 0, 180)
        c1, d1 = heads(A)
        drop += int((c1 < c0).sum()); hit += int((d1.argmax(1) == J[t]).sum()); rise += int((d1[idx, J[t]] > d0[idx, J[t]]).sum()); n += len(X)
    res[name] = {"mean_correctness_good": round(float(c0.mean()), 3), "correctness_drops_when_form_broken": round(drop / n, 3),
                 "deviation_top1_is_broken_joint": round(hit / n, 3), "deviation_at_broken_joint_rises": round(rise / n, 3),
                 "chance_top1": round(1 / 15, 3)}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(f"{ROOT}/heads.json", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{ROOT}/heads.json", path_in_repo="evals_compare/heads_property_test.json", repo_id=MD, repo_type="model")
