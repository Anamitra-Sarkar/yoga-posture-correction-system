"""Old vs new 3-head MLP on IDENTICAL held-out photos. CPU only. Reads models from HF (read-only) and the new runs from the private model repo."""
import glob, json, os, shutil, sys
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import numpy as np, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF,
                  allow_patterns=["code/**", "vol/eval/**", "vol/folds/**", "vol/photos/public_corpus.npz"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueH_mlp_*/**"])
os.makedirs("/models_src", exist_ok=True)
for f in os.listdir(f"{ROOT}/code/models_src"): shutil.copy(f"{ROOT}/code/models_src/{f}", f"/models_src/{f}")
sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal"); sys.path.insert(0, "/models_src")
import asanaai_train2 as T
T.VOL = f"{ROOT}/vol"
from mlp import Yoga3HeadMLP
# (tag, model file, encoder file, label) -- pairings per the repo README; dims are verified below, never assumed
OLD = [("old_0719_mlp_3head_model", "mlp_3head_model.pth", "mlp_3head_pose_encoder.npy", "19 Jul: original 3-head"),
       ("old_0903_mlp_3head_model_v2", "mlp_3head_model_v2.pth", "mlp_3head_pose_encoder.npy", "3 Sep: retrain v2"),
       ("old_0918_photodomain_v1", "mlp_3head_photodomain_v1.pth", "mlp_3head_photodomain_v1_encoder.npy", "18 Sep: photo-domain v1"),
       ("old_0921_v4_LIVE", "mlp_3head_v4_photos_x2000.pth", "mlp_3head_v4_encoder.npy", "21 Sep: v4 (live in the app)")]
enc_a = np.load(hf_hub_download(LIVE, "mlp_3head_pose_encoder.npy", token=HF), allow_pickle=True)
enc_b = np.load(hf_hub_download(LIVE, "mlp_3head_pose_encoder_v2.npy", token=HF), allow_pickle=True)
print("encoders mlp_3head_pose_encoder.npy vs _v2.npy identical content+order:", list(enc_a) == list(enc_b), "| n =", len(enc_a), flush=True)
rows = []
for tag, mf, ef, label in OLD:
    d = f"{T.VOL}/runs/{tag}"; os.makedirs(d, exist_ok=True)
    shutil.copy(hf_hub_download(LIVE, mf, token=HF), f"{d}/mlp_3head_model_v2.pth")
    shutil.copy(hf_hub_download(LIVE, ef, token=HF), f"{d}/mlp_3head_pose_encoder_v2.npy")
    sd = torch.load(f"{d}/mlp_3head_model_v2.pth", map_location="cpu")
    head = [v.shape[0] for k, v in sd.items() if k.startswith("pose_head") and k.endswith("weight") and v.ndim == 2][-1]
    n_enc = len(np.load(f"{d}/mlp_3head_pose_encoder_v2.npy", allow_pickle=True))
    print(f"{tag}: pose-head outputs={head} encoder classes={n_enc} -> {'OK' if head == n_enc else 'MISMATCH (excluded)'}", flush=True)
    if head != n_enc: continue
    c = T.eval_photos(tag)["zero_z(app)"]; p = T.eval_public(tag, name=tag)
    rows.append((label, tag, c["overall_acc"], c["n_passing"], c["passing_poses"], p["n_passing"], p["passing_poses"], p["oov_false_alarm_rate"]))
o = T.eval_oof(prefix="cueH_mlp_fold", k=5, name="NEW_cueH_commons_oof")
p = T.eval_public("cueH_mlp_all", name="NEW_cueH_public")
rows.append(("NEW (today): 24 videos, cue labels, photos, mirror", "cueH_mlp_all", o["overall_acc"], o["n_passing"], o["passing_poses"], p["n_passing"], p["passing_poses"], p["oov_false_alarm_rate"]))
print("\n================ SUMMARY (same held-out photos for every model) ================")
for r in rows:
    print(f"{r[0]}\n   Commons-422 overall acc {r[2]:.3f} | poses passing 70/70: {r[3]} {r[4]}\n   Public held-out ({1685} photos) poses passing: {r[5]} {r[6]} | other-pose false alarms {r[7]}")
json.dump(rows, open(f"{ROOT}/compare_summary.json", "w"), indent=1, default=str)
api.upload_file(path_or_fileobj=f"{ROOT}/compare_summary.json", path_in_repo="evals_compare/compare_summary.json", repo_id=MD, repo_type="model")
ev = f"{T.VOL}/runs/_oof"
if os.path.isdir(ev): api.upload_folder(folder_path=ev, path_in_repo="evals_compare/detail", repo_id=MD, repo_type="model")
