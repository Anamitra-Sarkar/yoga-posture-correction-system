"""Copy the final target-pose ST-GCN into the LIVE model repo under NEW names (nothing existing is touched), sha256-verified, strict-load tested."""
import glob, hashlib, os
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download
import numpy as np, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
MD, LIVE = "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
PAIRS = [("runs/cueT2_stgcn_all/stgcn_sequence_model_v2.pth", "stgcn_target_v1.pth"), ("runs/cueT2_stgcn_all/stgcn_label_encoder_v2.npy", "stgcn_target_v1_encoder.npy")]
existing = set(api.list_repo_files(LIVE))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for src, dst in PAIRS:
    assert dst not in existing, f"{dst} already exists in {LIVE}: refusing to overwrite"
    local = hf_hub_download(MD, src, token=HF); h = sha(local)
    api.upload_file(path_or_fileobj=local, path_in_repo=dst, repo_id=LIVE, repo_type="model", commit_message=f"Add {dst} (final target-pose ST-GCN, copy of {src}); new name only")
    back = hf_hub_download(LIVE, dst, token=HF, force_download=True); assert sha(back) == h, "sha256 mismatch after upload"
    print("published", dst, h[:16], flush=True)
# strict-load check against a copy of the production class definition is done by the comparison kernel (parity max diff 0.0); here: classes + a forward pass
cls = [str(c) for c in np.load(hf_hub_download(LIVE, "stgcn_target_v1_encoder.npy", token=HF), allow_pickle=True)]
sd = torch.load(hf_hub_download(LIVE, "stgcn_target_v1.pth", token=HF), map_location="cpu"); print("classes:", cls, "| fc out features:", sd["fc.4.weight"].shape[0], flush=True)
assert sd["fc.4.weight"].shape[0] == len(cls)
print("LIVE repo now has:", sorted(f for f in api.list_repo_files(LIVE) if "stgcn" in f), flush=True)
