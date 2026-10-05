"""Deployed /analyse_sequence vs a local run of the PUBLISHED stgcn_target_v1 on held-out-fold test windows: must agree on label AND confidence."""
import glob, importlib.util, json, sys, time
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import numpy as np, requests, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/originals/**", "vol/stgcn/cueT2_f[012]/test_*.npy"])
spec = importlib.util.spec_from_file_location("stgcn_trainer", f"{ROOT}/code/originals/train_stgcn_gpu.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
cls = [str(c) for c in np.load(hf_hub_download(LIVE, "stgcn_target_v1_encoder.npy", token=HF), allow_pickle=True)]
m = tr.YogaSequenceLSTM(99, 128, 2, len(cls)); m.load_state_dict(torch.load(hf_hub_download(LIVE, "stgcn_target_v1.pth", token=HF), map_location="cpu")); m.eval()
X = np.concatenate([np.load(f"{ROOT}/vol/stgcn/cueT2_f{i}/test_feats.npy") for i in range(3)]); Y = np.concatenate([np.load(f"{ROOT}/vol/stgcn/cueT2_f{i}/test_labels.npy", allow_pickle=True) for i in range(3)]).astype(str)
sel = np.random.default_rng(5).choice(len(X), 60, replace=False); X, Y = X[sel], Y[sel]
with torch.no_grad(): P = torch.softmax(m(tr.SequenceDataset(tr.normalize_coordinate_sequence(X), np.zeros(len(X), dtype=np.int64)).X), 1).numpy()
agree, rows = 0, []
for w, p, y in zip(X, P, Y):
    r = None
    for _ in range(6):
        try:
            r = requests.post("https://arko007-yoga-pose.hf.space/api/analyse_sequence", json={"coordinates": w.tolist()}, timeout=90)
            if r.status_code == 200: break
            time.sleep(2)
        except Exception: time.sleep(2)
    if r is None or r.status_code != 200: continue
    j = r.json(); local = cls[int(p.argmax())]; same = j["sequence_pose"] == local and abs(j["confidence"] - float(p.max())) < 1e-3
    agree += same; rows.append({"truth": y, "local": local, "endpoint": j["sequence_pose"], "local_conf": round(float(p.max()), 3), "endpoint_conf": round(j["confidence"], 3), "fallback": j["requires_static_fallback"]})
print(f"PARITY deployed endpoint vs local stgcn_target_v1: {agree}/{len(rows)} identical (label and confidence)", flush=True)
print("endpoint label vocabulary seen:", sorted({r['endpoint'] for r in rows}), flush=True)
conf = [r for r in rows if not r["fallback"]]; print("confident (not falling back):", len(conf), "of", len(rows), "| correct among confident:", sum(r["truth"] == r["endpoint"] for r in conf), flush=True)
json.dump(rows, open(f"{ROOT}/e2e.json", "w"), indent=1); api.upload_file(path_or_fileobj=f"{ROOT}/e2e.json", path_in_repo="evals_stgcn/seq_endpoint_parity_target_v1.json", repo_id=MD, repo_type="model")
