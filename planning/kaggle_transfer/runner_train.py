PHASE = "stgcn"   # "stgcn" | "mlp"   (set by the pusher)
"""Kaggle 2xT4 kernel: ORIGINAL trainers (unmodified) via asanaai_train2's own wrappers, two GPUs in parallel,
each finished run uploaded to HF at once. Then the same held-out evaluations used all along."""
import json, os, shutil, subprocess, sys, threading
import glob
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"credential file {name} not found under /kaggle/input"
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=HF)
assert api.whoami()["name"] == "Arko007", "wrong HF account"
ROOT = "/kaggle/working"
DS, MD = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs"
need = {"stgcn": ["code/**", "vol/stgcn/cueT2_*/**", "vol/eval/**", "vol/folds/**"],
        "mlp": ["code/**", "vol/csv/cue2/vH/**", "vol/eval/**", "vol/folds/**", "vol/photos/public_corpus.npz"]}[PHASE]
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, allow_patterns=need, token=HF)
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", allow_patterns=["runs/cueT_mlp_*/**"], token=HF)
os.makedirs("/opt/orig", exist_ok=True); os.makedirs("/models_src", exist_ok=True)
for f in os.listdir(f"{ROOT}/code/originals"):
    shutil.copy(f"{ROOT}/code/originals/{f}", f"/opt/orig/{f}")
for f in os.listdir(f"{ROOT}/code/models_src"):
    shutil.copy(f"{ROOT}/code/models_src/{f}", f"/models_src/{f}")
print("data ready:", subprocess.run(["du", "-sh", f"{ROOT}/vol"], capture_output=True, text=True).stdout.strip(), flush=True)

jobs = {
 "stgcn": ([{"kind": "stgcn", "feats_tag": "cueT2_f0", "tag": "cueT2_stgcn_f0"}, {"kind": "stgcn", "feats_tag": "cueT2_f2", "tag": "cueT2_stgcn_f2"}],
           [{"kind": "stgcn", "feats_tag": "cueT2_f1", "tag": "cueT2_stgcn_f1"}, {"kind": "stgcn", "feats_tag": "cueT2_all", "tag": "cueT2_stgcn_all"}]),
 "mlp": ([{"kind": "mlp", "csv_rel": f"csv/cue2/vH/mlp_{n}.csv", "tag": f"cueH_mlp_{n}"} for n in ("fold0", "fold2", "fold4")],
         [{"kind": "mlp", "csv_rel": f"csv/cue2/vH/mlp_{n}.csv", "tag": f"cueH_mlp_{n}"} for n in ("fold1", "fold3", "all")]),
}[PHASE]
env = {**os.environ, "HF_REAL": HF, "PYTHONUNBUFFERED": "1"}
import collections, time
LIVE = collections.deque(maxlen=400)
def heartbeat():
    while True:
        time.sleep(600)
        try:
            open(f"{ROOT}/live_{PHASE}.txt", "w").write("\n".join(LIVE))
            api.upload_file(path_or_fileobj=f"{ROOT}/live_{PHASE}.txt", path_in_repo=f"logs/live_{PHASE}.txt", repo_id=MD, repo_type="model")
        except Exception as e:
            print("heartbeat failed:", e, flush=True)
threading.Thread(target=heartbeat, daemon=True).start()
procs = []
for gpu, js in zip(("0", "1"), jobs):
    p = subprocess.Popen([sys.executable, f"{ROOT}/code/kaggle_worker.py", "--gpu", gpu, "--jobs", json.dumps(js)],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    procs.append(p)
    def pump(p=p, g=gpu):
        for line in p.stdout:
            if line.startswith(("Epoch", "[gpu", "Using", "Unique", "exit", "saved", "Train size")) or "Error" in line or "Traceback" in line:
                LIVE.append(f"[w{g}] {line.rstrip()[:230]}")
                print(f"[w{g}] {line.rstrip()[:230]}", flush=True)
    threading.Thread(target=pump, daemon=True).start()
for p in procs:
    p.wait()
open(f"{ROOT}/live_{PHASE}.txt", "w").write("\n".join(LIVE))
try:
    api.upload_file(path_or_fileobj=f"{ROOT}/live_{PHASE}.txt", path_in_repo=f"logs/final_{PHASE}.txt", repo_id=MD, repo_type="model")
except Exception as e:
    print("final log upload failed:", e, flush=True)
print("workers finished:", [p.returncode for p in procs], flush=True)
if any(p.returncode for p in procs):
    print("ABORT: a worker failed, skipping evaluation", flush=True)
    sys.exit(1)

# ---- the same held-out evaluations used all along (asanaai_train2's functions, via the modal stand-in)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal")
import asanaai_train2 as T
T.VOL = f"{ROOT}/vol"
if PHASE == "stgcn":
    T.eval_stgcn_oof(prefix="cueT2_stgcn_f", k=3, name="stgcn_target_heldout_video", test_prefix="cueT2_f")
else:
    import numpy as np
    T.eval_oof(prefix="cueH_mlp_fold", k=5, name="mlpH_commons_oof")
    T.eval_public("cueH_mlp_all", name="mlpH_public")
    cls = sorted({c.replace("imperfect_", "") for c in np.load(f"{T.VOL}/runs/cueH_mlp_all/mlp_3head_pose_encoder_v2.npy", allow_pickle=True)} - {"transition/unknown"})
    T.calibrate("cueH_mlp_all", "cueH_mlp_fold", ",".join(cls))
ev = f"{T.VOL}/runs/_oof"
if os.path.isdir(ev):
    api.upload_folder(folder_path=ev, path_in_repo=f"evals_{PHASE}", repo_id=MD, repo_type="model", commit_message=f"evals {PHASE}")
    print("evals uploaded", flush=True)
