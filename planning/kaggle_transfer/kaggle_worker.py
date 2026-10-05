"""One GPU worker. Runs a list of ORIGINAL-trainer jobs through asanaai_train2's own wrappers, then uploads each
finished run to the private HF model repo IMMEDIATELY (a cancelled session must never again lose finished models)."""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--jobs", required=True)
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
# the original trainers `pip install --upgrade` torch/huggingface_hub on Kaggle at every start (and race when two start together);
# Kaggle's own torch works on the T4, so make pip a no-op for them (environment only, trainer code untouched)
os.environ["PIP_NO_INDEX"] = "1"
ROOT = "/kaggle/working"
sys.path.insert(0, f"{ROOT}/code/shim"); sys.path.insert(0, f"{ROOT}/code/modal")
import asanaai_train2 as T                      # noqa: E402  (imports the shim 'modal')
T.VOL = f"{ROOT}/vol"
T.HOME = f"/home/w{a.gpu}"
# the ORIGINAL trainers find their inputs by file name in the working directory, so link the data right there
T.DATA = T.HOME
os.makedirs(T.DATA, exist_ok=True)
from huggingface_hub import HfApi               # noqa: E402
api = HfApi(token=os.environ["HF_REAL"])
for job in json.loads(a.jobs):
    t0 = time.time()
    print(f"[gpu{a.gpu}] START {job}", flush=True)
    if job["kind"] == "stgcn":
        res = T.train_stgcn2(job["feats_tag"], job["tag"])
    else:
        res = T.train_mlp2(job["csv_rel"], job["tag"])
    print(f"[gpu{a.gpu}] DONE {job['tag']} in {(time.time()-t0)/60:.1f} min -> {res}", flush=True)
    d = f"{T.VOL}/runs/{job['tag']}"
    if os.path.isdir(d) and os.listdir(d):
        api.upload_folder(folder_path=d, path_in_repo=f"runs/{job['tag']}", repo_id="Arko007/asanaai-conference-runs",
                          repo_type="model", commit_message=f"{job['tag']}")
        print(f"[gpu{a.gpu}] uploaded {job['tag']} to HF", flush=True)
    else:
        print(f"[gpu{a.gpu}] ABORT: nothing saved for {job['tag']} -> stopping this worker to protect GPU quota", flush=True)
        sys.exit(3)
