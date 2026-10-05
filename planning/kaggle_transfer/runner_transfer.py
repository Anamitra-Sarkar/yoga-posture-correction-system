"""Kaggle CPU kernel: Modal volume -> private HF dataset (and models -> private HF model repo). Tokens come from the private input dataset asanaai-conf-creds (files modal_token_id, modal_token_secret, hf_token).
Transcripts are deliberately NOT uploaded (instructors' words are copyrighted). Idempotent: a prefix already present on HF is skipped. Uploads per path."""
import json, os, shutil, subprocess, sys, time
import glob
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"credential file {name} not found under /kaggle/input"
    return open(hits[0]).read().strip()
os.environ["MODAL_TOKEN_ID"] = cred("modal_token_id")
os.environ["MODAL_TOKEN_SECRET"] = cred("modal_token_secret")
HF = cred("hf_token")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], check=True)
from huggingface_hub import HfApi
api = HfApi(token=HF)
assert api.whoami()["name"] == "Arko007", "wrong HF account"
DS, MD = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs"
# (volume path, repo, repo_type, path_in_repo)  -- smallest / most valuable first
PATHS = [("cue", DS, "dataset", "vol/cue"),
         ("labelsrc", DS, "dataset", "vol/labelsrc"), ("eval", DS, "dataset", "vol/eval"), ("folds", DS, "dataset", "vol/folds"),
         ("csv/cue", DS, "dataset", "vol/csv/cue"), ("photos", DS, "dataset", "vol/photos"),
         ("runs", MD, "model", "runs"),
         ("csv/cue2", DS, "dataset", "vol/csv/cue2"), ("landmarks", DS, "dataset", "vol/landmarks"),
         ("landmarks_new", DS, "dataset", "vol/landmarks_new"),
         ("stgcn/cueT2_f0", DS, "dataset", "vol/stgcn/cueT2_f0"), ("stgcn/cueT2_f1", DS, "dataset", "vol/stgcn/cueT2_f1"),
         ("stgcn/cueT2_f2", DS, "dataset", "vol/stgcn/cueT2_f2"), ("stgcn/cueT2_all", DS, "dataset", "vol/stgcn/cueT2_all")]
report = []
for vp, repo, rt, dst in PATHS:
    t0 = time.time()
    try:
        have = [x for x in api.list_repo_tree(repo, path_in_repo=dst, repo_type=rt)] if True else []
    except Exception:
        have = []
    if have:
        report.append({"path": vp, "status": "already_on_hf"}); print("skip (already on HF):", vp, flush=True); continue
    tmp = f"/kaggle/working/stage/{vp.replace('/', '__')}"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    r = subprocess.run(["modal", "volume", "get", "asanaai-data", vp, tmp, "--force"], capture_output=True, text=True)
    if r.returncode != 0:
        report.append({"path": vp, "status": "modal_get_failed", "err": (r.stderr or r.stdout)[-300:]}); print("FAILED get", vp, r.stderr[-200:], flush=True); continue
    base = os.path.join(tmp, os.path.basename(vp))
    src = base if os.path.exists(base) else tmp
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(src) for f in fs) if os.path.isdir(src) else os.path.getsize(src)
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=repo, repo_type=rt, commit_message=f"transfer {vp}")
    else:
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type=rt, commit_message=f"transfer {vp}")
    shutil.rmtree(tmp, ignore_errors=True)
    report.append({"path": vp, "status": "ok", "mb": round(size / 1e6, 1), "secs": round(time.time() - t0)})
    print("OK", vp, f"{size/1e6:.0f} MB", f"{time.time()-t0:.0f}s", flush=True)
json.dump(report, open("/kaggle/working/transfer_report.json", "w"), indent=1)
api.upload_file(path_or_fileobj="/kaggle/working/transfer_report.json", path_in_repo="transfer_report.json", repo_id=DS, repo_type="dataset")
print(json.dumps(report, indent=1))
