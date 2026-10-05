"""Feasibility probe for an offline CLIFF occlusion experiment: can we get the code, a checkpoint, the SMPL model and run a forward pass?"""
import glob, os, subprocess, sys, types, pickle, json
def sh(cmd, tail=25):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True); out = (r.stdout + r.stderr).strip().splitlines()
    print(f"$ {cmd}\n" + "\n".join(out[-tail:]), flush=True); return r.returncode
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
W = "/kaggle/working"; os.chdir(W)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv"); sh("python --version; pip list 2>/dev/null | grep -i -E '^(torch|torchvision|numpy|scipy|opencv|mediapipe|smplx|gdown|chumpy|trimesh) '")
# 1) code
sh("git clone --depth 1 --filter=blob:none --sparse https://github.com/huawei-noah/noah-research.git cl && cd cl && git sparse-checkout set CLIFF && ls CLIFF CLIFF/models CLIFF/common 2>&1 | head -40")
sh("cat cl/CLIFF/requirements.txt 2>/dev/null | head -30; sed -n 1,80p cl/CLIFF/demo.py", tail=120)
# 2) Google Drive folder listing (no download)
sh("pip install -q gdown 2>&1 | tail -1")
try:
    import gdown
    files = gdown.download_folder(url="https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP", skip_download=True, quiet=True, use_cookies=False)
    print("DRIVE FOLDER:", [(f.path, f.id) for f in files], flush=True)
except Exception as e:
    print("drive listing failed:", repr(e)[:300], flush=True)
