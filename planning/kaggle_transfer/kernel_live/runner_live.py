"""Replay the held-out public photos against the LIVE Space (cascade ON) with the same request shape the web client sends."""
import glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
import numpy as np, requests
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"
snapshot_download("Arko007/Yoga-1M", repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/backend/app/utils/**", "code/backend/app/__init__.py", "vol/photos/public_corpus.npz"])
sys.path.insert(0, f"{ROOT}/code/backend")
from app.utils.geometry import extract_angles_from_landmarks, compute_orientation
URL = "https://arko007-yoga-pose.hf.space/api/analyse_frame"
p = np.load(f"{ROOT}/vol/photos/public_corpus.npz", allow_pickle=True); te = np.where(p["split"] == "test")[0]
y = p["labels"][te].astype(str)
FAILS = []
def call(j):
    pts = p["landmarks"][j][:, :3].astype(np.float64)
    ang = [float(a) for a in extract_angles_from_landmarks(pts, True)]
    body = {"angles": ang, "orientation": compute_orientation(pts)}
    last = None
    for _ in range(6):
        try:
            r = requests.post(URL, json=body, timeout=90)
            if r.status_code == 200: return r.json()
            last = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e: last = f"EXC {type(e).__name__}: {str(e)[:160]}"
        time.sleep(3)
    FAILS.append({"photo": int(j), "last_error": last, "has_nan_angle": any(a != a for a in ang), "angles_head": ang[:4]})
    return None
t0 = time.time()
with ThreadPoolExecutor(8) as ex: out = list(ex.map(call, te))
print("failed requests:", sum(o is None for o in out), "| wall seconds:", round(time.time() - t0))
for f in FAILS: print("FAIL", json.dumps(f))
pred = np.array([o["pose_id"] if o else "ERR" for o in out]); inv = y != "transition/unknown"
cl = [c for c in sorted(set(y[inv])) if (y == c).sum() >= 10]
ok = [str(c) for c in cl if (pred[y == c] == c).mean() >= .7 and (pred == c).sum() and (y[pred == c] == c).mean() >= .7]
print("cascade active in all responses:", all(o and o.get("cascade") and o["cascade"]["active"] for o in out))
print(f"LIVE SPACE, cascade ON, {len(y)} held-out public photos: overall {(pred == y).mean():.3f} | other-pose false alarm {(pred[~inv] != 'transition/unknown').mean():.3f} | poses passing 70/70: {len(ok)} {ok}")
