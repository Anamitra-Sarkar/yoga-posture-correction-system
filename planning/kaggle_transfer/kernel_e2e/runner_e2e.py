"""End-to-end production check on CPU: (1) the real pytest suite, (2) every held-out photo through the REAL /analyse_frame endpoint with the REAL
models, cascade OFF (= production today) and ON, (3) endpoint-vs-offline agreement, latency, memory."""
import glob, json, os, resource, subprocess, sys, time
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token"); os.environ["HF_TOKEN"] = HF
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "httpx", "pytest", "scipy"], check=False)
from huggingface_hub import HfApi, snapshot_download
import numpy as np
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs"
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/backend/**", "vol/eval/**", "vol/folds/**", "vol/photos/public_corpus.npz"])
BE = f"{ROOT}/code/backend"
# ---------- (1) pytest -- the whole suite, honestly reported
env = {**os.environ, "PYTHONPATH": BE}
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "tests/test_cascade.py"], cwd=BE, env=env, capture_output=True, text=True)
print("=== pytest tests/test_cascade.py ===\n" + r.stdout[-1800:] + r.stderr[-600:], flush=True)
r2 = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "tests"], cwd=BE, env=env, capture_output=True, text=True)
print("=== pytest FULL SUITE ===\n" + r2.stdout[-2500:] + r2.stderr[-500:], flush=True)
# ---------- (2) real endpoint, real models
os.environ.update(ENABLE_POSE_CASCADE="1", GATE_REPO=MD, GATE_MODEL_FILE="runs/cueH_mlp_all/mlp_3head_model_v2.pth",
                  GATE_ENCODER_FILE="runs/cueH_mlp_all/mlp_3head_pose_encoder_v2.npy")
sys.path.insert(0, BE); os.chdir(BE)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.config import settings
from app.routers import pose as P
from app.services import hf_loader
from app.services.cascade import UNKNOWN, collapse
from app.utils.geometry import extract_angles_from_landmarks, compute_orientation
from app.utils.rules_classifier import sanitize_pose
hf_loader.initialize_models()
print("memory after loading live models (MB):", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
app = FastAPI(); app.include_router(P.router, prefix="/api"); client = TestClient(app)
import torch
def payload(lm, world, target=None):
    pts = lm[:, :3].astype(np.float64)
    d = {"angles": [float(a) for a in extract_angles_from_landmarks(pts, True)], "orientation": compute_orientation(pts)}
    if world is not None: d["world_angles"] = [float(a) for a in extract_angles_from_landmarks(world.astype(np.float64), False)]
    if target: d["target_pose"] = target
    return d
def expected_h4(angles):
    m, cls = hf_loader.get_mlp_model(); g, gc = hf_loader.get_gate_model()
    x = torch.tensor([angles], dtype=torch.float32)
    with torch.no_grad():
        vp = cls[m(x)[0].argmax(1).item()]; gp = torch.softmax(g(x)[0], 1)[0].numpy()
    gd = collapse({c: float(p) for c, p in zip(gc, gp)}); return UNKNOWN if max(gd, key=gd.get) == UNKNOWN else sanitize_pose(vp)
def run_set(name, lms, worlds, ys, min_n):
    res, lat = {}, []
    for mode in ("OFF (production today)", "ON (cascade)"):
        settings.ENABLE_POSE_CASCADE = mode.startswith("ON")
        preds, agree = [], 0
        for i in range(len(lms)):
            pl = payload(lms[i], None if worlds is None else worlds[i])
            t = time.perf_counter(); r = client.post("/api/analyse_frame", json=pl); lat.append((time.perf_counter() - t) * 1000)
            assert r.status_code == 200, r.text
            body = r.json(); preds.append(body["pose_id"])
            if mode.startswith("ON"):
                assert body["cascade"]["active"], body["cascade"]
                agree += int(body["pose_id"] == expected_h4(pl["angles"]))
        p = np.array(preds); inv = ys != UNKNOWN
        cl = [c for c in sorted(set(ys[inv])) if (ys == c).sum() >= min_n]
        ok = [str(c) for c in cl if (p[ys == c] == c).mean() >= .7 and (p == c).sum() and (ys[p == c] == c).mean() >= .7]
        res[mode] = {"overall": round(float((p == ys).mean()), 3), "macro_recall": round(float(np.mean([(p[ys == c] == c).mean() for c in cl])), 3),
                     "pass": ok, "false_alarm": round(float((p[~inv] != UNKNOWN).mean()), 3) if (~inv).sum() else None,
                     "endpoint_vs_offline_H4_agreement": (agree / len(lms)) if mode.startswith("ON") else None}
        print(f"[{name}] {mode}: {res[mode]}", flush=True)
    res["latency_ms"] = {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1), "max": round(float(max(lat)), 1), "n_requests": len(lat)}
    print(f"[{name}] latency per /analyse_frame request, in-process CPU: {res['latency_ms']}", flush=True); return res
out = {}
z = np.load(f"{ROOT}/vol/eval/photo_corpus.npz", allow_pickle=True); t = z["split"] == "test"
out["frozen103"] = run_set("frozen-103 Commons (3-way vote path)", z["landmarks"][t], z["world"][t], z["labels"][t].astype(str), 5)
p = np.load(f"{ROOT}/vol/photos/public_corpus.npz", allow_pickle=True); te = p["split"] == "test"
out["public_heldout"] = run_set("public held-out", p["landmarks"][te], None, p["labels"][te].astype(str), 10)
# guided + legacy-request sanity on real models
r = client.post("/api/analyse_frame", json={**payload(z["landmarks"][t][0], None), "target_pose": "warrior_2"}).json()
print("guided block:", {k: r["guided"][k] for k in ("target_pose", "matches", "target_correctness")}, flush=True)
print("peak RSS MB:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
json.dump({"pytest_cascade": r.__class__.__name__ and "see log", "e2e": out}, open(f"{ROOT}/e2e.json", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{ROOT}/e2e.json", path_in_repo="evals_compare/e2e_endpoint.json", repo_id=MD, repo_type="model")
