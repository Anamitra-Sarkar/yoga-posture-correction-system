"""Replay the frozen 103 Commons test photos through the LIVE Space /api/analyse_frame, exactly as the web client sends them
(zero-z angles, world angles, orientation). Compare with the endpoint test run on Kaggle (OFF=40.8%, ON=60.2% overall on this set)."""
import json, sys, time
import numpy as np, requests
sys.path.insert(0, "backend")
from app.utils.geometry import extract_angles_from_landmarks, compute_orientation
URL = sys.argv[1].rstrip("/") + "/api/analyse_frame"
z = np.load("planning/photo_corpus/photo_corpus.npz", allow_pickle=True)
t = np.where(z["split"] == "test")[0]
y = z["labels"][t].astype(str); preds, lat, casc = [], [], []
for k, i in enumerate(t):
    pts = z["landmarks"][i][:, :3].astype(np.float64)
    body = {"angles": [float(a) for a in extract_angles_from_landmarks(pts, True)], "orientation": compute_orientation(pts),
            "world_angles": [float(a) for a in extract_angles_from_landmarks(z["world"][i].astype(np.float64), False)]}
    t0 = time.perf_counter()
    for attempt in range(6):
        r = requests.post(URL, json=body, timeout=90)
        if r.status_code == 200: break
        time.sleep(5)
    lat.append((time.perf_counter() - t0) * 1000); assert r.status_code == 200, (r.status_code, r.text[:200])
    j = r.json(); preds.append(j["pose_id"]); casc.append(j.get("cascade"))
p = np.array(preds)
print(f"requests: {len(p)} | overall accuracy vs labels: {(p == y).mean():.3f}")
print("cascade block in responses:", casc[1] if casc[1] else None)
print("latency over the internet, ms: p50 %.0f  p95 %.0f  (first request, incl. any cold start: %.0f)" % (np.percentile(lat[1:], 50), np.percentile(lat[1:], 95), lat[0]))
print("unknown share:", round(float((p == 'transition/unknown').mean()), 3))
