"""Does the live CSV equal angles(stored landmarks) for each of the 12 old videos?
(verify showed re-extraction vs CSV = 14 deg median for oUgpXY7QhpQ; find out whether
the CSV and the volume's landmark files agree with EACH OTHER before blaming extraction.)"""
import json, os, sys
import modal

SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
app = modal.App("asanaai-diag")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy<2", "pandas")
         .add_local_file(f"{SRC}/extract_features_safe.py", "/opt/orig/extract_features_safe.py", copy=True))
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
ANGLES = ["elbow_l","elbow_r","shoulder_l","shoulder_r","hip_l","hip_r","knee_l","knee_r",
          "ankle_l","ankle_r","trunk_l","trunk_r","neck","hip_abduct_l","hip_abduct_r"]

@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=12288, timeout=3000)
def diag():
    import numpy as np, pandas as pd
    sys.path.insert(0, "/opt/orig")
    from extract_features_safe import extract_features_from_landmarks
    vol.reload()
    csv = [f for f in os.listdir(f"{VOL}/csv") if f.endswith(".csv")][0]
    df = pd.read_csv(f"{VOL}/csv/{csv}", usecols=["video_id", "frame_num"] + ANGLES)
    print("csv", csv, df.shape, "videos:", df.video_id.nunique())
    out = {}
    for f in sorted(os.listdir(f"{VOL}/landmarks")):
        yid = f.replace("landmarks_", "").replace(".npy", "")
        v = df[df.video_id == yid].sort_values("frame_num")
        a = extract_features_from_landmarks(f"{VOL}/landmarks/{f}")[ANGLES].values
        n = min(len(a), len(v))
        d = np.abs(a[:n] - v[ANGLES].values[:n]) if n else np.array([[np.nan]])
        r = {"lm_frames": int(len(a)), "csv_rows": int(len(v)), "median": float(np.nanmedian(d)),
             "max": float(np.nanmax(d)), "frames_fnum_contiguous": bool((np.diff(v.frame_num.values) == 1).all()) if len(v) > 1 else None,
             "frame_num_min_max": [int(v.frame_num.min()), int(v.frame_num.max())] if len(v) else None}
        # also try aligning by frame_num explicitly (in case rows are not 1..N in order)
        if len(v):
            idx = (v.frame_num.values - 1).clip(0, len(a) - 1)
            r["median_by_frame_num"] = float(np.median(np.abs(a[idx] - v[ANGLES].values)))
        out[yid] = r
        print(yid, r, flush=True)
    json.dump(out, open(f"{VOL}/runs/diag_csv_vs_landmarks.json", "w"), indent=1)
    vol.commit()

@app.local_entrypoint()
def main():
    diag.remote()
