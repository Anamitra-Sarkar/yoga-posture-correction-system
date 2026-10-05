"""Landmark + angle extraction for the 12 NEW videos, with the ORIGINAL scripts.

Runs the user's own `extract_landmarks.py` (subprocess, untouched, default
args = complexity 1, max_width 640, conf 0.5/0.5, every frame, zeros when
undetected) and `extract_features_safe.extract_features_from_landmarks`
(occlusion interpolation vis>=0.5, z:=0, 15 angles).

`verify` FIRST re-extracts an OLD video (oUgpXY7QhpQ) and compares against the
stored landmarks and the live CSV rows, because a different feature space would
not error -- it would just quietly train a worthless model with a believable
score.

Stages:  verify | fetch | extract
"""
import json
import os
import subprocess
import sys

import modal

SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
app = modal.App("asanaai-extract")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless==4.10.0.84",
                 "numpy<2", "pandas", "huggingface_hub")
    .add_local_file(f"{SRC}/extract_landmarks.py", "/opt/orig/extract_landmarks.py", copy=True)
    .add_local_file(f"{SRC}/extract_features_safe.py", "/opt/orig/extract_features_safe.py", copy=True)
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
secret = modal.Secret.from_name("arko007-hf-token")
RAW = "Arko007/yoga-dataset-raw"
NEW_DIR = "new_videos_2026-10"
NEW_IDS = ("v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk "
           "hHhxKkskHDg JHjV-wFTwSw EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow 4ZBUDd4bsyA").split()
ANGLES = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l",
          "knee_r", "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck",
          "hip_abduct_l", "hip_abduct_r"]


def _token():
    for k in ("HF_TOKEN_ARKO007", "HF_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def _run_extractor(video, out):
    subprocess.run([sys.executable, "/opt/orig/extract_landmarks.py",
                    "--video", video, "--output", out], check=True)


@app.function(image=image, volumes={VOL: vol}, secrets=[secret], cpu=4, memory=8192, timeout=3600)
def verify(ytid: str = "oUgpXY7QhpQ"):
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, "/opt/orig")
    from extract_features_safe import extract_features_from_landmarks

    vol.reload()
    vp = hf_hub_download(RAW, f"Yoga_Dataset_Raw/{ytid}.mp4", repo_type="dataset", token=_token())
    out = f"/tmp/landmarks_{ytid}.npy"
    _run_extractor(vp, out)
    new, old = np.load(out), np.load(f"{VOL}/landmarks/landmarks_{ytid}.npy")
    res = {"video": ytid, "shape_new": list(new.shape), "shape_old": list(old.shape)}
    n = min(len(new), len(old))
    d = np.abs(new[:n, :, :2] - old[:n, :, :2])
    res["landmark_median_abs_diff"] = float(np.median(d))
    res["landmark_frac_frames_maxdiff_lt_1e-3"] = float((d.max(axis=(1, 2)) < 1e-3).mean())
    a_new = extract_features_from_landmarks(out)[ANGLES].values[:n]
    a_old = extract_features_from_landmarks(f"{VOL}/landmarks/landmarks_{ytid}.npy")[ANGLES].values[:n]
    dd = np.abs(a_new - a_old)
    res["angle_new_vs_oldlandmarks_median"] = float(np.median(dd))
    res["angle_new_vs_oldlandmarks_p95"] = float(np.percentile(dd, 95))
    res["angle_frac_within_1deg"] = float((dd.max(axis=1) < 1.0).mean())
    csv = [f for f in os.listdir(f"{VOL}/csv") if f.endswith(".csv")][0]
    df = pd.read_csv(f"{VOL}/csv/{csv}", usecols=["video_id", "frame_num"] + ANGLES)
    v = df[df.video_id == ytid].sort_values("frame_num")
    m = min(len(v), n)
    dc = np.abs(a_new[:m] - v[ANGLES].values[:m])
    res["csv_rows"] = int(len(v))
    res["angle_new_vs_csv_median"] = float(np.median(dc))
    res["angle_new_vs_csv_p95"] = float(np.percentile(dc, 95))
    res["angle_frac_within_1deg_of_csv"] = float((dc.max(axis=1) < 1.0).mean())
    os.makedirs(f"{VOL}/runs", exist_ok=True)
    json.dump(res, open(f"{VOL}/runs/verify_{ytid}.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(res, indent=1))
    return res


@app.function(image=image, volumes={VOL: vol}, secrets=[secret], timeout=3600)
def fetch():
    """HF dataset folder -> Modal volume, entirely server-side."""
    import shutil
    from huggingface_hub import snapshot_download

    p = snapshot_download(RAW, repo_type="dataset", token=_token(), local_dir="/tmp/hf",
                          allow_patterns=[f"{NEW_DIR}/*"])
    os.makedirs(f"{VOL}/hold_videos", exist_ok=True)
    got = {}
    for f in sorted(os.listdir(f"{p}/{NEW_DIR}")):
        shutil.copy(f"{p}/{NEW_DIR}/{f}", f"{VOL}/hold_videos/{f}")
        got[f] = os.path.getsize(f"{VOL}/hold_videos/{f}")
    vol.commit()
    print(json.dumps(got, indent=1))
    return got


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=3 * 3600)
def extract_one(ytid: str):
    """Never raises: a failed video must not cancel the other eleven."""
    import traceback

    import numpy as np

    sys.path.insert(0, "/opt/orig")
    from extract_features_safe import extract_features_from_landmarks

    try:
        vol.reload()
        os.makedirs(f"{VOL}/landmarks_new", exist_ok=True)
        os.makedirs(f"{VOL}/new_angles", exist_ok=True)
        src = f"{VOL}/hold_videos/{ytid}.mp4"
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
                             "-of", "json", src], capture_output=True, text=True)
        info = json.loads(pr.stdout or "{}")
        codec = (info.get("streams") or [{}])[0].get("codec_name")
        print(ytid, "ffprobe:", json.dumps(info), flush=True)
        used = src
        if codec != "h264":  # OpenCV-headless cannot decode AV1/VP9 -> would give 0 frames
            used = f"/tmp/{ytid}_h264.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an", "-c:v", "libx264",
                            "-crf", "16", "-preset", "veryfast", "-pix_fmt", "yuv420p", used], check=True)
            print(ytid, f"transcoded {codec} -> h264", flush=True)
        lm = f"{VOL}/landmarks_new/landmarks_{ytid}.npy"
        _run_extractor(used, lm)
        arr = np.load(lm)
        if arr.ndim != 3 or len(arr) == 0:
            raise RuntimeError(f"extractor produced shape {arr.shape} (codec={codec})")
        df = extract_features_from_landmarks(lm)
        df.insert(0, "frame_num", np.arange(1, len(df) + 1))
        df.to_csv(f"{VOL}/new_angles/{ytid}.csv", index=False)
        vol.commit()
        det = float((arr[:, :, 3].max(axis=1) > 0).mean())
        return {"id": ytid, "ok": True, "frames": int(len(df)), "codec": codec, "frac_frames_detected": round(det, 4)}
    except Exception as e:  # noqa: BLE001
        return {"id": ytid, "ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-600:]}


@app.local_entrypoint()
def main(stage: str = "verify", only: str = ""):
    if stage == "verify":
        verify.remote()
    elif stage == "fetch":
        fetch.remote()
    elif stage == "extract":
        for r in extract_one.map(only.split() if only else NEW_IDS):
            print(r, flush=True)
    else:
        raise SystemExit("stage = verify|fetch|extract")
