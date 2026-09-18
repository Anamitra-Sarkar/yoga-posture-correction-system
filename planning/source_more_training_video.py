"""Expand the AsanaAI training corpus with genuinely DIVERSE yoga video.

WHY MORE VIDEO, AND WHY *DIVERSE* VIDEO
---------------------------------------
The existing corpus is 12 vinyasa-flow videos. Measured consequences:
  * real-world macro across the broader vocabulary is ~26%, with several
    classes at 0%;
  * landmark augmentation (mirroring / rotation / scale / noise) was tried and
    did NOT help (24.2% vs 25.8% macro) -- synthesising variation around a
    narrow corpus cannot manufacture diversity the corpus never had;
  * several classes are simply absent or near-absent from vinyasa flows
    (chair_pose, triangle, chaturanga, corpse, seated_forward).

So the lever is genuine coverage: different instructors, body types, camera
angles, indoor/outdoor, and crucially *pose-targeted* clips rather than more
of the same flow sequences.

SELECTION STRATEGY
------------------
Query per-pose and per-style rather than "yoga flow", so the corpus gains the
classes it is actually missing instead of more mountain_pose and lunge frames.

Runs in a GitHub Codespace. Downloads video there, extracts landmarks there,
and emits only compact .npy landmark arrays -- video never leaves the
Codespace and never touches the local machine.
"""
import os, json, glob, subprocess, argparse
import numpy as np

OUT = os.environ.get("OUT_DIR", "newvideo")
VID = os.path.join(OUT, "_video")

# Pose-targeted queries, weighted toward the classes the current corpus lacks.
QUERIES = [
    ("chair_pose",    "utkatasana chair pose yoga tutorial hold"),
    ("triangle",      "trikonasana triangle pose yoga tutorial"),
    ("chaturanga",    "chaturanga dandasana tutorial yoga"),
    ("corpse",        "savasana final relaxation yoga"),
    ("seated_forward","paschimottanasana seated forward bend tutorial"),
    ("warrior_1",     "virabhadrasana 1 warrior one pose tutorial"),
    ("tree_pose",     "vrksasana tree pose balance tutorial"),
    ("child_pose",    "balasana child pose yoga tutorial"),
    ("table_top",     "tabletop pose cat cow yoga tutorial"),
    ("plank",         "plank pose phalakasana yoga hold tutorial"),
    ("mixed_hatha",   "hatha yoga full class beginners 30 minutes"),
    ("mixed_iyengar", "iyengar yoga standing poses class"),
]


def have(cmd):
    return subprocess.run(["bash", "-lc", f"command -v {cmd}"],
                          capture_output=True).returncode == 0


def download(tag, query, n, maxres=480):
    """yt-dlp search. Caps resolution -- MediaPipe does not benefit from 1080p
    and it wastes Codespace disk/time."""
    os.makedirs(VID, exist_ok=True)
    cmd = [
        "yt-dlp", f"ytsearch{n}:{query}",
        "--match-filter", "duration > 120 & duration < 3600",
        "-f", f"bv*[height<={maxres}]+ba/b[height<={maxres}]",
        "--merge-output-format", "mp4",
        "-o", os.path.join(VID, f"{tag}__%(id)s.%(ext)s"),
        "--write-info-json", "--no-playlist", "--ignore-errors",
        "--no-warnings", "--quiet", "--progress",
    ]
    print(f"\n=== {tag}: {query!r} (n={n}) ===", flush=True)
    subprocess.run(cmd, check=False)


def extract(sample_fps=3, maxw=640):
    """MediaPipe Tasks API over sampled frames -> [N,33,4] per video.

    Matches how kaggle_process.py built the existing corpus, so new data is
    directly poolable with the old rather than being a second distribution.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import requests

    task = os.path.join(OUT, "pose_landmarker_heavy.task")
    if not os.path.exists(task):
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        open(task, "wb").write(requests.get(url, timeout=300).content)

    lm_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
        output_segmentation_masks=False)

    manifest = {}
    for path in sorted(glob.glob(os.path.join(VID, "*.mp4"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(OUT, f"landmarks_{stem}.npy")
        if os.path.exists(dst):
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / sample_fps)))
        rows, idx, ts = [], 0, 0
        with mp_vision.PoseLandmarker.create_from_options(lm_opts) as lmk:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    h, w = frame.shape[:2]
                    if w > maxw:
                        frame = cv2.resize(frame, (maxw, int(h * maxw / w)))
                    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ts += int(1000 / sample_fps)
                    r = lmk.detect_for_video(img, ts)
                    if r.pose_landmarks:
                        rows.append([[p.x, p.y, p.z, p.visibility]
                                     for p in r.pose_landmarks[0]])
                    else:
                        rows.append(np.zeros((33, 4), dtype=np.float32).tolist())
                idx += 1
        cap.release()
        if len(rows) >= 120:          # need at least a couple of 60-frame windows
            arr = np.array(rows, dtype=np.float32)
            np.save(dst, arr)
            manifest[stem] = {"frames": int(arr.shape[0]),
                              "detected": int((arr[:, :, 3].mean(1) > 0).sum())}
            print(f"  {stem[:58]:<58} {arr.shape[0]} frames", flush=True)
        os.remove(path)               # video never persists or leaves the box

    json.dump(manifest, open(os.path.join(OUT, "new_video_manifest.json"), "w"), indent=1)
    print(f"\nextracted {len(manifest)} videos", flush=True)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=3)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.skip_download:
        if not have("yt-dlp"):
            subprocess.run(["bash", "-lc", "pip install -q yt-dlp"], check=False)
        for tag, q in QUERIES:
            download(tag, q, a.per_query)
    extract()
