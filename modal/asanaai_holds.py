"""AsanaAI non-vinyasa hold-pose data pipeline, on Modal.

WHY THIS EXISTS
---------------
Every one of the 12 videos behind the original 654,488-row CSV is a vinyasa
FLOW. Nobody in them holds anything: `pose_label` is 99.4% "transition/
unknown" and the real-pose rows are the few frames where a flow happens to
pass through a recognisable shape. That is why the deployed model knew ~3
poses in the real world while reporting 90.86% validation.

These 13 videos are single-pose HOLD tutorials. One video is (mostly) one
asana, held for seconds at a time, which buys three things the flows cannot:
  * labels that come from the video's identity instead of a classifier's guess
  * frames where the pose is actually static, which is what a user does
  * clean 60-frame windows for the ST-GCN that contain one pose, not a blur

EXTRACTION MUST MATCH THE ORIGINAL BIT-FOR-BIT
----------------------------------------------
Verified locally before writing this: recomputing the 15 angles from the
stored landmarks with `z` zeroed reproduces the live CSV to a max absolute
error of 2.8e-14, while leaving `z` raw is off by up to 69 degrees. So the
settings below are not "reasonable defaults", they are the measured ones:
MediaPipe Pose, static_image_mode=False, model_complexity=1, max_width=640,
min_detection/tracking_confidence=0.5, EVERY frame, undetected frames stored
as zeros, then occlusion interpolation at visibility>=0.5, then z:=0.
Deviating from any of these puts the new rows in a different feature space
than the 654k they are appended to -- which would not error, it would just
quietly produce a worthless model with a believable score.
"""
import json
import os

import modal

app = modal.App("asanaai-holds")

# numpy<2 because mediapipe 0.10.x is built against the 1.x ABI.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install(
        "yt-dlp",
        "mediapipe==0.10.14",
        "opencv-python-headless==4.10.0.84",
        "numpy<2",
        "pandas",
        "huggingface_hub",
        "scikit-learn",
    )
)

vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"

URLS = [
    "https://youtu.be/v7AYKMP6rOE", "https://youtu.be/ZiQh8jA5tVM",
    "https://youtu.be/O2EY79Ys_qg", "https://youtu.be/dAqQqmaI9vY",
    "https://youtu.be/4K2xTVRDJgA", "https://youtu.be/6CueZ4zujMk",
    "https://youtu.be/hHhxKkskHDg", "https://youtu.be/JHjV-wFTwSw",
    "https://youtu.be/EvMTrP8eRvM", "https://youtu.be/149Iac5fmoE",
    "https://youtu.be/Eml2xnoLpYE", "https://youtu.be/i6TzP2COtow",
    "https://youtu.be/4ZBUDd4bsyA",
]


@app.function(image=image, volumes={VOL: vol}, timeout=1800)
def probe():
    """Metadata only -- no video bytes. Answers two questions before any
    expensive work: can this IP talk to YouTube at all, and what pose is
    each video actually about (needed to assign labels)."""
    import subprocess

    out = []
    for u in URLS:
        try:
            r = subprocess.run(
                ["yt-dlp", "-J", "--no-warnings", "--skip-download", u],
                capture_output=True, text=True, timeout=180,
            )
            if r.returncode != 0:
                out.append({"url": u, "error": (r.stderr or "")[-400:]})
                continue
            j = json.loads(r.stdout)
            out.append({
                "url": u,
                "id": j.get("id"),
                "title": j.get("title"),
                "channel": j.get("channel"),
                "duration": j.get("duration"),
                "fps": j.get("fps"),
                "width": j.get("width"),
                "height": j.get("height"),
                "chapters": [
                    {"t": c.get("start_time"), "title": c.get("title")}
                    for c in (j.get("chapters") or [])
                ],
                "desc_head": (j.get("description") or "")[:600],
            })
        except Exception as e:
            out.append({"url": u, "error": f"{type(e).__name__}: {e}"[:400]})

    os.makedirs(f"{VOL}/meta", exist_ok=True)
    with open(f"{VOL}/meta/probe.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    res = probe.remote()
    ok = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    print(f"\n=== {len(ok)} reachable / {len(bad)} failed ===\n")
    for r in ok:
        print(f"[{r['id']}] {r['title']}")
        print(f"    channel={r['channel']}  {r['duration']}s  "
              f"{r['width']}x{r['height']} @{r['fps']}fps  "
              f"chapters={len(r['chapters'])}")
    for r in bad:
        print(f"FAILED {r['url']}: {r['error']}")
