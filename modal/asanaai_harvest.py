"""Commons + Openverse real-photo harvest on Modal, one container per pose class (the original script is sequential
and rate-limited; separate containers parallelise it). Uses the ORIGINAL harvest_photo_corpus_v2.py unmodified:
its query tables and helper functions are imported by exec'ing everything above its trailing `main()` call.
Labels here are SEARCH-QUERY labels (weak) -> they are only used after the geometry-consistency filter in assembly.
"""
import hashlib
import json
import os
import time

import modal

PL = "/home/anamitra/yoga_posture_workspace/planning"
app = modal.App("asanaai-harvest")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless==4.10.0.84", "numpy<2", "requests")
    .add_local_file(f"{PL}/harvest_photo_corpus_v2.py", "/opt/orig/harvest_photo_corpus_v2.py", copy=True)
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
CLASSES = ["mountain_pose", "child_pose", "standing_forward_fold", "cobra_pose", "lunge_pose", "plank", "corpse",
           "seated_staff", "seated_easy_pose", "warrior_1", "chair_pose", "triangle", "downward_dog", "warrior_2",
           "tree_pose", "table_top", "upward_dog", "upward_salute", "seated_forward", "halfway_lift", "chaturanga"]
EXTRA_PRIORITY = {"mountain_pose", "lunge_pose", "corpse", "triangle", "downward_dog", "seated_staff"}


@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=8192, timeout=3 * 3600, max_containers=21)
def harvest_class(label: str, per_pose: int = 150, priority_per_pose: int = 400, min_vis: float = 0.55):
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    os.environ["OUT_DIR"] = "/tmp/harvest"
    os.makedirs("/tmp/harvest", exist_ok=True)
    src = open("/opt/orig/harvest_photo_corpus_v2.py").read().rsplit("\nmain()", 1)[0]
    ns = {"__name__": "harvest_v2"}
    exec(compile(src, "harvest_photo_corpus_v2.py", "exec"), ns)
    ns["PRIORITY"] |= EXTRA_PRIORITY
    import requests
    task = "/tmp/harvest/pose_landmarker_heavy.task"
    open(task, "wb").write(requests.get(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/"
        "pose_landmarker_heavy.task", timeout=300).content)
    lmk = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task), running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1, output_segmentation_masks=False))
    budget = priority_per_pose if label in ns["PRIORITY"] else per_pose
    cands = ns["gather"](label, ns["QUERIES"][label], budget)
    sess, seen = ns["sess"], set()
    L, K, META, rej = [], [], [], {"no_person": 0, "low_vis": 0, "fetch": 0}
    for c in cands:
        if len(L) >= budget:
            break
        ck = ns["canonical_key"](c["url"])
        if ck in seen:
            continue
        seen.add(ck)
        try:
            r = sess.get(c["url"], timeout=45)
            if r.status_code == 429:
                time.sleep(8)
                continue
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                rej["fetch"] += 1
                continue
            key = hashlib.sha1(np.ascontiguousarray(img)).hexdigest()
            h, w = img.shape[:2]
            if max(h, w) > 1400:
                s = 1400.0 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.pose_landmarks:
                rej["no_person"] += 1
                continue
            lm = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]], dtype=np.float32)
            if float(lm[[11, 12, 23, 24, 25, 26, 27, 28], 3].mean()) < min_vis:
                rej["low_vis"] += 1
                continue
            L.append(lm); K.append(key)
            META.append({"id": ns["stable_id"](c), "source": c["source"], "license": c["license"],
                         "split": ns["split_of"](ns["stable_id"](c))})
        except Exception:
            rej["fetch"] += 1
        time.sleep(0.9)
    os.makedirs(f"{VOL}/photos/harvest", exist_ok=True)
    np.savez_compressed(f"{VOL}/photos/harvest/{label}.npz", landmarks=np.array(L, dtype=np.float32).reshape(-1, 33, 4),
                        keys=np.array(K), label=label)
    json.dump(META, open(f"{VOL}/photos/harvest/{label}_meta.json", "w"))
    vol.commit()
    return {"label": label, "candidates": len(cands), "kept": len(L), "rejected": rej}


@app.local_entrypoint()
def main():
    for r in harvest_class.map(CLASSES):
        print(r, flush=True)
