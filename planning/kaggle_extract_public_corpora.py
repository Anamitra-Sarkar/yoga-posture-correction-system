"""Extract MediaPipe joints from the large public yoga photo datasets.

WHY THIS, RATHER THAN MORE FRAMES
---------------------------------
The existing video corpus is 654,488 frames, but they come from only 12
videos. Frames are densely sampled, so consecutive frames of a held pose are
near-duplicates: the independent unit is closer to (video x pose), of which
there are 26. Adding more frames from those same 12 videos adds almost no
information, which is exactly why 448 real photographs beat 654,488 frames
(47.5% vs 38.9% macro on the held-out photo set).

What is actually missing is INDEPENDENT sources: different bodies, camera
heights, focal lengths, indoor/outdoor, and non-idealised form. Several large
public datasets provide precisely that, already organised by Sanskrit pose
name, and they mount on Kaggle so nothing is downloaded locally.

WHAT IT DOES
------------
Walks every mounted dataset, runs MediaPipe Pose on each image, computes the
same 15 zero-z angle features the production path uses, maps folder names onto
the project's 23-class vocabulary, and writes one corpus.

DEDUPLICATION IS NOT OPTIONAL HERE
----------------------------------
These datasets overlap heavily -- shrutisaxena's "1. 1.png" and
tr1gg3rtrash's "File1.png" are byte-identical (157390 bytes each). Without
deduplication the same photograph would be counted several times, inflating
apparent volume and, far worse, landing in both the train and test split.
Images are therefore keyed by a SHA1 of their decoded pixel content, so the
same photo is recognised across datasets regardless of filename.

The train/test split uses the same deterministic hash the existing photo
corpus uses, so the frozen benchmark stays comparable.
"""
import os
import sys
import glob
import json
import math
import time
import hashlib
import subprocess
import collections

import numpy as np


# Kaggle's CPU image has no mediapipe. --no-deps keeps the stock protobuf:
# letting mediapipe pin its own downgrades protobuf below what TensorFlow
# needs, which broke earlier runs on this project.
def _ensure_mediapipe():
    try:
        import mediapipe  # noqa: F401
        return
    except ImportError:
        pass
    for spec in ("mediapipe==0.10.14", "mediapipe"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", spec],
                       check=False)
        try:
            import mediapipe  # noqa: F401
            print(f"installed {spec}", flush=True)
            return
        except ImportError:
            continue
    raise SystemExit("FATAL: could not import mediapipe")


_ensure_mediapipe()

OUT = "/kaggle/working"
MIN_VIS = 0.55

# Folder-name aliases -> the project's 23-class vocabulary.
# Matched LONGEST-FIRST: "chaturanga dandasana" must win over "dandasana",
# and "virabhadrasana iii" (warrior 3, not in our vocabulary) must be rejected
# before "virabhadrasana i" can match it.
ALIASES = [
    ("chaturanga dandasana", "chaturanga"),
    ("four limbed staff", "chaturanga"),
    ("adho mukha svanasana", "downward_dog"),
    ("downward facing dog", "downward_dog"),
    ("downward dog", "downward_dog"),
    ("urdhva mukha svanasana", "upward_dog"),
    ("upward facing dog", "upward_dog"),
    ("upward dog", "upward_dog"),
    ("urdhva hastasana", "upward_salute"),
    ("upward salute", "upward_salute"),
    ("ardha uttanasana", "halfway_lift"),
    ("half forward bend", "halfway_lift"),
    ("standing forward bend", "standing_forward_fold"),
    ("uttanasana", "standing_forward_fold"),
    ("paschimottanasana", "seated_forward"),
    ("seated forward bend", "seated_forward"),
    ("utthita trikonasana", "triangle"),
    ("extended triangle", "triangle"),
    ("trikonasana", "triangle"),
    ("virabhadrasana i i", "warrior_2"),
    ("virabhadrasana ii", "warrior_2"),
    ("virabhadrasana 2", "warrior_2"),
    ("warrior ii", "warrior_2"),
    ("warrior 2", "warrior_2"),
    ("virabhadrasana i", "warrior_1"),
    ("virabhadrasana 1", "warrior_1"),
    ("warrior i", "warrior_1"),
    ("warrior 1", "warrior_1"),
    ("utkatasana", "chair_pose"),
    ("chair pose", "chair_pose"),
    ("bhujangasana", "cobra_pose"),
    ("cobra", "cobra_pose"),
    ("balasana", "child_pose"),
    ("child", "child_pose"),
    ("savasana", "corpse"),
    ("shavasana", "corpse"),
    ("corpse", "corpse"),
    ("phalakasana", "plank"),
    ("kumbhakasana", "plank"),
    ("plank", "plank"),
    ("sukhasana", "seated_easy_pose"),
    ("easy pose", "seated_easy_pose"),
    ("padmasana", "seated_easy_pose"),
    ("lotus", "seated_easy_pose"),
    ("dandasana", "seated_staff"),
    ("staff pose", "seated_staff"),
    ("tadasana", "mountain_pose"),
    ("mountain pose", "mountain_pose"),
    ("vrksasana", "tree_pose"),
    ("vrikshasana", "tree_pose"),
    ("tree pose", "tree_pose"),
    ("anjaneyasana", "lunge_pose"),
    ("low lunge", "lunge_pose"),
    ("crescent", "lunge_pose"),
    ("lunge", "lunge_pose"),
    ("bharmanasana", "table_top"),
    ("marjaryasana", "table_top"),
    ("bitilasana", "table_top"),
    ("cat cow", "table_top"),
    ("table top", "table_top"),
]
# Poses that exist in these datasets but NOT in our vocabulary. Listed
# explicitly so they are skipped rather than silently mis-mapped by a
# substring (warrior 3 must not become warrior_1).
REJECT = ["virabhadrasana iii", "virabhadrasana 3", "warrior iii", "warrior 3",
          "ardha chandrasana", "half moon"]

ALIASES.sort(key=lambda kv: -len(kv[0]))


def map_label(folder: str):
    f = folder.lower().replace("_", " ").replace("-", " ")
    f = " ".join(f.split())
    for bad in REJECT:
        if bad in f:
            return None
    for alias, target in ALIASES:
        if alias in f:
            return target
    return None


NOSE = 0
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
HP_L, HP_R, KN_L, KN_R, AN_L, AN_R, HE_L, HE_R = 23, 24, 25, 26, 27, 28, 29, 30


def angles_from_landmarks(pts):
    p = np.asarray(pts, dtype=np.float64)[:, :3].copy()
    p[:, 2] = 0.0

    def ang(a, b, c):
        ba, bc = p[a] - p[b], p[c] - p[b]
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nb == 0 or nc == 0:
            return 180.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ba, bc) / (nb * nc))))))

    sm = (p[SH_L] + p[SH_R]) / 2.0
    hm = (p[HP_L] + p[HP_R]) / 2.0
    ba, bc = p[NOSE] - sm, hm - sm
    nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    neck = 180.0 if (nb == 0 or nc == 0) else math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(ba, bc) / (nb * nc))))))
    return [ang(SH_L, EL_L, WR_L), ang(SH_R, EL_R, WR_R), ang(HP_L, SH_L, EL_L), ang(HP_R, SH_R, EL_R),
            ang(SH_L, HP_L, KN_L), ang(SH_R, HP_R, KN_R), ang(HP_L, KN_L, AN_L), ang(HP_R, KN_R, AN_R),
            ang(KN_L, AN_L, HE_L), ang(KN_R, AN_R, HE_R), ang(SH_L, HP_L, HP_R), ang(SH_R, HP_R, HP_L),
            neck, ang(HP_R, HP_L, KN_L), ang(HP_L, HP_R, KN_R)]


def split_of(stable_id, test_frac=0.25):
    """Identical to the existing photo corpus, so splits stay comparable."""
    h = int(hashlib.sha1(stable_id.encode()).hexdigest()[:8], 16)
    return "test" if (h % 100) < int(test_frac * 100) else "train"


def main():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import requests

    print("=== mounted inputs ===", flush=True)
    for d in sorted(glob.glob("/kaggle/input/*")):
        print(" ", d, flush=True)

    task = os.path.join(OUT, "pose_landmarker_heavy.task")
    if not os.path.exists(task):
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        open(task, "wb").write(requests.get(url, timeout=300).content)
    lmk = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task),
            running_mode=mp_vision.RunningMode.IMAGE, num_poses=1,
            output_segmentation_masks=False))

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for root, _dirs, names in os.walk("/kaggle/input"):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    print(f"\ncandidate image files: {len(files)}", flush=True)

    # label each by the nearest ancestor folder that maps to our vocabulary
    labelled = []
    unmapped = collections.Counter()
    for f in files:
        lab = None
        parts = f.split(os.sep)
        for seg in reversed(parts[:-1]):
            lab = map_label(seg)
            if lab:
                break
        if lab:
            labelled.append((f, lab))
        else:
            unmapped[parts[-2] if len(parts) > 1 else "?"] += 1
    print(f"mapped to our 23-class vocabulary: {len(labelled)}", flush=True)
    print(f"unmapped folders (top 15): {unmapped.most_common(15)}", flush=True)

    lms, wlms, labels, splits, metas = [], [], [], [], []
    seen = set()
    stats = collections.Counter()
    rej = collections.Counter()
    t0 = time.time()

    for i, (path, lab) in enumerate(labelled):
        if i % 1000 == 0 and i:
            print(f"  {i}/{len(labelled)}  kept={len(labels)}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(
                os.path.join(OUT, "public_corpus_partial.npz"),
                landmarks=np.array(lms, dtype=np.float32),
                world=np.array(wlms, dtype=np.float32),
                labels=np.array(labels), split=np.array(splits))
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                rej["unreadable"] += 1
                continue
            # hash the DECODED pixels: the same photo appears under different
            # filenames across these datasets
            key = hashlib.sha1(np.ascontiguousarray(img)).hexdigest()
            if key in seen:
                rej["duplicate"] += 1
                continue
            seen.add(key)

            h, w = img.shape[:2]
            if max(h, w) > 1400:
                s = 1400.0 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.pose_landmarks:
                rej["no_person"] += 1
                continue
            lm = np.array([[q.x, q.y, q.z, q.visibility]
                           for q in res.pose_landmarks[0]], dtype=np.float32)
            if float(lm[[11, 12, 23, 24, 25, 26, 27, 28], 3].mean()) < MIN_VIS:
                rej["low_visibility"] += 1
                continue
            wl = (np.array([[q.x, q.y, q.z] for q in res.pose_world_landmarks[0]],
                           dtype=np.float32) if res.pose_world_landmarks
                  else np.zeros((33, 3), dtype=np.float32))
            lms.append(lm)
            wlms.append(wl)
            labels.append(lab)
            splits.append(split_of("pub:" + key))
            metas.append({"id": "pub:" + key, "source": path.split(os.sep)[3]
                          if len(path.split(os.sep)) > 3 else "?", "label": lab})
            stats[lab] += 1
        except Exception as e:
            rej[f"exc:{type(e).__name__}"] += 1

    lms = np.array(lms, dtype=np.float32)
    wlms = np.array(wlms, dtype=np.float32)
    labels = np.array(labels)
    splits = np.array(splits)
    ang = np.array([angles_from_landmarks(x) for x in lms], dtype=np.float32) \
        if len(lms) else np.zeros((0, 15), np.float32)

    np.savez_compressed(os.path.join(OUT, "public_corpus.npz"),
                        landmarks=lms, world=wlms, labels=labels,
                        split=splits, angles=ang)
    json.dump({"per_class": dict(stats), "rejected": dict(rej),
               "total": int(len(labels)),
               "train": int((splits == "train").sum()),
               "test": int((splits == "test").sum()),
               "meta": metas},
              open(os.path.join(OUT, "public_corpus_manifest.json"), "w"), indent=1)

    print("\n=== PER-CLASS ===")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v}")
    print(f"\nrejected: {dict(rej)}")
    print(f"TOTAL {len(labels)}  train={int((splits=='train').sum())}  "
          f"test={int((splits=='test').sum())}")
    print(f"elapsed {time.time()-t0:.0f}s")


main()
