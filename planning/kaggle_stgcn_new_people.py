"""Extract ST-GCN sequences from a multi-PERSON yoga video dataset.

WHY THIS AND NOT MORE PHOTOS
----------------------------
The ST-GCN consumes 60-frame skeletal sequences, so the photo corpora that
rescued the MLP are useless to it -- a photograph has no temporal extent. It
needs video, and specifically video of DIFFERENT PEOPLE.

The existing sequence corpus is 54,488 windows, but every one comes from the
same collection of 12 vinyasa videos, and 34,578 of them (63%) are
transition/unknown. Its reported 63.0% macro is honest as far as it goes --
it uses a video-level holdout, unlike the MLP's leaky 90.86% -- but the two
held-out videos come from the same shoot as the training ones: same framing
conventions, same production style, often the same instructor. That measures
generalisation across clips, not across bodies.

nandwalritik/yoga-pose-videos-dataset is organised as
"<PersonName>_<Asana>.mp4" across roughly 19 named people. That makes a
CROSS-PERSON holdout possible, which is a far stronger claim: it answers "does
this work on someone it has never seen", which is the only question that
matters for an app strangers will install.

TIMESCALE MUST MATCH
--------------------
Frames are sampled at 3 fps, identical to the original pipeline, because a
60-frame window then spans the same ~20 seconds the deployed model was trained
to expect. Sampling denser would silently change what a "window" means and
make the new data incomparable with the old.

PERSON GROUPING IS DELIBERATELY CONSERVATIVE
--------------------------------------------
The filenames contain "veena" and "Veena", "Santosh" and "Santosh2". Treating
those as different people would let the same body appear on both sides of the
holdout and quietly inflate the result -- exactly the leak that made the MLP's
90.86% meaningless. Names are therefore lowercased and stripped of trailing
digits, and ambiguous near-duplicates are merged rather than split, because
the failure mode of over-merging (a slightly smaller training set) is much
cheaper than the failure mode of under-merging (a dishonest number).
"""
import os
import re
import sys
import glob
import json
import time
import subprocess
import collections

import numpy as np


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
SAMPLE_FPS = 3          # must match the original corpus; see module docstring
WINDOW = 60             # frames per sequence, as the deployed model expects
STRIDE = 12             # same overlap convention as the existing corpus
MAX_SECONDS = 90        # cap per clip so one long video cannot eat the budget

ASANA_TO_POSE = {
    "bhujangasana": "cobra_pose",
    "padmasana": "seated_easy_pose",
    "shavasana": "corpse",
    "savasana": "corpse",
    "tadasana": "mountain_pose",
    "trikonasana": "triangle",
    "vrikshasana": "tree_pose",
    "vrksasana": "tree_pose",
    "virabhadrasana": "warrior_2",
    "adhomukhasvanasana": "downward_dog",
    "balasana": "child_pose",
    "utkatasana": "chair_pose",
    "uttanasana": "standing_forward_fold",
    "dandasana": "seated_staff",
    "phalakasana": "plank",
}


def parse_name(stem: str):
    """'Abhay_Bhujangasana' -> ('abhay', 'cobra_pose')."""
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None
    person_raw, asana_raw = parts[0], "_".join(parts[1:])
    # conservative person id: lowercase, drop trailing digits, so
    # veena/Veena and Santosh/Santosh2 collapse to one body
    person = re.sub(r"\d+$", "", person_raw.strip().lower())
    asana = re.sub(r"[^a-z]", "", asana_raw.lower())
    pose = None
    for key, val in ASANA_TO_POSE.items():
        if key in asana:
            pose = val
            break
    return person, pose


def normalize_sequence(coords: np.ndarray) -> np.ndarray:
    """Pelvis-centre then scale-normalise, mirroring
    geometry.normalize_coordinate_sequence so the new windows land in the same
    space as the deployed model's training data."""
    seq = coords.reshape(coords.shape[0], 33, 3).astype(np.float32)
    hip_mid = (seq[:, 23, :] + seq[:, 24, :]) / 2.0
    seq = seq - hip_mid[:, None, :]
    sh_mid = (seq[:, 11, :] + seq[:, 12, :]) / 2.0
    scale = np.linalg.norm(sh_mid, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    seq = seq / scale[:, None, :]
    return seq.reshape(coords.shape[0], 99)


def main():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import requests

    print("=== mounted inputs ===", flush=True)
    for d in sorted(glob.glob("/kaggle/input/*")):
        print(" ", d, flush=True)

    vids = []
    for root, _d, names in os.walk("/kaggle/input"):
        for n in names:
            if n.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                vids.append(os.path.join(root, n))
    print(f"videos found: {len(vids)}", flush=True)

    task = os.path.join(OUT, "pose_landmarker_heavy.task")
    if not os.path.exists(task):
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        open(task, "wb").write(requests.get(url, timeout=300).content)

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
        output_segmentation_masks=False)

    feats, labels, people, sources = [], [], [], []
    per_person = collections.Counter()
    per_pose = collections.Counter()
    skipped = collections.Counter()
    t0 = time.time()

    for vi, path in enumerate(sorted(vids)):
        stem = os.path.splitext(os.path.basename(path))[0]
        person, pose = parse_name(stem)
        if not person or not pose:
            skipped["unmapped_name"] += 1
            continue
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(fps / SAMPLE_FPS)))
            rows, idx, ts = [], 0, 0
            max_frames = int(MAX_SECONDS * fps)
            with mp_vision.PoseLandmarker.create_from_options(opts) as lmk:
                while idx < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % step == 0:
                        h, w = frame.shape[:2]
                        if w > 640:
                            frame = cv2.resize(frame, (640, int(h * 640 / w)))
                        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        ts += int(1000 / SAMPLE_FPS)
                        r = lmk.detect_for_video(img, ts)
                        if r.pose_landmarks:
                            rows.append([c for q in r.pose_landmarks[0]
                                         for c in (q.x, q.y, q.z)])
                    idx += 1
            cap.release()

            if len(rows) < WINDOW:
                skipped["too_short"] += 1
                continue
            arr = np.array(rows, dtype=np.float32)
            n = 0
            for s in range(0, len(arr) - WINDOW + 1, STRIDE):
                feats.append(normalize_sequence(arr[s:s + WINDOW]))
                labels.append(f"hold:{pose}")
                people.append(person)
                sources.append(stem)
                n += 1
            per_person[person] += n
            per_pose[pose] += n
            if vi % 10 == 0:
                print(f"  [{vi}/{len(vids)}] {stem[:40]:<40} {person:<10} "
                      f"{pose:<20} +{n} windows  {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            skipped[f"exc:{type(e).__name__}"] += 1

    F = np.array(feats, dtype=np.float32)
    L = np.array(labels)
    P = np.array(people)
    S = np.array(sources)
    np.save(os.path.join(OUT, "newpeople_feats.npy"), F)
    np.save(os.path.join(OUT, "newpeople_labels.npy"), L)
    np.save(os.path.join(OUT, "newpeople_person.npy"), P)
    np.save(os.path.join(OUT, "newpeople_source.npy"), S)
    json.dump({"per_person": dict(per_person), "per_pose": dict(per_pose),
               "skipped": dict(skipped), "windows": int(len(L)),
               "people": sorted(set(P.tolist()))},
              open(os.path.join(OUT, "newpeople_manifest.json"), "w"), indent=1)

    print(f"\nshape {F.shape}")
    print(f"distinct people: {len(set(P.tolist()))} -> {sorted(set(P.tolist()))}")
    print("\n=== windows per pose ===")
    for k, v in per_pose.most_common():
        print(f"  {k:<24} {v}")
    print(f"\nskipped: {dict(skipped)}")
    print(f"elapsed {time.time()-t0:.0f}s")


main()
