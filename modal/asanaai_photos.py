"""Real-photo corpus from two public HF datasets, extracted on Modal with the SAME recipe as the earlier corpus:
PoseLandmarker HEAVY (IMAGE mode, 1 pose), >=0.55 mean visibility of shoulders/hips/knees/ankles, labels mapped onto
the project's vocabulary by the same alias table, de-duplicated by SHA1 of decoded pixels, split by the same hash.
Sources: rotemvahava/yoga-poses-107 (5,994 photos, Sanskrit names), AdityasArsenal/Yoga-pose-Data-Set (2,134, 5 classes).
"""
import collections
import hashlib
import json
import os

import modal

app = modal.App("asanaai-photos")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("mediapipe==0.10.14", "opencv-python-headless==4.10.0.84", "numpy<2", "pyarrow", "huggingface_hub", "requests")
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
MIN_VIS = 0.55
TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/"
            "pose_landmarker_heavy.task")
ALIASES = [
    ("chaturanga dandasana", "chaturanga"), ("adho mukha svanasana", "downward_dog"), ("downward dog", "downward_dog"),
    ("urdhva mukha svanasana", "upward_dog"), ("urdhva hastasana", "upward_salute"), ("upward salute", "upward_salute"),
    ("ardha uttanasana", "halfway_lift"), ("uttanasana", "standing_forward_fold"), ("paschimottanasana", "seated_forward"),
    ("utthita trikonasana", "triangle"), ("trikonasana", "triangle"), ("virabhadrasana ii", "warrior_2"),
    ("virabhadrasana i", "warrior_1"), ("utkatasana", "chair_pose"), ("bhujangasana", "cobra_pose"),
    ("balasana", "child_pose"), ("savasana", "corpse"), ("phalakasana", "plank"), ("sukhasana", "seated_easy_pose"),
    ("padmasana", "seated_easy_pose"), ("dandasana", "seated_staff"), ("tadasana", "mountain_pose"),
    ("vrksasana", "tree_pose"), ("vriksasana", "tree_pose"), ("anjaneyasana", "lunge_pose"),
    ("ashwa sanchalanasana", "lunge_pose"), ("bitilasana", "table_top"), ("marjaryasana", "table_top"),
]
ALIASES.sort(key=lambda kv: -len(kv[0]))
# variants that merely CONTAIN a mapped name but are different asanas -> skip instead of mis-mapping
REJECT = ["virabhadrasana iii", "makara adho mukha", "adho mukha vriksasana", "parivrtta trikonasana",
          "salamba bhujangasana", "dwi pada viparita dandasana", "ananda balasana", "eka pada", "supta"]
EXACT = {"downdog": "downward_dog", "plank": "plank", "tree": "tree_pose", "warrior2": "warrior_2"}  # 5-class set; 'goddess' skipped


def map_label(name: str):
    f = " ".join(name.lower().replace("_", " ").replace("-", " ").split())
    if f in EXACT:
        return EXACT[f]
    if any(b in f for b in REJECT):
        return None
    for alias, target in ALIASES:
        if alias in f:
            return target
    return None


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=7200, max_containers=24)
def extract_part(repo: str, fname: str, names: list, part: int, nparts: int, oov: bool = True):
    import cv2
    import mediapipe as mp
    import numpy as np
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import hf_hub_download
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    task = "/tmp/pose_landmarker_heavy.task"
    if not os.path.exists(task):
        open(task, "wb").write(requests.get(TASK_URL, timeout=300).content)
    lmk = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task), running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1, output_segmentation_masks=False))
    path = hf_hub_download(repo, fname, repo_type="dataset")
    pf = pq.ParquetFile(path)
    lms, wls, labs, keys, raws = [], [], [], [], []
    rej = collections.Counter()
    for bi, batch in enumerate(pf.iter_batches(batch_size=32, columns=["image", "label"])):
        if bi % nparts != part:
            continue
        for img_d, li in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            raw = names[li]
            lab = map_label(raw)
            if lab is None:
                if not oov:
                    rej["unmapped"] += 1
                    continue
                lab = "transition/unknown"  # a real photo of a pose outside our vocabulary = "none of our poses"
            img = cv2.imdecode(np.frombuffer(img_d["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                rej["unreadable"] += 1
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
            lm = np.array([[q.x, q.y, q.z, q.visibility] for q in res.pose_landmarks[0]], dtype=np.float32)
            if float(lm[[11, 12, 23, 24, 25, 26, 27, 28], 3].mean()) < MIN_VIS:
                rej["low_visibility"] += 1
                continue
            lms.append(lm); labs.append(lab); keys.append(key); raws.append(raw)
    safe = repo.replace("/", "_")
    os.makedirs(f"{VOL}/photos/shards", exist_ok=True)
    np.savez_compressed(f"{VOL}/photos/shards/{safe}__{os.path.basename(fname)}__{part}.npz",
                        landmarks=np.array(lms, dtype=np.float32).reshape(-1, 33, 4), labels=np.array(labs),
                        keys=np.array(keys), raw=np.array(raws), src=repo)
    vol.commit()
    return {"repo": repo, "file": os.path.basename(fname), "part": part, "kept": len(labs), "rejected": dict(rej)}


@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=8192, timeout=1800)
def merge():
    import glob
    import numpy as np
    vol.reload()
    L, Y, K, S, R = [], [], [], [], []
    seen, dup = set(), 0
    for f in sorted(glob.glob(f"{VOL}/photos/shards/*.npz")):
        z = np.load(f, allow_pickle=True)
        src = str(z["src"])
        for lm, y, k, r in zip(z["landmarks"], z["labels"], z["keys"], z["raw"]):
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            L.append(lm); Y.append(str(y)); K.append(str(k)); S.append(src); R.append(str(r))
    split = ["test" if (int(hashlib.sha1(("pub:" + k).encode()).hexdigest()[:8], 16) % 100) < 25 else "train" for k in K]
    np.savez_compressed(f"{VOL}/photos/public_corpus.npz", landmarks=np.array(L, dtype=np.float32), labels=np.array(Y),
                        split=np.array(split), keys=np.array(K), src=np.array(S), raw=np.array(R))
    per = collections.Counter(Y)
    rep = {"total": len(Y), "duplicates_removed": dup, "per_class": dict(per.most_common()),
           "per_source": dict(collections.Counter(S)), "train": split.count("train"), "test": split.count("test")}
    json.dump(rep, open(f"{VOL}/photos/public_corpus_report.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(rep, indent=1))
    return rep


@app.local_entrypoint()
def main(nparts: int = 4):
    import json as _j
    import urllib.parse
    import urllib.request
    from huggingface_hub import HfApi
    jobs = []
    for repo in ("rotemvahava/yoga-poses-107", "AdityasArsenal/Yoga-pose-Data-Set"):
        inf = _j.load(urllib.request.urlopen("https://datasets-server.huggingface.co/info?dataset=" + urllib.parse.quote(repo), timeout=30))
        names = next(iter(inf["dataset_info"].values()))["features"]["label"]["names"]
        mapped = {n: map_label(n) for n in names}
        print(repo, "mapped classes:", {k: v for k, v in mapped.items() if v})
        for f in HfApi().list_repo_files(repo, repo_type="dataset"):
            if f.endswith(".parquet"):
                for p in range(nparts):
                    jobs.append((repo, f, names, p, nparts, True))
    for r in extract_part.starmap(jobs):
        print(r, flush=True)
    merge.remote()
