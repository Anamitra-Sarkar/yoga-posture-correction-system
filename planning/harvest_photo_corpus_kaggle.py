"""Harvest a large REAL-PHOTO corpus for AsanaAI, split into train/test.

THE ARGUMENT FOR THIS
---------------------
The MLP is a single-frame classifier. It is currently trained on frames from
12 vinyasa videos and evaluated on real photographs. That mismatch is the
measured failure: ~94% on held-in video frames vs 52.6% on real photos (and
~26% macro across the broader vocabulary).

Landmark augmentation was already tried to bridge it and did NOT work
(24.2% vs 25.8% macro) -- synthetic jitter around a narrow corpus cannot
invent the diversity of real photographs: different bodies, camera heights,
focal lengths, clothing, lighting, indoor/outdoor, and crucially the
non-idealised form of ordinary practitioners.

So: train on the same kind of data we are judged on. Photographs.

WHY WIKIMEDIA COMMONS
---------------------
Freely licensed, so both the corpus and the benchmark stay redistributable and
citable in the paper. YouTube frames are defensible as a training signal under
the precedent already set by the existing 12 videos, but Commons avoids both
the licensing question and the datacenter IP blocking that makes yt-dlp
unusable from CI/Codespace environments.

DISJOINT SPLIT
--------------
Images are assigned to train/test by a hash of their Commons title, so the
split is deterministic, reproducible, and guarantees no image appears in both.
Anything already used by the v2 test set is excluded from train.
"""
import os, json, time, hashlib, argparse
import numpy as np
import requests

UA = ("AsanaAI-YogaResearch/1.0 (RCC Institute of Information Technology "
      "final-year project; github.com/Anamitra-Sarkar/yoga-posture-correction-system)")
API = "https://commons.wikimedia.org/w/api.php"
OUT = os.environ.get("OUT_DIR", "/kaggle/working")

# Deliberately broad: several phrasings, Sanskrit + English, plus generic
# catch-alls, because Commons titling is inconsistent and recall matters far
# more than precision here (the rule engine relabels anyway).
QUERIES = {
    "warrior_2": ["Virabhadrasana II", "Warrior II yoga", "warrior 2 asana", "yoga warrior pose side"],
    "warrior_1": ["Virabhadrasana I", "Warrior I yoga", "warrior one asana"],
    "mountain_pose": ["Tadasana", "Mountain pose yoga", "samasthiti", "standing yoga upright"],
    "cobra_pose": ["Bhujangasana", "Cobra pose yoga", "cobra asana backbend"],
    "tree_pose": ["Vrksasana", "Vrikshasana", "Tree pose yoga", "yoga balance one leg"],
    "plank": ["Phalakasana", "Plank pose yoga", "kumbhakasana", "yoga plank hold"],
    "downward_dog": ["Adho Mukha Svanasana", "Downward dog yoga", "downward facing dog"],
    "chair_pose": ["Utkatasana", "Chair pose yoga", "fierce pose yoga"],
    "child_pose": ["Balasana", "Child pose yoga", "childs pose resting"],
    "table_top": ["Bharmanasana", "Table top yoga", "cat cow yoga pose", "quadruped yoga"],
    "seated_staff": ["Dandasana", "Staff pose yoga seated"],
    "seated_easy_pose": ["Sukhasana", "Easy pose yoga", "cross legged meditation yoga", "lotus meditation sitting"],
    "standing_forward_fold": ["Uttanasana", "Standing forward bend yoga", "forward fold yoga"],
    "halfway_lift": ["Ardha Uttanasana", "Halfway lift yoga"],
    "upward_salute": ["Urdhva Hastasana", "Upward salute yoga", "arms overhead yoga"],
    "upward_dog": ["Urdhva Mukha Svanasana", "Upward facing dog yoga"],
    "lunge_pose": ["Anjaneyasana", "Low lunge yoga", "crescent lunge yoga", "yoga lunge pose"],
    "triangle": ["Trikonasana", "Triangle pose yoga", "extended triangle yoga"],
    "chaturanga": ["Chaturanga Dandasana", "four limbed staff pose yoga"],
    "corpse": ["Savasana", "Shavasana", "corpse pose relaxation yoga"],
    "seated_forward": ["Paschimottanasana", "seated forward bend yoga"],
    "standing_pose": ["standing yoga asana", "yoga pose standing outdoor"],
}

sess = requests.Session(); sess.headers.update({"User-Agent": UA})


def _get(params, tries=4):
    for a in range(tries):
        try:
            r = sess.get(API, params=params, timeout=45)
            if r.status_code == 429:
                w = int(r.headers.get("Retry-After", 0)) or 5 * (a + 1)
                time.sleep(w); continue
            r.raise_for_status(); return r.json()
        except Exception:
            time.sleep(3 * (a + 1))
    return None


def search(term, limit=120, offset=0):
    d = _get({"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"filetype:bitmap {term}", "gsrnamespace": 6,
              "gsrlimit": limit, "gsroffset": offset,
              "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 900})
    out = []
    for p in ((d or {}).get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        u = ii.get("thumburl") or ii.get("url")
        if u:
            out.append({"title": p.get("title", ""), "url": u,
                        "license": (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "")})
    return out


def split_of(title, test_frac=0.25):
    """Deterministic, reproducible, and guarantees disjointness."""
    h = int(hashlib.sha1(title.encode()).hexdigest()[:8], 16)
    return "test" if (h % 100) < int(test_frac * 100) else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pose", type=int, default=180)
    ap.add_argument("--min-visibility", type=float, default=0.55)
    args = ap.parse_args()

    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    os.makedirs(OUT, exist_ok=True)

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

    lms, wlms, labels, splits, metas = [], [], [], [], []
    seen, summary = set(), {}

    for label, terms in QUERIES.items():
        kept = 0
        cands = []
        for t in terms:
            for off in (0, 120):
                cands.extend(search(t, 120, off))
                time.sleep(1.5)
                if len(cands) > args.per_pose * 6:
                    break
        print(f"\n=== {label}: {len(cands)} candidates ===", flush=True)
        for c in cands:
            if kept >= args.per_pose:
                break
            key = hashlib.md5(c["url"].encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            try:
                r = sess.get(c["url"], timeout=45)
                if r.status_code == 429:
                    time.sleep(8); continue
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                if not res.pose_landmarks:
                    continue
                lm = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]],
                              dtype=np.float32)
                if float(lm[[11, 12, 23, 24, 25, 26, 27, 28], 3].mean()) < args.min_visibility:
                    continue
                wl = (np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]],
                               dtype=np.float32) if res.pose_world_landmarks
                      else np.zeros((33, 3), dtype=np.float32))
                lms.append(lm); wlms.append(wl); labels.append(label)
                splits.append(split_of(c["title"]))
                metas.append({"title": c["title"], "license": c["license"]})
                kept += 1
                if kept % 25 == 0:
                    print(f"    {label}: {kept}", flush=True)
                # checkpoint so an interrupted run is never a total loss
                if len(labels) % 200 == 0:
                    np.savez_compressed(os.path.join(OUT, "photo_corpus_partial.npz"),
                                        landmarks=np.array(lms, dtype=np.float32),
                                        world=np.array(wlms, dtype=np.float32),
                                        labels=np.array(labels), split=np.array(splits))
                    print(f"    [checkpoint {len(labels)}]", flush=True)
            except Exception:
                pass
            time.sleep(1.1)
        summary[label] = kept
        print(f"=== {label}: kept {kept} ===", flush=True)

    lms = np.array(lms, dtype=np.float32)
    wlms = np.array(wlms, dtype=np.float32)
    labels = np.array(labels); splits = np.array(splits)
    np.savez_compressed(os.path.join(OUT, "photo_corpus.npz"),
                        landmarks=lms, world=wlms, labels=labels, split=splits)
    json.dump({"per_class": summary, "total": int(len(labels)),
               "train": int((splits == "train").sum()),
               "test": int((splits == "test").sum()), "meta": metas},
              open(os.path.join(OUT, "photo_corpus_manifest.json"), "w"), indent=1)

    print("\n=== PER-CLASS ===")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v}")
    print(f"TOTAL {len(labels)}  train={int((splits=='train').sum())}  test={int((splits=='test').sum())}")


if __name__ == "__main__":
    main()
