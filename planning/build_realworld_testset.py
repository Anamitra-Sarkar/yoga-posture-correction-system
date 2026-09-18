"""Build a REAL statistically-usable real-world test set for AsanaAI.

WHY
---
The existing real-world test set is 26 photos across 11 classes -- 1 to 5 per
class. At that size every per-class number is noise, so we cannot tell whether
any model change actually helped. This is the binding constraint on the whole
project: measurement, not modelling.

Target: 30-60 usable images per pose where they exist, honestly reporting the
achieved n per class (some rare asanas genuinely do not have that many usable
full-body photographs on Commons, and that is a finding, not a failure).

SOURCE / LICENSING
------------------
Wikimedia Commons only. Everything there is freely licensed, which keeps the
test set redistributable and citable in the paper -- unlike scraped YouTube
frames, which are fine as an implicit-fair-use training signal but awkward as a
published benchmark.

Run this in a GitHub Codespace (network + CPU), never on the local machine.
Outputs a compact .npz of landmarks + labels; the images themselves are NOT
kept, so nothing large travels anywhere.
"""
import os, io, json, time, hashlib, argparse
import numpy as np
import requests

UA = ("AsanaAI-YogaResearch/1.0 (RCC Institute of Information Technology "
      "final-year project; github.com/Anamitra-Sarkar/yoga-posture-correction-system)")
API = "https://commons.wikimedia.org/w/api.php"
OUT = os.environ.get("OUT_DIR", "testset_v2")

# Search terms per pose. Multiple phrasings + Sanskrit names, because Commons
# titles are inconsistent and a single query badly under-recalls.
QUERIES = {
    "warrior_2":      ["Virabhadrasana II", "Warrior II pose yoga", "warrior 2 yoga asana"],
    "warrior_1":      ["Virabhadrasana I", "Warrior I pose yoga", "warrior one yoga asana"],
    "mountain_pose":  ["Tadasana", "Mountain pose yoga", "samasthiti yoga"],
    "cobra_pose":     ["Bhujangasana", "Cobra pose yoga"],
    "tree_pose":      ["Vrksasana", "Vrikshasana", "Tree pose yoga"],
    "plank":          ["Phalakasana", "Plank pose yoga", "kumbhakasana"],
    "downward_dog":   ["Adho Mukha Svanasana", "Downward dog yoga", "downward facing dog pose"],
    "chair_pose":     ["Utkatasana", "Chair pose yoga"],
    "child_pose":     ["Balasana", "Child pose yoga"],
    "table_top":      ["Bharmanasana", "Table top pose yoga", "quadruped yoga pose"],
    "seated_staff":   ["Dandasana", "Staff pose yoga"],
    "seated_easy_pose": ["Sukhasana", "Easy pose yoga", "cross legged sitting yoga"],
    "standing_forward_fold": ["Uttanasana", "Standing forward bend yoga"],
    "halfway_lift":   ["Ardha Uttanasana", "Halfway lift yoga"],
    "upward_salute":  ["Urdhva Hastasana", "Upward salute yoga"],
    "upward_dog":     ["Urdhva Mukha Svanasana", "Upward facing dog yoga"],
    "lunge_pose":     ["Anjaneyasana", "Low lunge yoga", "crescent lunge yoga"],
    "standing_pose":  ["standing yoga asana"],
    "triangle":       ["Trikonasana", "Triangle pose yoga"],
    "chaturanga":     ["Chaturanga Dandasana", "four limbed staff pose"],
    "corpse":         ["Savasana", "Shavasana", "corpse pose yoga"],
    "seated_forward": ["Paschimottanasana", "seated forward bend yoga"],
}

sess = requests.Session()
sess.headers.update({"User-Agent": UA})


def _get(params, tries=4):
    """Commons rate-limits hard without a UA and without pacing."""
    for a in range(tries):
        try:
            r = sess.get(API, params=params, timeout=45)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0)) or (5 * (a + 1))
                print(f"    429, backing off {wait}s", flush=True); time.sleep(wait); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    retry {a+1}: {str(e)[:70]}", flush=True); time.sleep(3 * (a + 1))
    return None


def search(term, limit=60):
    d = _get({"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f'filetype:bitmap {term}', "gsrnamespace": 6,
              "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|extmetadata",
              "iiurlwidth": 900})
    out = []
    for p in ((d or {}).get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        if not url:
            continue
        lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "")
        out.append({"title": p.get("title", ""), "url": url, "license": lic})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pose", type=int, default=45)
    ap.add_argument("--min-visibility", type=float, default=0.55)
    args = ap.parse_args()

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    os.makedirs(OUT, exist_ok=True)

    # mediapipe >=0.10.30 removed the legacy mp.solutions.pose API, so use the
    # Tasks API. This is also what kaggle_process.py used to build the TRAINING
    # corpus, so the test set is extracted the same way the training data was.
    task_path = os.path.join(OUT, "pose_landmarker_heavy.task")
    if not os.path.exists(task_path):
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        with open(task_path, "wb") as fh:
            fh.write(requests.get(url, timeout=180).content)
    landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1, output_segmentation_masks=False))

    all_lm, all_lab, all_meta = [], [], []
    seen = set()
    summary = {}

    for label, terms in QUERIES.items():
        kept = 0
        cands = []
        for t in terms:
            cands.extend(search(t))
            time.sleep(2.0)
        print(f"\n=== {label}: {len(cands)} candidates ===", flush=True)
        for c in cands:
            if kept >= args.per_pose:
                break
            h = hashlib.md5(c["url"].encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            try:
                r = sess.get(c["url"], timeout=45)
                if r.status_code == 429:
                    time.sleep(10); continue
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                res = landmarker.detect(mp_img)
                if not res.pose_landmarks:
                    continue
                lm = np.array([[p.x, p.y, p.z, p.visibility]
                               for p in res.pose_landmarks[0]], dtype=np.float32)
                # require a genuinely full-body detection, else the sample says
                # more about framing than about the pose
                key = [11, 12, 23, 24, 25, 26, 27, 28]
                if float(lm[key, 3].mean()) < args.min_visibility:
                    continue
                wl = None
                if res.pose_world_landmarks:
                    wl = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]],
                                  dtype=np.float32)
                all_lm.append(lm)
                all_lab.append(label)
                all_meta.append({"title": c["title"], "license": c["license"],
                                 "world": wl is not None})
                if wl is not None:
                    np.save(f"{OUT}/world_{len(all_lm)-1}.npy", wl)
                kept += 1
            except Exception as e:
                print(f"    skip: {str(e)[:60]}", flush=True)
            time.sleep(1.5)
        summary[label] = kept
        print(f"=== {label}: kept {kept} ===", flush=True)

    np.savez_compressed(f"{OUT}/realworld_testset_v2.npz",
                        landmarks=np.array(all_lm, dtype=np.float32),
                        labels=np.array(all_lab))
    json.dump({"per_class": summary, "total": len(all_lm), "meta": all_meta},
              open(f"{OUT}/manifest_v2.json", "w"), indent=1)
    print("\n=== FINAL PER-CLASS COUNTS ===")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v}")
    print(f"TOTAL usable images: {len(all_lm)}")


if __name__ == "__main__":
    main()
