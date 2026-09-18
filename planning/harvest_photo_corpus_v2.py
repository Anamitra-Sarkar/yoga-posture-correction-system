"""Harvest a MUCH larger real-photo corpus -- Commons + Openverse.

WHY A SECOND HARVEST
--------------------
The first harvest is what turned the MLP around: trained on real photos it
scores 35.5% macro on a held-out photo split vs 14.1% for the video-only model,
a 2.5x gain. But it only found 422 usable photos across 22 classes, and the
per-class floor is where the model still fails outright:

    upward_dog 1   standing_forward_fold 11   plank 14   cobra_pose 16
    chair_pose 21  ... and halfway_lift / chaturanga at 0

Those are exactly the five classes measuring 0% accuracy. With ~10-16 training
photos each, that is a data shortage, not a modelling failure, and no amount of
retraining fixes it. So: find more photos.

TWO CHANGES THAT ACTUALLY INCREASE RECALL
-----------------------------------------
1. OPENVERSE. api.openverse.org aggregates openly-licensed images from Flickr,
   museums and others. It is a genuinely different pool from Commons -- far more
   ordinary-practitioner photography, which is precisely the distribution the
   app is judged on -- and it stays redistributable and citable, so the paper's
   licensing argument is unchanged. No auth needed.

2. DEEPER, WIDER QUERIES. The first pass took only offsets 0 and 120 from each
   term. Weak classes get many more phrasings (Sanskrit transliteration
   variants, IAST spellings, plain-English descriptions) and pagination runs
   until a term is genuinely exhausted.

THE SPLIT IS DELIBERATELY UNCHANGED
-----------------------------------
split_of() still hashes a stable per-image id, so every Commons image keeps the
split it had in v1. The original 103-image test set therefore survives intact
inside the new one, and the 35.5% number stays directly comparable -- the
training script evaluates on BOTH the frozen original test set and the expanded
one, so a gain can never be an artefact of an easier benchmark.
"""
import os, json, time, hashlib, argparse, subprocess, sys
from urllib.parse import unquote
import numpy as np
import requests


# Kaggle's CPU image has no mediapipe. --no-deps keeps the stock protobuf:
# letting mediapipe pin its own downgrades protobuf below what TensorFlow
# needs, which broke earlier runs on this project with an opaque
# "cannot import name runtime_version" ImportError.
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
    raise SystemExit("FATAL: could not import mediapipe after install attempts")

_ensure_mediapipe()

UA = ("AsanaAI-YogaResearch/1.0 (RCC Institute of Information Technology "
      "final-year project; github.com/Anamitra-Sarkar/yoga-posture-correction-system)")
COMMONS = "https://commons.wikimedia.org/w/api.php"
OPENVERSE = "https://api.openverse.org/v1/images/"
OUT = os.environ.get("OUT_DIR", "/kaggle/working")

# Weighted toward the starved classes. Sanskrit is transliterated many ways and
# photo captions are inconsistent, so recall matters far more than precision --
# every image is relabelled by its query class anyway and a wrong pose simply
# adds noise, while a missing pose adds nothing at all.
QUERIES = {
    # ---- the five 0%-accuracy classes get the deepest coverage ----
    "upward_dog": ["Urdhva Mukha Svanasana", "Urdhva Mukha Shvanasana",
                   "upward facing dog yoga", "upward dog pose", "up dog yoga pose",
                   "yoga backbend arms straight prone", "cobra upward dog asana"],
    "standing_forward_fold": ["Uttanasana", "Uttanaasana", "standing forward bend yoga",
                              "forward fold yoga", "standing forward fold asana",
                              "yoga touching toes standing", "hastapadasana",
                              "standing head to knees yoga"],
    "plank": ["Phalakasana", "Kumbhakasana", "plank pose yoga", "yoga plank hold",
              "high plank yoga", "yoga push up position", "santolanasana"],
    "cobra_pose": ["Bhujangasana", "Bhujangaasana", "cobra pose yoga",
                   "cobra asana backbend", "yoga prone backbend chest lift",
                   "sphinx pose yoga", "serpent pose yoga"],
    "chair_pose": ["Utkatasana", "Utkataasana", "chair pose yoga", "fierce pose yoga",
                   "awkward pose yoga", "yoga squat arms overhead", "powerful pose yoga"],
    # ---- classes with zero usable photos in v1 ----
    "halfway_lift": ["Ardha Uttanasana", "halfway lift yoga", "half forward fold yoga",
                     "yoga flat back forward bend", "standing half forward bend"],
    "chaturanga": ["Chaturanga Dandasana", "four limbed staff pose yoga",
                   "chaturanga yoga", "low plank yoga pose", "yoga lowering push up"],
    # ---- thin but non-zero ----
    "table_top": ["Bharmanasana", "table top yoga", "cat cow yoga pose",
                  "quadruped yoga", "marjaryasana", "bitilasana", "yoga all fours"],
    "seated_forward": ["Paschimottanasana", "seated forward bend yoga",
                       "Pashchimottanasana", "yoga seated fold legs straight"],
    "upward_salute": ["Urdhva Hastasana", "upward salute yoga", "arms overhead yoga",
                      "Hasta Uttanasana", "yoga reaching up standing"],
    "standing_pose": ["standing yoga asana", "yoga pose standing outdoor",
                      "yoga standing posture practice"],
    "child_pose": ["Balasana", "child pose yoga", "childs pose resting",
                   "yoga kneeling forward rest", "Shashankasana"],
    "corpse": ["Savasana", "Shavasana", "corpse pose relaxation yoga",
               "yoga lying relaxation final"],
    "warrior_2": ["Virabhadrasana II", "warrior II yoga", "warrior 2 asana",
                  "yoga warrior pose side", "warrior two yoga"],
    "lunge_pose": ["Anjaneyasana", "low lunge yoga", "crescent lunge yoga",
                   "yoga lunge pose", "high lunge yoga"],
    "downward_dog": ["Adho Mukha Svanasana", "downward dog yoga",
                     "downward facing dog", "down dog yoga pose"],
    # ---- already reasonably covered; kept so the corpus stays balanced-ish ----
    "warrior_1": ["Virabhadrasana I", "warrior I yoga", "warrior one asana"],
    "mountain_pose": ["Tadasana", "mountain pose yoga", "samasthiti", "standing yoga upright"],
    "tree_pose": ["Vrksasana", "Vrikshasana", "tree pose yoga", "yoga balance one leg"],
    "triangle": ["Trikonasana", "triangle pose yoga", "extended triangle yoga"],
    "seated_staff": ["Dandasana", "staff pose yoga seated"],
    "seated_easy_pose": ["Sukhasana", "easy pose yoga", "cross legged meditation yoga"],
}

# Classes already well covered do not need another 180 images; the starved ones
# do. Spending the (rate-limited) request budget uniformly would mostly buy more
# seated_easy_pose, which is already the largest class at 77.
PRIORITY = {"upward_dog", "standing_forward_fold", "plank", "cobra_pose",
            "chair_pose", "halfway_lift", "chaturanga", "table_top",
            "seated_forward", "upward_salute", "child_pose", "standing_pose"}

sess = requests.Session()
sess.headers.update({"User-Agent": UA})


def _get(url, params, tries=4):
    for a in range(tries):
        try:
            r = sess.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 0)) or 5 * (a + 1))
                continue
            if r.status_code >= 500:
                time.sleep(3 * (a + 1)); continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(3 * (a + 1))
    return None


def search_commons(term, limit=120, offset=0):
    d = _get(COMMONS, {"action": "query", "format": "json", "generator": "search",
                       "gsrsearch": f"filetype:bitmap {term}", "gsrnamespace": 6,
                       "gsrlimit": limit, "gsroffset": offset,
                       "prop": "imageinfo", "iiprop": "url|extmetadata",
                       "iiurlwidth": 900})
    out = []
    for p in ((d or {}).get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        u = ii.get("thumburl") or ii.get("url")
        if u:
            out.append({"id": p.get("title", ""), "url": u, "source": "commons",
                        "license": (ii.get("extmetadata", {})
                                    .get("LicenseShortName", {}) or {}).get("value", "")})
    return out


def search_openverse(term, page=1, page_size=100):
    """Openverse is a different image pool from Commons, which is the entire
    point -- more ordinary-practitioner photography, the distribution the app
    is actually judged on."""
    d = _get(OPENVERSE, {"q": term, "page": page, "page_size": page_size,
                         "license_type": "all-cc", "mature": "false"})
    out = []
    for r in ((d or {}).get("results") or []):
        u = r.get("url") or r.get("thumbnail")
        if not u:
            continue
        out.append({"id": f"openverse:{r.get('id')}", "url": u, "source": "openverse",
                    "license": f"{r.get('license','')} {r.get('license_version','')}".strip()})
    return out


def canonical_key(url):
    """Collapse the SAME underlying image to one key across both sources.

    Openverse mirrors a lot of Wikimedia Commons, but hands back the full
    upload.wikimedia.org URL while our Commons search returns a width-900
    thumburl -- different strings, same photo. Deduping on the raw URL would
    therefore admit the same image twice under two different stable ids, which
    could place it in BOTH train and test and quietly leak the benchmark. So
    Wikimedia URLs collapse to their filename, everything else to its URL.
    """
    if "wikimedia.org" in url or "wikipedia.org" in url:
        name = url.split("?")[0].rstrip("/").split("/")[-1]
        # a thumb URL ends in e.g. "900px-Bhujangasana.jpg"
        if "px-" in name:
            name = name.split("px-", 1)[1]
        return "wm:" + name.lower()
    return "url:" + url.split("?")[0]


def stable_id(c):
    """The id whose hash decides train/test -- source-independent by design.

    A Wikimedia image must get the SAME id whether we found it through the
    Commons search or through Openverse's mirror of it, or its split would
    depend on which query happened to reach it first and v1's frozen test set
    would silently stop being frozen. MediaWiki file titles are recoverable
    from the URL: percent-decode the filename and swap underscores for spaces,
    which reproduces the "File:Some Name.jpg" form v1 hashed.
    """
    url = c["url"]
    if "wikimedia.org" in url or "wikipedia.org" in url:
        name = unquote(url.split("?")[0].rstrip("/").split("/")[-1])
        if "px-" in name:
            name = name.split("px-", 1)[1]
        return "File:" + name.replace("_", " ")
    return c["id"]


def split_of(stable_id, test_frac=0.25):
    """UNCHANGED from v1 on purpose: every Commons image keeps the split it was
    given before, so the original 103-image test set survives inside the new
    corpus and the 35.5% baseline stays directly comparable."""
    h = int(hashlib.sha1(stable_id.encode()).hexdigest()[:8], 16)
    return "test" if (h % 100) < int(test_frac * 100) else "train"


def gather(label, terms, budget):
    """Interleave the two sources so a slow/empty one never starves the other."""
    cands, seen_ids = [], set()
    for t in terms:
        for off in (0, 120, 240, 360):
            for c in search_commons(t, 120, off):
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"]); cands.append(c)
            time.sleep(1.2)
            if len(cands) > budget * 8:
                break
        for page in (1, 2, 3):
            got = search_openverse(t, page)
            for c in got:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"]); cands.append(c)
            time.sleep(1.0)
            if not got:
                break
        if len(cands) > budget * 8:
            break
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-pose", type=int, default=120)
    ap.add_argument("--priority-per-pose", type=int, default=300)
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
    seen_keys, summary, rejected = set(), {}, {}

    # starved classes first, so a wall-clock timeout costs the classes that
    # already have enough rather than the ones that have none
    order = sorted(QUERIES, key=lambda k: (k not in PRIORITY, k))

    for label in order:
        budget = args.priority_per_pose if label in PRIORITY else args.per_pose
        cands = gather(label, QUERIES[label], budget)
        n_cm = sum(1 for c in cands if c["source"] == "commons")
        print(f"\n=== {label}: {len(cands)} candidates "
              f"({n_cm} commons / {len(cands)-n_cm} openverse), budget {budget} ===",
              flush=True)
        kept, no_person, low_vis = 0, 0, 0
        for c in cands:
            if kept >= budget:
                break
            ck = canonical_key(c["url"])
            if ck in seen_keys:
                continue
            seen_keys.add(ck)
            try:
                r = sess.get(c["url"], timeout=45)
                if r.status_code == 429:
                    time.sleep(8); continue
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if max(h, w) > 1400:      # MediaPipe gains nothing from huge images
                    s = 1400.0 / max(h, w)
                    img = cv2.resize(img, (int(w * s), int(h * s)))
                res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                if not res.pose_landmarks:
                    no_person += 1; continue
                lm = np.array([[p.x, p.y, p.z, p.visibility]
                               for p in res.pose_landmarks[0]], dtype=np.float32)
                if float(lm[[11, 12, 23, 24, 25, 26, 27, 28], 3].mean()) < args.min_visibility:
                    low_vis += 1; continue
                wl = (np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]],
                               dtype=np.float32) if res.pose_world_landmarks
                      else np.zeros((33, 3), dtype=np.float32))
                lms.append(lm); wlms.append(wl); labels.append(label)
                # Hash the canonical key, not the per-source id: a Commons
                # image found through Openverse must land in the same split it
                # had in v1, otherwise the frozen 103-image test set is no
                # longer frozen and the 35.5% baseline stops being comparable.
                splits.append(split_of(stable_id(c)))
                metas.append({"id": stable_id(c), "source": c["source"],
                              "license": c["license"]})
                kept += 1
                if kept % 25 == 0:
                    print(f"    {label}: {kept}", flush=True)
                if len(labels) % 200 == 0:
                    np.savez_compressed(os.path.join(OUT, "photo_corpus_v2_partial.npz"),
                                        landmarks=np.array(lms, dtype=np.float32),
                                        world=np.array(wlms, dtype=np.float32),
                                        labels=np.array(labels), split=np.array(splits))
                    json.dump(metas, open(os.path.join(OUT, "photo_corpus_v2_partial_meta.json"), "w"))
                    print(f"    [checkpoint {len(labels)}]", flush=True)
            except Exception:
                pass
            time.sleep(0.9)
        summary[label] = kept
        rejected[label] = {"no_person": no_person, "low_visibility": low_vis}
        print(f"=== {label}: kept {kept}  (rejected: {no_person} no-person, "
              f"{low_vis} low-visibility) ===", flush=True)

    lms = np.array(lms, dtype=np.float32); wlms = np.array(wlms, dtype=np.float32)
    labels = np.array(labels); splits = np.array(splits)
    np.savez_compressed(os.path.join(OUT, "photo_corpus_v2.npz"),
                        landmarks=lms, world=wlms, labels=labels, split=splits)
    json.dump({"per_class": summary, "rejected": rejected, "total": int(len(labels)),
               "train": int((splits == "train").sum()),
               "test": int((splits == "test").sum()), "meta": metas},
              open(os.path.join(OUT, "photo_corpus_v2_manifest.json"), "w"), indent=1)

    print("\n=== PER-CLASS ===")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {k:<26} {v}")
    print(f"TOTAL {len(labels)}  train={int((splits=='train').sum())}  "
          f"test={int((splits=='test').sum())}")


main()
