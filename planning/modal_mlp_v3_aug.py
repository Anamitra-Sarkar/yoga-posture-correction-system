"""AsanaAI MLP v3: per-class real-world baseline + augmentation retrain, on Modal.

Why: the model is going free-form (pose selection removed), so it must name
whatever pose the user is in across all 23 classes. Training data is 12 vinyasa
videos with near-identical camera framing, which is the root cause of weak
real-world generalization. We augment LANDMARK COORDINATES (not angles, so the
geometry stays physically valid) to simulate camera viewpoint/distance/noise
variation, then re-derive the same 15 angle features with zero_z=True so the
training features match the inference path exactly.

Runs entirely on Modal; the local machine has ~250MB free RAM.
HF account: Arko007 ONLY (secret arko007-hf-token -> HF_TOKEN_ARKO007).
Never overwrites existing HF artifacts; new names only.
"""
import modal

app = modal.App("asanaai-mlp-v3-aug")

vol = modal.Volume.from_name("asanaai-mlp-v3", create_if_missing=True)

base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "numpy<2.0.0",
        "pandas",
        "scikit-learn",
        "torch",
        "mediapipe==0.10.14",
        "opencv-python-headless",
        "requests",
        "huggingface_hub>=0.34.0",
    )
)

VOL = "/data"
REPO = "Arko007/yoga-posture-models"

# ---------------------------------------------------------------- shared code

FEATURE_NAMES = [
    "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
    "hip_l", "hip_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r", "trunk_l", "trunk_r",
    "neck", "hip_abduct_l", "hip_abduct_r",
]

# Sanskrit + English search terms per class. transition/unknown is deliberately
# absent: it is not a photographable posture, it is the absence of one.
POSE_QUERIES = {
    "chair_pose": ["Utkatasana", "chair pose yoga"],
    "chaturanga": ["Chaturanga Dandasana", "chaturanga yoga pose"],
    "child_pose": ["Balasana", "child pose yoga"],
    "cobra_pose": ["Bhujangasana", "cobra pose yoga"],
    "corpse": ["Savasana", "shavasana yoga"],
    "downward_dog": ["Adho Mukha Svanasana", "downward facing dog yoga"],
    "halfway_lift": ["Ardha Uttanasana", "halfway lift yoga"],
    "lunge_pose": ["Anjaneyasana", "low lunge yoga pose"],
    "mountain_pose": ["Tadasana", "mountain pose yoga"],
    "plank": ["Phalakasana", "plank pose yoga"],
    "seated_easy_pose": ["Sukhasana", "easy pose yoga sitting"],
    "seated_forward": ["Paschimottanasana", "seated forward bend yoga"],
    "seated_staff": ["Dandasana", "staff pose yoga"],
    "standing_forward_fold": ["Uttanasana", "standing forward fold yoga"],
    "standing_pose": ["standing yoga asana", "Samasthiti yoga"],
    "table_top": ["Bharmanasana", "tabletop pose yoga"],
    "tree_pose": ["Vrksasana", "tree pose yoga"],
    "triangle": ["Trikonasana", "triangle pose yoga"],
    "upward_dog": ["Urdhva Mukha Svanasana", "upward facing dog yoga"],
    "upward_salute": ["Urdhva Hastasana", "upward salute yoga"],
    "warrior_1": ["Virabhadrasana I", "warrior one pose yoga"],
    "warrior_2": ["Virabhadrasana II", "warrior two pose yoga"],
}

UA = ("AsanaAI-YogaPoseResearch/1.0 (RCC Institute of Information Technology "
      "final-year project; github.com/Anamitra-Sarkar/yoga-posture-correction-system)")

# Wikimedia's search happily returns images that merely contain a human body
# (a family snapshot, a journal scan) for a yoga query. Keeping those would
# silently poison the labels, so a candidate's FILE TITLE must positively
# identify the pose. Regexes (not substrings) because "Virabhadrasana I" is a
# prefix of "Virabhadrasana II" -- matching naively would mix warrior 1 and 2,
# the exact confusion this project already fights in the rule engine.
TITLE_PATTERNS = {
    "chair_pose": r"utkatasana|chair[- ]pose",
    "chaturanga": r"chaturanga",
    "child_pose": r"balasana|child'?s?[- ]pose",
    "cobra_pose": r"bhujangasana|cobra[- ]pose",
    "corpse": r"savasana|shavasana|corpse[- ]pose",
    "downward_dog": r"adho[- ]mukha[- ]svanasana|downward",
    "halfway_lift": r"ardha[- ]uttanasana|halfway",
    "lunge_pose": r"anjaneyasana|lunge",
    "mountain_pose": r"tadasana|mountain[- ]pose",
    "plank": r"phalakasana|plank",
    "seated_easy_pose": r"sukhasana|easy[- ]pose|lotus",
    "seated_forward": r"paschimottanasana|seated[- ]forward",
    "seated_staff": r"dandasana|staff[- ]pose",
    "standing_forward_fold": r"uttanasana(?!\s*ardha)|forward[- ]fold|forward[- ]bend",
    "standing_pose": r"tadasana|samasthiti|standing",
    "table_top": r"bharmanasana|table[- ]?top",
    "tree_pose": r"vrksasana|vrikshasana|tree[- ]pose",
    "triangle": r"trikonasana|triangle[- ]pose",
    "upward_dog": r"urdhva[- ]mukha[- ]svanasana|upward[- ]facing[- ]dog",
    "upward_salute": r"urdhva[- ]hastasana|upward[- ]salute",
    # \b...\b so "I" cannot match inside "II"
    "warrior_1": r"virabhadrasana\s*i\b(?!i)|warrior\s*(i\b(?!i)|1\b|one\b)",
    "warrior_2": r"virabhadrasana\s*ii\b|warrior\s*(ii\b|2\b|two\b)",
}

# Reject a candidate whose title matches ANOTHER class more specifically than
# its own -- e.g. an "Ardha Uttanasana" (halfway lift) hit inside the
# standing_forward_fold query.
TITLE_EXCLUDE = {
    "standing_forward_fold": r"ardha[- ]uttanasana|halfway",
    "warrior_1": r"virabhadrasana\s*ii\b|warrior\s*(ii\b|2\b|two\b)",
    "warrior_2": r"virabhadrasana\s*i\b(?!i)|warrior\s*(i\b(?!i)|1\b|one\b)",
    "seated_easy_pose": r"dandasana|paschimottanasana",
    "standing_pose": r"utkatasana|uttanasana|virabhadrasana|vrksasana",
    "plank": r"chaturanga",
}


# Wikimedia Commons curates per-asana CATEGORIES, which are a far better label
# source than free-text search: membership is human-assigned and means "this
# file depicts this asana". Search matches file DESCRIPTIONS, so title-based
# filtering threw away most on-topic hits (yield collapsed to ~1/class).
# Categories give precision AND recall. Title filtering is kept below only as a
# fallback for classes with no usable category.
POSE_CATEGORIES = {
    "chair_pose": ["Utkatasana"],
    "chaturanga": ["Chaturanga Dandasana"],
    "child_pose": ["Balasana"],
    "cobra_pose": ["Bhujangasana"],
    "corpse": ["Shavasana", "Savasana"],
    "downward_dog": ["Adho Mukha Svanasana"],
    "halfway_lift": ["Ardha Uttanasana"],
    "lunge_pose": ["Anjaneyasana", "Lunging asanas"],
    "mountain_pose": ["Tadasana"],
    "plank": ["Phalakasana"],
    "seated_easy_pose": ["Sukhasana"],
    "seated_forward": ["Paschimottanasana"],
    "seated_staff": ["Dandasana"],
    "standing_forward_fold": ["Uttanasana"],
    "standing_pose": ["Standing asanas"],
    "table_top": ["Bharmanasana"],
    "tree_pose": ["Vrksasana"],
    "triangle": ["Utthita Trikonasana", "Trikonasana"],
    "upward_dog": ["Urdhva Mukha Svanasana"],
    "upward_salute": ["Urdhva Hastasana"],
    "warrior_1": ["Virabhadrasana I"],
    "warrior_2": ["Virabhadrasana II"],
}


def _title_ok(cls: str, title: str) -> bool:
    import re
    t = title.lower()
    pat = TITLE_PATTERNS.get(cls)
    if not pat or not re.search(pat, t):
        return False
    ex = TITLE_EXCLUDE.get(cls)
    if ex and re.search(ex, t):
        return False
    return True


def _angles(points, zero_z=True):
    """Verbatim port of backend/app/utils/geometry.py::extract_angles_from_landmarks."""
    import numpy as np
    SHOULDER_L, SHOULDER_R = 11, 12
    ELBOW_L, ELBOW_R = 13, 14
    WRIST_L, WRIST_R = 15, 16
    HIP_L, HIP_R = 23, 24
    KNEE_L, KNEE_R = 25, 26
    ANKLE_L, ANKLE_R = 27, 28
    HEEL_L, HEEL_R = 29, 30
    NOSE = 0

    def ang(a, b, c):
        ba, bc = a - b, c - b
        nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nba == 0 or nbc == 0:
            return 180.0
        cos = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    if points.shape[0] < 31:
        return [0.0] * 15
    p = points.copy().astype(np.float64)
    if zero_z:
        p[:, 2] = 0.0
    sm = (p[SHOULDER_L] + p[SHOULDER_R]) / 2.0
    hm = (p[HIP_L] + p[HIP_R]) / 2.0
    return [
        ang(p[SHOULDER_L], p[ELBOW_L], p[WRIST_L]),
        ang(p[SHOULDER_R], p[ELBOW_R], p[WRIST_R]),
        ang(p[HIP_L], p[SHOULDER_L], p[ELBOW_L]),
        ang(p[HIP_R], p[SHOULDER_R], p[ELBOW_R]),
        ang(p[SHOULDER_L], p[HIP_L], p[KNEE_L]),
        ang(p[SHOULDER_R], p[HIP_R], p[KNEE_R]),
        ang(p[HIP_L], p[KNEE_L], p[ANKLE_L]),
        ang(p[HIP_R], p[KNEE_R], p[ANKLE_R]),
        ang(p[KNEE_L], p[ANKLE_L], p[HEEL_L]),
        ang(p[KNEE_R], p[ANKLE_R], p[HEEL_R]),
        ang(p[SHOULDER_L], p[HIP_L], p[HIP_R]),
        ang(p[SHOULDER_R], p[HIP_R], p[HIP_L]),
        ang(p[NOSE], sm, hm),
        ang(p[HIP_R], p[HIP_L], p[KNEE_L]),
        ang(p[HIP_L], p[HIP_R], p[KNEE_R]),
    ]


def _model_cls():
    """Verbatim port of backend/app/models/mlp.py (architecture unchanged)."""
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, dim, dropout=0.3):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.block(x)

    class Yoga3HeadMLP(nn.Module):
        def __init__(self, input_dim, num_poses, num_joints=15):
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU())
            self.res1 = ResBlock(256, dropout=0.3)
            self.res2 = ResBlock(256, dropout=0.3)
            self.pose_head = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, num_poses))
            self.correctness_head = nn.Sequential(
                nn.Linear(256, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(64, 1))
            self.deviation_head = nn.Sequential(
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128, num_joints))

        def forward(self, x):
            f = self.input_layer(x)
            f = self.res1(f)
            f = self.res2(f)
            return (self.pose_head(f),
                    self.correctness_head(f).squeeze(-1),
                    self.deviation_head(f))

    return Yoga3HeadMLP


def _hf_token():
    import os
    for k in ("HF_TOKEN_ARKO007", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    raise RuntimeError("no Arko007 HF token in env")


# ------------------------------------------------------- 1. source test photos

@app.function(image=base, volumes={VOL: vol}, timeout=60 * 60 * 5)
def source_test_photos(per_class: int = 14):
    """Source real-world photos per class from Wikimedia Commons into the volume.

    Paced + UA'd + backoff on 429 (Wikimedia rate-limits aggressively).
    Only keeps images where MediaPipe finds a full-body landmark set, since a
    photo we cannot extract landmarks from tells us nothing about the MLP.
    """
    import os, json, time, random, requests, cv2, numpy as np
    import mediapipe as mp

    outdir = f"{VOL}/testset"
    os.makedirs(outdir, exist_ok=True)
    manifest_path = f"{VOL}/testset/manifest.json"
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})

    def get(url, **kw):
        for attempt in range(4):
            try:
                r = sess.get(url, timeout=60, **kw)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 8)) + 2 * attempt
                    print(f"  429, backing off {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except Exception as e:
                if attempt == 3:
                    print(f"  giving up on {url[:70]}: {type(e).__name__}", flush=True)
                    return None
                time.sleep(3 + 3 * attempt)
        return None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                        min_detection_confidence=0.5)

    for cls, queries in POSE_QUERIES.items():
        have = len(manifest.get(cls, []))
        if have >= per_class:
            print(f"[{cls}] already have {have}, skip", flush=True)
            continue
        manifest.setdefault(cls, [])
        seen_titles = {e["title"] for e in manifest[cls]}

        # Build the candidate list: curated CATEGORY members first (their
        # membership is the label), then free-text search as a fallback where a
        # category is missing or too small. `trusted` marks candidates whose
        # label comes from category membership, so they skip title filtering.
        candidates = []  # (page_dict, trusted)
        for cat in POSE_CATEGORIES.get(cls, []):
            r = get("https://commons.wikimedia.org/w/api.php", params={
                "action": "query", "format": "json", "generator": "categorymembers",
                "gcmtitle": f"Category:{cat}", "gcmtype": "file", "gcmlimit": "60",
                "prop": "imageinfo", "iiprop": "url", "iiurlwidth": "900",
            })
            time.sleep(random.uniform(1.5, 2.5))
            if r is None:
                continue
            pages = (r.json().get("query", {}) or {}).get("pages", {}) or {}
            got = list(pages.values())
            print(f"  category '{cat}': {len(got)} files", flush=True)
            candidates += [(pg, True) for pg in got]

        if len(candidates) < per_class * 2:
            for q in queries:
                r = get("https://commons.wikimedia.org/w/api.php", params={
                    "action": "query", "format": "json", "generator": "search",
                    "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": "6",
                    "gsrlimit": "60", "prop": "imageinfo", "iiprop": "url",
                    "iiurlwidth": "900",
                })
                time.sleep(random.uniform(1.5, 2.5))
                if r is None:
                    continue
                pages = (r.json().get("query", {}) or {}).get("pages", {}) or {}
                candidates += [(pg, False) for pg in pages.values()]

        if True:
            for pg, trusted in candidates:
                if len(manifest[cls]) >= per_class:
                    break
                title = pg.get("title", "")
                if title in seen_titles or not title:
                    continue
                # Label-integrity gate. Category members are already curated as
                # depicting this asana, so they pass; free-text search hits must
                # additionally have a title that names the pose, because search
                # readily returns unrelated photos that merely contain a person.
                if not trusted and not _title_ok(cls, title):
                    continue
                ii = (pg.get("imageinfo") or [{}])[0]
                url = ii.get("thumburl") or ii.get("url")
                # No extension filter here: Wikimedia thumb URLs often don't end
                # in a bare image extension (e.g. ".../900px-Foo.pdf.jpg", SVG
                # rasterisations, size-prefixed names), and an endswith() check
                # silently rejected every candidate on the first run. cv2.imdecode
                # below is the reliable filter -- it returns None for anything
                # that isn't a decodable image.
                if not url:
                    continue
                ir = get(url)
                time.sleep(random.uniform(1.5, 2.5))
                if ir is None:
                    continue
                arr = np.frombuffer(ir.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.pose_landmarks:
                    continue
                lms = res.pose_landmarks.landmark
                # require the full-body joints the 15 features depend on
                need = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]
                if min(lms[i].visibility for i in need) < 0.5:
                    continue
                pts = np.array([[l.x, l.y, l.z] for l in lms])
                fn = f"{cls}_{len(manifest[cls])}.npy"
                np.save(f"{outdir}/{fn}", pts)
                manifest[cls].append({"title": title, "landmarks": fn})
                seen_titles.add(title)
                print(f"[{cls}] kept {len(manifest[cls])}: {title[:65]}", flush=True)

        json.dump(manifest, open(manifest_path, "w"), indent=1)
        vol.commit()
        print(f"=== {cls}: {len(manifest[cls])} usable ===", flush=True)

    pose.close()
    total = sum(len(v) for v in manifest.values())
    print(f"TOTAL usable test images: {total}")
    return {c: len(v) for c, v in manifest.items()}


@app.function(image=base, volumes={VOL: vol}, timeout=60 * 20)
def diagnose_sourcing(cls: str = "warrior_2"):
    """Report how many candidates survive each filter stage, to find the leak."""
    import time, random, requests, cv2, numpy as np
    import mediapipe as mp

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    stats = {"pages": 0, "url_ok": 0, "downloaded": 0, "decoded": 0,
             "landmarks": 0, "vis_pass": 0}
    vis_samples = []

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                        min_detection_confidence=0.5)
    need = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]

    for q in POSE_QUERIES[cls]:
        r = sess.get("https://commons.wikimedia.org/w/api.php", timeout=60, params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": "6",
            "gsrlimit": "20", "prop": "imageinfo", "iiprop": "url",
            "iiurlwidth": "900"})
        print(f"query {q!r} -> HTTP {r.status_code}", flush=True)
        if r.status_code != 200:
            continue
        pages = (r.json().get("query", {}) or {}).get("pages", {}) or {}
        stats["pages"] += len(pages)
        for _, pg in list(pages.items())[:8]:
            ii = (pg.get("imageinfo") or [{}])[0]
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            stats["url_ok"] += 1
            try:
                ir = sess.get(url, timeout=60)
                time.sleep(random.uniform(1.5, 2.5))
                if ir.status_code != 200:
                    continue
                stats["downloaded"] += 1
                img = cv2.imdecode(np.frombuffer(ir.content, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                stats["decoded"] += 1
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.pose_landmarks:
                    continue
                stats["landmarks"] += 1
                lms = res.pose_landmarks.landmark
                mv = min(lms[i].visibility for i in need)
                vis_samples.append(round(float(mv), 3))
                if mv >= 0.5:
                    stats["vis_pass"] += 1
            except Exception as e:
                print("  err", type(e).__name__, flush=True)
    pose.close()
    print("STAGE COUNTS:", stats, flush=True)
    print("min-visibility samples:", sorted(vis_samples), flush=True)
    return {"stats": stats, "vis_samples": sorted(vis_samples)}


# --------------------------------------------------- 2. evaluate a checkpoint

@app.function(image=base, volumes={VOL: vol},
              secrets=[modal.Secret.from_name("arko007-hf-token")], timeout=60 * 60)
def evaluate(ckpt: str, encoder: str, from_hf: bool = True):
    """Per-class real-world accuracy of one checkpoint on the sourced photo set."""
    import os, json, numpy as np, torch
    from huggingface_hub import hf_hub_download

    tok = _hf_token()
    if from_hf:
        cpath = hf_hub_download(REPO, ckpt, token=tok)
        epath = hf_hub_download(REPO, encoder, token=tok)
    else:
        cpath, epath = f"{VOL}/{ckpt}", f"{VOL}/{encoder}"

    classes = list(np.load(epath, allow_pickle=True))
    Model = _model_cls()
    m = Model(input_dim=15, num_poses=len(classes))
    m.load_state_dict(torch.load(cpath, map_location="cpu"))
    m.eval()

    manifest = json.load(open(f"{VOL}/testset/manifest.json"))
    per_class, rows = {}, []
    for cls, entries in sorted(manifest.items()):
        if cls not in classes or not entries:
            continue
        correct = 0
        for e in entries:
            pts = np.load(f"{VOL}/testset/{e['landmarks']}")
            a = np.array(_angles(pts, zero_z=True), dtype=np.float32)[None, :]
            with torch.no_grad():
                logits, _, _ = m(torch.from_numpy(a))
            pred = classes[int(logits.argmax(1))]
            ok = pred == cls
            correct += ok
            rows.append({"true": cls, "pred": pred, "ok": bool(ok),
                         "title": e["title"][:70]})
        per_class[cls] = {"n": len(entries), "correct": int(correct),
                          "acc": round(correct / len(entries), 4)}

    n_tot = sum(v["n"] for v in per_class.values())
    c_tot = sum(v["correct"] for v in per_class.values())
    macro = float(np.mean([v["acc"] for v in per_class.values()])) if per_class else 0.0
    out = {"ckpt": ckpt, "per_class": per_class, "n": n_tot,
           "overall": round(c_tot / n_tot, 4) if n_tot else 0.0,
           "macro": round(macro, 4), "classes_covered": len(per_class)}
    print(json.dumps(out["per_class"], indent=1))
    print(f"\n{ckpt}: overall={out['overall']} macro={out['macro']} "
          f"n={n_tot} classes={len(per_class)}")
    json.dump({"summary": out, "rows": rows},
              open(f"{VOL}/eval_{ckpt.replace('/', '_')}.json", "w"), indent=1)
    vol.commit()
    return out


# ------------------------------------- 3. augmented feature build + retraining

@app.function(image=base, volumes={VOL: vol}, gpu="a10g",
              secrets=[modal.Secret.from_name("arko007-hf-token")], timeout=60 * 60 * 5)
def build_and_train(aug_per_frame: int = 3, epochs: int = 60, seed: int = 42):
    """Regenerate features with landmark-space augmentation, then retrain.

    Augmentations (applied to the 33x3 landmark cloud BEFORE angle extraction,
    so joint geometry stays physically valid):
      - horizontal mirror with correct L/R index swap
      - yaw rotation +-40 deg   (camera viewing angle: the dominant real-world variable)
      - pitch/roll +-15 deg     (camera tilt)
      - scale 0.85-1.15, translation jitter (subject distance/framing)
      - gaussian landmark noise (MediaPipe's own detection jitter)
    Validation is a VIDEO-LEVEL holdout: no window/frame from a held-out video
    is ever trained on, because neighbouring frames are near-identical and a
    random split leaks (that is why 94% val coexisted with 52.6% real-world).
    """
    import os, json, glob, numpy as np, pandas as pd, torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| torch:", torch.__version__, flush=True)

    # ---- mapping: landmark file title -> youtube video_id (matches
    # compile_master_dataset.py's get_youtube_id) so labels join correctly
    title2vid = {}
    for ij in glob.glob(f"{VOL}/raw/*.info.json"):
        try:
            vid = json.load(open(ij)).get("id")
            if vid:
                title2vid[os.path.basename(ij)[: -len(".info.json")]] = vid
        except Exception:
            pass
    print("info.json mappings:", len(title2vid), flush=True)

    df = pd.read_csv(f"{VOL}/raw/master_mlp_dataset_fully_classified.csv")
    print("master csv:", df.shape, flush=True)

    def map_base_pose(lbl):
        if lbl.startswith("imperfect_"):
            lbl = lbl.replace("imperfect_", "")
        if lbl == "child":
            return "child_pose"
        return lbl

    # label lookup: (video_id, frame_num) -> imperfect_pose_label
    lab = {}
    for vid, fn, l in zip(df["video_id"].values, df["frame_num"].values,
                          df["imperfect_pose_label"].values):
        lab[(vid, int(fn))] = l

    L_IDX = [11, 13, 15, 23, 25, 27, 29, 31, 1, 2, 3, 7, 9]
    R_IDX = [12, 14, 16, 24, 26, 28, 30, 32, 4, 5, 6, 8, 10]

    def mirror(p):
        q = p.copy()
        q[:, 0] = -q[:, 0]
        q[L_IDX + R_IDX] = q[R_IDX + L_IDX]
        return q

    # MediaPipe normalises x by image WIDTH and y by image HEIGHT, so for 16:9
    # source video the landmark space is anisotropic (y is stretched ~1.78x
    # relative to x). Rotating that space directly would shear the skeleton into
    # physically impossible shapes. So: undo the aspect stretch -> rotate in
    # (approximately) isotropic space -> reapply it, which yields the angles a
    # real camera at the new viewpoint would actually have produced. All 12
    # source videos are standard 16:9 YouTube uploads.
    ASPECT = 9.0 / 16.0

    def rot(p, yaw, pitch, roll):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        q = p.copy()
        q[:, 1] *= ASPECT                      # -> isotropic
        c = q.mean(0, keepdims=True)
        q = (q - c) @ (Rz @ Rx @ Ry).T + c
        q[:, 1] /= ASPECT                      # -> back to MediaPipe's space
        return q

    # Accumulate per VIDEO and concatenate once at the end. Building one giant
    # Python list of ~3.3M 15-float lists would peak well over a GB before
    # np.asarray ever runs; per-video numpy blocks keep peak memory near the
    # size of the final array.
    Xb, Yb, Yr, Vb = [], [], [], []
    for f in sorted(glob.glob(f"{VOL}/raw/landmarks_*.npy")):
        title = os.path.basename(f)[len("landmarks_"): -len(".npy")]
        vid = title2vid.get(title)
        if vid is None:
            print("  !! no video_id for", title[:50], flush=True)
            continue
        lms = np.load(f)  # [N,33,4]
        n_kept = 0
        vX, vY, vR = [], [], []
        for i in range(lms.shape[0]):
            l = lab.get((vid, i + 1))
            if l is None:
                continue
            pts = lms[i, :, :3].astype(np.float64)
            variants = [pts, mirror(pts)]
            for _ in range(aug_per_frame):
                q = pts if rng.random() < 0.5 else mirror(pts)
                q = rot(q,
                        np.radians(rng.uniform(-40, 40)),
                        np.radians(rng.uniform(-15, 15)),
                        np.radians(rng.uniform(-15, 15)))
                # NOTE: uniform scale and translation are deliberately NOT used
                # here -- the 15 features are joint ANGLES, which are invariant
                # to both, so they would be pure no-ops. What does change an
                # angle is an ANISOTROPIC stretch, which is what differing
                # camera aspect ratios / lens characteristics actually produce.
                q[:, 0] *= rng.uniform(0.9, 1.1)
                q[:, 1] *= rng.uniform(0.9, 1.1)
                q = q + rng.normal(0, 0.006, q.shape)  # mediapipe-like jitter
                variants.append(q)
            for q in variants:
                vX.append(_angles(q, zero_z=True))
                vR.append(l)
                vY.append(map_base_pose(l))
            n_kept += 1
        if not vX:
            print(f"  {title[:45]}: no labelled frames, skipped", flush=True)
            continue
        Xb.append(np.asarray(vX, dtype=np.float32))
        Yb.append(np.asarray(vY))
        Yr.append(np.asarray(vR))
        Vb.append(np.full(len(vX), vid))
        print(f"  {title[:45]}: {n_kept} frames -> {len(vX)} rows", flush=True)
        del vX, vY, vR

    X = np.concatenate(Xb)
    Ybase = np.concatenate(Yb)
    Yraw = np.concatenate(Yr)
    Vid = np.concatenate(Vb)
    del Xb, Yb, Yr, Vb
    print("augmented dataset:", X.shape, flush=True)

    # Class list pinned to the LIVE encoder's exact set AND ORDER, so the new
    # checkpoint stays a valid pair with the encoder the backend already loads.
    # A silent reordering here would corrupt every prediction in production.
    from huggingface_hub import hf_hub_download
    live_enc = hf_hub_download(REPO, "mlp_3head_pose_encoder.npy", token=_hf_token())
    classes = list(np.load(live_enc, allow_pickle=True))
    cls2i = {c: i for i, c in enumerate(classes)}
    keep = np.array([b in cls2i for b in Ybase])
    X, Ybase, Yraw, Vid = X[keep], Ybase[keep], Yraw[keep], Vid[keep]
    y_pose = np.array([cls2i[b] for b in Ybase], dtype=np.int64)
    y_corr = np.array([0.0 if (l.startswith("imperfect_") or l == "transition/unknown")
                       else 1.0 for l in Yraw], dtype=np.float32)
    print("after class filter:", X.shape, "| classes:", len(classes), flush=True)

    # ---- video-level holdout
    vids = sorted(set(Vid.tolist()))
    rng2 = np.random.default_rng(seed)
    hold = set(rng2.choice(vids, size=max(2, len(vids) // 5), replace=False).tolist())
    tr = np.array([v not in hold for v in Vid])
    va = ~tr
    print(f"holdout videos: {sorted(hold)} | train={tr.sum()} val={va.sum()}", flush=True)

    cnt = np.bincount(y_pose[tr], minlength=len(classes))
    w = 1.0 / np.sqrt(np.where(cnt == 0, 1, cnt))
    w = w / w.sum()
    wt = torch.tensor(w, dtype=torch.float32, device=dev)

    Model = _model_cls()
    model = Model(input_dim=15, num_poses=len(classes)).to(dev)
    pose_crit = nn.CrossEntropyLoss(weight=wt)
    corr_crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    def loader(mask, shuffle):
        return DataLoader(TensorDataset(
            torch.from_numpy(X[mask]), torch.from_numpy(y_pose[mask]),
            torch.from_numpy(y_corr[mask])), batch_size=512, shuffle=shuffle,
            num_workers=2, drop_last=shuffle)

    trl, val = loader(tr, True), loader(va, False)
    best_macro, best_state = -1.0, None

    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xb, pb, cb in trl:
            xb, pb, cb = xb.to(dev), pb.to(dev), cb.to(dev)
            opt.zero_grad()
            pl, cl, _ = model(xb)
            loss = pose_crit(pl, pb) + 0.3 * corr_crit(cl, cb)
            loss.backward()
            opt.step()
            tl += loss.item()

        model.eval()
        hit = np.zeros(len(classes))
        tot = np.zeros(len(classes))
        with torch.no_grad():
            for xb, pb, cb in val:
                pl, _, _ = model(xb.to(dev))
                pr = pl.argmax(1).cpu().numpy()
                for t, p in zip(pb.numpy(), pr):
                    tot[t] += 1
                    hit[t] += (t == p)
        present = tot > 0
        macro = float((hit[present] / tot[present]).mean())
        overall = float(hit.sum() / tot.sum())
        sched.step(macro)
        if macro > best_macro:
            best_macro, best_state = macro, {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}
        print(f"ep {ep:02d}/{epochs} loss={tl/max(1,len(trl)):.4f} "
              f"holdout macro={macro:.4f} overall={overall:.4f} "
              f"(classes present {int(present.sum())})", flush=True)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{VOL}/mlp_3head_model_v3_aug.pth")
    np.save(f"{VOL}/mlp_3head_pose_encoder_v3_aug.npy", np.array(classes, dtype=object))
    vol.commit()
    print(f"saved. best video-level holdout macro={best_macro:.4f}")
    return {"best_holdout_macro": round(best_macro, 4), "rows": int(X.shape[0]),
            "holdout_videos": sorted(hold), "classes": len(classes)}


@app.function(image=base, volumes={VOL: vol},
              secrets=[modal.Secret.from_name("arko007-hf-token")], timeout=60 * 90)
def holdout_compare(holdout: str, ckpts: str):
    """Fair per-class comparison of checkpoints on held-out VIDEO frames.

    Uses ORIGINAL (un-augmented) frames only: augmented rows are synthetic and
    must not be counted as real-world evidence. Both checkpoints see byte-identical
    inputs, and no frame from these videos was trained on (video-level holdout),
    so neighbouring-frame leakage cannot inflate the result the way a random
    frame split does.

    holdout: comma-separated video_ids;  ckpts: comma-separated
             "name.pth:encoder.npy:hf|vol" specs.
    """
    import os, json, glob, numpy as np, torch
    from huggingface_hub import hf_hub_download
    import pandas as pd

    tok = _hf_token()
    hold = set(holdout.split(","))

    title2vid = {}
    for ij in glob.glob(f"{VOL}/raw/*.info.json"):
        try:
            v = json.load(open(ij)).get("id")
            if v:
                title2vid[os.path.basename(ij)[: -len(".info.json")]] = v
        except Exception:
            pass

    df = pd.read_csv(f"{VOL}/raw/master_mlp_dataset_fully_classified.csv")
    lab = {(v, int(f)): l for v, f, l in zip(
        df["video_id"], df["frame_num"], df["imperfect_pose_label"])}

    def base_pose(l):
        if l.startswith("imperfect_"):
            l = l.replace("imperfect_", "")
        return "child_pose" if l == "child" else l

    X, Y = [], []
    for f in sorted(glob.glob(f"{VOL}/raw/landmarks_*.npy")):
        vid = title2vid.get(os.path.basename(f)[len("landmarks_"): -len(".npy")])
        if vid not in hold:
            continue
        lms = np.load(f)
        for i in range(lms.shape[0]):
            l = lab.get((vid, i + 1))
            if l is None:
                continue
            X.append(_angles(lms[i, :, :3].astype(np.float64), zero_z=True))
            Y.append(base_pose(l))
        print(f"  holdout {vid}: {len(X)} cumulative original frames", flush=True)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y)
    print(f"holdout original frames: {X.shape}", flush=True)

    Model = _model_cls()
    results = {}
    for spec in ckpts.split(","):
        name, enc, src = spec.split(":")
        cp = hf_hub_download(REPO, name, token=tok) if src == "hf" else f"{VOL}/{name}"
        ep = hf_hub_download(REPO, enc, token=tok) if src == "hf" else f"{VOL}/{enc}"
        classes = list(np.load(ep, allow_pickle=True))
        m = Model(input_dim=15, num_poses=len(classes))
        m.load_state_dict(torch.load(cp, map_location="cpu"))
        m.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                pl, _, _ = m(torch.from_numpy(X[i:i + 8192]))
                preds.append(pl.argmax(1).numpy())
        pred = np.concatenate(preds)
        pred_lbl = np.array([classes[p] for p in pred])
        per = {}
        for c in sorted(set(Y.tolist())):
            msk = Y == c
            per[c] = {"n": int(msk.sum()),
                      "acc": round(float((pred_lbl[msk] == c).mean()), 4)}
        macro = round(float(np.mean([v["acc"] for v in per.values()])), 4)
        overall = round(float((pred_lbl == Y).mean()), 4)
        results[name] = {"per_class": per, "macro": macro, "overall": overall}
        print(f"\n{name}: macro={macro} overall={overall}", flush=True)

    names = list(results)
    allc = sorted({c for r in results.values() for c in r["per_class"]})
    print(f"\n{'class':24s} {'n':>6s} " + " ".join(f"{n[:16]:>17s}" for n in names))
    print("-" * (32 + 18 * len(names)))
    for c in allc:
        n0 = results[names[0]]["per_class"].get(c, {}).get("n", 0)
        cells = " ".join(
            f"{results[n]['per_class'].get(c, {}).get('acc', float('nan')):>17.3f}"
            for n in names)
        print(f"{c:24s} {n0:6d} {cells}")
    print("-" * (32 + 18 * len(names)))
    print(f"{'MACRO':24s} {'':6s} " + " ".join(f"{results[n]['macro']:>17.3f}" for n in names))
    print(f"{'OVERALL':24s} {len(Y):6d} " + " ".join(f"{results[n]['overall']:>17.3f}" for n in names))
    json.dump(results, open(f"{VOL}/holdout_compare.json", "w"), indent=1)
    vol.commit()
    return {n: {"macro": results[n]["macro"], "overall": results[n]["overall"]}
            for n in names}


@app.local_entrypoint()
def hcmp(holdout: str, ckpts: str = ("mlp_3head_model_v2.pth:mlp_3head_pose_encoder.npy:hf,"
                                     "mlp_3head_model_v3_aug.pth:mlp_3head_pose_encoder_v3_aug.npy:vol")):
    print(holdout_compare.remote(holdout=holdout, ckpts=ckpts))


@app.function(image=base, volumes={VOL: vol},
              secrets=[modal.Secret.from_name("arko007-hf-token")], timeout=60 * 30)
def upload_v3():
    """Publish v3 under NEW names. Never touches existing artifacts."""
    from huggingface_hub import HfApi
    api = HfApi(token=_hf_token())
    print("HF account:", api.whoami().get("name"))
    for fn in ("mlp_3head_model_v3_aug.pth", "mlp_3head_pose_encoder_v3_aug.npy"):
        api.upload_file(path_or_fileobj=f"{VOL}/{fn}", path_in_repo=fn,
                        repo_id=REPO, repo_type="model",
                        commit_message="MLP v3: landmark-space augmentation, video-level holdout")
        print("uploaded", fn)
    return {"uploaded": True}


@app.local_entrypoint()
def diag(cls: str = "warrior_2"):
    print(diagnose_sourcing.remote(cls=cls))


@app.local_entrypoint()
def src(per_class: int = 12):
    print(source_test_photos.remote(per_class=per_class))


@app.local_entrypoint()
def train(aug_per_frame: int = 3, epochs: int = 60):
    print(build_and_train.remote(aug_per_frame=aug_per_frame, epochs=epochs))


@app.local_entrypoint()
def compare():
    """Head-to-head per-class real-world comparison: live v2 vs augmented v3."""
    old = evaluate.remote(ckpt="mlp_3head_model_v2.pth",
                          encoder="mlp_3head_pose_encoder.npy", from_hf=True)
    new = evaluate.remote(ckpt="mlp_3head_model_v3_aug.pth",
                          encoder="mlp_3head_pose_encoder_v3_aug.npy", from_hf=False)
    keys = sorted(set(old["per_class"]) | set(new["per_class"]))
    print(f"\n{'class':24s} {'n':>3s}  {'v2':>7s}  {'v3_aug':>7s}   delta")
    print("-" * 60)
    for k in keys:
        o = old["per_class"].get(k, {})
        n = new["per_class"].get(k, {})
        oa, na = o.get("acc"), n.get("acc")
        nn = n.get("n", o.get("n", 0))
        d = ("" if oa is None or na is None
             else f"{na - oa:+.3f}")
        print(f"{k:24s} {nn:3d}  {('-' if oa is None else f'{oa:.3f}'):>7s}  "
              f"{('-' if na is None else f'{na:.3f}'):>7s}   {d}")
    print("-" * 60)
    print(f"{'OVERALL':24s} {old['n']:3d}  {old['overall']:.3f}  {new['overall']:.3f}   "
          f"{new['overall'] - old['overall']:+.3f}")
    print(f"{'MACRO':24s} {'':3s}  {old['macro']:.3f}  {new['macro']:.3f}   "
          f"{new['macro'] - old['macro']:+.3f}")
