"""Cue-verified pose labels for the 24 videos.

A frame is labelled P only if ALL hold:
  1. CUE      the instructor named P (Whisper transcript) within [t0-1s, t1+30s] of the frame,
  2. GEOMETRY the app's own rule engine (backend rules_classifier.classify_pose, with orientation) says P,
  3. HOLD     the 15 angles are steady (mean |change| per joint over +-7 frames < STABLE_DEG).
Everything else is 'transition/unknown'. Blocks shorter than 15 frames are dropped (original clean_labels rule).
Angles are the APP's recipe: occlusion-interpolated landmarks, z zeroed.
"""
import json
import os
import re
import sys

import modal

app = modal.App("asanaai-cuelabel")
B = "/home/anamitra/yoga_posture_workspace/backend/app"
SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "pandas")
    .add_local_file(f"{B}/utils/geometry.py", "/srv/app/utils/geometry.py", copy=True)
    .add_local_file(f"{B}/utils/rules_classifier.py", "/srv/app/utils/rules_classifier.py", copy=True)
    .add_local_file(f"{SRC}/extract_features_safe.py", "/opt/orig/extract_features_safe.py", copy=True)
    .run_commands("touch /srv/app/__init__.py /srv/app/utils/__init__.py")
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
ANG = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r",
       "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck", "hip_abduct_l", "hip_abduct_r"]
OLD = "HmZFwoUU3WQ SZU7Sbgu57o oUgpXY7QhpQ 7ciS93shMNQ P8uHMMmWMHQ 4ORRiN2_aVI RQMtwbhXD7A 8ibxmzJziHU Eml2xnoLpYE QiebZSlTw_U L-z1HLkS_-Y s-1vMbAgYWU".split()
NEW = ("v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk hHhxKkskHDg JHjV-wFTwSw "
       "EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow 4ZBUDd4bsyA").split()
STABLE_DEG, MIN_RUN, PRE_S, POST_S = 8.0, 15, 1.0, 30.0
CUES = {
    "mountain_pose": r"mountain pose|\btadasana\b|ताड़ासन|ताडासन|माउंटेन",
    "cobra_pose": r"\bcobra\b|bhujangasana|भुजंगासन|कोबरा",
    "warrior_2": r"warrior (two|2|ii|too|to)\b|virabhadrasana (two|2|ii)|वारियर (टू|2|दो)|वीरभद्रासन (२|2|दो)",
    "warrior_1": r"warrior (one|1|i)\b|virabhadrasana (one|1|i)\b",
    "downward_dog": r"down(ward)?[- ]?(facing )?dog|adho mukha|अधोमुख|डाउनवर्ड",
    "child_pose": r"child'?s? pose|\bbalasana\b|बालासन|चाइल्ड",
    "tree_pose": r"tree pose|vrik?sh?asana|वृक्षासन|ट्री पोज",
    "triangle": r"triangle|trikonasana|त्रिकोणासन",
    "chair_pose": r"chair pose|utkatasana|उत्कटासन",
    "standing_forward_fold": r"forward (fold|bend)|uttanasana|उत्तानासन",
    "seated_easy_pose": r"easy (seat|pose)|sukhasana|cross[- ]?legged|सुखासन",
    "seated_staff": r"staff pose|dandasana|दंडासन",
    "corpse": r"corpse|sh?avasana|शवासन",
    "plank": r"\bplank\b|phalakasana",
    "upward_salute": r"upward salute|urdhva hastasana",
    "lunge_pose": r"\blunge\b|crescent",
    "table_top": r"table ?top|all fours",
    "upward_dog": r"up(ward)?[- ]?(facing )?dog|urdhva mukha",
}


def _ang(a, b, c):
    import numpy as np
    ba, bc = a - b, c - b
    nb, nc = np.linalg.norm(ba, axis=1), np.linalg.norm(bc, axis=1)
    with np.errstate(all="ignore"):
        cos = np.clip((ba * bc).sum(1) / (nb * nc), -1.0, 1.0)
    out = np.degrees(np.arccos(cos))
    out[(nb == 0) | (nc == 0)] = 180.0
    return out


def angles_zero_z(c):
    """Vectorised twin of the original extractor with z zeroed (c = interpolated landmarks)."""
    import numpy as np
    p = c[:, :, :3].copy()
    p[:, :, 2] = 0.0
    g = lambda i: p[:, i, :]  # noqa: E731
    sm, hm = (g(11) + g(12)) / 2.0, (g(23) + g(24)) / 2.0
    cols = [_ang(g(11), g(13), g(15)), _ang(g(12), g(14), g(16)), _ang(g(23), g(11), g(13)), _ang(g(24), g(12), g(14)),
            _ang(g(11), g(23), g(25)), _ang(g(12), g(24), g(26)), _ang(g(23), g(25), g(27)), _ang(g(24), g(26), g(28)),
            _ang(g(25), g(27), g(29)), _ang(g(26), g(28), g(30)), _ang(g(11), g(23), g(24)), _ang(g(12), g(24), g(23)),
            _ang(g(0), sm, hm), _ang(g(24), g(23), g(25)), _ang(g(23), g(24), g(26))]
    return np.stack(cols, axis=1).astype(np.float32)


def clean_runs(lab, min_run):
    import numpy as np
    lab = lab.copy()
    i, n = 0, len(lab)
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        if lab[i] != "transition/unknown" and j - i < min_run:
            lab[i:j] = "transition/unknown"
        i = j
    return lab


@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=12288, timeout=3600)
def label_one(vid: str):
    import numpy as np
    sys.path.insert(0, "/srv"); sys.path.insert(0, "/opt/orig")
    import extract_features_safe as fe
    from app.utils.geometry import compute_orientation
    from app.utils.rules_classifier import classify_pose

    vol.reload()
    try:
        tr = json.load(open(f"{VOL}/transcripts/{vid}.json"))
        lp = f"{VOL}/landmarks/landmarks_{vid}.npy" if vid in OLD else f"{VOL}/landmarks_new/landmarks_{vid}.npy"
        lm = np.load(lp)
        n = len(lm)
        dur = tr["duration"]
        spf = dur / n  # seconds per frame (robust to fps rounding)
        c = fe.interpolate_occlusions(lm, visibility_threshold=0.5)
        a = angles_zero_z(c)
        # rule-engine call per frame, exactly as the app does (angles dict + orientation)
        rule = np.empty(n, dtype=object)
        for t in range(n):
            ad = dict(zip(ANG, a[t].tolist()))
            rule[t] = classify_pose(ad, compute_orientation(c[t]))
        # hold = steady angles
        k = 7
        v = np.full(n, 1e9, dtype=np.float32)
        if n > 2 * k:
            v[k:n - k] = np.abs(a[2 * k:] - a[:n - 2 * k]).mean(axis=1)
        stable = v < STABLE_DEG
        # cues
        cues = []
        for ch in tr["chunks"]:
            txt = re.sub(r"[^\w\s'ऀ-ॿ]", " ", ch["text"].lower().replace("-", " "))
            for pose, rx in CUES.items():
                if re.search(rx, txt):
                    cues.append((pose, ch["t0"], ch["t1"]))
        lab = np.full(n, "transition/unknown", dtype=object)
        win = {}
        for pose, t0, t1 in cues:
            f0, f1 = max(0, int((t0 - PRE_S) / spf)), min(n, int((t1 + POST_S) / spf) + 1)
            win.setdefault(pose, []).append((f0, f1))
        for pose, spans in win.items():
            m = np.zeros(n, dtype=bool)
            for f0, f1 in spans:
                m[f0:f1] = True
            hit = m & (rule == pose) & stable
            lab[hit] = pose
        raw_pos = lab.astype("U32")
        pos = clean_runs(raw_pos, MIN_RUN)                    # short positive blips are dropped...
        dropped_blip = (raw_pos != "transition/unknown") & (pos == "transition/unknown")
        # three-way split: positive pose | transition (moving, or steady with no pose recognised) | excluded (ambiguous)
        neg = (~stable) | (rule == "transition/unknown")
        lab = np.where(pos != "transition/unknown", pos,
                       np.where(neg & ~dropped_blip, "transition/unknown", "__ignore__")).astype("U32")
        os.makedirs(f"{VOL}/cue", exist_ok=True)
        np.savez_compressed(f"{VOL}/cue/{vid}.npz", labels=lab, rule=rule.astype("U32"), stable=stable)
        # angle-speed percentiles help justify the STABLE_DEG choice
        cnt = {p: int((lab == p).sum()) for p in sorted(set(lab.tolist())) if p not in ("transition/unknown", "__ignore__")}
        cue_n = {}
        for p, _, _ in cues:
            cue_n[p] = cue_n.get(p, 0) + 1
        vol.commit()
        return {"id": vid, "ok": True, "frames": n, "frames_vs_transcript": [n, tr.get("n_video_frames")],
                "cues": cue_n, "labelled_frames": cnt, "labelled_seconds": {p: round(c_ * spf, 1) for p, c_ in cnt.items()},
                "frac_positive": round(float(np.isin(lab, list(cnt)).mean()), 4),
                "frac_transition": round(float((lab == "transition/unknown").mean()), 4),
                "frac_excluded": round(float((lab == "__ignore__").mean()), 4),
                "speed_p25_p50_p75": [round(float(np.percentile(v[v < 1e8], q)), 2) for q in (25, 50, 75)]}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"id": vid, "ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-400:]}


@app.local_entrypoint()
def main(only: str = ""):
    ids = only.split() if only else OLD + NEW
    res = []
    for r in label_one.map(ids):
        print(json.dumps(r, ensure_ascii=False), flush=True)
        res.append(r)
    os.makedirs("/tmp", exist_ok=True)
    tot = {}
    for r in res:
        for p, c in (r.get("labelled_seconds") or {}).items():
            tot[p] = round(tot.get(p, 0) + c, 1)
    print("TOTAL labelled seconds per pose:", json.dumps(dict(sorted(tot.items(), key=lambda kv: -kv[1]))))
