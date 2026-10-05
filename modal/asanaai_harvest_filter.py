"""Clean the Commons/Openverse harvest before it is allowed anywhere near training.
Harvest labels are SEARCH-QUERY labels (weak). A photo is kept only if
  (a) it is NOT a duplicate (pixel hash vs public corpus / other harvest classes, Commons id vs the 422-photo benchmark), and
  (b) the app's rule engine calls it the queried pose (or a close sibling), OR a model that never saw Commons photos
      (cue2_mlp_vN) ranks the queried pose in its top-2.
"""
import collections
import json
import os
import sys

import modal

B = "/home/anamitra/yoga_posture_workspace/backend/app"
app = modal.App("asanaai-harvest-filter")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy<2")
    .add_local_file(f"{B}/utils/geometry.py", "/srv/app/utils/geometry.py", copy=True)
    .add_local_file(f"{B}/utils/rules_classifier.py", "/srv/app/utils/rules_classifier.py", copy=True)
    .add_local_dir(f"{B}/models", remote_path="/models_src", copy=True)
    .add_local_file("/home/anamitra/yoga_posture_workspace/planning/photo_corpus/photo_corpus_manifest.json",
                    "/opt/commons_manifest.json", copy=True)
    .run_commands("touch /srv/app/__init__.py /srv/app/utils/__init__.py")
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
COMPAT = {"cobra_pose": {"upward_dog"}, "upward_dog": {"cobra_pose"}, "warrior_2": {"warrior_1", "lunge_pose"},
          "warrior_1": {"warrior_2", "lunge_pose"}, "lunge_pose": {"warrior_1", "warrior_2"},
          "mountain_pose": {"standing_pose", "upward_salute"}, "standing_pose": {"mountain_pose"},
          "upward_salute": {"mountain_pose"}, "standing_forward_fold": {"halfway_lift"}, "halfway_lift": {"standing_forward_fold"},
          "plank": {"chaturanga"}, "chaturanga": {"plank"}, "seated_easy_pose": {"seated_staff"},
          "seated_staff": {"seated_easy_pose"}, "child_pose": {"table_top"}, "table_top": {"child_pose"}}
ANG = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r",
       "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck", "hip_abduct_l", "hip_abduct_r"]


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=3600)
def filt(model_tag: str = "cue2_mlp_vN"):
    import glob
    import numpy as np
    import torch
    sys.path.insert(0, "/srv"); sys.path.insert(0, "/models_src")
    from app.utils.geometry import compute_orientation, extract_angles_from_landmarks
    from app.utils.rules_classifier import classify_pose
    from mlp import Yoga3HeadMLP
    vol.reload()
    cls = list(np.load(f"{VOL}/runs/{model_tag}/mlp_3head_pose_encoder_v2.npy", allow_pickle=True))
    m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls))
    m.load_state_dict(torch.load(f"{VOL}/runs/{model_tag}/mlp_3head_model_v2.pth", map_location="cpu")); m.eval()
    base = [c.replace("imperfect_", "") for c in cls]
    commons_ids = {str(e.get("id") or e.get("title")) for e in
                   (lambda j: j if isinstance(j, list) else (j.get("meta") or []))(json.load(open("/opt/commons_manifest.json")))}
    pub_keys = set(np.load(f"{VOL}/photos/public_corpus.npz", allow_pickle=True)["keys"].astype(str).tolist())
    L, Y, S, ID, WHY, seen = [], [], [], [], [], set()
    rep = {}
    for f in sorted(glob.glob(f"{VOL}/photos/harvest/*.npz")):
        z = np.load(f, allow_pickle=True)
        label = str(z["label"])
        meta = json.load(open(f.replace(".npz", "_meta.json")))
        r = collections.Counter()
        for lm, key, md in zip(z["landmarks"], z["keys"].astype(str), meta):
            if md["id"] in commons_ids:
                r["dup_commons_benchmark"] += 1; continue
            if key in pub_keys or key in seen:
                r["dup_pixels"] += 1; continue
            seen.add(key)
            ang = extract_angles_from_landmarks(lm[:, :3].copy(), zero_z=True)
            rule = classify_pose(dict(zip(ANG, ang)), compute_orientation(lm[:, :3]))
            with torch.no_grad():
                top2 = torch.softmax(m(torch.tensor([ang], dtype=torch.float32))[0], 1)[0].topk(2).indices.tolist()
            top2 = [base[i] for i in top2]
            why = "rule" if rule == label else "rule_sibling" if rule in COMPAT.get(label, ()) else "model_top2" if label in top2 else None
            if why is None:
                r["dropped_label_disagrees"] += 1; continue
            r[why] += 1
            L.append(lm); Y.append(label); S.append(md["split"]); ID.append(md["id"]); WHY.append(why)
        rep[label] = dict(r)
    np.savez_compressed(f"{VOL}/photos/harvest_kept.npz", landmarks=np.array(L, dtype=np.float32), labels=np.array(Y),
                        split=np.array(S), ids=np.array(ID), why=np.array(WHY))
    rep["_total_kept"] = len(Y)
    rep["_kept_per_class"] = dict(collections.Counter(Y).most_common())
    json.dump(rep, open(f"{VOL}/photos/harvest_filter_report.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(rep, indent=1))
    return rep


@app.local_entrypoint()
def main(model_tag: str = "cue2_mlp_vN"):
    filt.remote(model_tag)
