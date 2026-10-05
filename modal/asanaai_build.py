"""Build the two training CSVs: old 654k master + the 12 new videos.

LABELS come from the ORIGINAL 3-stage chain, run in the space the master CSV lives in (RAW-z):
  1. RandomForest(100, balanced, seed 42) on the hand-labelled cleaned CSV + clean_labels(min_duration=15)
     (compile_master_dataset.py, Case B)
  2. imperfect_* labels from centroids of {triangle, plank, seated_forward, upward_dog, corpse}, 90 deg threshold
     (experiments/compile_imperfect_dataset.py, logic copied verbatim)
  3. rule cascade over remaining transition/unknown (experiments/classify_all_movements.py, called UNMODIFIED)

Variant A (faithful):      old master as-is (raw-z angles) + new rows with raw-z angles.
Variant B (serve-matched): same rows, same labels, but EVERY angle column recomputed with z zeroed, i.e. the
                           feature space backend/app/utils/geometry.py feeds the model at inference.
Only the 15 angle columns differ between A and B; labels are byte-identical.
"""
import json
import os
import sys

import modal

SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
app = modal.App("asanaai-build")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "pandas", "scikit-learn")
    .add_local_file(f"{SRC}/extract_features_safe.py", "/opt/orig/extract_features_safe.py", copy=True)
    .add_local_file(f"{SRC}/compile_master_dataset.py", "/opt/orig/compile_master_dataset.py", copy=True)
    .add_local_file(f"{SRC}/experiments/classify_all_movements.py", "/opt/orig/classify_all_movements.py", copy=True)
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
ANG = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r",
       "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck", "hip_abduct_l", "hip_abduct_r"]
NEW_IDS = ("v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk "
           "hHhxKkskHDg JHjV-wFTwSw EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow 4ZBUDd4bsyA").split()
KEEP = ["mountain_pose", "cobra_pose", "warrior_2"]


def _ang(a, b, c):
    import numpy as np
    ba, bc = a - b, c - b
    dot = (ba * bc).sum(1)
    nb, nc = np.linalg.norm(ba, axis=1), np.linalg.norm(bc, axis=1)
    with np.errstate(all="ignore"):
        cos = np.clip(dot / (nb * nc), -1.0, 1.0)
    out = np.degrees(np.arccos(cos))
    out[(nb == 0) | (nc == 0)] = 180.0
    return out


def angles_vec(lm, zero_z, fe):
    """Vectorised twin of extract_features_safe.extract_features_from_landmarks (float32, same math)."""
    import numpy as np
    c = fe.interpolate_occlusions(lm, visibility_threshold=0.5)
    p = c[:, :, :3].copy()
    if zero_z:
        p[:, :, 2] = 0.0
    g = lambda i: p[:, i, :]  # noqa: E731
    sm, hm = (g(11) + g(12)) / 2.0, (g(23) + g(24)) / 2.0
    cols = [_ang(g(11), g(13), g(15)), _ang(g(12), g(14), g(16)), _ang(g(23), g(11), g(13)),
            _ang(g(24), g(12), g(14)), _ang(g(11), g(23), g(25)), _ang(g(12), g(24), g(26)),
            _ang(g(23), g(25), g(27)), _ang(g(24), g(26), g(28)), _ang(g(25), g(27), g(29)),
            _ang(g(26), g(28), g(30)), _ang(g(11), g(23), g(24)), _ang(g(12), g(24), g(23)),
            _ang(g(0), sm, hm), _ang(g(24), g(23), g(25)), _ang(g(23), g(24), g(26))]
    return np.stack(cols, axis=1).astype(np.float32)


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=28672, timeout=2 * 3600)
def build(tag: str = "full", ids: list = None):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    sys.path.insert(0, "/opt/orig")
    import extract_features_safe as fe
    import compile_master_dataset as cmd
    import classify_all_movements as cam

    vol.reload()
    ids = ids or [i for i in NEW_IDS if os.path.exists(f"{VOL}/landmarks_new/landmarks_{i}.npy")]
    outdir = f"{VOL}/csv/build_{tag}"
    os.makedirs(outdir, exist_ok=True)
    rep = {"ids": ids}

    # ---- sanity: vectorised angles == original per-frame function, and == the live CSV (raw z)
    master = pd.read_csv(f"{VOL}/csv/master_mlp_dataset_fully_classified.csv")
    assert list(master.columns) == ["video_id", "frame_num", "difficulty"] + ANG + ["pose_label", "imperfect_pose_label"]
    rep["master_rows"] = int(len(master))
    old_ids = sorted(master.video_id.unique())
    lm0 = np.load(f"{VOL}/landmarks/landmarks_{old_ids[0]}.npy")
    tmp = "/tmp/_t.npy"; np.save(tmp, lm0[:3000])
    ref_zero = fe.extract_features_from_landmarks(tmp)[ANG].values  # ORIGINAL (zero-z) per-frame function
    rep["vec_vs_original_zeroz_maxabs"] = float(np.abs(angles_vec(lm0[:3000], True, fe) - ref_zero).max())
    full = angles_vec(lm0, False, fe)  # whole video: interpolation is video-global in the original
    mfull = master[master.video_id == old_ids[0]].sort_values("frame_num")[ANG].values
    dd = np.abs(full - mfull)
    rep["vec_rawz_vs_csv_p999"] = float(np.percentile(dd, 99.9))
    rep["vec_rawz_vs_csv_max"] = float(dd.max())
    rep["vec_rawz_vs_csv_median"] = float(np.median(dd))
    print("sanity", rep, flush=True)
    assert rep["vec_vs_original_zeroz_maxabs"] < 1e-3 and rep["vec_rawz_vs_csv_p999"] < 1e-2

    # ---- hand-labelled set in the MASTER's raw-z space.
    # regen_gt_zero_z.py rebuilt mlp_dataset_cleaned_zero_z.csv from the SAME landmark files as the master, row for row,
    # keeping the original hand labels. Join those labels to the master's own raw-z angles by (video_id, frame_num).
    cl_zero = pd.read_csv(f"{VOL}/labelsrc/mlp_dataset_cleaned_zero_z.csv")
    labeller_csv = cl_zero[["video_id", "frame_num", "pose_label"]].merge(
        master[["video_id", "frame_num"] + ANG], on=["video_id", "frame_num"], how="inner")
    rep["labeller_rows"] = int(len(labeller_csv))
    rep["labeller_classes"] = labeller_csv.pose_label.value_counts().to_dict()
    rep["labeller_space"] = "raw"
    print("labeller", rep["labeller_rows"], rep["labeller_classes"], flush=True)

    # ---- stage 1: RandomForest on hand-labelled frames (original recipe)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(labeller_csv[ANG].values, labeller_csv["pose_label"].values)

    rows_raw, rows_zero, per_video = [], [], {}
    for vid in ids:
        lm = np.load(f"{VOL}/landmarks_new/landmarks_{vid}.npy")
        a_raw, a_zero = angles_vec(lm, False, fe), angles_vec(lm, True, fe)
        feats = a_raw if rep["labeller_space"] == "raw" else a_zero
        lab = clf.predict(feats)
        lab = cmd.clean_labels(lab, min_duration=15)
        base = pd.DataFrame({"video_id": vid, "frame_num": np.arange(1, len(lm) + 1), "difficulty": "advanced"})
        d_raw = pd.concat([base, pd.DataFrame(a_raw, columns=ANG)], axis=1)
        d_raw["pose_label"] = lab
        # stage 2: imperfect_* via centroids (verbatim logic of compile_imperfect_dataset.py)
        d_raw["imperfect_pose_label"] = d_raw["pose_label"]
        targets = ["triangle", "plank", "seated_forward", "upward_dog", "corpse"]
        cen = {t: labeller_csv[labeller_csv.pose_label == t][ANG].mean().values for t in targets
               if (labeller_csv.pose_label == t).any()}
        tm = d_raw["pose_label"] == "transition/unknown"
        X = d_raw.loc[tm, ANG].values
        if cen and len(X):
            names = list(cen)
            D = np.stack([np.linalg.norm(X - cen[n], axis=1) for n in names], axis=1)
            near, dist = D.argmin(1), D.min(1)
            d_raw.loc[tm, "imperfect_pose_label"] = [f"imperfect_{names[k]}" if v < 90.0 else "transition/unknown"
                                                     for k, v in zip(near, dist)]
        per_video[vid] = {"frames": int(len(d_raw)),
                          "after_stage1": d_raw["pose_label"].value_counts().head(12).to_dict()}
        # stage 3: ORIGINAL rule cascade, unmodified, output path redirected
        cam.OUTPUT_CSV = "/tmp/_cam_out.csv"
        cam.classify_frames(d_raw)
        per_video[vid]["final"] = d_raw["imperfect_pose_label"].value_counts().head(14).to_dict()
        d_zero = d_raw.copy()
        d_zero[ANG] = a_zero
        rows_raw.append(d_raw); rows_zero.append(d_zero)

    new_raw, new_zero = pd.concat(rows_raw, ignore_index=True), pd.concat(rows_zero, ignore_index=True)
    assert (new_raw["imperfect_pose_label"].values == new_zero["imperfect_pose_label"].values).all()
    new_raw.to_csv(f"{outdir}/new_rows_A.csv", index=False)
    new_zero.to_csv(f"{outdir}/new_rows_B.csv", index=False)

    # ---- variant A: old master untouched + new raw-z rows
    A = pd.concat([master, new_raw[master.columns]], ignore_index=True)
    A.to_csv(f"{outdir}/master_plus_new_A.csv", index=False)
    # ---- variant B: every old row's angles recomputed with z zeroed from the stored landmarks
    B = master.copy()
    for vid in old_ids:
        lm = np.load(f"{VOL}/landmarks/landmarks_{vid}.npy")
        az = angles_vec(lm, True, fe)
        idx = B.index[B.video_id == vid]
        B.loc[idx, ANG] = az[B.loc[idx, "frame_num"].values - 1]
    B = pd.concat([B, new_zero[master.columns]], ignore_index=True)
    assert (A["imperfect_pose_label"].values == B["imperfect_pose_label"].values).all()
    B.to_csv(f"{outdir}/master_plus_new_B.csv", index=False)

    dist_old = master["imperfect_pose_label"].value_counts()
    dist_new = new_raw["imperfect_pose_label"].value_counts()
    rep.update(rows_A=int(len(A)), rows_B=int(len(B)), new_rows=int(len(new_raw)), per_video=per_video,
               dist_old=dist_old.head(30).to_dict(), dist_new=dist_new.head(30).to_dict(),
               keepers={k: {"old": int(dist_old.get(k, 0)), "new": int(dist_new.get(k, 0))} for k in KEEP})
    json.dump(rep, open(f"{outdir}/build_report.json", "w"), indent=1, default=str)
    vol.commit()
    print(json.dumps({k: rep[k] for k in ["rows_A", "rows_B", "new_rows", "keepers", "labeller_space"]}, indent=1))
    return rep


@app.local_entrypoint()
def main(tag: str = "full", only: str = ""):
    build.remote(tag, only.split() if only else None)


@app.function(image=image, volumes={VOL: vol}, timeout=600)
def show(tag: str = "test"):
    """Print a compact summary of a build report from inside Modal (nothing is downloaded locally)."""
    vol.reload()
    r = json.load(open(f"{VOL}/csv/build_{tag}/build_report.json"))
    print("rows A/B/new:", r["rows_A"], r["rows_B"], r["new_rows"])
    print("labeller classes:", r["labeller_classes"])
    print("OLD master dist:", r["dist_old"])
    print("NEW rows dist:", r["dist_new"])
    for v, d in r["per_video"].items():
        print(v, d["frames"], "stage1:", d["after_stage1"], "\n     final:", d["final"])


@app.local_entrypoint()
def report(tag: str = "test"):
    show.remote(tag)
