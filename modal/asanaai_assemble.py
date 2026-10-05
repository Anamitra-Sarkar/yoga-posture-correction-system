"""Assemble training sets from cue-verified labels + real photos.

  csv/cue/cue_full.csv        every frame of all 24 videos (zero-z angles, master schema); labels incl. '__ignore__'
  csv/cue/mlp_fold{k}.csv     MLP set: cue-positive video frames + transition negatives + photos NOT in fold k (replicated)
  csv/cue/mlp_all.csv         same, all 422 photos (the deployable model)
  folds/photo_folds.json      photo index -> fold (stratified, seeded) so out-of-fold scoring covers every photo
Photos are never seen by the model that scores them.
"""
import json
import os
import sys

import modal

app = modal.App("asanaai-assemble")
SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
image = (modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2", "pandas")
         .add_local_file(f"{SRC}/extract_features_safe.py", "/opt/orig/extract_features_safe.py", copy=True))
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
ANG = ["elbow_l", "elbow_r", "shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r",
       "ankle_l", "ankle_r", "trunk_l", "trunk_r", "neck", "hip_abduct_l", "hip_abduct_r"]
OLD = "HmZFwoUU3WQ SZU7Sbgu57o oUgpXY7QhpQ 7ciS93shMNQ P8uHMMmWMHQ 4ORRiN2_aVI RQMtwbhXD7A 8ibxmzJziHU Eml2xnoLpYE QiebZSlTw_U L-z1HLkS_-Y s-1vMbAgYWU".split()
NEW = ("v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk hHhxKkskHDg JHjV-wFTwSw "
       "EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow 4ZBUDd4bsyA").split()
K = 5
VIDEO_STRIDE, CLASS_CAP, NEG_RATIO, PHOTO_SHARE, REP_MIN, REP_MAX = 2, 30000, 1.0, 0.25, 20, 300


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


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=2 * 3600)
def assemble(min_positive_frames: int = 300):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/opt/orig")
    import extract_features_safe as fe

    vol.reload()
    rng = np.random.default_rng(42)
    os.makedirs(f"{VOL}/csv/cue", exist_ok=True)
    os.makedirs(f"{VOL}/folds", exist_ok=True)

    # ---- video frames
    parts = []
    ids = [v for v in OLD + NEW if os.path.exists(f"{VOL}/cue/{v}.npz")]
    print("videos with cue labels:", len(ids), ids, flush=True)
    for vid in ids:
        lp = f"{VOL}/landmarks/landmarks_{vid}.npy" if vid in OLD else f"{VOL}/landmarks_new/landmarks_{vid}.npy"
        lm = np.load(lp)
        lab = np.load(f"{VOL}/cue/{vid}.npz")["labels"]
        assert len(lab) == len(lm), (vid, len(lab), len(lm))
        a = angles_zero_z(fe.interpolate_occlusions(lm, visibility_threshold=0.5))
        d = pd.DataFrame(a, columns=ANG)
        d.insert(0, "video_id", vid)
        d.insert(1, "frame_num", np.arange(1, len(lm) + 1))
        d.insert(2, "difficulty", "cue")
        d["pose_label"] = lab
        d["imperfect_pose_label"] = lab
        parts.append(d)
        print(vid, len(d), flush=True)
    full = pd.concat(parts, ignore_index=True)
    full.to_csv(f"{VOL}/csv/cue/cue_full.csv", index=False)
    cnt = full.pose_label.value_counts()
    print(cnt.to_string(), flush=True)

    # ---- which poses have enough cue-verified video support to be classes
    pos_cnt = cnt.drop(labels=["transition/unknown", "__ignore__"], errors="ignore")
    video_support = {p: int(full[full.pose_label == p].video_id.nunique()) for p in pos_cnt.index}
    keep = [p for p in pos_cnt.index if pos_cnt[p] >= min_positive_frames]

    # ---- MLP video rows: positives (stride, cap) + transition negatives
    vid_rows = []
    for p in keep:
        r = full[full.pose_label == p].iloc[::VIDEO_STRIDE]
        if len(r) > CLASS_CAP:
            r = r.iloc[np.sort(rng.choice(len(r), CLASS_CAP, replace=False))]
        vid_rows.append(r)
    pos_df = pd.concat(vid_rows, ignore_index=True)
    neg_all = full[full.pose_label == "transition/unknown"]
    neg_df = neg_all.iloc[np.sort(rng.choice(len(neg_all), min(len(neg_all), int(NEG_RATIO * len(pos_df))), replace=False))]
    video_mlp = pd.concat([pos_df, neg_df], ignore_index=True)

    # ---- photos (real, human-labelled); angles by the app's recipe (z zeroed, no interpolation: single images)
    z = np.load(f"{VOL}/eval/photo_corpus.npz", allow_pickle=True)
    plm, plab = z["landmarks"], z["labels"].astype(str)
    pang = np.concatenate([angles_zero_z(plm[i:i + 1]) for i in range(len(plm))])
    fold = np.zeros(len(plm), dtype=int)
    for c in sorted(set(plab.tolist())):
        idx = rng.permutation(np.where(plab == c)[0])
        for j, i in enumerate(idx):
            fold[i] = j % K
    json.dump({"fold": fold.tolist(), "labels": plab.tolist()}, open(f"{VOL}/folds/photo_folds.json", "w"))

    def photo_rows(mask):
        sel = np.where(mask)[0]
        out = []
        for c in sorted(set(plab[sel].tolist())):
            ci = sel[plab[sel] == c]
            vid_n = int((video_mlp.pose_label == c).sum())
            rep = int(np.clip(round(PHOTO_SHARE / (1 - PHOTO_SHARE) * max(vid_n, 1) / max(len(ci), 1)), REP_MIN, REP_MAX))
            for r in range(rep):
                d = pd.DataFrame(pang[ci], columns=ANG)
                d.insert(0, "video_id", [f"photo_{i}" for i in ci])
                d.insert(1, "frame_num", r + 1)
                d.insert(2, "difficulty", "photo")
                d["pose_label"] = c
                d["imperfect_pose_label"] = c
                out.append(d)
        return pd.concat(out, ignore_index=True)

    rep = {"keep_from_video": keep, "video_support_videos": video_support,
           "video_positive_rows": {p: int((video_mlp.pose_label == p).sum()) for p in keep},
           "transition_rows": int(len(neg_df)), "photo_class_counts": {c: int((plab == c).sum()) for c in sorted(set(plab))}}
    for k in list(range(K)) + ["all"]:
        mask = np.ones(len(plab), dtype=bool) if k == "all" else fold != k
        ph = photo_rows(mask)
        df = pd.concat([video_mlp, ph], ignore_index=True)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df.to_csv(f"{VOL}/csv/cue/mlp_{'all' if k == 'all' else 'fold' + str(k)}.csv", index=False)
        rep[f"rows_{k}"] = int(len(df))
        rep[f"classes_{k}"] = int(df.imperfect_pose_label.nunique())
    json.dump(rep, open(f"{VOL}/csv/cue/assemble_report.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(rep, indent=1))
    return rep


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3600)
def assemble_v2(variant: str, neg_ratio: float = 0.3, pub_rep: int = 2, use_oov: bool = True, oov_cap: int = 4000,
                class_cap: int = 8000, folds: bool = False, min_positive_frames: int = 300, mirror: bool = False,
                commons: bool = True, harvest: bool = False, targets: str = ""):
    """cue-verified video positives + transition negatives + public real photos (train split only) +
    out-of-vocabulary real photos as 'none of our poses' + the 422 Commons photos (fold-wise).
    The public TEST split never enters any training set: it is the tuning/validation set."""
    import numpy as np
    import pandas as pd
    vol.reload()
    rng = np.random.default_rng(42)
    full = pd.read_csv(f"{VOL}/csv/cue/cue_full.csv")
    T = [t for t in targets.split(",") if t]
    def to_target(lab):  # non-target poses become "other" (= transition/unknown); __ignore__ stays excluded
        lab = np.asarray(lab).astype(object).copy()
        if T:
            lab = np.array([l if (l in T or l in ("transition/unknown", "__ignore__")) else "transition/unknown" for l in lab], dtype=object)
        return lab
    if T:
        full["pose_label"] = to_target(full.pose_label.values); full["imperfect_pose_label"] = full["pose_label"]
    cnt = full.pose_label.value_counts().drop(labels=["transition/unknown", "__ignore__"], errors="ignore")
    keep = [p for p in cnt.index if cnt[p] >= min_positive_frames]
    parts = []
    for p in keep:
        r = full[full.pose_label == p].iloc[::VIDEO_STRIDE]
        if len(r) > class_cap:
            r = r.iloc[np.sort(rng.choice(len(r), class_cap, replace=False))]
        parts.append(r)
    pos = pd.concat(parts, ignore_index=True)
    neg_all = full[full.pose_label == "transition/unknown"]
    neg = neg_all.iloc[np.sort(rng.choice(len(neg_all), min(len(neg_all), int(neg_ratio * len(pos))), replace=False))]
    video = pd.concat([pos, neg], ignore_index=True)

    pc = np.load(f"{VOL}/photos/public_corpus.npz", allow_pickle=True)
    tr = pc["split"] == "train"
    plab = to_target(pc["labels"].astype(str)) if T else pc["labels"]
    pang = np.concatenate([angles_zero_z(pc["landmarks"][i:i + 1]) for i in range(len(plab))])

    def rows(idx, labels, ang, prefix, rep):
        out = []
        for r in range(rep):
            d = pd.DataFrame(ang[idx], columns=ANG)
            d.insert(0, "video_id", [f"{prefix}{i}" for i in idx])
            d.insert(1, "frame_num", r + 1)
            d.insert(2, "difficulty", "photo")
            d["pose_label"] = labels[idx]
            d["imperfect_pose_label"] = labels[idx]
            out.append(d)
        return pd.concat(out, ignore_index=True)

    pub_pos = np.where(tr & (plab != "transition/unknown"))[0]
    pub_neg = np.where(tr & (plab == "transition/unknown"))[0]
    if len(pub_neg) > oov_cap:
        pub_neg = rng.choice(pub_neg, oov_cap, replace=False)
    pub_df = rows(pub_pos, plab, pang, "pub_", pub_rep)
    oov_df = rows(pub_neg, plab, pang, "oov_", 1) if use_oov and len(pub_neg) else None

    z = np.load(f"{VOL}/eval/photo_corpus.npz", allow_pickle=True)
    clab = to_target(z["labels"].astype(str)).astype(str) if T else z["labels"].astype(str)
    cang = np.concatenate([angles_zero_z(z["landmarks"][i:i + 1]) for i in range(len(clab))])
    fold = np.array(json.load(open(f"{VOL}/folds/photo_folds.json"))["fold"])
    base_n = len(video) + len(pub_df)
    outdir = f"{VOL}/csv/cue2/{variant}"
    os.makedirs(outdir, exist_ok=True)
    rep = {"variant": variant, "neg_ratio": neg_ratio, "pub_rep": pub_rep, "use_oov": use_oov,
           "video_pos_rows": {p: int((video.pose_label == p).sum()) for p in keep}, "video_neg_rows": int(len(neg)),
           "public_train_pos": {c: int((plab[pub_pos] == c).sum()) for c in sorted(set(plab[pub_pos]))},
           "oov_rows": int(len(pub_neg)) if use_oov else 0}
    for k in (list(range(K)) if folds else []) + ["all"]:
        sel = np.ones(len(clab), bool) if k == "all" else fold != k
        ci_all = np.where(sel)[0]
        parts = [video, pub_df] + ([oov_df] if oov_df is not None else [])
        for c in (sorted(set(clab[ci_all].tolist())) if commons else []):
            ci = ci_all[clab[ci_all] == c]
            r = int(np.clip(round(0.15 * base_n / len(set(clab)) / max(len(ci), 1)), 10, 100))
            parts.append(rows(ci, clab, cang, "commons_", r))
        if harvest and os.path.exists(f"{VOL}/photos/harvest_kept.npz"):
            hk = np.load(f"{VOL}/photos/harvest_kept.npz", allow_pickle=True)
            htr = np.where(hk["split"] == "train")[0]
            hang = np.concatenate([angles_zero_z(hk["landmarks"][i:i + 1]) for i in range(len(hk["labels"]))])
            parts.append(rows(htr, (to_target(hk["labels"].astype(str)) if T else hk["labels"].astype(str)), hang, "harv_", 2))
        df = pd.concat(parts, ignore_index=True)
        if mirror:  # left-right twin of every row: swap each _l/_r angle pair (neck is symmetric)
            m = df.copy()
            for a in ANG:
                if a.endswith("_l"):
                    b = a[:-2] + "_r"
                    m[a], m[b] = df[b].values, df[a].values
            m["video_id"] = m["video_id"].astype(str) + "_m"
            df = pd.concat([df, m], ignore_index=True)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        name = "mlp_all.csv" if k == "all" else f"mlp_fold{k}.csv"
        df.to_csv(f"{outdir}/{name}", index=False)
        rep[f"rows_{k}"] = int(len(df))
        rep[f"class_rows_{k}"] = df.imperfect_pose_label.value_counts().to_dict() if k == "all" else None
    json.dump(rep, open(f"{outdir}/assemble_report.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(rep, indent=1))
    return rep


@app.local_entrypoint()
def main(min_positive_frames: int = 300, v2: str = "", neg_ratio: float = 0.3, pub_rep: int = 2, oov: int = 1, folds: int = 0, mirror: int = 0, commons: int = 1, harvest: int = 0, targets: str = ""):
    if v2:
        assemble_v2.remote(v2, neg_ratio, pub_rep, bool(oov), 4000, 8000, bool(folds), 300, bool(mirror), bool(commons), bool(harvest), targets)
    else:
        assemble.remote(min_positive_frames)
