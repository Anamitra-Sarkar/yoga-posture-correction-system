"""Locks in WHY mlp_3head_photodomain_v1 replaced mlp_3head_model_v2 as the
live checkpoint on 2026-09-20, and re-verifies it against a live HF pull
rather than trusting a cached number.

The switch was triggered by re-measuring mlp_3head_model_v2 -- the checkpoint
that had been live since 2026-09-03 -- against the app's actual judging
condition: a single held-out 103-photo set spanning the full 23-class
vocabulary. It scored 10.5% macro / 13.6% overall. That is materially worse
than the 52.6% headline this project had been quoting, because that number
covered only 6 well-supported poses on a 19-photo set, not the full
vocabulary the app is actually judged on. mlp_3head_photodomain_v1 -- trained
on the same video corpus plus oversampled real photographs -- was measured on
the IDENTICAL 103-photo set and scored 35.5% macro / 45.6% overall, a >3x
macro gain.

Requires network (downloads from the public Arko007/yoga-posture-models HF
repo) and the local frozen photo-corpus test split, so it is skipped rather
than failed when either is unavailable -- this is a re-verification test, not
a pure-function unit test, and should not block an offline `pytest tests/`.
"""
import glob
import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

PHOTO_CORPUS = os.path.join(
    os.path.dirname(__file__), "..", "..", "planning", "photo_corpus", "photo_corpus.npz"
)
HF_REPO = "Arko007/yoga-posture-models"


def _hf_token():
    for path in (
        os.path.expanduser("~/Downloads/API_Keys_and_Secrets/hf_token"),
        os.environ.get("HF_TOKEN", ""),
    ):
        if path and os.path.exists(path):
            return open(path).read().strip()
        if path and not os.path.exists(path):
            continue
    return os.environ.get("HF_TOKEN")


pytestmark = pytest.mark.skipif(
    not os.path.exists(PHOTO_CORPUS), reason="frozen photo-corpus test set not present locally"
)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.block(x)


class Yoga3HeadMLP(nn.Module):
    """Verbatim production architecture (app/models/mlp.py), duplicated here
    deliberately: this test must keep working even if that file is refactored,
    since it is re-verifying a HISTORICAL measurement against the checkpoint
    bytes, not against whatever the current architecture module says."""

    def __init__(self, input_dim, num_poses, num_joints=15):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU())
        self.res1 = ResBlock(256, 0.3)
        self.res2 = ResBlock(256, 0.3)
        self.pose_head = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
                                       nn.Dropout(0.2), nn.Linear(128, num_poses))
        self.correctness_head = nn.Sequential(nn.Linear(256, 64), nn.BatchNorm1d(64), nn.GELU(),
                                              nn.Dropout(0.2), nn.Linear(64, 1))
        self.deviation_head = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
                                            nn.Dropout(0.2), nn.Linear(128, num_joints))

    def forward(self, x):
        f = self.res2(self.res1(self.input_layer(x)))
        return self.pose_head(f), self.correctness_head(f).squeeze(-1), self.deviation_head(f)


NOSE = 0
SH_L, SH_R, EL_L, EL_R, WR_L, WR_R = 11, 12, 13, 14, 15, 16
HP_L, HP_R, KN_L, KN_R, AN_L, AN_R, HE_L, HE_R = 23, 24, 25, 26, 27, 28, 29, 30


def angles_from_landmarks(pts):
    """Identical to geometry.extract_angles_from_landmarks(zero_z=True) -- the
    production inference path always zeroes z, so the test must too."""
    p = np.asarray(pts, dtype=np.float64)[:, :3].copy()
    p[:, 2] = 0.0

    def ang(a, b, c):
        ba, bc = p[a] - p[b], p[c] - p[b]
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nb == 0 or nc == 0:
            return 180.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ba, bc) / (nb * nc))))))

    sm = (p[SH_L] + p[SH_R]) / 2.0
    hm = (p[HP_L] + p[HP_R]) / 2.0
    ba, bc = p[NOSE] - sm, hm - sm
    nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    neck = 180.0 if (nb == 0 or nc == 0) else math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(ba, bc) / (nb * nc))))))
    return [ang(SH_L, EL_L, WR_L), ang(SH_R, EL_R, WR_R), ang(HP_L, SH_L, EL_L), ang(HP_R, SH_R, EL_R),
            ang(SH_L, HP_L, KN_L), ang(SH_R, HP_R, KN_R), ang(HP_L, KN_L, AN_L), ang(HP_R, KN_R, AN_R),
            ang(KN_L, AN_L, HE_L), ang(KN_R, AN_R, HE_R), ang(SH_L, HP_L, HP_R), ang(SH_R, HP_R, HP_L),
            neck, ang(HP_R, HP_L, KN_L), ang(HP_L, HP_R, KN_R)]


def _load_frozen_test_set():
    z = np.load(PHOTO_CORPUS, allow_pickle=True)
    lm, lab, sp = z["landmarks"], z["labels"].astype(str), z["split"].astype(str)
    te = sp == "test"
    X = np.array([angles_from_landmarks(x) for x in lm[te]], dtype=np.float32)
    return X, lab[te]


def _macro(model, classes, X, y):
    cidx = {c: i for i, c in enumerate(classes)}
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(torch.tensor(X))
    pred = [classes[i] for i in logits.argmax(1).numpy()]
    accs = []
    for c in sorted(set(y.tolist())):
        idx = [i for i in range(len(y)) if y[i] == c]
        if c not in cidx:
            accs.append(0.0)
            continue
        accs.append(sum(1 for i in idx if pred[i] == y[i]) / len(idx))
    overall = sum(1 for a, b in zip(y.tolist(), pred) if a == b) / len(y)
    return float(np.mean(accs)), overall


def _download(filename):
    from huggingface_hub import hf_hub_download
    token = _hf_token()
    return hf_hub_download(repo_id=HF_REPO, filename=filename, token=token)


@pytest.mark.network
def test_photodomain_beats_v2_on_the_frozen_photo_set():
    try:
        v2_path = _download("mlp_3head_model_v2.pth")
        v2_enc_path = _download("mlp_3head_pose_encoder.npy")
        pd_path = _download("mlp_3head_photodomain_v1.pth")
        pd_enc_path = _download("mlp_3head_photodomain_v1_encoder.npy")
    except Exception as e:
        pytest.skip(f"could not reach Hugging Face Hub: {e}")

    X, y = _load_frozen_test_set()
    assert len(y) == 103, f"expected the frozen 103-photo test set, got {len(y)}"

    v2_classes = list(np.load(v2_enc_path, allow_pickle=True))
    pd_classes = list(np.load(pd_enc_path, allow_pickle=True))
    assert v2_classes == pd_classes, (
        "the two checkpoints' encoders diverged -- the swap in hf_loader.py "
        "assumed an identical vocabulary and order; re-check before trusting "
        "any comparison between them")

    v2 = Yoga3HeadMLP(15, len(v2_classes))
    v2.load_state_dict(torch.load(v2_path, map_location="cpu"))
    pd = Yoga3HeadMLP(15, len(pd_classes))
    pd.load_state_dict(torch.load(pd_path, map_location="cpu"))

    v2_macro, v2_overall = _macro(v2, v2_classes, X, y)
    pd_macro, pd_overall = _macro(pd, pd_classes, X, y)

    # Loose bands, not exact floats: BatchNorm eval-mode behaviour and any
    # future retrain of either checkpoint can shift a percentage point or two
    # without changing the conclusion. What this test guards is the ORDERING
    # and the MAGNITUDE of the gap, which is what justified the swap.
    assert pd_macro > v2_macro, (
        f"photodomain_v1 macro {pd_macro:.3f} no longer beats model_v2 "
        f"{v2_macro:.3f} on the frozen set -- the checkpoint choice in "
        f"hf_loader.py needs re-justifying, not just this test updating")
    assert pd_macro > 0.30, f"photodomain_v1 macro dropped to {pd_macro:.3f} (was 0.355)"
    assert v2_macro < 0.20, (
        f"model_v2 macro is now {v2_macro:.3f}, well above the measured 0.105 -- "
        f"if this moved, the whole comparison should be re-run before trusting it")
