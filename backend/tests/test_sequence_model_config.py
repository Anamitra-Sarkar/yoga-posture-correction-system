"""The ST-GCN checkpoint is selectable by env var, defaults to the checkpoint live since 2026-09-20 (so deploying
the code alone changes nothing), and both label vocabularies produce a sensible /analyse_sequence response."""
import importlib

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import pose as pose_router
from app.services import hf_loader


def _reload_settings(monkeypatch, **env):
    import app.config as cfg
    for k in ("STGCN_MODEL_FILE", "STGCN_ENCODER_FILE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(cfg).settings


def test_default_checkpoint_is_the_one_already_live(monkeypatch):
    s = _reload_settings(monkeypatch)
    assert s.STGCN_MODEL_FILE == "stgcn_transitions_v1.pth"
    assert s.STGCN_ENCODER_FILE == "stgcn_transitions_v1_encoder.npy"


def test_env_selects_the_target_pose_checkpoint(monkeypatch):
    s = _reload_settings(monkeypatch, STGCN_MODEL_FILE="stgcn_target_v1.pth", STGCN_ENCODER_FILE="stgcn_target_v1_encoder.npy")
    assert (s.STGCN_MODEL_FILE, s.STGCN_ENCODER_FILE) == ("stgcn_target_v1.pth", "stgcn_target_v1_encoder.npy")
    _reload_settings(monkeypatch)  # restore defaults for later tests


class _Fixed(nn.Module):
    """Stub ST-GCN returning fixed logits, so the endpoint's label handling is tested exactly."""
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("l", torch.tensor([logits], dtype=torch.float32))

    def forward(self, x):
        return self.l.expand(x.shape[0], -1)


def _client(monkeypatch, classes, logits):
    monkeypatch.setattr(pose_router, "get_stgcn_model", lambda: (_Fixed(logits), classes))
    app = FastAPI(); app.include_router(pose_router.router, prefix="/api")
    return TestClient(app)


WINDOW = {"coordinates": [[0.5] * 99 for _ in range(60)]}


@pytest.mark.parametrize("classes,top,expect_pose,expect_kind", [
    (["child_pose", "transition/unknown"], 0, "child_pose", "hold"),                  # stgcn_target_v1 vocabulary
    (["hold:child_pose", "transition:a->b", "unrecognized"], 0, "child_pose", "hold"),  # stgcn_transitions_v1 vocabulary
    (["child_pose", "transition/unknown"], 1, "transition/unknown", "unrecognized"),
])
def test_both_vocabularies_are_served(monkeypatch, classes, top, expect_pose, expect_kind):
    logits = [0.0] * len(classes); logits[top] = 9.0
    r = _client(monkeypatch, classes, logits).post("/api/analyse_sequence", json=WINDOW)
    assert r.status_code == 200
    j = r.json()
    assert j["sequence_pose"] == expect_pose and j["sequence_kind"] == expect_kind
    assert j["requires_static_fallback"] == (expect_pose == "transition/unknown")  # confident hold -> no fallback
