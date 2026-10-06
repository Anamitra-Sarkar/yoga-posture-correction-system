"""Pose cascade + guided report. Pure-logic tests need no models; endpoint tests use tiny stub models so every
branch of analyse_frame is exercised exactly (flag off / gate rejects / gate agrees / gate unavailable / guided / legacy request)."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import pose as pose_router
from app.services import hf_loader
from app.services.cascade import UNKNOWN, cascade_decide, collapse, guided_report, top_k
from app.utils.geometry import FEATURE_NAMES
from app.utils.rules_classifier import _POSE_FEATURE_BANDS

BANDED = next(p for p, b in _POSE_FEATURE_BANDS.items() if b)         # a pose that has angle bands
BANDLESS = next(p for p, b in _POSE_FEATURE_BANDS.items() if not b)   # tree_pose / lunge_pose: no bands
ANG = {n: 90.0 for n in FEATURE_NAMES}
DEVS = {n: 12.0 for n in FEATURE_NAMES}


def test_collapse_sums_imperfect_onto_base():
    assert collapse({"warrior_2": 0.5, "imperfect_warrior_2": 0.25, UNKNOWN: 0.25}) == {"warrior_2": 0.75, UNKNOWN: 0.25}


def test_gate_rejects_even_when_namer_is_confident():
    r = cascade_decide("warrior_2", {"warrior_2": 0.99, UNKNOWN: 0.01}, {UNKNOWN: 0.7, "warrior_2": 0.3}, ANG, 0.9, DEVS)
    assert r.pose_id == UNKNOWN and r.gated and r.correctness == 0.0 and r.correctness_source == "none"


def test_gate_pass_returns_namer_pose_and_gate_correctness():
    r = cascade_decide(BANDED, {BANDED: 0.8, "plank": 0.2}, {BANDED: 0.6, UNKNOWN: 0.4}, ANG, 0.83, DEVS)
    assert r.pose_id == BANDED and not r.gated and r.correctness == pytest.approx(0.83) and r.correctness_source == "gate_head"
    assert r.deviations_source == "rule_bands"                       # banded pose -> deviations come from the bands, not the weak head
    assert r.candidates[0][0] == BANDED


def test_bandless_pose_reports_no_per_joint_deviations_not_the_gate_heads_guess():
    r = cascade_decide(BANDLESS, {BANDLESS: 0.9}, {BANDLESS: 0.8, UNKNOWN: 0.2}, ANG, 0.7, DEVS)   # DEVS = 12 deg everywhere (the weak head's output)
    assert r.pose_id == BANDLESS and r.deviations_source == "none_no_bands"
    assert set(r.deviations) == set(FEATURE_NAMES) and all(v == 0.0 for v in r.deviations.values())
    assert r.correctness == pytest.approx(0.7) and r.correctness_source == "gate_head"             # the form score still comes from the gate


def test_namer_unknown_stays_unknown_without_being_counted_as_gated():
    r = cascade_decide(UNKNOWN, {UNKNOWN: 0.9, BANDED: 0.1}, {BANDED: 0.9, UNKNOWN: 0.1}, ANG, 0.8, DEVS)
    assert r.pose_id == UNKNOWN and not r.gated


def test_gate_disagreeing_on_which_pose_does_not_veto():
    # the gate only vetoes "none of mine"; if it names a DIFFERENT real pose the namer's name stands and gate_pose is reported
    r = cascade_decide(BANDED, {BANDED: 0.7}, {"plank": 0.8, UNKNOWN: 0.2}, ANG, 0.6, DEVS)
    assert r.pose_id == BANDED and r.gate_pose == "plank"


def test_top_k_orders_descending():
    assert [n for n, _ in top_k({"a": .1, "b": .6, "c": .3}, 2)] == ["b", "c"]


def test_guided_report_scores_against_the_chosen_pose_not_the_detected_one():
    g = guided_report(BANDED, "plank", ANG)
    assert g["target_pose"] == BANDED and g["matches"] is False and g["target_has_bands"] is True
    assert guided_report(BANDED, BANDED, ANG)["matches"] is True
    assert guided_report(BANDLESS, BANDLESS, ANG)["target_has_bands"] is False


# ---------------------------------------------------------------- endpoint tests with stub models
class Stub(nn.Module):
    def __init__(self, classes, top, corr=0.5):
        super().__init__()
        self.logits = torch.full((1, len(classes)), -5.0)
        self.logits[0, classes.index(top)] = 5.0
        self.corr = torch.tensor([float(np.log(corr / (1 - corr)))])
        self.dev = torch.full((1, 15), 0.1)

    def forward(self, x):
        return self.logits, self.corr, self.dev


CLASSES = [UNKNOWN, "mountain_pose", "warrior_2", "plank"]


@pytest.fixture
def client(monkeypatch):
    app = FastAPI(); app.include_router(pose_router.router, prefix="/api")
    return TestClient(app), monkeypatch


def payload(**kw):
    p = {"angles": [170.0] * 15}
    p.update(kw); return p


def install(monkeypatch, namer_top, gate_top=None, cascade_on=True, gate_corr=0.9):
    monkeypatch.setattr(pose_router, "get_mlp_model", lambda: (Stub(CLASSES, namer_top, 0.5), CLASSES))
    monkeypatch.setattr(settings, "ENABLE_POSE_CASCADE", cascade_on)
    monkeypatch.setattr(pose_router, "get_gate_model",
                        lambda: (Stub(CLASSES, gate_top, gate_corr), CLASSES) if gate_top else (None, []))


def test_flag_off_is_the_original_response_shape(client):
    c, mp = client; install(mp, "mountain_pose", gate_top=None, cascade_on=False)
    r = c.post("/api/analyse_frame", json=payload()).json()
    assert r["cascade"] is None and r["candidates"] is None and r["guided"] is None
    assert {"pose_id", "correctness_score", "deviations", "motion_state"} <= set(r)


def test_cascade_gate_rejects(client):
    c, mp = client; install(mp, "mountain_pose", gate_top=UNKNOWN)
    r = c.post("/api/analyse_frame", json=payload()).json()
    assert r["pose_id"] == UNKNOWN and r["cascade"]["active"] and r["cascade"]["gated"]


def test_cascade_gate_agrees_uses_gate_correctness_and_returns_candidates(client):
    c, mp = client; install(mp, "mountain_pose", gate_top="mountain_pose", gate_corr=0.9)
    r = c.post("/api/analyse_frame", json=payload()).json()
    assert r["pose_id"] == "mountain_pose" and r["correctness_score"] == pytest.approx(0.9, abs=1e-3)
    assert r["cascade"]["correctness_source"] == "gate_head" and len(r["candidates"]) == 3


def test_cascade_requested_but_gate_unavailable_falls_back_and_says_so(client):
    c, mp = client; install(mp, "mountain_pose", gate_top=None, cascade_on=True)
    mp.setattr(hf_loader, "gate_load_error", "FileNotFoundError: no gate checkpoint")
    r = c.post("/api/analyse_frame", json=payload()).json()
    assert r["cascade"]["active"] is False and "no gate checkpoint" in r["cascade"]["reason"]
    assert r["pose_id"]                                                       # still answered by the original hybrid path


def test_guided_block_present_only_with_target(client):
    c, mp = client; install(mp, "mountain_pose", gate_top="mountain_pose")
    assert c.post("/api/analyse_frame", json=payload()).json()["guided"] is None
    g = c.post("/api/analyse_frame", json=payload(target_pose="warrior_2")).json()["guided"]
    assert g["target_pose"] == "warrior_2" and g["matches"] is False and set(g["target_deviations"]) == set(FEATURE_NAMES)


def test_bad_input_still_rejected(client):
    c, mp = client; install(mp, "mountain_pose", gate_top="mountain_pose")
    assert c.post("/api/analyse_frame", json={"angles": [1.0] * 14}).status_code == 400
