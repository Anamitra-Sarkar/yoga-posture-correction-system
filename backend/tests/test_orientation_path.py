"""The orientation path, end to end through the API layer.

The 15 angle features are all RELATIVE joint angles, so they are invariant to
rotating the whole body. Measured on the real-photo corpus, that blindness is
not theoretical: the rule engine called corpse "mountain_pose" 11 times out of
18, because lying flat and standing upright produce nearly the same 15-vector.

These tests pin the two properties the fix has to have:

  * WITH orientation, a lying body is no longer read as a standing one;
  * WITHOUT orientation, behaviour is byte-identical to before, so every
    older client (and the Expo companion until it is updated) keeps working
    exactly as it did.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.geometry import FEATURE_NAMES, compute_orientation  # noqa: E402
from app.utils.rules_classifier import classify_pose  # noqa: E402


def angles(**over):
    a = {f: 180.0 for f in FEATURE_NAMES}
    a.update(over)
    return a


# Limbs extended, arms by the sides. Identical features for standing and lying
# -- which is precisely the collision being fixed.
EXTENDED = angles(shoulder_l=20.0, shoulder_r=20.0, neck=160.0,
                  trunk_l=100.0, trunk_r=100.0)


def test_extended_body_reads_as_standing_without_orientation():
    # unchanged legacy behaviour: no orientation, so the 2D chain decides
    assert classify_pose(EXTENDED) == "mountain_pose"


def test_same_angles_read_as_corpse_when_lying():
    out = classify_pose(EXTENDED, {"torso_incline": 100.0, "leg_torso_ratio": 1.2})
    assert out == "corpse", out


def test_same_angles_still_read_as_mountain_when_upright():
    out = classify_pose(EXTENDED, {"torso_incline": 5.0, "leg_torso_ratio": 1.3})
    assert out == "mountain_pose", out


def test_orientation_can_only_add_never_silently_change_a_2d_call():
    """A pose the 2D chain already gets right, with an orientation consistent
    with it, must not be re-labelled."""
    upright = {"torso_incline": 6.0, "leg_torso_ratio": 1.3}
    salute = angles(shoulder_l=150.0, shoulder_r=150.0, trunk_l=100.0, trunk_r=100.0)
    assert classify_pose(salute) == "upward_salute"
    assert classify_pose(salute, upright) == "upward_salute"


def test_malformed_orientation_falls_back_instead_of_crashing():
    for bad in ({}, {"torso_incline": 5.0}, {"leg_torso_ratio": 1.0},
                {"torso_incline": None, "leg_torso_ratio": None}):
        assert classify_pose(EXTENDED, bad) == "mountain_pose", bad


def test_compute_orientation_matches_the_physical_meaning():
    p = np.zeros((33, 3))
    # standing: shoulders above hips, ankles below
    p[11] = [0.45, 0.30, 0]; p[12] = [0.55, 0.30, 0]
    p[23] = [0.47, 0.55, 0]; p[24] = [0.53, 0.55, 0]
    p[27] = [0.47, 0.95, 0]; p[28] = [0.53, 0.95, 0]
    o = compute_orientation(p)
    assert o["torso_incline"] < 10.0, o

    # lying: torso horizontal across the frame
    q = np.zeros((33, 3))
    q[11] = [0.20, 0.50, 0]; q[12] = [0.20, 0.56, 0]
    q[23] = [0.50, 0.50, 0]; q[24] = [0.50, 0.56, 0]
    q[27] = [0.90, 0.50, 0]; q[28] = [0.90, 0.56, 0]
    o2 = compute_orientation(q)
    assert 80.0 < o2["torso_incline"] < 100.0, o2


def test_compute_orientation_returns_empty_on_degenerate_landmarks():
    assert compute_orientation(np.zeros((33, 3))) == {}


def test_frame_input_accepts_and_forwards_orientation():
    from app.routers.pose import FrameInput
    fi = FrameInput(angles=[180.0] * 15,
                    orientation={"torso_incline": 100.0, "leg_torso_ratio": 1.2})
    assert fi.orientation["torso_incline"] == 100.0
    # and an older client that omits it must still validate
    assert FrameInput(angles=[180.0] * 15).orientation is None
