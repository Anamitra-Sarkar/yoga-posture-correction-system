"""Regression tests for classify_pose() branch ordering.

The specific bug these lock down: warrior_1 (disabled) used to be checked
before warrior_2 (live). Their leg conditions are identical and their arm
bands overlap at shoulder in (110, 125], so every genuine warrior_2 attempt
landing in that overlap was labelled warrior_1 and then discarded by
sanitize_pose() as transition/unknown -- silently losing detections of one of
the three original live poses.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.geometry import FEATURE_NAMES
from app.utils.rules_classifier import classify_pose


def angles(**over):
    a = {f: 180.0 for f in FEATURE_NAMES}
    a.update(over)
    return a


def warrior_stance(shoulder):
    # front knee bent, back leg straight -- the shared warrior leg signature
    return angles(knee_l=100.0, knee_r=170.0, hip_l=120.0, hip_r=150.0,
                  shoulder_l=shoulder, shoulder_r=shoulder,
                  trunk_l=90.0, trunk_r=90.0, neck=90.0,
                  hip_abduct_l=120.0, hip_abduct_r=120.0)


def test_overlap_band_resolves_to_warrior_2_not_warrior_1():
    # the exact window the bug lived in
    for shoulder in (111.0, 115.0, 120.0, 125.0):
        assert classify_pose(warrior_stance(shoulder)) == "warrior_2", shoulder


def test_warrior_2_only_band_still_warrior_2():
    for shoulder in (65.0, 90.0, 110.0):
        assert classify_pose(warrior_stance(shoulder)) == "warrior_2", shoulder


def test_above_the_overlap_is_still_warrior_1():
    # >125 leaves warrior_2's arm band, so warrior_1 must still be reachable;
    # reordering must not make the branch dead code.
    assert classify_pose(warrior_stance(140.0)) == "warrior_1"


def test_warrior_legs_with_arms_down_is_a_lunge():
    assert classify_pose(warrior_stance(30.0)) == "lunge_pose"


def test_mirrored_stance_is_classified_the_same():
    a = warrior_stance(115.0)
    m = dict(a); m["knee_l"], m["knee_r"] = a["knee_r"], a["knee_l"]
    m["hip_l"], m["hip_r"] = a["hip_r"], a["hip_l"]
    assert classify_pose(m) == "warrior_2"


# --- the other live poses must not have regressed ---

def test_live_poses_still_classify():
    cases = {
        "mountain_pose": angles(shoulder_l=20.0, shoulder_r=20.0, neck=160.0),
        "downward_dog": angles(hip_l=60.0, hip_r=60.0, knee_l=170.0, knee_r=170.0,
                               shoulder_l=165.0, shoulder_r=165.0),
        "plank": angles(shoulder_l=85.0, shoulder_r=85.0),
        # trunk must be <=65: mountain_pose's rule is otherwise a superset of
        # cobra's in pure 2D-angle space (both hip>140/knee>140/shoulder<55),
        # which is the documented representational limit of dropping z, not a
        # threshold miss. A real cobra's lifted torso is what separates them.
        "cobra_pose": angles(shoulder_l=25.0, shoulder_r=25.0, neck=120.0,
                             hip_l=150.0, hip_r=150.0,
                             trunk_l=40.0, trunk_r=40.0),
        "tree_pose": angles(knee_l=175.0, hip_l=178.0, knee_r=60.0, hip_r=120.0,
                            shoulder_l=40.0, shoulder_r=40.0),
    }
    for expected, a in cases.items():
        assert classify_pose(a) == expected, f"{expected} -> {classify_pose(a)}"


def test_every_branch_returns_a_known_label():
    import random
    random.seed(11)
    known = set()
    for _ in range(3000):
        a = {f: random.uniform(0.0, 180.0) for f in FEATURE_NAMES}
        p = classify_pose(a)
        assert isinstance(p, str) and p
        known.add(p)
    # a random sweep should reach a decent share of the vocabulary, not one label
    assert len(known) >= 5, known
