"""Unit tests for the free-form (no pose selection) analysis helpers.

These three functions carry the whole redesign: they decide whether the user
is holding a pose or moving between poses, and they turn the Digital Twin
calibration into a second, personalised correctness score. They are pure, so
they can be tested exactly -- no model, no network, no HF Space.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.routers.pose import (
    MOTION_HOLD_MAX_DEG_PER_SEC as THRESH,
    classify_motion_state,
    apply_calibration,
    correctness_from_deviations,
)


# ---------------- motion state ----------------

def test_missing_motion_is_unknown_not_guessed():
    # Old clients send no motion signal; we must not invent one.
    assert classify_motion_state(None, "warrior_2") == "unknown"
    assert classify_motion_state(None, "transition/unknown") == "unknown"


def test_still_and_recognised_is_holding():
    assert classify_motion_state(0.0, "warrior_2") == "holding"
    assert classify_motion_state(THRESH, "warrior_2") == "holding"


def test_still_but_unnameable_is_unrecognized_not_transitioning():
    # The point of the redesign: a pose we can't name is NOT the same as a
    # transition, and the old catch-all conflated them.
    assert classify_motion_state(2.0, "transition/unknown") == "unrecognized"


def test_moving_is_transitioning_regardless_of_pose_label():
    assert classify_motion_state(THRESH + 0.01, "warrior_2") == "transitioning"
    assert classify_motion_state(400.0, "transition/unknown") == "transitioning"


# ---------------- calibration ----------------

def _devs():
    return {"knee_l": 10.0, "knee_r": 4.0, "elbow_l": 0.0}


def test_joint_inside_user_range_is_forgiven():
    out = apply_calibration(_devs(), {"knee_l": 95.0, "knee_r": 170.0, "elbow_l": 180.0},
                            {"knee_l": {"min": 80.0, "max": 110.0}})
    assert out["knee_l"] == 0.0
    assert out["knee_r"] == 4.0          # untouched


def test_joint_outside_user_range_is_not_forgiven():
    out = apply_calibration(_devs(), {"knee_l": 40.0, "knee_r": 170.0, "elbow_l": 180.0},
                            {"knee_l": {"min": 80.0, "max": 110.0}})
    assert out["knee_l"] == 10.0


def test_range_is_inclusive_at_both_ends():
    ang = {"knee_l": 80.0, "knee_r": 170.0, "elbow_l": 180.0}
    cal = {"knee_l": {"min": 80.0, "max": 110.0}}
    assert apply_calibration(_devs(), ang, cal)["knee_l"] == 0.0
    ang["knee_l"] = 110.0
    assert apply_calibration(_devs(), ang, cal)["knee_l"] == 0.0


def test_malformed_or_unknown_calibration_entries_are_ignored():
    devs = _devs()
    ang = {"knee_l": 95.0, "knee_r": 170.0, "elbow_l": 180.0}
    # unknown joint, missing max, missing min -- none may raise or corrupt
    out = apply_calibration(devs, ang, {
        "not_a_joint": {"min": 0.0, "max": 180.0},
        "knee_r": {"min": 100.0},
        "elbow_l": {"max": 180.0},
    })
    assert out == devs


def test_apply_calibration_does_not_mutate_its_input():
    devs = _devs()
    apply_calibration(devs, {"knee_l": 95.0}, {"knee_l": {"min": 80.0, "max": 110.0}})
    assert devs["knee_l"] == 10.0


# ---------------- personal score ----------------

def test_forgiving_nothing_leaves_the_score_unchanged():
    u = {"knee_l": 10.0, "knee_r": 4.0}
    assert correctness_from_deviations(dict(u), u, 0.6) == 0.6


def test_forgiving_everything_gives_a_perfect_personal_score():
    u = {"knee_l": 10.0, "knee_r": 4.0}
    c = {"knee_l": 0.0, "knee_r": 0.0}
    assert correctness_from_deviations(c, u, 0.6) == 1.0


def test_personal_score_is_never_below_universal():
    u = {"knee_l": 10.0, "knee_r": 4.0}
    for cal in ({"knee_l": 10.0, "knee_r": 4.0}, {"knee_l": 0.0, "knee_r": 4.0},
                {"knee_l": 5.0, "knee_r": 0.0}, {"knee_l": 0.0, "knee_r": 0.0}):
        assert correctness_from_deviations(cal, u, 0.6) >= 0.6


def test_partial_forgiveness_closes_that_fraction_of_the_gap():
    # knee_l is 10 of the 14 total deviation; forgiving it forgives 10/14,
    # so the score should close 10/14 of the 0.4 gap up to 1.0.
    u = {"knee_l": 10.0, "knee_r": 4.0}
    got = correctness_from_deviations({"knee_l": 0.0, "knee_r": 4.0}, u, 0.6)
    assert abs(got - (0.6 + (10.0 / 14.0) * 0.4)) < 1e-9


def test_already_perfect_pose_stays_at_one():
    u = {"knee_l": 0.0, "knee_r": 0.0}
    assert correctness_from_deviations(dict(u), u, 1.0) == 1.0


def test_score_stays_inside_zero_to_one():
    u = {"knee_l": 10.0}
    assert 0.0 <= correctness_from_deviations({"knee_l": 0.0}, u, 0.99) <= 1.0
