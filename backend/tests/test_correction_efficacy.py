"""The correction-efficacy loop's server side.

An open-loop coach repeats the same sentence at a user who did not respond to
it, which is the single most obviously "deaf" behaviour automated coaching
has. These tests pin the three things the client needs for the loop to work:

  * the response names WHICH joint the cue is trying to move, so the client can
    measure that joint specifically rather than inferring from the whole-body
    score (which drifts for unrelated reasons);
  * repeating a cue that did not work escalates instead of repeating verbatim;
  * escalation ends in BACKING OFF, not in pushing harder -- a cue that has
    failed twice usually means the range of motion is not there today, and
    "try harder" is exactly the injury this system exists to prevent.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.correction import generate_safe_correction  # noqa: E402

# warrior_2/knee_l has a real template, so the target joint is deterministic
POSE = "warrior_2"
DEVS = {"knee_l": 24.0, "knee_r": 2.0}


def _text(attempt, language="en"):
    t, safe, joint = generate_safe_correction(
        pose_id=POSE, deviations=DEVS, language=language,
        groq_api_key=None, attempt=attempt)
    return t, safe, joint


def test_response_names_the_targeted_joint():
    _t, _s, joint = _text(0)
    assert joint == "knee_l", joint


def test_joint_is_the_worst_deviation_not_the_first():
    t, _s, joint = generate_safe_correction(
        pose_id=POSE, deviations={"knee_l": 3.0, "knee_r": 30.0},
        language="en", groq_api_key=None, attempt=0)
    assert joint == "knee_r", joint


def test_first_attempt_is_the_plain_cue():
    t0, _s, _j = _text(0)
    assert "degrees off" not in t0
    assert "Ease out" not in t0


def test_second_attempt_adds_the_measured_magnitude():
    t0, _, _ = _text(0)
    t1, _, _ = _text(1)
    assert t1 != t0, "a cue that did not work must not be repeated verbatim"
    assert "24 degrees" in t1, t1
    assert t0.rstrip() in t1, "the escalated cue should still contain the original instruction"


def test_third_attempt_backs_off_rather_than_pushing():
    t2, safe, joint = _text(2)
    low = t2.lower()
    assert "ease out" in low, t2
    for word in ("more", "deeper", "further", "harder"):
        assert word not in low, f"escalation must not push harder: {t2}"
    assert safe is True
    assert joint == "knee_l"


def test_escalation_is_capped_and_does_not_crash():
    for a in (3, 10, 99):
        t, safe, _j = _text(a)
        assert "ease out" in t.lower()
        assert safe is True


def test_escalation_is_localised():
    for lang in ("hi", "bn"):
        t1, _, _ = _text(1, lang)
        t2, _, _ = _text(2, lang)
        assert t1 and t2
        # the localised strings must not silently fall back to English
        assert "degrees off" not in t1, (lang, t1)
        assert "Ease out" not in t2, (lang, t2)


def test_no_target_joint_when_nothing_exceeds_threshold():
    t, safe, joint = generate_safe_correction(
        pose_id=POSE, deviations={"knee_l": 1.0, "knee_r": 2.0},
        language="en", groq_api_key=None, attempt=0)
    assert joint is None
    assert safe is True
    assert t  # still returns the generic encouragement
