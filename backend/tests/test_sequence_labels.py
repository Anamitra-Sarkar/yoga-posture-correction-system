"""The sequence endpoint must behave identically under both label vocabularies.

The original 15-class checkpoint emits plain pose names plus one catch-all
"transition/unknown". The transition-aware checkpoint (63.0% macro, 24 classes)
emits "hold:<pose>", "transition:<A>-><B>" and "unrecognized". Loading the
better checkpoint should therefore be a one-line change in hf_loader, with no
client breakage -- these tests are what makes that claim checkable.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.routers.pose import parse_sequence_label


# ---- original vocabulary ----

def test_plain_pose_name_is_a_hold():
    assert parse_sequence_label("warrior_2") == ("warrior_2", "hold", None, None)


def test_old_catch_all_maps_to_unrecognized():
    assert parse_sequence_label("transition/unknown") == \
        ("transition/unknown", "unrecognized", None, None)


# ---- transition-aware vocabulary ----

def test_hold_prefix_is_stripped():
    assert parse_sequence_label("hold:child_pose") == ("child_pose", "hold", None, None)


def test_named_transition_exposes_both_endpoints():
    d, k, a, b = parse_sequence_label("transition:warrior_2->plank")
    assert (k, a, b) == ("transition", "warrior_2", "plank")


def test_named_transition_does_not_leak_the_raw_label_to_old_clients():
    # an old client renders sequence_pose directly; "transition:a->b" would
    # show up as if it were the name of a pose
    d, _, _, _ = parse_sequence_label("transition:warrior_2->plank")
    assert d == "transition/unknown"


def test_new_unrecognized_label_matches_the_old_catch_all():
    assert parse_sequence_label("unrecognized") == \
        ("transition/unknown", "unrecognized", None, None)


def test_malformed_transition_label_does_not_crash():
    d, k, a, b = parse_sequence_label("transition:onlyone")
    assert k == "transition" and a == "onlyone" and b is None


# ---- the tuned threshold must survive the vocabulary change ----

def test_child_pose_threshold_is_found_under_both_vocabularies():
    from app.routers.pose import SEQUENCE_FALLBACK_THRESHOLDS
    for label in ("child_pose", "hold:child_pose"):
        display = parse_sequence_label(label)[0]
        assert display in SEQUENCE_FALLBACK_THRESHOLDS, label
