"""
Deterministic biomechanical rule classifier for single-frame pose classification.

Replaces the learned MLP for this task. The MLP was trained on 3D (x,y,z) joint
angles, but MediaPipe's monocular z-depth estimate is only reliable at the
consistent camera distance/framing seen in demo videos -- on arbitrary real-world
photos and webcam framing it degrades badly (measured: 0-3% accuracy on a 35-image
real-world test set, confirmed via a 40-trial randomized threshold sweep that this
holds regardless of the specific rule thresholds chosen). Dropping z and using pure
2D (image-plane) angles is far more robust (~46% on the same test set) since 2D
projection is what a camera actually captures reliably.

Callers must zero out the z component before computing angles (see geometry.ts /
extract_angles_from_landmarks) -- this module only defines the classification and
deviation logic on top of whatever 15 angles it's given.
"""
from typing import Dict, List, Tuple

FEATURE_NAMES = [
    "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
    "hip_l", "hip_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r", "trunk_l", "trunk_r",
    "neck", "hip_abduct_l", "hip_abduct_r"
]

# Pulled from live detection after real-world testing: plank was consistently
# misread as mountain_pose by the MLP (no rule engine coverage exists for it
# either), tree_pose scattered across child_pose/lunge-type predictions, and
# chair_pose kept misfiring even after the warrior/chair rule-priority fix
# (see classify_pose's chair_pose comment below -- reordering and the
# leg-symmetry check helped but weren't sufficient in live real-world use).
# Applied as a final safety net after classification, regardless of whether
# the pose call came from the MLP or the rule engine, so neither can ever
# surface as a detected pose until the underlying issue is fixed.
DISABLED_POSES = {"plank", "tree_pose", "chair_pose"}


def sanitize_pose(pose_id: str) -> str:
    return "transition/unknown" if pose_id in DISABLED_POSES else pose_id


def _between(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def classify_pose(a: Dict[str, float]) -> str:
    hip_l, hip_r = a["hip_l"], a["hip_r"]
    knee_l, knee_r = a["knee_l"], a["knee_r"]
    shoulder_l, shoulder_r = a["shoulder_l"], a["shoulder_r"]
    trunk_l, trunk_r = a["trunk_l"], a["trunk_r"]
    neck = a["neck"]
    hip_abduct_l, hip_abduct_r = a["hip_abduct_l"], a["hip_abduct_r"]

    if hip_l > 140 and hip_r > 140 and knee_l > 140 and knee_r > 140 and shoulder_l < 55 and shoulder_r < 55 and trunk_l > 65 and trunk_r > 65:
        return "mountain_pose"
    if hip_l > 140 and hip_r > 140 and knee_l > 140 and knee_r > 140 and shoulder_l > 115 and shoulder_r > 115 and trunk_l > 65 and trunk_r > 65:
        return "upward_salute"
    if _between(hip_l, 45, 130) and _between(hip_r, 45, 130) and knee_l > 110 and knee_r > 110 and shoulder_l > 95 and shoulder_r > 95:
        return "downward_dog"
    if hip_l > 120 and hip_r > 120 and knee_l > 120 and knee_r > 120 and _between(shoulder_l, 5, 50) and _between(shoulder_r, 5, 50) and neck >= 80:
        return "cobra_pose"
    if hip_l < 90 and hip_r < 90 and knee_l < 90 and knee_r < 90 and shoulder_l > 85 and shoulder_r > 85:
        return "child_pose"
    if _between(hip_l, 60, 120) and _between(hip_r, 60, 120) and knee_l > 135 and knee_r > 135 and trunk_l >= 60 and trunk_r >= 60:
        return "seated_staff"
    if _between(hip_l, 50, 120) and _between(hip_r, 50, 120) and knee_l < 125 and knee_r < 125 and trunk_l >= 60 and trunk_r >= 60:
        return "seated_easy_pose"
    # tree_pose rule intentionally removed: live real-world testing showed it
    # consistently scattering across child_pose/lunge-type predictions rather
    # than reliably identifying tree_pose itself. Pulled from the vocabulary
    # (see PRODUCTION_DISABLED_POSES below) until the underlying detection is
    # fixed, rather than keep offering an unreliable rule.
    # Chair pose (Utkatasana) is a SYMMETRIC bent-knee stance; warrior_1/2/lunge
    # are all defined by an ASYMMETRIC one-bent-one-straight leg pair. Checking
    # chair_pose first, with an explicit symmetry requirement, stops the (much
    # wider) warrior/lunge leg conditions from preempting a genuinely symmetric
    # chair stance just because one knee's 2D angle reads slightly differently
    # than the other -- and conversely stops legitimate warrior_2 attempts that
    # fall outside its arm-angle band from being coincidentally caught by
    # chair_pose's own wide hip/knee band, since real warrior stances have a
    # large (>30 deg) knee asymmetry that chair_pose now explicitly excludes.
    if (_between(hip_l, 75, 140) and _between(hip_r, 75, 140)
            and _between(knee_l, 75, 140) and _between(knee_r, 75, 140)
            and abs(knee_l - knee_r) < 30
            and shoulder_l > 95 and shoulder_r > 95):
        return "chair_pose"
    w1_legs = (knee_l < 120 and knee_r > 130) or (knee_r < 120 and knee_l > 130)
    w1_arms = shoulder_l > 110 and shoulder_r > 110
    if w1_legs and w1_arms:
        return "warrior_1"
    w2_legs = (knee_l < 120 and knee_r > 130) or (knee_r < 120 and knee_l > 130)
    w2_arms = _between(shoulder_l, 65, 125) and _between(shoulder_r, 65, 125)
    if w2_legs and w2_arms:
        return "warrior_2"
    if (knee_l < 120 and knee_r > 130) or (knee_r < 120 and knee_l > 130):
        return "lunge_pose"
    if hip_l < 70 and hip_r < 70 and knee_l > 120 and knee_r > 120:
        return "standing_forward_fold"
    if _between(hip_l, 70, 115) and _between(hip_r, 70, 115) and knee_l > 130 and knee_r > 130:
        return "halfway_lift"
    if _between(hip_l, 60, 125) and _between(hip_r, 60, 125) and _between(knee_l, 60, 125) and _between(knee_r, 60, 125) and _between(shoulder_l, 60, 125) and _between(shoulder_r, 60, 125):
        return "table_top"
    if hip_l > 140 and hip_r > 140 and knee_l > 140 and knee_r > 140:
        return "standing_pose"
    return "transition/unknown"


# For each pose, the features its rule actually constrains, and the (lo, hi) band
# used for that constraint -- reused here to score how centered the user's angles
# are within the "correct" band, not just whether they cleared the pass/fail line.
_POSE_FEATURE_BANDS: Dict[str, List[Tuple[str, float, float]]] = {
    "mountain_pose": [("hip_l", 140, 180), ("hip_r", 140, 180), ("knee_l", 140, 180), ("knee_r", 140, 180),
                       ("shoulder_l", 0, 55), ("shoulder_r", 0, 55), ("trunk_l", 65, 180), ("trunk_r", 65, 180)],
    "upward_salute": [("hip_l", 140, 180), ("hip_r", 140, 180), ("knee_l", 140, 180), ("knee_r", 140, 180),
                       ("shoulder_l", 115, 180), ("shoulder_r", 115, 180), ("trunk_l", 65, 180), ("trunk_r", 65, 180)],
    "downward_dog": [("hip_l", 45, 130), ("hip_r", 45, 130), ("knee_l", 110, 180), ("knee_r", 110, 180),
                      ("shoulder_l", 95, 180), ("shoulder_r", 95, 180)],
    "cobra_pose": [("hip_l", 120, 180), ("hip_r", 120, 180), ("knee_l", 120, 180), ("knee_r", 120, 180),
                   ("shoulder_l", 5, 50), ("shoulder_r", 5, 50), ("neck", 80, 180)],
    "child_pose": [("hip_l", 0, 90), ("hip_r", 0, 90), ("knee_l", 0, 90), ("knee_r", 0, 90),
                    ("shoulder_l", 85, 180), ("shoulder_r", 85, 180)],
    "seated_staff": [("hip_l", 60, 120), ("hip_r", 60, 120), ("knee_l", 135, 180), ("knee_r", 135, 180),
                      ("trunk_l", 60, 180), ("trunk_r", 60, 180)],
    "seated_easy_pose": [("hip_l", 50, 120), ("hip_r", 50, 120), ("knee_l", 0, 125), ("knee_r", 0, 125),
                          ("trunk_l", 60, 180), ("trunk_r", 60, 180)],
    "tree_pose": [("knee_l", 140, 180), ("knee_r", 140, 180), ("hip_l", 140, 180), ("hip_r", 140, 180)],
    "warrior_1": [("shoulder_l", 110, 180), ("shoulder_r", 110, 180)],
    "warrior_2": [("shoulder_l", 65, 125), ("shoulder_r", 65, 125)],
    "lunge_pose": [],
    "standing_forward_fold": [("hip_l", 0, 70), ("hip_r", 0, 70), ("knee_l", 120, 180), ("knee_r", 120, 180)],
    "halfway_lift": [("hip_l", 70, 115), ("hip_r", 70, 115), ("knee_l", 130, 180), ("knee_r", 130, 180)],
    "chair_pose": [("hip_l", 75, 140), ("hip_r", 75, 140), ("knee_l", 75, 140), ("knee_r", 75, 140),
                    ("shoulder_l", 95, 180), ("shoulder_r", 95, 180)],
    "table_top": [("hip_l", 60, 125), ("hip_r", 60, 125), ("knee_l", 60, 125), ("knee_r", 60, 125),
                   ("shoulder_l", 60, 125), ("shoulder_r", 60, 125)],
    "standing_pose": [("hip_l", 140, 180), ("hip_r", 140, 180), ("knee_l", 140, 180), ("knee_r", 140, 180)],
}


def hybrid_classify(
    mlp_pose: str,
    mlp_correctness: float,
    mlp_devs: Dict[str, float],
    angles_2d: Dict[str, float],
    world_angles: "Dict[str, float] | None" = None,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Combines the learned MLP's pose call with the deterministic 2D-rule engine
    (see module docstring) and, when available, a second independent rule
    call over MediaPipe's `pose_world_landmarks` (metric-scale 3D, distinct
    from the default normalized landmarks' unreliable z). Three sources
    voting beats two: world-landmarks genuinely carries depth information the
    2D path deliberately discards, so when the 2D path and MLP disagree,
    checking whether either one agrees with the independent 3D read gives a
    real tiebreak instead of always defaulting to one side.

    Without world_angles, falls back to the original 2-way MLP-vs-2D-rules
    comparison (unchanged behavior for any caller that hasn't been updated to
    send world landmarks yet).
    """
    rule_pose_2d = classify_pose(angles_2d)

    if world_angles is None:
        if rule_pose_2d == mlp_pose:
            predicted_pose, correctness, devs = mlp_pose, mlp_correctness, mlp_devs
        else:
            predicted_pose = rule_pose_2d
            correctness, devs = score_pose(rule_pose_2d, angles_2d)
        return sanitize_pose(predicted_pose), correctness, devs

    rule_pose_world = classify_pose(world_angles)

    if mlp_pose == rule_pose_2d:
        # MLP and the proven 2D-rule path already agree -- keep the MLP's
        # richer learned correctness/deviation output, same as the 2-way case.
        predicted_pose, correctness, devs = mlp_pose, mlp_correctness, mlp_devs
    elif rule_pose_2d == rule_pose_world:
        # Two independent geometric reads agree even though the MLP differs --
        # trust them over the MLP.
        predicted_pose = rule_pose_2d
        correctness, devs = score_pose(rule_pose_2d, angles_2d)
    elif mlp_pose == rule_pose_world:
        # MLP's call is corroborated by real depth information the 2D path
        # threw away -- worth trusting over the lone 2D-rules dissent.
        predicted_pose = mlp_pose
        correctness, devs = score_pose(mlp_pose, angles_2d)
    else:
        # All three disagree: fall back to the 2D-rules path, the one
        # independently validated at ~45.7% real-world accuracy.
        predicted_pose = rule_pose_2d
        correctness, devs = score_pose(rule_pose_2d, angles_2d)

    return sanitize_pose(predicted_pose), correctness, devs


def score_pose(pose_id: str, a: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Returns (correctness_score in [0,1], per-feature deviation in degrees).
    Deviation is distance from the matched pose's expected band for each
    constrained feature (0 if already inside the band); unconstrained
    features get 0 (no signal to report, not "perfect").
    """
    deviations = {name: 0.0 for name in FEATURE_NAMES}
    bands = _POSE_FEATURE_BANDS.get(pose_id, [])
    if not bands:
        # transition/unknown or a pose with no tracked bands (e.g. lunge_pose,
        # which is intentionally a broad catch-all with no strict form check)
        return (0.5 if pose_id != "transition/unknown" else 0.0), deviations

    total_dev = 0.0
    for name, lo, hi in bands:
        val = a.get(name, 0.0)
        if val < lo:
            dev = lo - val
        elif val > hi:
            dev = val - hi
        else:
            dev = 0.0
        deviations[name] = dev
        total_dev += dev

    mean_dev = total_dev / len(bands)
    correctness = max(0.0, min(1.0, 1.0 - mean_dev / 45.0))
    return correctness, deviations
