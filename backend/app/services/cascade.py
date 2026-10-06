"""Two-stage pose cascade (validated on the real production functions; see docs/CASCADE.md).

Stage 1 -- the GATE (a second 3-head MLP, today's model): "is this one of my poses at all?" and the form score.
Stage 2 -- the NAMER (the existing live MLP): "which pose?"
The rule engine no longer overrides either model on the pose NAME; it keeps the job it is good at, per-joint
deviations from each pose's angle bands.

Why this exists: through hybrid_classify the rule engine wins every MLP/rules disagreement, so swapping in a better
MLP changed nothing (identical to rules-only: 36.9% overall, 78% false alarms on held-out photos). The cascade measured
78.9% overall / 20.8% false alarms / 7 poses at 70-70 on the same photos, chosen on a validation half and confirmed on an
untouched half. This module is pure logic (no torch) so it is unit-tested directly.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.utils.geometry import FEATURE_NAMES
from app.utils.rules_classifier import _POSE_FEATURE_BANDS, sanitize_pose, score_pose

UNKNOWN = "transition/unknown"


def collapse(dist: Dict[str, float]) -> Dict[str, float]:
    """Sum probability mass of 'imperfect_<pose>' classes onto the base pose."""
    out: Dict[str, float] = {}
    for name, p in dist.items():
        base = name.replace("imperfect_", "")
        out[base] = out.get(base, 0.0) + float(p)
    return out


def top_k(dist: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(dist.items(), key=lambda kv: -kv[1])[:k]


@dataclass
class CascadeResult:
    pose_id: str
    correctness: float
    deviations: Dict[str, float]
    gated: bool                      # stage 1 said "not one of my poses"
    gate_pose: str                   # stage 1's own top class (collapsed)
    gate_other_prob: float
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    correctness_source: str = "none"
    deviations_source: str = "none"


def cascade_decide(
    namer_pose: str,
    namer_dist: Dict[str, float],
    gate_dist: Dict[str, float],
    angles: Dict[str, float],
    gate_correctness: float,
    gate_devs: Dict[str, float],
) -> CascadeResult:
    gd = collapse(gate_dist)
    gate_pose = max(gd, key=gd.get) if gd else UNKNOWN
    other_p = gd.get(UNKNOWN, 0.0)
    cands = top_k(collapse(namer_dist), 3)
    gated = gate_pose == UNKNOWN
    pose = UNKNOWN if gated else sanitize_pose(namer_pose.replace("imperfect_", ""))
    if pose == UNKNOWN:
        c, d = score_pose(UNKNOWN, angles)
        return CascadeResult(UNKNOWN, c, d, gated, gate_pose, other_p, cands, "none", "none")
    # Form score from the gate's correctness head (the one that reacts most to a broken joint in testing);
    # per-joint deviations from the pose's own angle bands where it has them (definitional), else the head's output.
    if _POSE_FEATURE_BANDS.get(pose):
        _, devs = score_pose(pose, angles)
        dsrc = "rule_bands"
    else:
        # No angle bands (tree_pose, lunge_pose). The only other per-joint source is the gate's deviation head, which is close to
        # chance (docs/BENCHMARKS.md section 3: top joint correct 12.4% vs 6.7% chance) -- it painted correct Trees with a coral
        # standing leg and named the wrong joint in cues. Say "no per-joint evidence" instead of inventing some.
        devs = {n: 0.0 for n in FEATURE_NAMES}
        dsrc = "none_no_bands"
    return CascadeResult(pose, float(gate_correctness), devs, False, gate_pose, other_p, cands, "gate_head", dsrc)


def guided_report(target_pose: str, detected_pose: str, angles: Dict[str, float]) -> Dict:
    """Guided mode: the user chose a pose. Score the body against THAT pose's bands and say whether the detector agrees."""
    corr, devs = score_pose(target_pose, angles)
    return {
        "target_pose": target_pose,
        "matches": detected_pose == target_pose,
        "target_correctness": corr,
        "target_deviations": devs,
        "target_has_bands": bool(_POSE_FEATURE_BANDS.get(target_pose)),
    }
