import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.config import settings
from app.services.hf_loader import get_mlp_model, get_stgcn_model
from app.utils.geometry import FEATURE_NAMES, normalize_coordinate_sequence
from app.utils.rules_classifier import hybrid_classify

router = APIRouter()

# Input & Output Schemas
class FrameInput(BaseModel):
    angles: List[float]
    # Optional second signal: angles derived from MediaPipe's pose_world_landmarks
    # (metric-scale 3D, separately calibrated from the default normalized
    # landmarks). Older clients that haven't been updated yet simply omit this,
    # and hybrid_classify falls back to its original 2-way MLP-vs-2D-rules logic.
    world_angles: Optional[List[float]] = None
    # Mean absolute angular velocity across the client's recent frame buffer,
    # in degrees/second. Clients compute this locally (they hold the frame
    # history; this endpoint is stateless because the Space may run multiple
    # workers). Used to tell a genuine POSE HOLD apart from a TRANSITION --
    # see MOTION_HOLD_MAX_DEG_PER_SEC. Omitted by older clients, in which case
    # motion_state is reported as "unknown" and behaviour is unchanged.
    motion: Optional[float] = None
    # The user's Digital Twin calibration profile: {joint: {min, max, resting}}.
    # When supplied, a SECOND, personalised correctness score is returned
    # alongside the universal one, crediting joints the user is already within
    # their own calibrated comfortable range for.
    calibration: Optional[Dict[str, Dict[str, float]]] = None

class FrameResponse(BaseModel):
    pose_id: str
    correctness_score: float
    deviations: Dict[str, float]
    # "holding" | "transitioning" | "unrecognized" | "unknown"
    motion_state: str = "unknown"
    # Personalised correctness, present only when `calibration` was supplied.
    personal_correctness_score: Optional[float] = None
    # Deviations after the user's calibrated range is taken into account --
    # what a personalised correction should actually be based on.
    calibrated_deviations: Optional[Dict[str, float]] = None

class SequenceInput(BaseModel):
    coordinates: List[List[float]] # Shape [60, 99]

class SequenceResponse(BaseModel):
    sequence_pose: str
    confidence: float
    requires_static_fallback: bool

# Default fallback threshold is 0.70. child_pose gets a lower one: the
# sequence model handles it well (962 training sequences, ~0.62 confidence
# on real held-out data), so trusting it down to 0.55 avoids routing to the
# static single-frame fallback for this pose specifically.
SEQUENCE_FALLBACK_THRESHOLDS = {
    "child_pose": 0.55,
}
DEFAULT_SEQUENCE_FALLBACK_THRESHOLD = 0.70

# Mean absolute angular velocity (deg/s) at or below which the body counts as
# held still rather than moving. Chosen from the kinematics rather than tuned
# on a metric: MediaPipe landmark jitter alone produces a few deg/s on a
# genuinely motionless subject, while a real vinyasa transition sweeps major
# joints through 60-120 deg in well under a second (i.e. hundreds of deg/s).
# 15 deg/s sits comfortably in the empty gap between those two regimes, so it
# does not need to be precise to be reliable.
MOTION_HOLD_MAX_DEG_PER_SEC = 15.0


def classify_motion_state(motion: Optional[float], predicted_pose: str) -> str:
    """Split the old catch-all 'transition/unknown' into distinct states.

    Returns "holding" (still, and we recognise the pose), "transitioning"
    (actively moving between poses), "unrecognized" (still, but the pose isn't
    one we can name), or "unknown" (client didn't send a motion signal).
    """
    if motion is None:
        return "unknown"
    if motion > MOTION_HOLD_MAX_DEG_PER_SEC:
        return "transitioning"
    return "unrecognized" if predicted_pose == "transition/unknown" else "holding"


def apply_calibration(
    devs: Dict[str, float],
    angles: Dict[str, float],
    calibration: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Zero out deviations for joints already inside the user's own safe range.

    The Digital Twin calibration captures what each joint can actually do for
    THIS user, so a joint inside its calibrated range shouldn't be reported as
    an error even if it sits outside the universal ideal band -- that's the
    difference between "wrong" and "different body".
    """
    out = dict(devs)
    for joint, rng in calibration.items():
        if joint not in out or joint not in angles:
            continue
        lo, hi = rng.get("min"), rng.get("max")
        if lo is None or hi is None:
            continue
        if lo <= angles[joint] <= hi:
            out[joint] = 0.0
    return out


def correctness_from_deviations(
    calibrated: Dict[str, float],
    universal: Dict[str, float],
    universal_score: float,
) -> float:
    """Scale the universal correctness up by how much calibration forgave.

    Derived from the same deviations the universal score came from, so the two
    numbers stay directly comparable: personal >= universal always, and the two
    coincide when calibration forgives nothing.
    """
    tracked = [j for j, d in universal.items() if d > 0.0]
    if not tracked:
        return universal_score
    total_universal = sum(universal[j] for j in tracked)
    if total_universal <= 0.0:
        return universal_score
    total_remaining = sum(calibrated.get(j, 0.0) for j in tracked)
    forgiven_fraction = 1.0 - (total_remaining / total_universal)
    # Close that fraction of the remaining gap to a perfect score.
    return max(0.0, min(1.0, universal_score + forgiven_fraction * (1.0 - universal_score)))

@router.post("/analyse_frame", response_model=FrameResponse)
def analyse_frame(data: FrameInput):
    if len(data.angles) != 15:
        raise HTTPException(status_code=400, detail="Frame inputs must contain exactly 15 angle features.")

    try:
        angles_dict = {FEATURE_NAMES[idx]: data.angles[idx] for idx in range(15)}

        model, classes = get_mlp_model()
        x_tensor = torch.tensor([data.angles], dtype=torch.float32).to(settings.DEVICE)

        with torch.no_grad():
            pose_logits, correctness_logit, deviations_pred = model(x_tensor)
            _, pose_idx = pose_logits.max(1)
            mlp_pose = classes[pose_idx.item()]
            mlp_correctness = torch.sigmoid(correctness_logit).item()
            devs_deg = (deviations_pred[0].cpu().numpy() * 180.0).tolist()
            mlp_devs = {FEATURE_NAMES[idx]: min(180.0, max(0.0, float(devs_deg[idx]))) for idx in range(15)}

        # Deterministic rule engine as a real-time sanity check on the MLP's
        # pose call (see Section "Real-World Generalization Gap" in the paper):
        # the MLP is trained on 3D angle features whose z-depth component is
        # only reliable at the camera framing of training video, not arbitrary
        # webcam/photo distances. Combined with an optional independent
        # world-landmarks rule vote (see hybrid_classify) when the client sends
        # one, for a 3-way tiebreak instead of a straight 2-way override.
        world_angles_dict = None
        if data.world_angles is not None and len(data.world_angles) == 15:
            world_angles_dict = {FEATURE_NAMES[idx]: data.world_angles[idx] for idx in range(15)}

        predicted_pose, correctness_prob, devs_dict = hybrid_classify(
            mlp_pose, mlp_correctness, mlp_devs, angles_dict, world_angles_dict
        )

        # Motion state. The old vocabulary collapsed two completely different
        # situations into the single "transition/unknown" label: the user
        # actively MOVING between poses, and the user holding a posture we
        # simply don't recognise. Those need opposite responses -- you don't
        # correct someone's alignment mid-vinyasa, but you do want to tell
        # them when a held posture isn't being recognised. Angular velocity
        # separates them cleanly and needs no model.
        motion_state = classify_motion_state(data.motion, predicted_pose)

        # Personalised correctness, when the client sent a calibration profile.
        personal_score = None
        calibrated_devs = None
        if data.calibration:
            calibrated_devs = apply_calibration(devs_dict, angles_dict, data.calibration)
            personal_score = correctness_from_deviations(calibrated_devs, devs_dict, correctness_prob)

        return FrameResponse(
            pose_id=predicted_pose,
            correctness_score=correctness_prob,
            deviations=devs_dict,
            motion_state=motion_state,
            personal_correctness_score=personal_score,
            calibrated_deviations=calibrated_devs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")

@router.post("/analyse_sequence", response_model=SequenceResponse)
def analyse_sequence(data: SequenceInput):
    if len(data.coordinates) != 60 or any(len(frame) != 99 for frame in data.coordinates):
        raise HTTPException(status_code=400, detail="Sequence inputs must contain exactly 60 frames of 99 coordinates each.")
        
    try:
        model, classes = get_stgcn_model()
        coords_arr = np.array(data.coordinates, dtype=np.float32)
        normalized_coords = normalize_coordinate_sequence(coords_arr)
        
        x_tensor = torch.from_numpy(normalized_coords).unsqueeze(0).float().to(settings.DEVICE)
        
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
            
            predicted_seq = classes[idx.item()]
            confidence = conf.item()
            
        threshold = SEQUENCE_FALLBACK_THRESHOLDS.get(predicted_seq, DEFAULT_SEQUENCE_FALLBACK_THRESHOLD)
        requires_fallback = (predicted_seq == "transition/unknown" or confidence < threshold)
        
        return SequenceResponse(
            sequence_pose=predicted_seq,
            confidence=confidence,
            requires_static_fallback=requires_fallback
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence analysis failed: {str(e)}")
