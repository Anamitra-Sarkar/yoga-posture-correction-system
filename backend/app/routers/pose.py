import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.config import settings
from app.services.hf_loader import get_mlp_model, get_stgcn_model
from app.utils.geometry import FEATURE_NAMES, normalize_coordinate_sequence
from app.utils.rules_classifier import classify_pose, score_pose

router = APIRouter()

# Input & Output Schemas
class FrameInput(BaseModel):
    angles: List[float]

class FrameResponse(BaseModel):
    pose_id: str
    correctness_score: float
    deviations: Dict[str, float]

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
        # webcam/photo distances. When the rule engine -- which never depended
        # on that unreliable z signal -- disagrees with the MLP's pose call,
        # its independently-derived answer is trusted instead. When they
        # agree, the MLP's richer learned correctness/deviation output is kept.
        rule_pose = classify_pose(angles_dict)

        if rule_pose == mlp_pose:
            predicted_pose = mlp_pose
            correctness_prob = mlp_correctness
            devs_dict = mlp_devs
        else:
            predicted_pose = rule_pose
            correctness_prob, devs_dict = score_pose(rule_pose, angles_dict)

        return FrameResponse(
            pose_id=predicted_pose,
            correctness_score=correctness_prob,
            deviations=devs_dict
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
