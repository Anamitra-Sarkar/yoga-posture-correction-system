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
        # webcam/photo distances. Combined with an optional independent
        # world-landmarks rule vote (see hybrid_classify) when the client sends
        # one, for a 3-way tiebreak instead of a straight 2-way override.
        world_angles_dict = None
        if data.world_angles is not None and len(data.world_angles) == 15:
            world_angles_dict = {FEATURE_NAMES[idx]: data.world_angles[idx] for idx in range(15)}

        predicted_pose, correctness_prob, devs_dict = hybrid_classify(
            mlp_pose, mlp_correctness, mlp_devs, angles_dict, world_angles_dict
        )

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
