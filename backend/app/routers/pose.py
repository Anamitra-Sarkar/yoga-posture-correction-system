import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.config import settings
from app.services.hf_loader import get_mlp_model, get_stgcn_model
from app.utils.geometry import FEATURE_NAMES, normalize_coordinate_sequence

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

@router.post("/analyse_frame", response_model=FrameResponse)
def analyse_frame(data: FrameInput):
    if len(data.angles) != 15:
        raise HTTPException(status_code=400, detail="Frame inputs must contain exactly 15 angle features.")
        
    try:
        model, classes = get_mlp_model()
        x_tensor = torch.tensor([data.angles], dtype=torch.float32).to(settings.DEVICE)
        
        with torch.no_grad():
            pose_logits, correctness_logit, deviations_pred = model(x_tensor)
            
            _, pose_idx = pose_logits.max(1)
            predicted_pose = classes[pose_idx.item()]
            
            correctness_prob = torch.sigmoid(correctness_logit).item()
            
            # Scale deviations back to degrees
            devs_deg = (deviations_pred[0].cpu().numpy() * 180.0).tolist()
            devs_dict = {FEATURE_NAMES[idx]: max(0.0, float(devs_deg[idx])) for idx in range(15)}
            
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
            
        requires_fallback = (predicted_seq == "transition/unknown" or confidence < 0.70)
        
        return SequenceResponse(
            sequence_pose=predicted_seq,
            confidence=confidence,
            requires_static_fallback=requires_fallback
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence analysis failed: {str(e)}")
