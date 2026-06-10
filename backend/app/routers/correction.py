from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.services.correction import generate_safe_correction
from app.services.occlusion import fuse_and_recover_occlusions

router = APIRouter()

# Input & Output Schemas
class CorrectionInput(BaseModel):
    pose_id: str
    deviations: Dict[str, float]
    language: str = "en"
    groq_api_key: Optional[str] = None

class CorrectionResponse(BaseModel):
    correction_text: str
    is_safe: bool

class OcclusionInput(BaseModel):
    mp_landmarks: List[List[float]] # Shape [33, 4]
    cliff_landmarks: Optional[List[List[float]]] = None # Shape [33, 3]

class OcclusionResponse(BaseModel):
    fused_landmarks: List[List[float]]
    occluded_joints_recovered: List[str]
    method_used: str

@router.post("/generate_correction", response_model=CorrectionResponse)
def generate_correction(data: CorrectionInput):
    try:
        text, safe = generate_safe_correction(
            pose_id=data.pose_id,
            deviations=data.deviations,
            language=data.language,
            groq_api_key=data.groq_api_key
        )
        return CorrectionResponse(
            correction_text=text,
            is_safe=safe
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correction generation failed: {str(e)}")

@router.post("/occlusion_recovery", response_model=OcclusionResponse)
def occlusion_recovery(data: OcclusionInput):
    if len(data.mp_landmarks) != 33 or any(len(pt) != 4 for pt in data.mp_landmarks):
        raise HTTPException(status_code=400, detail="mp_landmarks must contain exactly 33 points with (x, y, z, visibility).")
        
    try:
        fused, recovered, method = fuse_and_recover_occlusions(
            mp_landmarks=data.mp_landmarks,
            cliff_landmarks=data.cliff_landmarks
        )
        return OcclusionResponse(
            fused_landmarks=fused,
            occluded_joints_recovered=recovered,
            method_used=method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Occlusion recovery failed: {str(e)}")
