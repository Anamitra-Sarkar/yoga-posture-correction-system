import numpy as np
from typing import List, Optional

JOINT_INDEX_MAP = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

def fuse_and_recover_occlusions(
    mp_landmarks: List[List[float]],
    cliff_landmarks: Optional[List[List[float]]] = None,
    visibility_threshold: float = 0.5
) -> tuple:
    """
    Implements two-stream fusion. If pre-computed CLIFF 3D coordinates are provided,
    they override low-visibility joints. Otherwise, falls back to a symmetric 
    kinematic solver to mirror corresponding visible joint positions.
    """
    mp_arr = np.array(mp_landmarks, dtype=np.float32) # Shape [33, 4]
    fused = mp_arr.copy()
    recovered_joints = []
    
    # Identify occluded joints (visibility score < threshold)
    occluded_indices = [i for i in range(33) if mp_arr[i, 3] < visibility_threshold]
    
    if not occluded_indices:
        return fused.tolist(), [], "None (All joints visible)"
        
    # Case 1: CLIFF coordinates are provided
    if cliff_landmarks and len(cliff_landmarks) == 33:
        cliff_arr = np.array(cliff_landmarks, dtype=np.float32)
        for idx in occluded_indices:
            fused[idx, :3] = cliff_arr[idx, :3]
            fused[idx, 3] = 1.0 # Set visibility as recovered
            recovered_joints.append(JOINT_INDEX_MAP[idx])
        return fused.tolist(), recovered_joints, "Two-Stream Fusion (MediaPipe + CLIFF)"
        
    # Case 2: Fallback to local 3D Symmetric Kinematic solver
    for idx in occluded_indices:
        # Map symmetrical joints to query partner information
        partner_map = {
            13: 14, 14: 13, # elbow l/r
            15: 16, 16: 15, # wrist l/r
            25: 26, 26: 25, # knee l/r
            27: 28, 28: 27  # ankle l/r
        }
        if idx in partner_map:
            partner_idx = partner_map[idx]
            # Mirror the partner's coordinate space along body midplane (invert X)
            if mp_arr[partner_idx, 3] >= visibility_threshold:
                fused[idx, 0] = -mp_arr[partner_idx, 0]
                fused[idx, 1] = mp_arr[partner_idx, 1]
                fused[idx, 2] = mp_arr[partner_idx, 2]
                fused[idx, 3] = 0.8 # Confidence flag
                recovered_joints.append(JOINT_INDEX_MAP[idx])
                
    return fused.tolist(), recovered_joints, "Symmetric Kinematic solver (4GB RAM Optimization Fallback)"
