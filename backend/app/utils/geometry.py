import numpy as np

FEATURE_NAMES = [
    "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
    "hip_l", "hip_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r", "trunk_l", "trunk_r",
    "neck", "hip_abduct_l", "hip_abduct_r"
]

def calculate_angle_3d(a, b, c):
    """Calculates the 3D angle between vector BA and vector BC. Point B is the vertex."""
    ba = a - b
    bc = c - b
    
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 180.0
        
    cosine_angle = dot_product / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))

def normalize_coordinate_sequence(coords: np.ndarray) -> np.ndarray:
    """
    Translates joints to be pelvis-centered (midpoint of left and right hips)
    and scales by hip-width to ensure translation and scale invariance.
    coords shape: [60, 99]
    """
    coords_reshaped = coords.reshape(60, 33, 3)
    hip_l = coords_reshaped[:, 23, :]
    hip_r = coords_reshaped[:, 24, :]
    pelvis = (hip_l + hip_r) / 2.0
    
    # Translate
    coords_normalized = coords_reshaped - pelvis[:, None, :]
    
    # Scale by hip width
    hip_width = np.linalg.norm(hip_l - hip_r, axis=-1, keepdims=True)
    hip_width = np.where(hip_width < 1e-5, 1.0, hip_width)
    coords_normalized = coords_normalized / hip_width[:, None, :]
    
    return coords_normalized.reshape(60, 99)
