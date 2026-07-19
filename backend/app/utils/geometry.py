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

SHOULDER_L, SHOULDER_R = 11, 12
ELBOW_L, ELBOW_R = 13, 14
WRIST_L, WRIST_R = 15, 16
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28
HEEL_L, HEEL_R = 29, 30
NOSE = 0

def extract_angles_from_landmarks(points: np.ndarray) -> list:
    """
    Mirrors frontend/src/utils/geometry.ts's extractAnglesFromLandmarks, so the
    Gradio demo (which runs MediaPipe server-side on an uploaded/webcam image)
    computes the exact same 15 biomechanical features, in the same order as
    FEATURE_NAMES, that the client-side pipeline sends to /api/analyse_frame.
    points shape: [33, 3] (x, y, z per MediaPipe landmark)
    """
    if points.shape[0] < 31:
        return [0.0] * 15

    shoulder_mid = (points[SHOULDER_L] + points[SHOULDER_R]) / 2.0
    hip_mid = (points[HIP_L] + points[HIP_R]) / 2.0

    return [
        calculate_angle_3d(points[SHOULDER_L], points[ELBOW_L], points[WRIST_L]),
        calculate_angle_3d(points[SHOULDER_R], points[ELBOW_R], points[WRIST_R]),
        calculate_angle_3d(points[HIP_L], points[SHOULDER_L], points[ELBOW_L]),
        calculate_angle_3d(points[HIP_R], points[SHOULDER_R], points[ELBOW_R]),
        calculate_angle_3d(points[SHOULDER_L], points[HIP_L], points[KNEE_L]),
        calculate_angle_3d(points[SHOULDER_R], points[HIP_R], points[KNEE_R]),
        calculate_angle_3d(points[HIP_L], points[KNEE_L], points[ANKLE_L]),
        calculate_angle_3d(points[HIP_R], points[KNEE_R], points[ANKLE_R]),
        calculate_angle_3d(points[KNEE_L], points[ANKLE_L], points[HEEL_L]),
        calculate_angle_3d(points[KNEE_R], points[ANKLE_R], points[HEEL_R]),
        calculate_angle_3d(points[SHOULDER_L], points[HIP_L], points[HIP_R]),
        calculate_angle_3d(points[SHOULDER_R], points[HIP_R], points[HIP_L]),
        calculate_angle_3d(points[NOSE], shoulder_mid, hip_mid),
        calculate_angle_3d(points[HIP_R], points[HIP_L], points[KNEE_L]),
        calculate_angle_3d(points[HIP_L], points[HIP_R], points[KNEE_R]),
    ]

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
