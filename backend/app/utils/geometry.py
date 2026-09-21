import math
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

def extract_angles_from_landmarks(points: np.ndarray, zero_z: bool = True) -> list:
    """
    Mirrors frontend/src/utils/geometry.ts's extractAnglesFromLandmarks, so the
    Gradio demo (which runs MediaPipe server-side on an uploaded/webcam image)
    computes the exact same 15 biomechanical features, in the same order as
    FEATURE_NAMES, that the client-side pipeline sends to /api/analyse_frame.
    points shape: [33, 3] (x, y, z per MediaPipe landmark)

    MediaPipe's default (image-normalized) z-depth estimate is only reliable
    at the consistent camera distance/framing seen in demo videos; on
    arbitrary real-world camera framing it degrades badly (measured 0-3%
    real-world pose accuracy with z included, vs ~46% with it dropped,
    confirmed across a 40-trial randomized threshold sweep). zero_z=True (the
    default) reproduces that proven 2D-only path. zero_z=False is for the
    separately-calibrated `pose_world_landmarks` (metric-scale 3D) signal,
    which keeps genuine depth information and is used as an independent
    second vote alongside the 2D path rather than as a replacement for it
    (see rules_classifier.hybrid_classify).
    """
    if points.shape[0] < 31:
        return [0.0] * 15

    points = points.copy()
    if zero_z:
        points[:, 2] = 0.0

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


# --- Global body orientation -------------------------------------------------
#
# The 15 features in FEATURE_NAMES are all RELATIVE joint angles
# (shoulder-hip-knee and friends), which makes them invariant to rotating the
# whole body. Measured consequence on the real-photo corpus: a person lying
# flat in corpse and a person standing in mountain_pose produce nearly the same
# 15-vector, and the rule engine called corpse "mountain_pose" 11 times out of
# 18. No threshold change can fix that -- the information is simply absent from
# the feature vector.
#
# These two scalars restore it. Both are translation- and scale-invariant and
# come from landmarks the client already has, so they cost nothing to compute
# and nothing extra to transmit beyond two floats.

def compute_orientation(points: np.ndarray) -> dict:
    """Global orientation cues that the relative joint angles cannot express.

    torso_incline: degrees between the shoulders->hips vector and image-down.
        ~0 standing, ~90 lying flat or in plank, ~180 inverted (downward dog).
        Measured medians: mountain 6, cobra 43, table_top 61, triangle 71,
        plank 80, corpse 107, downward_dog 148.

    leg_torso_ratio: mean hip->ankle distance over shoulder->hip distance, in
        image space. Cleanly separates cross-legged sitting (0.57) from every
        standing pose (~1.2-1.4), which bounding-box aspect ratio does NOT do
        reliably (a standing person with arms down is just as tall and narrow
        as a seated one photographed head-on).

    Returns {} when the landmarks are degenerate, so callers can treat missing
    orientation the same as an older client that never sends it.
    """
    p = np.asarray(points, dtype=np.float64)[:, :2]
    shoulder_mid = (p[SHOULDER_L] + p[SHOULDER_R]) / 2.0
    hip_mid = (p[HIP_L] + p[HIP_R]) / 2.0

    torso_vec = hip_mid - shoulder_mid
    torso_len = float(np.linalg.norm(torso_vec))
    if torso_len < 1e-9:
        return {}

    # MediaPipe's y axis grows downward, so image-down is (0, +1) and the
    # inclination is just the angle of the torso vector against it.
    cos_i = float(torso_vec[1] / torso_len)
    torso_incline = math.degrees(math.acos(max(-1.0, min(1.0, cos_i))))

    leg_len = (float(np.linalg.norm(p[ANKLE_L] - p[HIP_L])) +
               float(np.linalg.norm(p[ANKLE_R] - p[HIP_R]))) / 2.0

    return {
        "torso_incline": torso_incline,
        "leg_torso_ratio": leg_len / torso_len,
    }
