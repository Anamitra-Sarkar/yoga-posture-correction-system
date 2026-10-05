import os
import sys
import json
import numpy as np
from fastapi.testclient import TestClient

# Ensure workspace is in python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("/home/anamitra/Projects_and_Code/Scripts_and_Source/")

from app.main import app
from app.utils.geometry import FEATURE_NAMES
from extract_features_safe import calculate_angle_3d, interpolate_occlusions

LANDMARKS_PATH = "/home/anamitra/yoga_raw_dataset/landmarks_20170209_How to do a Vinyasa (Flow).npy"

# Mock User Digital Twin Profile (from Stage 1: User Calibration)
MOCK_DIGITAL_TWIN = {
    "knee_l": (80.0, 160.0),
    "knee_r": (80.0, 160.0),
    "hip_l": (60.0, 150.0),
    "hip_r": (60.0, 150.0),
    "shoulder_l": (30.0, 150.0),
    "shoulder_r": (30.0, 150.0)
}

# MediaPipe landmark indices
SHOULDER_L, SHOULDER_R = 11, 12
ELBOW_L, ELBOW_R = 13, 14
WRIST_L, WRIST_R = 15, 16
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28
HEEL_L, HEEL_R = 29, 30
NOSE = 0

def extract_frame_angles(pts):
    """Computes the 15 joint angles for a single frame landmarks pts [33, 3]"""
    shoulder_mid = (pts[SHOULDER_L] + pts[SHOULDER_R]) / 2.0
    hip_mid = (pts[HIP_L] + pts[HIP_R]) / 2.0
    
    raw_angles = [
        calculate_angle_3d(pts[SHOULDER_L], pts[ELBOW_L], pts[WRIST_L]), # elbow_l
        calculate_angle_3d(pts[SHOULDER_R], pts[ELBOW_R], pts[WRIST_R]), # elbow_r
        calculate_angle_3d(pts[HIP_L], pts[SHOULDER_L], pts[ELBOW_L]), # shoulder_l
        calculate_angle_3d(pts[HIP_R], pts[SHOULDER_R], pts[ELBOW_R]), # shoulder_r
        calculate_angle_3d(pts[SHOULDER_L], pts[HIP_L], pts[KNEE_L]), # hip_l
        calculate_angle_3d(pts[SHOULDER_R], pts[HIP_R], pts[KNEE_R]), # hip_r
        calculate_angle_3d(pts[HIP_L], pts[KNEE_L], pts[ANKLE_L]), # knee_l
        calculate_angle_3d(pts[HIP_R], pts[KNEE_R], pts[ANKLE_R]), # knee_r
        calculate_angle_3d(pts[KNEE_L], pts[ANKLE_L], pts[HEEL_L]), # ankle_l
        calculate_angle_3d(pts[KNEE_R], pts[ANKLE_R], pts[HEEL_R]), # ankle_r
        calculate_angle_3d(pts[SHOULDER_L], pts[HIP_L], pts[HIP_R]), # trunk_l
        calculate_angle_3d(pts[SHOULDER_R], pts[HIP_R], pts[HIP_L]), # trunk_r
        calculate_angle_3d(pts[NOSE], shoulder_mid, hip_mid), # neck
        calculate_angle_3d(pts[HIP_R], pts[HIP_L], pts[KNEE_L]), # hip_abduct_l
        calculate_angle_3d(pts[HIP_L], pts[HIP_R], pts[KNEE_R]) # hip_abduct_r
    ]
    return [float(x) for x in raw_angles]

def run_simulation():
    print("="*80)
    print("SMART YOGA POSTURE CORRECTION SYSTEM - PIPELINE SIMULATION")
    print(f"Source Video Landmarks: {LANDMARKS_PATH}")
    print("="*80)
    
    if not os.path.exists(LANDMARKS_PATH):
        print(f"Error: Landmarks file not found at {LANDMARKS_PATH}.")
        return
        
    # Load landmarks
    print("\nStage 3 & 4: Loading & Cleaning Raw Landmarks (Occlusion Interpolation)...")
    landmarks = np.load(LANDMARKS_PATH)
    cleaned_landmarks = interpolate_occlusions(landmarks, visibility_threshold=0.5)
    num_frames = len(cleaned_landmarks)
    print(f"Loaded {num_frames} frames of 3D skeletal data.")
    
    # Initialize TestClient
    client = TestClient(app)
    
    # Session Logs data
    session_pose_log = []
    session_correctness_log = []
    session_flow_log = []
    
    # Simulate processing video in chunks of 60 frames (Stage 5: Temporal buffer)
    step_size = 60
    print(f"\nProcessing video in temporal windows of size {step_size}...")
    
    with TestClient(app) as client:
        for start_idx in range(0, num_frames - step_size, 150): # Skip frames to simulate real-time sampling
            end_idx = start_idx + step_size
            
            # 1. Stage 3 & 4: Load raw frame landmarks and run Occlusion Recovery
            raw_frame = landmarks[end_idx - 1].tolist() # Shape [33, 4]
            occ_response = client.post("/api/occlusion_recovery", json={"mp_landmarks": raw_frame})
            if occ_response.status_code == 200:
                fused_frame = np.array(occ_response.json()["fused_landmarks"])[:, :3] # [33, 3]
                recovered = occ_response.json()["occluded_joints_recovered"]
            else:
                fused_frame = cleaned_landmarks[end_idx - 1, :, :3]
                recovered = []
                
            # 2. Stage 5: Extract angles from the fused coordinate output
            last_frame_angles = extract_frame_angles(fused_frame)
            
            # Use rolling window buffer coordinates for sequence flow
            window_coords = cleaned_landmarks[start_idx:end_idx, :, :3]
            flat_coords = window_coords.reshape(60, 99).tolist()
            
            # 3. Stage 7: Sequence Flow Analysis
            seq_response = client.post("/api/analyse_sequence", json={"coordinates": flat_coords})
            if seq_response.status_code != 200:
                continue
            seq_data = seq_response.json()
            
            # 4. Stage 6: Frame-level static classification
            frame_response = client.post("/api/analyse_frame", json={"angles": last_frame_angles})
            if frame_response.status_code != 200:
                continue
            frame_data = frame_response.json()
            
            # 4. Stage 8: Personalised Digital Twin Comparison & Safety Checks
            pose_id = frame_data["pose_id"]
            correctness = frame_data["correctness_score"]
            deviations = frame_data["deviations"]
            
            # Adjust deviations based on user-calibrated safe ranges (Digital Twin Comparison)
            dt_deviations = {}
            for joint, val in deviations.items():
                if joint in MOCK_DIGITAL_TWIN:
                    low_limit, high_limit = MOCK_DIGITAL_TWIN[joint]
                    angle_val = last_frame_angles[FEATURE_NAMES.index(joint)]
                    # If joint angle falls inside user's safe mobility limits, override deviation to 0
                    if low_limit <= angle_val <= high_limit:
                        dt_deviations[joint] = 0.0
                    else:
                        dt_deviations[joint] = val
                else:
                    dt_deviations[joint] = val
            
            # 5. Stage 9 & 10: Generate Safety-Bounded Corrections
            corr_response = client.post("/api/generate_correction", json={
                "pose_id": pose_id,
                "deviations": dt_deviations,
                "language": "en"
            })
            corr_text = corr_response.json()["correction_text"] if corr_response.status_code == 200 else "Maintain alignment."
            
            # 6. Log results
            session_pose_log.append(pose_id)
            session_correctness_log.append(correctness)
            session_flow_log.append(seq_data["sequence_pose"])
            
            print(f"\nFrame {end_idx:04d} | Window Sequence Pose: {seq_data['sequence_pose']} (Conf: {seq_data['confidence']:.2f})")
            if recovered:
                print(f"  🔧 Occlusion Recovery (Stage 4): Fused {len(recovered)} joint(s) {recovered}")
            print(f"  --> Static Pose ID: {pose_id} | Correctness Score: {correctness*100:.1f}%")
            if correctness < 0.70 and pose_id != 'transition/unknown':
                print(f"  ⚠️ ALERT (Stage 11: Audio Guidance): \"{corr_text}\"")
                
    # Stage 13: Session Logging & Summary Generation
    print("\n" + "="*80)
    print("Stage 13: Compiling Session Logs & Analytics Summary...")
    print("="*80)
    
    unique_poses = list(set(session_pose_log))
    avg_correctness = float(np.mean(session_correctness_log)) if session_correctness_log else 0.0
    
    summary = {
        "video_source": os.path.basename(LANDMARKS_PATH),
        "total_frames_processed": num_frames,
        "average_correctness_score": f"{avg_correctness*100:.2f}%",
        "poses_detected": unique_poses,
        "sequence_flow_trajectory": session_flow_log[:15] # first 15 steps
    }
    
    summary_path = "/home/anamitra/yoga_posture_workspace/session_summary_20170209.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print(json.dumps(summary, indent=4))
    print(f"\nSession summary JSON successfully compiled and saved to: {summary_path}")
    print("="*80)

if __name__ == "__main__":
    run_simulation()
