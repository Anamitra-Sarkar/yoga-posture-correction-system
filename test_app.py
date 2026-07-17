import sys
import os

# Ensure the modular backend directory is in the python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from fastapi.testclient import TestClient
from app.main import app
from app.utils.geometry import FEATURE_NAMES

def test_backend():
    print("Initializing FastAPI TestClient and loading models from HF Hub...")
    with TestClient(app) as client:
        # 1. Test Root endpoint (health check & metadata)
        print("\n1. Testing health check endpoint '/'...")
        response = client.get("/")
        assert response.status_code == 200
        res_data = response.json()
        print("Health check response:", res_data)
        
        # 2. Test Frame Analysis Endpoint
        print("\n2. Testing '/api/analyse_frame' endpoint...")
        # Provide 15 dummy angle features (e.g. 180 degrees representing straight limbs)
        dummy_angles = [160.0] * 15 
        response = client.post("/api/analyse_frame", json={"angles": dummy_angles})
        if response.status_code != 200:
            print("Error details:", response.text)
        assert response.status_code == 200
        frame_res = response.json()
        print("Frame analysis response:")
        print(f"  Pose ID: {frame_res['pose_id']}")
        print(f"  Correctness Score: {frame_res['correctness_score']:.4f}")
        print(f"  Sample Deviation (knee_l): {frame_res['deviations']['knee_l']:.2f}")
        
        # 3. Test Sequence Analysis Endpoint
        print("\n3. Testing '/api/analyse_sequence' endpoint...")
        # Provide 60 frames of 99 coordinates (all zero coordinates)
        dummy_sequence = [[0.0] * 99] * 60
        response = client.post("/api/analyse_sequence", json={"coordinates": dummy_sequence})
        assert response.status_code == 200
        seq_res = response.json()
        print("Sequence analysis response:")
        print(f"  Sequence Pose: {seq_res['sequence_pose']}")
        print(f"  Confidence: {seq_res['confidence']:.4f}")
        print(f"  Static Fallback Required: {seq_res['requires_static_fallback']}")
        
        # 4. Test Correction Generation Endpoint
        print("\n4. Testing '/api/generate_correction' endpoint...")
        response = client.post("/api/generate_correction", json={
            "pose_id": "warrior_2",
            "deviations": {"knee_l": 25.0, "shoulder_l": 0.0},
            "language": "en"
        })
        assert response.status_code == 200
        corr_res = response.json()
        print("Correction response (EN):", corr_res)
        
        response = client.post("/api/generate_correction", json={
            "pose_id": "warrior_2",
            "deviations": {"knee_l": 25.0, "shoulder_l": 0.0},
            "language": "hi"
        })
        assert response.status_code == 200
        corr_res_hi = response.json()
        print("Correction response (HI):", corr_res_hi)
        
        # 5. Test Occlusion Recovery Endpoint
        print("\n5. Testing '/api/occlusion_recovery' endpoint...")
        # Create 33 dummy points, setting left elbow (index 13) visibility to 0.1 (occluded)
        # while right elbow (index 14) visibility is 1.0 (visible)
        # Symmetrically position hips (23, 24) and shoulders (11, 12) around x=0
        dummy_landmarks = [[1.0, 2.0, 3.0, 1.0]] * 33
        dummy_landmarks[11] = [-1.0, 2.0, 3.0, 1.0] # Left shoulder
        dummy_landmarks[12] = [1.0, 2.0, 3.0, 1.0]  # Right shoulder
        dummy_landmarks[23] = [-1.0, 2.0, 3.0, 1.0] # Left hip
        dummy_landmarks[24] = [1.0, 2.0, 3.0, 1.0]  # Right hip
        dummy_landmarks[13] = [0.0, 0.0, 0.0, 0.1]  # Left elbow occluded
        dummy_landmarks[14] = [2.0, 2.0, 2.0, 1.0]  # Right elbow visible at (2,2,2)
        
        response = client.post("/api/occlusion_recovery", json={"mp_landmarks": dummy_landmarks})
        assert response.status_code == 200
        occ_res = response.json()
        print("Occlusion recovery response:")
        print(f"  Method Used: {occ_res['method_used']}")
        print(f"  Recovered Joint list: {occ_res['occluded_joints_recovered']}")
        # Mirroring right elbow (2,2,2) relative to body center should give left elbow X = -2
        print(f"  Recovered left_elbow coordinate: {occ_res['fused_landmarks'][13][:3]}")
        assert occ_res['fused_landmarks'][13][0] == -2.0
        
        print("\n✅ ALL BACKEND TEST PASSES SUCCESSFULLY! Models loaded and endpoints working properly.")

if __name__ == "__main__":
    test_backend()
