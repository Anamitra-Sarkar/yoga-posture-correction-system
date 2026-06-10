import os
import sys
from fastapi.testclient import TestClient

# Ensure workspace is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

def load_groq_key():
    key_path = "/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt"
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()
    return None

def test_groq_integration():
    groq_key = load_groq_key()
    if not groq_key:
        print("❌ Error: Groq API key not found in /home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt")
        sys.exit(1)
        
    print("Groq API Key loaded successfully.")
    print("="*60)
    print("Testing Root App Groq Integration...")
    print("="*60)
    
    with TestClient(app) as client:
        # Request with Groq Key
        payload = {
            "pose_id": "warrior_2",
            "deviations": {"knee_l": 25.0},
            "language": "en",
            "groq_api_key": groq_key
        }
        print(f"Sending request to /api/generate_correction with payload: {payload}")
        response = client.post("/api/generate_correction", json=payload)
        assert response.status_code == 200, f"Failed with status: {response.status_code}"
        
        res_data = response.json()
        print("Response from Root App:")
        print(f"  Correction Text: {res_data['correction_text']}")
        print(f"  Is Safe: {res_data['is_safe']}")
        
    print("\n✅ Groq API integration in root app tested successfully!")

if __name__ == "__main__":
    test_groq_integration()
