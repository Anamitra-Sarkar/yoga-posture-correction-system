import os
import sys
from fastapi.testclient import TestClient

# Ensure backend directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main import app as backend_app

def load_groq_key():
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key
    key_path = os.environ.get("GROQ_API_KEY_FILE", "")
    if key_path and os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()
    return None

def test_groq_backend():
    groq_key = load_groq_key()
    if not groq_key:
        print("❌ Error: Groq API key not found. Set GROQ_API_KEY or GROQ_API_KEY_FILE.")
        sys.exit(1)
        
    print("Groq API Key loaded successfully.")
    print("="*60)
    print("Testing Modular Backend App Groq Integration...")
    print("="*60)
    
    with TestClient(backend_app) as client:
        # Request with Groq Key
        payload = {
            "pose_id": "warrior_2",
            "deviations": {"knee_l": 25.0},
            "language": "hi", # Test translation to Hindi via Groq
            "groq_api_key": groq_key
        }
        print(f"Sending request to /api/generate_correction with payload: {payload}")
        response = client.post("/api/generate_correction", json=payload)
        assert response.status_code == 200, f"Failed with status: {response.status_code}"
        
        res_data = response.json()
        print("Response from Backend App:")
        print(f"  Correction Text: {res_data['correction_text']}")
        print(f"  Is Safe: {res_data['is_safe']}")
        
    print("\n✅ Groq API integration in backend tested successfully!")

if __name__ == "__main__":
    test_groq_backend()
