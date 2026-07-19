import os
import torch

class Settings:
    # API metadata
    TITLE: str = "Smart Yoga Posture Correction System API"
    VERSION: str = "1.0"
    
    # Hugging Face Settings
    HF_REPO: str = "Arko007/yoga-posture-models"
    HF_TOKEN: str = os.environ.get("HF_TOKEN") or None
    
    # Device mapping (CPU optimization for low-resource environments like 4GB RAM)
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Thread bounding — HF's free "cpu-basic" tier gives 2 vCPUs; 1 left the
    # second core idle on every inference call.
    CPU_THREADS: int = 2

settings = Settings()
