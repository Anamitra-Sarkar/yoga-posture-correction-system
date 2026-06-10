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
    
    # Thread bounding
    CPU_THREADS: int = 1

settings = Settings()
