import os
import torch

class Settings:
    # API metadata
    TITLE: str = "Smart Yoga Posture Correction System API"
    VERSION: str = "1.0"
    
    # Hugging Face Settings
    HF_REPO: str = "Arko007/yoga-posture-models"
    HF_TOKEN: str = os.environ.get("HF_TOKEN") or None
    
    # Two-stage pose cascade (see app/services/cascade.py). OFF by default so that
    # deploying this code cannot change behaviour by itself; flip ENABLE_POSE_CASCADE=1
    # only after the gate checkpoint is present in GATE_REPO. If the gate cannot be
    # loaded the endpoint falls back to the original hybrid path and says so in the response.
    ENABLE_POSE_CASCADE: bool = os.environ.get("ENABLE_POSE_CASCADE", "0") == "1"
    GATE_REPO: str = os.environ.get("GATE_REPO", "Arko007/yoga-posture-models")
    GATE_MODEL_FILE: str = os.environ.get("GATE_MODEL_FILE", "mlp_3head_gate_v1.pth")
    GATE_ENCODER_FILE: str = os.environ.get("GATE_ENCODER_FILE", "mlp_3head_gate_v1_encoder.npy")

    # Sequence (ST-GCN) checkpoint. Defaults to the one that has been live since 2026-09-20 so deploying
    # this code changes nothing by itself; set STGCN_MODEL_FILE / STGCN_ENCODER_FILE as Space variables
    # (e.g. stgcn_target_v1.pth / stgcn_target_v1_encoder.npy) to switch, delete them to roll back.
    STGCN_MODEL_FILE: str = os.environ.get("STGCN_MODEL_FILE", "stgcn_transitions_v1.pth")
    STGCN_ENCODER_FILE: str = os.environ.get("STGCN_ENCODER_FILE", "stgcn_transitions_v1_encoder.npy")

    # Device mapping (CPU optimization for low-resource environments like 4GB RAM)
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Thread bounding — HF's free "cpu-basic" tier gives 2 vCPUs; 1 left the
    # second core idle on every inference call.
    CPU_THREADS: int = 2

settings = Settings()
