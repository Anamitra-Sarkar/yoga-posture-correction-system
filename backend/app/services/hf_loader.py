import os
import sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from app.config import settings
from app.models.mlp import Yoga3HeadMLP
from app.models.sequence import YogaSequenceLSTM

# Singleton pointers for loaded models and classes
mlp_model = None
stgcn_model = None
mlp_classes = []
stgcn_classes = []

def initialize_models():
    """Download models and label encoders from Hugging Face Hub and load them into memory."""
    global mlp_model, stgcn_model, mlp_classes, stgcn_classes
    
    if mlp_model is not None and stgcn_model is not None:
        return # Models are already loaded
        
    print("Fetching models and encoders from Hugging Face Hub...", flush=True)
    
    try:
        # Download files from Hugging Face Hub
        mlp_path = hf_hub_download(repo_id=settings.HF_REPO, filename="mlp_3head_model.pth", token=settings.HF_TOKEN)
        mlp_enc_path = hf_hub_download(repo_id=settings.HF_REPO, filename="mlp_3head_pose_encoder.npy", token=settings.HF_TOKEN)
        
        stgcn_path = hf_hub_download(repo_id=settings.HF_REPO, filename="stgcn_sequence_model.pth", token=settings.HF_TOKEN)
        stgcn_enc_path = hf_hub_download(repo_id=settings.HF_REPO, filename="stgcn_label_encoder.npy", token=settings.HF_TOKEN)
        
        # Load encoders
        mlp_classes = list(np.load(mlp_enc_path, allow_pickle=True))
        stgcn_classes = list(np.load(stgcn_enc_path, allow_pickle=True))
        
        # Load 3-head MLP Model
        mlp_model = Yoga3HeadMLP(input_dim=15, num_poses=len(mlp_classes))
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=settings.DEVICE))
        mlp_model.eval()
        
        # Load ST-GCN Sequence model
        stgcn_model = YogaSequenceLSTM(input_dim=99, hidden_dim=128, num_layers=2, num_classes=len(stgcn_classes))
        stgcn_model.load_state_dict(torch.load(stgcn_path, map_location=settings.DEVICE))
        stgcn_model.eval()
        
        print("All models successfully fetched and loaded into memory.", flush=True)
        
    except Exception as e:
        print(f"CRITICAL ERROR during model loading: {e}", file=sys.stderr)
        raise e

def get_mlp_model():
    initialize_models()
    return mlp_model, mlp_classes

def get_stgcn_model():
    initialize_models()
    return stgcn_model, stgcn_classes
