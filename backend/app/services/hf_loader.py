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
        # mlp_3head_photodomain_v1.pth (2026-09-20). mlp_3head_model_v2.pth (the
        # zero_z-retrained checkpoint, live until today) was re-measured against
        # the app's real judging condition -- a single held-out 103-photo set
        # spanning the full 23-class vocabulary, identical for both models -- and
        # scored only 10.5% macro / 13.6% overall. That is the "model is too bad"
        # the live app was actually shipping; the previously-quoted 52.6% covered
        # only 6 well-supported poses on a 19-photo set, not this benchmark.
        # photodomain_v1 was trained on the SAME video corpus plus real photographs
        # (oversampled so ~319 photos aren't drowned by ~650k video frames), and
        # scores 35.5% macro / 45.6% overall on the identical 103-photo set -- a
        # >3x macro gain, verified before this swap, not assumed from its own
        # training run. Same 23-class vocabulary, same order, as confirmed by
        # diffing the two encoder files -- so this is a pure checkpoint swap, no
        # downstream remapping. Previous weights remain at mlp_3head_model_v2.pth
        # (and the original mlp_3head_model.pth before it) for instant rollback.
        mlp_path = hf_hub_download(repo_id=settings.HF_REPO, filename="mlp_3head_photodomain_v1.pth", token=settings.HF_TOKEN)
        mlp_enc_path = hf_hub_download(repo_id=settings.HF_REPO, filename="mlp_3head_photodomain_v1_encoder.npy", token=settings.HF_TOKEN)

        # stgcn_transitions_v1.pth (2026-09-20). The previous stgcn_sequence_model.pth
        # was the original 15-class checkpoint that had NEVER been shown a
        # transition -- confirmed live, /api/analyse_sequence always answered
        # "transition/unknown". This one is trained on relabelled hold/transition
        # windows (24 classes: "hold:<pose>" / "transition:<A>-><B>" / "unrecognized"),
        # measured 63.0% macro with all 8 transition classes learned (72.7-100%
        # accuracy on the named ones), vs 54.6% for the old 15-class scheme with
        # zero transition coverage. Its residual-block keys were originally named
        # "block1.res.*"; this file has them remapped to "block1.residual.*" and was
        # strict-load-verified against the production YogaSequenceLSTM class with a
        # real forward pass before upload. The routing layer that makes
        # /api/analyse_sequence behave identically under either label vocabulary
        # (parse_sequence_label in routers/pose.py) shipped and was tested BEFORE
        # this swap, so this line is the only behavioural change.
        stgcn_path = hf_hub_download(repo_id=settings.HF_REPO, filename="stgcn_transitions_v1.pth", token=settings.HF_TOKEN)
        stgcn_enc_path = hf_hub_download(repo_id=settings.HF_REPO, filename="stgcn_transitions_v1_encoder.npy", token=settings.HF_TOKEN)
        
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
