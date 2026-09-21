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
        # mlp_3head_v4_photos_x2000.pth (2026-09-21). Two improvements over
        # mlp_3head_photodomain_v1, measured on the identical frozen 103-photo
        # set and re-verified here from the saved weights rather than trusting
        # the training run's own report:
        #
        #   pose macro   35.5% -> 47.8%
        #
        # and, more importantly, ALL THREE HEADS ARE TRAINED. The photo-domain
        # scripts optimised only the pose head, so correctness_head and
        # deviation_head shipped at random initialisation -- and hybrid_classify
        # serves both directly whenever the MLP and the 2D rules agree, which is
        # the confident, common case. The per-joint coaching numbers the app has
        # been showing were therefore noise. Verified non-random here:
        # correctness spans [0.000, 1.000] with std 0.420, deviations reach
        # 131 degrees, neither of which a freshly-initialised head produces.
        #
        # The gain comes from data, not architecture: a 10x photo corpus (587 ->
        # 6,242) extracted from four public datasets, which finally gave the
        # starved classes real support. cobra_pose 0% -> 100%, seated_staff
        # 0% -> 80%, triangle -> 100%, corpse -> 100%.
        #
        # Trade-off worth knowing: heavier photo oversampling helps diverse
        # real-world imagery and costs same-shoot video accuracy (photos_x2000
        # scores 47.8% photo / 65.9% held-out video; photos_x1 is 30.8% /
        # 84.3%). x2000 is chosen because the app is judged on arbitrary users
        # in arbitrary rooms, which is the diverse-imagery condition -- the
        # held-out videos are 3 clips from the same 12-video shoot and flatter
        # a model that has memorised that production style.
        # mlp_3head_v4_photos_x365.pth is the balanced alternative.
        #
        # Encoder verified identical in content AND order to the previous
        # checkpoint's, so this is a pure swap with no relabelling. Every
        # earlier checkpoint remains on HF untouched for rollback.
        mlp_path = hf_hub_download(repo_id=settings.HF_REPO, filename="mlp_3head_v4_photos_x2000.pth", token=settings.HF_TOKEN)
        mlp_enc_path = hf_hub_download(repo_id=settings.HF_REPO, filename="mlp_3head_v4_encoder.npy", token=settings.HF_TOKEN)

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
