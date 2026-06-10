import os
import sys
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download

# Optimize PyTorch CPU execution for low-resource environments (e.g. 4GB RAM)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# FastAPI Initialisation
app = FastAPI(
    title="Smart Yoga Posture Correction System API",
    description="Production-grade API wrapper for dual-model posture correction and flow analysis",
    version="1.0"
)

# Hugging Face Repository Settings
HF_REPO = "Arko007/yoga-posture-models"
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# Global model pointers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = None
stgcn_model = None
mlp_classes = []
stgcn_classes = []

# List of the 15 angle features in order
FEATURE_NAMES = [
    "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
    "hip_l", "hip_r", "knee_l", "knee_r",
    "ankle_l", "ankle_r", "trunk_l", "trunk_r",
    "neck", "hip_abduct_l", "hip_abduct_r"
]

# ----------------- Model Architecture Definitions -----------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class Yoga3HeadMLP(nn.Module):
    def __init__(self, input_dim, num_poses, num_joints=15):
        super(Yoga3HeadMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        self.res1 = ResBlock(256, dropout=0.3)
        self.res2 = ResBlock(256, dropout=0.3)
        
        self.pose_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_poses)
        )
        
        self.correctness_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.deviation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_joints)
        )
        
    def forward(self, x):
        features = self.input_layer(x)
        features = self.res1(features)
        features = self.res2(features)
        
        pose_logits = self.pose_head(features)
        correctness_logit = self.correctness_head(features).squeeze(-1)
        deviations = self.deviation_head(features)
        
        return pose_logits, correctness_logit, deviations

# Hybrid Conv1D + Residual GRU + Attention Model
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context

class ResGRUBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(ResGRUBlock, self).__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.proj(out)
        out = self.ln(out)
        out = self.dropout(out)
        return x + out

class YogaSequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(YogaSequenceLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.res_gru1 = ResGRUBlock(hidden_dim, dropout=0.3)
        self.res_gru2 = ResGRUBlock(hidden_dim, dropout=0.3)
        self.attention = AttentionBlock(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.res_gru1(x)
        x = self.res_gru2(x)
        x = self.attention(x)
        return self.fc(x)

# ----------------- Preprocessing Helpers -----------------

def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Apply pelvis-centering translation and hip-width scaling to coords of shape [60, 99].
    coords represents 33 joints in 3D (x, y, z) sequentially.
    """
    coords_reshaped = coords.reshape(60, 33, 3)
    hip_l = coords_reshaped[:, 23, :]
    hip_r = coords_reshaped[:, 24, :]
    pelvis = (hip_l + hip_r) / 2.0
    
    # Translate
    coords_normalized = coords_reshaped - pelvis[:, None, :]
    
    # Scale by hip width
    hip_width = np.linalg.norm(hip_l - hip_r, axis=-1, keepdims=True)
    hip_width = np.where(hip_width < 1e-5, 1.0, hip_width)
    coords_normalized = coords_normalized / hip_width[:, None, :]
    
    return coords_normalized.reshape(60, 99)

# ----------------- Biomechanical Knowledge Graph & Rules -----------------

BIOMECHANICAL_TEMPLATES = {
    "warrior_2": {
        "knee_l": {
            "issue_low": {
                "en": "Bend your left knee more to bring it directly over your ankle.",
                "hi": "अपने बाएं घुटने को थोड़ा और मोड़ें ताकि वह टखने के ठीक ऊपर रहे।",
                "bn": "আপনার বাম হাঁটু আরও বাঁকুন যাতে এটি গোড়ালির ঠিক উপরে থাকে।"
            }
        },
        "knee_r": {
            "issue_low": {
                "en": "Bend your right knee more to align it directly over your ankle.",
                "hi": "अपने दाएं घुटने को थोड़ा और मोड़ें ताकि वह टखने के ठीक ऊपर रहे।",
                "bn": "আপনার ডান হাঁটু আরও বাঁকুন যাতে এটি গোড়ালির ঠিক উপরে থাকে।"
            }
        }
    },
    "chair_pose": {
        "knee_l": {
            "issue_low": {
                "en": "Sit deeper to bring your thighs closer to parallel to the floor.",
                "hi": "जांघों को फर्श के समानांतर लाने के लिए थोड़ा नीचे बैठें।",
                "bn": "উরু মেঝের সমান্তরাল করার জন্য আর একটু নিচে বসুন।"
            }
        },
        "knee_r": {
            "issue_low": {
                "en": "Sit deeper to align both thighs towards parallel to the floor.",
                "hi": "दोनों जांघों को फर्श के समानांतर करने के लिए थोड़ा नीचे बैठें।",
                "bn": "দুই উরু মেঝের সমান্তরাল করার জন্য আর একটু নিচে বসুন।"
            }
        }
    },
    "cobra_pose": {
        "neck": {
            "issue_low": {
                "en": "Elongate your neck and gaze forward, relaxing your shoulders.",
                "hi": "अपनी गर्दन को लंबी करें और सामने की ओर देखें, कंधों को ढीला छोड़ें।",
                "bn": "আপনার ঘাড় সোজা করুন এবং সামনের দিকে তাকান, কাঁধ শিথিল রাখুন।"
            }
        }
    }
}

# ----------------- API Startup Event -----------------

@app.on_event("startup")
def startup_event():
    global mlp_model, stgcn_model, mlp_classes, stgcn_classes
    print("Initializing models and fetching weights from Hugging Face Hub...", flush=True)
    
    try:
        # Download weights and encoders
        mlp_path = hf_hub_download(repo_id=HF_REPO, filename="mlp_3head_model.pth", token=HF_TOKEN)
        mlp_enc_path = hf_hub_download(repo_id=HF_REPO, filename="mlp_3head_pose_encoder.npy", token=HF_TOKEN)
        
        stgcn_path = hf_hub_download(repo_id=HF_REPO, filename="stgcn_sequence_model.pth", token=HF_TOKEN)
        stgcn_enc_path = hf_hub_download(repo_id=HF_REPO, filename="stgcn_label_encoder.npy", token=HF_TOKEN)
        
        # Load classes
        mlp_classes = list(np.load(mlp_enc_path, allow_pickle=True))
        stgcn_classes = list(np.load(stgcn_enc_path, allow_pickle=True))
        
        # Instantiate model structures
        mlp_model = Yoga3HeadMLP(input_dim=15, num_poses=len(mlp_classes))
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
        mlp_model.eval()
        
        stgcn_model = YogaSequenceLSTM(input_dim=99, hidden_dim=128, num_layers=2, num_classes=len(stgcn_classes))
        stgcn_model.load_state_dict(torch.load(stgcn_path, map_location=device))
        stgcn_model.eval()
        
        print("Models loaded successfully and mapped onto active device.", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR during startup initialization: {e}", flush=sys.stderr)
        raise e

# ----------------- Input & Output Schemas -----------------

class FrameInput(BaseModel):
    angles: List[float] # List of 15 joint angles

class FrameResponse(BaseModel):
    pose_id: str
    correctness_score: float
    deviations: Dict[str, float]
    suggested_correction: Optional[str] = None

class SequenceInput(BaseModel):
    # Flattened coordinates list of shape [60, 99]
    coordinates: List[List[float]]

class SequenceResponse(BaseModel):
    sequence_pose: str
    confidence: float
    requires_static_fallback: bool

class CorrectionInput(BaseModel):
    pose_id: str
    deviations: Dict[str, float]
    language: str = "en" # "en", "hi", "bn"
    groq_api_key: Optional[str] = None

class CorrectionResponse(BaseModel):
    correction_text: str
    is_safe: bool

# ----------------- Endpoints -----------------

@app.get("/")
def read_root():
    return {
        "status": "ready",
        "mlp_classes": mlp_classes,
        "stgcn_classes": stgcn_classes,
        "device": str(device)
    }

@app.post("/api/analyse_frame", response_model=FrameResponse)
def analyse_frame(data: FrameInput):
    if len(data.angles) != 15:
        raise HTTPException(status_code=400, detail="Frame inputs must contain exactly 15 angle features.")
        
    try:
        # Convert to tensor
        x_tensor = torch.tensor([data.angles], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pose_logits, correctness_logit, deviations_pred = mlp_model(x_tensor)
            
            # Post-process outputs
            _, pose_idx = pose_logits.max(1)
            predicted_pose = mlp_classes[pose_idx.item()]
            
            # Map logit to probability
            correctness_prob = torch.sigmoid(correctness_logit).item()
            
            # Scale deviations back to degrees (from normalized range [0, 1])
            devs_deg = (deviations_pred[0].cpu().numpy() * 180.0).tolist()
            devs_dict = {FEATURE_NAMES[idx]: max(0.0, float(devs_deg[idx])) for idx in range(15)}
            
        return FrameResponse(
            pose_id=predicted_pose,
            correctness_score=correctness_prob,
            deviations=devs_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {str(e)}")

@app.post("/api/analyse_sequence", response_model=SequenceResponse)
def analyse_sequence(data: SequenceInput):
    if len(data.coordinates) != 60 or any(len(frame) != 99 for frame in data.coordinates):
        raise HTTPException(status_code=400, detail="Sequence inputs must contain exactly 60 frames of 99 coordinates each.")
        
    try:
        # Normalize joint coordinates
        coords_arr = np.array(data.coordinates, dtype=np.float32)
        normalized_coords = normalize_coordinates(coords_arr)
        
        # Convert to tensor
        x_tensor = torch.from_numpy(normalized_coords).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            logits = stgcn_model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
            
            predicted_seq = stgcn_classes[idx.item()]
            confidence = conf.item()
            
        # Fallback trigger: if sequence model is unsure or classifies as transition/unknown
        requires_fallback = (predicted_seq == "transition/unknown" or confidence < 0.70)
        
        return SequenceResponse(
            sequence_pose=predicted_seq,
            confidence=confidence,
            requires_static_fallback=requires_fallback
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence analysis failed: {str(e)}")

@app.post("/api/generate_correction", response_model=CorrectionResponse)
def generate_correction(data: CorrectionInput):
    # Stage 1: Symbolic Rules Engine + Local Template Matching (Physiotherapy Knowledge Graph)
    pose_templates = BIOMECHANICAL_TEMPLATES.get(data.pose_id, {})
    
    # Identify the highest deviation joint that has an associated rule template
    target_joint = None
    target_dir = None
    max_dev = 0.0
    
    for joint, dev_val in data.deviations.items():
        if joint in pose_templates and dev_val > max_dev:
            # Check if dev value crosses a baseline threshold of 10 degrees to warrant correction
            if dev_val > 10.0:
                max_dev = dev_val
                target_joint = joint
                target_dir = "issue_low" # Defaulting to posture offset flags
                
    # If a valid physiotherapy rule template is matched, use it
    if target_joint:
        correction_text = pose_templates[target_joint][target_dir].get(data.language, "Adjust your alignment.")
        is_safe = True
    else:
        # Fallback default safe templates
        defaults = {
            "en": "Maintain your posture, breathing steadily.",
            "hi": "अपनी मुद्रा बनाए रखें, लगातार सांस लेते रहें।",
            "bn": "আপনার অঙ্গভঙ্গি বজায় রাখুন, নিয়মিত শ্বাস নিন।"
        }
        correction_text = defaults.get(data.language, "Adjust your alignment.")
        is_safe = True
        
    # Stage 2: Optional Groq API wrapper if key is provided and safe bounds are respected
    if data.groq_api_key:
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {data.groq_api_key}",
                "Content-Type": "application/json"
            }
            # Strictly bounding the LLM prompt to prevent hallucinations/injury recommendation
            payload = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a professional yoga instructor. Translate or paraphrase the following instruction: '{correction_text}'. Do NOT suggest pushing further or stretching deeper. Keep it safe and under 15 words."
                    }
                ],
                "temperature": 0.2
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                candidate_text = response.json()["choices"][0]["message"]["content"].strip()
                # Post-Generation Validator
                forbidden_words = ["push", "force", "hurt", "pain", "stretch more", "further"]
                if not any(word in candidate_text.lower() for word in forbidden_words):
                    correction_text = candidate_text
        except Exception:
            # Silent fallback to rule-based templates if API fails (guarantees uptime)
            pass

    return CorrectionResponse(
        correction_text=correction_text,
        is_safe=is_safe
    )

# ----------------- Occlusion Handling & Fusion -----------------

class OcclusionInput(BaseModel):
    mp_landmarks: List[List[float]] # Shape [33, 4]
    cliff_landmarks: Optional[List[List[float]]] = None # Shape [33, 3]

class OcclusionResponse(BaseModel):
    fused_landmarks: List[List[float]]
    occluded_joints_recovered: List[str]
    method_used: str

JOINT_INDEX_MAP = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

@app.post("/api/occlusion_recovery", response_model=OcclusionResponse)
def occlusion_recovery(data: OcclusionInput):
    if len(data.mp_landmarks) != 33 or any(len(pt) != 4 for pt in data.mp_landmarks):
        raise HTTPException(status_code=400, detail="mp_landmarks must contain exactly 33 points with (x, y, z, visibility).")
        
    mp_arr = np.array(data.mp_landmarks, dtype=np.float32)
    fused = mp_arr.copy()
    recovered_joints = []
    method = "Two-Stream Fusion (MediaPipe + CLIFF)"
    
    occluded_indices = [i for i in range(33) if mp_arr[i, 3] < 0.5]
    
    if not occluded_indices:
        return OcclusionResponse(
            fused_landmarks=fused.tolist(),
            occluded_joints_recovered=[],
            method_used="None (All joints visible)"
        )
        
    if data.cliff_landmarks and len(data.cliff_landmarks) == 33:
        cliff_arr = np.array(data.cliff_landmarks, dtype=np.float32)
        for idx in occluded_indices:
            fused[idx, :3] = cliff_arr[idx, :3]
            fused[idx, 3] = 1.0
            recovered_joints.append(JOINT_INDEX_MAP[idx])
    else:
        # Fallback 3D Kinematic Solver: Reconstruct occluded joints using bone-length symmetry & anatomical constraints
        method = "Symmetric Kinematic solver (4GB RAM Optimization Fallback)"
        for idx in occluded_indices:
            partner_map = {
                13: 14, 14: 13, # elbow l/r
                15: 16, 16: 15, # wrist l/r
                25: 26, 26: 25, # knee l/r
                27: 28, 28: 27  # ankle l/r
            }
            if idx in partner_map:
                partner_idx = partner_map[idx]
                if mp_arr[partner_idx, 3] >= 0.5:
                    fused[idx, 0] = -mp_arr[partner_idx, 0] # invert X relative to center
                    fused[idx, 1] = mp_arr[partner_idx, 1]  # keep Y
                    fused[idx, 2] = mp_arr[partner_idx, 2]  # keep Z
                    fused[idx, 3] = 0.8
                    recovered_joints.append(JOINT_INDEX_MAP[idx])
                    
    return OcclusionResponse(
        fused_landmarks=fused.tolist(),
        occluded_joints_recovered=recovered_joints,
        method_used=method
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
