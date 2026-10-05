"""Old vs new ST-GCN on the SAME held-out videos, plus (1) a train/serve PARITY check and (2) a FRAME-RATE sweep.
 - new = cueT2_stgcn_f{0,1,2}: each scored ONLY on the videos it never trained on (3 folds by video).
 - old = the three ST-GCNs in the live model repo (stgcn_sequence_model.pth, stgcn_sequence_model_v2.pth, stgcn_transitions_v1.pth = LIVE).
   They were trained before the 12 new videos existed, so the NEW-video subset is clean for them; the 12 old videos may be in-sample for them (flagged).
 - rate sweep: the web app feeds ~0.5-2 fps into the 60-frame buffer; training windows are 60 CONSECUTIVE native-fps frames (~2.4 s).
   We rebuild the held-out windows sampling every k-th frame (k=1 = training scale) and re-score everything.
"""
import collections, glob, importlib.util, json, os, sys, time
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import numpy as np, pandas as pd, requests, torch
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"; DS, MD, LIVE = "Arko007/Yoga-1M", "Arko007/asanaai-conference-runs", "Arko007/yoga-posture-models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device", DEV, flush=True)
snapshot_download(DS, repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/originals/**", "vol/csv/cue/cue_full.csv", "vol/landmarks/**", "vol/landmarks_new/**", "vol/stgcn/cueT2_f[012]/test_*.npy"])
snapshot_download(MD, repo_type="model", local_dir=f"{ROOT}/vol", token=HF, allow_patterns=["runs/cueT2_stgcn_f[012]/**"])
sys.path.insert(0, f"{ROOT}/code/originals")
spec = importlib.util.spec_from_file_location("stgcn_trainer", f"{ROOT}/code/originals/train_stgcn_gpu.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
import generate_sequence_features as gsf
U = "transition/unknown"
TARGETS = "downward_dog,warrior_2,tree_pose,triangle,seated_easy_pose,child_pose,corpse,plank".split(",")
OLD_IDS = "HmZFwoUU3WQ SZU7Sbgu57o oUgpXY7QhpQ 7ciS93shMNQ P8uHMMmWMHQ 4ORRiN2_aVI RQMtwbhXD7A 8ibxmzJziHU Eml2xnoLpYE QiebZSlTw_U L-z1HLkS_-Y s-1vMbAgYWU".split()
SEQ, STRIDE = gsf.SEQ_LENGTH, gsf.STRIDE

# ---------------------------------------------------------------- models
def load_model(path, enc):
    cls = [str(c) for c in np.load(enc, allow_pickle=True)]
    m = tr.YogaSequenceLSTM(99, 128, 2, len(cls)); m.load_state_dict(torch.load(path, map_location="cpu")); return m.to(DEV).eval(), cls
MODELS = {}
for i in range(3):
    MODELS[f"NEW f{i}"] = load_model(f"{ROOT}/vol/runs/cueT2_stgcn_f{i}/stgcn_sequence_model_v2.pth", f"{ROOT}/vol/runs/cueT2_stgcn_f{i}/stgcn_label_encoder_v2.npy")
OLD = {"OLD orig (Jul, stgcn_sequence_model)": ("stgcn_sequence_model.pth", "stgcn_label_encoder.npy"),
       "OLD v2 (stgcn_sequence_model_v2)": ("stgcn_sequence_model_v2.pth", "stgcn_label_encoder_v2.npy"),
       "OLD LIVE (stgcn_transitions_v1)": ("stgcn_transitions_v1.pth", "stgcn_transitions_v1_encoder.npy")}
for name, (p, e) in OLD.items():
    try:
        MODELS[name] = load_model(hf_hub_download(LIVE, p, token=HF), hf_hub_download(LIVE, e, token=HF)); print(name, "classes:", MODELS[name][1], flush=True)
    except Exception as ex: print("could not load", name, ex, flush=True)

def probs(model_cls, X):
    m, cls = model_cls
    Xn = tr.normalize_coordinate_sequence(X.astype(np.float32)); t = tr.SequenceDataset(Xn, np.zeros(len(Xn), dtype=np.int64)).X
    out = []
    with torch.no_grad():
        for b in range(0, len(t), 256): out.append(torch.softmax(m(t[b:b + 256].to(DEV)), 1).cpu().numpy())
    return np.concatenate(out), cls
def to_target(label):
    l = str(label)
    if l.startswith("hold:"): l = l[5:]
    return l if l in TARGETS else U
THR = lambda disp: 0.55 if disp == "child_pose" else 0.70
def decide(P, cls):
    """returns (argmax label mapped to the 8-pose vocabulary, 'answered' mask = what production would let through)"""
    idx = P.argmax(1); conf = P.max(1); lab = np.array([to_target(cls[i]) for i in idx])
    ans = np.array([(lab[j] != U) and conf[j] >= THR(lab[j]) for j in range(len(lab))])
    return lab, ans, conf

# ---------------------------------------------------------------- windows
df = pd.read_csv(f"{ROOT}/vol/csv/cue/cue_full.csv", usecols=["video_id", "frame_num", "imperfect_pose_label"])
COORD, LAB = {}, {}
for vid in df.video_id.unique():
    lp = f"{ROOT}/vol/landmarks/landmarks_{vid}.npy" if vid in OLD_IDS else f"{ROOT}/vol/landmarks_new/landmarks_{vid}.npy"
    lm = np.load(lp); COORD[vid] = lm[:, :, :3].reshape(len(lm), -1).astype(np.float32)
    lab = df[df.video_id == vid].sort_values("frame_num")["imperfect_pose_label"].values
    LAB[vid] = np.array([l if (l in TARGETS or l in (U, "__ignore__")) else U for l in lab], dtype=object); assert len(lab) == len(COORD[vid])
print("videos", len(COORD), "frames", sum(len(v) for v in COORD.values()), flush=True)
def build_meta(step):
    meta = []
    for vid in COORD:
        span = SEQ * step; lab = LAB[vid]
        for st in range(0, len(COORD[vid]) - span + 1, STRIDE * step):
            w = lab[st:st + span:step]; ign = w == "__ignore__"
            if ign.mean() > 0.15: continue
            meta.append((vid, st, gsf.label_window(w[~ign])))
    Y = np.array([m[2] for m in meta]); V = np.array([m[0] for m in meta]); pos = Y != U
    rng = np.random.default_rng(7); tr_idx = np.where(~pos)[0]
    keep = rng.choice(tr_idx, min(len(tr_idx), int(1.5 * pos.sum())), replace=False)
    sel = np.sort(np.concatenate([np.where(pos)[0], keep]))
    return [meta[i] for i in sel], Y[sel], V[sel]
def fold_assign(Y, V, nfold=3):
    vids = sorted(set(V.tolist())); secs = {v: collections.Counter(Y[(V == v) & (Y != U)].tolist()) for v in vids}
    load = [collections.Counter() for _ in range(nfold)]; fo = {}
    for v in sorted(vids, key=lambda v: -sum(secs[v].values())):
        b = min(range(nfold), key=lambda f: sum((load[f][c] + n) ** 2 for c, n in secs[v].items()) + 0.001 * sum(load[f].values())); fo[v] = b; load[b].update(secs[v])
    return fo
def materialise(meta, idx, step): return np.stack([COORD[meta[i][0]][meta[i][1]:meta[i][1] + SEQ * step:step] for i in idx]).astype(np.float32)

meta1, Y1, V1 = build_meta(1); FO = fold_assign(Y1, V1); F1 = np.array([FO[v] for v in V1])
print("fold_of_video", FO, flush=True)
# sanity: rebuilt test windows must equal the ones the models were scored on
for f in range(3):
    saved = np.load(f"{ROOT}/vol/stgcn/cueT2_f{f}/test_feats.npy"); mine = materialise(meta1, np.where(F1 == f)[0], 1)
    ok = saved.shape == mine.shape and np.allclose(saved, mine); print(f"fold {f}: rebuilt test windows == saved test windows: {ok} {saved.shape} {mine.shape}", flush=True); assert ok

# ---------------------------------------------------------------- scoring
def score(P, T, min_n=20, bar=0.70):
    rows = {}
    for c in sorted(set(T.tolist())):
        t, pm = T == c, P == c; n = int(t.sum()); rec = float((P[t] == c).mean()); prec = float((T[pm] == c).mean()) if pm.sum() else 0.0
        rows[c] = {"n": n, "recall": round(rec, 3), "precision": round(prec, 3), "pass": bool(n >= min_n and rec >= bar and prec >= bar and c != U)}
    poses = [c for c in rows if c != U and rows[c]["n"] >= min_n]; ok = [c for c in poses if rows[c]["pass"]]
    return {"overall": round(float((P == T).mean()), 3), "n_pass": len(ok), "pass": ok, "macro_recall": round(float(np.mean([rows[c]["recall"] for c in poses])), 3) if poses else None,
            "macro_precision": round(float(np.mean([rows[c]["precision"] for c in poses])), 3) if poses else None, "per_pose": rows}
def gated(P, T, ans):
    """production view: only 'answered' windows reach the UI. coverage = share of true-pose windows answered; precision = of answered, share correct."""
    pose = T != U; cov = float(ans[pose].mean()) if pose.any() else None
    prec_ans = float((P[ans] == T[ans]).mean()) if ans.any() else None
    return {"coverage_of_true_pose_windows": None if cov is None else round(cov, 3), "precision_of_answers": None if prec_ans is None else round(prec_ans, 3), "answers": int(ans.sum())}
def run_all(step, subset_new_only=False):
    meta, Y, V = (meta1, Y1, V1) if step == 1 else build_meta(step)
    F = np.array([FO[v] for v in V]); newmask = np.array([v not in OLD_IDS for v in V])
    res = {}; allP = {k: np.empty(len(Y), dtype=object) for k in MODELS if k.startswith("OLD")}; allP["NEW (held-out fold models)"] = np.empty(len(Y), dtype=object)
    allA = {k: np.zeros(len(Y), bool) for k in allP}
    for f in range(3):
        idx = np.where(F == f)[0]
        if len(idx) == 0: continue
        X = materialise(meta, idx, step)
        P, cls = probs(MODELS[f"NEW f{f}"], X); lab, ans, _ = decide(P, cls); allP["NEW (held-out fold models)"][idx] = lab; allA["NEW (held-out fold models)"][idx] = ans
        for k in [k for k in MODELS if k.startswith("OLD")]:
            P, cls = probs(MODELS[k], X); lab, ans, _ = decide(P, cls); allP[k][idx] = lab; allA[k][idx] = ans
    for subset, mask in (("new_videos_only(clean for old)", newmask), ("all_24_videos(old may be in-sample)", np.ones(len(Y), bool))):
        res[subset] = {}
        for k in allP:
            res[subset][k] = {**{kk: vv for kk, vv in score(allP[k][mask].astype(str), Y[mask]).items()}, "production_gate": gated(allP[k][mask].astype(str), Y[mask], allA[k][mask]), "n_windows": int(mask.sum())}
    return res

# ---------------------------------------------------------------- 1) comparison at training scale
t0 = time.time(); cmp1 = run_all(1); print("comparison done", round(time.time() - t0), "s", flush=True)
for subset, d in cmp1.items():
    print(f"\n=== {subset}  (argmax, mapped to the 8-pose vocabulary; pass = n>=20 & recall>=.70 & precision>=.70)", flush=True)
    for k, r in d.items():
        print(f"{k:<42} n={r['n_windows']:<5} overall {r['overall']:.3f} | macro recall {r['macro_recall']} prec {r['macro_precision']} | pass {r['n_pass']} {r['pass']} | prod-gate {r['production_gate']}", flush=True)
        if "NEW" in k or "LIVE" in k:
            print("     " + " ".join(f"{c.split('_')[0][:7]}:{x['recall']:.2f}/{x['precision']:.2f}(n{x['n']})" for c, x in r['per_pose'].items() if c != U), flush=True)

# ---------------------------------------------------------------- 2) frame-rate sweep
rate = {}
for k in (1, 2, 4, 8, 12, 25):
    t0 = time.time(); r = run_all(k)["new_videos_only(clean for old)"]
    rate[k] = {m: {kk: r[m][kk] for kk in ("overall", "n_pass", "pass", "macro_recall", "macro_precision", "n_windows")} for m in r}
    print(f"\n[frame step k={k}: each window spans {SEQ * k} frames ~ {SEQ * k / 25:.1f} s at 25 fps; equals ~{25 / k:.1f} fps into the buffer] ({round(time.time() - t0)} s)", flush=True)
    for m, v in rate[k].items(): print(f"   {m:<42} overall {v['overall']:.3f} | macro recall {v['macro_recall']} | pass {v['n_pass']} {v['pass']} | n={v['n_windows']}", flush=True)

# ---------------------------------------------------------------- 3) parity: backend model code vs trainer code, and the DEPLOYED endpoint vs the same weights run locally
BACKEND_SEQ = r'''import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MediaPipe skeletal anatomical connections
MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

def get_normalized_adjacency():
    """
    Builds the normalized skeletal adjacency matrix A_norm for the 33 MediaPipe joints.
    A_norm = D^(-1/2) * (A + I) * D^(-1/2)
    """
    num_nodes = 33
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Add skeletal edges
    for i, j in MEDIAPIPE_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
        
    # Add self loops
    A_loop = A + np.eye(num_nodes, dtype=np.float32)
    
    # Calculate degree matrix D
    deg = np.sum(A_loop, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    
    # Symmetric normalization
    A_norm = D_inv_sqrt.dot(A_loop).dot(D_inv_sqrt)
    return torch.tensor(A_norm, dtype=torch.float32)


class SpatialGraphConv(nn.Module):
    """
    True Spatial Graph Convolutional Layer.
    Aggregates joint features along the normalized skeletal adjacency matrix A.
    """
    def __init__(self, in_channels, out_channels, A_norm):
        super(SpatialGraphConv, self).__init__()
        self.register_buffer('A', A_norm)  # Keep adjacency matrix on the correct device
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, num_nodes, seq_len] -> [N, C, 33, T]
        # 1. Project features: W * X
        x = self.conv(x)  # [N, C_out, 33, T]
        
        # 2. Graph aggregation: A * (W * X)
        # We multiply adjacency matrix A (shape [33, 33]) across the node dimension (dim=2)
        out = torch.einsum('vw,ncwt->ncvt', self.A, x)
        return out


class TemporalConv(nn.Module):
    """
    Temporal Convolutional Layer.
    Convolves features along the temporal dimension for each node independently.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.3):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, num_nodes, seq_len] -> [N, C, 33, T]
        # Swap node and temporal dimensions to convolve over time: [N, C, T, 33]
        x = x.transpose(2, 3)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        # Swap back: [N, C, 33, T]
        return x.transpose(2, 3)


class STGCNBlock(nn.Module):
    """
    A single Spatio-Temporal Graph Convolutional Block containing:
    1. Spatial Graph Convolution (GCN)
    2. Temporal Graph Convolution (TCN)
    """
    def __init__(self, in_channels, out_channels, A_norm, stride=1, dropout=0.3):
        super(STGCNBlock, self).__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A_norm)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, stride=stride, dropout=dropout)
        
        # Residual skip connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
            
    def forward(self, x):
        # x shape: [N, C_in, 33, T]
        res = self.residual(x)
        x = self.gcn(x)
        x = F.gelu(x)
        x = self.tcn(x)
        return F.gelu(x + res)


# True ST-GCN Sequence Classifier (maintaining class name YogaSequenceLSTM for backend loading compatibility)
class YogaSequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(YogaSequenceLSTM, self).__init__()
        A_norm = get_normalized_adjacency()
        
        # Input channels: 3 (x, y, z joint coordinates)
        self.block1 = STGCNBlock(3, 64, A_norm, stride=1, dropout=0.2)
        self.block2 = STGCNBlock(64, 128, A_norm, stride=1, dropout=0.3)
        self.block3 = STGCNBlock(128, 256, A_norm, stride=1, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_dim] -> [batch_size, 60, 99]
        # Reshape to [batch_size, seq_len, 33, 3] and permute to [batch_size, channels=3, nodes=33, seq_len=60]
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 60, 33, 3).permute(0, 3, 2, 1) # [batch_size, 3, 33, 60]
        
        # ST-GCN convolutions
        out = self.block1(x_reshaped)  # [batch_size, 64, 33, 60]
        out = self.block2(out)         # [batch_size, 128, 33, 60]
        out = self.block3(out)         # [batch_size, 256, 33, 60]
        
        # Global Pooling over both Spatial nodes (33) and Temporal frames (60)
        # out shape: [batch_size, 256]
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        
        return self.fc(out)
'''
BACKEND_NORM = r'''def normalize_coordinate_sequence(coords: np.ndarray) -> np.ndarray:
    """
    Translates joints to be pelvis-centered (midpoint of left and right hips)
    and scales by hip-width to ensure translation and scale invariance.
    coords shape: [60, 99]
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
    
    return coords_normalized.reshape(60, 99)'''
par = {}
try:
    ns = {}; exec(BACKEND_SEQ, ns); exec("import numpy as np\n" + BACKEND_NORM, ns)
    X0 = np.load(f"{ROOT}/vol/stgcn/cueT2_f0/test_feats.npy")[:64]
    m, cls = MODELS["NEW f0"]; bm = ns["YogaSequenceLSTM"](99, 128, 2, len(cls)); bm.load_state_dict(m.state_dict()); bm = bm.to(DEV).eval()
    with torch.no_grad():
        a = bm(torch.from_numpy(np.stack([ns["normalize_coordinate_sequence"](x) for x in X0])).float().to(DEV)).cpu().numpy()
        b = m(tr.SequenceDataset(tr.normalize_coordinate_sequence(X0), np.zeros(len(X0), dtype=np.int64)).X.to(DEV)).cpu().numpy()
    par["backend_vs_trainer_max_abs_logit_diff"] = float(np.abs(a - b).max()); print("\nPARITY backend model+normalisation vs trainer: max |logit diff| =", par["backend_vs_trainer_max_abs_logit_diff"], flush=True)
except Exception as ex: par["backend_vs_trainer_error"] = repr(ex); print("parity error", ex, flush=True)
try:
    live_m = MODELS["OLD LIVE (stgcn_transitions_v1)"]; Xs = np.load(f"{ROOT}/vol/stgcn/cueT2_f1/test_feats.npy"); sel = np.random.default_rng(3).choice(len(Xs), 40, replace=False); Xs = Xs[sel]
    P, cls = probs(live_m, Xs); agree, rows = 0, []
    for w, p in zip(Xs, P):
        r = None
        for _ in range(6):
            try:
                r = requests.post("https://arko007-yoga-pose.hf.space/api/analyse_sequence", json={"coordinates": w.tolist()}, timeout=90)
                if r.status_code == 200: break
                time.sleep(2)
            except Exception: time.sleep(2)
        if r is None or r.status_code != 200: continue
        j = r.json(); raw = cls[int(p.argmax())]; disp = raw[5:] if raw.startswith("hold:") else ("transition/unknown" if raw.startswith("transition:") or raw == "unrecognized" else raw)
        same = (j["sequence_pose"] == disp) and abs(j["confidence"] - float(p.max())) < 1e-3; agree += same; rows.append((disp, j["sequence_pose"], round(float(p.max()), 3), round(j["confidence"], 3)))
    par["endpoint_vs_local_agree"] = f"{agree}/{len(rows)}"; par["endpoint_examples"] = rows[:8]; print("PARITY deployed /analyse_sequence vs local run of stgcn_transitions_v1: agree", par["endpoint_vs_local_agree"], rows[:5], flush=True)
except Exception as ex: par["endpoint_error"] = repr(ex); print("endpoint parity error", ex, flush=True)

out = {"comparison_training_scale": cmp1, "frame_rate_sweep": {str(k): v for k, v in rate.items()}, "parity": par, "fold_of_video": FO}
json.dump(out, open(f"{ROOT}/cmp.json", "w"), indent=1, default=str)
api.upload_file(path_or_fileobj=f"{ROOT}/cmp.json", path_in_repo="evals_stgcn/stgcn_old_vs_new_compare.json", repo_id=MD, repo_type="model"); print("uploaded", flush=True)
