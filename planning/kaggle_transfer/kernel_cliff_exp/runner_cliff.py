"""Offline experiment: does CLIFF recover OCCLUDED joints better than the symmetric-mirror fallback (and than MediaPipe's own guess)?

Real yoga video frames -> MediaPipe on the clean frame = reference (only frames where every key joint is clearly visible) ->
a grey box hides one limb -> recover the hidden joints four ways and compare against the reference:
  M0 raw       MediaPipe run on the occluded image (its guess for hidden joints)
  M1 mirror    the deployed fallback (backend/app/services/occlusion.py) driven by M0 with the hidden joints marked invisible
  M2 CLIFF     CLIFF (ResNet-50) run on the occluded image, SMPL joints projected to the image with the predicted camera
  M3 CLIFF+al  M2 aligned to M0's VISIBLE joints (2-D similarity fit): the realistic 'fusion' use of CLIFF
Error = 2-D distance / torso length (hip-mid to shoulder-mid), plus joint-angle error for the affected angle.
CLIFF is inference-only here (never trained). Split by symmetric vs asymmetric poses: the mirror can only be right for symmetric ones.
"""
import glob, json, os, pickle, subprocess, sys, time, traceback, types
def sh(cmd, tail=15):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True); out = (r.stdout + r.stderr).strip().splitlines()
    print(f"$ {cmd}\n" + "\n".join(out[-tail:]), flush=True); return r.returncode
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token"); W = "/kaggle/working"; os.chdir(W)
sh("pip install -q smplx yacs trimesh opencv-python-headless mediapipe requests gdown scipy 2>&1 | tail -2")
import numpy as np, cv2, requests, torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"; MD = "Arko007/asanaai-conference-runs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("device", DEV, torch.__version__, flush=True)

# ---------------- CLIFF code, checkpoint, SMPL ----------------
sh("git clone --depth 1 --filter=blob:none --sparse https://github.com/huawei-noah/noah-research.git cl && cd cl && git sparse-checkout set CLIFF && ls CLIFF/models/cliff_res50 CLIFF/common && grep -rn 'torchgeometry\\|tgm\\.\\|pyrender' CLIFF --include=*.py | head -12", tail=40)
CL = f"{W}/cl/CLIFF"; sys.path.insert(0, CL)
for m in ("torchgeometry", "pyrender"):
    try: __import__(m)
    except Exception: sys.modules[m] = types.ModuleType(m); print("stubbed", m, flush=True)
import gdown
files = gdown.download_folder(url="https://drive.google.com/drive/folders/1EmSZwaDULhT9m1VvH7YOpCXwBWgYrgwP", skip_download=True, quiet=True, use_cookies=False)
print("drive files:", [f.path for f in files], flush=True)
ck = [f for f in files if "res50" in f.path][0]
os.makedirs(f"{W}/ckpt", exist_ok=True)
CKPT = gdown.download(id=ck.id, output=f"{W}/ckpt/res50.pt", quiet=True)
# smpl_mean_params.npz is not in the Drive folder. In CLIFF it only seeds three registered BUFFERS (init_pose/init_shape/init_cam) that the
# checkpoint's state_dict overwrites, so a neutral stand-in (identity 6-D rotations, zero shape, default camera) is enough to build the model.
MEAN = f"{W}/ckpt/smpl_mean_params.npz"
np.savez(MEAN, pose=np.tile(np.array([1, 0, 0, 1, 0, 0], dtype=np.float32), 24), shape=np.zeros(10, dtype=np.float32), cam=np.array([0.9, 0.0, 0.0], dtype=np.float32))
print("downloaded:", CKPT, os.path.getsize(CKPT) // 1e6, "MB", flush=True)

class _Ch:  # chumpy stand-in so the SMPL pickle loads without chumpy
    def __setstate__(self, s): self.__dict__.update(s)
for name in ("chumpy", "chumpy.ch", "chumpy.reordering", "chumpy.ch_ops", "chumpy.utils"):
    mod = types.ModuleType(name); mod.Ch = _Ch; sys.modules[name] = mod
smpl_pkl = hf_hub_download("Arko007/smpl-models", "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl", token=HF)
d = pickle.load(open(smpl_pkl, "rb"), encoding="latin1"); out = {}
for k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f"):
    v = d[k]
    if hasattr(v, "x"): v = v.x
    if hasattr(v, "toarray"): v = v.toarray()
    out[k] = np.array(v)
os.makedirs(f"{W}/smpl", exist_ok=True); pickle.dump(out, open(f"{W}/smpl/SMPL_NEUTRAL.pkl", "wb"), protocol=2)   # plain-numpy pickle: this smplx version reads pickles only
print("SMPL arrays:", {k: v.shape for k, v in out.items()}, flush=True)
import smplx
smpl = smplx.SMPL(model_path=f"{W}/smpl/SMPL_NEUTRAL.pkl", gender="neutral", batch_size=1).to(DEV)
from models.cliff_res50.cliff import CLIFF
from common.imutils import process_image
from common.utils import cam_crop2full, estimate_focal_length, strip_prefix_if_present
cliff = CLIFF(MEAN).to(DEV)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)["model"]; sd = strip_prefix_if_present(sd, prefix="module.")
print("buffers present in checkpoint:", [k for k in sd if k.startswith("init_")], flush=True)
print("load_state_dict:", cliff.load_state_dict(sd, strict=False), flush=True); cliff.eval()
print("CLIFF params (M):", round(sum(p.numel() for p in cliff.parameters()) / 1e6, 1), flush=True)

def run_cliff(img_bgr, bbox, device=DEV):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    norm_img, center, scale, _, _, _ = process_image(rgb, bbox)
    focal = estimate_focal_length(h, w)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    norm_img = t(norm_img)[None]; center = t(center)[None]; scale = t(scale).reshape(1); focal_t = t([focal]); ih = t([h]); iw = t([w])
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    bbox_info = torch.stack([cx - iw / 2., cy - ih / 2., b], dim=-1)
    bbox_info[:, :2] = bbox_info[:, :2] / focal_t.unsqueeze(-1) * 2.8
    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_t) / (0.06 * focal_t)
    with torch.no_grad():
        rotmat, betas, cam_crop = cliff.to(device)(norm_img, bbox_info)
        cam_full = cam_crop2full(cam_crop, center, scale, torch.stack((ih, iw), dim=-1), focal_t)
        o = smpl.to(device)(betas=betas, body_pose=rotmat[:, 1:], global_orient=rotmat[:, [0]], pose2rot=False, transl=cam_full)
    j3 = o.joints[0, :24].cpu().numpy()
    return np.stack([focal * j3[:, 0] / j3[:, 2] + w / 2.0, focal * j3[:, 1] / j3[:, 2] + h / 2.0], axis=1)  # 24 x 2 pixels

# ---------------- MediaPipe + frames ----------------
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
TASK = f"{W}/pose_landmarker_full.task"
open(TASK, "wb").write(requests.get("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", timeout=120).content)
lmk = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=TASK), running_mode=vision.RunningMode.IMAGE, num_poses=1))
def mp_landmarks(img_bgr):
    r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks: return None
    h, w = img_bgr.shape[:2]
    return np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]], dtype=np.float64), (w, h)
VID = "https://huggingface.co/datasets/Arko007/yoga-dataset-raw/resolve/main/new_videos_2026-10/{}.mp4"
CLIPS = [("mountain_pose", "149Iac5fmoE", 433, 14), ("corpse", "149Iac5fmoE", 769, 25), ("mountain_pose", "v7AYKMP6rOE", 925, 20), ("seated_easy_pose", "EvMTrP8eRvM", 44, 17),
         ("seated_easy_pose", "4K2xTVRDJgA", 566, 17), ("child_pose", "O2EY79Ys_qg", 545, 49), ("tree_pose", "JHjV-wFTwSw", 1304, 19), ("lunge_pose", "JHjV-wFTwSw", 1829, 13),
         ("downward_dog", "O2EY79Ys_qg", 1000, 20), ("triangle", "v7AYKMP6rOE", 1500, 20)]
frames = []
os.makedirs(f"{W}/fr", exist_ok=True)
for pose, vid, s0, ln in CLIPS:
    pre = f"{W}/fr/{vid}_{s0}"
    sh(f"ffmpeg -loglevel error -y -ss {s0} -i {VID.format(vid)} -t {ln} -an -vf 'fps=0.5,scale=720:-2' {pre}_%02d.jpg", tail=2)
    for p in sorted(glob.glob(pre + "_*.jpg"))[:8]: frames.append((pose, vid, p))
print("frames extracted:", len(frames), flush=True)

KEY = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MP2SMPL = {23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8, 11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21}
SCEN = {"left_leg": ([25, 27, 29, 31], [25, 27], ("knee_l", (23, 25, 27))), "right_arm": ([14, 16, 18, 20, 22], [14, 16], ("elbow_r", (12, 14, 16))),
        "both_feet": ([27, 28, 29, 30, 31, 32], [27, 28], ("knee_l", (23, 25, 27)))}
PARTNER = {13: 14, 14: 13, 15: 16, 16: 15, 25: 26, 26: 25, 27: 28, 28: 27}
sys.path.insert(0, "/kaggle/working/be"); snapshot_download("Arko007/Yoga-1M", repo_type="dataset", local_dir="/kaggle/working/be", token=HF, allow_patterns=["code/backend/app/__init__.py", "code/backend/app/services/__init__.py", "code/backend/app/services/occlusion.py"])
sys.path.insert(0, "/kaggle/working/be/code/backend")
from app.services.occlusion import fuse_and_recover_occlusions

def angle(a, b, c):
    ba, bc = a - b, c - b; n = np.linalg.norm(ba) * np.linalg.norm(bc)
    return 180.0 if n == 0 else float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1, 1))))
def similarity(src, dst):  # 2-D Umeyama: scale, rotation, translation mapping src -> dst
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src)); D = np.diag([1, np.sign(np.linalg.det(U) * np.linalg.det(Vt))]); R = U @ D @ Vt
    sc = (S * np.diag(D)).sum() / max(1e-9, s0.var(0).sum()); return lambda p: sc * (p - ms) @ R.T + md

rows = []; t_cliff_gpu = []; t_cliff_cpu = []
for pose, vid, path in frames:
    try:
        img = cv2.imread(path); got = mp_landmarks(img)
        if got is None: continue
        L, (w, h) = got
        if not all(L[i, 3] > 0.6 for i in KEY): continue
        gt = L[:, :2] * [w, h]; torso = np.linalg.norm(gt[[11, 12]].mean(0) - gt[[23, 24]].mean(0))
        if torso < 40: continue
        asym = float(np.mean([abs(gt[a, 1] - gt[b, 1]) for a, b in ((13, 14), (15, 16), (25, 26), (27, 28))]) / torso)
        x0, y0, x1, y1 = gt[:, 0].min(), gt[:, 1].min(), gt[:, 0].max(), gt[:, 1].max(); pad = 0.12 * max(x1 - x0, y1 - y0)
        bbox = [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]
        for scen, (boxjoints, hidden, (aname, (ia, ib, ic))) in SCEN.items():
            pts = gt[boxjoints]; m = 0.20 * torso
            bx0, by0, bx1, by1 = [int(v) for v in (pts[:, 0].min() - m, pts[:, 1].min() - m, pts[:, 0].max() + m, pts[:, 1].max() + m)]
            occ = img.copy(); cv2.rectangle(occ, (max(0, bx0), max(0, by0)), (min(w - 1, bx1), min(h - 1, by1)), (128, 128, 128), -1)
            g2 = mp_landmarks(occ)
            if g2 is None: continue
            L2 = g2[0]
            res = {"raw": L2[:, :2] * [w, h]}
            fz = L2.copy(); fz[hidden, 3] = 0.1
            fused, rec, _ = fuse_and_recover_occlusions(fz.tolist()); res["mirror"] = np.array(fused)[:, :2] * [w, h]
            t0 = time.time(); c2 = run_cliff(occ, bbox); t_cliff_gpu.append(time.time() - t0)
            if len(t_cliff_cpu) < 6:
                torch.set_num_threads(2); t0 = time.time(); run_cliff(occ, bbox, "cpu"); t_cliff_cpu.append(time.time() - t0)
                cliff.to(DEV); smpl.to(DEV)
            full = res["raw"].copy(); cl = res["raw"].copy()
            for mpi, si in MP2SMPL.items(): cl[mpi] = c2[si]
            res["cliff"] = cl
            anchors = [j for j in MP2SMPL if j not in hidden and L2[j, 3] > 0.5]
            if len(anchors) >= 4:
                T = similarity(np.array([c2[MP2SMPL[j]] for j in anchors]), res["raw"][anchors]); al = res["raw"].copy()
                for j in hidden: al[j] = T(c2[MP2SMPL[j]][None])[0]
                res["cliff_aligned"] = al
            for meth, P in res.items():
                nme = float(np.mean([np.linalg.norm(P[j] - gt[j]) for j in hidden]) / torso)
                ang = abs(angle(P[ia], P[ib], P[ic]) - angle(gt[ia], gt[ib], gt[ic]))
                rows.append({"pose": pose, "video": vid, "scenario": scen, "method": meth, "nme": nme, "angle_err": ang, "asym": asym, "mirror_recovered": len(rec)})
    except Exception as e:
        print("frame failed:", path, repr(e)[:200], flush=True); traceback.print_exc()
print("rows:", len(rows), "frames used:", len({(r['video'], r['pose'], r['asym']) for r in rows}), flush=True)

def summ(sel):
    out = {}
    for m in ("raw", "mirror", "cliff", "cliff_aligned"):
        v = [r for r in sel if r["method"] == m]
        if v: out[m] = {"n": len(v), "median_nme": round(float(np.median([r["nme"] for r in v])), 3), "mean_nme": round(float(np.mean([r["nme"] for r in v])), 3),
                        "median_angle_err": round(float(np.median([r["angle_err"] for r in v])), 1), "mean_angle_err": round(float(np.mean([r["angle_err"] for r in v])), 1),
                        "within_0.15torso": round(float(np.mean([r["nme"] < 0.15 for r in v])), 3)}
    return out
report = {"all": summ(rows), "symmetric(asym<0.2)": summ([r for r in rows if r["asym"] < 0.2]), "asymmetric(asym>=0.2)": summ([r for r in rows if r["asym"] >= 0.2])}
for sc in SCEN: report["scenario:" + sc] = summ([r for r in rows if r["scenario"] == sc])
for pz in sorted({r["pose"] for r in rows}): report["pose:" + pz] = summ([r for r in rows if r["pose"] == pz])
report["timing"] = {"cliff_res50_gpu_ms_median": round(1000 * float(np.median(t_cliff_gpu)), 1) if t_cliff_gpu else None, "cliff_res50_cpu_2threads_ms_median": round(1000 * float(np.median(t_cliff_cpu[1:] or t_cliff_cpu)), 0) if t_cliff_cpu else None}
report["n_frames_extracted"] = len(frames)
print(json.dumps(report, indent=1), flush=True)
json.dump({"report": report, "rows": rows}, open(f"{W}/cliff_exp.json", "w"))
api.upload_file(path_or_fileobj=f"{W}/cliff_exp.json", path_in_repo="evals_compare/cliff_occlusion_experiment.json", repo_id=MD, repo_type="model"); print("uploaded", flush=True)
