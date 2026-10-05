"""LIVE-PIPELINE REPLAY: real clips -> real MediaPipe (Tasks API, full model) -> the exact requests the web client sends -> the LIVE production API.
Clips come from the 12 NEW videos: the live pose-naming MLP (v4) and the live sequence model never trained on them. The gate (cascade) DID train on these
videos, which can only flatter recall -- stated in the report. Orientation is computed from raw (not occlusion-fused) landmarks: a small deviation from the browser."""
import glob, json, os, subprocess, sys, time, collections
from concurrent.futures import ThreadPoolExecutor
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe", "opencv-python-headless", "requests"], check=False)
import cv2, numpy as np, requests, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"
snapshot_download("Arko007/Yoga-1M", repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["code/backend/app/__init__.py", "code/backend/app/utils/*.py", "vol/cue/*.npz"])
sys.path.insert(0, f"{ROOT}/code/backend")
from app.utils.geometry import extract_angles_from_landmarks, compute_orientation
API = "https://arko007-yoga-pose.hf.space/api"
VID = "https://huggingface.co/datasets/Arko007/yoga-dataset-raw/resolve/main/new_videos_2026-10/{}.mp4"
TASK = f"{ROOT}/pose_landmarker_full.task"
open(TASK, "wb").write(requests.get("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task", timeout=120).content)
CLIPS = [("mountain_pose", "149Iac5fmoE", 433, 14), ("corpse", "149Iac5fmoE", 769, 25), ("mountain_pose", "v7AYKMP6rOE", 925, 20),
         ("seated_easy_pose", "EvMTrP8eRvM", 44, 17), ("seated_easy_pose", "4K2xTVRDJgA", 566, 17), ("child_pose", "O2EY79Ys_qg", 545, 49),
         ("tree_pose", "JHjV-wFTwSw", 1304, 19), ("lunge_pose", "JHjV-wFTwSw", 1829, 13)]
# moving stretches (no pose held): longest runs of UNSTABLE frames in the new videos, from the cue files' own 'stable' flag
NEW = "v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk hHhxKkskHDg EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow".split()
def dur(v):
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download("Arko007/yoga-dataset-raw", f"new_videos_2026-10/{v}.info.json", repo_type="dataset", token=HF)))["duration"]
    except Exception: return None
moving = []
for v in NEW:
    z = np.load(f"{ROOT}/vol/cue/{v}.npz", allow_pickle=True); st = z["stable"]; d = dur(v)
    if d is None: continue
    spf = d / len(st); i = 0
    while i < len(st):
        if not st[i]:
            j = i
            while j < len(st) and not st[j]: j += 1
            if (j - i) * spf >= 10: moving.append(((j - i) * spf, v, i * spf))
            i = j
        else: i += 1
seen = set()
for secs, v, s0 in sorted(moving, reverse=True):
    if v in seen: continue
    seen.add(v); CLIPS.append(("MOVING", v, int(s0), int(min(secs, 40))))
    if len(seen) >= 3: break
print("clips:", CLIPS, flush=True)

def post(path, body, tries=6):
    for k in range(tries):
        try:
            r = requests.post(f"{API}/{path}", json=body, timeout=90)
            if r.status_code == 200: return r.json()
            if r.status_code == 429: time.sleep(2 + 2 * k)
        except Exception: time.sleep(1 + k)
    return None
def run_clip(expected, vid, start, length):
    clip = f"{ROOT}/clip_{vid}_{start}.mp4"
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-i", VID.format(vid), "-t", str(length), "-an", "-vf", "scale=640:-2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", clip], capture_output=True, text=True)
    if not os.path.exists(clip) or os.path.getsize(clip) < 1000: return {"clip": f"{vid}@{start}", "error": "ffmpeg: " + r.stderr[-200:]}
    cap = cv2.VideoCapture(clip); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lmk = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=TASK),
          running_mode=vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))
    frames, coords, hist, n = [], [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(n * 1000 / fps)); n += 1
        if not res.pose_landmarks: continue
        lm = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]], dtype=np.float64)
        wl = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], dtype=np.float64)
        ang = [float(a) for a in extract_angles_from_landmarks(lm[:, :3], True)]
        t = (n - 1) / fps; hist.append((t, ang)); hist = [h for h in hist if t - h[0] <= 1.5]       # same 1.5 s window as the web hook
        motion = None
        if len(hist) >= 2 and t - hist[0][0] > 0: motion = float(np.mean(np.abs(np.array(ang) - np.array(hist[0][1]))) / (t - hist[0][0]))
        coords.append(lm[:, :3].reshape(-1).tolist())
        frames.append({"t": t, "idx": len(coords) - 1, "body": {"angles": ang, "world_angles": [float(a) for a in extract_angles_from_landmarks(wl, False)],
                       "orientation": compute_orientation(lm[:, :3]), **({"motion": motion} if motion is not None else {})}})
    cap.release()
    if not frames: return {"clip": f"{vid}@{start}", "error": "no person detected"}
    step = max(1, int(round(fps / 6)))                                                              # ~6 frame calls per second
    fcalls = frames[::step]
    with ThreadPoolExecutor(4) as ex: fres = list(ex.map(lambda f: post("analyse_frame", f["body"]), fcalls))
    scalls = [f for f in frames if f["idx"] >= 59][::max(1, int(round(fps / 2)))]                    # sequence call twice a second once 60 frames exist
    with ThreadPoolExecutor(4) as ex: sres = list(ex.map(lambda f: post("analyse_sequence", {"coordinates": coords[f["idx"] - 59: f["idx"] + 1]}), scalls))
    # CORRECTNESS (the metric that matters): (a) score the live API gives the held pose, (b) the same frame with ONE joint deliberately broken by 45 deg
    # (left knee, index 6, and left elbow, index 0 -- the property test, run through the deployed endpoint). A working correctness head must score the broken copy LOWER.
    def broken(body, j):
        b = json.loads(json.dumps(body))
        for key in ("angles", "world_angles"):
            a = b[key]; a[j] = a[j] - 45.0 if a[j] > 135.0 else a[j] + 45.0
        return b
    cs = [(r or {}).get("correctness_score") for r in fres]
    cor = {}
    if expected != "MOVING":
        idxs = [i for i in range(0, len(fcalls), 3) if fres[i] and fres[i]["pose_id"] == expected and cs[i] is not None]   # frames the system named correctly
        cor["n_probe_frames"] = len(idxs)
        for name, j in (("knee", 6), ("elbow", 0)):
            with ThreadPoolExecutor(4) as ex: br = list(ex.map(lambda i: post("analyse_frame", broken(fcalls[i]["body"], j)), idxs))
            pairs = [(cs[i], b["correctness_score"]) for i, b in zip(idxs, br) if b and b.get("correctness_score") is not None]
            cor[name] = {"mean_held": round(float(np.mean([a for a, _ in pairs])), 3) if pairs else None,
                         "mean_broken": round(float(np.mean([b for _, b in pairs])), 3) if pairs else None,
                         "share_score_dropped": round(sum(b < a for a, b in pairs) / max(1, len(pairs)), 3), "n": len(pairs)}
        held = [cs[i] for i in range(len(fcalls)) if fres[i] and fres[i]["pose_id"] == expected and cs[i] is not None]
        cor["held_score_mean"] = round(float(np.mean(held)), 3) if held else None
        cor["held_score_p10_p90"] = [round(float(np.percentile(held, 10)), 3), round(float(np.percentile(held, 90)), 3)] if held else None
    fp = [r["pose_id"] if r else "ERR" for r in fres]; ms = [(r or {}).get("motion_state", "?") for r in fres]
    sp = [(r or {}).get("sequence_pose", "ERR") for r in sres]
    # What the web UI would show: once 60 frames exist the hook asks the sequence model first; when it is confident (requires_static_fallback == False)
    # the UI adopts the SEQUENCE pose, otherwise it uses the per-frame (MLP/cascade) answer.
    stimes = [f["t"] for f in scalls]
    def ui_pose(t, fpose):
        k = max([i for i, tt in enumerate(stimes) if tt <= t], default=None)
        if k is None or sres[k] is None: return fpose
        return fpose if sres[k]["requires_static_fallback"] else sres[k]["sequence_pose"]
    ui = [ui_pose(f["t"], p) for f, p in zip(fcalls, fp)]
    conf_calls = [r for r in sres if r and not r["requires_static_fallback"]]
    def frac(lst, val): return round(sum(x == val for x in lst) / max(1, len(lst)), 3)
    out = {"clip": f"{vid}@{start}+{length}s", "expected": expected, "fps": round(fps, 1), "frames_with_person": len(frames), "frame_calls": len(fp), "failed_frame_calls": fp.count("ERR"),
           "frame_pose_top": collections.Counter(fp).most_common(4), "motion_states": collections.Counter(ms).most_common(3),
           "seq_calls": len(sp), "seq_top": collections.Counter(sp).most_common(4),
           "seq_confident_share": round(len(conf_calls) / max(1, len(sres)), 3),
           "seq_confident_poses": collections.Counter(r["sequence_pose"] for r in conf_calls).most_common(3),
           "seq_conf_mean": round(float(np.mean([r["confidence"] for r in sres if r])) if any(sres) else 0.0, 3),
           "ui_top": collections.Counter(ui).most_common(3), "correctness": cor}
    if expected != "MOVING":
        out["ui_acc"] = frac(ui, expected); out["frame_acc"] = frac(fp, expected); out["frame_unknown"] = frac(fp, "transition/unknown"); out["seq_acc"] = frac(sp, expected)
        # what the app shows at its own 10 s cadence: majority label of each 10 s window
        wins = collections.defaultdict(list)
        for f, p in zip(fcalls, fp): wins[int(f["t"] // 10)].append(p)
        out["per_10s_window_majority"] = [collections.Counter(w).most_common(1)[0][0] for _, w in sorted(wins.items())]
    else:
        out["named_a_pose_share"] = round(1 - frac(fp, "transition/unknown"), 3); out["transitioning_share"] = frac(ms, "transitioning")
    return out
results = []
for c in CLIPS:
    t0 = time.time(); r = run_clip(*c); r["secs"] = round(time.time() - t0); results.append(r); print("RESULT", json.dumps(r, default=str), flush=True)
json.dump(results, open(f"{ROOT}/replay.json", "w"), indent=1, default=str)
api.upload_file(path_or_fileobj=f"{ROOT}/replay.json", path_in_repo="evals_compare/live_replay_v3.json", repo_id="Arko007/asanaai-conference-runs", repo_type="model")
