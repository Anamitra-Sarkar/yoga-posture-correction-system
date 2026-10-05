"""Find the best live-test clips: longest steady cue-verified holds per pose, and long moving stretches, with YouTube timestamps."""
import glob, json, os
def cred(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hits, name
    return open(hits[0]).read().strip()
HF = cred("hf_token")
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
api = HfApi(token=HF); assert api.whoami()["name"] == "Arko007"
ROOT = "/kaggle/working"
snapshot_download("Arko007/Yoga-1M", repo_type="dataset", local_dir=ROOT, token=HF, allow_patterns=["vol/cue/*.npz"])
OLD = "HmZFwoUU3WQ SZU7Sbgu57o oUgpXY7QhpQ 7ciS93shMNQ P8uHMMmWMHQ 4ORRiN2_aVI RQMtwbhXD7A 8ibxmzJziHU Eml2xnoLpYE QiebZSlTw_U L-z1HLkS_-Y s-1vMbAgYWU".split()
def info(v):
    cand = [f"Yoga_Dataset_Raw/{v}.info.json"] if v in OLD else [f"new_videos_2026-10/{v}.info.json"]
    for c in cand:
        try:
            j = json.load(open(hf_hub_download("Arko007/yoga-dataset-raw", c, repo_type="dataset", token=HF)))
            return j.get("duration"), j.get("title")
        except Exception as e:
            pass
    return None, None
TARGETS = ["downward_dog", "warrior_2", "tree_pose", "triangle", "seated_easy_pose", "child_pose", "corpse", "plank",
           "mountain_pose", "cobra_pose", "lunge_pose", "standing_forward_fold"]
runs = {p: [] for p in TARGETS}; moving = []
meta = {}
for f in sorted(glob.glob(f"{ROOT}/vol/cue/*.npz")):
    v = os.path.basename(f)[:-4]
    lab = np.load(f, allow_pickle=True)["labels"].astype(str); n = len(lab)
    dur, title = info(v); meta[v] = (dur, title)
    if not dur: continue
    spf = dur / n
    i = 0
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]: j += 1
        L = lab[i]; secs = (j - i) * spf
        if L in runs and secs >= 5: runs[L].append((secs, v, i * spf))
        if L == "transition/unknown" and secs >= 8: moving.append((secs, v, i * spf))
        i = j
def mmss(t): t = int(t); return f"{t // 60}:{t % 60:02d}"
out = ["VIDEOS: " + json.dumps({v: (round(d) if d else None, (t or "")[:60]) for v, (d, t) in meta.items()}, ensure_ascii=False)]
for p in TARGETS:
    out.append(f"\n## {p}")
    seen = {}
    for secs, v, st in sorted(runs[p], reverse=True):
        if seen.get(v, 0) >= 1: continue          # best segment per video, so the list spans different people/rooms
        seen[v] = 1
        out.append(f"CLIP|{p}|{v}|{round(st)}|{round(secs)}|https://youtu.be/{v}?t={int(st)}|starts {mmss(st)} for ~{int(secs)}s")
        if len(seen) >= 3: break
    if not seen: out.append("CLIP|" + p + "|none>=5s")
out.append("\n## moving (no pose held) -- the app should stay quiet / say 'transitioning'")
seen = {}
for secs, v, st in sorted(moving, reverse=True):
    if seen.get(v, 0) >= 1: continue
    seen[v] = 1; out.append(f"MOVE|{v}|{round(st)}|{round(secs)}|https://youtu.be/{v}?t={int(st)}|starts {mmss(st)} for ~{int(secs)}s")
    if len(seen) >= 4: break
print("\n".join(out))
open(f"{ROOT}/clips.txt", "w").write("\n".join(out))
