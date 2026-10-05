"""Which angle recipe does the live 654k CSV actually match? 4 variants x 3 videos,
and are the volume's landmark files byte-identical to the local ones?"""
import hashlib, json, os, sys
import modal

SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
app = modal.App("asanaai-diag2")
image = (modal.Image.debian_slim(python_version="3.11").pip_install("numpy<2", "pandas")
         .add_local_file(f"{SRC}/extract_features_safe.py", "/opt/orig/extract_features_safe.py", copy=True)
         .add_local_file("/home/anamitra/yoga_posture_workspace/modal/local_lm_sha.json", "/opt/local_lm_sha.json", copy=True))
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
ANG = ["elbow_l","elbow_r","shoulder_l","shoulder_r","hip_l","hip_r","knee_l","knee_r",
       "ankle_l","ankle_r","trunk_l","trunk_r","neck","hip_abduct_l","hip_abduct_r"]

def angles(lm, zero_z, interp, N, fe):
    import numpy as np
    c = fe.interpolate_occlusions(lm, 0.5) if interp else lm
    rows = []
    for f in range(min(N, len(c))):
        p = c[f, :, :3].copy()
        if zero_z: p[:, 2] = 0.0
        A = fe.calculate_angle_3d; sm = (p[11]+p[12])/2; hm = (p[23]+p[24])/2
        rows.append([A(p[11],p[13],p[15]),A(p[12],p[14],p[16]),A(p[23],p[11],p[13]),A(p[24],p[12],p[14]),
                     A(p[11],p[23],p[25]),A(p[12],p[24],p[26]),A(p[23],p[25],p[27]),A(p[24],p[26],p[28]),
                     A(p[25],p[27],p[29]),A(p[26],p[28],p[30]),A(p[11],p[23],p[24]),A(p[12],p[24],p[23]),
                     A(p[0],sm,hm),A(p[24],p[23],p[25]),A(p[23],p[24],p[26])])
    return np.array(rows)

@app.function(image=image, volumes={VOL: vol}, cpu=2, memory=12288, timeout=3000)
def diag2():
    import numpy as np, pandas as pd
    sys.path.insert(0, "/opt/orig"); import extract_features_safe as fe
    vol.reload()
    loc = json.load(open("/opt/local_lm_sha.json")); res = {"sha_same": {}, "variants": {}}
    for y, h in loc.items():
        hh = hashlib.sha256(open(f"{VOL}/landmarks/landmarks_{y}.npy", "rb").read()).hexdigest()
        res["sha_same"][y] = (hh == h)
    print("volume==local landmark files:", res["sha_same"], flush=True)
    csv = [f for f in os.listdir(f"{VOL}/csv") if f.endswith(".csv")][0]
    df = pd.read_csv(f"{VOL}/csv/{csv}", usecols=["video_id","frame_num"]+ANG)
    N = 12000
    for y in ["oUgpXY7QhpQ", "P8uHMMmWMHQ", "Eml2xnoLpYE"]:
        lm = np.load(f"{VOL}/landmarks/landmarks_{y}.npy")
        v = df[df.video_id == y].sort_values("frame_num")[ANG].values[:N]
        for name, (z, it) in {"interp+zeroZ": (True, True), "interp+rawZ": (False, True),
                              "nointerp+zeroZ": (True, False), "nointerp+rawZ": (False, False)}.items():
            a = angles(lm, z, it, N, fe)[:len(v)]
            d = np.abs(a - v)
            res["variants"][f"{y}|{name}"] = {"median": round(float(np.median(d)), 6), "p95": round(float(np.percentile(d, 95)), 4)}
            print(y, name, res["variants"][f"{y}|{name}"], flush=True)
    json.dump(res, open(f"{VOL}/runs/diag2.json", "w"), indent=1); vol.commit()

@app.local_entrypoint()
def main():
    diag2.remote()
