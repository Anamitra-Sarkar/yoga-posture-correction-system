"""Decisive fair comparison: v2 vs v3_aug on REAL-WORLD PHOTOS.

The video-level "holdout" comparison is invalid for this pair, because v2 was
trained on all 12 videos (including the held-out two) while v3_aug genuinely
excluded them. Real-world photos are unseen by BOTH models, so this is the only
apples-to-apples measurement.
"""
import modal

app = modal.App("asanaai-eval-photos")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "huggingface_hub")
    .add_local_dir("/home/anamitra/yoga_posture_workspace/backend/app/models", remote_path="/models_src")
)
mlp_vol = modal.Volume.from_name("asanaai-mlp-v3")

# MediaPipe indices, mirroring backend/app/utils/geometry.py exactly
NOSE = 0; SH_L, SH_R = 11, 12; EL_L, EL_R = 13, 14; WR_L, WR_R = 15, 16
HP_L, HP_R = 23, 24; KN_L, KN_R = 25, 26; AN_L, AN_R = 27, 28; HE_L, HE_R = 29, 30


@app.function(image=image, volumes={"/mlp": mlp_vol}, timeout=3600,
              secrets=[modal.Secret.from_name("arko007-hf-token")])
def run():
    import sys, os, json
    import numpy as np, torch
    sys.path.insert(0, "/models_src")
    from mlp import Yoga3HeadMLP
    from huggingface_hub import hf_hub_download

    def ang(a, b, c):
        ba, bc = a - b, c - b
        nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nb == 0 or nc == 0:
            return 180.0
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (nb * nc), -1.0, 1.0))))

    def angles_from_landmarks(pts):
        """Exactly the production path: zero z, then 15 vector angles."""
        p = np.array(pts, dtype=np.float64)[:, :3].copy()
        p[:, 2] = 0.0
        sm = (p[SH_L] + p[SH_R]) / 2.0
        hm = (p[HP_L] + p[HP_R]) / 2.0
        return [
            ang(p[SH_L], p[EL_L], p[WR_L]), ang(p[SH_R], p[EL_R], p[WR_R]),
            ang(p[HP_L], p[SH_L], p[EL_L]), ang(p[HP_R], p[SH_R], p[EL_R]),
            ang(p[SH_L], p[HP_L], p[KN_L]), ang(p[SH_R], p[HP_R], p[KN_R]),
            ang(p[HP_L], p[KN_L], p[AN_L]), ang(p[HP_R], p[KN_R], p[AN_R]),
            ang(p[KN_L], p[AN_L], p[HE_L]), ang(p[KN_R], p[AN_R], p[HE_R]),
            ang(p[SH_L], p[HP_L], p[HP_R]), ang(p[SH_R], p[HP_R], p[HP_L]),
            ang(p[NOSE], sm, hm),
            ang(p[HP_R], p[HP_L], p[KN_L]), ang(p[HP_L], p[HP_R], p[KN_R]),
        ]

    tok = os.environ.get("HF_TOKEN_ARKO007")
    models = {}
    # v2 = currently live, from HF
    try:
        mp = hf_hub_download("Arko007/yoga-posture-models", "mlp_3head_model_v2.pth", token=tok)
        ep = hf_hub_download("Arko007/yoga-posture-models", "mlp_3head_pose_encoder.npy", token=tok)
        cls = list(np.load(ep, allow_pickle=True))
        m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls)); m.load_state_dict(torch.load(mp, map_location="cpu")); m.eval()
        models["v2_live"] = (m, cls)
    except Exception as e:
        print("v2 load failed:", e, flush=True)
    # v3_aug from the volume
    try:
        cls = list(np.load("/mlp/mlp_3head_pose_encoder_v3_aug.npy", allow_pickle=True))
        m = Yoga3HeadMLP(input_dim=15, num_poses=len(cls))
        m.load_state_dict(torch.load("/mlp/mlp_3head_model_v3_aug.pth", map_location="cpu")); m.eval()
        models["v3_aug"] = (m, cls)
    except Exception as e:
        print("v3_aug load failed:", e, flush=True)

    man = json.load(open("/mlp/testset/manifest.json"))
    out = {}
    for name, (m, cls) in models.items():
        yt, yp, per = [], [], {}
        for pose, items in man.items():
            hits = tot = 0
            for it in items:
                fp = f"/mlp/testset/{it['landmarks']}"
                if not os.path.exists(fp):
                    continue
                arr = np.load(fp, allow_pickle=True)
                a = arr.reshape(-1).astype(np.float32) if arr.size == 15 else np.array(
                    angles_from_landmarks(arr.reshape(-1, arr.shape[-1])), dtype=np.float32)
                if a.shape[0] != 15:
                    continue
                with torch.no_grad():
                    pl, _, _ = m(torch.tensor([a]))
                pred = cls[int(pl.argmax(1))]
                yt.append(pose); yp.append(pred)
                tot += 1; hits += int(pred == pose)
            if tot:
                per[pose] = {"n": tot, "acc": round(hits / tot, 4)}
        if yt:
            macro = sum(v["acc"] for v in per.values()) / len(per)
            out[name] = {"per_class": per, "macro": round(macro, 4),
                         "overall": round(sum(1 for a, b in zip(yt, yp) if a == b) / len(yt), 4),
                         "n_total": len(yt)}
            print(f"{name}: n={len(yt)} macro={out[name]['macro']} overall={out[name]['overall']}", flush=True)

    with open("/mlp/eval_photos_fair.json", "w") as f:
        json.dump(out, f, indent=1)
    mlp_vol.commit()
    return out


@app.local_entrypoint()
def main():
    r = run.remote()
    for name, res in (r or {}).items():
        print(f"\n=== {name} === n={res['n_total']} macro={res['macro']*100:.1f}% overall={res['overall']*100:.1f}%")
        for c, v in sorted(res["per_class"].items()):
            print(f"   {c:<24} n={v['n']:<3} {v['acc']*100:.0f}%")
