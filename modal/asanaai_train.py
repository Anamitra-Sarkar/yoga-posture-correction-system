"""Run the ORIGINAL AsanaAI trainers, unmodified, on Modal GPU.

WHY THE SCRIPTS ARE NOT EDITED
------------------------------
The user's instruction is to retrain "the old model's code ... as it was
goated". So the two trainers are copied in byte-for-byte and executed as
subprocesses. Everything this wrapper does is arrange the *environment* so
that the original code takes its intended paths:

  * `/home/anamitra/yoga_raw_dataset/` is created, because
      - line 3 of each script pip-installs torch==2.4.1 at runtime when
        `/home/anamitra` is absent, which would replace this image's torch
        mid-run;
      - `resolve_path(..., is_output=True)` returns the default path only if
        its directory exists, otherwise a bare filename in the cwd.
    Creating the directory makes both behave exactly as they did locally.

  * HF_TOKEN is deliberately set to an invalid value. The scripts upload
    every time val improves, with `path_in_repo=os.path.basename(
    MODEL_SAVE_PATH)` -- i.e. `mlp_3head_model_v2.pth` and
    `stgcn_sequence_model_v2.pth`, BOTH OF WHICH ALREADY EXIST on
    Arko007/yoga-posture-models. The standing instruction is that the
    artifacts uploaded two months ago must never be replaced or changed.
    With a bad token the upload raises inside the existing try/except, is
    caught, prints a warning, and training continues untouched. Checkpoints
    are collected from the volume afterwards and published under NEW names.
    (The `open(~/yoga_retrain/.hf_token)` fallback sits OUTSIDE that try, so
    simply leaving HF_TOKEN unset would crash the run instead.)

The GPU-capability guard in the local copies matters here: the Kaggle
notebook version string-matches `get_arch_list()` and falls back to CPU on
L4/sm_89, which is exactly the card Modal hands out. The local copies run a
real kernel launch instead, so they correctly stay on GPU.
"""
import os
import subprocess
import sys

import modal

app = modal.App("asanaai-train")

SRC = "/home/anamitra/Projects_and_Code/Scripts_and_Source"
HOME = "/home/anamitra"
DATA = f"{HOME}/yoga_raw_dataset"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("pandas", "numpy<2", "scikit-learn", "huggingface_hub")
    .add_local_file(f"{SRC}/train_mlp_3head_gpu.py",
                    "/opt/orig/train_mlp_3head_gpu.py", copy=True)
    .add_local_file(f"{SRC}/train_stgcn_gpu.py",
                    "/opt/orig/train_stgcn_gpu.py", copy=True)
    .add_local_file(f"{SRC}/generate_sequence_features.py",
                    "/opt/orig/generate_sequence_features.py", copy=True)
)

vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
secret = modal.Secret.from_name("arko007-hf-token")


def _token():
    """The secret's key name is not visible from `modal secret list`."""
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
              "HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN_ARKO007"):
        v = os.environ.get(k)
        if v:
            print(f"[wrapper] HF token found in ${k} (len {len(v)})", flush=True)
            return v
    print(f"[wrapper] NO HF token. env keys: "
          f"{[k for k in os.environ if 'HF' in k.upper() or 'HUG' in k.upper()]}",
          flush=True)
    return None


@app.function(image=image, volumes={VOL: vol}, secrets=[secret], timeout=900)
def seed_csv():
    """Pull the original 654k-row CSV from HF into the volume (once)."""
    from huggingface_hub import hf_hub_download
    import shutil

    os.makedirs(f"{VOL}/csv", exist_ok=True)
    dest = f"{VOL}/csv/master_mlp_dataset_fully_classified.csv"
    if os.path.exists(dest):
        print(f"already present: {os.path.getsize(dest)/1e6:.1f} MB")
        return os.path.getsize(dest)

    p = hf_hub_download(repo_id="Arko007/Yoga-650k",
                        filename="master_mlp_dataset_fully_classified.csv",
                        repo_type="dataset", token=_token())
    shutil.copy(p, dest)
    vol.commit()
    sz = os.path.getsize(dest)
    print(f"seeded {sz/1e6:.1f} MB")

    import pandas as pd
    df = pd.read_csv(dest)
    print(f"rows={len(df)} videos={df.video_id.nunique()}")
    print(df.imperfect_pose_label.value_counts().head(30))
    return sz


def _prep_home(need_csv=True, need_stgcn=False):
    os.makedirs(DATA, exist_ok=True)
    if need_csv:
        src = f"{VOL}/csv/master_mlp_dataset_fully_classified.csv"
        dst = f"{DATA}/master_mlp_dataset_fully_classified.csv"
        if not os.path.exists(dst):
            os.symlink(src, dst)
        print(f"CSV -> {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
    if need_stgcn:
        for f in ("stgcn_master_feats.npy", "stgcn_master_labels.npy"):
            dst = f"{DATA}/{f}"
            if not os.path.exists(dst):
                os.symlink(f"{VOL}/stgcn/orig12/{f}", dst)
            print(f"{f} -> {os.path.getsize(dst)/1e6:.1f} MB", flush=True)


def _run(script, tag, outputs):
    env = dict(os.environ)
    # See module docstring: this MUST be invalid, not absent.
    env["HF_TOKEN"] = "upload-disabled-by-wrapper-see-docstring"
    env["PYTHONUNBUFFERED"] = "1"
    print(f"\n{'='*70}\nRUN {script}  tag={tag}\n{'='*70}", flush=True)
    r = subprocess.run([sys.executable, script], env=env, cwd=HOME)
    print(f"\nexit code {r.returncode}", flush=True)

    outdir = f"{VOL}/runs/{tag}"
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for f in outputs:
        for cand in (f"{HOME}/{f}", f"{HOME}/{os.path.basename(f)}"):
            if os.path.exists(cand):
                import shutil
                shutil.copy(cand, f"{outdir}/{os.path.basename(f)}")
                saved.append((os.path.basename(f), os.path.getsize(cand)))
                break
    vol.commit()
    print(f"saved to volume: {saved}", flush=True)
    return {"exit": r.returncode, "saved": saved}


@app.function(image=image, volumes={VOL: vol}, secrets=[secret],
              gpu="L4", timeout=14400)
def train_mlp(tag: str = "mlp_baseline"):
    _prep_home(need_csv=True)
    return _run("/opt/orig/train_mlp_3head_gpu.py", tag,
                ["mlp_3head_model_v2.pth", "mlp_3head_pose_encoder_v2.npy"])


@app.function(image=image, volumes={VOL: vol}, secrets=[secret],
              gpu="L4", timeout=21600)
def train_stgcn(tag: str = "stgcn_baseline"):
    _prep_home(need_csv=False, need_stgcn=True)
    return _run("/opt/orig/train_stgcn_gpu.py", tag,
                ["stgcn_sequence_model_v2.pth", "stgcn_label_encoder_v2.npy"])


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=24576, timeout=3600)
def build_stgcn(tag: str = "orig12"):
    """Run the ORIGINAL generate_sequence_features.py on the 12 original
    videos and check it reproduces the label array the old ST-GCN trained on.

    `find_landmark_file_by_id` looks for landmarks_{id}.npy first, so symlinks
    named that way are all the unmodified script needs. It writes its output
    next to the landmarks (VIDEO_DIR), which here is a plain container dir.
    """
    import glob
    import numpy as np

    os.makedirs(DATA, exist_ok=True)
    for f in glob.glob(f"{VOL}/landmarks/landmarks_*.npy"):
        dst = f"{DATA}/{os.path.basename(f)}"
        if not os.path.exists(dst):
            os.symlink(f, dst)
    dst = f"{DATA}/master_mlp_dataset_fully_classified.csv"
    if not os.path.exists(dst):
        os.symlink(f"{VOL}/csv/master_mlp_dataset_fully_classified.csv", dst)

    r = subprocess.run([sys.executable, "/opt/orig/generate_sequence_features.py"],
                       cwd=HOME, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print("exit", r.returncode, flush=True)
    if r.returncode != 0:
        return {"exit": r.returncode}

    new_l = np.load(f"{DATA}/stgcn_master_labels.npy", allow_pickle=True)
    ref_l = np.load(f"{VOL}/stgcn_ref/stgcn_master_labels.npy", allow_pickle=True)
    same_len = len(new_l) == len(ref_l)
    agree = float((new_l == ref_l).mean()) if same_len else None
    print(f"\nregenerated {len(new_l)} windows | reference {len(ref_l)} | "
          f"length match {same_len} | label agreement {agree}", flush=True)
    if not same_len:
        import collections
        a, b = collections.Counter(new_l.tolist()), collections.Counter(ref_l.tolist())
        for k in sorted(set(a) | set(b)):
            print(f"  {k:<24} regen {a.get(k,0):>6}   ref {b.get(k,0):>6}")

    out = f"{VOL}/stgcn/{tag}"
    os.makedirs(out, exist_ok=True)
    for f in ("stgcn_master_feats.npy", "stgcn_master_labels.npy"):
        import shutil
        shutil.copy(f"{DATA}/{f}", f"{out}/{f}")
    vol.commit()
    return {"exit": 0, "windows": len(new_l), "same_len": same_len, "agreement": agree}


@app.local_entrypoint()
def main(stage: str = "seed"):
    if stage == "seed":
        print(seed_csv.remote())
    elif stage == "mlp":
        print(train_mlp.remote("mlp_baseline_origcode"))
    elif stage == "build":
        print(build_stgcn.remote("orig12"))
    elif stage == "stgcn":
        print(train_stgcn.remote("stgcn_baseline_origcode"))
