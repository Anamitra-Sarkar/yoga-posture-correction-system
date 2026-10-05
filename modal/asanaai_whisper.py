"""Transcribe the 24 training videos so poses can be labelled by what the instructor SAYS (cue) AND what the body DOES.

Output: /data/transcripts/<video_id>.json = {fps, duration, n_video_frames, language, chunks:[{t0,t1,text}]}
Old 12 videos come from HF `Arko007/yoga-dataset-raw/Yoga_Dataset_Raw/<id>.mp4`, new 12 from the volume.
"""
import json
import os
import subprocess

import modal

app = modal.App("asanaai-whisper")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("transformers==4.44.2", "accelerate", "numpy<2", "huggingface_hub")
)
vol = modal.Volume.from_name("asanaai-data", create_if_missing=True)
VOL = "/data"
secret = modal.Secret.from_name("arko007-hf-token")
RAW = "Arko007/yoga-dataset-raw"
OLD = "HmZFwoUU3WQ SZU7Sbgu57o oUgpXY7QhpQ 7ciS93shMNQ P8uHMMmWMHQ 4ORRiN2_aVI RQMtwbhXD7A 8ibxmzJziHU Eml2xnoLpYE QiebZSlTw_U L-z1HLkS_-Y s-1vMbAgYWU".split()
NEW = ("v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk hHhxKkskHDg JHjV-wFTwSw "
       "EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow 4ZBUDd4bsyA").split()
HINDI = {"4K2xTVRDJgA", "4ZBUDd4bsyA"}


def _tok():
    for k in ("HF_TOKEN_ARKO007", "HF_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]


@app.function(image=image, volumes={VOL: vol}, secrets=[secret], gpu="L4", timeout=3 * 3600, max_containers=8)
def transcribe(vid: str):
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import pipeline

    try:
        vol.reload()
        if vid in OLD:
            path = hf_hub_download(RAW, f"Yoga_Dataset_Raw/{vid}.mp4", repo_type="dataset", token=_tok())
        else:
            path = f"{VOL}/hold_videos/{vid}.mp4"
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                             "stream=r_frame_rate,nb_read_frames:format=duration", "-of", "json", path],
                            capture_output=True, text=True)
        info = json.loads(pr.stdout)
        st = info["streams"][0]
        num, den = st["r_frame_rate"].split("/")
        fps, nfr, dur = float(num) / float(den), int(st.get("nb_read_frames") or 0), float(info["format"]["duration"])
        raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
                              "-f", "f32le", "-"], capture_output=True).stdout
        audio = np.frombuffer(raw, dtype=np.float32)
        print(vid, f"fps={fps:.3f} frames={nfr} dur={dur:.1f}s audio={len(audio)/16000:.1f}s", flush=True)
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
                        device="cuda:0", token=_tok())
        lang = "hi" if vid in HINDI else "en"
        out = pipe({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30, batch_size=16, return_timestamps=True,
                   generate_kwargs={"task": "transcribe", "language": lang})
        chunks = [{"t0": c["timestamp"][0], "t1": c["timestamp"][1] if c["timestamp"][1] is not None else c["timestamp"][0] + 3,
                   "text": c["text"].strip()} for c in out["chunks"]]
        res = {"id": vid, "fps": fps, "n_video_frames": nfr, "duration": dur, "language": lang, "chunks": chunks}
        os.makedirs(f"{VOL}/transcripts", exist_ok=True)
        json.dump(res, open(f"{VOL}/transcripts/{vid}.json", "w"), ensure_ascii=False)
        vol.commit()
        return {"id": vid, "ok": True, "fps": round(fps, 3), "frames": nfr, "chunks": len(chunks),
                "sample": " | ".join(c["text"] for c in chunks[10:14])[:260]}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"id": vid, "ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-500:]}


@app.local_entrypoint()
def main():
    for r in transcribe.map(OLD + NEW):
        print(r, flush=True)
