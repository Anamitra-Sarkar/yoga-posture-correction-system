"""Environment probe: internet, GPU, and which secrets exist on this Kaggle account. Prints presence only, never values."""
import os, subprocess, urllib.request

def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

print("GPU:", sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or "none")
try:
    import torch
    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(), "n_gpu:", torch.cuda.device_count())
except Exception as e:
    print("torch import failed:", e)
for u in ("https://huggingface.co", "https://modal.com", "https://pypi.org"):
    try:
        print(u, "->", urllib.request.urlopen(u, timeout=15).status)
    except Exception as e:
        print(u, "-> FAIL", type(e).__name__)
print("disk free (GB):", sh("df -BG /kaggle/working | tail -1"))
try:
    from kaggle_secrets import UserSecretsClient
    c = UserSecretsClient()
    for name in ("HF_TOKEN", "HF_TOKEN_ARKO007", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        try:
            v = c.get_secret(name)
            print("secret", name, "PRESENT (len %d)" % len(v))
            if name.startswith("HF_TOKEN"):
                from huggingface_hub import HfApi
                print("   -> authenticates as:", HfApi(token=v).whoami().get("name"))
        except Exception as e:
            print("secret", name, "absent")
except Exception as e:
    print("kaggle_secrets unavailable:", e)
