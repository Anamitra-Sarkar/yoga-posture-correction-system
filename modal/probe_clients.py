"""YouTube refuses Modal's datacenter IP with "confirm you're not a bot" on the
default web client. Different InnerTube clients have different bot-check
paths, so before resorting to cookies (a credential I would rather not move
onto a VM) try each client on one video and report which, if any, works."""
import json
import subprocess

import modal

app = modal.App("asanaai-probe-clients")
image = modal.Image.debian_slim(python_version="3.11").pip_install("yt-dlp")

CLIENTS = ["tv", "tv_simply", "android_vr", "mweb", "web_embedded",
           "web_safari", "ios", "android", "web_music", "default"]
TEST = "https://youtu.be/v7AYKMP6rOE"


@app.function(image=image, timeout=1800)
def try_clients():
    res = {}
    for c in CLIENTS:
        cmd = ["yt-dlp", "-J", "--no-warnings", "--skip-download"]
        if c != "default":
            cmd += ["--extractor-args", f"youtube:player_client={c}"]
        cmd.append(TEST)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                j = json.loads(r.stdout)
                res[c] = {"ok": True, "title": j.get("title"),
                          "formats": len(j.get("formats") or [])}
            else:
                err = (r.stderr or "").strip().splitlines()
                res[c] = {"ok": False, "err": (err[-1] if err else "")[:180]}
        except Exception as e:
            res[c] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:180]}
        print(f"{c:<14} {res[c]}", flush=True)
    return res


@app.local_entrypoint()
def main():
    r = try_clients.remote()
    good = [k for k, v in r.items() if v.get("ok")]
    print(f"\n=== WORKING CLIENTS: {good or 'NONE'} ===")
