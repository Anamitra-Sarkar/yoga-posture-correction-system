"""Second YouTube-reachability probe on Modal: the pieces probe 1 lacked.

Probe 1 used a bare `pip install yt-dlp` on slim Debian and failed with
"Sign in to confirm you're not a bot" on 10 player clients. The user's own
Colab script uses the same library, so the delta is (a) IP reputation, which
we cannot change, and (b) things current yt-dlp wants and a bare image lacks:
  * a JS runtime (deno) + yt-dlp-ejs for YouTube's n/sig challenge,
  * curl-cffi for browser TLS impersonation,
  * a PO-token provider (bgutil) for the web/mweb clients.
This probe turns those on one at a time, metadata only (no bytes), so we learn
which one (if any) moves the needle.
"""
import json
import os
import subprocess
import time

import modal

app = modal.App("asanaai-ytdlp-probe2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl", "git", "unzip", "ca-certificates", "nodejs", "npm")
    .run_commands("curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh")
    .pip_install("yt-dlp[default,curl-cffi]", "bgutil-ytdlp-pot-provider",
                 extra_options="--pre -U")
    .run_commands(
        # server half of the PO-token provider; version must match the pip plugin
        "V=$(pip show bgutil-ytdlp-pot-provider | awk '/^Version:/{print $2}') && "
        "echo plugin=$V && git clone --depth 1 --branch $V "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider /opt/bgutil && "
        "cd /opt/bgutil/server && npm ci && npx tsc"
    )
)

TEST = ["v7AYKMP6rOE", "149Iac5fmoE", "i6TzP2COtow"]


def run(args, timeout=240):
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    tail = (r.stderr or r.stdout).strip().splitlines()[-2:]
    return {"rc": r.returncode, "secs": round(time.time() - t, 1), "tail": tail,
            "out": r.stdout.strip()[:300]}


@app.function(image=image, timeout=3000)
def probe():
    res = {}
    res["versions"] = run(["yt-dlp", "--version"])["out"]
    res["deno"] = run(["deno", "--version"])["out"].splitlines()[:1]
    base = ["yt-dlp", "--simulate", "--no-warnings", "-f", "bv*[height<=480]",
            "--print", "%(id)s fmt=%(format_id)s %(height)sp"]
    srv = subprocess.Popen(["node", "/opt/bgutil/server/build/main.js"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(6)
    res["bgutil_server_alive"] = srv.poll() is None
    combos = {
        "A_default+deno": [],
        "B_impersonate_chrome": ["--impersonate", "chrome"],
        "C_bgutil_mweb": ["--extractor-args", "youtube:player_client=mweb"],
        "D_bgutil_web": ["--extractor-args", "youtube:player_client=web"],
        "E_bgutil_tv": ["--extractor-args", "youtube:player_client=tv"],
        "F_bgutil_imp_web_mweb_tv": ["--impersonate", "chrome",
                                     "--extractor-args", "youtube:player_client=web,mweb,tv"],
    }
    for name, extra in combos.items():
        res[name] = {}
        for vid in TEST:
            res[name][vid] = run(base + extra + [f"https://youtu.be/{vid}"])
    srv.terminate()
    print(json.dumps(res, indent=1))
    return res


@app.local_entrypoint()
def main():
    probe.remote()
