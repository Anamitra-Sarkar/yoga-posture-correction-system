"""Can Kaggle's network reach YouTube? Metadata-only probe, nothing downloaded."""
import json, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"], check=False)
IDS = ["v7AYKMP6rOE", "149Iac5fmoE", "4ZBUDd4bsyA"]
for v in IDS:
    for client in (None, "tv", "web_embedded", "mweb"):
        cmd = ["yt-dlp", "-J", "--no-warnings", "--skip-download"]
        if client:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
        cmd.append(f"https://youtu.be/{v}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        if r.returncode == 0:
            j = json.loads(r.stdout)
            print(f"PROBE {v} client={client}: OK title={j.get('title')!r} formats={len(j.get('formats') or [])}", flush=True)
        else:
            err = (r.stderr.strip().splitlines() or [""])[-1][:140]
            print(f"PROBE {v} client={client}: FAIL {err}", flush=True)
