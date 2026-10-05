"""Titles only, via YouTube's public oEmbed endpoint (no login, no media).
yt-dlp is bot-blocked from Modal, but oEmbed is a plain metadata API."""
import json
import urllib.parse
import urllib.request

import modal

app = modal.App("asanaai-titles")
image = modal.Image.debian_slim(python_version="3.11")

IDS = ["v7AYKMP6rOE", "ZiQh8jA5tVM", "O2EY79Ys_qg", "dAqQqmaI9vY", "4K2xTVRDJgA",
       "6CueZ4zujMk", "hHhxKkskHDg", "JHjV-wFTwSw", "EvMTrP8eRvM", "149Iac5fmoE",
       "Eml2xnoLpYE", "i6TzP2COtow", "4ZBUDd4bsyA"]


@app.function(image=image, timeout=300)
def titles():
    out = {}
    for v in IDS:
        u = ("https://www.youtube.com/oembed?format=json&url="
             + urllib.parse.quote(f"https://www.youtube.com/watch?v={v}"))
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}),
                    timeout=30) as r:
                j = json.loads(r.read())
            out[v] = {"title": j.get("title"), "author": j.get("author_name")}
        except Exception as e:
            out[v] = {"error": f"{type(e).__name__}: {e}"[:120]}
    return out


@app.local_entrypoint()
def main():
    for v, d in titles.remote().items():
        print(f"{v}  {d.get('title') or d.get('error')}   [{d.get('author','')}]")
