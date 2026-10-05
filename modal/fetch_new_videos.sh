#!/usr/bin/env bash
# Fetch the 12 NEW follow-along videos from this machine's home IP (YouTube
# hard-blocks Modal's datacenter IPs) and push them straight into the Modal
# volume, one at a time, deleting each local copy after upload so disk and
# RAM stay flat. Eml2xnoLpYE is deliberately absent: it is already in the CSV.
#
# 480p video-only is enough: the original extractor resized every frame to
# max_width=640 before MediaPipe, so a 640x360-class stream feeds MediaPipe
# the same pixel scale it always saw. No audio, so no ffmpeg merge step.
set -u
IDS="v7AYKMP6rOE ZiQh8jA5tVM O2EY79Ys_qg dAqQqmaI9vY 4K2xTVRDJgA 6CueZ4zujMk hHhxKkskHDg JHjV-wFTwSw EvMTrP8eRvM 149Iac5fmoE i6TzP2COtow 4ZBUDd4bsyA"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
ok=0; bad=0
for id in $IDS; do
  echo "=== $id ==="
  if yt-dlp -q --no-warnings -f "bv*[height<=480][ext=mp4]/bv*[height<=480]/b[height<=480]" \
        -o "$TMP/$id.%(ext)s" "https://youtu.be/$id" \
     && f="$(ls "$TMP/$id".* 2>/dev/null | head -1)" && [ -n "$f" ] \
     && modal volume put --force asanaai-data "$f" "/hold_videos/$id.mp4"; then
    echo "  ok  $(du -h "$f" | cut -f1)"; ok=$((ok+1))
  else
    echo "  FAILED $id"; bad=$((bad+1))
  fi
  rm -f "$TMP/$id".*
done
echo; echo "uploaded $ok, failed $bad"
