#!/usr/bin/env bash
# Records the demo video of the fitting lab web UI: docs/lab-video/fitting-lab-demo.mp4
#
#   docs/lab-video/make_video.sh            # reuse the recorded fits if present
#   docs/lab-video/make_video.sh --refit    # run the fits again (after lab changes)
#
# 1. record_runs.py runs the real fits once (takes a few minutes) into recording/
# 2. web/lab is built, so the video shows the current UI
# 3. Playwright drives the UI against replay_lab.py, which replays the recorded
#    fits at about 3 s each (record.spec.ts holds the storyline and captions)
# 4. ffmpeg turns the webm into an MP4
#
# Needs: uv sync --extra lab, npm ci in web/lab, npx playwright install chromium, ffmpeg.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
web="$root/web/lab"

if [[ "${1:-}" == "--refit" || ! -d "$here/recording" ]]; then
  (cd "$root" && uv run python "$here/record_runs.py" "$here/recording")
fi

npm --prefix "$web" run build
# The config resolves @playwright/test from here
ln -sfn "$web/node_modules" "$here/node_modules"
(cd "$web" && npx playwright test --config "$here/record.config.ts")

codec=libx264
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q " libx264 " || codec=libopenh264
ffmpeg -y -v error -ss 0.8 -i "$here/out/record-fitting-lab-demo/video.webm" \
  -vf "fps=25,format=yuv420p" -c:v "$codec" -b:v 3M -movflags +faststart \
  "$here/fitting-lab-demo.mp4"
echo "Wrote $here/fitting-lab-demo.mp4"
