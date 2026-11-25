// Records the demo video of the fitting lab (see make_video.sh): the built UI
// (src/mxlpy/lab/static) against replay_lab.py, which replays recorded fits.
import { defineConfig } from "@playwright/test";

const port = 8770;
const here = import.meta.dirname;

export default defineConfig({
  testDir: here,
  testMatch: "record.spec.ts",
  outputDir: `${here}/out`,
  timeout: 300_000,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    viewport: { width: 1280, height: 800 },
    video: { mode: "on", size: { width: 1280, height: 800 } },
    browserName: "chromium",
  },
  webServer: {
    command: `uv run python ${here}/replay_lab.py ${here}/recording ${port}`,
    cwd: `${here}/../..`,
    url: `http://127.0.0.1:${port}/api/overview`,
    timeout: 60_000,
    reuseExistingServer: false,
  },
});
