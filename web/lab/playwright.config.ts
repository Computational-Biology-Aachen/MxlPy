import { defineConfig, devices } from "@playwright/test";

// Smoke test of the built UI (src/mxlpy/lab/static) against a real lab run
// served by mxlpy (tests/e2e/serve.py). Build first: `npm run build`.
const port = 8766;

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 120_000,
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: `http://127.0.0.1:${port}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `uv run python web/lab/tests/e2e/serve.py ${port}`,
    cwd: "../..",
    url: `http://127.0.0.1:${port}/api/overview`,
    // Any response (401 without the token) means the server is up
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
