import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// The build is committed inside the Python package, which serves it
// (mxlpy.lab.server). `npm run dev` proxies the API to a running
// `mxlpy lab serve --no-token <run_dir>`.
const out = "../../src/mxlpy/lab/static";

export default defineConfig({
  plugins: [
    sveltekit({
      preprocess: [vitePreprocess()],
      adapter: adapter({
        pages: out,
        assets: out,
        fallback: "index.html",
        precompress: false,
        strict: true,
      }),
      paths: { relative: false },
      // A fixed version keeps the build reproducible (CI checks that the
      // committed build is up to date by rebuilding it)
      version: { name: "mxlpy-lab" },
      output: { bundleStrategy: "single" },
    }),
  ],
  optimizeDeps: { exclude: ["@computational-biology-aachen/design"] },
  server: {
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
