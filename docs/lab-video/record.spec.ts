// Storyline of the demo video (make_video.sh). Waits are for the viewer: the
// replay (replay_lab.py) makes every fit take about 3 s.
//
// Identifiability and uncertainty are opened only once they finished: their
// refits start at the optimum, so the live loss trace is flat to ~1e-14 and
// PlotChart's tick loop doesn't end on such a range (crashes the tab).
import { expect, test, type Locator, type Page } from "@playwright/test";

// Caption banner, a visible cursor, and a faster live-fit poll (3 s -> 0.8 s,
// so the ~3 s fits show the best fit so far more than once)
const overlay = () => {
  const interval = window.setInterval;
  // @ts-expect-error -- narrowed signature is fine here
  window.setInterval = (fn: TimerHandler, ms?: number, ...a: unknown[]) =>
    interval(fn, ms === 3000 ? 800 : ms, ...a);
  window.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = `
      #demo-cursor { position: fixed; z-index: 99999; width: 22px; height: 22px;
        margin: -11px 0 0 -11px; border-radius: 50%; pointer-events: none;
        background: rgba(246,168,0,.45); border: 2px solid rgb(246,168,0);
        transition: transform .15s; left: -50px; top: -50px; }
      #demo-cursor.down { transform: scale(.6); }
      #demo-caption { position: fixed; z-index: 99998; left: 50%; bottom: 28px;
        transform: translateX(-50%); max-width: 80%; padding: 12px 22px;
        border-radius: 10px; background: rgba(20,30,35,.88); color: #fff;
        font: 500 20px/1.35 "Space Grotesk", system-ui, sans-serif;
        text-align: center; pointer-events: none; transition: opacity .3s; }
      #demo-caption:empty { opacity: 0; padding: 0; }`;
    document.head.append(style);
    const cursor = Object.assign(document.createElement("div"), { id: "demo-cursor" });
    const caption = Object.assign(document.createElement("div"), { id: "demo-caption" });
    document.body.append(cursor, caption);
    const saved = sessionStorage.getItem("demo-cursor");
    if (saved) Object.assign(cursor.style, JSON.parse(saved));
    caption.textContent = sessionStorage.getItem("demo-caption") ?? "";
    addEventListener("mousemove", (e) => {
      cursor.style.left = `${e.clientX}px`;
      cursor.style.top = `${e.clientY}px`;
      sessionStorage.setItem("demo-cursor", JSON.stringify({ left: cursor.style.left, top: cursor.style.top }));
    });
    addEventListener("mousedown", () => cursor.classList.add("down"));
    addEventListener("mouseup", () => cursor.classList.remove("down"));
  });
};

let mouse = { x: 640, y: 400 };

async function caption(page: Page, text: string, hold = 0): Promise<void> {
  await page.evaluate((t) => {
    sessionStorage.setItem("demo-caption", t);
    const el = document.getElementById("demo-caption");
    if (el) el.textContent = t;
  }, text);
  if (hold) await page.waitForTimeout(hold);
}

async function moveTo(page: Page, target: Locator): Promise<void> {
  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) throw new Error("target not visible");
  const to = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  const steps = Math.max(8, Math.round(Math.hypot(to.x - mouse.x, to.y - mouse.y) / 25));
  await page.mouse.move(to.x, to.y, { steps });
  mouse = to;
  await page.waitForTimeout(250);
}

async function click(page: Page, target: Locator, after = 600): Promise<void> {
  await moveTo(page, target);
  await page.mouse.down();
  await page.waitForTimeout(90);
  await page.mouse.up();
  await page.waitForTimeout(after);
}

async function scrollBy(page: Page, dy: number, hold = 1200): Promise<void> {
  await page.evaluate((d) => window.scrollBy({ top: d, behavior: "smooth" }), dy);
  await page.waitForTimeout(hold);
}

test("fitting lab demo", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.addInitScript(overlay);
  const gate = page.getByRole("region", { name: "Review gate" });
  const steps = page.getByRole("navigation", { name: "Steps" });
  const waiting = gate.getByText("waiting for a decision");
  const step = (name: RegExp) => steps.getByRole("link", { name });

  // 1. Overview
  await page.goto("/?token=demo");
  await expect(page.getByRole("heading", { name: "Setup waits for your decision" })).toBeVisible();
  await page.mouse.move(mouse.x, mouse.y);
  await caption(page, "mxlpy lab serve: the server runs the fitting steps and stops at every review gate", 3500);

  // 2. Setup
  await click(page, page.getByRole("link", { name: /Review the results/ }), 800);
  await caption(page, "Step 0 · Setup: observables, noise and integrator, with the data", 2500);
  await scrollBy(page, 500, 1800);
  await scrollBy(page, -500, 800);
  await caption(page, "Accept the proposal: the run goes on by itself");
  await click(page, gate.getByRole("button", { name: "Accept proposal" }), 700);

  // 3. Parameter selection: leave out k_out
  await click(page, step(/Parameter selection/), 300);
  await expect(waiting).toBeVisible();
  await caption(page, "Step 1 · Parameters ranked by sensitivity; all four are proposed", 2200);
  await scrollBy(page, 450, 1500);
  await scrollBy(page, -450, 600);
  await caption(page, "Override the proposal: keep k_out fixed at its (wrong) start value");
  await click(page, page.getByRole("checkbox", { name: "k_out" }), 500);
  await click(page, gate.getByRole("button", { name: "Decide as edited" }), 1200);

  // 4. The fit runs live, then fails
  await click(page, step(/Fit/), 0);
  await caption(page, "Step 2 · Multi-start fit, live: best objective per start and the best fit so far");
  await expect(waiting).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(800);
  await caption(page, "The fit fails: χ² far above its degrees of freedom, structured residuals", 2500);
  await scrollBy(page, 520, 2500);
  await scrollBy(page, -520, 600);

  // 5. Change the decision: estimate k_out again
  await caption(page, "Go back and reopen the parameter decision");
  await click(page, step(/Parameter selection/), 600);
  await click(page, gate.getByRole("button", { name: "Reopen" }), 700);
  await caption(page, "The form is back at the proposal: accept it, estimating k_out as well");
  await page.waitForTimeout(1200);
  await click(page, gate.getByRole("button", { name: "Accept proposal" }), 900);
  await click(page, step(/Fit/), 0);
  await caption(page, "The fit reruns with all four parameters");
  await expect(waiting).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(800);
  await caption(page, "Now it passes: χ² = 172 for 176 degrees of freedom", 2500);
  await scrollBy(page, 520, 2500);
  await scrollBy(page, -520, 600);

  // 6. Pause, decide, unpause
  await caption(page, "Pause: the run stops before its next step, here or from any other process");
  await click(page, page.getByRole("button", { name: "Pause" }), 600);
  await caption(page, "Choose what comes next and decide");
  await click(page, page.getByRole("combobox", { name: "next" }), 300);
  await page.getByRole("combobox", { name: "next" }).selectOption("identifiability");
  await page.waitForTimeout(500);
  await click(page, gate.getByRole("button", { name: "Decide as edited" }), 1500);
  await caption(page, "Unpause: the run goes on");
  await click(page, page.getByRole("button", { name: "Unpause" }), 200);

  // 7. Identifiability
  await caption(page, "Step 4 · Identifiability runs: Fisher information, pulling test and profile likelihoods");
  await expect(step(/Identifiability/)).toContainText(/passed|failed/, { timeout: 20_000 });
  await page.waitForTimeout(600);
  await click(page, step(/Identifiability/), 0);
  await expect(waiting).toBeVisible();
  await page.waitForTimeout(800);
  await caption(page, "All four parameters are identifiable", 1500);
  await scrollBy(page, 600, 2500);
  await scrollBy(page, -600, 600);
  await click(page, gate.getByRole("button", { name: "Accept proposal" }), 600);

  // 8. Uncertainty
  await caption(page, "Step 5 · Uncertainty runs: prediction bands of the observables");
  await expect(step(/Uncertainty/)).toContainText(/done|passed/, { timeout: 20_000 });
  await page.waitForTimeout(600);
  await click(page, step(/Uncertainty/), 0);
  await expect(gate).toBeVisible();
  await page.waitForTimeout(800);
  await scrollBy(page, 500, 2800);
  await scrollBy(page, -500, 600);
  await click(page, gate.getByRole("button", { name: /Accept proposal|Continue/ }), 900);
  await caption(page, "The run is finished: everything lives in the run directory");
  await expect(page.getByRole("status")).toContainText("finished");
  await page.waitForTimeout(1500);
  await caption(page, "The fitted estimates and every decision stay in the run directory");
  await scrollBy(page, 2000, 3000);
  await caption(page, "");
  await page.waitForTimeout(800);

  expect(errors).toEqual([]);
});
