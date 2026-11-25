import { expect, test } from "@playwright/test";

const TOKEN = "e2e-token";

test("decide the gates of a run in the browser", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const gate = page.getByRole("region", { name: "Review gate" });
  const steps = page.getByRole("navigation", { name: "Steps" });

  await page.goto(`/?token=${TOKEN}`);
  // the token moved into a cookie
  await expect(page).toHaveURL(/\/$/);
  await expect(
    page.getByRole("heading", { name: "Setup waits for your decision" }),
  ).toBeVisible({ timeout: 60_000 });

  await page.getByRole("link", { name: /Review the results/ }).click();
  await expect(page.getByRole("heading", { name: "0 · Setup" })).toBeVisible();
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.getByRole("button", { name: "Accept proposal" }).click();
  await expect(gate.getByText("decided (user)")).toBeVisible();

  // the server goes on by itself and stops at the next gate
  await steps.getByRole("link", { name: /Parameter selection/ }).click();
  await expect(gate.getByText("waiting for a decision")).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.getByRole("checkbox", { name: "k1" })).toBeChecked();
  await expect(page.getByRole("table").first()).toBeVisible();
  await page.getByRole("button", { name: "Decide as edited" }).click();
  await expect(gate.getByText("decided (user)")).toBeVisible();

  // the fit runs; its gate shows the data with the fit and the starts
  await steps.getByRole("link", { name: /Fit/ }).click();
  await expect(gate.getByText("waiting for a decision")).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.getByText("Multi-start")).toBeVisible();
  expect(await page.locator("canvas").count()).toBeGreaterThanOrEqual(2);

  // pausing goes through the run directory
  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByRole("button", { name: "Unpause" })).toBeVisible();
  await page.getByRole("button", { name: "Unpause" }).click();
  await expect(page.getByRole("button", { name: "Pause" })).toBeVisible();

  expect(errors).toEqual([]);
});
