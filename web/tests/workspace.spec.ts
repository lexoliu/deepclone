import { test, expect } from "@playwright/test";
test("unsupported browsers receive an actionable message and cannot load", async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(navigator, "gpu", {
      value: undefined,
      configurable: true,
    }),
  );
  await page.route("**/models.json", (route) =>
    route.fulfill({
      json: [
        {
          size: "0.6b",
          manifest: "https://example.test/manifest.json",
          sha256: "0".repeat(64),
        },
      ],
    }),
  );
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Make it your voice." }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Generate speech" }),
  ).toBeDisabled();
  await page
    .getByLabel("Reference transcript")
    .fill("法律让人安心。規則への信頼。규칙에 대한 신뢰.");
  await expect(page.getByLabel("Reference transcript")).toHaveValue(
    "法律让人安心。規則への信頼。규칙에 대한 신뢰.",
  );
});
test("workspace remains usable at a narrow viewport", async ({ page }) => {
  await page.route("**/models.json", (route) => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByText("No models are available.", { exact: false }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});
test("published wasm glue exports the browser inference contract", async ({
  page,
}) => {
  await page.goto("/");
  const exports = await page.evaluate(async () => {
    const url = "/wasm/deepclone.js";
    const module = await import(url);
    await module.default();
    return [
      typeof module.ModelBuilder.create,
      typeof module.TextInput,
      typeof module.BrowserVoice,
    ];
  });
  expect(exports).toEqual(["function", "function", "function"]);
});
