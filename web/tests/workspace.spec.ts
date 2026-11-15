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
  await page.getByRole("button", { name: "Upload audio" }).click();
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

test("guided recording supplies multilingual passages and reports microphone denial", async ({
  page,
}) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      throw new DOMException("Microphone permission denied", "NotAllowedError");
    };
  });
  await page.goto("/");
  const language = page.getByLabel("Language you will read");
  for (const [value, tag] of [
    ["Chinese", "zh"],
    ["Japanese", "ja"],
    ["Korean", "ko"],
  ]) {
    await language.selectOption(value);
    await expect(page.locator(`p[lang="${tag}"]`)).toBeVisible();
  }
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(
    page.getByText("Microphone permission denied", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start recording" }),
  ).toBeEnabled();
});

test("recording requires confirmation and can be replaced", async ({
  browser,
}) => {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto("/");
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(page.getByLabel("Language you will read")).toBeDisabled();
  await expect(
    page.getByRole("status").filter({ hasText: "Recording" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(page.getByLabel("Review your recording")).toBeVisible();
  await page.getByRole("button", { name: "Use this recording" }).click();
  await expect(
    page.getByRole("button", { name: "Voice selected" }),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Record my voice", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Voice selected" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Record again" }).click();
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(
    page.getByRole("button", { name: "Use this recording" }),
  ).toBeEnabled();
  await page.getByLabel("Language you will read").selectOption("Japanese");
  await expect(page.getByLabel("Review your recording")).toHaveCount(0);
  await context.close();
});
