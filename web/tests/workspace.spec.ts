import { test, expect } from "@playwright/test";

function referenceWav() {
  const wav = Buffer.alloc(44 + 4800);
  wav.write("RIFF");
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(24000, 24);
  wav.writeUInt32LE(48000, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(4800, 40);
  return { name: "reference.wav", mimeType: "audio/wav", buffer: wav };
}

test.beforeEach(async ({ page }) => {
  // UI tests never download real weights; GPU contract tests initialize explicitly.
  await page.route("**/models.json", (route) => route.fulfill({ json: [] }));
});
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
    page.getByRole("button", { name: "Start recording" }),
  ).toBeEnabled();
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  for (const label of [
    "Context tokens",
    "Maximum audio frames",
    "Frames per audio chunk",
    "Random seed",
    "Model",
    "Generation settings",
  ]) {
    await expect(page.getByText(label, { exact: true })).toHaveCount(0);
  }
  await expect(page.locator("body")).not.toContainText(/Qwen|0\.6B|1\.7B/);
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
      typeof module.BrowserRuntime.create,
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

test("a GPU runtime remains reusable after a rejected model shard", async ({
  page,
}) => {
  await page.goto("/");
  const rejected = await page.evaluate(async () => {
    const url = "/wasm/deepclone.js";
    const module = await import(url);
    await module.default();
    const runtime = await module.BrowserRuntime.create();
    const first = runtime.builder(true);
    let rejected = false;
    try {
      first.upload(new Uint8Array([0]), false);
    } catch {
      rejected = true;
    } finally {
      first.free();
    }
    const second = runtime.builder(true);
    second.free();
    runtime.free();
    return rejected;
  });
  expect(rejected).toBe(true);
});

test("confirmed recordings survive refresh, replacement and deletion", async ({
  browser,
}) => {
  const context = await browser.newContext();
  await context.route("**/models.json", (route) => route.fulfill({ json: [] }));
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
    page.getByText("Saved on this device", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByLabel("Saved voice recording")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start recording" }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Record again" }).click();
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(
    page.getByRole("button", { name: "Use this recording" }),
  ).toBeEnabled();
  await page.getByLabel("Language you will read").selectOption("Japanese");
  await expect(page.getByLabel("Review your recording")).toHaveCount(0);
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.getByRole("button", { name: "Stop recording" }).click();
  await page.getByRole("button", { name: "Use this recording" }).click();
  await expect(
    page.getByText("Saved on this device", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByText("静かな朝に窓を開けると", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Delete voice" }).click();
  await expect(page.getByLabel("Saved voice recording")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Start recording" }),
  ).toBeEnabled();
  // Wait for the deletion transaction to commit, then verify a new page load.
  await page.waitForFunction(async () => {
    const request = indexedDB.open("deepclone-voices");
    const db = await new Promise<IDBDatabase>((resolve) => {
      request.onsuccess = () => resolve(request.result);
    });
    try {
      const read = db
        .transaction("reference")
        .objectStore("reference")
        .get("selected");
      return await new Promise<boolean>((resolve) => {
        read.onsuccess = () => resolve(read.result === undefined);
      });
    } finally {
      db.close();
    }
  });
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Start recording" }),
  ).toBeEnabled();
  await expect(page.getByLabel("Saved voice recording")).toHaveCount(0);
  await context.close();
});

test("background preparation stays usable and cancellation terminates an unresponsive worker", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", { value: {}, configurable: true });
    class TestWorker {
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror = null;
      postMessage(command: { kind: string; id?: number }) {
        if (command.kind === "load") {
          Object.assign(window, {
            preparationStarted: true,
            preparationCount:
              ((window as unknown as { preparationCount?: number })
                .preparationCount ?? 0) + 1,
            completePreparation: () =>
              this.onmessage?.(
                new MessageEvent("message", { data: { kind: "ready" } }),
              ),
          });
          this.onmessage?.(
            new MessageEvent("message", {
              data: {
                kind: "progress",
                completed: 1,
                total: 2,
                phase: "Technical model label",
              },
            }),
          );
        }
        if (command.kind === "reference") {
          Object.assign(window, {
            referenceCount:
              ((window as unknown as { referenceCount?: number })
                .referenceCount ?? 0) + 1,
          });
        }
        const event =
          command.kind === "reference"
            ? { kind: "conditioned" }
            : command.kind === "start"
              ? { kind: "started" }
              : command.kind === "text"
                ? { kind: "accepted", id: command.id }
                : command.kind === "finish"
                  ? { kind: "done" }
                  : null;
        if (event)
          this.onmessage?.(new MessageEvent("message", { data: event }));
      }
      terminate() {
        Object.assign(window, { terminated: true });
      }
    }
    Object.defineProperty(window, "Worker", { value: TestWorker });
  });
  await page.route("**/models.json", (route) =>
    route.fulfill({
      json: [
        {
          size: "0.6b",
          manifest: "https://example.test/model.json",
          sha256: "0".repeat(64),
        },
      ],
    }),
  );
  await page.goto("/");
  await page.waitForFunction(
    () =>
      (window as unknown as { preparationStarted: boolean }).preparationStarted,
  );
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Start recording" }),
  ).toBeEnabled();
  await expect(page.getByLabel("What should your voice say?")).toBeEnabled();
  await expect(page.locator("body")).not.toContainText("Technical model label");
  await page.getByRole("button", { name: "Upload audio", exact: true }).click();
  await page
    .getByLabel("Reference recording", { exact: true })
    .setInputFiles(referenceWav());
  await page
    .getByLabel("Reference transcript")
    .fill("法律让人安心。規則への信頼。규칙에 대한 신뢰.");
  await page
    .getByLabel("What should your voice say?")
    .fill("Hello from my voice.");
  await page.getByRole("button", { name: "Generate speech" }).click();
  await expect(page.getByRole("progressbar")).toHaveAttribute(
    "aria-valuenow",
    "50",
  );
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  expect(
    await page.evaluate(
      () => (window as unknown as { terminated: boolean }).terminated,
    ),
  ).toBe(true);
  await expect(page.getByText("Cancelled", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Generate speech" }),
  ).toBeEnabled();
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  await page.getByRole("button", { name: "Generate speech" }).click();
  await expect(page.getByRole("progressbar")).toBeVisible();
  await page.waitForFunction(
    () =>
      (window as unknown as { preparationCount: number }).preparationCount ===
      2,
  );
  await page.evaluate(() =>
    (
      window as unknown as { completePreparation: () => void }
    ).completePreparation(),
  );
  await expect(page.getByRole("link", { name: "Download WAV" })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Generate speech" }),
  ).toBeEnabled();
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  for (const expected of [1, 2]) {
    if (expected === 2)
      await page
        .getByLabel("Reference transcript")
        .fill("规则让人安心。規則を信頼する。규칙을 믿는다.");
    await page.getByRole("button", { name: "Generate speech" }).click();
    await expect(
      page.getByRole("link", { name: "Download WAV" }),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () => (window as unknown as { referenceCount: number }).referenceCount,
      ),
    ).toBe(expected);
  }
});

test("a crashed worker is discarded and the next request creates a fresh worker", async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(navigator, "gpu", { value: {}, configurable: true }),
  );
  await page.route("**/models.json", (route) =>
    route.fulfill({
      json: [
        {
          size: "0.6b",
          manifest: "https://example.test/model.json",
          sha256: "0".repeat(64),
        },
      ],
    }),
  );
  let created = 0;
  await page.context().route("**/inference.worker-*.js", (route) =>
    route.fulfill({
      path: new URL(
        ++created === 1
          ? "./fixtures/crashing-worker.js"
          : "./fixtures/unresponsive-worker.js",
        import.meta.url,
      ).pathname,
      contentType: "text/javascript",
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
      },
    }),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Upload audio", exact: true }).click();
  await page
    .getByLabel("Reference recording", { exact: true })
    .setInputFiles(referenceWav());
  await page
    .getByLabel("Reference transcript")
    .fill("法律让人安心。規則への信頼。규칙에 대한 신뢰.");
  await page
    .getByLabel("What should your voice say?")
    .fill("Hello from my voice.");
  await page.getByRole("button", { name: "Generate speech" }).click();
  await expect(page.getByRole("alert")).toHaveText(
    "Speech could not be prepared. Please try again.",
  );
  await expect(
    page.getByRole("button", { name: "Generate speech" }),
  ).toBeEnabled();
  expect(created).toBe(1);
  await page.getByRole("button", { name: "Generate speech" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Generating speech" }),
  ).toBeVisible();
  expect(created).toBe(2);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await expect(page.getByText("Cancelled", { exact: true })).toBeVisible();
});

test("cancel terminates a real worker that cannot process messages", async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(navigator, "gpu", { value: {}, configurable: true }),
  );
  await page.route("**/models.json", (route) =>
    route.fulfill({
      json: [
        {
          size: "0.6b",
          manifest: "https://example.test/model.json",
          sha256: "0".repeat(64),
        },
      ],
    }),
  );
  await page.context().route("**/inference.worker-*.js", (route) => {
    return route.fulfill({
      path: new URL("./fixtures/unresponsive-worker.js", import.meta.url)
        .pathname,
      contentType: "text/javascript",
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Upload audio", exact: true }).click();
  await page
    .getByLabel("Reference recording", { exact: true })
    .setInputFiles(referenceWav());
  await page
    .getByLabel("Reference transcript")
    .fill("法律让人安心。規則への信頼。규칙에 대한 신뢰.");
  await page
    .getByLabel("What should your voice say?")
    .fill("法律让人安心。規則への信頼。규칙에 대한 신뢰.");
  for (let attempt = 0; attempt < 2; attempt++) {
    await page.getByRole("button", { name: "Generate speech" }).click();
    await expect(
      page.getByRole("status").filter({ hasText: "Generating speech" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(page.getByText("Cancelled", { exact: true })).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Generate speech" }),
    ).toBeEnabled();
    await expect(
      page.getByRole("button", { name: "Upload audio", exact: true }),
    ).toBeEnabled();
  }
});
