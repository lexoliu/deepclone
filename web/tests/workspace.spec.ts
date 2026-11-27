import { test, expect, type Page } from "@playwright/test";

const CJK_SAMPLE = "法律让人安心。規則への信頼。규칙에 대한 신뢰.";

function referenceWavBytes(): number[] {
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
  return Array.from(wav);
}

async function seedSavedVoice(
  page: Page,
  {
    language = "English",
    transcript = CJK_SAMPLE,
  }: { language?: string; transcript?: string } = {},
) {
  await page.route("**/__seed_voice__", (route) =>
    route.fulfill({
      contentType: "text/html",
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
      body: "<!doctype html><title>seed</title>",
    }),
  );
  await page.goto("/__seed_voice__");
  await page.evaluate(
    async ({ bytes, savedLanguage, savedTranscript }) => {
      const db = await new Promise<IDBDatabase>((resolve, reject) => {
        const request = indexedDB.open("deepclone-voices", 1);
        request.onupgradeneeded = () => {
          if (!request.result.objectStoreNames.contains("reference")) {
            request.result.createObjectStore("reference");
          }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () =>
          reject(request.error ?? new Error("IndexedDB open failed"));
      });
      try {
        await new Promise<void>((resolve, reject) => {
          const tx = db.transaction("reference", "readwrite");
          tx.objectStore("reference").put(
            {
              file: new File([new Uint8Array(bytes)], "reference.wav", {
                type: "audio/wav",
              }),
              transcript: savedTranscript,
              source: "record",
              language: savedLanguage,
            },
            "selected",
          );
          tx.oncomplete = () => resolve();
          tx.onerror = () =>
            reject(tx.error ?? new Error("IndexedDB write failed"));
        });
      } finally {
        db.close();
      }
    },
    {
      bytes: referenceWavBytes(),
      savedLanguage: language,
      savedTranscript: transcript,
    },
  );
  await page.unroute("**/__seed_voice__");
}

for (const stop of ["failure", "cancellation"] as const) {
  test(`${stop} preserves received audio and retry clears it`, async ({
    page,
  }) => {
    await seedSavedVoice(page);
    await page.addInitScript(() =>
      Object.defineProperty(navigator, "gpu", {
        value: {},
        configurable: true,
      }),
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
    await page.context().route("**/inference.worker-*.js", (route) =>
      route.fulfill({
        path: new URL("./fixtures/partial-worker.js", import.meta.url).pathname,
        contentType: "text/javascript",
        headers: {
          "Cross-Origin-Embedder-Policy": "require-corp",
          "Cross-Origin-Resource-Policy": "same-origin",
        },
      }),
    );
    await page.goto("/");
    await page
      .getByTestId("prompt-input")
      .fill(stop === "failure" ? "fail" : "continue");
    await page.getByTestId("synthesize-button").click();
    if (stop === "cancellation") {
      await expect(page.getByTestId("voice-status")).toContainText(
        "Generating and playing",
      );
      await page.getByTestId("cancel-button").click();
    } else {
      await expect(page.getByRole("alert")).toContainText(
        "audio generated so far",
      );
    }
    await expect(page.getByTestId("voice-status")).toContainText(
      /incomplete audio saved/i,
    );
    const download = page.getByTestId("download-wav");
    const samples = await download.evaluate(async (link) => {
      const bytes = await (
        await fetch((link as HTMLAnchorElement).href)
      ).arrayBuffer();
      const view = new DataView(bytes);
      return {
        channels: view.getUint16(22, true),
        length: view.getUint32(40, true),
        pcm: Array.from({ length: (bytes.byteLength - 44) / 2 }, (_, i) =>
          view.getInt16(44 + i * 2, true),
        ),
      };
    });
    expect(samples).toEqual({
      channels: 2,
      length: 16,
      pcm: [8192, 8192, -16384, -16384, 24575, 24575, -32768, -32768],
    });
    await page.getByTestId("prompt-input").fill("retry");
    await page.getByTestId("synthesize-button").click();
    await expect(download).toHaveCount(0);
    await expect(page.getByRole("alert")).toHaveCount(0);
    await page.getByTestId("cancel-button").click();
  });
}

test.beforeEach(async ({ page }) => {
  // UI tests never download real weights; GPU contract tests initialize explicitly.
  await page.route("**/models.json", (route) => route.fulfill({ json: [] }));
});

test("unsupported browsers receive a localized alert on the main screen and cannot synthesize", async ({
  page,
}) => {
  await seedSavedVoice(page, { language: "Chinese" });
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
  await expect(page.locator("html")).toHaveAttribute("lang", "zh");
  await expect(
    page.getByRole("heading", { name: "用你的声音说话" }),
  ).toBeVisible();
  await expect(page.getByRole("alert")).toContainText(
    "当前浏览器不支持本机硬件加速语音合成",
  );
  await page.getByTestId("prompt-input").fill(CJK_SAMPLE);
  await expect(page.getByTestId("synthesize-button")).toBeDisabled();
  await expect(page.getByTestId("re-setup")).toBeEnabled();
});

test("mobile-first onboarding, UDHR passages, localization, and narrow viewports stay overflow-free", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "languages", {
      value: ["zh-CN", "en-US"],
      configurable: true,
    });
  });
  await page.setViewportSize({ width: 320, height: 568 });
  await page.goto("/");

  await expect(
    page.getByRole("heading", { name: "你的母语是什么？" }),
  ).toBeVisible();
  await expect(page.getByTestId("lang-Chinese")).toHaveAttribute(
    "aria-checked",
    "true",
  );
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  for (const retiredLabel of [
    "Context tokens",
    "Maximum audio frames",
    "Frames per audio chunk",
    "Random seed",
    "Model",
    "Generation settings",
    "Upload audio",
    "Reference transcript",
    "Stream as I type",
  ]) {
    await expect(page.getByText(retiredLabel, { exact: true })).toHaveCount(0);
  }
  await expect(page.locator("body")).not.toContainText(/Qwen|0\.6B|1\.7B/);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);

  const expectedPassages: readonly [string, string, string][] = [
    ["Chinese", "zh", "人人生而自由，在尊严和权利上一律平等。"],
    ["Japanese", "ja", "すべての人間は、生れながらにして自由であり"],
    ["Korean", "ko", "모든 인간은 태어날 때부터 자유로우며"],
    ["English", "en", "All human beings are born free and equal"],
  ];

  for (const [language, bcp47, snippet] of expectedPassages) {
    await page.getByTestId(`lang-${language}`).click();
    await page.getByTestId("continue-step-2").click();
    const passage = page.getByTestId("udhr-passage");
    await expect(passage).toHaveAttribute("lang", bcp47);
    await expect(passage).toContainText(snippet);
    await expect(passage).not.toContainText(
      /清晨打开窗户|On a quiet morning|静かな朝に窓を開けると/,
    );
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.getByTestId("step-back").click();
  }

  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});

test("microphone denial surfaces an actionable alert and keeps recording retryable", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "languages", {
      value: ["en-US"],
      configurable: true,
    });
    navigator.mediaDevices.getUserMedia = async () => {
      throw new DOMException("Microphone permission denied", "NotAllowedError");
    };
  });
  await page.goto("/");
  await page.getByTestId("continue-step-2").click();
  await page.getByTestId("start-recording").click();
  await expect(page.getByRole("alert")).toContainText(
    "Microphone permission was denied",
  );
  await expect(page.getByTestId("start-recording")).toBeEnabled();
});

test("completed voice setup persists to main screen across reloads, supports re-setup cancel, replacement, and deletion", async ({
  browser,
}) => {
  const context = await browser.newContext({ locale: "en-US" });
  await context.route("**/models.json", (route) => route.fulfill({ json: [] }));
  const page = await context.newPage();
  await page.goto("/");

  await expect(page.getByTestId("cancel-setup")).toHaveCount(0);
  await expect(page.getByTestId("delete-voice")).toHaveCount(0);
  await page.getByTestId("lang-English").click();
  await page.getByTestId("continue-step-2").click();

  await page.getByTestId("start-recording").click();
  await expect(page.getByTestId("step-back")).toBeDisabled();
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: /Recording · (0\.[2-9]|[1-9])/ }),
  ).toBeVisible();
  await page.getByTestId("stop-recording").click();
  await expect(page.getByTestId("review-audio")).toBeVisible();
  await page.getByTestId("complete-setup").click();

  await expect(page.getByTestId("prompt-input")).toBeVisible();
  await expect(page.getByTestId("start-recording")).toHaveCount(0);
  await page.reload();
  await expect(page.getByTestId("prompt-input")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Speak in your voice" }),
  ).toBeVisible();

  await page.getByTestId("re-setup").click();
  await expect(page.getByTestId("cancel-setup")).toBeVisible();
  await page.getByTestId("cancel-setup").click();
  await expect(page.getByTestId("prompt-input")).toBeVisible();

  await page.getByTestId("re-setup").click();
  await page.getByTestId("lang-Japanese").click();
  await expect(
    page.getByRole("heading", { name: "母語を選んでください" }),
  ).toBeVisible();
  await page.getByTestId("continue-step-2").click();
  await expect(page.getByTestId("udhr-passage")).toHaveAttribute("lang", "ja");
  await page.getByTestId("start-recording").click();
  await expect(
    page.getByRole("status").filter({ hasText: /録音中 · (0\.[2-9]|[1-9])/ }),
  ).toBeVisible();
  await page.getByTestId("stop-recording").click();
  await page.getByTestId("complete-setup").click();

  await expect(
    page.getByRole("heading", { name: "あなたの声で話す" }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "あなたの声で話す" }),
  ).toBeVisible();

  await page.getByTestId("re-setup").click();
  await page.getByTestId("delete-voice").click();
  await expect(page.getByTestId("continue-step-2")).toBeVisible();
  await expect(page.getByTestId("cancel-setup")).toHaveCount(0);
  await expect(page.getByTestId("delete-voice")).toHaveCount(0);

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
  await expect(page.getByTestId("continue-step-2")).toBeVisible();
  await expect(page.getByTestId("prompt-input")).toHaveCount(0);
  await context.close();
});

test("published wasm glue and security headers honor the browser inference contract", async ({
  page,
}) => {
  const response = await page.goto("/");
  expect(response).not.toBeNull();
  const headers = response!.headers();
  expect(headers["content-security-policy"]).toContain(
    "script-src 'self' 'wasm-unsafe-eval'",
  );
  expect(headers["content-security-policy"]).toContain(
    "https://assets.deepclone.lexo.cool",
  );
  expect(headers["permissions-policy"]).toContain("microphone=(self)");
  expect(headers["x-frame-options"]).toBe("DENY");

  const exports = await page.evaluate(async () => {
    const manifestResponse = await fetch("/wasm.json");
    if (!manifestResponse.ok) throw new Error("Missing /wasm.json");
    const manifest = (await manifestResponse.json()) as { url: string };
    const url = "/wasm/deepclone.js";
    const module = await import(url);
    await module.default({ module_or_path: manifest.url });
    return [
      manifest.url,
      typeof module.BrowserRuntime.create,
      typeof module.TextInput,
      typeof module.BrowserVoice,
    ];
  });
  expect(exports).toEqual([
    "/wasm/deepclone_bg.wasm",
    "function",
    "function",
    "function",
  ]);
});

test("a GPU runtime remains reusable after a rejected model shard", async ({
  page,
}) => {
  await page.goto("/");
  const rejected = await page.evaluate(async () => {
    const url = "/wasm/deepclone.js";
    const module = await import(url);
    await module.default();
    const runtime = await module.BrowserRuntime.create("{}");
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

test("background preparation stays usable, reuses conditioning, re-conditions on re-setup, and terminates unresponsive workers", async ({
  page,
}) => {
  await seedSavedVoice(page);
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
        if (command.kind === "finish")
          this.onmessage?.(
            new MessageEvent("message", {
              data: {
                kind: "audio",
                id: 0,
                samples: new Float32Array([0.25]),
                sampleRate: 24000,
              },
            }),
          );
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
  await expect(page.getByTestId("prompt-input")).toBeEnabled();
  await expect(page.getByTestId("re-setup")).toBeEnabled();
  await expect(page.locator("body")).not.toContainText("Technical model label");

  await page.getByTestId("prompt-input").fill(CJK_SAMPLE);
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByRole("progressbar")).toHaveAttribute(
    "aria-valuenow",
    "50",
  );
  await page.getByTestId("cancel-button").click();
  expect(
    await page.evaluate(
      () => (window as unknown as { terminated: boolean }).terminated,
    ),
  ).toBe(true);
  await expect(page.getByTestId("voice-status")).toHaveText("Cancelled");
  await expect(page.getByTestId("synthesize-button")).toBeEnabled();
  await expect(page.getByRole("progressbar")).toHaveCount(0);

  await page.getByTestId("synthesize-button").click();
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
  await expect(page.getByTestId("download-wav")).toBeVisible();
  await expect(page.getByTestId("synthesize-button")).toBeEnabled();
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => (window as unknown as { referenceCount: number }).referenceCount,
    ),
  ).toBe(1);

  // Synthesizing again with the same saved voice reuses speaker conditioning.
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByTestId("download-wav")).toBeVisible();
  expect(
    await page.evaluate(
      () => (window as unknown as { referenceCount: number }).referenceCount,
    ),
  ).toBe(1);

  // Re-running setup and saving a new recording triggers fresh conditioning.
  await page.getByTestId("re-setup").click();
  await page.getByTestId("continue-step-2").click();
  await page.getByTestId("start-recording").click();
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: /Recording · (0\.[2-9]|[1-9])/ }),
  ).toBeVisible();
  await page.getByTestId("stop-recording").click();
  await page.getByTestId("complete-setup").click();
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByTestId("download-wav")).toBeVisible();
  expect(
    await page.evaluate(
      () => (window as unknown as { referenceCount: number }).referenceCount,
    ),
  ).toBe(2);
});

test("a crashed worker is discarded and the next request creates a fresh worker", async ({
  page,
}) => {
  await seedSavedVoice(page);
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
  await page.getByTestId("prompt-input").fill(CJK_SAMPLE);
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByRole("alert")).toHaveText(
    "Speech could not be prepared. Please try again.",
  );
  await expect(page.getByTestId("synthesize-button")).toBeEnabled();
  expect(created).toBe(1);
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByTestId("voice-status")).toHaveText(
    "Generating speech",
  );
  expect(created).toBe(2);
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.getByTestId("cancel-button").click();
  await expect(page.getByTestId("voice-status")).toHaveText("Cancelled");
});

test("cancel terminates a real worker that cannot process messages", async ({
  page,
}) => {
  await seedSavedVoice(page);
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
  await page.context().route("**/inference.worker-*.js", (route) =>
    route.fulfill({
      path: new URL("./fixtures/unresponsive-worker.js", import.meta.url)
        .pathname,
      contentType: "text/javascript",
      headers: {
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Resource-Policy": "same-origin",
      },
    }),
  );
  await page.goto("/");
  await page.getByTestId("prompt-input").fill(CJK_SAMPLE);
  for (let attempt = 0; attempt < 2; attempt++) {
    await page.getByTestId("synthesize-button").click();
    await expect(page.getByTestId("voice-status")).toHaveText(
      "Generating speech",
    );
    await page.getByTestId("cancel-button").click();
    await expect(page.getByTestId("voice-status")).toHaveText("Cancelled");
    await expect(page.getByTestId("synthesize-button")).toBeEnabled();
    await expect(page.getByTestId("re-setup")).toBeEnabled();
  }
});
