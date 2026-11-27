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

  // Desktop viewport expands to a wide two-column layout without pinning CTAs to the bottom edge.
  await page.setViewportSize({ width: 1440, height: 900 });
  const headerBox = await page.getByTestId("app-header").boundingBox();
  const headingBox = await page.getByRole("heading").boundingBox();
  const radioGroupBox = await page.getByRole("radiogroup").boundingBox();
  const continueBox = await page.getByTestId("continue-step-2").boundingBox();
  expect(headerBox).not.toBeNull();
  expect(headingBox).not.toBeNull();
  expect(radioGroupBox).not.toBeNull();
  expect(continueBox).not.toBeNull();
  expect(headerBox!.width).toBeGreaterThan(950);
  expect(radioGroupBox!.x).toBeGreaterThan(headingBox!.x + headingBox!.width);
  expect(continueBox!.y + continueBox!.height).toBeLessThan(700);
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
    const manifestResponse = await fetch("/wasm.json", { cache: "no-cache" });
    if (!manifestResponse.ok) throw new Error("Missing /wasm.json");
    const manifest = (await manifestResponse.json()) as {
      js: string;
      url: string;
    };
    const module = await import(manifest.js);
    await module.default({ module_or_path: manifest.url });
    const hasher = new module.Sha256Hasher();
    hasher.update(new Uint8Array([97, 98]));
    hasher.update(new Uint8Array([99]));
    const abcDigest = hasher.finalize();
    return [
      manifest.js,
      manifest.url,
      typeof module.BrowserRuntime.create,
      typeof module.BrowserRuntime.prototype.load_conditioning_streaming,
      typeof module.BrowserRuntime.prototype.load_synthesis_streaming,
      typeof module.BrowserRuntime.prototype.reclaim_memory,
      typeof module.TextInput,
      typeof module.BrowserVoice,
      typeof module.BrowserVoice.prototype.set_reference_conditioning,
      typeof module.BrowserVoice.prototype.required_text_tokens,
      typeof module.BrowserVoice.prototype.set_text_embeddings,
      typeof module.BrowserConditioner,
      abcDigest,
    ];
  });
  expect(exports).toEqual([
    "/wasm/deepclone.js",
    "/wasm/deepclone_bg.wasm",
    "function",
    "function",
    "function",
    "function",
    "function",
    "function",
    "function",
    "function",
    "function",
    "function",
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  ]);
});

test("a GPU runtime remains reusable after a rejected model shard", async ({
  page,
}) => {
  await page.goto("/");
  const rejected = await page.evaluate(async () => {
    const manifestResponse = await fetch("/wasm.json", { cache: "no-cache" });
    if (!manifestResponse.ok) throw new Error("Missing /wasm.json");
    const manifest = (await manifestResponse.json()) as {
      js: string;
      url: string;
    };
    const module = await import(manifest.js);
    await module.default({ module_or_path: manifest.url });
    const runtime = await module.BrowserRuntime.create("{}");
    let firstRejected = false;
    try {
      runtime.load_conditioning_streaming(
        "not-json",
        () => {},
        "{}",
        "{}",
        "{}",
      );
    } catch {
      firstRejected = true;
    }
    runtime.reclaim_memory();
    let secondRejected = false;
    try {
      runtime.load_synthesis_streaming(
        true,
        "not-json",
        () => {},
        "{}",
        "{}",
        "{}",
        new Uint8Array(0),
        2048,
        0n,
      );
    } catch {
      secondRejected = true;
    }
    runtime.free();
    return firstRejected && secondRejected;
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
          const refCommand = command as {
            conditioning?: { modelSha256: string };
          };
          Object.assign(window, {
            referenceCount:
              ((window as unknown as { referenceCount?: number })
                .referenceCount ?? 0) + 1,
            lastReceivedConditioning:
              refCommand.conditioning?.modelSha256 ?? null,
          });
        }
        const event =
          command.kind === "reference"
            ? {
                kind: "conditioned",
                conditioning: {
                  modelSha256: "0".repeat(64),
                  audioVersion: 3,
                  speaker: new Float32Array([0.5]),
                  speakerShape: [1, 1, 1],
                  codes: new Uint32Array([1, 2]),
                  groups: 2,
                },
              }
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

  // Reloading restores cached conditioning from IndexedDB and forwards it to the worker.
  await page.reload();
  await page.waitForFunction(
    () =>
      (window as unknown as { preparationStarted: boolean }).preparationStarted,
  );
  await page.evaluate(() =>
    (
      window as unknown as { completePreparation: () => void }
    ).completePreparation(),
  );
  await page.getByTestId("prompt-input").fill(CJK_SAMPLE);
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByTestId("download-wav")).toBeVisible();
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { lastReceivedConditioning: string | null })
          .lastReceivedConditioning,
    ),
  ).toBe("0".repeat(64));

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
  expect(
    await page.evaluate(
      () =>
        (window as unknown as { lastReceivedConditioning: string | null })
          .lastReceivedConditioning,
    ),
  ).toBeNull();
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

test("legacy cached conditioning is ignored and quiet stereo-with-silent-channel recordings are normalized", async ({
  page,
}) => {
  // Build a 48 kHz 2-channel WAV with quiet speech-level sine on ch0 (~-40 dBFS) and silence on ch1.
  const rate = 48000;
  const frames = 24000;
  const wav = Buffer.alloc(44 + frames * 4);
  wav.write("RIFF");
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(rate, 24);
  wav.writeUInt32LE(rate * 4, 28);
  wav.writeUInt16LE(4, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(frames * 4, 40);
  for (let i = 0; i < frames; i++) {
    const sample = Math.round(
      0.014 * Math.sin((2 * Math.PI * 220 * i) / rate) * 32767,
    );
    wav.writeInt16LE(sample, 44 + i * 4);
    wav.writeInt16LE(0, 44 + i * 4 + 2);
  }

  await page.route("**/__seed_legacy__", (route) =>
    route.fulfill({
      contentType: "text/html",
      headers: {
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
      body: "<!doctype html><title>seed</title>",
    }),
  );
  await page.goto("/__seed_legacy__");
  await page.evaluate(
    async ({ bytes, transcript }) => {
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
              file: new File([new Uint8Array(bytes)], "legacy.wav", {
                type: "audio/wav",
              }),
              transcript,
              source: "record",
              language: "Chinese",
              // Legacy conditioning from an older audioVersion must trigger re-extraction.
              conditioning: {
                modelSha256: "0".repeat(64),
                audioVersion: 2,
                speaker: new Float32Array([0.1]),
                speakerShape: [1, 1, 1],
                codes: new Uint32Array([1, 2]),
                groups: 2,
              },
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
    { bytes: Array.from(wav), transcript: CJK_SAMPLE },
  );
  await page.unroute("**/__seed_legacy__");

  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", { value: {}, configurable: true });
    class RecordingWorker {
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror = null;
      postMessage(command: {
        kind: string;
        id?: number;
        skipVoice?: boolean;
        samples?: Float32Array;
        conditioning?: { modelSha256: string; audioVersion?: number };
      }) {
        if (command.kind === "load") {
          Object.assign(window, { initialSkipVoice: command.skipVoice });
          this.onmessage?.(
            new MessageEvent("message", { data: { kind: "ready" } }),
          );
          return;
        }
        if (command.kind === "reference") {
          const samples = command.samples ?? new Float32Array(0);
          let sumSq = 0;
          let max = 0;
          for (let i = 0; i < samples.length; i++) {
            const v = samples[i]!;
            sumSq += v * v;
            const abs = Math.abs(v);
            if (abs > max) max = abs;
          }
          Object.assign(window, {
            receivedConditioning: command.conditioning ?? null,
            receivedSampleCount: samples.length,
            receivedRms: Math.sqrt(sumSq / Math.max(1, samples.length)),
            receivedPeak: max,
          });
          this.onmessage?.(
            new MessageEvent("message", {
              data: {
                kind: "conditioned",
                conditioning: {
                  modelSha256: "0".repeat(64),
                  audioVersion: 3,
                  speaker: new Float32Array([0.5]),
                  speakerShape: [1, 1, 1],
                  codes: new Uint32Array([1, 2]),
                  groups: 2,
                },
              },
            }),
          );
          return;
        }
        if (command.kind === "start") {
          Object.assign(window, {
            receivedLanguage:
              (command as { language?: string }).language ?? null,
          });
          this.onmessage?.(
            new MessageEvent("message", { data: { kind: "started" } }),
          );
          return;
        }
        if (command.kind === "text") {
          this.onmessage?.(
            new MessageEvent("message", {
              data: { kind: "accepted", id: command.id },
            }),
          );
          return;
        }
        if (command.kind === "finish") {
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
          this.onmessage?.(
            new MessageEvent("message", { data: { kind: "done" } }),
          );
        }
      }
      terminate() {}
    }
    Object.defineProperty(window, "Worker", { value: RecordingWorker });
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
  await page.getByTestId("prompt-input").fill("你好很高兴见到你，这是一个测试");
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByTestId("download-wav")).toBeVisible();

  const observed = await page.evaluate(() => {
    const w = window as unknown as {
      initialSkipVoice?: boolean;
      receivedConditioning: unknown;
      receivedSampleCount: number;
      receivedRms: number;
      receivedPeak: number;
      receivedLanguage?: string | null;
    };
    return {
      initialSkipVoice: w.initialSkipVoice,
      receivedConditioning: w.receivedConditioning,
      receivedSampleCount: w.receivedSampleCount,
      receivedRms: w.receivedRms,
      receivedPeak: w.receivedPeak,
      receivedLanguage: w.receivedLanguage ?? null,
    };
  });
  expect(observed.initialSkipVoice).toBe(true);
  expect(observed.receivedConditioning).toBeNull();
  expect(observed.receivedSampleCount).toBe(12000);
  expect(observed.receivedRms).toBeGreaterThan(0.04);
  expect(observed.receivedRms).toBeLessThan(0.06);
  expect(observed.receivedPeak).toBeLessThanOrEqual(0.895);
  expect(observed.receivedLanguage).toBe("Chinese");
});

test("generated audio history persists across reloads and supports downloading, single delete, and clear all", async ({
  page,
}) => {
  await seedSavedVoice(page, { language: "Chinese" });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", { value: {}, configurable: true });
    let runCount = 0;
    class HistoryWorker {
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror = null;
      postMessage(command: { kind: string; id?: number }) {
        if (command.kind === "load") {
          this.onmessage?.(
            new MessageEvent("message", { data: { kind: "ready" } }),
          );
          return;
        }
        if (command.kind === "reference") {
          this.onmessage?.(
            new MessageEvent("message", {
              data: {
                kind: "conditioned",
                conditioning: {
                  modelSha256: "0".repeat(64),
                  audioVersion: 3,
                  speaker: new Float32Array([0.5]),
                  speakerShape: [1, 1, 1],
                  codes: new Uint32Array([1, 2]),
                  groups: 2,
                },
              },
            }),
          );
          return;
        }
        if (command.kind === "start") {
          Object.assign(window, {
            receivedLanguage:
              (command as { language?: string }).language ?? null,
          });
          this.onmessage?.(
            new MessageEvent("message", { data: { kind: "started" } }),
          );
          return;
        }
        if (command.kind === "text") {
          this.onmessage?.(
            new MessageEvent("message", {
              data: { kind: "accepted", id: command.id },
            }),
          );
          return;
        }
        if (command.kind === "finish") {
          runCount++;
          const samples = new Float32Array(2400 * runCount).fill(
            runCount === 1 ? 0.25 : -0.25,
          );
          this.onmessage?.(
            new MessageEvent("message", {
              data: {
                kind: "audio",
                id: runCount,
                samples,
                sampleRate: 24000,
              },
            }),
          );
          this.onmessage?.(
            new MessageEvent("message", { data: { kind: "done" } }),
          );
        }
      }
      terminate() {}
    }
    Object.defineProperty(window, "Worker", { value: HistoryWorker });
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

  await page.setViewportSize({ width: 320, height: 640 });
  await page.goto("/");
  await expect(page.getByTestId("history-section")).toHaveCount(0);

  await page.getByTestId("prompt-input").fill(CJK_SAMPLE);
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByTestId("download-wav")).toBeVisible();
  await expect(page.getByTestId("history-item")).toHaveCount(1);
  await expect(page.getByTestId("history-item-text").first()).toHaveText(
    CJK_SAMPLE,
  );

  const secondPrompt = "你好，这是第二条测试。";
  await page.getByTestId("prompt-input").fill(secondPrompt);
  await page.getByTestId("synthesize-button").click();
  await expect(page.getByTestId("history-item")).toHaveCount(2);
  await expect(page.getByTestId("history-item-text").nth(0)).toHaveText(
    secondPrompt,
  );
  await expect(page.getByTestId("history-item-text").nth(1)).toHaveText(
    CJK_SAMPLE,
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);

  // Reloading restores both history items with playable and downloadable WAV blobs.
  await page.reload();
  await expect(page.getByTestId("history-item")).toHaveCount(2);
  await expect(page.getByTestId("history-item-text").nth(0)).toHaveText(
    secondPrompt,
  );
  const wavHeader = await page
    .getByTestId("history-download")
    .first()
    .evaluate(async (link) => {
      const buf = await (
        await fetch((link as HTMLAnchorElement).href)
      ).arrayBuffer();
      const bytes = new Uint8Array(buf);
      return {
        riff: String.fromCharCode(...bytes.slice(0, 4)),
        wave: String.fromCharCode(...bytes.slice(8, 12)),
        size: bytes.byteLength,
      };
    });
  expect(wavHeader.riff).toBe("RIFF");
  expect(wavHeader.wave).toBe("WAVE");
  expect(wavHeader.size).toBeGreaterThan(44);

  // Deleting one item removes only that entry and persists across reload.
  await page.getByTestId("history-delete").first().click();
  await expect(page.getByTestId("history-item")).toHaveCount(1);
  await expect(page.getByTestId("history-item-text").first()).toHaveText(
    CJK_SAMPLE,
  );
  await page.reload();
  await expect(page.getByTestId("history-item")).toHaveCount(1);
  await expect(page.getByTestId("history-item-text").first()).toHaveText(
    CJK_SAMPLE,
  );

  // Clearing all removes remaining entries and hides the history section.
  await page.getByTestId("history-clear").click();
  await expect(page.getByTestId("history-section")).toHaveCount(0);
  await page.reload();
  await expect(page.getByTestId("history-section")).toHaveCount(0);
});

test("reference recorder requests stereo with noise suppression and AGC and saves lossless WAV", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(
      navigator.mediaDevices,
    );
    navigator.mediaDevices.getUserMedia = async (
      constraints?: MediaStreamConstraints,
    ) => {
      Object.assign(window, { capturedConstraints: constraints });
      return original(constraints);
    };
  });
  await page.goto("/");
  await page.getByTestId("lang-Chinese").click();
  await page.getByTestId("continue-step-2").click();
  await page.getByTestId("start-recording").click();
  await expect(
    page.getByRole("status").filter({ hasText: /录音中 · (0\.[3-9]|[1-9])/ }),
  ).toBeVisible();
  await page.getByTestId("stop-recording").click();
  await expect(page.getByTestId("review-audio")).toBeVisible();
  await page.getByTestId("complete-setup").click();
  await expect(page.getByTestId("prompt-input")).toBeVisible();

  const recorded = await page.evaluate(async () => {
    const constraints = (
      window as unknown as { capturedConstraints?: MediaStreamConstraints }
    ).capturedConstraints;
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open("deepclone-voices");
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    try {
      const saved = await new Promise<{ file: File }>((resolve, reject) => {
        const req = db
          .transaction("reference", "readonly")
          .objectStore("reference")
          .get("selected");
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      const buf = await saved.file.arrayBuffer();
      const bytes = new Uint8Array(buf);
      const view = new DataView(buf);
      return {
        audioConstraints: constraints?.audio,
        name: saved.file.name,
        type: saved.file.type,
        size: saved.file.size,
        riff: String.fromCharCode(...bytes.slice(0, 4)),
        wave: String.fromCharCode(...bytes.slice(8, 12)),
        channels: view.getUint16(22, true),
        sampleRate: view.getUint32(24, true),
      };
    } finally {
      db.close();
    }
  });

  expect(recorded.audioConstraints).toEqual({
    channelCount: { ideal: 2 },
    noiseSuppression: true,
    autoGainControl: true,
    echoCancellation: false,
  });
  expect(recorded.name).toBe("my-voice.wav");
  expect(recorded.type).toBe("audio/wav");
  expect(recorded.riff).toBe("RIFF");
  expect(recorded.wave).toBe("WAVE");
  expect(recorded.channels).toBeGreaterThanOrEqual(1);
  expect(recorded.channels).toBeLessThanOrEqual(2);
  expect(recorded.size).toBeGreaterThan(4096);
});
