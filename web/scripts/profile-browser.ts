import { chromium } from "@playwright/test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    cdp: { type: "string" },
    url: { type: "string" },
    reference: { type: "string" },
    transcript: { type: "string" },
    text: { type: "string" },
    language: { type: "string" },
    output: { type: "string" },
    runs: { type: "string" },
    timeout: { type: "string" },
    profile: { type: "boolean", default: false },
  },
});
for (const key of [
  "cdp",
  "url",
  "reference",
  "transcript",
  "text",
  "language",
  "output",
  "runs",
  "timeout",
] as const) {
  if (!values[key]) throw new Error(`Missing --${key}`);
}
const timeout = Number(values.timeout);
const runs = Number(values.runs);
if (
  !Number.isFinite(timeout) ||
  timeout <= 0 ||
  !Number.isSafeInteger(runs) ||
  runs <= 0
)
  throw new Error("Invalid timeout or run count");
await mkdir(values.output!, { recursive: true });
const browser = await chromium.connectOverCDP(values.cdp!);
const page = await browser.contexts()[0].newPage();
page.setDefaultTimeout(timeout);
const measurements: unknown[] = [];
const profiler = values.profile ? await browser.newBrowserCDPSession() : null;
let profileSession: string | undefined;
let commandId = 0;
const replies = new Map<
  number,
  { resolve(value: unknown): void; reject(reason: Error): void }
>();
profiler?.on("Target.receivedMessageFromTarget", ({ sessionId, message }) => {
  if (sessionId !== profileSession) return;
  const response = JSON.parse(message);
  const pending = replies.get(response.id);
  if (!pending) return;
  replies.delete(response.id);
  if (response.error) pending.reject(new Error(response.error.message));
  else pending.resolve(response.result);
});
async function profileCommand(method: string) {
  const id = ++commandId;
  const response = new Promise<unknown>((resolve, reject) => {
    replies.set(id, { resolve, reject });
  });
  const deadline = setTimeout(() => {
    replies.get(id)?.reject(new Error(`Worker profiler timed out: ${method}`));
    replies.delete(id);
  }, 10000);
  try {
    await profiler!.send("Target.sendMessageToTarget", {
      sessionId: profileSession,
      message: JSON.stringify({ id, method }),
    });
    return await response;
  } finally {
    clearTimeout(deadline);
    replies.delete(id);
  }
}
page.on("console", (event) => {
  if (event.type() === "error") process.stderr.write(`${event.text()}\n`);
  if (event.text().startsWith("Speech codec timing"))
    process.stdout.write(`${JSON.stringify({ codec: event.text() })}\n`);
});
try {
  await page.goto(values.url!);
  const adapter = await page.evaluate(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("A WebGPU adapter is required for profiling");
    const { info } = adapter;
    return {
      vendor: info.vendor,
      architecture: info.architecture,
      fallback: info.isFallbackAdapter,
    };
  });
  await page.getByRole("button", { name: "Upload audio", exact: true }).click();
  await page
    .getByLabel("Reference recording", { exact: true })
    .setInputFiles(values.reference!);
  const transcript: { text: string } = JSON.parse(
    await readFile(values.transcript!, "utf8"),
  );
  await page.getByLabel("Reference transcript").fill(transcript.text);
  await page.getByLabel("What should your voice say?").fill(values.text!);
  await page
    .getByLabel("Language", { exact: true })
    .selectOption(values.language!);
  if (profiler) {
    const { targetInfos } = await profiler.send("Target.getTargets");
    const target = targetInfos.find(
      (target) =>
        target.type === "worker" && target.url.includes("inference.worker"),
    );
    if (!target)
      throw new Error("The inference worker is unavailable for profiling");
    const attached = await profiler.send("Target.attachToTarget", {
      targetId: target.targetId,
      flatten: false,
    });
    profileSession = attached.sessionId;
    await profileCommand("Profiler.enable");
    await profileCommand("Profiler.start");
  }
  for (let run = 1; run <= runs; run++) {
    const measured = page.waitForEvent("console", {
      predicate: (event) => event.text().startsWith("Speech generation timing"),
      timeout,
    });
    await page.getByRole("button", { name: "Generate speech" }).click();
    const failed = page
      .getByRole("alert")
      .waitFor({ state: "visible", timeout })
      .then(async () => {
        throw new Error(await page.getByRole("alert").innerText());
      });
    const event = await Promise.race([measured, failed]);
    const timing = await event.args()[1].jsonValue();
    // Persist timing before transferring audio, including when the remote browser closes.
    await writeFile(
      join(values.output!, `run-${run}-timing.json`),
      JSON.stringify(timing, null, 2),
    );
    const download = page.getByRole("link", { name: "Download WAV" });
    await download.waitFor({ state: "visible", timeout });
    const audio = await download.evaluate(async (link: HTMLAnchorElement) => {
      const response = await fetch(link.href);
      if (!response.ok)
        throw new Error(`Audio fetch failed: ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      const context = new AudioContext();
      try {
        const decoded = await context.decodeAudioData(bytes.buffer.slice(0));
        return { bytes: Array.from(bytes), seconds: decoded.duration };
      } finally {
        await context.close();
      }
    });
    await writeFile(
      join(values.output!, `run-${run}.wav`),
      Uint8Array.from(audio.bytes),
    );
    const seconds = audio.seconds;
    const result = {
      run,
      adapter,
      browser: browser.version(),
      timing,
      audioSeconds: seconds,
    };
    measurements.push(result);
    await writeFile(
      join(values.output!, "timings.json"),
      JSON.stringify(measurements, null, 2),
    );
    process.stdout.write(`${JSON.stringify(result)}\n`);
  }
} catch (reason) {
  process.stderr.write(
    `${reason instanceof Error ? reason.stack : String(reason)}\n`,
  );
  throw reason;
} finally {
  try {
    if (profileSession) {
      const result = await profileCommand("Profiler.stop");
      await writeFile(
        join(values.output!, "worker-profile.json"),
        JSON.stringify(result),
      );
      await profiler!.send("Target.detachFromTarget", {
        sessionId: profileSession,
      });
    }
  } finally {
    await page.close();
    await browser.close();
  }
}
