import { chromium } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    cdp: { type: "string" },
    url: { type: "string" },
    shader: { type: "string" },
    output: { type: "string" },
    timeout: { type: "string" },
  },
});
for (const key of ["cdp", "url", "shader", "output", "timeout"] as const)
  if (!values[key]) throw new Error(`Missing --${key}`);
const timeout = Number(values.timeout);
if (!Number.isSafeInteger(timeout) || timeout <= 0)
  throw new Error("Timeout must be a positive integer");
const shader: { code: string; features: GPUFeatureName[] } = JSON.parse(
  await readFile(values.shader!, "utf8"),
);
const browser = await chromium.connectOverCDP(values.cdp!, { timeout });
const page = await browser.contexts()[0].newPage();
page.setDefaultTimeout(timeout);
let deadline: ReturnType<typeof setTimeout> | undefined;
try {
  await page.goto(values.url!);
  const result = await Promise.race([
    page.evaluate(async (shader) => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("A WebGPU adapter is required");
      const device = await adapter.requestDevice({
        requiredFeatures: shader.features,
      });
      try {
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code: shader.code });
        const [{ messages }, validation] = await Promise.all([
          module.getCompilationInfo(),
          device.popErrorScope(),
        ]);
        return {
          vendor: adapter.info.vendor,
          architecture: adapter.info.architecture,
          features: Array.from(device.features),
          messages: messages.map(({ message, type, lineNum, linePos }) => ({
            message,
            type,
            lineNum,
            linePos,
          })),
          validation: validation?.message ?? null,
        };
      } finally {
        device.destroy();
      }
    }, shader),
    new Promise<never>((_, reject) => {
      deadline = setTimeout(
        () => reject(new Error("Shader validation timed out")),
        timeout,
      );
    }),
  ]);
  await writeFile(values.output!, JSON.stringify(result, null, 2));
  process.stdout.write(`${JSON.stringify(result)}\n`);
} finally {
  clearTimeout(deadline);
  await page.close();
  await browser.close();
}
