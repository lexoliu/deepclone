import { chromium } from "@playwright/test";
import { readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    cdp: { type: "string" },
    url: { type: "string" },
    iterations: { type: "string" },
    elements: { type: "string" },
    output: { type: "string" },
    timeout: { type: "string" },
  },
});
for (const key of [
  "cdp",
  "url",
  "iterations",
  "elements",
  "output",
  "timeout",
] as const)
  if (!values[key]) throw new Error(`Missing --${key}`);
const iterations = Number(values.iterations),
  elements = Number(values.elements),
  timeout = Number(values.timeout);
for (const count of [iterations, elements, timeout])
  if (!Number.isSafeInteger(count) || count <= 0)
    throw new Error("Counts and timeout must be positive integers");
const shader = await readFile(
  new URL("./readback.wgsl", import.meta.url),
  "utf8",
);
const browser = await chromium.connectOverCDP(values.cdp!, { timeout });
const page = await browser.contexts()[0].newPage();
page.setDefaultTimeout(timeout);
let deadline: ReturnType<typeof setTimeout> | undefined;
try {
  await page.goto(values.url!);
  const result = await Promise.race([
    page.evaluate(
      async ({ shader, iterations, elements }) => {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("A hardware adapter is required");
        const { GPUBufferUsage, GPUMapMode } =
          globalThis as typeof globalThis & {
            GPUBufferUsage: {
              STORAGE: number;
              COPY_SRC: number;
              COPY_DST: number;
              MAP_READ: number;
            };
            GPUMapMode: { READ: number };
          };
        const device = await adapter.requestDevice();
        const size = elements * Uint32Array.BYTES_PER_ELEMENT;
        const storage = device.createBuffer({
          size,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
          size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        try {
          const pipeline = await device.createComputePipelineAsync({
            layout: "auto",
            compute: { module: device.createShaderModule({ code: shader }) },
          });
          const bindings = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: storage } }],
          });
          const roundtripMs: number[] = [];
          for (let iteration = 0; iteration < iterations; iteration++) {
            const start = performance.now();
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindings);
            pass.dispatchWorkgroups(Math.ceil(elements / 64));
            pass.end();
            encoder.copyBufferToBuffer(storage, 0, readback, 0, size);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(GPUMapMode.READ);
            const output = new Uint32Array(readback.getMappedRange()).slice();
            readback.unmap();
            if (!output.every((value) => value === iteration + 1))
              throw new Error("Incorrect GPU result");
            roundtripMs.push(performance.now() - start);
          }
          const sorted = roundtripMs.toSorted((a, b) => a - b);
          return {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            elements,
            iterations,
            roundtripMs,
            medianMs: sorted[Math.floor(sorted.length / 2)],
            totalMs: roundtripMs.reduce((a, b) => a + b, 0),
          };
        } finally {
          storage.destroy();
          readback.destroy();
          device.destroy();
        }
      },
      { shader, iterations, elements },
    ),
    new Promise<never>((_resolve, reject) => {
      deadline = setTimeout(
        () => reject(new Error("GPU readback probe timed out")),
        timeout,
      );
    }),
  ]);
  clearTimeout(deadline);
  await writeFile(values.output!, JSON.stringify(result, null, 2));
  process.stdout.write(`${JSON.stringify(result)}\n`);
} finally {
  clearTimeout(deadline);
  await page.close();
  await browser.close();
}
