import { test, expect, spyOn } from "bun:test";
import { asset } from "./assets";

async function withDownload(
  chunks: Uint8Array[],
  size: number,
  run: (
    read: () => Promise<Uint8Array>,
    progress: number[],
    writes: Uint8Array[],
  ) => Promise<void>,
) {
  const contents = new Uint8Array(chunks.flatMap((chunk) => Array.from(chunk)));
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", contents)),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
  const writes: Uint8Array[] = [];
  const progress: number[] = [];
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "caches");
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    value: {
      open: async () => ({
        match: async () => undefined,
        put: async (_url: string, response: Response) => {
          writes.push(new Uint8Array(await response.arrayBuffer()));
        },
        delete: async () => true,
      }),
    },
  });
  const fetch = spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(
      new ReadableStream({
        start(controller) {
          for (const chunk of chunks) controller.enqueue(chunk);
          controller.close();
        },
      }),
    ),
  );
  try {
    await run(
      () =>
        asset(
          {
            size: "0.6b",
            manifest: "https://example.test/manifest.json",
            sha256: hash,
          },
          { path: "weights.safetensors", size, sha256: hash },
          new AbortController().signal,
          (bytes) => progress.push(bytes),
        ),
      progress,
      writes,
    );
  } finally {
    fetch.mockRestore();
    if (descriptor) Object.defineProperty(globalThis, "caches", descriptor);
    else Reflect.deleteProperty(globalThis, "caches");
  }
}

test("download reports partial byte progress and caches only verified content", async () => {
  await withDownload(
    [new Uint8Array([1, 2]), new Uint8Array([3, 4])],
    4,
    async (read, progress, writes) => {
      expect(await read()).toEqual(new Uint8Array([1, 2, 3, 4]));
      expect(progress).toContain(2);
      expect(progress.at(-1)).toBe(4);
      expect(writes).toEqual([new Uint8Array([1, 2, 3, 4])]);
    },
  );
});

test("incomplete and oversized downloads never enter the cache", async () => {
  for (const size of [2, 4]) {
    await withDownload(
      [new Uint8Array([1, 2, 3])],
      size,
      async (read, _progress, writes) => {
        await expect(read()).rejects.toThrow(
          "Could not download file: weights.safetensors",
        );
        expect(writes).toHaveLength(0);
      },
    );
  }
});
