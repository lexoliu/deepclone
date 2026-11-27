import { test, expect, spyOn } from "bun:test";
import { createHash } from "node:crypto";
import {
  BytesStoredFile,
  TEXT_EMBEDDING_TENSOR,
  asset,
  ensureStoredAsset,
  findTextEmbedding,
  openModelStorage,
  readTextEmbeddingRows,
  uploadShardTensors,
  type OpfsDirectoryHandleLike,
  type OpfsFileHandleLike,
  type StreamingHasher,
  type SyncAccessHandleLike,
} from "./assets";

function buildSafetensors(
  entries: {
    name: string;
    dtype: string;
    shape: number[];
    bytes: Uint8Array;
  }[],
): Uint8Array {
  const header: Record<string, unknown> = {
    __metadata__: { format: "pt" },
  };
  let offset = 0;
  for (const entry of entries) {
    header[entry.name] = {
      dtype: entry.dtype,
      shape: entry.shape,
      data_offsets: [offset, offset + entry.bytes.byteLength],
    };
    offset += entry.bytes.byteLength;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + headerBytes.byteLength + offset);
  new DataView(out.buffer).setBigUint64(
    0,
    BigInt(headerBytes.byteLength),
    true,
  );
  out.set(headerBytes, 8);
  let cursor = 8 + headerBytes.byteLength;
  for (const entry of entries) {
    out.set(entry.bytes, cursor);
    cursor += entry.bytes.byteLength;
  }
  return out;
}

class MemoryOpfsDirectory implements OpfsDirectoryHandleLike {
  readonly dirs = new Map<string, MemoryOpfsDirectory>();
  readonly files = new Map<string, Uint8Array>();

  async getDirectoryHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OpfsDirectoryHandleLike> {
    let dir = this.dirs.get(name);
    if (!dir) {
      if (!options?.create) throw new Error(`Missing dir ${name}`);
      dir = new MemoryOpfsDirectory();
      this.dirs.set(name, dir);
    }
    return dir;
  }

  async getFileHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OpfsFileHandleLike> {
    if (!this.files.has(name)) {
      if (!options?.create) throw new Error(`Missing file ${name}`);
      this.files.set(name, new Uint8Array(0));
    }
    return {
      getFile: async () =>
        new Blob([
          (this.files.get(name) ??
            new Uint8Array(0)) as Uint8Array<ArrayBuffer>,
        ]),
      createSyncAccessHandle: async (): Promise<SyncAccessHandleLike> => {
        let open = true;
        return {
          getSize: () => this.files.get(name)?.byteLength ?? 0,
          read: (buffer, opts) => {
            if (!open) throw new Error("closed");
            const data = this.files.get(name) ?? new Uint8Array(0);
            const at = opts?.at ?? 0;
            const out = new Uint8Array(
              buffer.buffer,
              buffer.byteOffset,
              buffer.byteLength,
            );
            const count = Math.max(
              0,
              Math.min(out.byteLength, data.byteLength - at),
            );
            out.set(data.subarray(at, at + count));
            return count;
          },
          write: (buffer, opts) => {
            if (!open) throw new Error("closed");
            const data = this.files.get(name) ?? new Uint8Array(0);
            const at = opts?.at ?? 0;
            const src = new Uint8Array(
              buffer.buffer,
              buffer.byteOffset,
              buffer.byteLength,
            );
            const nextLen = Math.max(data.byteLength, at + src.byteLength);
            const next = new Uint8Array(nextLen);
            next.set(data);
            next.set(src, at);
            this.files.set(name, next);
            return src.byteLength;
          },
          truncate: (newSize) => {
            if (!open) throw new Error("closed");
            const data = this.files.get(name) ?? new Uint8Array(0);
            const next = new Uint8Array(newSize);
            next.set(data.subarray(0, Math.min(newSize, data.byteLength)));
            this.files.set(name, next);
          },
          flush: () => {},
          close: () => {
            open = false;
          },
        };
      },
    };
  }

  async removeEntry(name: string): Promise<void> {
    this.files.delete(name);
    this.dirs.delete(name);
  }

  async *entries(): AsyncIterableIterator<[string, { kind: string }]> {
    for (const key of this.dirs.keys()) yield [key, { kind: "directory" }];
    for (const key of this.files.keys()) yield [key, { kind: "file" }];
  }
}

function nodeSha256Hasher(): StreamingHasher {
  const hash = createHash("sha256");
  return {
    update: (chunk) => {
      hash.update(chunk);
    },
    finalize: () => hash.digest("hex"),
    free: () => {},
  };
}

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

test("safetensors streaming skips text embeddings and slices exact token rows", async () => {
  const embeddingBytes = new Uint8Array(4 * 3 * 2);
  for (let i = 0; i < embeddingBytes.length; i++) embeddingBytes[i] = i + 1;
  const smallBytes = new Uint8Array([10, 20, 30, 40]);
  const shardBytes = buildSafetensors([
    {
      name: TEXT_EMBEDDING_TENSOR,
      dtype: "F16",
      shape: [4, 3],
      bytes: embeddingBytes,
    },
    {
      name: "talker.model.norm.weight",
      dtype: "F16",
      shape: [2],
      bytes: smallBytes,
    },
  ]);
  const stored = new BytesStoredFile("weights-0012.safetensors", shardBytes);
  const uploaded: {
    name: string;
    dtype: string;
    shape: number[];
    bytes: number[];
    conditioning: boolean;
  }[] = [];
  await uploadShardTensors(
    stored,
    false,
    new Set([TEXT_EMBEDDING_TENSOR]),
    (name, dtype, shape, bytes, conditioning) => {
      uploaded.push({
        name,
        dtype,
        shape: Array.from(shape),
        bytes: Array.from(bytes),
        conditioning,
      });
    },
  );
  expect(uploaded).toEqual([
    {
      name: "talker.model.norm.weight",
      dtype: "F16",
      shape: [2],
      bytes: [10, 20, 30, 40],
      conditioning: false,
    },
  ]);

  const descriptor = await findTextEmbedding(stored);
  expect(descriptor).not.toBeNull();
  expect(descriptor!.vocabSize).toBe(4);
  expect(descriptor!.width).toBe(3);
  const sliced = await readTextEmbeddingRows(
    descriptor!,
    new Uint32Array([3, 1]),
  );
  expect(Array.from(sliced)).toEqual([
    19, 20, 21, 22, 23, 24, 7, 8, 9, 10, 11, 12,
  ]);
  await expect(
    readTextEmbeddingRows(descriptor!, new Uint32Array([4])),
  ).rejects.toThrow("exceeds text embedding vocabulary");
});

test("OPFS streaming verifies hashes, reuses verified files without refetch, and purges failed downloads", async () => {
  const root = new MemoryOpfsDirectory();
  const staleModel = await root.getDirectoryHandle(
    "deepclone-verified-models-v2",
    { create: true },
  );
  await staleModel.getDirectoryHandle("old-revision", { create: true });

  const payload = new Uint8Array([5, 6, 7, 8, 9, 10]);
  const sha256 = createHash("sha256").update(payload).digest("hex");
  const entry = {
    size: "0.6b" as const,
    manifest: "https://example.test/manifest.json",
    sha256: "a".repeat(64),
  };
  const directory = await openModelStorage(entry, root);
  expect((staleModel as MemoryOpfsDirectory).dirs.has("old-revision")).toBe(
    false,
  );

  let fetchCalls = 0;
  const fetchSpy = spyOn(globalThis, "fetch").mockImplementation(
    Object.assign(
      async () => {
        fetchCalls++;
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(payload.subarray(0, 2));
              controller.enqueue(payload.subarray(2));
              controller.close();
            },
          }),
        );
      },
      { preconnect: () => {} },
    ),
  );
  try {
    const progress: number[] = [];
    const first = await ensureStoredAsset(
      directory,
      entry,
      { path: "weights-0001.safetensors", size: payload.byteLength, sha256 },
      new AbortController().signal,
      nodeSha256Hasher,
      (bytes) => progress.push(bytes),
    );
    expect(fetchCalls).toBe(1);
    expect(progress).toEqual([2, 6]);
    expect(await first.readAll()).toEqual(payload);

    // Repeat access skips network completely once sidecar matches.
    const secondProgress: number[] = [];
    const second = await ensureStoredAsset(
      directory,
      entry,
      { path: "weights-0001.safetensors", size: payload.byteLength, sha256 },
      new AbortController().signal,
      nodeSha256Hasher,
      (bytes) => secondProgress.push(bytes),
    );
    expect(fetchCalls).toBe(1);
    expect(secondProgress).toEqual([6]);
    expect(await second.readAll()).toEqual(payload);

    // Checksum mismatch removes partial file and sidecar.
    await expect(
      ensureStoredAsset(
        directory,
        entry,
        {
          path: "bad.safetensors",
          size: payload.byteLength,
          sha256: "f".repeat(64),
        },
        new AbortController().signal,
        nodeSha256Hasher,
      ),
    ).rejects.toThrow("Model checksum mismatch.");
    expect(
      (directory as MemoryOpfsDirectory).files.has("bad.safetensors"),
    ).toBe(false);
    expect(
      (directory as MemoryOpfsDirectory).files.has(
        "bad.safetensors.verified.json",
      ),
    ).toBe(false);
  } finally {
    fetchSpy.mockRestore();
  }
});
