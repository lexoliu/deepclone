import type { Asset, Manifest, ModelEntry } from "./contracts";

export const TEXT_EMBEDDING_TENSOR = "talker.model.text_embedding.weight";
const LEGACY_CACHE_NAME = "deepclone-verified-models-v1";
const OPFS_ROOT_DIR = "deepclone-verified-models-v2";
const MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024;

export interface StreamingHasher {
  update(chunk: Uint8Array): void;
  finalize(): string;
  free(): void;
}

export interface SyncAccessHandleLike {
  getSize(): number;
  read(buffer: ArrayBufferView, options?: { at?: number }): number;
  write(buffer: ArrayBufferView, options?: { at?: number }): number;
  truncate(newSize: number): void;
  flush(): void;
  close(): void;
}

export interface OpfsFileHandleLike {
  getFile(): Promise<Blob>;
  createSyncAccessHandle(): Promise<SyncAccessHandleLike>;
}

export interface OpfsDirectoryHandleLike {
  getDirectoryHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OpfsDirectoryHandleLike>;
  getFileHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OpfsFileHandleLike>;
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
  entries?(): AsyncIterableIterator<[string, { kind: string }]>;
}

export interface StoredFile {
  readonly path: string;
  readonly size: number;
  readAll(): Promise<Uint8Array>;
  withReader<T>(
    fn: (readInto: (offset: number, target: Uint8Array) => void) => T,
  ): Promise<T>;
}

export interface SafetensorsTensorInfo {
  name: string;
  dtype: string;
  shape: number[];
  start: number;
  length: number;
}

export interface SafetensorsHeader {
  dataStart: number;
  tensors: SafetensorsTensorInfo[];
}

export interface TextEmbeddingDescriptor {
  file: StoredFile;
  dtype: string;
  vocabSize: number;
  width: number;
  elementBytes: number;
  dataStart: number;
}

export interface ScratchBuffer {
  buffer: Uint8Array;
}

interface VerifiedMetadata {
  size: number;
  sha256: string;
}

function validateFileName(path: string): void {
  if (
    !path ||
    path === "." ||
    path === ".." ||
    path.includes("/") ||
    path.includes("\\")
  ) {
    throw new Error(`Invalid model file path: ${path}`);
  }
}

function readExact(
  access: SyncAccessHandleLike,
  offset: number,
  target: Uint8Array,
): void {
  let received = 0;
  while (received < target.byteLength) {
    const view =
      received === 0 ? target : target.subarray(received, target.byteLength);
    const count = access.read(view, { at: offset + received });
    if (count <= 0) {
      throw new Error("Unexpected end of model file.");
    }
    received += count;
  }
}

function writeExact(
  access: SyncAccessHandleLike,
  offset: number,
  source: Uint8Array,
): void {
  let written = 0;
  while (written < source.byteLength) {
    const view =
      written === 0 ? source : source.subarray(written, source.byteLength);
    const count = access.write(view, { at: offset + written });
    if (count <= 0) {
      throw new Error("Could not write model file to browser storage.");
    }
    written += count;
  }
}

class OpfsStoredFile implements StoredFile {
  readonly path: string;
  readonly size: number;
  private readonly handle: OpfsFileHandleLike;

  constructor(path: string, size: number, handle: OpfsFileHandleLike) {
    this.path = path;
    this.size = size;
    this.handle = handle;
  }

  async readAll(): Promise<Uint8Array> {
    const out = new Uint8Array(this.size);
    await this.withReader((readInto) => {
      if (this.size > 0) readInto(0, out);
    });
    return out;
  }

  async withReader<T>(
    fn: (readInto: (offset: number, target: Uint8Array) => void) => T,
  ): Promise<T> {
    const access = await this.handle.createSyncAccessHandle();
    try {
      return fn((offset, target) => readExact(access, offset, target));
    } finally {
      access.close();
    }
  }
}

export class BytesStoredFile implements StoredFile {
  readonly path: string;
  readonly size: number;
  private readonly bytes: Uint8Array;

  constructor(path: string, bytes: Uint8Array) {
    this.path = path;
    this.size = bytes.byteLength;
    this.bytes = bytes;
  }

  async readAll(): Promise<Uint8Array> {
    return this.bytes.slice();
  }

  async withReader<T>(
    fn: (readInto: (offset: number, target: Uint8Array) => void) => T,
  ): Promise<T> {
    return fn((offset, target) => {
      if (offset < 0 || offset + target.byteLength > this.bytes.byteLength) {
        throw new Error("Unexpected end of model file.");
      }
      target.set(this.bytes.subarray(offset, offset + target.byteLength));
    });
  }
}

export function dtypeElementBytes(dtype: string): number {
  switch (dtype) {
    case "F16":
    case "BF16":
      return 2;
    case "F32":
    case "U32":
      return 4;
    default:
      throw new Error(`Unsupported safetensors dtype: ${dtype}`);
  }
}

export function parseSafetensorsHeader(
  readInto: (offset: number, target: Uint8Array) => void,
  fileSize: number,
): SafetensorsHeader {
  if (fileSize < 8) {
    throw new Error("Invalid safetensors file: header length is truncated.");
  }
  const lengthBytes = new Uint8Array(8);
  readInto(0, lengthBytes);
  const view = new DataView(
    lengthBytes.buffer,
    lengthBytes.byteOffset,
    lengthBytes.byteLength,
  );
  const headerLenBig = view.getBigUint64(0, true);
  if (
    headerLenBig === 0n ||
    headerLenBig > BigInt(MAX_SAFETENSORS_HEADER_BYTES)
  ) {
    throw new Error("Invalid safetensors header length.");
  }
  const headerLen = Number(headerLenBig);
  const dataStart = 8 + headerLen;
  if (dataStart > fileSize) {
    throw new Error("Invalid safetensors header: extends past file end.");
  }
  const headerBytes = new Uint8Array(headerLen);
  readInto(8, headerBytes);
  const parsed: unknown = JSON.parse(new TextDecoder().decode(headerBytes));
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Invalid safetensors header JSON.");
  }
  const tensors: SafetensorsTensorInfo[] = [];
  for (const [name, entry] of Object.entries(
    parsed as Record<string, unknown>,
  )) {
    if (name === "__metadata__") continue;
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`Invalid safetensors entry for ${name}.`);
    }
    const record = entry as {
      dtype?: unknown;
      shape?: unknown;
      data_offsets?: unknown;
    };
    if (typeof record.dtype !== "string" || !record.dtype) {
      throw new Error(`Invalid safetensors dtype for ${name}.`);
    }
    if (
      !Array.isArray(record.shape) ||
      record.shape.some(
        (dim) =>
          typeof dim !== "number" || !Number.isSafeInteger(dim) || dim < 0,
      )
    ) {
      throw new Error(`Invalid safetensors shape for ${name}.`);
    }
    if (
      !Array.isArray(record.data_offsets) ||
      record.data_offsets.length !== 2 ||
      typeof record.data_offsets[0] !== "number" ||
      typeof record.data_offsets[1] !== "number" ||
      !Number.isSafeInteger(record.data_offsets[0]) ||
      !Number.isSafeInteger(record.data_offsets[1]) ||
      record.data_offsets[0] < 0 ||
      record.data_offsets[1] < record.data_offsets[0] ||
      dataStart + record.data_offsets[1] > fileSize
    ) {
      throw new Error(`Invalid safetensors data offsets for ${name}.`);
    }
    tensors.push({
      name,
      dtype: record.dtype,
      shape: record.shape as number[],
      start: dataStart + record.data_offsets[0],
      length: record.data_offsets[1] - record.data_offsets[0],
    });
  }
  tensors.sort((a, b) => a.start - b.start);
  return { dataStart, tensors };
}

export async function findTextEmbedding(
  file: StoredFile,
): Promise<TextEmbeddingDescriptor | null> {
  return file.withReader((readInto) => {
    const header = parseSafetensorsHeader(readInto, file.size);
    const tensor = header.tensors.find(
      (item) => item.name === TEXT_EMBEDDING_TENSOR,
    );
    if (!tensor) return null;
    if (tensor.shape.length !== 2) {
      throw new Error("Text embedding tensor must have rank 2.");
    }
    const [vocabSize, width] = tensor.shape as [number, number];
    const elementBytes = dtypeElementBytes(tensor.dtype);
    if (tensor.length !== vocabSize * width * elementBytes) {
      throw new Error("Text embedding byte length does not match its shape.");
    }
    return {
      file,
      dtype: tensor.dtype,
      vocabSize,
      width,
      elementBytes,
      dataStart: tensor.start,
    };
  });
}

export async function uploadShardTensors(
  file: StoredFile,
  conditioning: boolean,
  skipTensors: ReadonlySet<string>,
  uploadTensor: (
    name: string,
    dtype: string,
    shape: Uint32Array,
    bytes: Uint8Array,
    conditioning: boolean,
  ) => void,
  scratch: ScratchBuffer = { buffer: new Uint8Array(0) },
): Promise<void> {
  await file.withReader((readInto) => {
    const header = parseSafetensorsHeader(readInto, file.size);
    for (const tensor of header.tensors) {
      if (skipTensors.has(tensor.name)) continue;
      if (scratch.buffer.byteLength < tensor.length) {
        scratch.buffer = new Uint8Array(tensor.length);
      }
      const view = scratch.buffer.subarray(0, tensor.length);
      if (tensor.length > 0) {
        readInto(tensor.start, view);
      }
      uploadTensor(
        tensor.name,
        tensor.dtype,
        Uint32Array.from(tensor.shape),
        view,
        conditioning,
      );
    }
  });
}

export async function readTextEmbeddingRows(
  descriptor: TextEmbeddingDescriptor,
  tokenIds: Uint32Array,
): Promise<Uint8Array> {
  const rowBytes = descriptor.width * descriptor.elementBytes;
  const out = new Uint8Array(tokenIds.length * rowBytes);
  await descriptor.file.withReader((readInto) => {
    for (let i = 0; i < tokenIds.length; i++) {
      const id = tokenIds[i]!;
      if (id >= descriptor.vocabSize) {
        throw new Error(`Token ${id} exceeds text embedding vocabulary.`);
      }
      const offset = descriptor.dataStart + id * rowBytes;
      readInto(offset, out.subarray(i * rowBytes, (i + 1) * rowBytes));
    }
  });
  return out;
}

export async function openModelStorage(
  entry: ModelEntry,
  rootOverride?: OpfsDirectoryHandleLike,
): Promise<OpfsDirectoryHandleLike> {
  if (!rootOverride && typeof caches !== "undefined") {
    // Reclaim legacy Cache Storage entries so mobile origins do not store two copies.
    await caches.delete(LEGACY_CACHE_NAME).catch(() => false);
  }
  let root = rootOverride;
  if (!root) {
    const storage = (
      navigator as Navigator & {
        storage?: { getDirectory?: () => Promise<OpfsDirectoryHandleLike> };
      }
    ).storage;
    if (!storage?.getDirectory) {
      throw new Error(
        "Browser file storage is unavailable. Use a regular browser profile rather than private browsing.",
      );
    }
    try {
      root = await storage.getDirectory();
    } catch (cause) {
      throw new Error(
        "Could not access browser storage. Check available browser storage and use a regular browser profile rather than private browsing.",
        { cause },
      );
    }
  }
  const modelsDir = await root.getDirectoryHandle(OPFS_ROOT_DIR, {
    create: true,
  });
  if (typeof modelsDir.entries === "function") {
    for await (const [name] of modelsDir.entries()) {
      if (name !== entry.sha256) {
        await modelsDir.removeEntry(name, { recursive: true }).catch(() => {});
      }
    }
  }
  return modelsDir.getDirectoryHandle(entry.sha256, { create: true });
}

function metadataName(path: string): string {
  return `${path}.verified.json`;
}

async function readVerifiedFile(
  directory: OpfsDirectoryHandleLike,
  file: Asset,
): Promise<OpfsStoredFile | null> {
  try {
    const metaHandle = await directory.getFileHandle(metadataName(file.path));
    const metaText = await (await metaHandle.getFile()).text();
    const meta = JSON.parse(metaText) as Partial<VerifiedMetadata>;
    if (meta.size !== file.size || meta.sha256 !== file.sha256) {
      return null;
    }
    const dataHandle = await directory.getFileHandle(file.path);
    const blob = await dataHandle.getFile();
    if (blob.size !== file.size) {
      return null;
    }
    return new OpfsStoredFile(file.path, file.size, dataHandle);
  } catch {
    return null;
  }
}

export async function ensureStoredAsset(
  directory: OpfsDirectoryHandleLike,
  entry: ModelEntry,
  file: Asset,
  signal: AbortSignal,
  createHasher: () => StreamingHasher,
  progress?: (received: number) => void,
): Promise<StoredFile> {
  validateFileName(file.path);
  const existing = await readVerifiedFile(directory, file);
  if (existing) {
    progress?.(file.size);
    return existing;
  }

  const metaFile = metadataName(file.path);
  await directory.removeEntry(metaFile).catch(() => {});
  const url = new URL(file.path, entry.manifest).href;
  const dataHandle = await directory.getFileHandle(file.path, { create: true });
  let access: SyncAccessHandleLike | undefined;
  let hasher: StreamingHasher | undefined;
  let finalized = false;

  try {
    const response = await fetch(url, { signal });
    if (!response.ok) {
      throw new Error(`Model download failed (${response.status}).`);
    }
    if (!response.body) {
      throw new Error("The download has no response body.");
    }
    access = await dataHandle.createSyncAccessHandle();
    access.truncate(0);
    hasher = createHasher();

    const reader = response.body.getReader();
    let received = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (received + value.byteLength > file.size) {
          throw new Error("The download exceeds its declared size.");
        }
        hasher.update(value);
        writeExact(access, received, value);
        received += value.byteLength;
        progress?.(received);
      }
      if (received !== file.size) {
        throw new Error("The download is incomplete.");
      }
    } finally {
      reader.releaseLock();
    }

    finalized = true;
    const digest = hasher.finalize();
    if (digest !== file.sha256) {
      throw new Error(
        "Model checksum mismatch. Clear this model's cache and retry.",
      );
    }
    access.flush();
    access.close();
    access = undefined;

    const metaHandle = await directory.getFileHandle(metaFile, {
      create: true,
    });
    const metaAccess = await metaHandle.createSyncAccessHandle();
    try {
      metaAccess.truncate(0);
      const encoded = new TextEncoder().encode(
        JSON.stringify({ size: file.size, sha256: file.sha256 }),
      );
      writeExact(metaAccess, 0, encoded);
      metaAccess.flush();
    } finally {
      metaAccess.close();
    }
    return new OpfsStoredFile(file.path, file.size, dataHandle);
  } catch (cause) {
    if (hasher && !finalized) {
      hasher.free();
    }
    if (access) {
      try {
        access.close();
      } catch {
        // Ignore close errors while cleaning up a failed download.
      }
    }
    await directory.removeEntry(metaFile).catch(() => {});
    await directory.removeEntry(file.path).catch(() => {});
    if (
      cause instanceof Error &&
      cause.message.startsWith("Model checksum mismatch.")
    ) {
      throw cause;
    }
    throw new Error(`Could not download file: ${file.path}`, { cause });
  }
}

export async function verify(
  bytes: Uint8Array,
  expected: string,
): Promise<void> {
  const hash = await crypto.subtle.digest(
    "SHA-256",
    bytes as Uint8Array<ArrayBuffer>,
  );
  const hex = Array.from(new Uint8Array(hash), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  if (hex !== expected)
    throw new Error(
      "Model checksum mismatch. Clear this model's cache and retry.",
    );
}
async function download(
  url: string,
  signal: AbortSignal,
  size?: number,
  progress?: (received: number) => void,
): Promise<Uint8Array> {
  const response = await fetch(url, { signal });
  if (!response.ok)
    throw new Error(`Model download failed (${response.status}).`);
  if (size === undefined) return new Uint8Array(await response.arrayBuffer());
  if (!response.body) throw new Error("The download has no response body.");
  const reader = response.body.getReader();
  const bytes = new Uint8Array(size);
  let received = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (received + value.length > size)
        throw new Error("The download exceeds its declared size.");
      bytes.set(value, received);
      received += value.length;
      progress?.(received);
    }
    if (received !== size) throw new Error("The download is incomplete.");
    return bytes;
  } finally {
    reader.releaseLock();
  }
}
export async function manifest(
  entry: ModelEntry,
  signal: AbortSignal,
): Promise<Manifest> {
  const bytes = await download(entry.manifest, signal);
  await verify(bytes, entry.sha256);
  const data: Manifest = JSON.parse(new TextDecoder().decode(bytes));
  if (
    data.format !== "deepclone-burn-v1" ||
    !Array.isArray(data.files) ||
    !Array.isArray(data.shards)
  )
    throw new Error("Unsupported model package.");
  return data;
}
export async function asset(
  entry: ModelEntry,
  file: Asset,
  signal: AbortSignal,
  progress?: (received: number) => void,
): Promise<Uint8Array> {
  const url = new URL(file.path, entry.manifest).href;
  const cache = await caches.open(LEGACY_CACHE_NAME);
  const cached = await cache.match(url);
  let bytes: Uint8Array;
  try {
    bytes = cached
      ? new Uint8Array(await cached.arrayBuffer())
      : await download(url, signal, file.size, progress);
  } catch (cause) {
    throw new Error(
      `Could not ${cached ? "read cached" : "download"} file: ${file.path}`,
      { cause },
    );
  }
  progress?.(bytes.length);
  if (bytes.byteLength !== file.size)
    throw new Error(`Incomplete model file: ${file.path}`);
  try {
    await verify(bytes, file.sha256);
  } catch (error) {
    await cache.delete(url);
    throw error;
  }
  if (!cached) {
    try {
      await cache.put(url, new Response(bytes as Uint8Array<ArrayBuffer>));
    } catch (cause) {
      throw new Error(
        `Could not cache ${file.path}. Check available browser storage and use a regular browser profile rather than private browsing.`,
        { cause },
      );
    }
  }
  return bytes;
}
