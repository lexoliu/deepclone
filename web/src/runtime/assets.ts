import type { Asset, Manifest, ModelEntry } from "./contracts";
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
  const cache = await caches.open("deepclone-verified-models-v1");
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
