import { test, expect } from "bun:test";
import { encodeWav } from "./audio";
import { verify } from "./assets";
test("WAV carries identical left and right channels across streamed chunks", async () => {
  const blob = encodeWav(
    [new Float32Array([-0.5, 0]), new Float32Array([0.25, 1])],
    24000,
  );
  const view = new DataView(await blob.arrayBuffer());
  expect(view.getUint16(22, true)).toBe(2);
  expect(view.getUint32(24, true)).toBe(24000);
  expect(view.getUint32(40, true)).toBe(16);
  for (let offset = 44; offset < view.byteLength; offset += 4)
    expect(view.getInt16(offset, true)).toBe(view.getInt16(offset + 2, true));
});
test("model hash verification rejects corrupted bytes", async () => {
  const bytes = new TextEncoder().encode("abc");
  await verify(
    bytes,
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
  await expect(
    verify(
      new TextEncoder().encode("abd"),
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    ),
  ).rejects.toThrow("checksum");
});
