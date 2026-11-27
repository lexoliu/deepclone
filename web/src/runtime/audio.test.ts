import { test, expect } from "bun:test";
import { encodeWav, mixActiveChannels, normalizeReferenceLevel } from "./audio";
import { verify } from "./assets";
import { REFERENCE_AUDIO_VERSION, isCurrentConditioning } from "./contracts";

function sineWave(
  length: number,
  amplitude: number,
  sampleRate = 24000,
  frequency = 220,
): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / sampleRate);
  }
  return out;
}

function rms(samples: Float32Array): number {
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    const value = samples[i]!;
    sumSq += value * value;
  }
  return Math.sqrt(sumSq / Math.max(1, samples.length));
}

function peak(samples: Float32Array): number {
  let max = 0;
  for (let i = 0; i < samples.length; i++) {
    const abs = Math.abs(samples[i]!);
    if (abs > max) max = abs;
  }
  return max;
}

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
test("mixActiveChannels ignores silent stereo channels without attenuating speech", () => {
  const speech = new Float32Array([0.05, -0.05, 0.025, -0.025]);
  const silence = new Float32Array(speech.length);
  expect(Array.from(mixActiveChannels([speech, silence]))).toEqual(
    Array.from(speech),
  );
  expect(Array.from(mixActiveChannels([silence, speech]))).toEqual(
    Array.from(speech),
  );
  const stereo = mixActiveChannels([
    new Float32Array([0.2, -0.2]),
    new Float32Array([0.6, -0.6]),
  ]);
  expect(stereo[0]).toBeCloseTo(0.4, 6);
  expect(stereo[1]).toBeCloseTo(-0.4, 6);
  expect(Array.from(mixActiveChannels([silence, silence]))).toEqual([
    0, 0, 0, 0,
  ]);
});
test("normalizeReferenceLevel lifts quiet speech to -22 dBFS while guarding silence and peaks", () => {
  const quiet = sineWave(24000, 0.014);
  const boosted = normalizeReferenceLevel(quiet, 24000);
  expect(rms(boosted)).toBeCloseTo(0.08, 2);
  expect(peak(boosted)).toBeLessThanOrEqual(0.891);

  // Leading and trailing silence must not inflate the gain applied to voiced segments.
  const padded = new Float32Array(24000 * 3);
  padded.set(sineWave(24000, 0.014), 24000);
  const normalizedPadded = normalizeReferenceLevel(padded, 24000);
  const activeSlice = normalizedPadded.subarray(24000, 48000);
  expect(rms(activeSlice)).toBeCloseTo(0.08, 2);

  // DC bias is removed before measuring level.
  const biased = sineWave(24000, 0.014);
  for (let i = 0; i < biased.length; i++) biased[i] = biased[i]! + 0.15;
  const debias = normalizeReferenceLevel(biased, 24000);
  let mean = 0;
  for (let i = 0; i < debias.length; i++) mean += debias[i]!;
  expect(Math.abs(mean / debias.length)).toBeLessThan(1e-5);
  expect(rms(debias)).toBeCloseTo(0.08, 2);

  // High-crest spikes respect the peak ceiling.
  const spiky = sineWave(24000, 0.02);
  spiky[100] = 0.5;
  const capped = normalizeReferenceLevel(spiky, 24000);
  expect(peak(capped)).toBeLessThanOrEqual(0.891);

  // Near-silent captures are left unboosted.
  const floor = sineWave(24000, 1e-4);
  const untouched = normalizeReferenceLevel(floor, 24000);
  expect(peak(untouched)).toBeLessThan(2e-4);

  // Very quiet signals respect the 16x maximum gain cap.
  const veryQuiet = sineWave(24000, 0.002);
  const gainCapped = normalizeReferenceLevel(veryQuiet, 24000);
  expect(peak(gainCapped)).toBeCloseTo(0.032, 3);
});
test("isCurrentConditioning rejects legacy entries without the current audio version", () => {
  const legacy = {
    modelSha256: "0".repeat(64),
    speaker: new Float32Array([0.5]),
    speakerShape: [1, 1, 1] as [number, number, number],
    codes: new Uint32Array([1, 2]),
    groups: 2,
  };
  expect(isCurrentConditioning(undefined)).toBe(false);
  expect(isCurrentConditioning(legacy)).toBe(false);
  expect(isCurrentConditioning({ ...legacy, audioVersion: 1 })).toBe(false);
  expect(
    isCurrentConditioning({
      ...legacy,
      audioVersion: REFERENCE_AUDIO_VERSION,
    }),
  ).toBe(true);
});
