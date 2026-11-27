import { expect, test } from "bun:test";
import {
  RECORDING_AUDIO_CONSTRAINTS,
  encodeChannelWav,
  selectRecordedChannels,
} from "./recorder";

test("recording constraints request stereo with noise suppression and AGC", () => {
  expect(RECORDING_AUDIO_CONSTRAINTS.channelCount).toEqual({ ideal: 2 });
  expect(RECORDING_AUDIO_CONSTRAINTS.noiseSuppression).toBe(true);
  expect(RECORDING_AUDIO_CONSTRAINTS.autoGainControl).toBe(true);
  expect(RECORDING_AUDIO_CONSTRAINTS.echoCancellation).toBe(false);
});

test("selectRecordedChannels preserves active stereo and collapses silent-channel stereo to mono", () => {
  const left = new Float32Array([0.2, -0.2, 0.1, -0.1]);
  const right = new Float32Array([0.15, -0.15, 0.08, -0.08]);
  const silence = new Float32Array(4);

  const stereo = selectRecordedChannels([left, right]);
  expect(stereo.length).toBe(2);
  expect(stereo[0]).toBe(left);
  expect(stereo[1]).toBe(right);

  const leftOnly = selectRecordedChannels([left, silence]);
  expect(leftOnly.length).toBe(1);
  expect(leftOnly[0]).toBe(left);

  const rightOnly = selectRecordedChannels([silence, right]);
  expect(rightOnly.length).toBe(1);
  expect(rightOnly[0]).toBe(right);
});

test("encodeChannelWav writes valid 16-bit PCM WAV headers for both mono and stereo", async () => {
  const stereoBlob = encodeChannelWav(
    [new Float32Array([0.5, -0.5]), new Float32Array([-0.25, 0.25])],
    48000,
  );
  const stereoView = new DataView(await stereoBlob.arrayBuffer());
  expect(stereoView.getUint16(22, true)).toBe(2);
  expect(stereoView.getUint32(24, true)).toBe(48000);
  expect(stereoView.getUint32(40, true)).toBe(8);
  expect(stereoView.getInt16(44, true)).toBe(16384);
  expect(stereoView.getInt16(46, true)).toBe(-8192);
  expect(stereoView.getInt16(48, true)).toBe(-16384);
  expect(stereoView.getInt16(50, true)).toBe(8192);

  const monoBlob = encodeChannelWav([new Float32Array([0.5, -1])], 24000);
  const monoView = new DataView(await monoBlob.arrayBuffer());
  expect(monoView.getUint16(22, true)).toBe(1);
  expect(monoView.getUint32(24, true)).toBe(24000);
  expect(monoView.getUint32(40, true)).toBe(4);
  expect(monoView.getInt16(44, true)).toBe(16384);
  expect(monoView.getInt16(46, true)).toBe(-32768);
});
