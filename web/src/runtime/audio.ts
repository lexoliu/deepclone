const REFERENCE_SAMPLE_RATE = 24000;
// Channels more than 40 dB below the loudest channel are treated as silent
// (e.g. mobile WebKit MediaRecorder packing a mono mic into stereo with a zero right channel).
const SILENT_CHANNEL_RATIO = 0.01;
// 50 ms frame window for gated active-speech RMS estimation.
const LEVEL_WINDOW_SECONDS = 0.05;
// Gate frames within 20 dB of the 95th-percentile frame RMS so pauses do not dilute speech level.
const ACTIVE_FRAME_RATIO = 0.1;
// Target -22 dBFS active speech RMS with a -1 dBFS peak ceiling for Mimi and speaker conditioning.
const TARGET_ACTIVE_RMS = 0.08;
const MAX_PEAK = 0.89;
// Cap boost at +24 dB (16x) and leave near-silent captures unboosted.
const MAX_GAIN = 16;
const MIN_ACTIVE_RMS = 1e-4;
const MIN_PEAK = 1e-3;

export function mixActiveChannels(
  channels: readonly Float32Array[],
): Float32Array {
  if (channels.length === 0) return new Float32Array(0);
  const length = channels[0]!.length;
  if (channels.length === 1) return channels[0]!.slice();
  const rms = channels.map((channel) => {
    let sumSq = 0;
    for (let i = 0; i < length; i++) {
      const sample = channel[i]!;
      sumSq += sample * sample;
    }
    return Math.sqrt(sumSq / Math.max(1, length));
  });
  let maxRms = 0;
  for (const value of rms) {
    if (value > maxRms) maxRms = value;
  }
  if (maxRms === 0) return new Float32Array(length);
  const threshold = maxRms * SILENT_CHANNEL_RATIO;
  const active = channels.filter((_, index) => rms[index]! >= threshold);
  if (active.length === 1) return active[0]!.slice();
  const mono = new Float32Array(length);
  const scale = 1 / active.length;
  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (let c = 0; c < active.length; c++) sum += active[c]![i]!;
    mono[i] = sum * scale;
  }
  return mono;
}

export function normalizeReferenceLevel(
  samples: Float32Array,
  sampleRate: number,
): Float32Array {
  const length = samples.length;
  if (length === 0) return samples.slice();
  let sum = 0;
  for (let i = 0; i < length; i++) sum += samples[i]!;
  const mean = sum / length;
  const centered = new Float32Array(length);
  let peak = 0;
  for (let i = 0; i < length; i++) {
    const value = samples[i]! - mean;
    centered[i] = value;
    const abs = Math.abs(value);
    if (abs > peak) peak = abs;
  }
  if (peak < MIN_PEAK) return centered;

  const windowSize = Math.max(1, Math.round(sampleRate * LEVEL_WINDOW_SECONDS));
  const frameRms: number[] = [];
  for (let start = 0; start < length; start += windowSize) {
    const end = Math.min(length, start + windowSize);
    const count = end - start;
    if (count < windowSize / 2 && frameRms.length > 0) continue;
    let sumSq = 0;
    for (let i = start; i < end; i++) {
      const value = centered[i]!;
      sumSq += value * value;
    }
    frameRms.push(Math.sqrt(sumSq / count));
  }
  if (frameRms.length === 0) return centered;
  frameRms.sort((a, b) => a - b);
  const p95 =
    frameRms[
      Math.min(frameRms.length - 1, Math.floor(frameRms.length * 0.95))
    ]!;
  const gate = p95 * ACTIVE_FRAME_RATIO;
  let voicedSq = 0;
  let voicedCount = 0;
  for (const value of frameRms) {
    if (value >= gate) {
      voicedSq += value * value;
      voicedCount++;
    }
  }
  const activeRms = Math.sqrt(voicedSq / Math.max(1, voicedCount));
  if (activeRms < MIN_ACTIVE_RMS) return centered;

  const gain = Math.min(
    TARGET_ACTIVE_RMS / activeRms,
    MAX_PEAK / peak,
    MAX_GAIN,
  );
  for (let i = 0; i < length; i++) {
    const scaled = centered[i]! * gain;
    centered[i] = scaled < -1 ? -1 : scaled > 1 ? 1 : scaled;
  }
  return centered;
}

export async function referenceAudio(file: File): Promise<Float32Array> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const channels: Float32Array[] = [];
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      channels.push(decoded.getChannelData(c));
    }
    const mono = mixActiveChannels(channels);
    let resampled: Float32Array;
    if (decoded.sampleRate === REFERENCE_SAMPLE_RATE) {
      resampled = mono;
    } else {
      // Qwen3-TTS's published reference frontend requires mono 24 kHz.
      const offline = new OfflineAudioContext(
        1,
        Math.ceil(decoded.duration * REFERENCE_SAMPLE_RATE),
        REFERENCE_SAMPLE_RATE,
      );
      const monoBuffer = offline.createBuffer(
        1,
        mono.length,
        decoded.sampleRate,
      );
      monoBuffer.copyToChannel(mono as Float32Array<ArrayBuffer>, 0);
      const source = offline.createBufferSource();
      source.buffer = monoBuffer;
      source.connect(offline.destination);
      source.start();
      resampled = (await offline.startRendering()).getChannelData(0).slice();
    }
    return normalizeReferenceLevel(resampled, REFERENCE_SAMPLE_RATE);
  } finally {
    await context.close();
  }
}
export class Player {
  private context: AudioContext;
  private end = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private waiting: Promise<void>[] = [];
  private chunks: Float32Array[] = [];
  private rate = 24000;
  private underruns = 0;
  private gapSeconds = 0;
  private consumerWaitMs = 0;
  private firstPlayback: number | null = null;
  constructor() {
    this.context = new AudioContext();
  }
  async resume() {
    await this.context.resume();
  }
  async enqueue(samples: Float32Array, rate: number) {
    this.rate = rate;
    this.chunks.push(samples);
    const buffer = this.context.createBuffer(2, samples.length, rate);
    buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
    buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 1);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    this.sources.add(source);
    const ended = new Promise<void>((resolve) => {
      source.onended = () => {
        source.disconnect();
        this.sources.delete(source);
        resolve();
      };
    });
    const now = this.context.currentTime;
    if (this.end > 0 && now > this.end) {
      this.underruns++;
      this.gapSeconds += now - this.end;
    }
    const start = Math.max(now, this.end);
    this.firstPlayback ??= performance.now() + (start - now) * 1000;
    source.start(start);
    this.end = start + buffer.duration;
    this.waiting.push(ended);
    // Bound playback to the current chunk plus one queued chunk.
    if (this.waiting.length >= 2) {
      const waiting = performance.now();
      await this.waiting.shift();
      this.consumerWaitMs += performance.now() - waiting;
    }
  }
  async finish() {
    await Promise.all(this.waiting);
    this.waiting = [];
  }
  wav() {
    return encodeWav(this.chunks, this.rate);
  }
  get hasAudio() {
    return this.chunks.some((chunk) => chunk.length > 0);
  }
  diagnostics() {
    return {
      underruns: this.underruns,
      gapSeconds: this.gapSeconds,
      consumerWaitMs: this.consumerWaitMs,
      firstPlayback: this.firstPlayback,
    };
  }
  async close() {
    if (this.context.state === "closed") return;
    for (const source of this.sources) source.stop();
    await this.context.close();
  }
}
export function encodeWav(chunks: Float32Array[], rate: number): Blob {
  const length = chunks.reduce((sum, part) => sum + part.length, 0);
  const bytes = new ArrayBuffer(44 + length * 4);
  const view = new DataView(bytes);
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++)
      view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + length * 4, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, length * 4, true);
  let index = 44;
  for (const chunk of chunks)
    for (const sample of chunk) {
      const value = Math.max(-1, Math.min(1, sample));
      const pcm = Math.round(value * (value < 0 ? 32768 : 32767));
      view.setInt16(index, pcm, true);
      view.setInt16(index + 2, pcm, true);
      index += 4;
    }
  return new Blob([bytes], { type: "audio/wav" });
}
