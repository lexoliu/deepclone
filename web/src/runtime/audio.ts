const REFERENCE_SAMPLE_RATE = 24000;
// Channels more than 40 dB below the loudest channel are treated as silent
// (e.g. mobile WebKit MediaRecorder packing a mono mic into stereo with a zero right channel).
const SILENT_CHANNEL_RATIO = 0.01;
// 2nd-order Butterworth high-pass cutoff (Hz) to strip DC offset, AC hum, and mic handling rumble.
const HIGH_PASS_CUTOFF_HZ = 80;
// Stationary Wiener noise reduction parameters: 512-point (21.3 ms) Hann STFT at 24 kHz with 75% overlap.
const DENOISE_FFT_SIZE = 512;
const DENOISE_HOP_SIZE = 128;
// Estimate stationary background noise from the quietest 15% of interior STFT frames.
const DENOISE_NOISE_PERCENTILE = 0.15;
// Apply stationary suppression only when quiet pauses exist (p90/p10 >= 10 dB, power ratio >= 10)
// and background noise is still material (SNR <= 30 dB, power ratio <= 1000).
const DENOISE_MIN_SNR_POWER_RATIO = 10;
const DENOISE_CLEAN_SNR_POWER_RATIO = 1000;
const DENOISE_STRENGTH = 1.2;
const DENOISE_FLOOR_GAIN = 0.15;
// Tap-resistant speech boundary detection: 20 ms window, 10 ms hop, 80 ms minimum active run, 60 ms padding.
const TRIM_WINDOW_SECONDS = 0.02;
const TRIM_HOP_SECONDS = 0.01;
const TRIM_MIN_RUN_FRAMES = 8;
const TRIM_PADDING_SECONDS = 0.06;
const TRIM_MIN_RESULT_SECONDS = 0.5;
const TRIM_NOISE_FLOOR_MULTIPLIER = 2.2;
const TRIM_SPEECH_PEAK_RATIO = 0.12;
// 50 ms frame window for gated active-speech RMS estimation.
const LEVEL_WINDOW_SECONDS = 0.05;
// Gate frames within 20 dB of the 95th-percentile frame RMS so pauses do not dilute speech level.
const ACTIVE_FRAME_RATIO = 0.1;
// Target -26 dBFS active speech RMS with a -1 dBFS peak ceiling so quiet mics reach
// Qwen3-TTS's natural speech range without over-amplifying room noise.
const TARGET_ACTIVE_RMS = 0.05;
const MAX_PEAK = 0.89;
// Cap boost at +24 dB (16x) and leave near-silent captures unboosted.
const MAX_GAIN = 16;
const MIN_ACTIVE_RMS = 1e-4;
const MIN_PEAK = 1e-3;

export function highPassFilter(
  samples: Float32Array,
  sampleRate: number,
  cutoffHz = HIGH_PASS_CUTOFF_HZ,
): Float32Array {
  const length = samples.length;
  if (length === 0) return samples.slice();
  const w0 = (2 * Math.PI * cutoffHz) / sampleRate;
  const cosW0 = Math.cos(w0);
  // Butterworth Q = 1 / sqrt(2) for maximally flat passband.
  const alpha = Math.sin(w0) / (2 * Math.SQRT1_2);
  const a0 = 1 + alpha;
  const b0 = (1 + cosW0) / 2 / a0;
  const b1 = -(1 + cosW0) / a0;
  const b2 = (1 + cosW0) / 2 / a0;
  const a1 = (-2 * cosW0) / a0;
  const a2 = (1 - alpha) / a0;
  const out = new Float32Array(length);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < length; i++) {
    const x0 = samples[i]!;
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    out[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return out;
}

function fftInPlace(
  re: Float64Array,
  im: Float64Array,
  inverse: boolean,
): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wStepRe = Math.cos(angle);
    const wStepIm = Math.sin(angle);
    for (let start = 0; start < n; start += len) {
      let wRe = 1;
      let wIm = 0;
      for (let k = 0; k < half; k++) {
        const uRe = re[start + k]!;
        const uIm = im[start + k]!;
        const vRe = re[start + k + half]! * wRe - im[start + k + half]! * wIm;
        const vIm = re[start + k + half]! * wIm + im[start + k + half]! * wRe;
        re[start + k] = uRe + vRe;
        im[start + k] = uIm + vIm;
        re[start + k + half] = uRe - vRe;
        im[start + k + half] = uIm - vIm;
        const nextRe = wRe * wStepRe - wIm * wStepIm;
        wIm = wRe * wStepIm + wIm * wStepRe;
        wRe = nextRe;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] = re[i]! / n;
      im[i] = im[i]! / n;
    }
  }
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor(sorted.length * p)),
  );
  return sorted[idx]!;
}

export function reduceStationaryNoise(
  samples: Float32Array,
  sampleRate = REFERENCE_SAMPLE_RATE,
): Float32Array {
  const length = samples.length;
  // Require at least ~0.25 s of audio so quiet-frame noise profiling is well-formed.
  if (length < Math.max(DENOISE_FFT_SIZE * 2, Math.round(sampleRate * 0.25))) {
    return samples.slice();
  }
  const nfft = DENOISE_FFT_SIZE;
  const hop = DENOISE_HOP_SIZE;
  const bins = nfft / 2 + 1;
  const win = new Float64Array(nfft);
  for (let i = 0; i < nfft; i++) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft);
  }
  const pad = nfft;
  const paddedLen = length + pad * 2;
  const padded = new Float64Array(paddedLen);
  for (let i = 0; i < length; i++) padded[pad + i] = samples[i]!;

  const frameCount = Math.floor((paddedLen - nfft) / hop) + 1;
  const innerStart = Math.floor(pad / hop);
  const innerEnd = frameCount - innerStart;
  if (innerEnd - innerStart < 8) return samples.slice();

  const specRe = new Float64Array(frameCount * bins);
  const specIm = new Float64Array(frameCount * bins);
  const power = new Float64Array(frameCount * bins);
  const frameEnergy = new Float64Array(frameCount);
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);

  for (let f = 0; f < frameCount; f++) {
    const start = f * hop;
    for (let i = 0; i < nfft; i++) {
      re[i] = padded[start + i]! * win[i]!;
      im[i] = 0;
    }
    fftInPlace(re, im, false);
    let energy = 0;
    const base = f * bins;
    for (let b = 0; b < bins; b++) {
      const r = re[b]!;
      const m = im[b]!;
      specRe[base + b] = r;
      specIm[base + b] = m;
      const p = r * r + m * m;
      power[base + b] = p;
      energy += p;
    }
    frameEnergy[f] = energy / bins;
  }

  const innerEnergies: number[] = [];
  for (let f = innerStart; f < innerEnd; f++)
    innerEnergies.push(frameEnergy[f]!);
  innerEnergies.sort((a, b) => a - b);
  const p10 = percentile(innerEnergies, 0.1);
  const p90 = percentile(innerEnergies, 0.9);
  const snrPowerRatio = p90 / Math.max(p10, 1e-18);
  if (
    p90 <= 1e-12 ||
    snrPowerRatio < DENOISE_MIN_SNR_POWER_RATIO ||
    snrPowerRatio > DENOISE_CLEAN_SNR_POWER_RATIO
  ) {
    return samples.slice();
  }

  const noiseCutoff = percentile(innerEnergies, DENOISE_NOISE_PERCENTILE);
  const noisePsd = new Float64Array(bins);
  let noiseCount = 0;
  for (let f = innerStart; f < innerEnd; f++) {
    if (frameEnergy[f]! <= noiseCutoff) {
      const base = f * bins;
      for (let b = 0; b < bins; b++)
        noisePsd[b] = noisePsd[b]! + power[base + b]!;
      noiseCount++;
    }
  }
  if (noiseCount === 0) return samples.slice();
  for (let b = 0; b < bins; b++) noisePsd[b] = noisePsd[b]! / noiseCount;

  const rawGain = new Float64Array(frameCount * bins);
  for (let f = 0; f < frameCount; f++) {
    const base = f * bins;
    for (let b = 0; b < bins; b++) {
      const p = power[base + b]!;
      const g = 1 - (DENOISE_STRENGTH * noisePsd[b]!) / (p + 1e-12);
      rawGain[base + b] = g < DENOISE_FLOOR_GAIN ? DENOISE_FLOOR_GAIN : g;
    }
  }

  const out = new Float64Array(paddedLen);
  const wSum = new Float64Array(paddedLen);
  for (let f = 0; f < frameCount; f++) {
    const base = f * bins;
    const prevBase = Math.max(0, f - 1) * bins;
    const nextBase = Math.min(frameCount - 1, f + 1) * bins;
    for (let b = 0; b < bins; b++) {
      const g =
        0.25 * rawGain[prevBase + b]! +
        0.5 * rawGain[base + b]! +
        0.25 * rawGain[nextBase + b]!;
      const r = specRe[base + b]! * g;
      const m = specIm[base + b]! * g;
      re[b] = r;
      im[b] = m;
      if (b > 0 && b < bins - 1) {
        re[nfft - b] = r;
        im[nfft - b] = -m;
      }
    }
    fftInPlace(re, im, true);
    const start = f * hop;
    for (let i = 0; i < nfft; i++) {
      const w = win[i]!;
      out[start + i] = out[start + i]! + re[i]! * w;
      wSum[start + i] = wSum[start + i]! + w * w;
    }
  }

  const result = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const denom = wSum[pad + i]!;
    result[i] = denom > 1e-9 ? out[pad + i]! / denom : samples[i]!;
  }
  return result;
}

export function trimReferenceEdges(
  samples: Float32Array,
  sampleRate: number,
): Float32Array {
  const length = samples.length;
  const win = Math.max(1, Math.round(sampleRate * TRIM_WINDOW_SECONDS));
  const hop = Math.max(1, Math.round(sampleRate * TRIM_HOP_SECONDS));
  if (length <= win * (TRIM_MIN_RUN_FRAMES + 2)) return samples.slice();

  const frameCount = Math.floor((length - win) / hop) + 1;
  const frameRms = new Float64Array(frameCount);
  const sorted: number[] = [];
  for (let f = 0; f < frameCount; f++) {
    const start = f * hop;
    let sumSq = 0;
    for (let i = start; i < start + win; i++) {
      const v = samples[i]!;
      sumSq += v * v;
    }
    const r = Math.sqrt(sumSq / win);
    frameRms[f] = r;
    sorted.push(r);
  }
  sorted.sort((a, b) => a - b);
  const noiseFloor = percentile(sorted, 0.1);
  const speechP95 = percentile(sorted, 0.95);
  if (speechP95 < MIN_ACTIVE_RMS) return samples.slice();

  const threshold = Math.max(
    noiseFloor * TRIM_NOISE_FLOOR_MULTIPLIER,
    speechP95 * TRIM_SPEECH_PEAK_RATIO,
  );
  let firstActive = -1;
  let lastActive = -1;
  let run = 0;
  for (let f = 0; f < frameCount; f++) {
    if (frameRms[f]! >= threshold) {
      run++;
      if (run >= TRIM_MIN_RUN_FRAMES) {
        if (firstActive < 0) firstActive = f - TRIM_MIN_RUN_FRAMES + 1;
        lastActive = f;
      }
    } else {
      run = 0;
    }
  }
  if (firstActive < 0 || lastActive < 0) return samples.slice();

  const pad = Math.round(sampleRate * TRIM_PADDING_SECONDS);
  const start = Math.max(0, firstActive * hop - pad);
  const end = Math.min(length, lastActive * hop + win + pad);
  if (end - start < Math.round(sampleRate * TRIM_MIN_RESULT_SECONDS)) {
    return samples.slice();
  }
  return samples.slice(start, end);
}

export function prepareReferenceSamples(
  samples: Float32Array,
  sampleRate: number,
): Float32Array {
  const filtered = highPassFilter(samples, sampleRate);
  const denoised = reduceStationaryNoise(filtered, sampleRate);
  const trimmed = trimReferenceEdges(denoised, sampleRate);
  return normalizeReferenceLevel(trimmed, sampleRate);
}

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
    return prepareReferenceSamples(resampled, REFERENCE_SAMPLE_RATE);
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
  get durationSeconds() {
    const total = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    return total / this.rate;
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
