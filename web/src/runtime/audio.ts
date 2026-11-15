export async function referenceAudio(file: File): Promise<Float32Array> {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    // Qwen3-TTS's published reference frontend requires mono 24 kHz.
    const offline = new OfflineAudioContext(
      1,
      Math.ceil(decoded.duration * 24000),
      24000,
    );
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    return (await offline.startRendering()).getChannelData(0).slice();
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
    source.start(Math.max(this.context.currentTime, this.end));
    this.end = Math.max(this.context.currentTime, this.end) + buffer.duration;
    this.waiting.push(ended);
    // Bound playback to the current chunk plus one queued chunk.
    if (this.waiting.length >= 2) await this.waiting.shift();
  }
  async finish() {
    await Promise.all(this.waiting);
    this.waiting = [];
  }
  wav() {
    return encodeWav(this.chunks, this.rate);
  }
  async close() {
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
