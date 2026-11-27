// Channels more than 40 dB below the loudest channel are treated as silent.
const SILENT_CHANNEL_RATIO = 0.01;
const WORKLET_URL = "/recorder-worklet.js";

export const RECORDING_AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  // Prefer stereo capture with hardware/OS noise suppression and AGC while
  // keeping acoustic echo cancellation off so voice capture is not downmixed
  // to narrowband telephony mono.
  channelCount: { ideal: 2 },
  noiseSuppression: true,
  autoGainControl: true,
  echoCancellation: false,
};

export interface ActiveRecording {
  stop(): Promise<File | null>;
  abort(): void;
}

function channelRms(samples: Float32Array): number {
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    const value = samples[i]!;
    sumSq += value * value;
  }
  return Math.sqrt(sumSq / Math.max(1, samples.length));
}

export function selectRecordedChannels(
  channels: readonly Float32Array[],
): Float32Array[] {
  if (channels.length === 0) return [];
  if (channels.length === 1) return [channels[0]!];
  const rms = channels.map(channelRms);
  let maxRms = 0;
  for (const value of rms) {
    if (value > maxRms) maxRms = value;
  }
  if (maxRms === 0) return [channels[0]!];
  const threshold = maxRms * SILENT_CHANNEL_RATIO;
  const active = channels.filter((_, index) => rms[index]! >= threshold);
  return active.length > 0 ? active : [channels[0]!];
}

export function encodeChannelWav(
  channels: readonly Float32Array[],
  sampleRate: number,
): Blob {
  const channelCount = Math.max(1, channels.length);
  const frameCount = channels[0]?.length ?? 0;
  const blockAlign = channelCount * 2;
  const dataBytes = frameCount * blockAlign;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const writeAscii = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, dataBytes, true);
  let offset = 44;
  for (let frame = 0; frame < frameCount; frame++) {
    for (let c = 0; c < channelCount; c++) {
      const sample = channels[c]?.[frame] ?? 0;
      const clamped = Math.max(-1, Math.min(1, sample));
      const pcm = Math.round(clamped * (clamped < 0 ? 32768 : 32767));
      view.setInt16(offset, pcm, true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function concatChunks(chunks: readonly Float32Array[]): Float32Array {
  const total = chunks.reduce((sum, part) => sum + part.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const part of chunks) {
    merged.set(part, offset);
    offset += part.length;
  }
  return merged;
}

function startMediaRecorderFallback(
  media: MediaStream,
  onError: () => void,
): ActiveRecording | null {
  if (typeof MediaRecorder === "undefined") return null;
  // Prefer audio/mp4 before WebM so Safari avoids WebKit's WebM mono-into-stereo
  // muxer bug that leaves the right channel silent.
  const mimeType = [
    "audio/mp4",
    "audio/webm;codecs=opus",
    "audio/ogg;codecs=opus",
  ].find((type) => MediaRecorder.isTypeSupported(type));
  if (!mimeType) return null;

  const recorder = new MediaRecorder(media, { mimeType });
  const chunks: Blob[] = [];
  let failed = false;
  let stopped = false;

  const finished = new Promise<File | null>((resolve) => {
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    recorder.onerror = () => {
      failed = true;
      media.getTracks().forEach((track) => track.stop());
      onError();
      resolve(null);
    };
    recorder.onstop = () => {
      media.getTracks().forEach((track) => track.stop());
      if (failed) {
        resolve(null);
        return;
      }
      const type = recorder.mimeType || chunks[0]?.type || mimeType;
      const blob = new Blob(chunks, { type });
      if (!blob.size) {
        resolve(null);
        return;
      }
      const name = type.includes("mp4")
        ? "my-voice.m4a"
        : type.includes("ogg")
          ? "my-voice.ogg"
          : "my-voice.webm";
      resolve(new File([blob], name, { type }));
    };
  });

  recorder.start();
  return {
    stop() {
      if (!stopped && recorder.state !== "inactive") {
        stopped = true;
        recorder.stop();
      }
      return finished;
    },
    abort() {
      failed = true;
      if (!stopped && recorder.state !== "inactive") {
        stopped = true;
        recorder.stop();
      }
      media.getTracks().forEach((track) => track.stop());
    },
  };
}

export async function startMicRecorder(
  media: MediaStream,
  onError: () => void,
): Promise<ActiveRecording | null> {
  if (typeof AudioContext !== "undefined") {
    let context: AudioContext | null = null;
    try {
      context = new AudioContext();
      if (context.audioWorklet) {
        await context.audioWorklet.addModule(WORKLET_URL);
        if (context.state === "suspended") await context.resume();
        const track = media.getAudioTracks()[0];
        const reportedChannels = track?.getSettings?.().channelCount;
        const source = context.createMediaStreamSource(media);
        const channelCount = Math.max(
          1,
          Math.min(2, reportedChannels ?? source.channelCount ?? 2),
        );
        const node = new AudioWorkletNode(context, "pcm-recorder", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [1],
          channelCount,
          channelCountMode: "explicit",
        });
        const silentSink = context.createGain();
        silentSink.gain.value = 0;

        const buffers: Float32Array[][] = [];
        node.port.onmessage = (event: MessageEvent<Float32Array[]>) => {
          const frameChannels = event.data;
          if (!Array.isArray(frameChannels) || frameChannels.length === 0) {
            return;
          }
          while (buffers.length < frameChannels.length) buffers.push([]);
          for (let c = 0; c < frameChannels.length; c++) {
            buffers[c]!.push(frameChannels[c]!);
          }
        };

        source.connect(node);
        node.connect(silentSink);
        silentSink.connect(context.destination);

        const activeContext = context;
        let closed = false;
        const teardown = async () => {
          if (closed) return;
          closed = true;
          node.port.onmessage = null;
          source.disconnect();
          node.disconnect();
          silentSink.disconnect();
          media.getTracks().forEach((item) => item.stop());
          await activeContext.close().catch(() => undefined);
        };

        return {
          async stop(): Promise<File | null> {
            const sampleRate = activeContext.sampleRate;
            await teardown();
            if (buffers.length === 0 || buffers[0]!.length === 0) return null;
            const merged = buffers.map(concatChunks);
            const selected = selectRecordedChannels(merged);
            if (selected.length === 0 || selected[0]!.length === 0) return null;
            const wav = encodeChannelWav(selected, sampleRate);
            return new File([wav], "my-voice.wav", { type: "audio/wav" });
          },
          abort() {
            void teardown();
          },
        };
      }
      await context.close().catch(() => undefined);
    } catch {
      await context?.close().catch(() => undefined);
    }
  }
  return startMediaRecorderFallback(media, onError);
}
