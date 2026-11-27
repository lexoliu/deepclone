export interface ModelEntry {
  size: "0.6b" | "1.7b";
  manifest: string;
  sha256: string;
}
export interface Asset {
  path: string;
  size: number;
  sha256: string;
}
export interface Manifest {
  format: "deepclone-burn-v1";
  files: Asset[];
  shards: { path: string; conditioning: boolean }[];
}
export interface SavedConditioning {
  modelSha256: string;
  speaker: Float32Array;
  speakerShape: [number, number, number];
  codes: Uint32Array;
  groups: number;
}
export type Command =
  | {
      kind: "load";
      model: ModelEntry;
      context: number;
      seed: bigint;
      skipVoice?: boolean;
    }
  | {
      kind: "reference";
      samples: Float32Array;
      transcript: string;
      conditioning?: SavedConditioning;
    }
  | {
      kind: "start";
      language: string;
      frames: number;
      chunkFrames: number;
      text?: string;
    }
  | { kind: "text"; text: string; id: number }
  | { kind: "finish" }
  | { kind: "ack"; id: number };
export type Event =
  | { kind: "progress"; completed: number; total: number; phase: string }
  | { kind: "ready" }
  | {
      kind: "conditioned";
      conditioning?: SavedConditioning;
      recycle?: boolean;
      recycleSamples?: Float32Array;
    }
  | { kind: "started" }
  | { kind: "done" }
  | { kind: "accepted"; id: number }
  | { kind: "audio"; id: number; samples: Float32Array; sampleRate: number }
  | { kind: "error"; message: string };
export interface WasmSha256Hasher {
  update(chunk: Uint8Array): void;
  finalize(): string;
  free(): void;
}
export interface ExtractedConditioning {
  speaker: Float32Array;
  speakerShape: number[];
  codes: Uint32Array;
  groups: number;
  frames: number;
}
export interface WasmConditioner {
  extract(samples: Float32Array, rate: number): Promise<ExtractedConditioning>;
  free(): void;
}
export interface WasmInput {
  push(text: string): Promise<void>;
  finish(): void;
  cancel(): void;
  free(): void;
}
export interface WasmVoice {
  set_reference_conditioning(
    speaker: Float32Array,
    speakerShape: Uint32Array,
    codes: Uint32Array,
    groups: number,
    transcript: string,
  ): Promise<void>;
  required_text_tokens(text: string, maximumFrames: number): Uint32Array;
  set_text_embeddings(
    tokenIds: Uint32Array,
    dtype: string,
    bytes: Uint8Array,
    width: number,
  ): void;
  run(
    input: WasmInput,
    language: string,
    frames: number,
    chunkFrames: number,
    callback: (samples: Float32Array, rate: number) => Promise<void>,
  ): Promise<void>;
  free(): void;
}
export interface WasmManifest {
  js: string;
  url: string;
}
export interface WasmModule {
  default(options?: { module_or_path: string }): Promise<unknown>;
  BrowserRuntime: { create(configuration: string): Promise<WasmRuntime> };
  TextInput: new (capacity: number) => WasmInput;
  Sha256Hasher: new () => WasmSha256Hasher;
}
export type StreamingShardReader = (
  shard: number,
  offset: number,
  target: Uint8Array,
) => void;
export interface WasmRuntime {
  load_conditioning_streaming(
    indexJson: string,
    readInto: StreamingShardReader,
    config: string,
    codec: string,
    mel: string,
  ): WasmConditioner;
  load_synthesis_streaming(
    fp16: boolean,
    indexJson: string,
    readInto: StreamingShardReader,
    config: string,
    codec: string,
    mel: string,
    tokenizer: Uint8Array,
    context: number,
    seed: bigint,
  ): WasmVoice;
  reclaim_memory(): void;
  free(): void;
}
