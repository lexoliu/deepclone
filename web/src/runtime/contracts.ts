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
export type Command =
  | { kind: "load"; model: ModelEntry; context: number; seed: bigint }
  | { kind: "reference"; samples: Float32Array; transcript: string }
  | { kind: "start"; language: string; frames: number; chunkFrames: number }
  | { kind: "text"; text: string; id: number }
  | { kind: "finish" }
  | { kind: "ack"; id: number };
export type Event =
  | { kind: "progress"; completed: number; total: number; phase: string }
  | { kind: "ready" }
  | { kind: "conditioned" }
  | { kind: "started" }
  | { kind: "done" }
  | { kind: "accepted"; id: number }
  | { kind: "audio"; id: number; samples: Float32Array; sampleRate: number }
  | { kind: "error"; message: string };
export interface WasmInput {
  push(text: string): Promise<void>;
  finish(): void;
  cancel(): void;
  free(): void;
}
export interface WasmVoice {
  reference(samples: Float32Array, rate: number, text: string): Promise<void>;
  run(
    input: WasmInput,
    language: string,
    frames: number,
    chunkFrames: number,
    callback: (samples: Float32Array, rate: number) => Promise<void>,
  ): Promise<void>;
  free(): void;
}
export interface WasmBuilder {
  upload(bytes: Uint8Array, conditioning: boolean): void;
  finish(
    config: string,
    codec: string,
    mel: string,
    tokenizer: Uint8Array,
    context: number,
    seed: bigint,
  ): WasmVoice;
  free(): void;
}
export interface WasmManifest {
  url: string;
}
export interface WasmModule {
  default(options?: { module_or_path: string }): Promise<unknown>;
  BrowserRuntime: { create(configuration: string): Promise<WasmRuntime> };
  TextInput: new (capacity: number) => WasmInput;
}
export interface WasmRuntime {
  builder(fp16: boolean): WasmBuilder;
  free(): void;
}
