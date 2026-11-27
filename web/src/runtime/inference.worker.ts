/// <reference lib="webworker" />
import type {
  Command,
  Event,
  Manifest,
  ModelEntry,
  SavedConditioning,
  WasmInput,
  WasmManifest,
  WasmModule,
  WasmRuntime,
  WasmVoice,
} from "./contracts";
import {
  TEXT_EMBEDDING_TENSOR,
  ensureStoredAsset,
  findTextEmbedding,
  manifest,
  openModelStorage,
  readTextEmbeddingRows,
  uploadShardTensors,
  type ScratchBuffer,
  type StoredFile,
  type TextEmbeddingDescriptor,
} from "./assets";

interface LoadedPackage {
  entry: ModelEntry;
  manifest: Manifest;
  files: Map<string, StoredFile>;
  config: string;
  codec: string;
  mel: string;
  textEmbedding: TextEmbeddingDescriptor;
  context: number;
  seed: bigint;
}

const SYNTHESIS_SKIP_TENSORS = new Set([TEXT_EMBEDDING_TENSOR]);
const CONDITIONING_SKIP_TENSORS = new Set<string>();

const scope = self as DedicatedWorkerGlobalScope;
let module: WasmModule | undefined;
let runtime: WasmRuntime | undefined;
let voice: WasmVoice | undefined;
let loadedPackage: LoadedPackage | undefined;
let input: WasmInput | undefined;
let operation: AbortController | undefined;
let running = false;
let audioId = 0;
const pending = new Map<
  number,
  { resolve: () => void; reject: (reason: Error) => void }
>();

function send(event: Event, transfer: Transferable[] = []) {
  scope.postMessage(event, transfer);
}

function getFile(pkg: LoadedPackage, path: string): StoredFile {
  const file = pkg.files.get(path);
  if (!file) throw new Error(`Missing model file: ${path}`);
  return file;
}

async function buildSynthesisVoice(pkg: LoadedPackage): Promise<WasmVoice> {
  if (!runtime) throw new Error("GPU runtime is not initialized.");
  const tokenizer = await getFile(pkg, "tokenizer.json").readAll();
  const builder = runtime.builder(true);
  const scratch: ScratchBuffer = { buffer: new Uint8Array(0) };
  let consumed = false;
  try {
    for (const shard of pkg.manifest.shards) {
      if (shard.conditioning) continue;
      await uploadShardTensors(
        getFile(pkg, shard.path),
        false,
        SYNTHESIS_SKIP_TENSORS,
        (name, dtype, shape, bytes, conditioning) => {
          builder.upload_tensor(name, dtype, shape, bytes, conditioning);
        },
        scratch,
      );
    }
    scratch.buffer = new Uint8Array(0);
    consumed = true;
    return builder.finish_synthesis(
      pkg.config,
      pkg.codec,
      pkg.mel,
      tokenizer,
      pkg.context,
      pkg.seed,
    );
  } finally {
    scratch.buffer = new Uint8Array(0);
    if (!consumed) builder.free();
  }
}

async function extractConditioning(
  pkg: LoadedPackage,
  samples: Float32Array,
): Promise<SavedConditioning> {
  if (!runtime) throw new Error("GPU runtime is not initialized.");
  const builder = runtime.builder(true);
  const scratch: ScratchBuffer = { buffer: new Uint8Array(0) };
  let consumed = false;
  let conditioner;
  try {
    for (const shard of pkg.manifest.shards) {
      if (!shard.conditioning) continue;
      await uploadShardTensors(
        getFile(pkg, shard.path),
        true,
        CONDITIONING_SKIP_TENSORS,
        (name, dtype, shape, bytes, conditioning) => {
          builder.upload_tensor(name, dtype, shape, bytes, conditioning);
        },
        scratch,
      );
    }
    scratch.buffer = new Uint8Array(0);
    consumed = true;
    conditioner = builder.finish_conditioning(pkg.config, pkg.codec, pkg.mel);
  } finally {
    scratch.buffer = new Uint8Array(0);
    if (!consumed) builder.free();
  }
  try {
    const extracted = await conditioner.extract(samples, 24000);
    if (extracted.speakerShape.length !== 3) {
      throw new Error("Invalid speaker embedding shape.");
    }
    return {
      modelSha256: pkg.entry.sha256,
      speaker: extracted.speaker,
      speakerShape: [
        extracted.speakerShape[0]!,
        extracted.speakerShape[1]!,
        extracted.speakerShape[2]!,
      ],
      codes: extracted.codes,
      groups: extracted.groups,
    };
  } finally {
    conditioner.free();
    runtime.reclaim_memory();
  }
}

async function execute(command: Command): Promise<void> {
  switch (command.kind) {
    case "load": {
      if (running || operation)
        throw new Error("An operation is already running.");
      operation = new AbortController();
      try {
        if (voice) {
          voice.free();
          voice = undefined;
          runtime?.reclaim_memory();
        }
        loadedPackage = undefined;
        if (!module) {
          const url = "/wasm/deepclone.js";
          const [loaded, wasmResponse] = await Promise.all([
            import(/* @vite-ignore */ url) as Promise<WasmModule>,
            fetch("/wasm.json", { signal: operation.signal }),
          ]);
          if (!wasmResponse.ok)
            throw new Error("WebAssembly configuration could not be loaded.");
          const wasmManifest = (await wasmResponse.json()) as WasmManifest;
          if (typeof wasmManifest.url !== "string" || !wasmManifest.url)
            throw new Error("Invalid WebAssembly configuration.");
          await loaded.default({ module_or_path: wasmManifest.url });
          module = loaded;
        }
        const model = await manifest(command.model, operation.signal);
        const directory = await openModelStorage(command.model);
        const total = model.files.reduce((sum, f) => sum + f.size, 0);
        const received = new Map<string, number>();
        const storedFiles = new Map<string, StoredFile>();
        const reportProgress = (path: string, bytes: number) => {
          received.set(path, bytes);
          const completed = Array.from(received.values()).reduce(
            (sum, value) => sum + value,
            0,
          );
          send({
            kind: "progress",
            completed,
            total,
            phase: "Preparing workspace",
          });
        };
        for (const file of model.files) {
          const stored = await ensureStoredAsset(
            directory,
            command.model,
            file,
            operation.signal,
            () => new module!.Sha256Hasher(),
            (bytes) => reportProgress(file.path, bytes),
          );
          storedFiles.set(file.path, stored);
        }
        let textEmbedding: TextEmbeddingDescriptor | null = null;
        for (const shard of model.shards) {
          if (shard.conditioning) continue;
          const file = storedFiles.get(shard.path);
          if (!file) throw new Error(`Missing model file: ${shard.path}`);
          const found = await findTextEmbedding(file);
          if (found) {
            textEmbedding = found;
            break;
          }
        }
        if (!textEmbedding) {
          throw new Error("Model package is missing text embeddings.");
        }
        const decoder = new TextDecoder();
        const readText = async (path: string) => {
          const file = storedFiles.get(path);
          if (!file) throw new Error(`Missing model file: ${path}`);
          return decoder.decode(await file.readAll());
        };
        const [config, codec, mel] = await Promise.all([
          readText("config.json"),
          readText("codec.json"),
          readText("mel.json"),
        ]);
        const pkg: LoadedPackage = {
          entry: command.model,
          manifest: model,
          files: storedFiles,
          config,
          codec,
          mel,
          textEmbedding,
          context: command.context,
          seed: command.seed,
        };
        if (!command.skipVoice) {
          if (!runtime) {
            const response = await fetch("/runtime.json", {
              signal: operation.signal,
            });
            if (!response.ok)
              throw new Error("GPU runtime configuration could not be loaded.");
            runtime = await module.BrowserRuntime.create(await response.text());
          }
          voice = await buildSynthesisVoice(pkg);
        }
        loadedPackage = pkg;
        operation.signal.throwIfAborted();
        send({ kind: "ready" });
      } finally {
        operation = undefined;
      }
      return;
    }
    case "reference": {
      if (!loadedPackage || !module || running || operation)
        throw new Error("Load a model before preparing a voice.");
      operation = new AbortController();
      try {
        const cached = command.conditioning;
        if (cached && cached.modelSha256 === loadedPackage.entry.sha256) {
          if (!voice) {
            if (!runtime) {
              const response = await fetch("/runtime.json", {
                signal: operation.signal,
              });
              if (!response.ok)
                throw new Error(
                  "GPU runtime configuration could not be loaded.",
                );
              runtime = await module.BrowserRuntime.create(
                await response.text(),
              );
            }
            voice = await buildSynthesisVoice(loadedPackage);
          }
          await voice.set_reference_conditioning(
            cached.speaker,
            Uint32Array.from(cached.speakerShape),
            cached.codes,
            cached.groups,
            command.transcript,
          );
          operation.signal.throwIfAborted();
          send({ kind: "conditioned" });
          return;
        }
        if (voice) {
          // Ask the coordinator to terminate this worker so Safari destroys the
          // synthesis GPUDevice before running FP32 conditioning in a fresh one.
          send({ kind: "conditioned", recycleSamples: command.samples }, [
            command.samples.buffer,
          ]);
          return;
        }
        if (!runtime) {
          const response = await fetch("/runtime.json", {
            signal: operation.signal,
          });
          if (!response.ok)
            throw new Error("GPU runtime configuration could not be loaded.");
          runtime = await module.BrowserRuntime.create(await response.text());
        }
        const extracted = await extractConditioning(
          loadedPackage,
          command.samples,
        );
        operation.signal.throwIfAborted();
        // Signal the coordinator to terminate this conditioning worker so
        // Safari immediately destroys its GPUDevice before synthesis loads.
        send({ kind: "conditioned", conditioning: extracted, recycle: true }, [
          extracted.speaker.buffer,
          extracted.codes.buffer,
        ]);
      } finally {
        operation = undefined;
      }
      return;
    }
    case "start": {
      if (!voice || !module || !loadedPackage || running || operation)
        throw new Error("The voice is not ready.");
      running = true;
      try {
        if (command.text !== undefined) {
          const tokenIds = voice.required_text_tokens(
            command.text,
            command.frames,
          );
          const bytes = await readTextEmbeddingRows(
            loadedPackage.textEmbedding,
            tokenIds,
          );
          voice.set_text_embeddings(
            tokenIds,
            loadedPackage.textEmbedding.dtype,
            bytes,
            loadedPackage.textEmbedding.width,
          );
        }
        input = new module.TextInput(2);
        send({ kind: "started" });
        let resumed = performance.now();
        let chunks = 0;
        await voice.run(
          input,
          command.language,
          command.frames,
          command.chunkFrames,
          async (samples, sampleRate) => {
            const ready = performance.now();
            const productionMs = ready - resumed;
            const audioSeconds = samples.length / sampleRate;
            const id = audioId++;
            const acknowledged = new Promise<void>((resolve, reject) =>
              pending.set(id, { resolve, reject }),
            );
            send({ kind: "audio", id, samples, sampleRate }, [samples.buffer]);
            await acknowledged;
            resumed = performance.now();
            console.info("Speech chunk timing", {
              chunk: chunks++,
              productionMs,
              consumerWaitMs: resumed - ready,
              audioSeconds,
            });
          },
        );
        send({ kind: "done" });
      } finally {
        input?.free();
        input = undefined;
        running = false;
      }
      return;
    }
    case "text":
      if (!input) throw new Error("No active text stream.");
      await input.push(command.text);
      send({ kind: "accepted", id: command.id });
      return;
    case "finish":
      if (!input) throw new Error("No active text stream.");
      input.finish();
      return;
    case "ack": {
      const request = pending.get(command.id);
      if (!request) throw new Error("Unknown audio acknowledgement.");
      pending.delete(command.id);
      request.resolve();
      return;
    }
  }
}
scope.onmessage = (event: MessageEvent<Command>) => {
  void execute(event.data).catch((error) => {
    send({
      kind: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  });
};
