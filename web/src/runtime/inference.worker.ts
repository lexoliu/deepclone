/// <reference lib="webworker" />
import type {
  Command,
  Event,
  WasmModule,
  WasmVoice,
  WasmInput,
  WasmRuntime,
} from "./contracts";
import { asset, manifest } from "./assets";
const scope = self as DedicatedWorkerGlobalScope;
let module: WasmModule | undefined;
let runtime: WasmRuntime | undefined;
let voice: WasmVoice | undefined;
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
        }
        if (!module) {
          const url = "/wasm/deepclone.js";
          const loaded = (await import(/* @vite-ignore */ url)) as WasmModule;
          await loaded.default();
          module = loaded;
        }
        const model = await manifest(command.model, operation.signal);
        const total = model.files.reduce((sum, f) => sum + f.size, 0);
        const received = new Map<string, number>();
        const files = new Map(model.files.map((file) => [file.path, file]));
        const read = async (path: string) => {
          const file = files.get(path);
          if (!file) throw new Error(`Missing model file: ${path}`);
          return asset(command.model, file, operation!.signal, (bytes) => {
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
          });
        };
        if (!runtime) {
          const response = await fetch("/runtime.json", {
            signal: operation.signal,
          });
          if (!response.ok)
            throw new Error("GPU runtime configuration could not be loaded.");
          runtime = await module.BrowserRuntime.create(await response.text());
        }
        const builder = runtime.builder(true);
        let consumed = false;
        try {
          for (const shard of model.shards) {
            const bytes = await read(shard.path);
            builder.upload(bytes, shard.conditioning);
          }
          const [config, codec, mel, tokenizer] = await Promise.all([
            read("config.json"),
            read("codec.json"),
            read("mel.json"),
            read("tokenizer.json"),
          ]);
          const decoder = new TextDecoder();
          consumed = true;
          voice = builder.finish(
            decoder.decode(config),
            decoder.decode(codec),
            decoder.decode(mel),
            tokenizer,
            command.context,
            command.seed,
          );
        } finally {
          if (!consumed) builder.free();
        }
        operation.signal.throwIfAborted();
        send({ kind: "ready" });
      } finally {
        operation = undefined;
      }
      return;
    }
    case "reference": {
      if (!voice || running || operation)
        throw new Error("Load a model before preparing a voice.");
      operation = new AbortController();
      try {
        await voice.reference(command.samples, 24000, command.transcript);
        operation.signal.throwIfAborted();
        send({ kind: "conditioned" });
      } finally {
        operation = undefined;
      }
      return;
    }
    case "start": {
      if (!voice || !module || running || operation)
        throw new Error("The voice is not ready.");
      running = true;
      input = new module.TextInput(2);
      send({ kind: "started" });
      try {
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
        input.free();
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
