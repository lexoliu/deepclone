/// <reference lib="webworker" />
import type {
  Command,
  Event,
  WasmModule,
  WasmVoice,
  WasmInput,
} from "./contracts";
import { asset, manifest } from "./assets";
const scope = self as DedicatedWorkerGlobalScope;
let module: WasmModule | undefined;
let voice: WasmVoice | undefined;
let input: WasmInput | undefined;
let operation: AbortController | undefined;
let running = false;
let cancelled = false;
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
      cancelled = false;
      operation = new AbortController();
      try {
        if (voice) {
          voice.free();
          voice = undefined;
        }
        if (!module) {
          const url = "/wasm/deepclone.js";
          module = (await import(/* @vite-ignore */ url)) as WasmModule;
          await module.default();
        }
        const model = await manifest(command.model, operation.signal);
        const total = model.files.reduce((sum, f) => sum + f.size, 0);
        let completed = 0;
        const files = new Map(model.files.map((file) => [file.path, file]));
        const read = async (path: string) => {
          const file = files.get(path);
          if (!file) throw new Error(`Missing model file: ${path}`);
          const bytes = await asset(command.model, file, operation!.signal);
          completed += bytes.length;
          send({ kind: "progress", completed, total, phase: "Loading model" });
          return bytes;
        };
        const builder = await module.ModelBuilder.create(true);
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
      cancelled = false;
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
      cancelled = false;
      running = true;
      input = new module.TextInput(2);
      send({ kind: "started" });
      try {
        await voice.run(
          input,
          command.language,
          command.frames,
          command.chunkFrames,
          async (samples, sampleRate) => {
            const id = audioId++;
            const copy = samples.slice();
            const acknowledged = new Promise<void>((resolve, reject) =>
              pending.set(id, { resolve, reject }),
            );
            send({ kind: "audio", id, samples: copy, sampleRate }, [
              copy.buffer,
            ]);
            await acknowledged;
          },
        );
        send({ kind: cancelled ? "cancelled" : "done" });
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
    case "cancel":
      cancelled = true;
      operation?.abort();
      input?.cancel();
      for (const p of pending.values()) p.reject(new Error("Cancelled"));
      if (!operation && !running) send({ kind: "cancelled" });
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
    if (cancelled) {
      if (!operation && !running) send({ kind: "cancelled" });
    } else {
      send({
        kind: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });
};
