import { useEffect, useRef, useState } from "react";
import type { Command, Event, ModelEntry } from "./contracts";
import { Player, referenceAudio } from "./audio";
export function useVoice() {
  const worker = useRef<Worker | null>(null);
  const player = useRef<Player | null>(null);
  const pending = useRef(
    new Map<string, { resolve: () => void; reject: (error: Error) => void }>(),
  );
  const preparation = useRef<Promise<void> | null>(null);
  const catalogAbort = useRef<AbortController | null>(null);
  const prepared = useRef(false);
  const sequence = useRef(0);
  const cancelling = useRef(false);
  const operation = useRef(0);
  const audioUrl = useRef<string | null>(null);
  const [status, setStatus] = useState("Your voice, ready for something new");
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [audio, setAudio] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);
  function send(command: Command, transfer: Transferable[] = []) {
    if (!worker.current) throw new Error("Inference worker is unavailable.");
    worker.current.postMessage(command, transfer);
  }
  function wait(key: string) {
    return new Promise<void>((resolve, reject) =>
      pending.current.set(key, { resolve, reject }),
    );
  }
  function fail(reason: unknown) {
    const value = reason instanceof Error ? reason : new Error(String(reason));
    console.error("Voice workspace failed", value);
    if (cancelling.current) return;
    setError("Speech could not be prepared. Please try again.");
    setBusy(false);
    setRunning(false);
    setStatus("Operation stopped");
  }
  useEffect(() => {
    const instance = new Worker(
      new URL("./inference.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.current = instance;
    const workerFailed = (reason: Error) => {
      for (const item of pending.current.values()) item.reject(reason);
      pending.current.clear();
      fail(reason);
    };
    instance.onerror = (event) =>
      workerFailed(new Error(event.message || "Inference worker failed."));
    instance.onmessage = (message: MessageEvent<Event>) => {
      const event = message.data;
      if (event.kind === "cancelled") {
        for (const request of pending.current.values())
          request.reject(new DOMException("Cancelled", "AbortError"));
        pending.current.clear();
        setRunning(false);
        setBusy(false);
        setStatus("Cancelled");
        return;
      }
      if (event.kind === "error") {
        workerFailed(new Error(event.message));
        return;
      }
      if (event.kind === "progress") {
        setProgress((event.completed / event.total) * 100);
        return;
      }
      if (event.kind === "audio") {
        setStatus("Generating and playing");
        setSeconds((value) => value + event.samples.length / event.sampleRate);
        void player
          .current!.enqueue(event.samples, event.sampleRate)
          .then(() =>
            instance.postMessage({
              kind: "ack",
              id: event.id,
            } satisfies Command),
          )
          .catch(fail);
        return;
      }
      if (event.kind === "ready") {
        prepared.current = true;
        setLoaded(true);
      }
      if (event.kind === "started") {
        setRunning(true);
        setStatus("Generating speech");
      }
      if (event.kind === "done") {
        const active = player.current;
        void active
          ?.finish()
          .then(() => {
            if (cancelling.current) return;
            if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
            audioUrl.current = URL.createObjectURL(active.wav());
            setAudio(audioUrl.current);
            setRunning(false);
            setBusy(false);
            setStatus("Ready to listen");
          })
          .catch(fail);
      }
      const key = event.kind === "accepted" ? `text:${event.id}` : event.kind;
      pending.current.get(key)?.resolve();
      pending.current.delete(key);
    };
    if ((navigator as Navigator & { gpu?: unknown }).gpu) {
      void ensurePrepared().catch((reason) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError"))
          console.error("Background preparation failed", reason);
      });
    }
    return () => {
      catalogAbort.current?.abort();
      for (const request of pending.current.values())
        request.reject(new DOMException("Workspace closed", "AbortError"));
      pending.current.clear();
      preparation.current = null;
      prepared.current = false;
      instance.terminate();
      worker.current = null;
      void player.current?.close();
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
    };
  }, []);
  function ensurePrepared(): Promise<void> {
    if (prepared.current) return Promise.resolve();
    if (preparation.current) return preparation.current;
    const controller = new AbortController();
    catalogAbort.current = controller;
    setProgress(0);
    const task = (async () => {
      const response = await fetch("/models.json", {
        signal: controller.signal,
      });
      if (!response.ok)
        throw new Error("The voice catalog could not be loaded.");
      const entries: ModelEntry[] = await response.json();
      const model = entries.find((entry) => entry.size === "0.6b");
      if (!model) throw new Error("The browser voice package is unavailable.");
      controller.signal.throwIfAborted();
      const ready = wait("ready");
      // Browser inference defaults: context capacity and reproducible sampling.
      send({ kind: "load", model, context: 2048, seed: 42n });
      await ready;
    })();
    preparation.current = task;
    void task.catch(() => {
      if (preparation.current === task) preparation.current = null;
    });
    return task;
  }
  async function push(text: string) {
    if (!text) return;
    const id = sequence.current++;
    const accepted = wait(`text:${id}`);
    send({ kind: "text", text, id });
    await accepted;
  }
  async function start(
    file: File,
    transcript: string,
    text: string,
    language: string,
    live: boolean,
  ) {
    cancelling.current = false;
    setError(null);
    setBusy(true);
    setSeconds(0);
    setAudio(null);
    setStatus("Preparing reference voice");
    const current = ++operation.current;
    try {
      const previous = player.current;
      const active = new Player();
      player.current = active;
      // Resume during the click gesture, before waiting for background preparation.
      const [, , , samples] = await Promise.all([
        active.resume(),
        previous?.close(),
        ensurePrepared(),
        referenceAudio(file),
      ]);
      if (current !== operation.current) return;
      const conditioned = wait("conditioned");
      send({ kind: "reference", samples, transcript }, [samples.buffer]);
      await conditioned;
      if (current !== operation.current) return;
      const started = wait("started");
      // Bound each utterance and deliver roughly one second per audio chunk.
      send({ kind: "start", language, frames: 512, chunkFrames: 12 });
      await started;
      if (current !== operation.current) return;
      await push(text);
      if (current !== operation.current) return;
      if (!live) send({ kind: "finish" });
    } catch (reason) {
      if (current === operation.current) {
        fail(reason);
        void player.current?.close();
      }
    }
  }
  function finish() {
    send({ kind: "finish" });
    setStatus("Finishing speech");
  }
  function cancel() {
    operation.current++;
    cancelling.current = true;
    catalogAbort.current?.abort();
    send({ kind: "cancel" });
    void player.current?.close();
    setStatus("Cancelling…");
  }
  return {
    status,
    loaded,
    busy,
    running,
    progress,
    error,
    audio,
    seconds,
    start,
    push,
    finish,
    cancel,
  };
}
