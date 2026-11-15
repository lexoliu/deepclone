import { useEffect, useRef, useState } from "react";
import type { Command, Event, ModelEntry } from "./contracts";
import { Player, referenceAudio } from "./audio";
export function useVoice() {
  const worker = useRef<Worker | null>(null);
  const player = useRef<Player | null>(null);
  const pending = useRef(
    new Map<string, { resolve: () => void; reject: (error: Error) => void }>(),
  );
  const sequence = useRef(0);
  const audioUrl = useRef<string | null>(null);
  const [status, setStatus] = useState("Choose a model to begin");
  const [loaded, setLoaded] = useState(false);
  const [loadedSize, setLoadedSize] = useState<string | null>(null);
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
    setError(value.message);
    setBusy(false);
    setRunning(false);
    setStatus("Operation stopped");
    for (const item of pending.current.values()) item.reject(value);
    pending.current.clear();
  }
  useEffect(() => {
    const instance = new Worker(
      new URL("./inference.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.current = instance;
    instance.onerror = (event) =>
      fail(new Error(event.message || "Inference worker failed."));
    instance.onmessage = (message: MessageEvent<Event>) => {
      const event = message.data;
      if (event.kind === "error") {
        fail(new Error(event.message));
        return;
      }
      if (event.kind === "progress") {
        setProgress((event.completed / event.total) * 100);
        setStatus(event.phase);
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
        setLoaded(true);
        setBusy(false);
        setStatus("Model ready");
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
    return () => {
      instance.terminate();
      worker.current = null;
      void player.current?.close();
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
    };
  }, []);
  async function load(model: ModelEntry, context: number, seed: bigint) {
    setError(null);
    setBusy(true);
    setLoaded(false);
    setProgress(0);
    setStatus("Preparing local model");
    const ready = wait("ready");
    send({ kind: "load", model, context, seed });
    try {
      await ready;
      setLoadedSize(model.size);
    } catch (reason) {
      fail(reason);
    }
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
    frames: number,
    chunkFrames: number,
    live: boolean,
  ) {
    setError(null);
    setBusy(true);
    setSeconds(0);
    setAudio(null);
    setStatus("Preparing reference voice");
    try {
      if (player.current) await player.current.close();
      player.current = new Player();
      await player.current.resume();
      const samples = await referenceAudio(file);
      const conditioned = wait("conditioned");
      send({ kind: "reference", samples, transcript }, [samples.buffer]);
      await conditioned;
      const started = wait("started");
      send({ kind: "start", language, frames, chunkFrames });
      await started;
      await push(text);
      if (!live) send({ kind: "finish" });
    } catch (reason) {
      fail(reason);
    }
  }
  function finish() {
    send({ kind: "finish" });
    setStatus("Finishing speech");
  }
  function cancel() {
    send({ kind: "cancel" });
    void player.current?.close();
    setRunning(false);
    setBusy(false);
    setStatus("Cancelled");
  }
  return {
    status,
    loaded,
    loadedSize,
    busy,
    running,
    progress,
    error,
    audio,
    seconds,
    load,
    start,
    push,
    finish,
    cancel,
  };
}
