import { useEffect, useRef, useState } from "react";
import {
  isCurrentConditioning,
  type Command,
  type Event,
  type ModelEntry,
  type SavedConditioning,
} from "./contracts";
import { Player, referenceAudio } from "./audio";

export type VoicePhase =
  | "ready"
  | "conditioning"
  | "generating"
  | "playing"
  | "done"
  | "cancelled"
  | "incomplete"
  | "stopped";

export type VoiceErrorKind =
  "prepare_failed" | "stopped_empty" | "stopped_partial";

interface ConditionedResult {
  conditioning?: SavedConditioning;
  recycle?: boolean;
  recycleSamples?: Float32Array;
}

export function useVoice(referenceReady = true, hasCachedConditioning = false) {
  const worker = useRef<Worker | null>(null);
  const player = useRef<Player | null>(null);
  const pending = useRef(
    new Map<
      string,
      {
        resolve: (value?: ConditionedResult) => void;
        reject: (error: Error) => void;
      }
    >(),
  );
  const preparation = useRef<Promise<void> | null>(null);
  const catalogAbort = useRef<AbortController | null>(null);
  const prepared = useRef(false);
  const hasCachedConditioningRef = useRef(hasCachedConditioning);
  hasCachedConditioningRef.current = hasCachedConditioning;
  const reference = useRef<{ file: File; transcript: string } | null>(null);
  const sequence = useRef(0);
  const cancelling = useRef(false);
  const operation = useRef(0);
  const audioUrl = useRef<string | null>(null);
  const timing = useRef<{
    requested: number;
    ready?: number;
    conditioned?: number;
    firstAudio?: number;
  } | null>(null);
  const [phase, setPhase] = useState<VoicePhase>("ready");
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorKind, setErrorKind] = useState<VoiceErrorKind | null>(null);
  const [audio, setAudio] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);

  function send(command: Command, transfer: Transferable[] = []) {
    if (!worker.current) throw new Error("Inference worker is unavailable.");
    worker.current.postMessage(command, transfer);
  }
  function wait(key: string) {
    return new Promise<ConditionedResult | undefined>((resolve, reject) =>
      pending.current.set(key, { resolve, reject }),
    );
  }
  function fail(reason: unknown) {
    const value = reason instanceof Error ? reason : new Error(String(reason));
    console.error("Voice workspace failed", value);
    if (cancelling.current) return;
    operation.current++;
    const active = player.current;
    const partial = saveOutput(active);
    stopWorker(value);
    player.current = null;
    void active?.close();
    setErrorKind(
      partial
        ? "stopped_partial"
        : timing.current?.conditioned === undefined
          ? "prepare_failed"
          : "stopped_empty",
    );
    setBusy(false);
    setRunning(false);
    setPhase(partial ? "incomplete" : "stopped");
  }
  function saveOutput(active: Player | null) {
    if (!active?.hasAudio) return false;
    if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
    audioUrl.current = URL.createObjectURL(active.wav());
    setAudio(audioUrl.current);
    console.info("Speech playback timing", active.diagnostics());
    return true;
  }
  function stopWorker(
    reason: Error = new DOMException("Cancelled", "AbortError"),
  ) {
    catalogAbort.current?.abort();
    if (worker.current) {
      worker.current.onmessage = null;
      worker.current.onerror = null;
      worker.current.terminate();
      worker.current = null;
    }
    for (const request of pending.current.values()) request.reject(reason);
    pending.current.clear();
    preparation.current = null;
    prepared.current = false;
    reference.current = null;
    setLoaded(false);
  }
  function initializeWorker(autoPrepare = true, skipVoice?: boolean) {
    const instance = new Worker(
      new URL("./inference.worker.ts", import.meta.url),
      { type: "module" },
    );
    worker.current = instance;
    const workerFailed = (reason: Error) => {
      if (instance !== worker.current) return;
      fail(reason);
    };
    instance.onerror = (event) =>
      workerFailed(new Error(event.message || "Inference worker failed."));
    instance.onmessage = (message: MessageEvent<Event>) => {
      if (instance !== worker.current) return;
      const event = message.data;
      if (event.kind === "error") {
        workerFailed(new Error(event.message));
        return;
      }
      if (event.kind === "progress") {
        setProgress((event.completed / event.total) * 100);
        return;
      }
      if (event.kind === "audio") {
        if (timing.current && timing.current.firstAudio === undefined) {
          timing.current.firstAudio = performance.now();
          console.info("Speech first audio", {
            elapsedMs: timing.current.firstAudio - timing.current.requested,
          });
        }
        setPhase("playing");
        setSeconds((value) => value + event.samples.length / event.sampleRate);
        void player
          .current!.enqueue(event.samples, event.sampleRate)
          .then(() => {
            if (instance !== worker.current) return;
            instance.postMessage({
              kind: "ack",
              id: event.id,
            } satisfies Command);
          })
          .catch((reason) => {
            if (instance === worker.current) fail(reason);
          });
        return;
      }
      if (event.kind === "ready") {
        prepared.current = true;
        setLoaded(true);
      }
      if (event.kind === "started") {
        setRunning(true);
        setPhase("generating");
      }
      if (event.kind === "done") {
        const completed = performance.now();
        const times = timing.current;
        if (times)
          console.info("Speech generation timing", {
            totalMs: completed - times.requested,
            preparationMs: times.ready! - times.requested,
            conditioningMs: times.conditioned! - times.ready!,
            inferenceMs: completed - times.conditioned!,
            firstAudioMs:
              times.firstAudio === undefined
                ? null
                : times.firstAudio - times.requested,
          });
        const active = player.current;
        void active
          ?.finish()
          .then(() => {
            if (instance !== worker.current || cancelling.current) return;
            saveOutput(active);
            player.current = null;
            void active.close();
            setRunning(false);
            setBusy(false);
            setPhase("done");
          })
          .catch((reason) => {
            if (instance === worker.current) fail(reason);
          });
      }
      const key = event.kind === "accepted" ? `text:${event.id}` : event.kind;
      pending.current.get(key)?.resolve(
        event.kind === "conditioned"
          ? {
              conditioning: event.conditioning,
              recycle: event.recycle,
              recycleSamples: event.recycleSamples,
            }
          : undefined,
      );
      pending.current.delete(key);
    };
    if (autoPrepare && (navigator as Navigator & { gpu?: unknown }).gpu) {
      const deferVoice = skipVoice ?? !hasCachedConditioningRef.current;
      void ensurePrepared(deferVoice).catch((reason) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError"))
          console.error("Background preparation failed", reason);
      });
    }
  }
  useEffect(() => {
    if (!referenceReady) return;
    initializeWorker(true, !hasCachedConditioningRef.current);
    return () => {
      operation.current++;
      stopWorker();
      void player.current?.close();
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
    };
  }, [referenceReady]);
  function ensurePrepared(skipVoice = false): Promise<void> {
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
      send({ kind: "load", model, context: 2048, seed: 42n, skipVoice });
      await ready;
    })();
    preparation.current = task;
    void task.catch(() => {
      if (preparation.current === task) preparation.current = null;
    });
    return task;
  }
  async function recycleWorker(
    skipVoice: boolean,
    current: number,
  ): Promise<boolean> {
    stopWorker();
    if (current !== operation.current) return false;
    initializeWorker(false);
    await ensurePrepared(skipVoice);
    return current === operation.current;
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
    language = "auto",
    savedConditioning?: SavedConditioning,
    onConditioned?: (conditioning: SavedConditioning) => void,
  ) {
    cancelling.current = false;
    setErrorKind(null);
    setBusy(true);
    setSeconds(0);
    if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
    audioUrl.current = null;
    setAudio(null);
    setPhase("conditioning");
    const current = ++operation.current;
    const times = { requested: performance.now() } as NonNullable<
      typeof timing.current
    >;
    timing.current = times;
    try {
      const validConditioning = isCurrentConditioning(savedConditioning)
        ? savedConditioning
        : undefined;
      const skipInitialVoice = !validConditioning;
      if (!worker.current) initializeWorker(true, skipInitialVoice);
      const previous = player.current;
      const active = new Player();
      player.current = active;
      // Resume during the click gesture, before waiting for background preparation.
      const reusable =
        reference.current?.file === file &&
        reference.current.transcript === transcript;
      const [, , , samples] = await Promise.all([
        active.resume(),
        previous?.close(),
        ensurePrepared(skipInitialVoice),
        reusable ? Promise.resolve(null) : referenceAudio(file),
      ]);
      if (current !== operation.current) return;
      times.ready = performance.now();
      if (samples) {
        reference.current = null;
        let currentSamples = samples;
        let activeConditioning = validConditioning;
        const firstConditioned = wait("conditioned");
        send(
          {
            kind: "reference",
            samples: currentSamples,
            transcript,
            conditioning: activeConditioning,
          },
          [currentSamples.buffer],
        );
        let result = await firstConditioned;
        if (current !== operation.current) return;
        if (result?.recycleSamples) {
          // Cached conditioning did not match the loaded model revision.
          currentSamples = result.recycleSamples;
          activeConditioning = undefined;
          if (!(await recycleWorker(true, current))) return;
          const extractedWait = wait("conditioned");
          send(
            {
              kind: "reference",
              samples: currentSamples,
              transcript,
            },
            [currentSamples.buffer],
          );
          result = await extractedWait;
          if (current !== operation.current) return;
        }
        if (result?.conditioning) {
          onConditioned?.(result.conditioning);
          activeConditioning = result.conditioning;
        }
        if (result?.recycle) {
          // Terminate the conditioning worker so Safari frees its GPUDevice
          // immediately, then load synthesis weights once in a fresh worker.
          if (!(await recycleWorker(false, current))) return;
          const empty = new Float32Array(0);
          const readyReference = wait("conditioned");
          send(
            {
              kind: "reference",
              samples: empty,
              transcript,
              conditioning: activeConditioning,
            },
            [empty.buffer],
          );
          await readyReference;
          if (current !== operation.current) return;
        }
        reference.current = { file, transcript };
      }
      times.conditioned = performance.now();
      if (current !== operation.current) return;
      const started = wait("started");
      // Bound each utterance and deliver roughly one second per audio chunk.
      send({ kind: "start", language, frames: 512, chunkFrames: 12, text });
      await started;
      if (current !== operation.current) return;
      await push(text);
      if (current !== operation.current) return;
      send({ kind: "finish" });
    } catch (reason) {
      if (current === operation.current) {
        fail(reason);
      }
    }
  }
  function cancel() {
    operation.current++;
    cancelling.current = true;
    const active = player.current;
    const partial = saveOutput(active);
    stopWorker();
    player.current = null;
    void active?.close();
    setRunning(false);
    setBusy(false);
    setErrorKind(null);
    setPhase(partial ? "incomplete" : "cancelled");
    initializeWorker(true, !hasCachedConditioningRef.current);
  }
  return {
    phase,
    loaded,
    busy,
    running,
    progress,
    errorKind,
    audio,
    seconds,
    start,
    cancel,
  };
}
