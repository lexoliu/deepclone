import { useEffect, useRef, useState } from "react";
import {
  ArrowDownIcon,
  ArrowRightIcon,
  SpeakerLoudIcon,
  StopIcon,
  UploadIcon,
} from "@radix-ui/react-icons";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { useVoice } from "@/runtime/use-voice";
import type { ModelEntry } from "@/runtime/contracts";

import { ReferenceRecorder } from "./ReferenceRecorder";

export default function App() {
  const [referenceMode, setReferenceMode] = useState<"record" | "upload">(
    "record",
  );
  const voice = useVoice();
  const [catalog, setCatalog] = useState<ModelEntry[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [size, setSize] = useState("0.6b");
  const [file, setFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState("");
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("auto");
  const [live, setLive] = useState(false);
  const [inputEnded, setInputEnded] = useState(false);
  const [frames, setFrames] = useState(512);
  const [context, setContext] = useState(2048);
  const [chunkFrames, setChunkFrames] = useState(12);
  const [seed, setSeed] = useState("42");
  const [formError, setFormError] = useState<string | null>(null);
  const sent = useRef("");
  const sending = useRef(Promise.resolve());
  const picker = useRef<HTMLInputElement>(null);
  const supported =
    typeof navigator !== "undefined" &&
    Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
  useEffect(() => {
    const controller = new AbortController();
    void fetch("/models.json", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok)
          throw new Error("The model catalog could not be loaded.");
        const entries: ModelEntry[] = await response.json();
        if (!entries.length) throw new Error("No models are available.");
        setCatalog(entries);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setCatalogError(String(error));
      });
    return () => controller.abort();
  }, []);
  const model = catalog.find((entry) => entry.size === size);
  const modelReady = voice.loaded && voice.loadedSize === size;
  const canGenerate =
    modelReady && file && transcript.trim() && text.trim() && !voice.busy;
  function updateText(value: string) {
    if (voice.running && live && !inputEnded) {
      if (!value.startsWith(sent.current)) {
        setFormError(
          "During streaming, append new text after the text already sent.",
        );
        return;
      }
      const delta = value.slice(sent.current.length);
      sent.current = value;
      sending.current = sending.current
        .then(() => voice.push(delta))
        .catch((error) => setFormError(String(error)));
    }
    setText(value);
  }
  async function generate() {
    if (!file) return;
    setFormError(null);
    sent.current = text;
    setInputEnded(false);
    await voice.start(
      file,
      transcript,
      text,
      language,
      frames,
      chunkFrames,
      live,
    );
  }
  const error = formError || voice.error || catalogError;
  return (
    <div className="min-h-[100dvh]">
      <header className="flex h-16 items-center justify-between border-b px-5 md:px-9">
        <a
          href="/"
          className="flex items-center gap-2.5 text-sm font-semibold tracking-tight"
        >
          <span className="flex size-7 items-center justify-center rounded-md border bg-white">
            <SpeakerLoudIcon />
          </span>
          deepclone
        </a>
        <span className="text-xs text-muted-foreground">
          Local voice workspace
        </span>
      </header>
      <div className="mx-auto grid max-w-7xl md:min-h-[calc(100dvh-64px)] md:grid-cols-[240px_1fr]">
        <aside className="border-b p-5 md:border-r md:border-b-0 md:p-7">
          <div className="mb-8 flex items-center gap-2 rounded-md bg-secondary/70 px-3 py-2 text-sm font-medium">
            <SpeakerLoudIcon />
            Voice cloning
          </div>
          <label htmlFor="model" className="mb-2 block text-xs font-medium">
            Model
          </label>
          <select
            id="model"
            value={size}
            disabled={voice.busy}
            onChange={(event) => setSize(event.target.value)}
            className="h-9 w-full rounded-md border bg-white px-2 text-sm"
          >
            <option value="0.6b">Qwen3-TTS · 0.6B</option>
            <option value="1.7b">Qwen3-TTS · 1.7B</option>
          </select>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">
            {size === "0.6b"
              ? "A smaller model for everyday use."
              : "A larger model with higher memory requirements."}{" "}
            Downloaded once and cached in this browser.
          </p>
          <Button
            className="mt-4 w-full"
            variant="outline"
            disabled={!model || !supported || voice.busy || modelReady}
            onClick={() =>
              model && void voice.load(model, context, BigInt(seed))
            }
          >
            {modelReady
              ? "Model loaded"
              : voice.busy
                ? "Loading…"
                : "Load model"}
          </Button>
          {voice.busy && !voice.loaded ? (
            <Progress
              className="mt-3 h-1"
              value={voice.progress}
              aria-label="Model loading progress"
            />
          ) : null}
          <div className="mt-7 border-t pt-5">
            <span className="text-xs font-medium">On this device</span>
            <p className="mt-2 text-xs leading-5 text-muted-foreground">
              Your reference recording and generated speech stay in your
              browser.
            </p>
          </div>
          <details className="mt-7 border-t pt-5">
            <summary className="cursor-pointer text-xs font-medium">
              Generation settings
            </summary>
            <div className="mt-4 space-y-4">
              {[
                { label: "Context tokens", value: context, set: setContext },
                {
                  label: "Maximum audio frames",
                  value: frames,
                  set: setFrames,
                },
                {
                  label: "Frames per audio chunk",
                  value: chunkFrames,
                  set: setChunkFrames,
                },
              ].map((setting) => (
                <div key={setting.label}>
                  <label className="mb-1 block text-xs" htmlFor={setting.label}>
                    {setting.label}
                  </label>
                  <Input
                    id={setting.label}
                    type="number"
                    min={1}
                    value={setting.value}
                    disabled={
                      voice.busy ||
                      (voice.loaded && setting.label === "Context tokens")
                    }
                    onChange={(event) =>
                      setting.set(Number(event.target.value))
                    }
                  />
                </div>
              ))}
              <div>
                <label htmlFor="seed" className="mb-1 block text-xs">
                  Random seed
                </label>
                <Input
                  id="seed"
                  inputMode="numeric"
                  value={seed}
                  disabled={voice.loaded || voice.busy}
                  onChange={(event) =>
                    setSeed(event.target.value.replace(/\D/g, ""))
                  }
                />
              </div>
            </div>
          </details>
        </aside>
        <main className="min-w-0 px-5 py-9 md:px-12 md:py-12">
          <div className="mb-10">
            <div className="mb-3 text-xs text-muted-foreground">
              Workspace / Voice cloning
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">
              Make it your voice.
            </h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Add a reference recording, then write something new.
            </p>
          </div>
          {!supported ? (
            <div
              role="alert"
              className="mb-7 rounded-md border border-destructive/30 bg-white p-4 text-sm"
            >
              WebGPU is unavailable in this browser. Open this page in a browser
              with WebGPU enabled.
            </div>
          ) : null}
          <div className="grid gap-9 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <section className="min-w-0">
              <div className="mb-5 flex items-center gap-2 text-sm font-medium">
                <span className="text-muted-foreground">01</span>Reference voice
              </div>
              <div
                className="mb-5 flex gap-1 rounded-md bg-secondary/60 p-1"
                role="group"
                aria-label="Reference source"
              >
                <Button
                  className="flex-1"
                  size="sm"
                  variant={referenceMode === "record" ? "outline" : "ghost"}
                  disabled={voice.busy}
                  onClick={() => {
                    if (referenceMode === "record") return;
                    setReferenceMode("record");
                    setFile(null);
                    setTranscript("");
                  }}
                >
                  Record my voice
                </Button>
                <Button
                  className="flex-1"
                  size="sm"
                  variant={referenceMode === "upload" ? "outline" : "ghost"}
                  disabled={voice.busy}
                  onClick={() => {
                    if (referenceMode === "upload") return;
                    setReferenceMode("upload");
                    setFile(null);
                    setTranscript("");
                  }}
                >
                  Upload audio
                </Button>
              </div>
              {referenceMode === "record" ? (
                <ReferenceRecorder
                  disabled={voice.busy}
                  onUse={(recording, words) => {
                    setFile(recording);
                    setTranscript(words);
                  }}
                />
              ) : (
                <>
                  <input
                    ref={picker}
                    type="file"
                    accept="audio/*"
                    className="sr-only"
                    aria-label="Reference recording"
                    disabled={voice.busy}
                    onChange={(event) =>
                      setFile(event.target.files?.[0] ?? null)
                    }
                  />
                  <button
                    type="button"
                    disabled={voice.busy}
                    onClick={() => picker.current?.click()}
                    onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => {
                      event.preventDefault();
                      if (!voice.busy)
                        setFile(event.dataTransfer.files[0] ?? null);
                    }}
                    className="flex min-h-36 w-full flex-col items-center justify-center rounded-lg border border-dashed bg-white px-5 text-center hover:bg-secondary/30 disabled:opacity-50"
                  >
                    <UploadIcon className="mb-3 size-5 text-muted-foreground" />
                    <span className="max-w-full truncate text-sm font-medium">
                      {file ? file.name : "Choose a recording"}
                    </span>
                    <span className="mt-1 text-xs text-muted-foreground">
                      {file
                        ? "Click to replace"
                        : "Drop an audio file here, or browse"}
                    </span>
                  </button>
                  <label
                    htmlFor="transcript"
                    className="mt-6 mb-2 block text-xs font-medium"
                  >
                    Reference transcript
                  </label>
                  <Textarea
                    id="transcript"
                    value={transcript}
                    onChange={(event) => setTranscript(event.target.value)}
                    disabled={voice.busy}
                    placeholder="The exact words spoken in your recording"
                    className="min-h-36 bg-white text-sm shadow-none"
                  />
                  <p className="mt-2 text-xs leading-5 text-muted-foreground">
                    Use a clear recording and its exact transcript for the
                    closest match.
                  </p>
                </>
              )}
            </section>
            <section className="min-w-0">
              <div className="mb-5 flex items-center gap-2 text-sm font-medium">
                <span className="text-muted-foreground">02</span>New speech
              </div>
              <label htmlFor="text" className="mb-2 block text-xs font-medium">
                What should your voice say?
              </label>
              <Textarea
                id="text"
                value={text}
                onChange={(event) => updateText(event.target.value)}
                disabled={voice.busy && (!voice.running || !live || inputEnded)}
                placeholder="Write the words you want to hear…"
                className="min-h-52 bg-white text-sm shadow-none"
              />
              <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
                <div>
                  <label
                    htmlFor="language"
                    className="mb-2 block text-xs font-medium"
                  >
                    Language
                  </label>
                  <select
                    id="language"
                    value={language}
                    disabled={voice.busy}
                    onChange={(event) => setLanguage(event.target.value)}
                    className="h-9 rounded-md border bg-white px-3 text-sm"
                  >
                    {[
                      "auto",
                      "English",
                      "Chinese",
                      "Japanese",
                      "Korean",
                      "French",
                      "German",
                      "Spanish",
                      "Italian",
                      "Portuguese",
                      "Russian",
                    ].map((value) => (
                      <option key={value} value={value}>
                        {value === "auto" ? "Automatic" : value}
                      </option>
                    ))}
                  </select>
                </div>
                <label className="flex items-center gap-2 pb-2 text-xs">
                  <input
                    type="checkbox"
                    checked={live}
                    disabled={voice.busy}
                    onChange={(event) => setLive(event.target.checked)}
                  />
                  Stream as I type
                </label>
              </div>
              <div className="mt-7 flex items-center gap-2">
                {voice.busy ? (
                  <>
                    <Button variant="outline" onClick={voice.cancel}>
                      <StopIcon />
                      Cancel
                    </Button>
                    {voice.running && live && !inputEnded ? (
                      <Button
                        onClick={() => {
                          setInputEnded(true);
                          void sending.current.then(voice.finish);
                        }}
                      >
                        Finish text
                        <ArrowRightIcon />
                      </Button>
                    ) : null}
                  </>
                ) : (
                  <Button
                    disabled={!canGenerate}
                    onClick={() => void generate()}
                  >
                    Generate speech
                    <ArrowRightIcon />
                  </Button>
                )}
              </div>
            </section>
          </div>
          {error ? (
            <div
              role="alert"
              className="mt-6 rounded-md border border-destructive/30 bg-white px-4 py-3 text-sm text-destructive"
            >
              {error}
            </div>
          ) : null}
          <section className="mt-10 border-t pt-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-sm font-medium">Output</h2>
                <p role="status" className="mt-1 text-xs text-muted-foreground">
                  {voice.status}
                  {voice.seconds > 0
                    ? ` · ${voice.seconds.toFixed(1)}s generated`
                    : ""}
                </p>
              </div>
              {voice.audio ? (
                <Button asChild variant="outline" size="sm">
                  <a href={voice.audio} download="deepclone.wav">
                    <ArrowDownIcon />
                    Download WAV
                  </a>
                </Button>
              ) : null}
            </div>
            {voice.audio ? (
              <audio controls src={voice.audio} className="mt-5 w-full" />
            ) : (
              <div className="mt-5 flex min-h-24 items-center justify-center gap-3 rounded-md bg-secondary/40 px-5 text-sm text-muted-foreground">
                <SpeakerLoudIcon />
                {voice.running
                  ? "Audio plays as it is generated."
                  : "Your generated speech will appear here."}
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
