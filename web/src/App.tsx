import { useRef, useState } from "react";
import {
  ArrowDownIcon,
  ArrowRightIcon,
  SpeakerLoudIcon,
  StopIcon,
  UploadIcon,
} from "@radix-ui/react-icons";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { useVoice } from "@/runtime/use-voice";

import { ReferenceRecorder } from "./ReferenceRecorder";

export default function App() {
  const [referenceMode, setReferenceMode] = useState<"record" | "upload">(
    "record",
  );
  const voice = useVoice();
  const [file, setFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState("");
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("auto");
  const [live, setLive] = useState(false);
  const [inputEnded, setInputEnded] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const sent = useRef("");
  const sending = useRef(Promise.resolve());
  const picker = useRef<HTMLInputElement>(null);
  const supported =
    typeof navigator !== "undefined" &&
    Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
  const canGenerate =
    supported && file && transcript.trim() && text.trim() && !voice.busy;
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
        .catch((error) => {
          console.error("Text stream failed", error);
          setFormError("Text could not be sent. Please try again.");
        });
    }
    setText(value);
  }
  async function generate() {
    if (!file) return;
    setFormError(null);
    sent.current = text;
    setInputEnded(false);
    await voice.start(file, transcript, text, language, live);
  }
  const error = formError || voice.error;
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
      <div className="mx-auto max-w-6xl">
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
              This browser cannot generate speech on your device. Try an updated
              browser with hardware acceleration enabled.
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
          {voice.busy && !voice.loaded ? (
            <div className="mt-6" role="status">
              <p className="mb-2 text-xs text-muted-foreground">
                Preparing your voice workspace…
              </p>
              <Progress
                value={voice.progress}
                aria-label="Preparation progress"
              />
            </div>
          ) : null}
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
