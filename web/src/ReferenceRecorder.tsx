import { useEffect, useRef, useState } from "react";
import { CheckIcon, ReloadIcon, StopIcon } from "@radix-ui/react-icons";
import { Button } from "@/components/ui/button";
import {
  RECORDING_AUDIO_CONSTRAINTS,
  startMicRecorder,
  type ActiveRecording,
} from "@/runtime/recorder";
import { COPY, LANGUAGES, type LanguageId } from "./i18n";

function MicWaveIcon() {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="size-4"
      aria-hidden="true"
    >
      <rect x="7" y="2.5" width="6" height="9.5" rx="3" />
      <path d="M4.5 9.5a5.5 5.5 0 0 0 11 0" />
      <path d="M10 15v2.5" />
    </svg>
  );
}

export function ReferenceRecorder({
  language,
  disabled,
  saving,
  onSave,
  onRecordingChange,
}: {
  language: LanguageId;
  disabled: boolean;
  saving: boolean;
  onSave: (file: File, transcript: string, language: LanguageId) => void;
  onRecordingChange?: (active: boolean) => void;
}) {
  const copy = COPY[language];
  const passage = LANGUAGES.find((entry) => entry.id === language)!;
  const [phase, setPhase] = useState<
    "idle" | "requesting" | "recording" | "review"
  >("idle");
  const [errorKind, setErrorKind] = useState<
    "unavailable" | "denied" | "unsupported" | "stopped" | "empty" | null
  >(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);
  const recorder = useRef<ActiveRecording | null>(null);
  const recorded = useRef<File | null>(null);
  const url = useRef<string | null>(null);
  const alive = useRef(true);

  function clearPreview() {
    recorded.current = null;
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
    setPreview(null);
    setSeconds(0);
  }

  useEffect(() => {
    clearPreview();
    setErrorKind(null);
    setPhase("idle");
  }, [language]);

  useEffect(() => {
    onRecordingChange?.(phase === "recording" || phase === "requesting");
  }, [phase, onRecordingChange]);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      recorder.current?.abort();
      recorder.current = null;
      if (url.current) URL.revokeObjectURL(url.current);
    };
  }, []);

  useEffect(() => {
    if (phase !== "recording") return;
    const start = performance.now();
    const timer = setInterval(
      () => setSeconds((performance.now() - start) / 1000),
      100,
    );
    return () => clearInterval(timer);
  }, [phase]);

  async function start() {
    clearPreview();
    setErrorKind(null);
    setPhase("requesting");
    let media: MediaStream | null = null;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        setErrorKind("unavailable");
        setPhase("idle");
        return;
      }
      media = await navigator.mediaDevices.getUserMedia({
        audio: RECORDING_AUDIO_CONSTRAINTS,
      });
      if (!alive.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      const active = await startMicRecorder(media, () => {
        recorder.current = null;
        if (!alive.current) return;
        setErrorKind("stopped");
        setPhase("idle");
      });
      if (!alive.current) {
        active?.abort();
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      if (!active) {
        media.getTracks().forEach((track) => track.stop());
        setErrorKind("unsupported");
        setPhase("idle");
        return;
      }
      recorder.current = active;
      setPhase("recording");
    } catch (reason) {
      media?.getTracks().forEach((track) => track.stop());
      if (!alive.current) return;
      const denied =
        reason instanceof DOMException &&
        (reason.name === "NotAllowedError" ||
          reason.name === "PermissionDeniedError");
      setErrorKind(denied ? "denied" : "unavailable");
      setPhase("idle");
    }
  }

  async function stop() {
    const active = recorder.current;
    if (!active) return;
    recorder.current = null;
    const file = await active.stop();
    if (!alive.current) return;
    if (!file || !file.size) {
      setErrorKind("empty");
      setPhase("idle");
      return;
    }
    recorded.current = file;
    url.current = URL.createObjectURL(file);
    setPreview(url.current);
    setPhase("review");
  }

  const errorMessage =
    errorKind === "unavailable"
      ? copy.micUnavailable
      : errorKind === "denied"
        ? copy.micDenied
        : errorKind === "unsupported"
          ? copy.micEncodeUnsupported
          : errorKind === "stopped"
            ? copy.micStopped
            : errorKind === "empty"
              ? copy.micEmpty
              : null;

  return (
    <div className="flex flex-1 flex-col">
      <div className="surface-card rounded-3xl p-6 sm:p-8 lg:p-10">
        <div className="mb-3.5 text-xs font-medium tracking-wide text-muted-foreground sm:mb-4 sm:text-sm">
          {passage.udhrCitation}
        </div>
        <p
          data-testid="udhr-passage"
          lang={passage.bcp47}
          className="text-[1.125rem] leading-[1.85] tracking-[-0.01em] text-pretty text-foreground sm:text-xl lg:text-[1.35rem] lg:leading-[1.9]"
        >
          {passage.udhrText}
        </p>
      </div>

      {preview && phase === "review" ? (
        <div className="step-enter surface-card mt-4 rounded-2xl p-4 sm:p-5">
          <div className="mb-2.5 text-xs font-medium text-muted-foreground">
            {copy.reviewLabel}
          </div>
          <audio
            data-testid="review-audio"
            aria-label={copy.reviewLabel}
            controls
            src={preview}
            className="h-10 w-full"
          />
        </div>
      ) : null}

      {errorMessage ? (
        <p
          role="alert"
          className="mt-4 rounded-2xl border border-destructive/25 bg-white px-4 py-3 text-sm text-destructive"
        >
          {errorMessage}
        </p>
      ) : null}

      <div className="mt-auto pt-8 md:mt-6 md:pt-0">
        {phase === "recording" ? (
          <div className="space-y-3 md:flex md:items-center md:justify-between md:gap-4 md:space-y-0 md:rounded-2xl md:border md:bg-white md:px-5 md:py-3">
            <div
              role="status"
              className="flex items-center justify-center gap-2.5 text-sm font-medium tabular-nums md:justify-start"
            >
              <span className="relative flex size-2.5">
                <span className="record-pulse absolute inline-flex h-full w-full rounded-full bg-accent" />
                <span className="relative inline-flex size-2.5 rounded-full bg-accent" />
              </span>
              <span>{copy.recordingStatus(seconds.toFixed(1))}</span>
            </div>
            <Button
              data-testid="stop-recording"
              size="lg"
              className="h-14 w-full rounded-2xl bg-accent text-base font-medium text-accent-foreground hover:bg-accent/90 md:h-12 md:w-auto md:min-w-44 md:px-7"
              onClick={() => void stop()}
            >
              <StopIcon className="size-4" />
              {copy.stopRecording}
            </Button>
          </div>
        ) : phase === "review" ? (
          <div className="grid grid-cols-1 gap-2.5 sm:flex sm:justify-end sm:gap-3">
            <Button
              data-testid="record-again"
              variant="outline"
              size="lg"
              disabled={disabled || saving}
              className="h-13 rounded-2xl bg-white text-base font-medium sm:w-auto sm:min-w-40 sm:px-6"
              onClick={() => void start()}
            >
              <ReloadIcon className="size-4" />
              {copy.recordAgain}
            </Button>
            <Button
              data-testid="complete-setup"
              size="lg"
              disabled={disabled || saving || !recorded.current}
              className="h-13 rounded-2xl text-base font-medium sm:w-auto sm:min-w-48 sm:px-7"
              onClick={() => {
                if (recorded.current) {
                  onSave(recorded.current, passage.udhrText, language);
                }
              }}
            >
              <CheckIcon className="size-4" />
              {saving ? copy.savingVoice : copy.saveVoice}
            </Button>
          </div>
        ) : (
          <div className="md:flex md:justify-end">
            <Button
              data-testid="start-recording"
              size="lg"
              disabled={disabled || phase === "requesting"}
              className="h-14 w-full rounded-2xl text-base font-medium md:h-13 md:w-auto md:min-w-52 md:px-8"
              onClick={() => void start()}
            >
              <MicWaveIcon />
              {phase === "requesting" ? copy.waitingMic : copy.startRecording}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
