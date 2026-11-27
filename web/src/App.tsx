import { useEffect, useState } from "react";
import {
  ArrowDownIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  CheckIcon,
  ReloadIcon,
  StopIcon,
  TrashIcon,
} from "@radix-ui/react-icons";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { useReference } from "@/runtime/use-reference";
import { useVoice } from "@/runtime/use-voice";
import {
  COPY,
  LANGUAGES,
  detectBrowserLanguage,
  resolveLanguage,
  type LanguageId,
} from "./i18n";
import { ReferenceRecorder } from "./ReferenceRecorder";

function VoiceMark() {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      className="size-4"
      aria-hidden="true"
    >
      <path d="M4 8.5v3" />
      <path d="M8 5.5v9" />
      <path d="M12 3.5v13" />
      <path d="M16 7v6" />
    </svg>
  );
}

export default function App() {
  const voice = useVoice();
  const reference = useReference();
  const [setupOpen, setSetupOpen] = useState(false);
  const [step, setStep] = useState<1 | 2>(1);
  const [recordingBusy, setRecordingBusy] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState<LanguageId>(() =>
    detectBrowserLanguage(),
  );
  const [text, setText] = useState("");

  useEffect(() => {
    if (reference.value?.language) {
      setSelectedLanguage(resolveLanguage(reference.value.language));
    }
  }, [reference.value?.language]);

  const hasSavedVoice = Boolean(
    reference.value?.file && reference.value.transcript.trim(),
  );
  const inSetup = !hasSavedVoice || setupOpen;
  const activeLanguage = inSetup
    ? selectedLanguage
    : resolveLanguage(reference.value?.language);
  const copy = COPY[activeLanguage];

  useEffect(() => {
    const bcp47 =
      LANGUAGES.find((entry) => entry.id === activeLanguage)?.bcp47 ?? "en";
    document.documentElement.lang = bcp47;
  }, [activeLanguage]);

  const supported =
    typeof navigator !== "undefined" &&
    Boolean((navigator as Navigator & { gpu?: unknown }).gpu);

  const file = reference.value?.file ?? null;
  const transcript = reference.value?.transcript ?? "";
  const canGenerate =
    supported &&
    Boolean(file && transcript.trim() && text.trim()) &&
    !voice.busy;

  async function generate() {
    if (!file || !transcript.trim() || !text.trim()) return;
    await voice.start(file, transcript, text, "auto");
  }

  const statusText =
    voice.phase === "conditioning"
      ? copy.statusConditioning
      : voice.phase === "generating"
        ? copy.statusGenerating
        : voice.phase === "playing"
          ? copy.statusPlaying
          : voice.phase === "done"
            ? copy.statusDone
            : voice.phase === "cancelled"
              ? copy.statusCancelled
              : voice.phase === "incomplete"
                ? copy.statusIncomplete
                : voice.phase === "stopped"
                  ? copy.statusStopped
                  : copy.statusReady;

  const voiceError =
    voice.errorKind === "prepare_failed"
      ? copy.errorPrepareFailed
      : voice.errorKind === "stopped_empty"
        ? copy.errorStoppedEmpty
        : voice.errorKind === "stopped_partial"
          ? copy.errorStoppedPartial
          : null;

  const referenceError =
    reference.errorKind === "restore_failed"
      ? copy.errorRestoreFailed
      : reference.errorKind === "save_failed"
        ? copy.errorSaveFailed
        : null;

  const errorMessage = voiceError || referenceError;

  if (!reference.ready) {
    return (
      <div className="mx-auto flex min-h-[100dvh] w-full max-w-[32rem] flex-col px-5 pt-5 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:px-6 sm:py-8">
        <div className="flex h-11 items-center justify-between">
          <div className="h-6 w-28 animate-pulse rounded-lg bg-secondary" />
          <div className="h-8 w-20 animate-pulse rounded-full bg-secondary" />
        </div>
        <div className="mt-8 space-y-3">
          <div className="h-8 w-52 animate-pulse rounded-xl bg-secondary" />
          <div className="h-5 w-72 animate-pulse rounded-lg bg-secondary" />
        </div>
        <div className="mt-7 h-64 animate-pulse rounded-3xl bg-white/70" />
      </div>
    );
  }

  return (
    <div className="relative min-h-[100dvh]">
      <div
        aria-hidden="true"
        className="grain-overlay pointer-events-none fixed inset-0"
      />
      <div className="relative mx-auto flex min-h-[100dvh] w-full max-w-[32rem] flex-col px-5 pt-4 pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:px-6 sm:py-8">
        <header className="flex h-12 items-center justify-between gap-3">
          {inSetup && step === 2 ? (
            <button
              type="button"
              data-testid="step-back"
              disabled={recordingBusy || reference.saving}
              onClick={() => setStep(1)}
              className="inline-flex h-9 items-center gap-1.5 rounded-full px-2.5 text-sm font-medium text-muted-foreground hover:bg-secondary/80 hover:text-foreground disabled:opacity-40"
            >
              <ArrowLeftIcon className="size-4" />
              {copy.back}
            </button>
          ) : inSetup && step === 1 && hasSavedVoice ? (
            <button
              type="button"
              data-testid="cancel-setup"
              disabled={reference.saving}
              onClick={() => setSetupOpen(false)}
              className="inline-flex h-9 items-center gap-1.5 rounded-full px-2.5 text-sm font-medium text-muted-foreground hover:bg-secondary/80 hover:text-foreground disabled:opacity-40"
            >
              <ArrowLeftIcon className="size-4" />
              {copy.cancel}
            </button>
          ) : (
            <a
              href="/"
              className="inline-flex items-center gap-2 text-sm font-semibold tracking-tight"
            >
              <span className="flex size-8 items-center justify-center rounded-xl border bg-white text-foreground">
                <VoiceMark />
              </span>
              <span>deepclone</span>
            </a>
          )}

          {inSetup ? (
            <span className="text-xs font-medium tabular-nums text-muted-foreground">
              {copy.stepIndicator(step, 2)}
            </span>
          ) : (
            <button
              type="button"
              data-testid="re-setup"
              disabled={voice.busy || reference.saving}
              onClick={() => {
                setSelectedLanguage(resolveLanguage(reference.value?.language));
                setStep(1);
                setSetupOpen(true);
              }}
              className="inline-flex h-9 items-center gap-1.5 rounded-full border bg-white px-3.5 text-xs font-medium text-foreground hover:bg-secondary/70 disabled:opacity-40"
            >
              <ReloadIcon className="size-3.5" />
              {copy.resetSetup}
            </button>
          )}
        </header>

        <main className="mt-6 flex flex-1 flex-col">
          {!supported && !inSetup ? (
            <div
              role="alert"
              className="mb-5 rounded-2xl border border-destructive/25 bg-white px-4 py-3.5 text-sm leading-relaxed text-destructive"
            >
              {copy.errorUnsupportedBrowser}
            </div>
          ) : null}

          {inSetup && referenceError ? (
            <div
              role="alert"
              className="mb-5 rounded-2xl border border-destructive/25 bg-white px-4 py-3.5 text-sm leading-relaxed text-destructive"
            >
              {referenceError}
            </div>
          ) : null}

          {inSetup && step === 1 ? (
            <section
              key="setup-step-1"
              className="step-enter flex flex-1 flex-col"
            >
              <div className="mb-6">
                <h1 className="text-[1.75rem] leading-[1.15] font-semibold tracking-tight text-balance sm:text-3xl">
                  {copy.step1Title}
                </h1>
                <p className="mt-2.5 text-sm leading-relaxed text-pretty text-muted-foreground sm:text-base">
                  {copy.step1Subtitle}
                </p>
              </div>

              <div
                role="radiogroup"
                aria-label={copy.languageGroupLabel}
                className="grid grid-cols-2 gap-2.5"
              >
                {LANGUAGES.map((item) => {
                  const checked = selectedLanguage === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      role="radio"
                      aria-checked={checked}
                      data-testid={`lang-${item.id}`}
                      lang={item.bcp47}
                      onClick={() => setSelectedLanguage(item.id)}
                      className={`flex min-h-13 items-center justify-between rounded-2xl border px-4 py-3 text-left text-base font-medium ${
                        checked
                          ? "border-accent bg-accent text-accent-foreground"
                          : "border-border bg-white text-foreground hover:bg-secondary/60"
                      }`}
                    >
                      <span className="truncate">{item.nativeLabel}</span>
                      {checked ? (
                        <CheckIcon className="size-4 shrink-0" />
                      ) : null}
                    </button>
                  );
                })}
              </div>

              {hasSavedVoice ? (
                <div className="mt-5 flex justify-center">
                  <button
                    type="button"
                    data-testid="delete-voice"
                    disabled={reference.saving}
                    onClick={() => {
                      reference.save(null);
                      setSetupOpen(false);
                      setStep(1);
                    }}
                    className="inline-flex h-9 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-muted-foreground hover:bg-secondary/80 hover:text-destructive disabled:opacity-40"
                  >
                    <TrashIcon className="size-3.5" />
                    {copy.deleteSavedVoice}
                  </button>
                </div>
              ) : null}

              <div className="mt-auto pt-8">
                <Button
                  data-testid="continue-step-2"
                  size="lg"
                  className="h-14 w-full rounded-2xl text-base font-medium"
                  onClick={() => setStep(2)}
                >
                  {copy.continueStep}
                  <ArrowRightIcon className="size-4" />
                </Button>
              </div>
            </section>
          ) : inSetup && step === 2 ? (
            <section
              key="setup-step-2"
              className="step-enter flex flex-1 flex-col"
            >
              <div className="mb-6">
                <h1 className="text-[1.75rem] leading-[1.15] font-semibold tracking-tight text-balance sm:text-3xl">
                  {copy.step2Title}
                </h1>
                <p className="mt-2.5 text-sm leading-relaxed text-pretty text-muted-foreground sm:text-base">
                  {copy.step2Subtitle}
                </p>
              </div>

              <ReferenceRecorder
                language={selectedLanguage}
                disabled={voice.busy}
                saving={reference.saving}
                onRecordingChange={setRecordingBusy}
                onSave={(recordedFile, spokenTranscript, language) => {
                  reference.save({
                    file: recordedFile,
                    transcript: spokenTranscript,
                    source: "record",
                    language,
                  });
                  setSetupOpen(false);
                  setStep(1);
                }}
              />
            </section>
          ) : (
            <section
              key="main-workspace"
              className="step-enter flex flex-1 flex-col"
            >
              <div className="mb-6">
                <h1 className="text-[1.75rem] leading-[1.15] font-semibold tracking-tight text-balance sm:text-3xl">
                  {copy.mainTitle}
                </h1>
                <p className="mt-2 text-sm leading-relaxed text-pretty text-muted-foreground sm:text-base">
                  {copy.mainSubtitle}
                </p>
              </div>

              <div className="surface-card rounded-3xl p-4 sm:p-5">
                <label
                  htmlFor="speech-text"
                  className="mb-2.5 block text-xs font-medium text-muted-foreground"
                >
                  {copy.promptLabel}
                </label>
                <Textarea
                  id="speech-text"
                  data-testid="prompt-input"
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  disabled={voice.busy}
                  placeholder={copy.promptPlaceholder}
                  className="min-h-40 border-0 bg-transparent px-0 py-1 text-base leading-relaxed shadow-none focus-visible:ring-0 sm:min-h-48 sm:text-base"
                />
              </div>

              {voice.busy && !voice.loaded ? (
                <div
                  className="surface-card mt-4 rounded-2xl p-4"
                  role="status"
                >
                  <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                    <span>{copy.preparingModel}</span>
                    <span className="font-mono tabular-nums">
                      {Math.round(voice.progress)}%
                    </span>
                  </div>
                  <Progress
                    value={voice.progress}
                    aria-label={copy.preparationProgressLabel}
                  />
                </div>
              ) : null}

              {errorMessage ? (
                <div
                  role="alert"
                  className="mt-4 rounded-2xl border border-destructive/25 bg-white px-4 py-3 text-sm leading-relaxed text-destructive"
                >
                  {errorMessage}
                </div>
              ) : null}

              {voice.audio || voice.busy || voice.phase !== "ready" ? (
                <div className="surface-card mt-4 rounded-2xl p-4">
                  <div className="flex items-center justify-between gap-3">
                    <p
                      role="status"
                      data-testid="voice-status"
                      className="text-xs font-medium tabular-nums text-muted-foreground"
                    >
                      {statusText}
                      {voice.seconds > 0
                        ? ` · ${copy.generatedDuration(voice.seconds.toFixed(1))}`
                        : ""}
                    </p>
                    {voice.audio ? (
                      <Button asChild variant="outline" size="sm">
                        <a
                          data-testid="download-wav"
                          href={voice.audio}
                          download="deepclone.wav"
                          className="rounded-xl"
                        >
                          <ArrowDownIcon className="size-3.5" />
                          {copy.downloadWav}
                        </a>
                      </Button>
                    ) : null}
                  </div>
                  {voice.audio ? (
                    <audio
                      controls
                      src={voice.audio}
                      className="mt-3.5 h-10 w-full"
                    />
                  ) : null}
                </div>
              ) : null}

              <div className="mt-auto pt-8">
                {voice.busy ? (
                  <Button
                    data-testid="cancel-button"
                    variant="outline"
                    size="lg"
                    className="h-14 w-full rounded-2xl bg-white text-base font-medium"
                    onClick={voice.cancel}
                  >
                    <StopIcon className="size-4" />
                    {copy.cancel}
                  </Button>
                ) : (
                  <Button
                    data-testid="synthesize-button"
                    size="lg"
                    disabled={!canGenerate}
                    className="h-14 w-full rounded-2xl text-base font-medium"
                    onClick={() => void generate()}
                  >
                    {copy.synthesize}
                    <ArrowRightIcon className="size-4" />
                  </Button>
                )}
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
