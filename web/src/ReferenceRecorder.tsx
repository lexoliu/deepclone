import { useEffect, useRef, useState } from "react";
import {
  SpeakerLoudIcon,
  StopIcon,
  ReloadIcon,
  CheckIcon,
} from "@radix-ui/react-icons";
import { Button } from "@/components/ui/button";
const passages = [
  {
    language: "Chinese",
    label: "中文",
    text: "清晨打开窗户，远处传来轻轻的鸟鸣。阳光落在桌边，新的一天已经开始。今天我想放慢脚步，看看沿途的风景，也把心里的故事分享给你。",
  },
  {
    language: "English",
    label: "English",
    text: "On a quiet morning, I like to open the window and listen to the world outside. Every voice carries a different story, and today I would like to share a little of mine.",
  },
  {
    language: "Japanese",
    label: "日本語",
    text: "静かな朝に窓を開けると、遠くから鳥の声が聞こえてきます。新しい一日が始まります。今日は少しゆっくり歩いて、いつもと違う景色を見つけてみたいと思います。",
  },
  {
    language: "Korean",
    label: "한국어",
    text: "조용한 아침에 창문을 열면 멀리서 새소리가 들려옵니다. 새로운 하루가 시작됩니다. 오늘은 조금 천천히 걸으며 평소에 지나쳤던 풍경을 찾아보고 싶습니다.",
  },
  {
    language: "French",
    label: "Français",
    text: "Par un matin tranquille, j'aime ouvrir la fenêtre et écouter les bruits du dehors. Chaque voix raconte une histoire différente, et aujourd'hui, je voudrais partager un peu de la mienne.",
  },
  {
    language: "German",
    label: "Deutsch",
    text: "An einem ruhigen Morgen öffne ich gern das Fenster und lausche den Geräuschen draußen. Jede Stimme erzählt eine andere Geschichte, und heute möchte ich ein wenig von meiner eigenen erzählen.",
  },
  {
    language: "Spanish",
    label: "Español",
    text: "En una mañana tranquila, me gusta abrir la ventana y escuchar los sonidos de afuera. Cada voz cuenta una historia diferente, y hoy me gustaría compartir un poco de la mía.",
  },
  {
    language: "Italian",
    label: "Italiano",
    text: "In una mattina tranquilla, mi piace aprire la finestra e ascoltare i suoni del mondo fuori. Ogni voce racconta una storia diversa, e oggi vorrei condividere un po' della mia.",
  },
  {
    language: "Portuguese",
    label: "Português",
    text: "Numa manhã tranquila, gosto de abrir a janela e ouvir os sons lá fora. Cada voz conta uma história diferente, e hoje gostaria de partilhar um pouco da minha.",
  },
  {
    language: "Russian",
    label: "Русский",
    text: "Тихим утром я люблю открывать окно и слушать звуки за окном. Каждый голос рассказывает свою историю, и сегодня мне хотелось бы поделиться небольшой частью моей.",
  },
];
export function ReferenceRecorder({
  disabled,
  onUse,
  initialLanguage = "English",
}: {
  disabled: boolean;
  initialLanguage?: string;
  onUse: (file: File | null, text: string, language: string) => void;
}) {
  const [language, setLanguage] = useState(initialLanguage);
  const [phase, setPhase] = useState<
    "idle" | "requesting" | "recording" | "review" | "used"
  >("idle");
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const recorded = useRef<File | null>(null);
  const url = useRef<string | null>(null);
  const alive = useRef(true);
  const passage = passages.find((item) => item.language === language)!;
  function clear() {
    recorded.current = null;
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
    setPreview(null);
    setSeconds(0);
    onUse(null, "", language);
  }
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      const active = recorder.current;
      if (active && active.state !== "inactive") active.stop();
      stream.current?.getTracks().forEach((track) => track.stop());
      if (url.current) URL.revokeObjectURL(url.current);
    };
  }, []);
  useEffect(() => {
    if (phase !== "recording") return;
    const start = performance.now();
    const timer = setInterval(
      () => setSeconds((performance.now() - start) / 1000),
      250,
    );
    return () => clearInterval(timer);
  }, [phase]);
  async function start() {
    clear();
    setError(null);
    setPhase("requesting");
    try {
      if (
        !navigator.mediaDevices?.getUserMedia ||
        typeof MediaRecorder === "undefined"
      )
        throw new Error(
          "Microphone recording is unavailable. Use HTTPS or localhost, or upload a recording.",
        );
      const media = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      if (!alive.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current = media;
      const mimeType = [
        "audio/webm;codecs=opus",
        "audio/mp4",
        "audio/ogg;codecs=opus",
      ].find((type) => MediaRecorder.isTypeSupported(type));
      if (!mimeType)
        throw new Error(
          "This browser cannot encode microphone audio. Please upload a recording.",
        );
      const active = new MediaRecorder(media, { mimeType });
      recorder.current = active;
      const chunks: Blob[] = [];
      let failed = false;
      active.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      active.onerror = () => {
        failed = true;
        media.getTracks().forEach((track) => track.stop());
        if (!alive.current) return;
        setError("The microphone stopped unexpectedly. Please record again.");
        setPhase("idle");
      };
      active.onstop = () => {
        media.getTracks().forEach((track) => track.stop());
        stream.current = null;
        if (!alive.current || failed) return;
        const type = active.mimeType || chunks[0]?.type;
        const blob = new Blob(chunks, { type });
        if (!blob.size) {
          setError(
            "No audio was captured. Check your microphone and try again.",
          );
          setPhase("idle");
          return;
        }
        recorded.current = new File(
          [blob],
          type.includes("mp4")
            ? "my-voice.m4a"
            : type.includes("ogg")
              ? "my-voice.ogg"
              : "my-voice.webm",
          { type },
        );
        url.current = URL.createObjectURL(blob);
        setPreview(url.current);
        setPhase("review");
      };
      active.start();
      setPhase("recording");
    } catch (reason) {
      stream.current?.getTracks().forEach((track) => track.stop());
      if (!alive.current) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      setPhase("idle");
    }
  }
  return (
    <div>
      <label
        htmlFor="recording-language"
        className="mb-2 block text-xs font-medium"
      >
        Language you will read
      </label>
      <select
        id="recording-language"
        value={language}
        disabled={disabled || phase === "recording" || phase === "requesting"}
        onChange={(event) => {
          clear();
          setLanguage(event.target.value);
          setPhase("idle");
        }}
        className="h-9 w-full rounded-md border bg-white px-3 text-sm"
      >
        {passages.map((item) => (
          <option key={item.language} value={item.language}>
            {item.label}
          </option>
        ))}
      </select>
      <div className="mt-4 rounded-md border bg-white p-4">
        <div className="mb-2 text-xs font-medium text-muted-foreground">
          Read this aloud, in your natural voice
        </div>
        <p
          className="text-sm leading-7"
          lang={
            {
              Chinese: "zh",
              English: "en",
              Japanese: "ja",
              Korean: "ko",
              French: "fr",
              German: "de",
              Spanish: "es",
              Italian: "it",
              Portuguese: "pt",
              Russian: "ru",
            }[language]
          }
        >
          {passage.text}
        </p>
      </div>
      <p className="mt-3 text-xs leading-5 text-muted-foreground">
        Find a quiet space. Read the complete passage at a comfortable pace,
        then stop recording.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        {phase === "recording" ? (
          <>
            <Button onClick={() => recorder.current?.stop()}>
              <StopIcon />
              Stop recording
            </Button>
            <span role="status" className="font-mono text-xs">
              Recording · {seconds.toFixed(1)}s
            </span>
          </>
        ) : phase === "review" || phase === "used" ? (
          <>
            <Button
              variant="outline"
              disabled={disabled}
              onClick={() => void start()}
            >
              <ReloadIcon />
              Record again
            </Button>
            <Button
              disabled={disabled || phase === "used"}
              onClick={() => {
                onUse(recorded.current, passage.text, language);
                setPhase("used");
              }}
            >
              <CheckIcon />
              {phase === "used" ? "Voice selected" : "Use this recording"}
            </Button>
          </>
        ) : (
          <Button
            variant="outline"
            disabled={disabled || phase === "requesting"}
            onClick={() => void start()}
          >
            <SpeakerLoudIcon />
            {phase === "requesting"
              ? "Waiting for microphone…"
              : "Start recording"}
          </Button>
        )}
      </div>
      {preview ? (
        <audio
          aria-label="Review your recording"
          controls
          src={preview}
          className="mt-4 w-full"
        />
      ) : null}
      {error ? (
        <p role="alert" className="mt-3 text-xs text-destructive">
          {error}
        </p>
      ) : null}
    </div>
  );
}
