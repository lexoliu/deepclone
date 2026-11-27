export type LanguageId =
  | "Chinese"
  | "English"
  | "Japanese"
  | "Korean"
  | "French"
  | "German"
  | "Spanish"
  | "Italian"
  | "Portuguese"
  | "Russian";

export interface LanguageOption {
  id: LanguageId;
  bcp47: string;
  nativeLabel: string;
  udhrCitation: string;
  udhrText: string;
}

export const LANGUAGES: readonly LanguageOption[] = [
  {
    id: "Chinese",
    bcp47: "zh",
    nativeLabel: "中文",
    udhrCitation: "世界人权宣言 · 第一条",
    udhrText:
      "人人生而自由，在尊严和权利上一律平等。他们赋有理性和良心，并应以兄弟关系的精神相对待。",
  },
  {
    id: "English",
    bcp47: "en",
    nativeLabel: "English",
    udhrCitation: "Universal Declaration of Human Rights · Article 1",
    udhrText:
      "All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.",
  },
  {
    id: "Japanese",
    bcp47: "ja",
    nativeLabel: "日本語",
    udhrCitation: "世界人権宣言 · 第1条",
    udhrText:
      "すべての人間は、生れながらにして自由であり、かつ、尊厳と権利とについて平等である。人間は、理性と良心とを授けられており、互いに同胞の精神をもって行動しなければならない。",
  },
  {
    id: "Korean",
    bcp47: "ko",
    nativeLabel: "한국어",
    udhrCitation: "세계인권선언 · 제1조",
    udhrText:
      "모든 인간은 태어날 때부터 자유로우며 그 존엄과 권리에 있어 동등하다. 인간은 천부적으로 이성과 양심을 부여받았으며 서로 형제애의 정신으로 행동하여야 한다.",
  },
  {
    id: "French",
    bcp47: "fr",
    nativeLabel: "Français",
    udhrCitation: "Déclaration universelle des droits de l’homme · Article 1",
    udhrText:
      "Tous les êtres humains naissent libres et égaux en dignité et en droits. Ils sont doués de raison et de conscience et doivent agir les uns envers les autres dans un esprit de fraternité.",
  },
  {
    id: "German",
    bcp47: "de",
    nativeLabel: "Deutsch",
    udhrCitation: "Allgemeine Erklärung der Menschenrechte · Artikel 1",
    udhrText:
      "Alle Menschen sind frei und gleich an Würde und Rechten geboren. Sie sind mit Vernunft und Gewissen begabt und sollen einander im Geist der Brüderlichkeit begegnen.",
  },
  {
    id: "Spanish",
    bcp47: "es",
    nativeLabel: "Español",
    udhrCitation: "Declaración Universal de Derechos Humanos · Artículo 1",
    udhrText:
      "Todos los seres humanos nacen libres e iguales en dignidad y derechos y, dotados como están de razón y conciencia, deben comportarse fraternalmente los unos con los otros.",
  },
  {
    id: "Italian",
    bcp47: "it",
    nativeLabel: "Italiano",
    udhrCitation: "Dichiarazione Universale dei Diritti Umani · Articolo 1",
    udhrText:
      "Tutti gli esseri umani nascono liberi ed eguali in dignità e diritti. Essi sono dotati di ragione e di coscienza e devono agire gli uni verso gli altri in spirito di fratellanza.",
  },
  {
    id: "Portuguese",
    bcp47: "pt",
    nativeLabel: "Português",
    udhrCitation: "Declaração Universal dos Direitos Humanos · Artigo 1.º",
    udhrText:
      "Todos os seres humanos nascem livres e iguais em dignidade e em direitos. Dotados de razão e de consciência, devem agir uns para com os outros em espírito de fraternidade.",
  },
  {
    id: "Russian",
    bcp47: "ru",
    nativeLabel: "Русский",
    udhrCitation: "Всеобщая декларация прав человека · Статья 1",
    udhrText:
      "Все люди рождаются свободными и равными в своем достоинстве и правах. Они наделены разумом и совестью и должны поступать в отношении друг друга в духе братства.",
  },
];

const PREFIX_MAP: Record<string, LanguageId> = {
  zh: "Chinese",
  en: "English",
  ja: "Japanese",
  ko: "Korean",
  fr: "French",
  de: "German",
  es: "Spanish",
  it: "Italian",
  pt: "Portuguese",
  ru: "Russian",
};

export function isLanguageId(value: string | undefined): value is LanguageId {
  return LANGUAGES.some((entry) => entry.id === value);
}

export function detectBrowserLanguage(tags?: readonly string[]): LanguageId {
  const candidates =
    tags ??
    (typeof navigator === "undefined"
      ? []
      : navigator.languages?.length
        ? navigator.languages
        : [navigator.language]);
  for (const tag of candidates) {
    if (!tag) continue;
    const prefix = tag.toLowerCase().split(/[-_]/)[0] ?? "";
    const match = PREFIX_MAP[prefix];
    if (match) return match;
  }
  return "English";
}

export function resolveLanguage(saved: string | undefined): LanguageId {
  return isLanguageId(saved) ? saved : detectBrowserLanguage();
}

const LATIN_LANGUAGES = new Set<LanguageId>([
  "English",
  "French",
  "German",
  "Spanish",
  "Italian",
  "Portuguese",
]);

export function detectSynthesisLanguage(
  text: string,
  referenceLanguage: LanguageId,
): string {
  const hasHangul = /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]/.test(text);
  const hasKana = /[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]/.test(text);
  const hasHan = /\p{Script=Han}/u.test(text);
  const hasCyrillic = /\p{Script=Cyrillic}/u.test(text);
  const nonLatinFamilies =
    Number(hasHangul) +
    Number(hasKana) +
    Number(hasCyrillic) +
    Number(hasHan && !hasKana && !hasHangul);
  if (nonLatinFamilies > 1) return "auto";
  if (hasHangul) return "Korean";
  if (hasKana) return "Japanese";
  if (hasHan) return "Chinese";
  if (hasCyrillic) return "Russian";
  if (
    /\p{Script=Latin}/u.test(text) &&
    LATIN_LANGUAGES.has(referenceLanguage)
  ) {
    return referenceLanguage;
  }
  return "auto";
}

export interface UiCopy {
  stepIndicator: (current: number, total: number) => string;
  back: string;
  cancel: string;
  step1Title: string;
  step1Subtitle: string;
  languageGroupLabel: string;
  continueStep: string;
  deleteSavedVoice: string;
  step2Title: string;
  step2Subtitle: string;
  startRecording: string;
  waitingMic: string;
  stopRecording: string;
  recordingStatus: (seconds: string) => string;
  reviewLabel: string;
  recordAgain: string;
  saveVoice: string;
  savingVoice: string;
  micUnavailable: string;
  micDenied: string;
  micEncodeUnsupported: string;
  micStopped: string;
  micEmpty: string;
  resetSetup: string;
  mainTitle: string;
  mainSubtitle: string;
  promptLabel: string;
  promptPlaceholder: string;
  synthesize: string;
  preparingModel: string;
  preparationProgressLabel: string;
  statusReady: string;
  statusConditioning: string;
  statusGenerating: string;
  statusPlaying: string;
  statusDone: string;
  statusCancelled: string;
  statusIncomplete: string;
  statusStopped: string;
  errorUnsupportedBrowser: string;
  errorPrepareFailed: string;
  errorStoppedEmpty: string;
  errorStoppedPartial: string;
  errorRestoreFailed: string;
  errorSaveFailed: string;
  downloadWav: string;
  generatedDuration: (seconds: string) => string;
  historyTitle: string;
  clearHistory: string;
  deleteHistoryItem: string;
  historyPartial: string;
  errorHistoryFailed: string;
}

export const COPY: Record<LanguageId, UiCopy> = {
  Chinese: {
    stepIndicator: (current, total) => `第 ${current} / ${total} 步`,
    back: "返回",
    cancel: "取消",
    step1Title: "你的母语是什么？",
    step1Subtitle: "选择你最熟悉、最自然的语言来完成声音录入。",
    languageGroupLabel: "选择母语",
    continueStep: "继续",
    deleteSavedVoice: "删除已保存的声音",
    step2Title: "请自然朗读以下文字",
    step2Subtitle: "找一个安静的地方，用平时的语速完整读完即可。",
    startRecording: "开始录音",
    waitingMic: "正在请求麦克风…",
    stopRecording: "完成朗读",
    recordingStatus: (s) => `录音中 · ${s}s`,
    reviewLabel: "试听你的录音",
    recordAgain: "重新录制",
    saveVoice: "完成设置",
    savingVoice: "正在保存…",
    micUnavailable: "当前环境不支持麦克风录音，请使用 HTTPS 连接。",
    micDenied: "麦克风权限未开启，请在浏览器设置中允许访问后重试。",
    micEncodeUnsupported: "当前浏览器不支持录音编码。",
    micStopped: "录音意外中断，请重新录制。",
    micEmpty: "未采集到声音，请检查麦克风后重试。",
    resetSetup: "重设声音",
    mainTitle: "用你的声音说话",
    mainSubtitle: "输入任意文字，在你的设备上直接生成语音。",
    promptLabel: "想让你的声音说什么？",
    promptPlaceholder: "输入你想听到的内容…",
    synthesize: "生成语音",
    preparingModel: "正在本机准备语音模型…",
    preparationProgressLabel: "模型准备进度",
    statusReady: "声音已就绪",
    statusConditioning: "正在提取声纹特征",
    statusGenerating: "正在合成语音",
    statusPlaying: "正在合成并播放",
    statusDone: "合成完成",
    statusCancelled: "已取消",
    statusIncomplete: "已保存部分生成音频",
    statusStopped: "已停止",
    errorUnsupportedBrowser:
      "当前浏览器不支持本机硬件加速语音合成，请使用最新版 Chrome 并开启硬件加速。",
    errorPrepareFailed: "无法准备语音合成，请重试。",
    errorStoppedEmpty: "语音合成在生成音频前中断，请重试。",
    errorStoppedPartial: "语音合成提前中断，已为你保留当前生成的音频。",
    errorRestoreFailed: "无法读取已保存的声音，请重新录入。",
    errorSaveFailed: "无法在此设备上保存你的声音，请重试。",
    downloadWav: "下载 WAV",
    generatedDuration: (s) => `已生成 ${s}s`,
    historyTitle: "历史记录",
    clearHistory: "清空历史",
    deleteHistoryItem: "删除这条记录",
    historyPartial: "部分生成",
    errorHistoryFailed: "无法更新历史记录，请重试。",
  },
  English: {
    stepIndicator: (current, total) => `Step ${current} of ${total}`,
    back: "Back",
    cancel: "Cancel",
    step1Title: "What is your native language?",
    step1Subtitle:
      "Choose the language you speak most naturally to set up your voice.",
    languageGroupLabel: "Native language",
    continueStep: "Continue",
    deleteSavedVoice: "Delete saved voice",
    step2Title: "Read this passage aloud",
    step2Subtitle:
      "Find a quiet spot and read at a relaxed, natural pace from start to finish.",
    startRecording: "Start recording",
    waitingMic: "Waiting for microphone…",
    stopRecording: "Finish recording",
    recordingStatus: (s) => `Recording · ${s}s`,
    reviewLabel: "Review your recording",
    recordAgain: "Record again",
    saveVoice: "Complete setup",
    savingVoice: "Saving…",
    micUnavailable: "Microphone recording requires HTTPS or localhost.",
    micDenied:
      "Microphone permission was denied. Allow access in browser settings and try again.",
    micEncodeUnsupported: "This browser cannot encode microphone audio.",
    micStopped: "The microphone stopped unexpectedly. Please record again.",
    micEmpty: "No audio was captured. Check your microphone and try again.",
    resetSetup: "Re-setup",
    mainTitle: "Speak in your voice",
    mainSubtitle: "Write anything and synthesize speech right on your device.",
    promptLabel: "What should your voice say?",
    promptPlaceholder: "Type what you want your voice to say…",
    synthesize: "Generate speech",
    preparingModel: "Preparing voice model on your device…",
    preparationProgressLabel: "Preparation progress",
    statusReady: "Your voice is ready",
    statusConditioning: "Preparing your voice",
    statusGenerating: "Generating speech",
    statusPlaying: "Generating and playing",
    statusDone: "Ready to listen",
    statusCancelled: "Cancelled",
    statusIncomplete: "Incomplete audio saved",
    statusStopped: "Operation stopped",
    errorUnsupportedBrowser:
      "This browser cannot generate speech on your device. Try an updated browser with hardware acceleration enabled.",
    errorPrepareFailed: "Speech could not be prepared. Please try again.",
    errorStoppedEmpty:
      "Speech generation stopped before audio was produced. Please try again.",
    errorStoppedPartial:
      "Speech generation stopped before finishing. You can listen to or download the audio generated so far.",
    errorRestoreFailed:
      "Your saved voice could not be opened. Please set it up again.",
    errorSaveFailed:
      "Your voice could not be saved on this device. Please try again.",
    downloadWav: "Download WAV",
    generatedDuration: (s) => `${s}s generated`,
    historyTitle: "History",
    clearHistory: "Clear all",
    deleteHistoryItem: "Delete recording",
    historyPartial: "Partial",
    errorHistoryFailed: "History could not be updated on this device.",
  },
  Japanese: {
    stepIndicator: (current, total) => `ステップ ${current} / ${total}`,
    back: "戻る",
    cancel: "キャンセル",
    step1Title: "母語を選んでください",
    step1Subtitle: "もっとも自然に話せる言語で声の登録を行います。",
    languageGroupLabel: "母語の選択",
    continueStep: "次へ",
    deleteSavedVoice: "保存した声を削除",
    step2Title: "次の文章を声に出して読んでください",
    step2Subtitle:
      "静かな場所で、普段どおりの速さで最後まで読み上げてください。",
    startRecording: "録音を開始",
    waitingMic: "マイクを準備中…",
    stopRecording: "録音を完了",
    recordingStatus: (s) => `録音中 · ${s}s`,
    reviewLabel: "録音を確認",
    recordAgain: "録り直す",
    saveVoice: "設定を完了",
    savingVoice: "保存中…",
    micUnavailable: "マイク録音には HTTPS 接続が必要です。",
    micDenied:
      "マイクの許可がありません。ブラウザ設定で許可してから再試行してください。",
    micEncodeUnsupported: "このブラウザは音声録音に対応していません。",
    micStopped: "録音が中断されました。もう一度録音してください。",
    micEmpty: "音声が記録されませんでした。マイクを確認してください。",
    resetSetup: "声を再設定",
    mainTitle: "あなたの声で話す",
    mainSubtitle: "テキストを入力すると、端末上でそのまま音声を生成します。",
    promptLabel: "何を話しますか？",
    promptPlaceholder: "読み上げたい文章を入力…",
    synthesize: "音声を生成",
    preparingModel: "端末上で音声モデルを準備しています…",
    preparationProgressLabel: "準備の進捗",
    statusReady: "準備完了",
    statusConditioning: "声の特徴を準備中",
    statusGenerating: "音声を生成中",
    statusPlaying: "生成しながら再生中",
    statusDone: "再生できます",
    statusCancelled: "キャンセルしました",
    statusIncomplete: "途中までの音声を保存しました",
    statusStopped: "停止しました",
    errorUnsupportedBrowser:
      "このブラウザは端末上での音声生成に対応していません。ハードウェアアクセラレーションが有効な最新の Chrome をお試しください。",
    errorPrepareFailed: "音声の準備に失敗しました。もう一度お試しください。",
    errorStoppedEmpty: "音声が生成される前に停止しました。再試行してください。",
    errorStoppedPartial:
      "途中で生成が停止しました。ここまでに生成された音声を再生・保存できます。",
    errorRestoreFailed:
      "保存された声を読み込めませんでした。再設定してください。",
    errorSaveFailed: "この端末に声を保存できませんでした。再試行してください。",
    downloadWav: "WAV を保存",
    generatedDuration: (s) => `${s}s 生成`,
    historyTitle: "履歴",
    clearHistory: "すべて削除",
    deleteHistoryItem: "この履歴を削除",
    historyPartial: "途中まで",
    errorHistoryFailed: "履歴を更新できませんでした。",
  },
  Korean: {
    stepIndicator: (current, total) => `${current} / ${total}단계`,
    back: "뒤로",
    cancel: "취소",
    step1Title: "모국어가 무엇인가요?",
    step1Subtitle:
      "가장 자연스럽게 말할 수 있는 언어를 선택해 목소리를 등록하세요.",
    languageGroupLabel: "모국어 선택",
    continueStep: "계속",
    deleteSavedVoice: "저장된 목소리 삭제",
    step2Title: "아래 문장을 소리 내어 읽어 주세요",
    step2Subtitle: "조용한 곳에서 평소 말투와 속도로 끝까지 읽어 주세요.",
    startRecording: "녹음 시작",
    waitingMic: "마이크 요청 중…",
    stopRecording: "녹음 완료",
    recordingStatus: (s) => `녹음 중 · ${s}s`,
    reviewLabel: "녹음 들어보기",
    recordAgain: "다시 녹음",
    saveVoice: "설정 완료",
    savingVoice: "저장 중…",
    micUnavailable: "마이크 녹음은 HTTPS 환경에서만 사용할 수 있습니다.",
    micDenied:
      "마이크 권한이 거부되었습니다. 브라우저 설정에서 허용한 뒤 다시 시도해 주세요.",
    micEncodeUnsupported: "이 브라우저는 오디오 녹음을 지원하지 않습니다.",
    micStopped: "녹음이 예기치 않게 중단되었습니다. 다시 녹음해 주세요.",
    micEmpty: "녹음된 소리가 없습니다. 마이크를 확인해 주세요.",
    resetSetup: "목소리 재설정",
    mainTitle: "내 목소리로 말하기",
    mainSubtitle: "원하는 문장을 입력하면 기기에서 바로 음성을 합성합니다.",
    promptLabel: "어떤 말을 할까요?",
    promptPlaceholder: "내 목소리로 들을 문장을 입력하세요…",
    synthesize: "음성 생성",
    preparingModel: "기기에서 음성 모델을 준비하는 중…",
    preparationProgressLabel: "준비 진행률",
    statusReady: "목소리가 준비되었습니다",
    statusConditioning: "목소리 특징 준비 중",
    statusGenerating: "음성 생성 중",
    statusPlaying: "생성 및 재생 중",
    statusDone: "생성 완료",
    statusCancelled: "취소됨",
    statusIncomplete: "일부 생성된 오디오 저장됨",
    statusStopped: "중지됨",
    errorUnsupportedBrowser:
      "이 브라우저는 기기 내 음성 생성을 지원하지 않습니다. 하드웨어 가속이 켜진 최신 Chrome을 사용해 주세요.",
    errorPrepareFailed: "음성 준비에 실패했습니다. 다시 시도해 주세요.",
    errorStoppedEmpty:
      "오디오가 생성되기 전에 중단되었습니다. 다시 시도해 주세요.",
    errorStoppedPartial:
      "완료 전에 생성이 중단되었습니다. 지금까지 생성된 오디오를 듣거나 저장할 수 있습니다.",
    errorRestoreFailed:
      "저장된 목소리를 불러올 수 없습니다. 다시 설정해 주세요.",
    errorSaveFailed:
      "이 기기에 목소리를 저장할 수 없습니다. 다시 시도해 주세요.",
    downloadWav: "WAV 다운로드",
    generatedDuration: (s) => `${s}s 생성됨`,
    historyTitle: "기록",
    clearHistory: "전체 삭제",
    deleteHistoryItem: "기록 삭제",
    historyPartial: "일부 생성",
    errorHistoryFailed: "기록을 업데이트할 수 없습니다.",
  },
  French: {
    stepIndicator: (current, total) => `Étape ${current} sur ${total}`,
    back: "Retour",
    cancel: "Annuler",
    step1Title: "Quelle est votre langue maternelle ?",
    step1Subtitle:
      "Choisissez la langue que vous parlez le plus naturellement pour configurer votre voix.",
    languageGroupLabel: "Langue maternelle",
    continueStep: "Continuer",
    deleteSavedVoice: "Supprimer la voix enregistrée",
    step2Title: "Lisez ce texte à voix haute",
    step2Subtitle:
      "Installez-vous au calme et lisez l’intégralité du passage à votre rythme naturel.",
    startRecording: "Commencer l’enregistrement",
    waitingMic: "Accès au microphone…",
    stopRecording: "Terminer l’enregistrement",
    recordingStatus: (s) => `Enregistrement · ${s}s`,
    reviewLabel: "Écouter votre enregistrement",
    recordAgain: "Recommencer",
    saveVoice: "Terminer la configuration",
    savingVoice: "Enregistrement…",
    micUnavailable: "L’enregistrement nécessite une connexion HTTPS.",
    micDenied:
      "Accès au microphone refusé. Autorisez-le dans les réglages du navigateur.",
    micEncodeUnsupported: "Ce navigateur ne peut pas encoder l’audio.",
    micStopped: "Le microphone s’est arrêté. Veuillez recommencer.",
    micEmpty: "Aucun son capturé. Vérifiez votre microphone.",
    resetSetup: "Reconfigurer",
    mainTitle: "Parlez avec votre voix",
    mainSubtitle:
      "Écrivez votre texte et générez la parole directement sur votre appareil.",
    promptLabel: "Que doit dire votre voix ?",
    promptPlaceholder: "Écrivez le texte à prononcer…",
    synthesize: "Générer la voix",
    preparingModel: "Préparation du modèle vocal sur votre appareil…",
    preparationProgressLabel: "Progression de la préparation",
    statusReady: "Votre voix est prête",
    statusConditioning: "Préparation de votre voix",
    statusGenerating: "Génération en cours",
    statusPlaying: "Génération et lecture",
    statusDone: "Prêt à écouter",
    statusCancelled: "Annulé",
    statusIncomplete: "Audio partiel enregistré",
    statusStopped: "Opération arrêtée",
    errorUnsupportedBrowser:
      "Ce navigateur ne prend pas en charge la synthèse vocale locale. Utilisez un navigateur récent avec accélération matérielle.",
    errorPrepareFailed: "Impossible de préparer la synthèse. Réessayez.",
    errorStoppedEmpty: "La génération s’est arrêtée avant de produire du son.",
    errorStoppedPartial:
      "La génération s’est interrompue. Vous pouvez écouter ou télécharger l’audio déjà produit.",
    errorRestoreFailed:
      "Impossible d’ouvrir la voix enregistrée. Veuillez la reconfigurer.",
    errorSaveFailed:
      "Impossible d’enregistrer votre voix sur cet appareil. Réessayez.",
    downloadWav: "Télécharger WAV",
    generatedDuration: (s) => `${s}s générées`,
    historyTitle: "Historique",
    clearHistory: "Tout effacer",
    deleteHistoryItem: "Supprimer l’enregistrement",
    historyPartial: "Partiel",
    errorHistoryFailed: "Impossible de mettre à jour l’historique.",
  },
  German: {
    stepIndicator: (current, total) => `Schritt ${current} von ${total}`,
    back: "Zurück",
    cancel: "Abbrechen",
    step1Title: "Was ist Ihre Muttersprache?",
    step1Subtitle:
      "Wählen Sie die Sprache, die Sie am natürlichsten sprechen, um Ihre Stimme einzurichten.",
    languageGroupLabel: "Muttersprache",
    continueStep: "Weiter",
    deleteSavedVoice: "Gespeicherte Stimme löschen",
    step2Title: "Lesen Sie diesen Text laut vor",
    step2Subtitle:
      "Suchen Sie sich einen ruhigen Ort und lesen Sie den gesamten Absatz entspannt vor.",
    startRecording: "Aufnahme starten",
    waitingMic: "Warte auf Mikrofon…",
    stopRecording: "Aufnahme beenden",
    recordingStatus: (s) => `Aufnahme · ${s}s`,
    reviewLabel: "Aufnahme anhören",
    recordAgain: "Neu aufnehmen",
    saveVoice: "Einrichtung abschließen",
    savingVoice: "Speichern…",
    micUnavailable: "Für Mikrofonaufnahmen ist HTTPS erforderlich.",
    micDenied:
      "Mikrofonzugriff verweigert. Bitte in den Browsereinstellungen erlauben.",
    micEncodeUnsupported: "Dieser Browser unterstützt keine Audioaufnahme.",
    micStopped:
      "Das Mikrofon wurde unerwartet gestoppt. Bitte erneut aufnehmen.",
    micEmpty: "Kein Ton aufgenommen. Prüfen Sie Ihr Mikrofon.",
    resetSetup: "Neu einrichten",
    mainTitle: "Sprechen Sie mit Ihrer Stimme",
    mainSubtitle:
      "Geben Sie beliebigen Text ein und erzeugen Sie Sprache direkt auf Ihrem Gerät.",
    promptLabel: "Was soll Ihre Stimme sagen?",
    promptPlaceholder: "Schreiben Sie den gewünschten Text…",
    synthesize: "Sprache erzeugen",
    preparingModel: "Sprachmodell wird auf Ihrem Gerät vorbereitet…",
    preparationProgressLabel: "Vorbereitungsfortschritt",
    statusReady: "Ihre Stimme ist bereit",
    statusConditioning: "Stimme wird vorbereitet",
    statusGenerating: "Sprache wird erzeugt",
    statusPlaying: "Erzeugen und abspielen",
    statusDone: "Bereit zum Anhören",
    statusCancelled: "Abgebrochen",
    statusIncomplete: "Unvollständiges Audio gespeichert",
    statusStopped: "Vorgang gestoppt",
    errorUnsupportedBrowser:
      "Dieser Browser unterstützt keine lokale Sprachsynthese. Nutzen Sie einen aktuellen Browser mit Hardwarebeschleunigung.",
    errorPrepareFailed: "Sprachsynthese konnte nicht vorbereitet werden.",
    errorStoppedEmpty: "Die Erzeugung wurde vor der Audioausgabe beendet.",
    errorStoppedPartial:
      "Die Erzeugung wurde vorzeitig beendet. Das bisher erzeugte Audio bleibt verfügbar.",
    errorRestoreFailed:
      "Die gespeicherte Stimme konnte nicht geöffnet werden. Bitte neu einrichten.",
    errorSaveFailed:
      "Ihre Stimme konnte auf diesem Gerät nicht gespeichert werden. Bitte erneut versuchen.",
    downloadWav: "WAV herunterladen",
    generatedDuration: (s) => `${s}s erzeugt`,
    historyTitle: "Verlauf",
    clearHistory: "Alle löschen",
    deleteHistoryItem: "Eintrag löschen",
    historyPartial: "Unvollständig",
    errorHistoryFailed: "Der Verlauf konnte nicht aktualisiert werden.",
  },
  Spanish: {
    stepIndicator: (current, total) => `Paso ${current} de ${total}`,
    back: "Atrás",
    cancel: "Cancelar",
    step1Title: "¿Cuál es tu lengua materna?",
    step1Subtitle:
      "Elige el idioma que hablas con mayor naturalidad para configurar tu voz.",
    languageGroupLabel: "Lengua materna",
    continueStep: "Continuar",
    deleteSavedVoice: "Eliminar voz guardada",
    step2Title: "Lee este texto en voz alta",
    step2Subtitle:
      "Busca un lugar tranquilo y lee el párrafo completo a tu ritmo habitual.",
    startRecording: "Iniciar grabación",
    waitingMic: "Esperando el micrófono…",
    stopRecording: "Finalizar grabación",
    recordingStatus: (s) => `Grabando · ${s}s`,
    reviewLabel: "Escuchar grabación",
    recordAgain: "Grabar de nuevo",
    saveVoice: "Completar configuración",
    savingVoice: "Guardando…",
    micUnavailable: "La grabación requiere HTTPS o localhost.",
    micDenied:
      "Permiso de micrófono denegado. Habilítalo en los ajustes del navegador.",
    micEncodeUnsupported: "Este navegador no puede codificar audio.",
    micStopped: "El micrófono se detuvo inesperadamente. Graba de nuevo.",
    micEmpty: "No se capturó audio. Revisa tu micrófono.",
    resetSetup: "Reconfigurar",
    mainTitle: "Habla con tu propia voz",
    mainSubtitle:
      "Escribe cualquier texto y genera voz directamente en tu dispositivo.",
    promptLabel: "¿Qué quieres que diga tu voz?",
    promptPlaceholder: "Escribe lo que quieres escuchar…",
    synthesize: "Generar voz",
    preparingModel: "Preparando el modelo de voz en tu dispositivo…",
    preparationProgressLabel: "Progreso de preparación",
    statusReady: "Tu voz está lista",
    statusConditioning: "Preparando tu voz",
    statusGenerating: "Generando voz",
    statusPlaying: "Generando y reproduciendo",
    statusDone: "Listo para escuchar",
    statusCancelled: "Cancelado",
    statusIncomplete: "Audio parcial guardado",
    statusStopped: "Operación detenida",
    errorUnsupportedBrowser:
      "Este navegador no admite síntesis de voz en el dispositivo. Usa un navegador actualizado con aceleración por hardware.",
    errorPrepareFailed: "No se pudo preparar la voz. Inténtalo de nuevo.",
    errorStoppedEmpty: "La generación se detuvo antes de producir audio.",
    errorStoppedPartial:
      "La generación se detuvo antes de terminar. Puedes escuchar o descargar el audio generado hasta ahora.",
    errorRestoreFailed:
      "No se pudo abrir la voz guardada. Configúrala de nuevo.",
    errorSaveFailed:
      "No se pudo guardar tu voz en este dispositivo. Inténtalo de nuevo.",
    downloadWav: "Descargar WAV",
    generatedDuration: (s) => `${s}s generados`,
    historyTitle: "Historial",
    clearHistory: "Borrar todo",
    deleteHistoryItem: "Eliminar grabación",
    historyPartial: "Parcial",
    errorHistoryFailed: "No se pudo actualizar el historial.",
  },
  Italian: {
    stepIndicator: (current, total) => `Passaggio ${current} di ${total}`,
    back: "Indietro",
    cancel: "Annulla",
    step1Title: "Qual è la tua lingua madre?",
    step1Subtitle:
      "Scegli la lingua che parli con più naturalezza per configurare la tua voce.",
    languageGroupLabel: "Lingua madre",
    continueStep: "Continua",
    deleteSavedVoice: "Elimina voce salvata",
    step2Title: "Leggi questo testo ad alta voce",
    step2Subtitle:
      "Trova un ambiente silenzioso e leggi l’intero brano con il tuo ritmo naturale.",
    startRecording: "Avvia registrazione",
    waitingMic: "In attesa del microfono…",
    stopRecording: "Termina registrazione",
    recordingStatus: (s) => `Registrazione · ${s}s`,
    reviewLabel: "Riascolta la registrazione",
    recordAgain: "Registra di nuovo",
    saveVoice: "Completa configurazione",
    savingVoice: "Salvataggio…",
    micUnavailable: "La registrazione richiede una connessione HTTPS.",
    micDenied:
      "Permesso microfono negato. Consentilo nelle impostazioni del browser.",
    micEncodeUnsupported: "Questo browser non supporta la codifica audio.",
    micStopped: "Il microfono si è interrotto. Registra di nuovo.",
    micEmpty: "Nessun audio acquisito. Controlla il microfono.",
    resetSetup: "Riconfigura",
    mainTitle: "Parla con la tua voce",
    mainSubtitle:
      "Scrivi qualsiasi testo e genera la voce direttamente sul tuo dispositivo.",
    promptLabel: "Cosa deve dire la tua voce?",
    promptPlaceholder: "Scrivi le parole che vuoi ascoltare…",
    synthesize: "Genera voce",
    preparingModel: "Preparazione del modello vocale sul dispositivo…",
    preparationProgressLabel: "Avanzamento preparazione",
    statusReady: "La tua voce è pronta",
    statusConditioning: "Preparazione della voce",
    statusGenerating: "Generazione in corso",
    statusPlaying: "Generazione e riproduzione",
    statusDone: "Pronto per l’ascolto",
    statusCancelled: "Annullato",
    statusIncomplete: "Audio parziale salvato",
    statusStopped: "Operazione interrotta",
    errorUnsupportedBrowser:
      "Questo browser non supporta la sintesi vocale locale. Prova un browser aggiornato con accelerazione hardware.",
    errorPrepareFailed: "Impossibile preparare la sintesi. Riprova.",
    errorStoppedEmpty:
      "La generazione si è interrotta prima di produrre audio.",
    errorStoppedPartial:
      "La generazione si è interrotta prima della fine. Puoi ascoltare o scaricare l’audio generato finora.",
    errorRestoreFailed: "Impossibile aprire la voce salvata. Riconfigurala.",
    errorSaveFailed:
      "Impossibile salvare la tua voce su questo dispositivo. Riprova.",
    downloadWav: "Scarica WAV",
    generatedDuration: (s) => `${s}s generati`,
    historyTitle: "Cronologia",
    clearHistory: "Cancella tutto",
    deleteHistoryItem: "Elimina registrazione",
    historyPartial: "Parziale",
    errorHistoryFailed: "Impossibile aggiornare la cronologia.",
  },
  Portuguese: {
    stepIndicator: (current, total) => `Passo ${current} de ${total}`,
    back: "Voltar",
    cancel: "Cancelar",
    step1Title: "Qual é o seu idioma nativo?",
    step1Subtitle:
      "Escolha o idioma que você fala com mais naturalidade para configurar sua voz.",
    languageGroupLabel: "Idioma nativo",
    continueStep: "Continuar",
    deleteSavedVoice: "Excluir voz salva",
    step2Title: "Leia este texto em voz alta",
    step2Subtitle:
      "Encontre um local silencioso e leia todo o trecho em um ritmo natural.",
    startRecording: "Iniciar gravação",
    waitingMic: "Aguardando microfone…",
    stopRecording: "Concluir gravação",
    recordingStatus: (s) => `Gravando · ${s}s`,
    reviewLabel: "Ouvir sua gravação",
    recordAgain: "Gravar novamente",
    saveVoice: "Concluir configuração",
    savingVoice: "Salvando…",
    micUnavailable: "A gravação requer conexão HTTPS ou localhost.",
    micDenied:
      "Permissão do microfone negada. Permita o acesso nas configurações do navegador.",
    micEncodeUnsupported: "Este navegador não suporta codificação de áudio.",
    micStopped: "O microfone parou inesperadamente. Grave novamente.",
    micEmpty: "Nenhum áudio capturado. Verifique seu microfone.",
    resetSetup: "Reconfigurar",
    mainTitle: "Fale com a sua voz",
    mainSubtitle:
      "Escreva qualquer texto e gere fala diretamente no seu dispositivo.",
    promptLabel: "O que a sua voz deve dizer?",
    promptPlaceholder: "Digite o que você quer ouvir…",
    synthesize: "Gerar fala",
    preparingModel: "Preparando modelo de voz no seu dispositivo…",
    preparationProgressLabel: "Progresso da preparação",
    statusReady: "Sua voz está pronta",
    statusConditioning: "Preparando sua voz",
    statusGenerating: "Gerando fala",
    statusPlaying: "Gerando e reproduzindo",
    statusDone: "Pronto para ouvir",
    statusCancelled: "Cancelado",
    statusIncomplete: "Áudio parcial salvo",
    statusStopped: "Operação interrompida",
    errorUnsupportedBrowser:
      "Este navegador não suporta síntese de voz no dispositivo. Use um navegador atualizado com aceleração de hardware.",
    errorPrepareFailed: "Não foi possível preparar a fala. Tente novamente.",
    errorStoppedEmpty: "A geração parou antes de produzir áudio.",
    errorStoppedPartial:
      "A geração parou antes de concluir. Você pode ouvir ou baixar o áudio gerado até agora.",
    errorRestoreFailed:
      "Não foi possível abrir a voz salva. Configure-a novamente.",
    errorSaveFailed:
      "Não foi possível salvar sua voz neste dispositivo. Tente novamente.",
    downloadWav: "Baixar WAV",
    generatedDuration: (s) => `${s}s gerados`,
    historyTitle: "Histórico",
    clearHistory: "Limpar tudo",
    deleteHistoryItem: "Excluir gravação",
    historyPartial: "Parcial",
    errorHistoryFailed: "Não foi possível atualizar o histórico.",
  },
  Russian: {
    stepIndicator: (current, total) => `Шаг ${current} из ${total}`,
    back: "Назад",
    cancel: "Отмена",
    step1Title: "Какой ваш родной язык?",
    step1Subtitle:
      "Выберите язык, на котором вам удобнее всего говорить, для настройки голоса.",
    languageGroupLabel: "Родной язык",
    continueStep: "Продолжить",
    deleteSavedVoice: "Удалить сохранённый голос",
    step2Title: "Прочитайте этот текст вслух",
    step2Subtitle:
      "Найдите тихое место и спокойно прочитайте отрывок до конца в обычном темпе.",
    startRecording: "Начать запись",
    waitingMic: "Ожидание микрофона…",
    stopRecording: "Завершить запись",
    recordingStatus: (s) => `Запись · ${s}s`,
    reviewLabel: "Прослушать запись",
    recordAgain: "Записать заново",
    saveVoice: "Завершить настройку",
    savingVoice: "Сохранение…",
    micUnavailable: "Для записи с микрофона требуется HTTPS.",
    micDenied:
      "Доступ к микрофону запрещён. Разрешите его в настройках браузера.",
    micEncodeUnsupported: "Этот браузер не поддерживает кодирование звука.",
    micStopped: "Микрофон неожиданно остановился. Запишите заново.",
    micEmpty: "Звук не записан. Проверьте микрофон и повторите попытку.",
    resetSetup: "Настроить заново",
    mainTitle: "Говорите своим голосом",
    mainSubtitle:
      "Введите любой текст и синтезируйте речь прямо на вашем устройстве.",
    promptLabel: "Что должен сказать ваш голос?",
    promptPlaceholder: "Введите текст для озвучивания…",
    synthesize: "Создать речь",
    preparingModel: "Подготовка голосовой модели на устройстве…",
    preparationProgressLabel: "Прогресс подготовки",
    statusReady: "Ваш голос готов",
    statusConditioning: "Подготовка образца голоса",
    statusGenerating: "Генерация речи",
    statusPlaying: "Генерация и воспроизведение",
    statusDone: "Готово к прослушиванию",
    statusCancelled: "Отменено",
    statusIncomplete: "Частичное аудио сохранено",
    statusStopped: "Операция остановлена",
    errorUnsupportedBrowser:
      "Этот браузер не поддерживает синтез речи на устройстве. Используйте актуальный браузер с аппаратным ускорением.",
    errorPrepareFailed:
      "Не удалось подготовить синтез речи. Повторите попытку.",
    errorStoppedEmpty: "Генерация остановилась до создания аудио.",
    errorStoppedPartial:
      "Генерация прервалась до завершения. Вы можете прослушать или скачать уже созданное аудио.",
    errorRestoreFailed:
      "Не удалось открыть сохранённый голос. Настройте его заново.",
    errorSaveFailed:
      "Не удалось сохранить ваш голос на этом устройстве. Повторите попытку.",
    downloadWav: "Скачать WAV",
    generatedDuration: (s) => `Создано ${s}s`,
    historyTitle: "История",
    clearHistory: "Очистить всё",
    deleteHistoryItem: "Удалить запись",
    historyPartial: "Частично",
    errorHistoryFailed: "Не удалось обновить историю.",
  },
};
