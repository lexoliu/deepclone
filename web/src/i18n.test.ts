import { expect, test } from "bun:test";
import {
  COPY,
  LANGUAGES,
  detectBrowserLanguage,
  resolveLanguage,
} from "./i18n";

test("all ten languages provide UDHR Article 1 passages and localized UI copy", () => {
  expect(LANGUAGES.map((entry) => entry.id)).toEqual([
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Portuguese",
    "Russian",
  ]);
  const bcp47 = new Set(LANGUAGES.map((entry) => entry.bcp47));
  expect(bcp47.size).toBe(LANGUAGES.length);

  for (const entry of LANGUAGES) {
    expect(entry.nativeLabel.length).toBeGreaterThan(0);
    expect(entry.udhrCitation.length).toBeGreaterThan(0);
    expect(entry.udhrText.length).toBeGreaterThan(30);
    expect(entry.udhrText).not.toMatch(
      /清晨打开窗户|On a quiet morning|静かな朝に窓を開けると/,
    );
    const copy = COPY[entry.id];
    expect(copy.stepIndicator(1, 2)).toContain("1");
    expect(copy.recordingStatus("3.2")).toContain("3.2");
    expect(copy.generatedDuration("4.5")).toContain("4.5");
    for (const [key, value] of Object.entries(copy)) {
      if (typeof value === "string") {
        expect(value.trim().length, `${entry.id}.${key}`).toBeGreaterThan(0);
      }
    }
  }

  expect(LANGUAGES.find((entry) => entry.id === "Chinese")?.udhrText).toContain(
    "人人生而自由，在尊严和权利上一律平等",
  );
  expect(
    LANGUAGES.find((entry) => entry.id === "Japanese")?.udhrText,
  ).toContain("すべての人間は、生れながらにして自由であり");
  expect(LANGUAGES.find((entry) => entry.id === "Korean")?.udhrText).toContain(
    "모든 인간은 태어날 때부터 자유로우며",
  );
  expect(LANGUAGES.find((entry) => entry.id === "English")?.udhrText).toContain(
    "All human beings are born free and equal",
  );
});

test("browser language detection maps regional tags and falls back to English", () => {
  expect(detectBrowserLanguage(["zh-CN", "en-US"])).toBe("Chinese");
  expect(detectBrowserLanguage(["ja-JP"])).toBe("Japanese");
  expect(detectBrowserLanguage(["ko_KR"])).toBe("Korean");
  expect(detectBrowserLanguage(["sv-SE", "pt-BR"])).toBe("Portuguese");
  expect(detectBrowserLanguage(["nl-NL", "sv-SE"])).toBe("English");
  expect(resolveLanguage("Russian")).toBe("Russian");
  expect(resolveLanguage("Unknown")).toBe("English");
});
