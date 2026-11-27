import { expect, test } from "bun:test";
import {
  formatHistoryFilename,
  sortGenerationsNewestFirst,
  type SavedGeneration,
} from "./use-history";

test("sortGenerationsNewestFirst orders entries by timestamp descending with id tie-break", () => {
  const wav = new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" });
  const entries: SavedGeneration[] = [
    {
      id: "a",
      text: "older",
      createdAt: 1000,
      seconds: 1.2,
      complete: true,
      wav,
    },
    {
      id: "b",
      text: "same-ts-1",
      createdAt: 2000,
      seconds: 2.3,
      complete: true,
      wav,
    },
    {
      id: "c",
      text: "same-ts-2",
      createdAt: 2000,
      seconds: 0.8,
      complete: false,
      wav,
    },
  ];
  const sorted = sortGenerationsNewestFirst(entries);
  expect(sorted.map((item) => item.id)).toEqual(["c", "b", "a"]);
});

test("formatHistoryFilename produces a sortable WAV download name", () => {
  const ts = new Date(2026, 10, 27, 4, 5, 9).getTime();
  expect(formatHistoryFilename(ts)).toBe("deepclone-20261127-040509.wav");
});
