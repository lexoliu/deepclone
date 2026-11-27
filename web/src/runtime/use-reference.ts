import { openDB, type DBSchema } from "idb";
import { useEffect, useRef, useState } from "react";
import type { SavedConditioning } from "./contracts";

export interface SavedReference {
  file: File;
  transcript: string;
  source: "record" | "upload";
  language: string;
  conditioning?: SavedConditioning;
}
export type ReferenceErrorKind = "restore_failed" | "save_failed";

interface VoiceDatabase extends DBSchema {
  reference: { key: string; value: SavedReference };
}

function sameVoice(
  saved: SavedReference | undefined,
  file: File,
  transcript: string,
): saved is SavedReference {
  return Boolean(
    saved &&
    saved.transcript === transcript &&
    saved.file.name === file.name &&
    saved.file.size === file.size &&
    saved.file.lastModified === file.lastModified,
  );
}

export function useReference() {
  const [value, setValue] = useState<SavedReference | null>(null);
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const [errorKind, setErrorKind] = useState<ReferenceErrorKind | null>(null);
  const database = useRef<ReturnType<typeof openDB<VoiceDatabase>> | null>(
    null,
  );
  const writes = useRef(Promise.resolve());
  const sequence = useRef(0);
  useEffect(() => {
    let alive = true;
    const connection = openDB<VoiceDatabase>("deepclone-voices", 1, {
      upgrade(db) {
        db.createObjectStore("reference");
      },
      blocking() {
        void connection.then((db) => db.close());
      },
    });
    database.current = connection;
    void connection
      .then((db) => db.get("reference", "selected"))
      .then((saved) => {
        if (alive) setValue(saved ?? null);
      })
      .catch((reason) => {
        console.error("Saved voice could not be restored", reason);
        if (alive) setErrorKind("restore_failed");
      })
      .finally(() => {
        if (alive) setReady(true);
      });
    return () => {
      alive = false;
      void writes.current
        .then(() => connection.then((db) => db.close()))
        .catch((reason) =>
          console.error("Voice storage could not close", reason),
        );
    };
  }, []);
  function save(reference: SavedReference | null): Promise<boolean> {
    setSaving(true);
    setErrorKind(null);
    const current = ++sequence.current;
    // Commit to IndexedDB before updating UI state so reloads cannot lose a saved voice.
    const task = writes.current
      .then(async () => {
        const db = await database.current!;
        if (reference) await db.put("reference", reference, "selected");
        else await db.delete("reference", "selected");
        if (current === sequence.current) setValue(reference);
        return true;
      })
      .catch((reason) => {
        console.error("Voice storage failed", reason);
        if (current === sequence.current) setErrorKind("save_failed");
        return false;
      })
      .finally(() => {
        if (current === sequence.current) setSaving(false);
      });
    writes.current = task.then(() => undefined);
    return task;
  }
  function saveConditioning(
    file: File,
    transcript: string,
    conditioning: SavedConditioning,
  ): void {
    const task = writes.current
      .then(async () => {
        const db = await database.current;
        if (!db) return;
        const saved = await db.get("reference", "selected");
        if (!sameVoice(saved, file, transcript)) return;
        const updated: SavedReference = { ...saved, conditioning };
        await db.put("reference", updated, "selected");
        setValue((previous) =>
          previous && sameVoice(previous, file, transcript)
            ? { ...previous, conditioning }
            : previous,
        );
      })
      .catch((reason) => {
        console.error("Voice conditioning cache write failed", reason);
      });
    writes.current = task;
  }
  return { value, ready, saving, errorKind, save, saveConditioning };
}
