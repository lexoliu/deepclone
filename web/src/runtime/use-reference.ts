import { openDB, type DBSchema } from "idb";
import { useEffect, useRef, useState } from "react";

export interface SavedReference {
  file: File;
  transcript: string;
  source: "record" | "upload";
  language: string;
}
export type ReferenceErrorKind = "restore_failed" | "save_failed";

interface VoiceDatabase extends DBSchema {
  reference: { key: string; value: SavedReference };
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
  function save(reference: SavedReference | null) {
    setValue(reference);
    setSaving(true);
    setErrorKind(null);
    const current = ++sequence.current;
    // Serialize edits so an earlier transcript cannot overwrite a newer one.
    writes.current = writes.current
      .then(async () => {
        const db = await database.current!;
        if (reference) await db.put("reference", reference, "selected");
        else await db.delete("reference", "selected");
      })
      .catch((reason) => {
        console.error("Voice storage failed", reason);
        setErrorKind("save_failed");
      })
      .finally(() => {
        if (current === sequence.current) setSaving(false);
      });
  }
  return { value, ready, saving, errorKind, save };
}
