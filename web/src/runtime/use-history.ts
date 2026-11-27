import { openDB, type DBSchema } from "idb";
import { useEffect, useRef, useState } from "react";

export interface SavedGeneration {
  id: string;
  text: string;
  createdAt: number;
  seconds: number;
  complete: boolean;
  wav: Blob;
}

export interface HistoryItem extends SavedGeneration {
  audioUrl: string;
}

export type HistoryErrorKind =
  "restore_failed" | "save_failed" | "delete_failed";

interface HistoryDatabase extends DBSchema {
  generations: {
    key: string;
    value: SavedGeneration;
    indexes: { "by-created": number };
  };
}

export function sortGenerationsNewestFirst(
  entries: readonly SavedGeneration[],
): SavedGeneration[] {
  return [...entries].sort((a, b) =>
    b.createdAt !== a.createdAt
      ? b.createdAt - a.createdAt
      : b.id.localeCompare(a.id),
  );
}

export function formatHistoryFilename(createdAt: number): string {
  const date = new Date(createdAt);
  const pad = (value: number) => String(value).padStart(2, "0");
  return `deepclone-${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}.wav`;
}

function createGenerationId(sequence: number): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${sequence}`;
}

export function useHistory() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [ready, setReady] = useState(false);
  const [errorKind, setErrorKind] = useState<HistoryErrorKind | null>(null);
  const database = useRef<ReturnType<typeof openDB<HistoryDatabase>> | null>(
    null,
  );
  const writes = useRef(Promise.resolve());
  const urls = useRef(new Map<string, string>());
  const sequence = useRef(0);
  const alive = useRef(true);

  function toHistoryItem(entry: SavedGeneration): HistoryItem {
    let audioUrl = urls.current.get(entry.id);
    if (!audioUrl) {
      audioUrl = URL.createObjectURL(entry.wav);
      urls.current.set(entry.id, audioUrl);
    }
    return { ...entry, audioUrl };
  }

  function revokeUrl(id: string) {
    const existing = urls.current.get(id);
    if (!existing) return;
    URL.revokeObjectURL(existing);
    urls.current.delete(id);
  }

  useEffect(() => {
    alive.current = true;
    const connection = openDB<HistoryDatabase>("deepclone-history", 1, {
      upgrade(db) {
        const store = db.createObjectStore("generations", { keyPath: "id" });
        store.createIndex("by-created", "createdAt");
      },
      blocking() {
        void connection.then((db) => db.close());
      },
    });
    database.current = connection;
    void connection
      .then((db) => db.getAll("generations"))
      .then((saved) => {
        if (!alive.current) return;
        const ordered = sortGenerationsNewestFirst(saved);
        setItems(ordered.map(toHistoryItem));
      })
      .catch((reason) => {
        console.error("Generation history could not be restored", reason);
        if (alive.current) setErrorKind("restore_failed");
      })
      .finally(() => {
        if (alive.current) setReady(true);
      });
    return () => {
      alive.current = false;
      for (const url of urls.current.values()) URL.revokeObjectURL(url);
      urls.current.clear();
      void writes.current
        .then(() => connection.then((db) => db.close()))
        .catch((reason) =>
          console.error("History storage could not close", reason),
        );
    };
  }, []);

  function add(
    entry: Omit<SavedGeneration, "id" | "createdAt"> & { createdAt?: number },
  ): Promise<HistoryItem | null> {
    setErrorKind(null);
    const saved: SavedGeneration = {
      id: createGenerationId(++sequence.current),
      text: entry.text,
      createdAt: entry.createdAt ?? Date.now(),
      seconds: entry.seconds,
      complete: entry.complete,
      wav: entry.wav,
    };
    const task = writes.current
      .then(async () => {
        const db = await database.current;
        if (!db) return null;
        await db.put("generations", saved);
        if (!alive.current) return null;
        const item = toHistoryItem(saved);
        setItems((previous) => [item, ...previous]);
        return item;
      })
      .catch((reason) => {
        console.error("Generation history write failed", reason);
        if (alive.current) setErrorKind("save_failed");
        return null;
      });
    writes.current = task.then(() => undefined);
    return task;
  }

  function remove(id: string): Promise<boolean> {
    setErrorKind(null);
    const task = writes.current
      .then(async () => {
        const db = await database.current;
        if (!db) return false;
        await db.delete("generations", id);
        if (!alive.current) return false;
        revokeUrl(id);
        setItems((previous) => previous.filter((item) => item.id !== id));
        return true;
      })
      .catch((reason) => {
        console.error("Generation history deletion failed", reason);
        if (alive.current) setErrorKind("delete_failed");
        return false;
      });
    writes.current = task.then(() => undefined);
    return task;
  }

  function clear(): Promise<boolean> {
    setErrorKind(null);
    const task = writes.current
      .then(async () => {
        const db = await database.current;
        if (!db) return false;
        await db.clear("generations");
        if (!alive.current) return false;
        for (const url of urls.current.values()) URL.revokeObjectURL(url);
        urls.current.clear();
        setItems([]);
        return true;
      })
      .catch((reason) => {
        console.error("Generation history clear failed", reason);
        if (alive.current) setErrorKind("delete_failed");
        return false;
      });
    writes.current = task.then(() => undefined);
    return task;
  }

  return { items, ready, errorKind, add, remove, clear };
}
