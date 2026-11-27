import { ArrowDownIcon, TrashIcon } from "@radix-ui/react-icons";
import { Button } from "@/components/ui/button";
import { formatHistoryFilename, type HistoryItem } from "@/runtime/use-history";
import type { UiCopy } from "./i18n";

function formatTimestamp(createdAt: number, bcp47: string): string {
  try {
    return new Intl.DateTimeFormat(bcp47, {
      dateStyle: "short",
      timeStyle: "short",
    }).format(createdAt);
  } catch {
    return new Date(createdAt).toLocaleString();
  }
}

export function HistoryList({
  items,
  bcp47,
  copy,
  onDelete,
  onClear,
}: {
  items: readonly HistoryItem[];
  bcp47: string;
  copy: UiCopy;
  onDelete: (id: string) => void;
  onClear: () => void;
}) {
  if (items.length === 0) return null;

  return (
    <section
      data-testid="history-section"
      aria-label={copy.historyTitle}
      className="mt-8 lg:col-span-12 lg:mt-4"
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold tracking-tight text-foreground sm:text-base">
          {copy.historyTitle}
          <span className="ml-2 text-xs font-medium tabular-nums text-muted-foreground">
            {items.length}
          </span>
        </h2>
        <button
          type="button"
          data-testid="history-clear"
          onClick={onClear}
          className="inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-muted-foreground hover:bg-secondary/80 hover:text-destructive"
        >
          <TrashIcon className="size-3.5" />
          {copy.clearHistory}
        </button>
      </div>

      <ul className="space-y-3">
        {items.map((item) => (
          <li
            key={item.id}
            data-testid="history-item"
            className="surface-card rounded-2xl p-4 sm:p-5"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p
                  data-testid="history-item-text"
                  className="line-clamp-2 break-words text-sm font-medium leading-snug text-foreground"
                >
                  {item.text}
                </p>
                <p className="mt-1 text-xs tabular-nums text-muted-foreground">
                  {formatTimestamp(item.createdAt, bcp47)}
                  {" · "}
                  {copy.generatedDuration(item.seconds.toFixed(1))}
                  {item.complete ? "" : ` · ${copy.historyPartial}`}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-1.5">
                <Button asChild variant="outline" size="sm">
                  <a
                    data-testid="history-download"
                    href={item.audioUrl}
                    download={formatHistoryFilename(item.createdAt)}
                    className="rounded-xl"
                  >
                    <ArrowDownIcon className="size-3.5" />
                    <span className="hidden sm:inline">{copy.downloadWav}</span>
                  </a>
                </Button>
                <button
                  type="button"
                  data-testid="history-delete"
                  aria-label={copy.deleteHistoryItem}
                  onClick={() => onDelete(item.id)}
                  className="inline-flex size-8 items-center justify-center rounded-xl border border-transparent text-muted-foreground hover:border-border hover:bg-secondary/70 hover:text-destructive"
                >
                  <TrashIcon className="size-3.5" />
                </button>
              </div>
            </div>
            <audio controls src={item.audioUrl} className="mt-3 h-10 w-full" />
          </li>
        ))}
      </ul>
    </section>
  );
}
