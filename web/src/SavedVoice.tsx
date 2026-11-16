import { useEffect, useState } from "react";
import { ReloadIcon, TrashIcon, CheckIcon } from "@radix-ui/react-icons";
import { Button } from "@/components/ui/button";
import type { SavedReference } from "@/runtime/use-reference";

export function SavedVoice({
  reference,
  disabled,
  saving,
  error,
  onReplace,
  onDelete,
}: {
  reference: SavedReference;
  disabled: boolean;
  saving: boolean;
  error: string | null;
  onReplace: () => void;
  onDelete: () => void;
}) {
  const [preview, setPreview] = useState<string>();
  useEffect(() => {
    const url = URL.createObjectURL(reference.file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [reference.file]);
  return (
    <div className="rounded-md border bg-white p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        <CheckIcon />
        Voice selected
      </div>
      <p role="status" className="mt-2 text-xs text-muted-foreground">
        {saving
          ? "Saving your voice…"
          : error
            ? "Voice has not been saved"
            : "Saved on this device"}
      </p>
      <audio
        aria-label="Saved voice recording"
        controls
        src={preview}
        className="mt-4 w-full"
      />
      <p className="mt-4 text-sm leading-7">{reference.transcript}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button variant="outline" disabled={disabled} onClick={onReplace}>
          <ReloadIcon />
          Record again
        </Button>
        <Button
          variant="ghost"
          disabled={disabled || saving}
          onClick={onDelete}
        >
          <TrashIcon />
          Delete voice
        </Button>
      </div>
    </div>
  );
}
