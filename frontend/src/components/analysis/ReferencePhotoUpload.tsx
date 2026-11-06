import { AlertTriangle, UserRound, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../../api/client";
import { uploadReferencePhotos } from "../../api/references";

const fileKey = (f: File) => `${f.name}-${f.size}-${f.lastModified}`;

export function ReferencePhotoUpload({
  onChange,
}: {
  onChange: (result: { referenceSetId: string | null; photoCount: number }) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // One object URL per file, created once and revoked when the files change
  // or the component unmounts (creating them in render leaked every render).
  const previews = useMemo(
    () => files.map((file) => ({ key: fileKey(file), url: URL.createObjectURL(file) })),
    [files],
  );
  useEffect(() => {
    return () => previews.forEach((p) => URL.revokeObjectURL(p.url));
  }, [previews]);

  async function commit(nextFiles: File[]) {
    setFiles(nextFiles);
    setError(null);
    if (nextFiles.length === 0) {
      onChange({ referenceSetId: null, photoCount: 0 });
      return;
    }
    setUploading(true);
    try {
      const result = await uploadReferencePhotos(nextFiles);
      onChange({ referenceSetId: result.reference_set_id, photoCount: result.photo_count });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to upload reference photos");
      onChange({ referenceSetId: null, photoCount: 0 });
    } finally {
      setUploading(false);
    }
  }

  function addFiles(newFiles: FileList | null) {
    if (!newFiles) return;
    // Identical files are skipped so each preview keeps a unique stable key.
    const seen = new Set(files.map(fileKey));
    const fresh = Array.from(newFiles).filter((f) => {
      const k = fileKey(f);
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    const merged = [...files, ...fresh].slice(0, 6);
    void commit(merged);
  }

  function removeAt(i: number) {
    void commit(files.filter((_, idx) => idx !== i));
  }

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-text-secondary">Reference Photos</span>
        <span className="text-[11px] text-text-tertiary">{files.length} / 6 uploaded</span>
      </div>

      <div className="grid grid-cols-6 gap-2">
        {previews.map((preview, i) => (
          <div key={preview.key} className="group relative aspect-square overflow-hidden rounded-md border border-border-1 bg-bg-3">
            <img src={preview.url} alt={`Reference ${i + 1}`} className="h-full w-full object-cover" />
            <button
              onClick={() => removeAt(i)}
              aria-label={`Remove reference photo ${i + 1}`}
              className="absolute right-1 top-1 hidden h-5 w-5 items-center justify-center rounded-full bg-black/70 text-white group-hover:flex"
            >
              <X size={11} />
            </button>
          </div>
        ))}
        {files.length < 6 && (
          <button
            onClick={() => inputRef.current?.click()}
            className="flex aspect-square flex-col items-center justify-center gap-1 rounded-md border border-dashed border-border-2 text-text-tertiary transition-colors hover:border-accent-blue hover:text-accent-blue"
          >
            <UserRound size={18} />
            <span className="text-[9px] font-semibold">ADD</span>
          </button>
        )}
      </div>
      <input ref={inputRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => addFiles(e.target.files)} />

      <p className="mt-2 text-[11px] text-text-tertiary">Upload 1–6 clear reference images of the target person's face.</p>
      {uploading && <p className="mt-1 text-[11px] text-accent-blue">Uploading…</p>}
      {error && (
        <div className="mt-2 flex items-center gap-2 rounded-md border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[11.5px] text-accent-red">
          <AlertTriangle size={13} className="shrink-0" /> {error}
        </div>
      )}
    </div>
  );
}
