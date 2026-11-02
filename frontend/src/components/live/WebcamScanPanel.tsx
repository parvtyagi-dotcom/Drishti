import { useEffect, useState } from "react";
import { Check, Loader2, RefreshCw } from "lucide-react";
import { ApiError } from "../../api/client";
import { liveApi, type Webcam } from "../../api/live";

/**
 * Lists the webcams plugged into the server machine, each with a snapshot.
 *
 * A device index means nothing to an operator; the picture is how they tell
 * the laptop's own camera from the USB one. Picking one hands its index back
 * as the camera source.
 */
export function WebcamScanPanel({ selected, onPick }: { selected: string; onPick: (index: number) => void }) {
  const [scanning, setScanning] = useState(false);
  const [webcams, setWebcams] = useState<Webcam[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function scan() {
    setScanning(true);
    setError(null);
    try {
      setWebcams((await liveApi.webcamScan()).webcams);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The scan failed.");
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    void scan();
  }, []);

  return (
    <div className="flex flex-col gap-3 rounded border border-border-1 bg-bg-1 p-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-[12.5px] font-semibold text-text-primary">Webcams on this computer</div>
          <div className="text-[11.5px] text-text-tertiary">Pick one by its picture.</div>
        </div>
        <button
          onClick={() => void scan()}
          disabled={scanning}
          className="inline-flex shrink-0 items-center gap-1.5 rounded border border-border-2 bg-bg-3 px-3 py-1.5 text-[12px] font-semibold text-text-primary hover:bg-bg-4 disabled:opacity-40"
        >
          {scanning ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
          {scanning ? "Scanning…" : "Rescan"}
        </button>
      </div>

      {scanning && !webcams && (
        <div className="text-[11.5px] text-text-tertiary">Looking for webcams, this takes a few seconds…</div>
      )}

      {webcams && webcams.length === 0 && (
        <div className="text-[11.5px] leading-relaxed text-text-tertiary">
          No webcam answered. Check it is plugged in and not open in another app (Zoom, Teams, the Camera app),
          then rescan.
        </div>
      )}

      {webcams && webcams.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {webcams.map((w) => {
            const inUse = w.in_use_by !== null;
            const picked = selected === String(w.index);
            return (
              <button
                key={w.index}
                onClick={() => onPick(w.index)}
                disabled={inUse}
                className={`relative overflow-hidden rounded border text-left disabled:cursor-not-allowed disabled:opacity-50 ${
                  picked ? "border-accent-blue ring-1 ring-accent-blue" : "border-border-2 hover:border-text-tertiary"
                }`}
              >
                {w.snapshot ? (
                  <img src={w.snapshot} alt={`Webcam ${w.index}`} className="aspect-[4/3] w-full bg-black object-cover" />
                ) : (
                  <div className="flex aspect-[4/3] w-full items-center justify-center bg-bg-3 px-2 text-center text-[11px] text-text-tertiary">
                    {inUse ? `In use by ${w.in_use_by}` : "No picture"}
                  </div>
                )}
                <div className="flex items-center justify-between bg-bg-3 px-2 py-1 text-[11px]">
                  <span className="font-semibold text-text-primary">
                    {w.virtual ? `Phone app ${w.index}` : `Webcam ${w.index}`}
                  </span>
                  {picked ? (
                    <Check size={12} className="text-accent-blue" />
                  ) : (
                    w.width > 0 && <span className="font-mono text-text-tertiary">{w.width}×{w.height}</span>
                  )}
                </div>
              </button>
            );
          })}
        </div>
      )}

      {error && (
        <div className="rounded border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">{error}</div>
      )}
    </div>
  );
}
