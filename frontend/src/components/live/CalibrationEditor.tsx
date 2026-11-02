import { useEffect, useRef, useState } from "react";
import { Ruler, Trash2 } from "lucide-react";
import { ApiError } from "../../api/client";
import { liveApi, type GroundCalibration } from "../../api/live";

type Point = [number, number];

const CORNER_HINTS = [
  "Click the 1st corner of a rectangle on the ground",
  "Click the 2nd corner (along the width)",
  "Click the 3rd corner (across, along the depth)",
  "Click the 4th corner",
  "Enter the rectangle's real size and save",
];

/**
 * Ground calibration for one camera.
 *
 * The camera sees in perspective, so pixels mean different distances near
 * and far. Marking a rectangle that lies flat on the ground and giving its
 * real size lets behaviour analytics measure speed and distance in metres,
 * the same everywhere in the picture.
 */
export function CalibrationEditor({ cameraId, imageUrl }: { cameraId: string; imageUrl: string }) {
  const [saved, setSaved] = useState<GroundCalibration | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [width, setWidth] = useState("");
  const [depth, setDepth] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    let cancelled = false;
    setPoints([]);
    liveApi.getCalibration(cameraId).then(
      (c) => !cancelled && setSaved(c),
      () => !cancelled && setSaved({ calibrated: false }),
    );
    return () => {
      cancelled = true;
    };
  }, [cameraId]);

  function handleClick(e: React.MouseEvent<HTMLImageElement>) {
    const img = imgRef.current;
    if (!img) return;
    const r = img.getBoundingClientRect();
    const p: Point = [
      Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    ];
    setError(null);
    setPoints((pts) => (pts.length >= 4 ? [p] : [...pts, p]));
  }

  const w = parseFloat(width);
  const d = parseFloat(depth);
  const ready = points.length === 4 && w > 0 && d > 0;

  async function save() {
    if (!ready) return;
    setBusy(true);
    setError(null);
    try {
      setSaved(await liveApi.setCalibration(cameraId, { points, width_m: w, depth_m: d }));
      setPoints([]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save the calibration");
    } finally {
      setBusy(false);
    }
  }

  async function clear() {
    setBusy(true);
    try {
      setSaved(await liveApi.clearCalibration(cameraId));
    } finally {
      setBusy(false);
    }
  }

  const shown: Point[] = points.length > 0 ? points : (saved?.points ?? []);
  const svg = (pts: Point[]) => pts.map(([x, y]) => `${x * 100},${y * 100}`).join(" ");

  return (
    <div className="flex flex-col gap-2">
      <div className="text-[11.5px] leading-relaxed text-text-tertiary">
        Mark 4 corners of something rectangular lying flat on the ground (a parking bay, a stretch of road,
        a patch of fence line) and type its real size. Behaviour analytics then measures speed and distance in
        metres instead of pixels, so "running" means the same near the camera and far away.
      </div>

      <div className="relative inline-block w-full select-none overflow-hidden rounded-lg border border-border-1 bg-black">
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Camera view to mark the ground rectangle on"
          onClick={handleClick}
          className="block w-full cursor-crosshair"
          draggable={false}
        />
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
          {shown.length > 1 && (
            <polygon
              points={svg(shown)}
              fill={points.length > 0 ? "rgba(59,158,255,0.12)" : "rgba(52,199,89,0.10)"}
              stroke={points.length > 0 ? "var(--color-accent-blue)" : "var(--color-accent-green)"}
              strokeWidth="0.5"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
        {shown.map(([x, y], i) => (
          <div
            key={i}
            className="pointer-events-none absolute flex h-5 w-5 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-accent-blue text-[10px] font-bold text-on-accent"
            style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
          >
            {i + 1}
          </div>
        ))}
      </div>

      <div className="text-[11.5px] text-text-secondary">{CORNER_HINTS[Math.min(points.length, 4)]}</div>

      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1.5 text-[12px] text-text-secondary">
          Width 1→2
          <input
            type="number" min="0.1" step="0.1" value={width} onChange={(e) => setWidth(e.target.value)}
            placeholder="m"
            className="w-20 rounded border border-border-2 bg-bg-3 px-2 py-1.5 text-[12.5px] text-text-primary focus:border-accent-blue focus:outline-none"
          />
        </label>
        <label className="flex items-center gap-1.5 text-[12px] text-text-secondary">
          Depth 2→3
          <input
            type="number" min="0.1" step="0.1" value={depth} onChange={(e) => setDepth(e.target.value)}
            placeholder="m"
            className="w-20 rounded border border-border-2 bg-bg-3 px-2 py-1.5 text-[12.5px] text-text-primary focus:border-accent-blue focus:outline-none"
          />
        </label>
        <button
          onClick={() => void save()}
          disabled={!ready || busy}
          className="flex items-center gap-1.5 rounded bg-accent-blue px-3 py-1.5 text-[12px] font-semibold text-on-accent disabled:opacity-40"
        >
          <Ruler size={13} /> Save calibration
        </button>
        {points.length > 0 && (
          <button onClick={() => setPoints([])} className="rounded border border-border-2 px-3 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3">
            Reset
          </button>
        )}
      </div>

      {error && <div className="text-[11.5px] text-accent-red">{error}</div>}

      {saved?.calibrated && points.length === 0 && (
        <div className="flex items-center justify-between rounded border border-border-1 bg-bg-3 px-3 py-2 text-[12px] text-text-primary">
          <span>
            Calibrated: {saved.width_m} m × {saved.depth_m} m rectangle. Apply to camera to use it.
          </span>
          <button onClick={() => void clear()} disabled={busy} className="text-text-tertiary hover:text-accent-red" title="Remove calibration">
            <Trash2 size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
