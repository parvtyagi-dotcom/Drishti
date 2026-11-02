import { Map as MapIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { LiveCamera, LiveEvent } from "../../api/live";
import { mapApi, type SiteMap } from "../../api/map";
import { SitePicker } from "./SitePicker";

// A camera glows red for this long after an alarm-level event (as on Site Map).
const ALARM_GLOW_SEC = 60;

type Tone = "alarm" | "warn" | "off" | "ok";

const TONE: Record<Tone, { dot: string; ring: string }> = {
  alarm: { dot: "bg-accent-red", ring: "border-accent-red/50" },
  warn: { dot: "bg-accent-amber", ring: "border-accent-amber/50" },
  off: { dot: "bg-text-disabled", ring: "border-text-disabled/50" },
  ok: { dot: "bg-accent-blue", ring: "border-accent-blue/50" },
};

function tone(camera: LiveCamera | undefined, alarm: boolean): Tone {
  if (alarm) return "alarm";
  if (!camera || camera.state === "offline" || camera.state === "stopped") return "off";
  if (camera.state !== "online" || camera.degraded) return "warn";
  return "ok";
}

/** Cameras not yet placed on the plan are spread on an ellipse so the map
 * is never empty; the panel says so rather than passing it off as a layout. */
function autoPosition(i: number, n: number): [number, number] {
  const a = (i / Math.max(1, n)) * Math.PI * 2 - Math.PI / 2;
  return [0.5 + Math.cos(a) * 0.32, 0.52 + Math.sin(a) * 0.3];
}

export function OpsMap({
  map, cameras, events, now, onSiteChange,
}: {
  map: SiteMap | null;
  cameras: LiveCamera[];
  events: LiveEvent[];
  now: number;
  onSiteChange: (id: string) => void;
}) {
  const positions = map?.positions ?? {};
  const unplaced = cameras.filter((c) => !(c.name in positions));
  const alarmed = new Set(
    events.filter((e) => (e.alarm || e.severity === "CRITICAL") && now - e.wall_time <= ALARM_GLOW_SEC).map((e) => e.camera_name),
  );
  const byName = new Map(cameras.map((c) => [c.name, c]));
  const markers: { name: string; at: [number, number] }[] = [
    ...Object.entries(positions).map(([name, at]) => ({ name, at })),
    ...unplaced.map((c, i) => ({ name: c.name, at: autoPosition(i, unplaced.length) })),
  ];
  const reporting = cameras.filter((c) => c.state === "online").length;

  // Fit the whole plan inside the panel (letterboxed, never cropped) so a
  // marker lands exactly where it was placed on the Site Map page.
  const boxRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<[number, number]>([0, 0]);
  const [ratio, setRatio] = useState<number | null>(null);
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBox([el.clientWidth, el.clientHeight]));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const hasImage = !!map?.has_image;
  let stage = { left: 0, top: 0, width: box[0], height: box[1] };
  if (hasImage && ratio && box[0] && box[1]) {
    const width = Math.min(box[0], box[1] * ratio);
    const height = width / ratio;
    stage = { left: (box[0] - width) / 2, top: (box[1] - height) / 2, width, height };
  }

  return (
    <div ref={boxRef} className="relative h-full min-h-[380px] overflow-hidden rounded-xl border border-border-1 bg-[#0f1011]">
      {!hasImage && (
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              "radial-gradient(ellipse at 55% 55%, rgba(91,156,248,0.06), transparent 60%), linear-gradient(#1c2423 1px, transparent 1px), linear-gradient(90deg, #1c2423 1px, transparent 1px)",
            backgroundSize: "100% 100%, 40px 40px, 40px 40px",
          }}
        />
      )}

      {/* One wrapping bar, so the title and the controls never overlap on a narrow panel. */}
      <div className="pointer-events-none absolute inset-x-4 top-4 z-10 flex flex-wrap items-start justify-between gap-2">
        <div className="flex flex-col items-start gap-2">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-bg-1/60 px-1.5 py-0.5 backdrop-blur-sm">
            <span className="text-[14px] font-semibold text-text-primary">Live site map</span>
            <span className="flex items-center gap-1.5 text-[11.5px] text-accent-green">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-green" /> {reporting} {reporting === 1 ? "camera" : "cameras"} reporting
            </span>
          </div>
          <div className="rounded-lg border border-border-2 bg-bg-1/85 px-3 py-2.5 text-[11.5px] leading-[1.9] text-text-secondary backdrop-blur max-[700px]:hidden">
            {(
              [
                ["bg-accent-blue", "Camera online"],
                ["bg-accent-amber", "Degraded"],
                ["bg-accent-red", "Alarm, last minute"],
                ["bg-text-disabled", "Offline"],
              ] as const
            ).map(([c, l]) => (
              <div key={l} className="flex items-center gap-2">
                <span className={`h-2.5 w-2.5 rounded-full ${c}`} /> {l}
              </div>
            ))}
          </div>
        </div>

        <div className="pointer-events-auto flex min-w-0 flex-wrap items-center justify-end gap-2">
          <SitePicker sites={map?.sites ?? []} value={map?.site_id} onChange={onSiteChange} />
          <Link
            to="/map"
            className="flex shrink-0 items-center gap-1.5 rounded-lg border border-border-2 bg-bg-1/85 px-2.5 py-1.5 text-[12px] text-text-secondary backdrop-blur hover:text-text-primary"
          >
            <MapIcon size={13} /> {map?.has_image ? "Edit map" : "Upload site plan"}
          </Link>
        </div>
      </div>

      <div className="absolute" style={stage}>
        {hasImage && map && (
          <img
            src={mapApi.imageUrl(map.site_id, map.image_version)}
            alt="Site plan"
            onLoad={(e) => setRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight)}
            className="absolute inset-0 h-full w-full"
            draggable={false}
          />
        )}
      {markers.map(({ name, at: [x, y] }) => {
        const camera = byName.get(name);
        const t = tone(camera, alarmed.has(name));
        return (
          <div
            key={name}
            className="absolute flex -translate-x-1/2 -translate-y-1/2 flex-col items-center"
            style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
            title={`${name}${camera ? ` · ${camera.state}` : " · not connected"}${camera?.location ? ` · ${camera.location}` : ""}`}
          >
            {t === "alarm" && <span className="absolute top-[-34px] h-24 w-24 rounded-full bg-accent-red/25 blur-xl" />}
            <span className={`relative grid h-7 w-7 place-items-center rounded-full border ${TONE[t].ring}`}>
              <span className={`h-3.5 w-3.5 rounded-full ${TONE[t].dot} ${t === "alarm" ? "animate-pulse" : ""}`} />
            </span>
            <span className="mt-1 whitespace-nowrap text-[12px] font-medium text-text-primary [text-shadow:0_1px_3px_#000]">{name}</span>
          </div>
        );
      })}
      </div>

      {markers.length === 0 && (
        <div className="absolute inset-0 grid place-items-center text-[12.5px] text-text-tertiary">
          No cameras connected yet. Add one from Live Monitor.
        </div>
      )}

      {unplaced.length > 0 && (
        <div className="absolute bottom-3 left-4 z-10 text-[11px] text-text-tertiary">
          {unplaced.length === cameras.length ? "Auto layout. " : `${unplaced.length} not placed yet, shown in an auto layout. `}
          <Link to="/map" className="text-text-secondary underline-offset-2 hover:underline">Place cameras on the Site Map</Link>
        </div>
      )}
    </div>
  );
}
