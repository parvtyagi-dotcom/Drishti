import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Play, RotateCcw, Square } from "lucide-react";
import { playSiren } from "../../lib/siren";

/**
 * A scripted walk-through for presenting the Site Map without a live
 * incident: an intruder crosses the Chandigarh University campus and each
 * camera on the route lights up in turn as it picks them up, with the path
 * drawn between them.
 *
 * Nothing here touches the backend - the cameras are stand-ins drawn only
 * while the walk-through plays, never added to the system. Positions are
 * fractions of the site plan (data/site_map.jpg, the "main" site).
 */
interface DemoStop {
  camera: string;
  place: string;
  event: string;
  x: number;
  y: number;
  // Which side of the dot the label goes, so neighbours don't cover each other.
  side: "left" | "right";
}

const ROUTE: DemoStop[] = [
  { camera: "CAM-01", place: "Main Gate (NH5)", event: "Person detected entering", x: 0.595, y: 0.205, side: "right" },
  { camera: "CAM-02", place: "A-Block Quad", event: "Target confirmed (face match)", x: 0.635, y: 0.36, side: "right" },
  { camera: "CAM-03", place: "Block B1", event: "Target re-acquired", x: 0.565, y: 0.40, side: "left" },
  { camera: "CAM-04", place: "Central Plaza", event: "Tracked across cameras", x: 0.545, y: 0.49, side: "left" },
  { camera: "CAM-05", place: "Sports Ground", event: "Heading for restricted zone", x: 0.555, y: 0.58, side: "right" },
  { camera: "CAM-06", place: "NC Hostel Perimeter", event: "Zone intrusion — siren", x: 0.52, y: 0.745, side: "right" },
];

// The rest of the campus network: on the map for the whole walk-through, but
// off the intruder's route, so they never light up - the point is that only
// the cameras that actually see the person react.
const BACKGROUND: Omit<DemoStop, "event">[] = [
  { camera: "CAM-07", place: "North-West Entrance", x: 0.35, y: 0.09, side: "right" },
  { camera: "CAM-08", place: "Shivalik Hostel", x: 0.31, y: 0.31, side: "left" },
  { camera: "CAM-09", place: "D-Block Hostels", x: 0.375, y: 0.28, side: "right" },
  { camera: "CAM-10", place: "Sports Complex", x: 0.32, y: 0.47, side: "right" },
  { camera: "CAM-11", place: "Hostel Annexe", x: 0.475, y: 0.33, side: "right" },
  { camera: "CAM-12", place: "A1 Block", x: 0.665, y: 0.255, side: "right" },
  { camera: "CAM-13", place: "A3 Block", x: 0.675, y: 0.44, side: "right" },
  { camera: "CAM-14", place: "East Parking", x: 0.68, y: 0.56, side: "right" },
  { camera: "CAM-15", place: "South Hostels", x: 0.43, y: 0.81, side: "left" },
];

const STEP_MS = 2600;

function clockAt(offsetSec: number, start: number) {
  return new Date(start + offsetSec * 1000).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function IntrusionDemo() {
  // null: not playing and nothing shown. Otherwise the index of the camera
  // the intruder has just reached.
  const [step, setStep] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const startedAt = useRef(Date.now());

  useEffect(() => {
    if (!running || step === null) return;
    if (step >= ROUTE.length - 1) {
      setRunning(false);
      void playSiren(2);
      return;
    }
    const t = window.setTimeout(() => setStep((s) => (s === null ? s : s + 1)), STEP_MS);
    return () => window.clearTimeout(t);
  }, [running, step]);

  function toggle(e: ReactPointerEvent<HTMLButtonElement>) {
    e.stopPropagation();
    if (running) {
      setRunning(false);
      setStep(null);
      return;
    }
    startedAt.current = Date.now();
    setStep(0);
    setRunning(true);
  }

  const finished = step !== null && !running;
  const seen = step === null ? [] : ROUTE.slice(0, step + 1);
  const current = step === null ? null : ROUTE[step];

  return (
    <>
      {/* The button sits over the map's top-right corner. pointer-up is
          stopped so a click never reaches the map's place-a-camera handler. */}
      <div className="absolute right-3 top-3 z-20 flex items-center gap-2" onPointerUp={(e) => e.stopPropagation()}>
        {finished && (
          <button
            onPointerUp={(e) => { e.stopPropagation(); setStep(null); }}
            className="rounded-md border border-border-2 bg-bg-0/90 px-3 py-2 text-[12px] font-semibold text-text-secondary hover:text-text-primary"
          >
            Clear
          </button>
        )}
        <button
          onPointerUp={toggle}
          aria-label={running ? "Stop walk-through" : "Play intrusion walk-through"}
          className={`inline-flex items-center gap-1.5 rounded-md px-3.5 py-2 text-[12.5px] font-bold shadow-lg ${
            running ? "bg-accent-red text-white" : "bg-accent-blue text-on-accent"
          }`}
        >
          {running ? <Square size={13} fill="currentColor" /> : finished ? <RotateCcw size={14} /> : <Play size={14} fill="currentColor" />}
          {running ? "Stop" : finished ? "Replay" : "Play"}
        </button>
      </div>

      {step !== null && (
        <>
          {/* Path between the cameras that have seen the intruder so far. */}
          <svg className="pointer-events-none absolute inset-0 z-10 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <marker id="demo-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--color-accent-red)" />
              </marker>
            </defs>
            {seen.slice(1).map((s, i) => (
              <line
                key={s.camera}
                x1={seen[i].x * 100} y1={seen[i].y * 100} x2={s.x * 100} y2={s.y * 100}
                stroke="var(--color-accent-red)" strokeWidth={2.5} strokeDasharray="6 4"
                vectorEffect="non-scaling-stroke" markerEnd="url(#demo-arrow)"
              />
            ))}
          </svg>

          {/* The rest of the network: neutral throughout - the target never passes them. */}
          {BACKGROUND.map((s) => (
            <div
              key={s.camera}
              className="pointer-events-none absolute z-10 h-3 w-3 -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${s.x * 100}%`, top: `${s.y * 100}%` }}
            >
              <span className="block h-3 w-3 rounded-full border-2 border-bg-0 bg-text-tertiary" />
              <span
                className={`absolute top-1/2 -translate-y-1/2 whitespace-nowrap rounded bg-bg-0/75 px-1 py-[1px] text-[9.5px] font-semibold text-text-secondary ${
                  s.side === "right" ? "left-[calc(100%+6px)]" : "right-[calc(100%+6px)]"
                }`}
              >
                {s.camera} · {s.place}
              </span>
            </div>
          ))}

          {/* The route: neutral like the rest until the intruder reaches each one,
              so the path is not given away in advance. */}
          {ROUTE.map((s, i) => {
            const state = i < (step ?? 0) ? "passed" : i === step ? "now" : "waiting";
            return (
              <div
                key={s.camera}
                className={`pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-1/2 ${state === "waiting" ? "h-3 w-3" : "h-4 w-4"}`}
                style={{ left: `${s.x * 100}%`, top: `${s.y * 100}%` }}
              >
                <span className="relative flex h-full w-full items-center justify-center">
                  {state === "now" && <span className="absolute inline-flex h-10 w-10 animate-ping rounded-full bg-accent-red/50" />}
                  <span
                    className={`relative h-full w-full rounded-full border-2 border-bg-0 ${
                      state === "now" ? "bg-accent-red shadow-[0_0_0_6px_rgba(239,68,68,0.3)]"
                        : state === "passed" ? "bg-accent-amber"
                          : "bg-text-tertiary"
                    }`}
                  />
                </span>
                <span
                  className={`absolute top-1/2 -translate-y-1/2 whitespace-nowrap rounded font-semibold ${
                    s.side === "right" ? "left-[calc(100%+7px)]" : "right-[calc(100%+7px)]"
                  } ${
                    state === "now" ? "bg-accent-red px-1.5 py-[1px] text-[10.5px] text-white"
                      : state === "passed" ? "bg-bg-0/85 px-1.5 py-[1px] text-[10.5px] text-text-primary"
                        : "bg-bg-0/75 px-1 py-[1px] text-[9.5px] text-text-secondary"
                  }`}
                >
                  {s.camera} · {s.place}
                  {state !== "waiting" ? ` · ${clockAt(i * 40, startedAt.current)}` : ""}
                </span>
              </div>
            );
          })}

          {/* What just happened, for the audience. */}
          {current && (
            <div className="pointer-events-none absolute bottom-3 left-3 z-20 max-w-[340px] rounded-lg border border-accent-red/40 bg-bg-0/90 px-3.5 py-2.5 shadow-lg">
              <div className="mb-1 flex items-center gap-2 text-[10.5px] font-bold uppercase tracking-wide text-accent-red">
                <span className="h-2 w-2 animate-pulse rounded-full bg-accent-red" />
                {finished ? "Intrusion — patrol dispatched" : "Intruder tracked"}
                <span className="ml-auto font-normal normal-case text-text-tertiary">simulation · {step! + 1}/{ROUTE.length} · {ROUTE.length + BACKGROUND.length} cameras</span>
              </div>
              <div className="text-[13px] font-semibold text-text-primary">{current.camera} · {current.place}</div>
              <div className="text-[12px] text-text-secondary">{current.event}</div>
              <div className="mt-1 text-[11px] text-text-tertiary">{seen.map((s) => s.camera).join(" → ")}</div>
            </div>
          )}
        </>
      )}
    </>
  );
}
