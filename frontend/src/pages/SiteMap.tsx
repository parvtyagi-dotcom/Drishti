import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { ImageUp, Loader2, Map as MapIcon, MapPin, Pencil, Plus, Route, Trash2, X } from "lucide-react";
import { Panel } from "../components/common/Panel";
import type { SessionUser } from "../api/auth";
import { ApiError } from "../api/client";
import { liveApi, type LiveCamera, type LiveEvent } from "../api/live";
import { mapApi, type PathPrediction, type SiteMap as SiteMapState } from "../api/map";
import { eventMeta } from "../lib/eventMeta";
import { SitePicker } from "../components/dashboard/SitePicker";
import { IntrusionDemo } from "../components/sitemap/IntrusionDemo";
import { useSiteChoice } from "../hooks/useSiteChoice";

const REFRESH_MS = 3000;
// A camera glows red for this long after an alarm-level event.
const ALARM_GLOW_SEC = 60;
// How far back the watched person or vehicle's path is drawn.
const PATH_WINDOW_SEC = 30 * 60;
const SIGHTING_TYPES = new Set([
  "TARGET_CONFIRMED", "TARGET_REACQUIRED", "TARGET_CROSS_CAMERA_MATCH", "TARGET_ZONE_INTRUSION",
  "TARGET_BEHAVIOR_ALERT", "TARGET_VEHICLE_FOUND", "TARGET_VEHICLE_REACQUIRED",
]);

interface Stop {
  camera: string;
  time: number;
  type: string;
}

function clock(seconds: number) {
  return new Date(seconds * 1000).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit" });
}

/** Where the watched person or vehicle went: sightings in time order, with
 * repeated sightings on the same camera merged into one stop. */
function pursuitPath(events: LiveEvent[], now: number): Stop[] {
  const stops: Stop[] = [];
  events
    .filter((e) => SIGHTING_TYPES.has(e.type) && now - e.wall_time <= PATH_WINDOW_SEC)
    .sort((a, b) => a.wall_time - b.wall_time)
    .forEach((e) => {
      const last = stops[stops.length - 1];
      if (!last || last.camera !== e.camera_name) stops.push({ camera: e.camera_name, time: e.wall_time, type: e.type });
    });
  return stops;
}

// Cameras stay neutral until the watched person or vehicle passes them: red
// while alarming, amber once they are part of the path, grey otherwise. Only
// the cameras that actually saw the target stand out.
function markerColour(camera: LiveCamera | undefined, alarm: boolean, onPath: boolean) {
  if (alarm) return "bg-accent-red shadow-[0_0_0_6px_rgba(239,68,68,0.25)] animate-pulse";
  if (onPath) return "bg-accent-amber";
  if (!camera || camera.state === "offline" || camera.state === "stopped") return "bg-text-disabled opacity-50";
  return "bg-text-tertiary";
}

export function SiteMap({ user }: { user?: SessionUser | null }) {
  const canEdit = user?.role === "ADMIN" || user?.role === "OPERATOR";
  const [map, setMap] = useState<SiteMapState | null>(null);
  const [cameras, setCameras] = useState<LiveCamera[]>([]);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [placing, setPlacing] = useState<string | null>(null);
  const [dragging, setDragging] = useState<{ name: string; x: number; y: number } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PathPrediction | null>(null);
  const [site, chooseSite] = useSiteChoice();
  const [naming, setNaming] = useState<{ mode: "add" | "rename"; value: string } | null>(null);
  const boardRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const [m, c, e, p] = await Promise.all([
        mapApi.get(site), liveApi.listCameras(), liveApi.recentEvents(300),
        mapApi.prediction(site).catch(() => ({ prediction: null })),
      ]);
      setPrediction(p.prediction);
      setMap(m);
      setCameras(c);
      setEvents(e);
      setNow(Date.now() / 1000);
    } catch (err) {
      // A remembered site that was removed since: fall back to the default.
      if (err instanceof ApiError && err.status === 404 && site) return chooseSite(null);
      setError(err instanceof ApiError ? err.message : "Could not load the site map.");
    }
    // chooseSite only writes the choice, which re-runs this through `site`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [site]);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [load]);

  const byName = useMemo(() => new Map(cameras.map((c) => [c.name, c])), [cameras]);
  const alarmed = useMemo(
    () => new Set(events.filter((e) => (e.alarm || e.severity === "CRITICAL") && now - e.wall_time <= ALARM_GLOW_SEC).map((e) => e.camera_name)),
    [events, now],
  );
  const path = useMemo(() => pursuitPath(events, now), [events, now]);
  const recentAlarms = useMemo(
    () => events.filter((e) => e.alarm || e.severity === "CRITICAL" || e.severity === "HIGH").slice(-8).reverse(),
    [events],
  );

  const positions = { ...(map?.positions ?? {}) };
  if (dragging) positions[dragging.name] = [dragging.x, dragging.y];
  const placedNames = Object.keys(positions);
  const unplaced = cameras.map((c) => c.name).filter((n) => !(n in positions));
  const pathPoints = path.filter((s) => s.camera in positions);

  function fractionAt(clientX: number, clientY: number): [number, number] | null {
    const box = boardRef.current?.getBoundingClientRect();
    if (!box) return null;
    const clamp = (v: number) => Math.min(1, Math.max(0, v));
    return [clamp((clientX - box.left) / box.width), clamp((clientY - box.top) / box.height)];
  }

  async function save(name: string, x: number | null, y: number | null) {
    try {
      const { positions: saved } = await mapApi.place(map?.site_id ?? site, name, x, y);
      setMap((m) => (m ? { ...m, positions: saved } : m));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save the position.");
    }
  }

  function boardClick(e: ReactPointerEvent<HTMLDivElement>) {
    if (!placing || !canEdit) return;
    const at = fractionAt(e.clientX, e.clientY);
    if (!at) return;
    void save(placing, at[0], at[1]);
    setPlacing(null);
  }

  function startDrag(name: string, e: ReactPointerEvent<HTMLButtonElement>) {
    if (!canEdit) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    const [x, y] = positions[name];
    setDragging({ name, x, y });
  }

  function moveDrag(e: ReactPointerEvent<HTMLButtonElement>) {
    if (!dragging) return;
    const at = fractionAt(e.clientX, e.clientY);
    if (at) setDragging({ ...dragging, x: at[0], y: at[1] });
  }

  function endDrag() {
    if (!dragging) return;
    void save(dragging.name, dragging.x, dragging.y);
    setDragging(null);
  }

  async function upload(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await mapApi.uploadImage(map?.site_id ?? site, file);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not upload the plan.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function submitName() {
    if (!naming || !naming.value.trim()) return setNaming(null);
    try {
      if (naming.mode === "add") {
        const { id } = await mapApi.addSite(naming.value.trim());
        chooseSite(id);
      } else if (map) {
        await mapApi.renameSite(map.site_id, naming.value.trim());
        await load();
      }
      setNaming(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not save the site.");
    }
  }

  async function removeSite() {
    if (!map || !window.confirm(`Remove ${map.site_name} and its site plan? Camera positions on it are lost.`)) return;
    try {
      await mapApi.removeSite(map.site_id);
      chooseSite(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not remove the site.");
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-text-primary">
            Site Map{map && map.sites.length > 1 ? <span className="font-normal text-text-tertiary"> · {map.site_name}</span> : null}
          </h1>
          <p className="mt-1 max-w-[680px] text-[12.5px] text-text-tertiary">
            Every camera on a plan of the post. Markers turn red when a camera raises an alarm, and arrows trace
            where a watched person or vehicle has been seen across cameras in the last 30 minutes.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <SitePicker sites={map?.sites ?? []} value={map?.site_id} onChange={chooseSite} />
          {canEdit && naming && (
            <form
              onSubmit={(e) => { e.preventDefault(); void submitName(); }}
              className="flex items-center gap-1.5"
            >
              <input
                autoFocus
                value={naming.value}
                onChange={(e) => setNaming({ ...naming, value: e.target.value })}
                placeholder="BPIT Rohini"
                aria-label="Site name"
                maxLength={80}
                className="w-44 rounded border border-border-2 bg-bg-2 px-2.5 py-[7px] text-[12px] text-text-primary focus:border-accent-blue focus:outline-none"
              />
              <button type="submit" className="rounded border border-border-2 bg-bg-3 px-3 py-2 text-[12px] font-semibold text-text-primary hover:bg-bg-4">
                {naming.mode === "add" ? "Add" : "Save"}
              </button>
              <button type="button" onClick={() => setNaming(null)} aria-label="Cancel" className="p-1.5 text-text-tertiary hover:text-text-primary"><X size={14} /></button>
            </form>
          )}
          {canEdit && !naming && (
            <>
              <button onClick={() => setNaming({ mode: "add", value: "" })} className="inline-flex items-center gap-1.5 rounded border border-border-2 bg-bg-2 px-3 py-2 text-[12px] font-semibold text-text-secondary hover:bg-bg-3 hover:text-text-primary">
                <Plus size={13} /> Add site
              </button>
              {map && (
                <button onClick={() => setNaming({ mode: "rename", value: map.site_name })} aria-label="Rename site" title="Rename site" className="rounded border border-border-2 bg-bg-2 p-2 text-text-secondary hover:bg-bg-3 hover:text-text-primary">
                  <Pencil size={13} />
                </button>
              )}
              {map && map.sites.length > 1 && (
                <button onClick={() => void removeSite()} aria-label="Remove site" title="Remove site" className="rounded border border-border-2 bg-bg-2 p-2 text-text-secondary hover:border-accent-red/40 hover:text-accent-red">
                  <Trash2 size={13} />
                </button>
              )}
            </>
          )}
        {canEdit && (
          <div className="flex items-center gap-2">
            <input ref={fileRef} id="site-plan" type="file" accept="image/*" className="hidden" onChange={(e) => void upload(e.target.files?.[0])} />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="inline-flex items-center gap-1.5 rounded border border-border-2 bg-bg-2 px-3 py-2 text-[12px] font-semibold text-text-secondary hover:bg-bg-3 hover:text-text-primary disabled:opacity-40"
            >
              {uploading ? <Loader2 size={13} className="animate-spin" /> : <ImageUp size={13} />}
              {map?.has_image ? "Replace site plan" : "Upload site plan"}
            </button>
          </div>
        )}
        </div>
      </div>

      {error && (
        <div className="flex items-start justify-between gap-3 rounded border border-accent-red/35 bg-accent-red/10 px-4 py-3 text-[12.5px] text-accent-red">
          {error}
          <button onClick={() => setError(null)} aria-label="Dismiss"><X size={14} /></button>
        </div>
      )}

      {canEdit && unplaced.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded border border-border-1 bg-bg-2 px-3 py-2 text-[12px]">
          <span className="text-text-tertiary">{placing ? `Click the map where ${placing} is.` : "Not on the map yet, click one to place it:"}</span>
          {unplaced.map((name) => (
            <button
              key={name}
              onClick={() => setPlacing(placing === name ? null : name)}
              className={`inline-flex items-center gap-1 rounded border px-2 py-1 font-semibold ${
                placing === name ? "border-accent-blue bg-accent-blue/15 text-accent-blue" : "border-border-2 text-text-secondary hover:bg-bg-3"
              }`}
            >
              <MapPin size={12} /> {name}
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-[minmax(0,1fr)_300px] gap-4 max-[1100px]:grid-cols-1">
        <div className="overflow-hidden rounded-[10px] border border-border-1 bg-bg-2">
          <div
            ref={boardRef}
            onPointerUp={boardClick}
            className={`relative w-full select-none ${placing ? "cursor-crosshair" : ""}`}
            style={map?.has_image ? undefined : { aspectRatio: "16 / 9" }}
          >
            {map?.has_image ? (
              <img src={mapApi.imageUrl(map.site_id, map.image_version)} alt="Site plan" className="block w-full" draggable={false} />
            ) : (
              <div
                className="absolute inset-0 flex items-center justify-center text-[12.5px] text-text-tertiary"
                style={{
                  backgroundImage: "linear-gradient(var(--color-border-1) 1px, transparent 1px), linear-gradient(90deg, var(--color-border-1) 1px, transparent 1px)",
                  backgroundSize: "40px 40px",
                }}
              >
                {placedNames.length === 0 && (
                  <span className="flex items-center gap-2 rounded bg-bg-2/90 px-3 py-2">
                    <MapIcon size={14} /> Upload a site plan or satellite picture, then place each camera on it.
                  </span>
                )}
              </div>
            )}

            {/* The watched path, drawn under the markers. */}
            <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <marker id="path-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M0,0 L10,5 L0,10 z" fill="var(--color-accent-red)" />
                </marker>
              </defs>
              {pathPoints.slice(1).map((stop, i) => {
                const [x1, y1] = positions[pathPoints[i].camera];
                const [x2, y2] = positions[stop.camera];
                return (
                  <line key={`${stop.camera}-${stop.time}`} x1={x1 * 100} y1={y1 * 100} x2={x2 * 100} y2={y2 * 100}
                    stroke="var(--color-accent-red)" strokeWidth={2.5} strokeDasharray="6 4"
                    vectorEffect="non-scaling-stroke" markerEnd="url(#path-arrow)" />
                );
              })}
              {prediction && prediction.from[1] in positions && (() => {
                const [x1, y1] = positions[prediction.from[1]];
                const [x2, y2] = prediction.point;
                return (
                  <>
                    <line x1={x1 * 100} y1={y1 * 100} x2={x2 * 100} y2={y2 * 100}
                      stroke="var(--color-accent-amber)" strokeWidth={2.5} strokeDasharray="2 5"
                      vectorEffect="non-scaling-stroke" />
                    <circle cx={x2 * 100} cy={y2 * 100} r={2.2} fill="none" stroke="var(--color-accent-amber)"
                      strokeWidth={2} vectorEffect="non-scaling-stroke" className="animate-pulse" />
                  </>
                );
              })()}
            </svg>

            {placedNames.map((name) => {
              const [x, y] = positions[name];
              const camera = byName.get(name);
              const stop = [...pathPoints].reverse().find((s) => s.camera === name);
              return (
                <button
                  key={name}
                  onPointerDown={(e) => startDrag(name, e)}
                  onPointerMove={moveDrag}
                  onPointerUp={(e) => { e.stopPropagation(); endDrag(); }}
                  title={`${name}${camera ? ` · ${camera.state}` : " · not connected"}${camera?.location ? ` · ${camera.location}` : ""}`}
                  className={`absolute flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-1 ${canEdit ? "cursor-grab active:cursor-grabbing" : "cursor-default"}`}
                  style={{ left: `${x * 100}%`, top: `${y * 100}%`, touchAction: "none" }}
                >
                  <span className={`h-3.5 w-3.5 rounded-full border-2 border-bg-0 ${markerColour(camera, alarmed.has(name), !!stop)}`} />
                  <span className="whitespace-nowrap rounded bg-bg-0/85 px-1.5 py-[1px] text-[10.5px] font-semibold text-text-primary">
                    {name}{stop ? ` · ${clock(stop.time)}` : ""}
                  </span>
                </button>
              );
            })}

            <IntrusionDemo />
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <Panel title="Where to send the patrol" meta="prediction">
            {prediction ? (
              <div className="flex flex-col gap-1.5 text-[12.5px]">
                <div className="font-semibold text-accent-amber">{prediction.advice}</div>
                <div className="text-text-secondary">Path so far: {prediction.trail.join(" → ")}</div>
                <div className="text-[11.5px] text-text-tertiary">Worked out from a {prediction.basis}. A first guess for where to look, not a certainty.</div>
              </div>
            ) : (
              <div className="text-[12.5px] text-text-tertiary">
                Appears when alerts come from two or more cameras placed on this map in a row.
              </div>
            )}
          </Panel>
          <Panel title="Watched path" meta="last 30 min">
            {path.length === 0 ? (
              <div className="text-[12.5px] text-text-tertiary">No watched person or vehicle has been seen recently.</div>
            ) : (
              <ol className="flex flex-col gap-2">
                {path.map((s, i) => (
                  <li key={`${s.camera}-${s.time}`} className="flex items-start gap-2 text-[12px]">
                    <Route size={13} className="mt-[2px] shrink-0 text-accent-red" />
                    <span>
                      <span className="font-mono text-text-tertiary">{clock(s.time)}</span>{" "}
                      <span className="font-semibold text-text-primary">{s.camera}</span>
                      <span className="text-text-secondary"> · {eventMeta(s.type).label}</span>
                      {!(s.camera in positions) && <span className="text-text-tertiary"> (not on the map)</span>}
                      {i === path.length - 1 && <span className="ml-1 rounded bg-accent-red/15 px-1 text-[10px] font-bold text-accent-red">LATEST</span>}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </Panel>
          <Panel title="Recent alarms">
            {recentAlarms.length === 0 ? (
              <div className="text-[12.5px] text-text-tertiary">No alarms yet.</div>
            ) : (
              <ul className="flex flex-col gap-2">
                {recentAlarms.map((e, i) => (
                  <li key={e.event_id ?? i} className="text-[12px] text-text-secondary">
                    <span className="font-mono text-text-tertiary">{clock(e.wall_time)}</span>{" "}
                    <span className="font-semibold text-text-primary">{e.camera_name}</span> · {eventMeta(e.type).label}
                  </li>
                ))}
              </ul>
            )}
          </Panel>
          {canEdit && placedNames.length > 0 && (
            <div className="text-[11.5px] text-text-tertiary">Drag a marker to move it. To take a camera off the map, click its name below.</div>
          )}
          {canEdit && placedNames.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {placedNames.map((name) => (
                <button key={name} onClick={() => void save(name, null, null)}
                  className="inline-flex items-center gap-1 rounded border border-border-2 px-2 py-1 text-[11px] text-text-tertiary hover:bg-bg-3 hover:text-text-primary">
                  <X size={11} /> {name}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
