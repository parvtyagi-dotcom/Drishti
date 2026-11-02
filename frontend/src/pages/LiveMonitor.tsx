import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle, BellRing, Plus, Radio, RefreshCw, RotateCw, Settings2, Siren, Square, VolumeX, Wifi,
  WifiOff, X,
} from "lucide-react";
import { Panel } from "../components/common/Panel";
import { StatusBadge } from "../components/common/StatusBadge";
import { AddCameraForm } from "../components/live/AddCameraForm";
import { CameraHealthPanel } from "../components/live/CameraHealthPanel";
import { CameraSetupPanel } from "../components/live/CameraSetupPanel";
import { EvidencePanel } from "../components/live/EvidencePanel";
import { TalkDownControls } from "../components/live/TalkDownControls";
import { eventMeta } from "../lib/eventMeta";
import { liveApi, streamUrl, type LiveCamera, type LiveEvent } from "../api/live";
import { useLiveEvents } from "../hooks/useLiveEvents";
import { playAlarmSound, playSiren, vibrateAlert } from "../lib/siren";

const STATUS_POLL_MS = 2000;

const CAMERA_SEVERITY: Record<string, "green" | "amber" | "red" | "neutral"> = {
  online: "green",
  connecting: "amber",
  reconnecting: "amber",
  offline: "red",
  stopped: "neutral",
};

function cameraLabel(camera: LiveCamera): string {
  if (camera.degraded) return "DEGRADED";
  return camera.state.toUpperCase();
}

function CameraRow({
  camera, selected, onSelect,
}: { camera: LiveCamera; selected: boolean; onSelect: () => void }) {
  return (
    <button
      onClick={onSelect}
      className={`flex w-full flex-col gap-1.5 rounded border-l-2 px-3 py-2.5 text-left transition-colors ${
        selected
          ? "border-accent-blue bg-accent-blue/10"
          : "border-transparent hover:bg-bg-3"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[13px] font-semibold text-text-primary">{camera.name}</span>
        <StatusBadge
          label={cameraLabel(camera)}
          severity={camera.degraded ? "amber" : CAMERA_SEVERITY[camera.state] ?? "neutral"}
          pulse={camera.state === "online" && !camera.degraded}
        />
      </div>
      <div className="truncate text-[11px] text-text-tertiary">{camera.location ?? "—"}</div>
      <div className="flex items-center gap-3 font-mono text-[10.5px] text-text-tertiary">
        <span>{camera.measured_fps.toFixed(1)} fps</span>
        <span>{camera.width}×{camera.height}</span>
        {camera.reconnect_count > 0 && (
          <span className="text-accent-amber">↻{camera.reconnect_count}</span>
        )}
      </div>
      {camera.modules.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {camera.modules.map((m) => (
            <span key={m} className="rounded bg-bg-3 px-1.5 py-[1px] text-[10px] text-text-secondary">
              {m}
            </span>
          ))}
        </div>
      )}
    </button>
  );
}

function VerdictButtons({ event }: { event: LiveEvent }) {
  const [done, setDone] = useState<string | null>(null);
  if (!event.event_id || (event.severity !== "HIGH" && event.severity !== "CRITICAL")) return null;
  async function send(verdict: "false" | "real") {
    const note = verdict === "false" ? window.prompt("What was it? (e.g. stray dog, tree shadow) - optional") ?? undefined : undefined;
    try {
      const r = await liveApi.verdict(event.event_id!, verdict, note);
      setDone(verdict === "false"
        ? `Marked false (${r.pattern_marks} at this place and hour; 2 silence the siren next time)`
        : r.forgotten ? `Marked real; forgot ${r.forgotten} false marks here` : "Marked real");
    } catch {
      setDone("Could not save");
    }
  }
  if (done) return <div className="mt-1 text-[10.5px] text-text-tertiary">{done}</div>;
  return (
    <div className="mt-1 flex gap-1.5">
      <button onClick={() => void send("false")} className="rounded border border-border-2 px-1.5 py-0.5 text-[10.5px] text-text-tertiary hover:text-text-primary">False alarm</button>
      <button onClick={() => void send("real")} className="rounded border border-border-2 px-1.5 py-0.5 text-[10.5px] text-text-tertiary hover:text-text-primary">Real</button>
    </div>
  );
}

function EventRow({ event }: { event: LiveEvent }) {
  const meta = eventMeta(event.type);
  const Icon = meta.icon;
  const time = new Date(event.wall_time * 1000).toLocaleTimeString([], { hour12: false });
  return (
    <div className="flex items-start gap-2.5 border-b border-border-1 py-2.5 last:border-0">
      <Icon size={14} className="mt-[3px] shrink-0 opacity-80" />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="text-[12.5px] font-semibold text-text-primary">{meta.label}</span>
          {event.alarm && (
            <span className="rounded bg-accent-red/15 px-1.5 text-[10px] font-bold text-accent-red">
              SIREN
            </span>
          )}
          {event.likely_false && (
            <span className="rounded bg-accent-amber/15 px-1.5 text-[10px] font-bold text-accent-amber">
              LIKELY FALSE
            </span>
          )}
        </div>
        {/* The reason is the point: an operator has to be able to verify or
            dismiss an alert without reverse-engineering a score. */}
        <div className="mt-0.5 text-[11.5px] leading-snug text-text-secondary">{event.reason}</div>
        <VerdictButtons event={event} />
      </div>
      <div className="shrink-0 text-right">
        <div className="font-mono text-[10.5px] text-text-tertiary">{time}</div>
        <div className="text-[10px] text-text-tertiary">{event.camera_name}</div>
      </div>
    </div>
  );
}

export function LiveMonitor() {
  const [cameras, setCameras] = useState<LiveCamera[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streamNonce, setStreamNonce] = useState(() => Date.now());
  const [setupOpen, setSetupOpen] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const { events, connected, latestAlarm, clearAlarm } = useLiveEvents();

  const refresh = useCallback(async () => {
    try {
      const list = await liveApi.listCameras();
      setCameras(list);
      setError(null);
      setSelectedId((current) => current ?? list[0]?.camera_id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load cameras");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), STATUS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const selected = useMemo(
    () => cameras.find((c) => c.camera_id === selectedId) ?? null,
    [cameras, selectedId],
  );

  const alarms = useMemo(() => events.filter((e) => e.alarm), [events]);

  // Sound the siren here as well as on the dedicated siren page. The page was
  // built to run on a separate device on the table, like a real border post,
  // and it still does - but an operator watching this screen saw a red banner
  // and a SIREN badge and heard nothing, which reads as an alarm that failed
  // rather than one that was never meant to play here.
  const [audioBlocked, setAudioBlocked] = useState(false);
  const soundedAlarmRef = useRef<string | null>(null);

  useEffect(() => {
    if (!latestAlarm) return;
    // Keyed on the event, so re-renders while a banner is up stay silent.
    const key = latestAlarm.event_id ?? `${latestAlarm.type}@${latestAlarm.wall_time}`;
    if (soundedAlarmRef.current === key) return;
    soundedAlarmRef.current = key;
    vibrateAlert();
    void playAlarmSound(latestAlarm.type).then((played) => setAudioBlocked(!played));
  }, [latestAlarm]);

  const healthSummary = useMemo(() => {
    const online = cameras.filter((c) => c.state === "online" && !c.degraded).length;
    return { online, problems: cameras.length - online };
  }, [cameras]);

  const toggleCamera = async (camera: LiveCamera) => {
    try {
      if (camera.running) await liveApi.stop(camera.camera_id);
      else await liveApi.start(camera.camera_id);
      // Force the <img> to open a new MJPEG connection; the previous one is
      // dead the moment the session stops.
      setStreamNonce(Date.now());
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Camera control failed");
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-bold text-text-primary">
            <Radio size={18} className="text-accent-red" />
            Live Monitor
          </h1>
          <p className="mt-1 text-[12.5px] text-text-tertiary">
            Continuous analysis of connected cameras. Events appear as they happen.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge
            label={connected ? "EVENT FEED LIVE" : "FEED DISCONNECTED"}
            severity={connected ? "green" : "red"}
            pulse={connected}
          />
          <a
            href="/api/live/siren"
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 rounded border border-border-2 px-3 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3 hover:text-text-primary"
          >
            <Siren size={14} /> Open siren panel
          </a>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded border border-accent-red/35 bg-accent-red/10 px-4 py-3 text-[12.5px] text-accent-red">
          <AlertTriangle size={15} /> {error}
        </div>
      )}

      {/* Browsers refuse audio until the page has been interacted with, so a
          silent alarm is indistinguishable from a broken one. Say which it is,
          and offer the one click that fixes it. */}
      {audioBlocked && (
        <button
          onClick={() => void playSiren().then((played) => setAudioBlocked(!played))}
          className="flex w-full items-center gap-2.5 rounded border border-accent-amber/40 bg-accent-amber/10 px-4 py-3 text-left text-[12.5px] text-accent-amber"
        >
          <VolumeX size={16} className="shrink-0" />
          <span>
            <b>Alert sound blocked</b> — your browser muted it. Click here to enable the siren
            on this screen.
          </span>
        </button>
      )}

      {latestAlarm && (
        <button
          onClick={clearAlarm}
          className="flex w-full items-center gap-3 rounded border border-accent-red/45 bg-accent-red/12 px-4 py-3 text-left"
        >
          <BellRing size={18} className="shrink-0 animate-pulse text-accent-red" />
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-bold text-accent-red">
              {eventMeta(latestAlarm.type).label} · {latestAlarm.camera_name}
            </div>
            <div className="truncate text-[12px] text-text-secondary">{latestAlarm.reason}</div>
          </div>
          <span className="shrink-0 text-[11px] text-text-tertiary">dismiss</span>
        </button>
      )}

      {cameras.length > 0 && (
        <Panel
          title="Camera health"
          meta={
            healthSummary.problems > 0
              ? `${healthSummary.problems} needing attention`
              : `${healthSummary.online} online`
          }
        >
          <CameraHealthPanel cameras={cameras} onChanged={() => void refresh()} />
        </Panel>
      )}

      <div className="grid grid-cols-[260px_minmax(0,1fr)_320px] gap-4 max-[1500px]:grid-cols-[240px_minmax(0,1fr)] max-[1100px]:grid-cols-1">
        <Panel
          title="Cameras"
          meta={`${cameras.length}`}
          actions={
            <button
              onClick={() => setAddOpen((open) => !open)}
              className={`rounded border p-1.5 ${
                addOpen
                  ? "border-accent-blue bg-accent-blue/10 text-text-primary"
                  : "border-border-2 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
              }`}
              title={addOpen ? "Close" : "Add a camera"}
            >
              {addOpen ? <X size={13} /> : <Plus size={13} />}
            </button>
          }
          tight
        >
          {addOpen && (
            <div className="mb-3 border-b border-border-1 pb-3">
              <AddCameraForm
                onAdded={(camera) => {
                  // Phone pairing reports success without a camera to select -
                  // the QR code stays on screen until the phone connects.
                  if (camera) {
                    setAddOpen(false);
                    setSelectedId(camera.camera_id);
                  }
                  setStreamNonce(Date.now());
                  void refresh();
                }}
              />
            </div>
          )}
          <div className="flex flex-col gap-1">
            {cameras.length === 0 && (
              <div className="px-3 py-6 text-center text-[12px] text-text-tertiary">
                No cameras connected.
                <div className="mt-1 text-[11.5px]">
                  Use Connect Device to add a phone, or Add camera for an RTSP feed.
                </div>
              </div>
            )}
            {cameras.map((camera) => (
              <CameraRow
                key={camera.camera_id}
                camera={camera}
                selected={camera.camera_id === selectedId}
                onSelect={() => setSelectedId(camera.camera_id)}
              />
            ))}
          </div>
        </Panel>

        <Panel
          title={selected ? selected.name : "Live feed"}
          meta={selected?.location ?? undefined}
          actions={
            selected && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setSetupOpen((open) => !open)}
                  className={`flex items-center gap-1.5 rounded border px-2.5 py-1.5 text-[12px] ${
                    setupOpen
                      ? "border-accent-blue bg-accent-blue/10 text-text-primary"
                      : "border-border-2 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
                  }`}
                >
                  {setupOpen ? <X size={12} /> : <Settings2 size={12} />} Configure
                </button>
                {(selected.source_kind === "webcam" || selected.source_kind === "rtsp") && (
                  <button
                    onClick={async () => {
                      await liveApi.rotate(selected.camera_id, ((selected.rotate ?? 0) + 90) % 360);
                      await refresh();
                      setStreamNonce(Date.now());
                    }}
                    title="Rotate the picture 90° clockwise (for a sideways phone webcam)"
                    className="flex items-center gap-1.5 rounded border border-border-2 px-2.5 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3 hover:text-text-primary"
                  >
                    <RotateCw size={12} /> Rotate{selected.rotate ? ` ${selected.rotate}°` : ""}
                  </button>
                )}
                {selected.source_kind !== "file" && (
                  <select
                    value={selected.max_width >= 1280 ? "accuracy" : selected.max_width <= 640 ? "speed" : "balanced"}
                    onChange={async (e) => {
                      await liveApi.setQuality(selected.camera_id, e.target.value as "accuracy" | "balanced" | "speed");
                      await refresh();
                      setStreamNonce(Date.now());
                    }}
                    title="Accuracy: sharper, for plates and far faces. Speed: smoother, for a busy CPU."
                    className="rounded border border-border-2 bg-bg-3 px-2 py-1.5 text-[12px] text-text-secondary"
                  >
                    <option value="accuracy">Accuracy</option>
                    <option value="balanced">Balanced</option>
                    <option value="speed">Speed</option>
                  </select>
                )}
                <button
                  onClick={() => setStreamNonce(Date.now())}
                  title="Reconnect video"
                  className="rounded border border-border-2 p-1.5 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
                >
                  <RefreshCw size={13} />
                </button>
                <button
                  onClick={() => void toggleCamera(selected)}
                  className="flex items-center gap-1.5 rounded border border-border-2 px-2.5 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3 hover:text-text-primary"
                >
                  {selected.running ? <><Square size={12} /> Stop</> : <><Radio size={12} /> Start</>}
                </button>
              </div>
            )
          }
          tight
        >
          {selected && setupOpen ? (
            <div className="flex flex-col gap-4">
              <div className="rounded border border-border-1 bg-bg-1 px-3 py-2 text-[12px] text-text-tertiary">
                Configuring <span className="font-semibold text-text-primary">{selected.name}</span>.
                Zones are drawn on a still frame captured just now; the coordinates are relative, so
                they line up with the live feed.
              </div>
              <CameraSetupPanel
                camera={selected}
                onApplied={() => {
                  setSetupOpen(false);
                  setStreamNonce(Date.now());
                  void refresh();
                }}
              />
            </div>
          ) : selected ? (
            <div className="flex flex-col gap-2">
              <div className="relative overflow-hidden rounded bg-black">
                {selected.running ? (
                  <img
                    key={`${selected.camera_id}-${streamNonce}`}
                    src={streamUrl(selected.camera_id, 12, streamNonce)}
                    alt={`${selected.name} live feed`}
                    className="block max-h-[62vh] w-full object-contain"
                  />
                ) : (
                  <div className="flex aspect-video items-center justify-center gap-2 text-[13px] text-text-tertiary">
                    <WifiOff size={16} /> Camera stopped
                  </div>
                )}
                {selected.running && (
                  <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1.5 rounded bg-black/65 px-2 py-1 text-[11px] font-semibold tracking-wide text-white">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-accent-red" /> LIVE
                  </div>
                )}
              </div>
              <div className="grid grid-cols-4 gap-2 text-center max-[700px]:grid-cols-2">
                {[
                  { label: "Video rate", value: `${selected.measured_fps.toFixed(1)} fps` },
                  { label: "Analysis rate", value: selected.analysis_fps != null ? `${selected.analysis_fps.toFixed(1)} fps` : "off" },
                  { label: "Frames read", value: selected.frames_read.toLocaleString() },
                  { label: "Reconnects", value: String(selected.reconnect_count) },
                ].map((stat) => (
                  <div key={stat.label} className="rounded border border-border-1 bg-bg-1 px-2 py-2">
                    <div className="font-mono text-[14px] font-semibold text-text-primary">{stat.value}</div>
                    <div className="mt-0.5 text-[10.5px] text-text-tertiary">{stat.label}</div>
                  </div>
                ))}
              </div>
              <TalkDownControls camera={selected} />
              {selected.analysis_error && (
                <div className="rounded border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">
                  Analysis error: {selected.analysis_error}
                </div>
              )}
              {selected.modules.length === 0 && (
                <div className="rounded border border-border-1 bg-bg-1 px-3 py-2 text-[12px] text-text-tertiary">
                  No analysis modules enabled on this camera — showing raw video only.
                </div>
              )}
            </div>
          ) : (
            <div className="flex h-[40vh] items-center justify-center text-[13px] text-text-tertiary">
              Select a camera
            </div>
          )}
        </Panel>

        <div className="flex flex-col gap-4 max-[1500px]:col-span-full max-[1500px]:grid max-[1500px]:grid-cols-2 max-[1100px]:grid-cols-1">
          <Panel title="Active alerts" meta={`${alarms.length}`} tight>
            <div className="max-h-[36vh] overflow-y-auto px-2">
              {alarms.length === 0 ? (
                <div className="px-1 py-6 text-center text-[12px] text-text-tertiary">
                  No alerts requiring attention.
                </div>
              ) : (
                alarms.slice(0, 20).map((event, i) => (
                  <EventRow key={event.event_id ?? `${i}-${event.wall_time}`} event={event} />
                ))
              )}
            </div>
          </Panel>

          <Panel title="Evidence" tight>
            <div className="max-h-[38vh] overflow-y-auto px-1 py-1">
              <EvidencePanel />
            </div>
          </Panel>

          <Panel
            title="Event feed"
            meta={
              <span className="flex items-center gap-1">
                {connected ? <Wifi size={11} /> : <WifiOff size={11} />} {events.length}
              </span>
            }
            tight
          >
            <div className="max-h-[42vh] overflow-y-auto px-2">
              {events.length === 0 ? (
                <div className="px-1 py-6 text-center text-[12px] text-text-tertiary">
                  Waiting for events…
                </div>
              ) : (
                events.slice(0, 60).map((event, i) => (
                  <EventRow key={event.event_id ?? `${i}-${event.wall_time}`} event={event} />
                ))
              )}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
