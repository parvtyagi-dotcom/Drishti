import { useCallback, useEffect, useState } from "react";
import { Copy, Loader2, Plus, QrCode, RadioTower, Trash2, Zap } from "lucide-react";
import QRCode from "qrcode";
import { Panel } from "../components/common/Panel";
import type { SessionUser } from "../api/auth";
import { api, ApiError } from "../api/client";
import { liveApi, type LiveCamera, type LiveEvent } from "../api/live";
import { SENSOR_EVENT_TYPES, sensorsApi, type Sensor, type SensorKind } from "../api/sensors";
import { eventMeta } from "../lib/eventMeta";

const REFRESH_MS = 3000;

const SEVERITY_TEXT: Record<string, string> = {
  red: "text-accent-red",
  amber: "text-accent-amber",
  green: "text-accent-green",
  blue: "text-accent-blue",
  neutral: "text-text-tertiary",
};

function clock(seconds: number) {
  return new Date(seconds * 1000).toLocaleTimeString([], { hour12: false });
}

function PhoneLink({ sensor, base }: { sensor: Sensor; base: string | null }) {
  const [dataUrl, setDataUrl] = useState<string | null>(null);
  const url = base && sensor.key
    ? `${base}/api/sensors/hook/page#id=${sensor.sensor_id}&key=${encodeURIComponent(sensor.key)}`
    : null;
  useEffect(() => {
    if (!url) return;
    void QRCode.toDataURL(url, { width: 360, margin: 1 }).then(setDataUrl).catch(() => setDataUrl(null));
  }, [url]);
  if (!url) {
    return <p className="text-[12px] text-accent-amber">The server has no https address on the network, so a phone cannot use its microphone.</p>;
  }
  const curl = `curl -X POST ${base}/api/sensors/hook/${sensor.sensor_id} -H "X-Sensor-Key: ${sensor.key}"`;
  return (
    <div className="mt-3 grid gap-4 rounded-lg border border-border-1 bg-bg-1 p-4 md:grid-cols-[200px_1fr]">
      <div className="rounded bg-white p-2">{dataUrl && <img src={dataUrl} alt="Sensor page QR code" className="w-full" />}</div>
      <div className="space-y-3 text-[12px] text-text-tertiary">
        <p>
          <b className="text-text-secondary">Phone as a sensor:</b> scan this on a phone at the post. For a drone sensor
          the phone listens for a steady propeller hum; any sensor page also has a button to report a trip by hand.
        </p>
        <p>
          <b className="text-text-secondary">Real sensor hardware:</b> its controller sends one web request per trip. Keep
          the key secret: it lets this one sensor raise alarms.
        </p>
        <div className="flex items-center gap-2">
          <code className="min-w-0 flex-1 truncate rounded border border-border-1 bg-bg-2 px-2 py-1.5 font-mono text-[10.5px]">{curl}</code>
          <button
            onClick={() => void navigator.clipboard?.writeText(curl)}
            className="rounded border border-border-1 p-1.5 text-text-tertiary hover:text-text-primary"
            title="Copy the command"
          >
            <Copy size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}

export function Sensors({ user }: { user?: SessionUser | null }) {
  const isAdmin = user?.role === "ADMIN";
  const canTest = isAdmin || user?.role === "OPERATOR";
  const [sensors, setSensors] = useState<Sensor[]>([]);
  const [kinds, setKinds] = useState<SensorKind[]>([]);
  const [cameras, setCameras] = useState<LiveCamera[]>([]);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [base, setBase] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [showLink, setShowLink] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", kind: "fence", camera: "" });

  const refresh = useCallback(async () => {
    try {
      const [s, c, e] = await Promise.all([sensorsApi.list(), liveApi.listCameras(), liveApi.recentEvents(300)]);
      setSensors(s.sensors);
      setKinds(s.kinds);
      setCameras(c);
      setEvents(e.filter((ev) => SENSOR_EVENT_TYPES.has(ev.type)).reverse());
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load sensors.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!isAdmin) return;
    api.get<{ secure_base_url: string | null }>("/api/live/connect-info")
      .then((info) => setBase(info.secure_base_url))
      .catch(() => setBase(null));
  }, [isAdmin]);

  async function add() {
    if (!form.name.trim()) return;
    setBusy("add");
    try {
      await sensorsApi.add(form.name.trim(), form.kind, form.camera || null);
      setForm({ ...form, name: "" });
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not add the sensor.");
    } finally {
      setBusy(null);
    }
  }

  async function test(sensor: Sensor) {
    setBusy(sensor.sensor_id);
    try {
      await sensorsApi.test(sensor.sensor_id);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Test trip failed.");
    } finally {
      setBusy(null);
    }
  }

  async function remove(sensor: Sensor) {
    if (!window.confirm(`Remove ${sensor.name}? A device still using its key will stop being accepted.`)) return;
    try {
      await sensorsApi.remove(sensor.sensor_id);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not remove the sensor.");
    }
  }

  const lastFor = (id: string) => events.find((e) => (e.data as { sensor_id?: string } | undefined)?.sensor_id === id
    && e.type !== "SENSOR_TRIGGERED");

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-[20px] font-bold text-text-primary">Sensors</h1>
        <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-text-tertiary">
          Fence, ground and radar sensors trip on animals and wind. When one trips, Drishti checks the linked camera's
          video from 4 seconds before to 6 seconds after. If it sees a person or vehicle, the siren sounds. If the camera
          is down or blocked, the siren sounds too, because an unchecked alarm is never dismissed. If the camera saw nothing,
          the trip is logged quietly and raised again if it keeps happening. A camera cannot confirm a drone, so an acoustic
          drone alert always sounds.
        </p>
      </div>

      {error && <div className="rounded border border-accent-red/40 bg-accent-red/10 px-4 py-2 text-[12.5px] text-accent-red">{error}</div>}

      <Panel title="Post sensors" meta={`${sensors.length} configured`}>
        {sensors.length === 0 && <p className="text-[13px] text-text-tertiary">No sensors yet.{isAdmin ? " Add one below." : " An admin can add them."}</p>}
        <div className="divide-y divide-border-1">
          {sensors.map((s) => {
            const last = lastFor(s.sensor_id);
            const meta = last ? eventMeta(last.type) : null;
            return (
              <div key={s.sensor_id} className="py-3">
                <div className="flex flex-wrap items-center gap-3">
                  <RadioTower size={16} className="text-accent-blue" />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13.5px] font-semibold text-text-primary">{s.name}</div>
                    <div className="text-[12px] text-text-tertiary">
                      {s.label} · {s.checkable ? (s.camera ? `checked by ${s.camera}` : "no camera linked") : "sounds without a camera check"}
                    </div>
                    {last && meta && (
                      <div className={`mt-1 text-[12px] ${SEVERITY_TEXT[meta.severity]}`}>
                        {clock(last.wall_time)} · {meta.label}: {last.reason}
                      </div>
                    )}
                  </div>
                  {canTest && (
                    <button
                      onClick={() => void test(s)}
                      disabled={busy === s.sensor_id}
                      className="flex items-center gap-1.5 rounded border border-border-1 px-3 py-1.5 text-[12px] text-text-secondary hover:text-text-primary disabled:opacity-50"
                    >
                      {busy === s.sensor_id ? <Loader2 size={13} className="animate-spin" /> : <Zap size={13} />} Test trip
                    </button>
                  )}
                  {isAdmin && (
                    <>
                      <button
                        onClick={() => setShowLink(showLink === s.sensor_id ? null : s.sensor_id)}
                        className="flex items-center gap-1.5 rounded border border-border-1 px-3 py-1.5 text-[12px] text-text-secondary hover:text-text-primary"
                      >
                        <QrCode size={13} /> Connect
                      </button>
                      <button onClick={() => void remove(s)} className="rounded border border-border-1 p-1.5 text-text-tertiary hover:text-accent-red" title="Remove">
                        <Trash2 size={13} />
                      </button>
                    </>
                  )}
                </div>
                {showLink === s.sensor_id && <PhoneLink sensor={s} base={base} />}
              </div>
            );
          })}
        </div>

        {isAdmin && (
          <div className="mt-4 flex flex-wrap items-end gap-3 border-t border-border-1 pt-4">
            <label className="text-[12px] text-text-tertiary">
              Name
              <input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="Fence, east stretch"
                className="mt-1 block w-56 rounded border border-border-1 bg-bg-1 px-2 py-1.5 text-[13px] text-text-primary"
              />
            </label>
            <label className="text-[12px] text-text-tertiary">
              Kind
              <select
                value={form.kind}
                onChange={(e) => setForm({ ...form, kind: e.target.value })}
                className="mt-1 block rounded border border-border-1 bg-bg-1 px-2 py-1.5 text-[13px] text-text-primary"
              >
                {kinds.map((k) => <option key={k.kind} value={k.kind}>{k.label}</option>)}
              </select>
            </label>
            <label className="text-[12px] text-text-tertiary">
              Camera that checks it
              <select
                value={form.camera}
                onChange={(e) => setForm({ ...form, camera: e.target.value })}
                className="mt-1 block rounded border border-border-1 bg-bg-1 px-2 py-1.5 text-[13px] text-text-primary"
              >
                <option value="">None</option>
                {cameras.map((c) => <option key={c.camera_id} value={c.name}>{c.name}</option>)}
              </select>
            </label>
            <button
              onClick={() => void add()}
              disabled={busy === "add" || !form.name.trim()}
              className="flex items-center gap-1.5 rounded bg-accent-blue px-4 py-2 text-[12.5px] font-semibold text-white disabled:opacity-50"
            >
              <Plus size={14} /> Add sensor
            </button>
          </div>
        )}
      </Panel>

      <Panel title="Sensor alarms" meta="newest first">
        {events.length === 0 && <p className="text-[13px] text-text-tertiary">No sensor activity yet.</p>}
        <div className="divide-y divide-border-1">
          {events.slice(0, 30).map((e) => {
            const meta = eventMeta(e.type);
            const Icon = meta.icon;
            return (
              <div key={e.event_id} className="flex items-start gap-3 py-2.5 text-[12.5px]">
                <span className="w-16 shrink-0 font-mono text-text-tertiary">{clock(e.wall_time)}</span>
                <Icon size={14} className={`mt-0.5 shrink-0 ${SEVERITY_TEXT[meta.severity]}`} />
                <div className="min-w-0">
                  <span className={`font-semibold ${SEVERITY_TEXT[meta.severity]}`}>{meta.label}</span>
                  <span className="text-text-secondary"> · {e.reason}</span>
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
