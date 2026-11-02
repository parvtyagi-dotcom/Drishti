import { useState } from "react";
import { AlertTriangle, PlugZap } from "lucide-react";
import { api, ApiError } from "../../api/client";
import type { LiveCamera } from "../../api/live";
import { StatusBadge } from "../common/StatusBadge";

const STATE_SEVERITY: Record<string, "green" | "amber" | "red" | "neutral"> = {
  online: "green",
  connecting: "amber",
  reconnecting: "amber",
  offline: "red",
  stopped: "neutral",
};

function healthOf(camera: LiveCamera): { label: string; severity: "green" | "amber" | "red" | "neutral" } {
  if (camera.degraded) return { label: "DEGRADED", severity: "amber" };
  return { label: camera.state.toUpperCase(), severity: STATE_SEVERITY[camera.state] ?? "neutral" };
}

/**
 * Fleet health at a glance.
 *
 * Shows frame age alongside state, because the dangerous failure is not a
 * camera that disconnects - that is obvious - but one that stays connected
 * while its picture stops updating. Frame age is what separates the two.
 */
export function CameraHealthPanel({
  cameras, onChanged,
}: { cameras: LiveCamera[]; onChanged: () => void }) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function simulateFault(camera: LiveCamera) {
    setBusyId(camera.camera_id);
    setError(null);
    try {
      await api.post(`/api/live/cameras/${camera.camera_id}/simulate-fault?seconds=20`);
      onChanged();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not sever the feed");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      {error && (
        <div className="flex items-center gap-2 rounded border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] border-collapse text-[12px]">
          <thead>
            <tr className="border-b border-border-1 text-left text-[10.5px] uppercase tracking-wide text-text-tertiary">
              <th className="py-2 pr-3 font-semibold">Camera</th>
              <th className="py-2 pr-3 font-semibold">Health</th>
              <th className="py-2 pr-3 text-right font-semibold">Rate</th>
              <th className="py-2 pr-3 text-right font-semibold">Frame age</th>
              <th className="py-2 pr-3 text-right font-semibold">Reconnects</th>
              <th className="py-2 pr-3 text-right font-semibold">Uptime</th>
              <th className="py-2 font-semibold" />
            </tr>
          </thead>
          <tbody>
            {cameras.map((camera) => {
              const health = healthOf(camera);
              const stale = camera.frame_age_sec != null && camera.frame_age_sec > 3;
              return (
                <tr key={camera.camera_id} className="border-b border-border-1 last:border-0">
                  <td className="py-2.5 pr-3">
                    <div className="font-semibold text-text-primary">{camera.name}</div>
                    <div className="text-[10.5px] text-text-tertiary">
                      {camera.location ?? "—"} · {camera.source_kind}
                    </div>
                  </td>
                  <td className="py-2.5 pr-3">
                    <StatusBadge
                      label={health.label}
                      severity={health.severity}
                      pulse={camera.state === "online" && !camera.degraded}
                    />
                    {camera.error && (
                      <div className="mt-1 max-w-[220px] text-[10.5px] text-accent-red">{camera.error}</div>
                    )}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono text-text-secondary">
                    {camera.measured_fps.toFixed(1)}
                  </td>
                  <td className={`py-2.5 pr-3 text-right font-mono ${stale ? "text-accent-red" : "text-text-secondary"}`}>
                    {camera.frame_age_sec == null ? "—" : `${camera.frame_age_sec.toFixed(1)}s`}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono text-text-secondary">
                    {camera.reconnect_count}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono text-text-secondary">
                    {camera.uptime_sec == null ? "—" : `${Math.round(camera.uptime_sec)}s`}
                  </td>
                  <td className="py-2.5 text-right">
                    <button
                      onClick={() => void simulateFault(camera)}
                      disabled={busyId === camera.camera_id || !camera.running}
                      title="Genuinely sever this feed for 20s to rehearse recovery"
                      className="inline-flex items-center gap-1.5 rounded border border-border-2 px-2 py-1 text-[11px] text-text-secondary hover:bg-bg-3 hover:text-accent-amber disabled:opacity-35"
                    >
                      <PlugZap size={12} /> Sever feed
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] leading-relaxed text-text-tertiary">
        <b>Frame age</b> is the time since the last decoded frame. A camera can stay connected while
        its picture freezes, so age is the reliable signal — not connection state. Severing a feed
        really does drop it; the camera then recovers on its own.
      </p>
    </div>
  );
}
