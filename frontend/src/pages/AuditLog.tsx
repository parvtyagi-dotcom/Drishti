import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, FileText, Loader2, RefreshCw, ShieldAlert, ShieldCheck } from "lucide-react";
import { Panel } from "../components/common/Panel";
import { auditApi, type AuditVerification } from "../api/auth";
import { eventMeta } from "../lib/eventMeta";

interface AuditRow {
  row_id?: number;
  event_id?: string;
  type?: string;
  severity?: string;
  reason?: string;
  camera_name?: string;
  wall_time?: number;
  alarm?: boolean;
  hash?: string;
}

export function AuditLog() {
  const [rows, setRows] = useState<AuditRow[]>([]);
  const [verification, setVerification] = useState<AuditVerification | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [alarmsOnly, setAlarmsOnly] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setRows((await auditApi.log(200, alarmsOnly)) as AuditRow[]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the audit log");
    }
  }, [alarmsOnly]);

  useEffect(() => {
    void load();
  }, [load]);

  async function verify() {
    setVerifying(true);
    try {
      setVerification(await auditApi.verify());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setVerifying(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-text-primary">Audit Log</h1>
          <p className="mt-1 text-[12.5px] text-text-tertiary">
            The durable record of every event. Each entry is chained to the one before it, so any
            later alteration is detectable.
          </p>
        </div>
        {/* A printable page; the browser's Print turns it into a PDF. */}
        <a
          href="/api/live/handover?hours=12"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 rounded border border-border-2 bg-bg-2 px-3 py-2 text-[12px] font-semibold text-text-secondary hover:bg-bg-3 hover:text-text-primary"
        >
          <FileText size={13} /> Shift handover (last 12 h)
        </a>
      </div>

      {error && (
        <div className="rounded border border-accent-red/35 bg-accent-red/10 px-4 py-3 text-[12.5px] text-accent-red">
          {error}
        </div>
      )}

      <Panel
        title="Integrity"
        actions={
          <button
            onClick={() => void verify()}
            disabled={verifying}
            className="flex items-center gap-1.5 rounded border border-border-2 px-2.5 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3 hover:text-text-primary disabled:opacity-40"
          >
            {verifying ? <Loader2 size={13} className="animate-spin" /> : <ShieldCheck size={13} />}
            Verify chain
          </button>
        }
      >
        {!verification ? (
          <div className="text-[12.5px] text-text-tertiary">
            Run a verification to recompute every hash and confirm the record is unaltered.
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div
              className={`flex items-start gap-2.5 rounded border px-3 py-2.5 text-[12.5px] ${
                verification.valid
                  ? "border-accent-green/35 bg-accent-green/10 text-accent-green"
                  : "border-accent-red/40 bg-accent-red/10 text-accent-red"
              }`}
            >
              {verification.valid ? (
                <CheckCircle2 size={16} className="mt-[1px] shrink-0" />
              ) : (
                <ShieldAlert size={16} className="mt-[1px] shrink-0" />
              )}
              <span>{verification.summary}</span>
            </div>

            {/* Stated plainly: an integrity claim is only useful if its limits
                are visible alongside it. */}
            <div className="grid grid-cols-2 gap-3 max-[760px]:grid-cols-1">
              <div className="rounded border border-border-1 bg-bg-1 p-3">
                <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wide text-text-secondary">
                  Detects
                </div>
                <ul className="flex flex-col gap-1 text-[11.5px] text-text-tertiary">
                  {verification.guarantees.detects.map((item) => (
                    <li key={item}>· {item}</li>
                  ))}
                </ul>
              </div>
              <div className="rounded border border-border-1 bg-bg-1 p-3">
                <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wide text-text-secondary">
                  Does not detect
                </div>
                <ul className="flex flex-col gap-1 text-[11.5px] text-text-tertiary">
                  {verification.guarantees.does_not_detect.map((item) => (
                    <li key={item}>· {item}</li>
                  ))}
                </ul>
                <div className="mt-2 border-t border-border-1 pt-2 text-[11px] leading-relaxed text-text-tertiary">
                  {verification.guarantees.note}
                </div>
              </div>
            </div>
          </div>
        )}
      </Panel>

      <Panel
        title="Stored events"
        meta={`${rows.length}`}
        actions={
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-[11.5px] text-text-secondary">
              <input
                type="checkbox"
                checked={alarmsOnly}
                onChange={(e) => setAlarmsOnly(e.target.checked)}
              />
              Alarms only
            </label>
            <button
              onClick={() => void load()}
              className="rounded border border-border-2 p-1.5 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        }
        tight
      >
        <div className="max-h-[55vh] overflow-auto">
          <table className="w-full min-w-[720px] border-collapse text-[12px]">
            <thead className="sticky top-0 bg-bg-2">
              <tr className="border-b border-border-1 text-left text-[10.5px] uppercase tracking-wide text-text-tertiary">
                <th className="py-2 pr-3 font-semibold">#</th>
                <th className="py-2 pr-3 font-semibold">Time</th>
                <th className="py-2 pr-3 font-semibold">Event</th>
                <th className="py-2 pr-3 font-semibold">Reason</th>
                <th className="py-2 pr-3 font-semibold">Camera</th>
                <th className="py-2 font-semibold">Hash</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-text-tertiary">
                    No stored events yet.
                  </td>
                </tr>
              )}
              {rows.map((row) => (
                <tr key={row.event_id ?? row.row_id} className="border-b border-border-1 last:border-0">
                  <td className="py-2 pr-3 font-mono text-text-tertiary">{row.row_id}</td>
                  <td className="py-2 pr-3 font-mono text-text-tertiary">
                    {row.wall_time
                      ? new Date(row.wall_time * 1000).toLocaleTimeString([], { hour12: false })
                      : "—"}
                  </td>
                  <td className="py-2 pr-3">
                    <span className="font-semibold text-text-primary">
                      {eventMeta(row.type ?? "").label}
                    </span>
                    {row.alarm && (
                      <span className="ml-1.5 rounded bg-accent-red/15 px-1 text-[9.5px] font-bold text-accent-red">
                        SIREN
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-3 text-text-secondary">{row.reason}</td>
                  <td className="py-2 pr-3 text-text-tertiary">{row.camera_name}</td>
                  <td className="py-2 font-mono text-[10px] text-text-tertiary">
                    {row.hash?.slice(0, 12)}…
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
