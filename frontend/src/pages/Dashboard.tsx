import { ChevronRight, Radio, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { cancelJob } from "../api/jobs";
import { OpsMap } from "../components/dashboard/OpsMap";
import { PageBody, PageHeader } from "../components/layout/AppLayout";
import { Panel } from "../components/common/Panel";
import { ErrorState } from "../components/common/States";
import { JobStatusBadge, StatusBadge } from "../components/common/StatusBadge";
import { ProgressPanel } from "../components/jobs/ProgressPanel";
import { useDashboardData } from "../hooks/useDashboardData";
import { hourly, threatIndex, useOpsPicture } from "../hooks/useOpsPicture";
import { useSiteChoice } from "../hooks/useSiteChoice";
import { eventMeta } from "../lib/eventMeta";

function useNow(intervalMs = 5000) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now() / 1000), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

function ago(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

const SEVERITY_DOT: Record<string, string> = {
  CRITICAL: "bg-accent-red",
  HIGH: "bg-accent-orange",
  MEDIUM: "bg-accent-amber",
  LOW: "bg-accent-green",
  INFO: "bg-text-tertiary",
};

const isAlarm = (e: { alarm: boolean; severity: string }) => e.alarm || e.severity === "CRITICAL" || e.severity === "HIGH";
const isPlate = (e: { type: string }) => e.type.startsWith("PLATE_") || e.type.startsWith("TARGET_VEHICLE");

export function Dashboard() {
  const { error, jobs, metrics, activeJob, refetch } = useDashboardData();
  const [siteChoice, chooseSite] = useSiteChoice();
  const { cameras, events, map, connected } = useOpsPicture(siteChoice, () => chooseSite(null));
  const navigate = useNavigate();
  const [cancelling, setCancelling] = useState(false);
  const now = useNow();

  const online = cameras.filter((c) => c.state === "online").length;
  const degraded = cameras.filter((c) => c.state !== "online" || c.degraded).length;
  const day = useMemo(() => events.filter((e) => now - e.wall_time < 86400), [events, now]);
  const alarms24 = day.filter(isAlarm).length;
  const plates24 = day.filter(isPlate).length;
  const index = useMemo(() => threatIndex(events, now), [events, now]);
  const perHour = useMemo(() => hourly(events, now), [events, now]);
  const alarmsPerHour = useMemo(() => hourly(events, now, isAlarm), [events, now]);
  const platesPerHour = useMemo(() => hourly(events, now, isPlate), [events, now]);
  const threatLabel = index >= 60 ? "High" : index >= 25 ? "Elevated" : "Low";
  const threatDot = index >= 60 ? "red" : index >= 25 ? "orange" : "green";
  const site = (map && map.sites.length > 1 ? map.site_name : null) ?? cameras.find((c) => c.location)?.location;

  async function handleCancelActive() {
    if (!activeJob) return;
    setCancelling(true);
    try {
      await cancelJob(activeJob.job_id);
      refetch();
    } finally {
      setCancelling(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Site overview"
        subtitle={site ? `Drishti / ${site}` : "Drishti"}
        actions={
          <>
            <StatusBadge label={connected ? "Feed live" : "Feed offline"} severity={connected ? "green" : "amber"} pulse={connected} />
            <button
              onClick={() => navigate("/live")}
              className="flex items-center gap-2 rounded-lg bg-text-primary px-4 py-2 text-[12.5px] font-semibold text-bg-0 hover:bg-white"
            >
              <Radio size={14} /> Live monitor
            </button>
          </>
        }
      />
      <PageBody>
        {error && <ErrorState message={error} onRetry={refetch} />}

        <div className="flex flex-col gap-3.5">
          {/* KPI strip */}
          <div className="grid grid-cols-5 gap-3 max-[1200px]:grid-cols-3 max-[700px]:grid-cols-2">
            <Kpi label="Cameras online" value={`${online}/${cameras.length}`} note={degraded ? `${degraded} degraded or offline` : "All healthy"} dot={degraded ? "amber" : "green"} />
            <Kpi label="Alarms, 24h" value={alarms24} note={alarms24 ? "Needs review" : "None raised"} dot={alarms24 ? "red" : "green"} />
            <Kpi label="Events, 24h" value={day.length} note={`${perHour[23]} in the last hour`} dot="blue" />
            <Kpi label="Plates read, 24h" value={plates24} note={`${metrics.platesRead.toLocaleString()} from forensic jobs`} dot="blue" />
            <Kpi label="Threat index" value={`${index} /100`} note={threatLabel} dot={threatDot} />
          </div>

          {/* Map and alerts fill the screen height left over on a desktop */}
          <div className="grid h-[calc(100vh-430px)] min-h-[420px] grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)] gap-3.5 max-[1100px]:h-auto max-[1100px]:grid-cols-1">
            <OpsMap map={map} cameras={cameras} events={events} now={now} onSiteChange={chooseSite} />

            <div className="surface flex min-h-[380px] flex-col overflow-hidden rounded-xl border border-border-1">
              <div className="flex items-center justify-between px-4 pb-2 pt-4">
                <h2 className="text-[14px] font-semibold text-text-primary">Real-time alerts</h2>
                <Link to="/events" className="text-[12px] text-text-tertiary hover:text-text-primary">View all</Link>
              </div>
              <div className="grid grid-cols-[48px_minmax(0,1fr)_minmax(0,128px)] border-b border-border-1 px-4 pb-2 text-[11px] text-text-tertiary">
                <span>Time</span>
                <span>Alert</span>
                <span>Camera</span>
              </div>
              {events.length === 0 ? (
                <div className="flex flex-1 flex-col items-center justify-center gap-2 text-[12.5px] text-text-tertiary">
                  <ShieldCheck size={20} className="text-accent-green" /> All quiet. No live events yet.
                </div>
              ) : (
                <div className="flex-1 overflow-y-auto">
                  {events.slice(0, 40).map((e, i) => (
                    <button
                      key={e.event_id ?? i}
                      onClick={() => navigate("/live")}
                      title={e.reason}
                      className="grid w-full grid-cols-[48px_minmax(0,1fr)_minmax(0,128px)] items-center border-b border-border-1/70 px-4 py-2.5 text-left text-[12.5px] hover:bg-bg-3/60"
                    >
                      <span className="font-mono text-[11.5px] text-text-tertiary">-{ago(now - e.wall_time)}</span>
                      <span className="flex min-w-0 items-center gap-2">
                        <span className={`h-[7px] w-[7px] shrink-0 rounded-full ${SEVERITY_DOT[e.severity] ?? "bg-text-tertiary"}`} />
                        <span className={`truncate text-text-primary ${e.alarm ? "font-semibold" : ""}`}>
                          {eventMeta(e.type).label}{e.plate ? ` · ${e.plate}` : ""}
                        </span>
                      </span>
                      <span className="truncate text-text-secondary">{e.camera_name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Trends */}
          <div className="grid h-[150px] grid-cols-4 gap-3.5 max-[1100px]:grid-cols-2 max-[1100px]:h-auto">
            <Trend label="Events per hour" value={day.length} kind="bars" data={perHour} color="var(--color-accent-red)" />
            <Trend label="Alarms per hour" value={alarms24} kind="area" data={alarmsPerHour} color="var(--color-accent-orange)" />
            <Trend label="Plate reads per hour" value={plates24} kind="area" data={platesPerHour} color="var(--color-accent-green)" />
            <Trend
              label="Camera frame rate"
              value={cameras.length ? `${Math.round(cameras.reduce((a, c) => a + c.measured_fps, 0) / cameras.length)} fps` : "—"}
              kind="bars"
              data={cameras.map((c) => c.measured_fps)}
              names={cameras.map((c) => c.name)}
              color="var(--color-accent-blue)"
            />
          </div>

          {/* Forensic work */}
          <div className="grid grid-cols-2 gap-3.5 max-[1000px]:grid-cols-1">
            <Panel title={activeJob ? "Active analysis" : "Forensic analysis"} meta={activeJob ? undefined : "Nothing processing"}>
              {activeJob ? (
                <ProgressPanel job={activeJob} onCancel={handleCancelActive} cancelling={cancelling} onView={() => navigate(`/results/${activeJob.job_id}`)} />
              ) : (
                <div className="flex items-center justify-between gap-3">
                  <div className="text-[12.5px] text-text-tertiary">
                    {metrics.videosAnalyzed.toLocaleString()} videos analysed · {metrics.targetsIdentified} targets · {metrics.vehiclesDetected.toLocaleString()} vehicles
                  </div>
                  <button onClick={() => navigate("/analysis")} className="shrink-0 rounded-lg border border-border-2 px-3 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3 hover:text-text-primary">
                    New analysis
                  </button>
                </div>
              )}
            </Panel>

            <Panel title="Recent analyses" meta={jobs.length ? `${jobs.length} total` : undefined} tight>
              {jobs.length === 0 ? (
                <div className="py-6 text-center text-[12.5px] text-text-tertiary">No analyses yet</div>
              ) : (
                <div className="flex flex-col">
                  {jobs.slice(0, 4).map((job) => (
                    <button
                      key={job.job_id}
                      onClick={() => navigate(job.status === "done" ? `/results/${job.job_id}` : `/processing/${job.job_id}`)}
                      className="flex items-center justify-between gap-3 rounded-lg px-2 py-2 text-left hover:bg-bg-3/60"
                    >
                      <div className="min-w-0">
                        <div className="font-mono text-[12px] text-text-primary">Job {job.job_id.slice(0, 8)}</div>
                        <div className="truncate text-[11px] text-text-tertiary">{job.summary?.modules_run?.join(" · ") || job.kind}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <JobStatusBadge status={job.status} />
                        <ChevronRight size={14} className="text-text-tertiary" />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
          </div>
        </div>
      </PageBody>
    </>
  );
}

const DOT = { red: "bg-accent-red", amber: "bg-accent-amber", green: "bg-accent-green", blue: "bg-accent-blue", orange: "bg-accent-orange" } as const;
const NOTE = { red: "text-accent-red", amber: "text-accent-amber", green: "text-accent-green", blue: "text-accent-blue", orange: "text-accent-orange" } as const;

function Kpi({ label, value, note, dot }: { label: string; value: string | number; note: string; dot: keyof typeof DOT }) {
  return (
    <div className="surface rounded-xl border border-border-1 px-4 py-3.5">
      <div className="flex items-center justify-between text-[12px] text-text-secondary">
        {label}
        <span className={`h-2 w-2 rounded-full ${DOT[dot]}`} />
      </div>
      <div className="mt-1.5 tabular-nums text-[26px] font-semibold leading-tight tracking-tight text-text-primary">
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      <div className={`mt-0.5 truncate text-[11.5px] ${NOTE[dot]}`}>{note}</div>
    </div>
  );
}

function Trend({
  label, value, kind, data, color, names,
}: {
  label: string;
  value: string | number;
  kind: "bars" | "area";
  data: number[];
  color: string;
  names?: string[];
}) {
  const max = Math.max(1, ...data);
  return (
    <div className="surface flex flex-col rounded-xl border border-border-1 px-4 py-3">
      <div className="text-[12px] text-text-secondary">{label}</div>
      <div className="mb-2 mt-0.5 tabular-nums text-[18px] font-semibold text-text-primary">
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      <div className="relative min-h-[56px] flex-1">
        {kind === "area" ? (
          <AreaChart data={data} max={max} color={color} />
        ) : (
          <div className="absolute inset-0 flex items-end gap-[3px]">
            {data.length === 0 && <div className="m-auto text-[11.5px] text-text-tertiary">No data yet</div>}
            {data.map((v, i) => (
              <div
                key={i}
                title={names ? `${names[i]}: ${v.toFixed(1)} fps` : `${v} events`}
                className="max-w-[48px] flex-1 rounded-t-[2px]"
                style={{ height: `${Math.max(3, (v / max) * 100)}%`, background: color, opacity: v ? 1 : 0.25 }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function AreaChart({ data, max, color }: { data: number[]; max: number; color: string }) {
  const w = 300;
  const h = 60;
  const step = w / (data.length - 1);
  const pts = data.map((v, i) => `${(i * step).toFixed(1)},${(h - (v / max) * (h - 6) - 2).toFixed(1)}`);
  const line = `M${pts.join(" L")}`;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
      <path d={`${line} L${w},${h} L0,${h} Z`} fill={color} opacity={0.14} />
      <path d={line} fill="none" stroke={color} strokeWidth={2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
