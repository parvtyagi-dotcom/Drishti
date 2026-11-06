import { Loader2, X } from "lucide-react";
import type { JobStatusOut } from "../../api/types";
import { formatDuration, formatPercent } from "../../lib/format";
import { JobStatusBadge } from "../common/StatusBadge";

interface ProgressPanelProps {
  job: JobStatusOut;
  onCancel?: () => void;
  cancelling?: boolean;
  onView?: () => void;
}

/** Live progress of one forensic analysis job: status, bar, throughput, ETA. */
export function ProgressPanel({ job, onCancel, cancelling = false, onView }: ProgressPanelProps) {
  const active = job.status === "running" || job.status === "pending";
  const progress = Math.max(0, Math.min(1, job.progress ?? 0));
  const frames =
    job.frames_processed != null
      ? `${job.frames_processed.toLocaleString()}${job.total_frames ? ` / ${job.total_frames.toLocaleString()}` : ""} frames`
      : "—";

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <JobStatusBadge status={job.status} />
          <span className="truncate font-mono text-[11.5px] text-text-tertiary">{job.job_id.slice(0, 8)}</span>
        </div>
        <span className="text-[13px] font-semibold tabular-nums text-text-primary">{formatPercent(progress)}</span>
      </div>

      <div
        className="h-2 w-full overflow-hidden rounded-full bg-bg-3"
        role="progressbar"
        aria-label="Analysis progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(progress * 100)}
      >
        <div
          className={`h-full rounded-full transition-[width] duration-500 ${
            job.status === "failed" ? "bg-accent-red" : job.status === "done" ? "bg-accent-green" : "bg-accent-blue"
          }`}
          style={{ width: `${progress * 100}%` }}
        />
      </div>

      <dl className="grid grid-cols-4 gap-2 text-[11.5px] max-[640px]:grid-cols-2">
        <Stat label="Frames" value={frames} />
        <Stat label="Speed" value={job.processing_fps != null ? `${job.processing_fps.toFixed(1)} fps` : "—"} />
        <Stat label="Elapsed" value={formatDuration(job.elapsed_sec)} />
        <Stat label="Remaining" value={active ? formatDuration(job.eta_sec) : "—"} />
      </dl>

      {job.error && (
        <div className="rounded-md border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">
          {job.error}
        </div>
      )}

      {(onCancel || onView) && (
        <div className="flex justify-end gap-2">
          {onView && (
            <button
              type="button"
              onClick={onView}
              className="rounded-lg border border-border-2 px-3 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3 hover:text-text-primary"
            >
              View
            </button>
          )}
          {onCancel && active && (
            <button
              type="button"
              onClick={onCancel}
              disabled={cancelling}
              className="inline-flex items-center gap-1.5 rounded-lg border border-accent-red/35 px-3 py-1.5 text-[12px] text-accent-red hover:bg-accent-red/10 disabled:opacity-50"
            >
              {cancelling ? <Loader2 size={13} className="animate-spin" /> : <X size={13} />}
              {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border-1 bg-bg-2 px-2.5 py-2">
      <dt className="text-text-tertiary">{label}</dt>
      <dd className="mt-0.5 font-medium tabular-nums text-text-primary">{value}</dd>
    </div>
  );
}
