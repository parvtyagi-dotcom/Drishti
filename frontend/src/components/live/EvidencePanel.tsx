import { useCallback, useEffect, useState } from "react";
import { Clapperboard, Clock, ImageIcon, VideoOff, X } from "lucide-react";
import { evidenceAssetUrl, liveApi, type EvidenceBundle } from "../../api/live";
import { eventMeta } from "../../lib/eventMeta";

const REFRESH_MS = 5000;

/**
 * Evidence captured for alerts.
 *
 * A clip appears a few seconds after its snapshot, because the footage
 * following an event has to be recorded before it can be assembled. The panel
 * says "clip pending" during that window rather than showing a broken player,
 * so a missing clip never looks like a lost recording.
 *
 * Some alerts can never have a clip at all - a camera going down has no
 * footage of itself - and those are labelled "no footage" instead, because a
 * bundle stuck on "pending" forever reads as a system that lost the recording.
 */
export function EvidencePanel() {
  const [bundles, setBundles] = useState<EvidenceBundle[]>([]);
  const [open, setOpen] = useState<EvidenceBundle | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setBundles(await liveApi.listEvidence(24));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load evidence");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  if (error) {
    return <div className="px-1 py-4 text-[12px] text-accent-red">{error}</div>;
  }

  if (bundles.length === 0) {
    return (
      <div className="px-1 py-6 text-center text-[12px] text-text-tertiary">
        No evidence yet. Bundles are captured automatically when an alert is raised.
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-3 gap-2 max-[900px]:grid-cols-2">
        {bundles.map((bundle) => {
          const meta = eventMeta(bundle.event_type ?? "");
          return (
            <button
              key={`${bundle.camera_id}-${bundle.bundle_id}`}
              onClick={() => setOpen(bundle)}
              className="group overflow-hidden rounded border border-border-1 bg-bg-1 text-left transition-colors hover:border-accent-blue"
            >
              <div className="relative aspect-video bg-black">
                {bundle.has_snapshot ? (
                  <img
                    src={evidenceAssetUrl(bundle, "snapshot.jpg")}
                    alt={bundle.reason ?? "Evidence snapshot"}
                    loading="lazy"
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-text-tertiary">
                    <ImageIcon size={18} />
                  </div>
                )}
                <span
                  className={`absolute left-1.5 top-1.5 rounded px-1.5 py-[1px] text-[9.5px] font-bold ${
                    bundle.severity === "CRITICAL"
                      ? "bg-accent-red text-white"
                      : "bg-black/70 text-white"
                  }`}
                >
                  {bundle.severity ?? "EVENT"}
                </span>
                <span className="absolute bottom-1.5 right-1.5 flex items-center gap-1 rounded bg-black/70 px-1.5 py-[1px] text-[9.5px] text-white">
                  {bundle.clip_state === "ready" ? (
                    <><Clapperboard size={9} /> clip</>
                  ) : bundle.clip_state === "unavailable" ? (
                    <span className="flex items-center gap-1 text-text-tertiary">
                      <VideoOff size={9} /> no footage
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-text-tertiary">
                      <Clock size={9} /> clip pending
                    </span>
                  )}
                </span>
              </div>
              <div className="p-2">
                <div className="truncate text-[11.5px] font-semibold text-text-primary">{meta.label}</div>
                <div className="truncate text-[10.5px] text-text-tertiary">
                  {bundle.camera_name ?? bundle.camera_id} ·{" "}
                  {new Date(bundle.created_at * 1000).toLocaleTimeString([], { hour12: false })}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
          onClick={() => setOpen(null)}
        >
          <div
            className="max-h-full w-full max-w-3xl overflow-y-auto rounded-lg border border-border-2 bg-bg-2"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3 border-b border-border-1 px-4 py-3">
              <div className="min-w-0">
                <div className="text-[13.5px] font-bold text-text-primary">
                  {eventMeta(open.event_type ?? "").label}
                </div>
                <div className="text-[12px] text-text-secondary">{open.reason}</div>
                <div className="mt-0.5 text-[11px] text-text-tertiary">
                  {open.camera_name ?? open.camera_id} ·{" "}
                  {new Date(open.created_at * 1000).toLocaleString()}
                </div>
              </div>
              <button
                onClick={() => setOpen(null)}
                className="shrink-0 rounded border border-border-2 p-1.5 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
              >
                <X size={14} />
              </button>
            </div>

            <div className="p-4">
              {open.has_clip ? (
                <video
                  key={open.bundle_id}
                  src={evidenceAssetUrl(open, "clip.mp4")}
                  controls
                  autoPlay
                  className="w-full rounded bg-black"
                />
              ) : (
                <div className="flex flex-col gap-3">
                  {open.has_snapshot && (
                    <img
                      src={evidenceAssetUrl(open, "snapshot.jpg")}
                      alt="Evidence snapshot"
                      className="w-full rounded bg-black"
                    />
                  )}
                  <div className="rounded border border-border-1 bg-bg-1 px-3 py-2 text-[12px] text-text-tertiary">
                    {open.clip_state === "unavailable"
                      ? "No footage was recorded for this alert. The feed had already stopped when it was raised, so there is nothing to replay — the snapshot and event record are the evidence."
                      : "The clip is still being assembled — footage after the event has to be recorded before it can be written."}
                  </div>
                </div>
              )}
              <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
                <a
                  href={evidenceAssetUrl(open, "snapshot.jpg")}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded border border-border-2 px-2.5 py-1.5 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
                >
                  Open snapshot
                </a>
                <a
                  href={`/api/live/evidence/${open.camera_id}/${open.bundle_id}/event.json`}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded border border-border-2 px-2.5 py-1.5 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
                >
                  Event record (JSON)
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
