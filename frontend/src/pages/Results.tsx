import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { getJobEvidence, getJobEvents, jobOutputUrl } from "../api/jobs";
import type { BwEvent, EvidenceItem } from "../api/types";
import { titleCase } from "../lib/format";
import { PageBody, PageHeader } from "../components/layout/AppLayout";
import { Panel, SectionTitle } from "../components/common/Panel";
import { MetricCard } from "../components/common/MetricCard";
import { StatusBadge } from "../components/common/StatusBadge";
import { EmptyState, ErrorState, LoadingBlock } from "../components/common/States";
import { VideoPlayer, type VideoPlayerHandle } from "../components/video/VideoPlayer";
import { EventTimeline } from "../components/events/EventTimeline";
import { EvidenceGrid } from "../components/evidence/EvidenceGrid";
import { Modal } from "../components/common/Modal";
import { jobEvidenceFileUrl } from "../api/jobs";
import { TargetCard } from "../components/target/TargetCard";
import { PlateResultCard } from "../components/anpr/PlateResultCard";
import { PlatesTable } from "../components/anpr/PlatesTable";
import { ZoneResultsTable } from "../components/zone/ZoneResultsTable";
import { BehaviorResultsList } from "../components/behavior/BehaviorResultsList";
import { LowlightSummaryCard } from "../components/common/LowlightSummaryCard";
import { deriveZoneIntrusions } from "../lib/deriveZoneIntrusions";
import { useJobPolling } from "../hooks/useJobPolling";
import { Radar, Siren, VolumeX } from "lucide-react";
import { playSiren, vibrateAlert } from "../lib/siren";
import { alertsReachedBy, type PlaybackAlert } from "../lib/alertPlayback";

// Which findings are worth interrupting someone for is the backend's
// decision - it stamps `alarm` from the severity each rule earned. This page
// used to keep its own list of types instead, and the two drifted: a zone
// crossing became CRITICAL and sounded the siren on a live camera while the
// same crossing stayed silent here, because this list predated the change.
// Following the stamp means a new alarm-worthy event works everywhere at once.
function isAlertEvent(event: BwEvent): boolean {
  return event.alarm === true;
}

// How long the on-screen alert banner stays up after the siren sounds.
const ALERT_BANNER_MS = 5000;

/** Stable identity for an alert. Forensic events carry an event_id only
 *  sometimes, and type-plus-timestamp identifies a moment either way. */
function alertKey(event: BwEvent): string {
  return event.event_id ?? `${event.type}@${event.timestamp_sec}`;
}

export function Results() {
  const { jobId } = useParams<{ jobId: string }>();
  const { job, error: jobError } = useJobPolling(jobId);
  const [events, setEvents] = useState<BwEvent[] | null>(null);
  const [evidence, setEvidence] = useState<EvidenceItem[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<BwEvent | null>(null);
  const [evidenceModal, setEvidenceModal] = useState<EvidenceItem | null>(null);
  const playerRef = useRef<VideoPlayerHandle>(null);

  const [audioBlocked, setAudioBlocked] = useState(false);
  const [activeAlert, setActiveAlert] = useState<BwEvent | null>(null);
  // Alerts already sounded, so one pass of playback sounds each one once.
  const soundedRef = useRef<Set<string>>(new Set());
  const lastTimeRef = useRef(0);

  useEffect(() => {
    if (job?.status !== "done" || !jobId) return;
    Promise.all([getJobEvents(jobId), getJobEvidence(jobId)])
      .then(([ev, ec]) => {
        setEvents(ev);
        setEvidence(ec);
      })
      .catch((err) => setLoadError(err instanceof ApiError ? err.message : "Failed to load analysis results"));
  }, [job?.status, jobId]);

  // The banner marks a moment in the footage, so it clears itself rather than
  // lingering over a part of the video it no longer describes.
  useEffect(() => {
    if (!activeAlert) return;
    const timer = window.setTimeout(() => setActiveAlert(null), ALERT_BANNER_MS);
    return () => window.clearTimeout(timer);
  }, [activeAlert]);

  const summary = job?.summary;
  const plateCount = summary?.anpr?.detected_plates?.length ?? 0;
  const zoneIntrusions = useMemo(() => deriveZoneIntrusions(events ?? [], summary?.zones ?? []), [events, summary?.zones]);

  // Alerts in playback order. Sounding the siren as the footage reaches each
  // one is the whole point: an alert that fires the moment analysis finishes
  // tells the operator only that the job is done, while one tied to the
  // playhead sounds exactly when the target is on screen and boxed - which is
  // what they are being asked to look at.
  const alertEvents = useMemo(
    () => (events ?? [])
      .filter(isAlertEvent)
      .sort((a, b) => a.timestamp_sec - b.timestamp_sec),
    [events],
  );

  const playbackAlerts: PlaybackAlert[] = useMemo(
    () => alertEvents.map((e) => ({ key: alertKey(e), timestampSec: e.timestamp_sec })),
    [alertEvents],
  );

  function handleTimeUpdate(now: number) {
    const previous = lastTimeRef.current;
    lastTimeRef.current = now;

    const { due, sounded } = alertsReachedBy(playbackAlerts, previous, now, soundedRef.current);
    soundedRef.current = sounded;
    if (due.length === 0) return;

    // Several findings a fraction of a second apart are one moment, not one
    // alarm each, so they are all marked heard but sounded once.
    const first = alertEvents.find((e) => alertKey(e) === due[0].key) ?? null;
    setActiveAlert(first);
    vibrateAlert();
    void playSiren().then((played) => setAudioBlocked(!played));
  }

  function jumpToEvent(ev: BwEvent) {
    setSelectedEvent(ev);
    playerRef.current?.seekTo(ev.timestamp_sec);
  }

  if (!jobId) return null;

  if (jobError && !job) return <PageShell><ErrorState message={jobError} /></PageShell>;
  if (!job) return <PageShell><LoadingBlock label="Loading analysis…" /></PageShell>;
  if (job.status !== "done") {
    return (
      <PageShell>
        <EmptyState icon={Radar} title={`Analysis ${job.status}`} description="This job hasn't completed yet. Return to the processing view to watch its progress." />
      </PageShell>
    );
  }
  if (loadError) return <PageShell><ErrorState message={loadError} /></PageShell>;
  if (!events || !evidence) return <PageShell><LoadingBlock label="Loading events and evidence…" /></PageShell>;

  const title = summary?.person_id ? "Target Person Identification" : "Drishti Analysis Results";

  return (
    <>
      <PageHeader
        title={title}
        subtitle={`Job ${jobId.slice(0, 8)}`}
        actions={<StatusBadge label="COMPLETED" severity="green" />}
      />
      <PageBody>
        {/* Browsers block sound until the page has been interacted with, so a
            silent alert is indistinguishable from a broken one. Say which it
            is, and offer the one click that fixes it. */}
        {audioBlocked && (
          <button
            onClick={() => void playSiren().then((played) => setAudioBlocked(!played))}
            className="mb-4 flex w-full items-center gap-2.5 rounded border border-accent-amber/40 bg-accent-amber/10 px-4 py-3 text-left text-[12.5px] text-accent-amber"
          >
            <VolumeX size={16} className="shrink-0" />
            <span>
              <b>Alert sound blocked</b> — your browser muted it. Click here to enable the siren,
              then play the video.
            </span>
          </button>
        )}

        {/* The siren is tied to the playhead, so say so: an operator who has
            not pressed play yet should know the alerts are waiting in the
            footage rather than wonder why nothing sounded. */}
        {activeAlert ? (
          <div className="mb-4 flex items-center gap-2.5 rounded border border-accent-red/50 bg-accent-red/10 px-4 py-3 text-[12.5px] text-accent-red">
            <Siren size={16} className="shrink-0" />
            <span>
              <b>{titleCase(activeAlert.type)}</b> at {activeAlert.timestamp_sec.toFixed(1)}s — the
              target is on screen now.
            </span>
          </div>
        ) : alertEvents.length > 0 ? (
          <div className="mb-4 flex items-center gap-2.5 rounded border border-border-1 bg-bg-1 px-4 py-2.5 text-[12px] text-text-tertiary">
            <Siren size={14} className="shrink-0 text-accent-red" />
            <span>
              {alertEvents.length} alert{alertEvents.length === 1 ? "" : "s"} in this footage — the
              siren sounds as playback reaches each one.
            </span>
          </div>
        ) : null}

        <div className="mb-5 grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-4">
          <MetricCard label="Target Track" value={summary?.person_id?.confirmed ? `#${summary.person_id.target_track_id}` : "—"} accent={summary?.person_id?.confirmed ? "red" : "default"} />
          <MetricCard label="Video Duration" value={summary?.video_info ? `${summary.video_info.duration_sec.toFixed(1)}s` : "—"} />
          <MetricCard label="Frames Processed" value={job.frames_processed ?? "—"} />
          <MetricCard label="Events" value={events.length} accent="blue" />
          <MetricCard label="Plates Read" value={summary?.anpr ? plateCount : "—"} />
          <MetricCard label="Zone Intrusions" value={summary?.zones ? zoneIntrusions.length : "—"} accent={zoneIntrusions.length ? "amber" : "default"} />
        </div>

        <div className="grid grid-cols-[1fr_360px] gap-5 max-[1200px]:grid-cols-1">
          <div className="flex flex-col gap-5">
            <VideoPlayer ref={playerRef} src={jobOutputUrl(jobId)} onTimeUpdate={handleTimeUpdate} />

            {summary?.person_id && (
              <Panel title="Target Visualization">
                <TargetCard summary={summary.person_id} events={events} />
                <p className="mt-3 text-[11px] text-text-tertiary">
                  The red target overlay is already rendered into the output video by the backend — this panel shows supporting detail only.
                </p>
              </Panel>
            )}

            {summary?.anpr && (
              <Panel title="ANPR Results" meta={`${plateCount} plate${plateCount === 1 ? "" : "s"} detected`}>
                {summary.anpr.searched_plate && (
                  <div className="mb-5">
                    <SectionTitle>Plate Search Result</SectionTitle>
                    <PlateResultCard
                      target={summary.anpr.target!}
                      searchedPlate={summary.anpr.searched_plate}
                      onJumpToVideo={summary.anpr.target?.first_seen_sec != null ? () => playerRef.current?.seekTo(summary.anpr!.target!.first_seen_sec!) : undefined}
                    />
                  </div>
                )}
                <SectionTitle>Plates Detected</SectionTitle>
                <PlatesTable plates={summary.anpr.detected_plates} />
              </Panel>
            )}

            {summary?.zones && (
              <Panel title="Zone Intrusions">
                <ZoneResultsTable intrusions={zoneIntrusions} />
              </Panel>
            )}

            {summary?.behavior && (
              <Panel title="Behavioral Analytics">
                <BehaviorResultsList events={events} />
              </Panel>
            )}

            {summary?.lowlight && (
              <Panel title="Low-Light Enhancement">
                <LowlightSummaryCard summary={summary.lowlight} />
              </Panel>
            )}

            <Panel title="Evidence" meta={`${evidence.length} item${evidence.length === 1 ? "" : "s"}`}>
              <EvidenceGrid jobId={jobId} items={evidence} onOpen={setEvidenceModal} />
            </Panel>
          </div>

          <div className="flex flex-col gap-5">
            <Panel title="Event Timeline" meta={`${events.length}`}>
              <div className="max-h-[640px] overflow-y-auto">
                <EventTimeline events={events} selectedId={selectedEvent?.event_id} onSelect={jumpToEvent} />
              </div>
            </Panel>
          </div>
        </div>
      </PageBody>

      {evidenceModal && (
        <Modal title={evidenceModal.event_type ? titleCase(evidenceModal.event_type) : "Evidence"} onClose={() => setEvidenceModal(null)}>
          <img src={jobEvidenceFileUrl(jobId, evidenceModal.filename)} alt="Evidence" className="w-full rounded-md border border-border-1" />
          <div className="mt-3 flex justify-between text-[12px] text-text-tertiary">
            <span>{evidenceModal.track_id != null ? `Track #${evidenceModal.track_id}` : "—"}</span>
            <span>Frame {evidenceModal.frame_index ?? "—"}</span>
          </div>
        </Modal>
      )}
    </>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <>
      <PageHeader title="Analysis Results" />
      <PageBody>
        <Panel>{children}</Panel>
      </PageBody>
    </>
  );
}
