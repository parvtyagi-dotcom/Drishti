import { useEffect, useMemo, useState } from "react";
import { liveApi, type LiveCamera, type LiveEvent } from "../api/live";
import { ApiError } from "../api/client";
import { mapApi, type SiteMap } from "../api/map";
import { useLiveEvents } from "./useLiveEvents";

const POLL_MS = 15000;
const HISTORY = 500;

/** Cameras, the site plan and live events for the Overview: history from the REST backlog,
 * kept current by the event websocket. */
export function useOpsPicture(site: string | null, onMissingSite: () => void) {
  const [cameras, setCameras] = useState<LiveCamera[]>([]);
  const [history, setHistory] = useState<LiveEvent[]>([]);
  const [map, setMap] = useState<SiteMap | null>(null);
  const live = useLiveEvents();

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      void liveApi.listCameras().then((c) => !cancelled && setCameras(c)).catch(() => {});
      void liveApi.recentEvents(HISTORY).then((e) => !cancelled && setHistory(e)).catch(() => {});
      void mapApi
        .get(site)
        .then((m) => !cancelled && setMap(m))
        // A remembered site that was removed since: fall back to the default.
        .catch((err) => err instanceof ApiError && err.status === 404 && onMissingSite());
    };
    load();
    const id = window.setInterval(load, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
    // onMissingSite only resets the choice; reloading on its identity would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [site]);

  const events = useMemo(() => {
    const seen = new Set<string>();
    const out: LiveEvent[] = [];
    for (const e of [...live.events, ...history]) {
      const key = e.event_id ?? `${e.camera_id}:${e.type}:${e.wall_time}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(e);
    }
    return out.sort((a, b) => b.wall_time - a.wall_time);
  }, [live.events, history]);

  return { cameras, events, map, connected: live.connected };
}

const SEVERITY_WEIGHT: Record<string, number> = { CRITICAL: 25, HIGH: 12, MEDIUM: 5, LOW: 2, INFO: 0 };

/** 0-100 pressure score from the last hour of events, weighted by severity.
 * A rule of thumb for the gauge, not a model output. */
export function threatIndex(events: LiveEvent[], nowSec: number): number {
  let score = 0;
  for (const e of events) {
    if (nowSec - e.wall_time > 3600) continue;
    score += SEVERITY_WEIGHT[e.severity] ?? 0;
  }
  return Math.min(100, Math.round(score));
}

/** Count of matching events per hour for the last 24 hours, oldest first. */
export function hourly(events: LiveEvent[], nowSec: number, match: (e: LiveEvent) => boolean = () => true): number[] {
  const buckets = new Array<number>(24).fill(0);
  for (const e of events) {
    const age = nowSec - e.wall_time;
    if (age < 0 || age >= 24 * 3600 || !match(e)) continue;
    buckets[23 - Math.floor(age / 3600)] += 1;
  }
  return buckets;
}
