import { useEffect, useRef, useState } from "react";
import type { LiveEvent } from "../api/live";

const MAX_EVENTS = 200;
const RECONNECT_DELAY_MS = 2000;

export interface LiveEventStream {
  events: LiveEvent[];
  connected: boolean;
  /** Most recent alarm-worthy event, for the alert banner. */
  latestAlarm: LiveEvent | null;
  clearAlarm: () => void;
}

/**
 * Subscribes to the backend's live event websocket.
 *
 * The socket reconnects on its own: an operator dashboard left open all shift
 * must not go quietly deaf because the backend restarted once. Events are
 * capped in memory because a live feed has no end - the full history lives
 * server-side, this is only what the panel needs to render.
 */
export function useLiveEvents(): LiveEventStream {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [latestAlarm, setLatestAlarm] = useState<LiveEvent | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const closedByUs = useRef(false);

  useEffect(() => {
    closedByUs.current = false;

    const connect = () => {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${window.location.host}/api/live/ws/events`);
      socketRef.current = ws;

      ws.onopen = () => setConnected(true);

      ws.onmessage = (msg) => {
        let event: LiveEvent;
        try {
          event = JSON.parse(msg.data) as LiveEvent;
        } catch {
          return;
        }
        setEvents((prev) => {
          // Belt and braces against a redelivered event (a reconnect replays
          // the backlog): the server dedupes its own race, this covers ours.
          if (event.event_id && prev.some((e) => e.event_id === event.event_id)) return prev;
          return [event, ...prev].slice(0, MAX_EVENTS);
        });
        if (event.alarm) setLatestAlarm(event);
      };

      ws.onclose = () => {
        setConnected(false);
        if (closedByUs.current) return;
        timerRef.current = window.setTimeout(connect, RECONNECT_DELAY_MS);
      };

      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      closedByUs.current = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
      socketRef.current?.close();
    };
  }, []);

  return { events, connected, latestAlarm, clearAlarm: () => setLatestAlarm(null) };
}
