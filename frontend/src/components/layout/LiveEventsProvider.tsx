import type { ReactNode } from "react";
import { LiveEventsContext, useLiveEventsSocket } from "../../hooks/useLiveEvents";

/** Owns the one live-events websocket for the signed-in app. */
export function LiveEventsProvider({ children }: { children: ReactNode }) {
  const stream = useLiveEventsSocket();
  return <LiveEventsContext.Provider value={stream}>{children}</LiveEventsContext.Provider>;
}
