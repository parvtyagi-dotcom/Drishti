import { useEffect, useRef, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { googleLogout } from "@react-oauth/google";
import { authApi, type SessionUser } from "../../api/auth";
import { useLiveEvents } from "../../hooks/useLiveEvents";
import { playAlarmSound, unlockAudioOnFirstGesture } from "../../lib/siren";
import { Sidebar } from "./Sidebar";

// Alarms older than this are replayed backlog from a reconnect, not news.
const FRESH_ALARM_SEC = 20;

/** Sounds the siren on this machine for live alarms on every page, so the
 * operator hears it while looking at the map or a search, not only on Live
 * Monitor. Live Monitor plays its own (with its "sound blocked" banner). */
function GlobalSiren() {
  const { latestAlarm } = useLiveEvents();
  const { pathname } = useLocation();
  const sounded = useRef<string | null>(null);

  useEffect(() => unlockAudioOnFirstGesture(), []);

  useEffect(() => {
    if (!latestAlarm || pathname.startsWith("/live")) return;
    const key = latestAlarm.event_id ?? `${latestAlarm.type}@${latestAlarm.wall_time}`;
    if (sounded.current === key) return;
    sounded.current = key;
    if (Date.now() / 1000 - latestAlarm.wall_time > FRESH_ALARM_SEC) return;
    void playAlarmSound(latestAlarm.type);
  }, [latestAlarm, pathname]);

  return null;
}

export function AppLayout({
  children,
  user,
  onSignedOut,
}: {
  children: ReactNode;
  user?: SessionUser | null;
  onSignedOut?: () => void;
}) {
  return (
    <div className="flex min-h-screen">
      <GlobalSiren />
      <Sidebar user={user} onSignedOut={onSignedOut} />
      <div className="flex min-w-0 flex-1 flex-col">{children}</div>
    </div>
  );
}

/** Who is signed in, and how to stop being signed in. */
export function SessionFooter({
  user,
  onSignedOut,
}: {
  user?: SessionUser | null;
  onSignedOut?: () => void;
}) {
  if (!user) return null;
  return (
    <div className="border-t border-border-1 px-1 pt-3">
      <div className="mb-2 flex items-center gap-2">
        {user.picture && (
          <img
            src={user.picture}
            alt=""
            referrerPolicy="no-referrer"
            className="h-8 w-8 shrink-0 rounded-full"
          />
        )}
        <div className="min-w-0">
          <div className="truncate text-[12px] font-semibold text-text-primary">
            {user.name || user.username}
          </div>
          <div className="truncate text-[11px] text-text-tertiary">
            {user.email || <span className="capitalize">{user.role}</span>}
          </div>
        </div>
      </div>
      <button
        onClick={() => {
          if (user.email) googleLogout();
          void authApi.logout().finally(() => onSignedOut?.());
        }}
        className="w-full rounded-full bg-bg-3 py-1.5 text-[12px] font-medium text-text-secondary hover:bg-bg-4 hover:text-text-primary"
      >
        Sign out
      </button>
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="glass sticky top-0 z-10 flex min-h-[76px] shrink-0 items-center justify-between gap-4 border-b border-border-1 px-8 py-3 max-[1366px]:px-6">
      <div className="flex min-w-0 flex-col gap-0.5">
        <h1 className="truncate text-[24px] font-bold leading-tight tracking-[-0.025em] text-text-primary">{title}</h1>
        {subtitle && <p className="truncate text-[13px] text-text-secondary">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}

export function PageBody({ children, narrow }: { children: ReactNode; narrow?: boolean }) {
  return <div className={`page-enter flex-1 p-8 max-[1366px]:p-6 ${narrow ? "max-w-[1040px]" : ""}`}>{children}</div>;
}
