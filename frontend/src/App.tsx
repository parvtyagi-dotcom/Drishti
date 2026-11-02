import { useCallback, useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { authApi, type SessionState } from "./api/auth";
import { Analysis } from "./pages/Analysis";
import { AnprSearch } from "./pages/AnprSearch";
import { AuditLog } from "./pages/AuditLog";
import { Dashboard } from "./pages/Dashboard";
import { Devices } from "./pages/Devices";
import { Events } from "./pages/Events";
import { Jobs } from "./pages/Jobs";
import { LiveMonitor } from "./pages/LiveMonitor";
import { Login } from "./pages/Login";
import { Processing } from "./pages/Processing";
import { Results } from "./pages/Results";
import { Settings } from "./pages/Settings";
import { SiteMap } from "./pages/SiteMap";
import { Sensors } from "./pages/Sensors";
import { AppearanceSearch } from "./pages/AppearanceSearch";
import { SatelliteCheck } from "./pages/SatelliteCheck";
import { SystemStatusPage } from "./pages/SystemStatusPage";

export default function App() {
  const [session, setSession] = useState<SessionState | null>(null);

  const refreshSession = useCallback(async () => {
    try {
      setSession(await authApi.me());
    } catch {
      // The API is unreachable. Treat that as signed-out rather than
      // rendering a dashboard that cannot load anything.
      setSession({ configured: true, authenticated: false, user: null });
    }
  }, []);

  useEffect(() => {
    void refreshSession();
  }, [refreshSession]);

  if (session === null) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-bg-0 text-[13px] text-text-tertiary">
        Connecting to Drishti…
      </div>
    );
  }

  // Everything behind this point reaches camera feeds, evidence and the audit
  // trail, so the whole application is gated rather than individual pages.
  if (!session.authenticated) {
    return <Login session={session} onSignedIn={() => void refreshSession()} />;
  }

  return (
    <AppLayout user={session.user} onSignedOut={() => void refreshSession()}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/live" element={<LiveMonitor />} />
        <Route path="/map" element={<SiteMap user={session.user} />} />
        <Route path="/sensors" element={<Sensors user={session.user} />} />
        <Route path="/devices" element={<Devices />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/processing/:jobId" element={<Processing />} />
        <Route path="/results/:jobId" element={<Results />} />
        <Route path="/jobs" element={<Jobs />} />
        <Route path="/events" element={<Events />} />
        <Route path="/audit" element={<AuditLog />} />
        <Route path="/anpr-search" element={<AnprSearch />} />
        <Route path="/appearance-search" element={<AppearanceSearch />} />
        <Route path="/satellite" element={<SatelliteCheck />} />
        <Route path="/system" element={<SystemStatusPage />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </AppLayout>
  );
}
