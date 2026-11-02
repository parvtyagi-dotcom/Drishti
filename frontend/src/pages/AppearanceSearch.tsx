import { useEffect, useState, type FormEvent } from "react";
import { Info, Loader2, Search } from "lucide-react";
import { Panel } from "../components/common/Panel";
import { ApiError } from "../api/client";
import { liveApi, type LiveCamera } from "../api/live";
import { searchApi, type AppearanceMatch, type AppearanceStatus } from "../api/search";

const EXAMPLES = ["person in a red shirt", "person carrying a backpack", "white car", "motorcycle", "person in dark clothes"];

function when(seconds: number) {
  return new Date(seconds * 1000).toLocaleString([], {
    hour12: false, month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

const FIELD = "mt-1 block rounded border border-border-1 bg-bg-1 px-2 py-2 text-[13px] text-text-primary";

export function AppearanceSearch() {
  const [status, setStatus] = useState<AppearanceStatus | null>(null);
  const [cameras, setCameras] = useState<LiveCamera[]>([]);
  const [q, setQ] = useState("");
  const [kind, setKind] = useState("");
  const [camera, setCamera] = useState("");
  const [hours, setHours] = useState(24);
  const [results, setResults] = useState<AppearanceMatch[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    searchApi.status().then(setStatus).catch(() => setStatus({ available: false, message: "Could not reach search." }));
    liveApi.listCameras().then(setCameras).catch(() => setCameras([]));
  }, []);

  async function run(text: string) {
    if (text.trim().length < 2) return;
    setQ(text);
    setBusy(true);
    setError(null);
    try {
      const r = await searchApi.search(text.trim(), kind, camera, hours);
      setResults(r.results);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Search failed.");
    } finally {
      setBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    void run(q);
  }

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-[20px] font-bold text-text-primary">Appearance Search</h1>
        <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-text-tertiary">
          Describe a person or vehicle in plain words. Every person and vehicle the live cameras track leaves one picture,
          and these are ranked by how well they match your words. It runs on this machine with an open image-text model
          (CLIP). It ranks pictures for you to check; it does not identify anyone and never raises an alarm.
        </p>
      </div>

      {status && !status.available && (
        <div className="flex items-start gap-2 rounded border border-accent-amber/40 bg-accent-amber/10 px-4 py-3 text-[12.5px] text-accent-amber">
          <Info size={15} className="mt-0.5 shrink-0" /> {status.message}
        </div>
      )}

      <Panel title="Search" meta={status?.available ? `${(status.sightings ?? 0).toLocaleString()} pictures indexed` : undefined}>
        <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
          <label className="min-w-[260px] flex-1 text-[12px] text-text-tertiary">
            Description
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="man in a red jacket"
              className="mt-1 block w-full rounded border border-border-1 bg-bg-1 px-3 py-2 text-[14px] text-text-primary"
            />
          </label>
          <label className="text-[12px] text-text-tertiary">
            Looking for
            <select value={kind} onChange={(e) => setKind(e.target.value)} className={FIELD}>
              <option value="">People and vehicles</option>
              <option value="person">People</option>
              <option value="vehicle">Vehicles</option>
            </select>
          </label>
          <label className="text-[12px] text-text-tertiary">
            Camera
            <select value={camera} onChange={(e) => setCamera(e.target.value)} className={FIELD}>
              <option value="">All cameras</option>
              {cameras.map((c) => <option key={c.camera_id} value={c.name}>{c.name}</option>)}
            </select>
          </label>
          <label className="text-[12px] text-text-tertiary">
            Within
            <select value={hours} onChange={(e) => setHours(Number(e.target.value))} className={FIELD}>
              <option value={1}>last hour</option>
              <option value={6}>last 6 hours</option>
              <option value={24}>last 24 hours</option>
              <option value={168}>last 7 days</option>
            </select>
          </label>
          <button
            type="submit"
            disabled={busy || !status?.available || q.trim().length < 2}
            className="flex items-center gap-1.5 rounded bg-accent-blue px-4 py-2 text-[13px] font-semibold text-white disabled:opacity-50"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />} Search
          </button>
        </form>
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => void run(ex)}
              disabled={!status?.available}
              className="rounded-full border border-border-1 px-3 py-1 text-[11.5px] text-text-tertiary hover:text-text-primary disabled:opacity-50"
            >
              {ex}
            </button>
          ))}
        </div>
      </Panel>

      {error && <div className="rounded border border-accent-red/40 bg-accent-red/10 px-4 py-2 text-[12.5px] text-accent-red">{error}</div>}

      {results && (
        <Panel title="Best matches" meta="best first · check each picture yourself">
          {results.length === 0 ? (
            <p className="text-[13px] text-text-tertiary">
              Nothing indexed in that time and camera range yet. A camera adds pictures only while zone or behaviour analysis is on.
            </p>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-3">
              {results.map((r, i) => (
                <div key={r.id} className="overflow-hidden rounded-lg border border-border-1 bg-bg-1">
                  <div className="flex h-44 items-center justify-center bg-black">
                    <img src={searchApi.cropUrl(r.id)} alt={`${r.kind} on ${r.camera_name}`} loading="lazy" className="max-h-full max-w-full object-contain" />
                  </div>
                  <div className="space-y-0.5 px-2.5 py-2 text-[11.5px]">
                    <div className="flex justify-between font-semibold text-text-primary">
                      <span>#{i + 1} {r.kind}</span>
                      <span className="font-mono text-text-tertiary">{r.score.toFixed(2)}</span>
                    </div>
                    <div className="text-text-secondary">{r.camera_name} · track {r.track_id}</div>
                    <div className="text-text-tertiary">{when(r.wall_time)}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}
    </div>
  );
}
