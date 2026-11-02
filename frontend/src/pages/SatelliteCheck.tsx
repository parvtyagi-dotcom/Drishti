import { useEffect, useState } from "react";
import { CalendarClock, Info, Loader2, MapPin, Satellite, Search } from "lucide-react";
import { Panel } from "../components/common/Panel";
import { ApiError } from "../api/client";
import {
  satelliteApi, type FetchJob, type Place, type SatelliteChange, type SatellitePair, type SatelliteResult,
} from "../api/satellite";

const KIND_COLOUR: Record<SatelliteChange["kind"], string> = {
  vegetation_cleared: "#f59e0b",
  vegetation_growth: "#22c55e",
  track: "#ef4444",
  bare_ground: "#eab308",
  structure: "#38bdf8",
};

function days(a: string, b: string) {
  return Math.round((Date.parse(b) - Date.parse(a)) / 86400000);
}

function Overlay({ result, selected, onSelect }: {
  result: SatelliteResult; selected: number | null; onSelect: (id: number) => void;
}) {
  return (
    <svg viewBox={`0 0 ${result.width} ${result.height}`} className="absolute inset-0 h-full w-full">
      {result.changes.map((c) => {
        const colour = KIND_COLOUR[c.kind];
        const on = selected === c.id;
        const [x, y, w, h] = c.box;
        return (
          <g key={c.id} onClick={() => onSelect(c.id)} className="cursor-pointer">
            <polygon points={c.polygon.map((p) => p.join(",")).join(" ")} fill={colour} fillOpacity={on ? 0.45 : 0.18}
              stroke={colour} strokeWidth={on ? 2 : 1} vectorEffect="non-scaling-stroke" />
            <rect x={x - 2} y={y - 2} width={w + 4} height={h + 4} fill="none" stroke={colour}
              strokeOpacity={on ? 1 : 0.6} strokeDasharray={on ? undefined : "4 3"} vectorEffect="non-scaling-stroke" />
            <text x={x} y={Math.max(9, y - 4)} fontSize={Math.max(9, result.width / 60)} fill={colour}
              fontWeight={700} stroke="#000" strokeWidth={0.6} paintOrder="stroke">{c.id}</text>
          </g>
        );
      })}
    </svg>
  );
}

function isoDaysAgo(n: number) {
  return new Date(Date.now() - n * 86400000).toISOString().slice(0, 10);
}

/** Search any place by name, pick two dates, and fetch a fresh pair - the way
 * you would look somewhere up on a map site. Needs internet; saved pairs don't. */
function PlaceSearch({ onFetched }: { onFetched: (pairId: string) => void }) {
  const [query, setQuery] = useState("");
  const [places, setPlaces] = useState<Place[] | null>(null);
  const [place, setPlace] = useState<Place | null>(null);
  const [before, setBefore] = useState(isoDaysAgo(60));
  const [after, setAfter] = useState(isoDaysAgo(3));
  const [sizeKm, setSizeKm] = useState(5);
  const [searching, setSearching] = useState(false);
  const [job, setJob] = useState<FetchJob | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function search() {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    setPlace(null);
    try {
      const r = await satelliteApi.places(query);
      setPlaces(r.places);
      if (r.places.length === 0) setError("No place found by that name.");
    } catch (err) {
      setPlaces(null);
      setError(err instanceof ApiError ? err.message : "Place search failed.");
    } finally {
      setSearching(false);
    }
  }

  async function fetchImages() {
    if (!place) return;
    setError(null);
    try {
      let current = await satelliteApi.fetch({ name: place.name, lat: place.lat, lon: place.lon, before, after, size_km: sizeKm });
      setJob(current);
      while (current.status === "running") {
        await new Promise((r) => setTimeout(r, 2000));
        current = await satelliteApi.fetchStatus(current.job_id);
        setJob(current);
      }
      if (current.status === "done") onFetched(current.pair_id);
      else setError(current.error ?? "The fetch failed.");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The fetch failed.");
      setJob(null);
    }
  }

  const running = job?.status === "running";
  const fieldClass = "rounded-lg border border-border-2 bg-bg-1 px-3 py-2 text-[13px] text-text-primary";

  return (
    <Panel title="Search any place" meta="needs internet">
      <form onSubmit={(e) => { e.preventDefault(); void search(); }} className="relative flex gap-2">
        <div className="relative flex-1">
          <Search size={15} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Type a place, e.g. Uri, Kashmir or Tanot, Rajasthan"
            className={`${fieldClass} w-full pl-9`} aria-label="Place name" />
        </div>
        <button type="submit" disabled={searching || !query.trim()}
          className="flex items-center gap-1.5 rounded-lg bg-accent-blue px-4 text-[13px] font-medium text-on-accent disabled:opacity-50">
          {searching && <Loader2 size={13} className="animate-spin" />} Search
        </button>
      </form>

      {places && places.length > 0 && !place && (
        <ul className="mt-2 overflow-hidden rounded-lg border border-border-2">
          {places.map((p) => (
            <li key={`${p.lat},${p.lon}`}>
              <button onClick={() => setPlace(p)}
                className="flex w-full items-start gap-2 border-b border-border-1 px-3 py-2 text-left text-[12.5px] text-text-primary last:border-0 hover:bg-bg-3">
                <MapPin size={14} className="mt-0.5 shrink-0 text-accent-red" />
                <span>{p.name} <span className="text-text-tertiary">· {p.lat.toFixed(3)}, {p.lon.toFixed(3)}</span></span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {place && (
        <div className="mt-3 space-y-3">
          <div className="flex items-start justify-between gap-3 rounded-lg border border-border-2 bg-bg-1 px-3 py-2 text-[12.5px]">
            <span className="flex items-start gap-2 text-text-primary">
              <MapPin size={14} className="mt-0.5 shrink-0 text-accent-red" /> {place.name}
            </span>
            <button onClick={() => setPlace(null)} disabled={running} className="shrink-0 text-accent-blue disabled:opacity-50">Change</button>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1 text-[12px] text-text-secondary">
              Before
              <input type="date" value={before} max={after} onChange={(e) => setBefore(e.target.value)} className={fieldClass} disabled={running} />
            </label>
            <span className="pb-2 text-text-tertiary">→</span>
            <label className="flex flex-col gap-1 text-[12px] text-text-secondary">
              After
              <input type="date" value={after} min={before} max={isoDaysAgo(0)} onChange={(e) => setAfter(e.target.value)} className={fieldClass} disabled={running} />
            </label>
            <label className="flex flex-col gap-1 text-[12px] text-text-secondary">
              Area
              <select value={sizeKm} onChange={(e) => setSizeKm(Number(e.target.value))} className={fieldClass} disabled={running}>
                {[3, 5, 10].map((km) => <option key={km} value={km}>{km} × {km} km</option>)}
              </select>
            </label>
            <button onClick={() => void fetchImages()} disabled={running}
              className="flex items-center gap-1.5 rounded-lg bg-accent-blue px-4 py-2 text-[13px] font-medium text-on-accent disabled:opacity-50">
              {running ? <Loader2 size={13} className="animate-spin" /> : <Satellite size={14} />}
              {running ? "Fetching…" : "Get satellite images"}
            </button>
          </div>
          <p className="text-[11.5px] text-text-tertiary">
            Finds the clearest picture within about two weeks of each date. Pick dates a week or more apart.
          </p>
          {job && job.log.length > 0 && (
            <pre className="max-h-32 overflow-y-auto rounded-lg border border-border-1 bg-bg-0 p-2 font-mono text-[11px] text-text-secondary">
              {job.log.join("\n")}
            </pre>
          )}
        </div>
      )}

      {error && <p className="mt-2 text-[12.5px] text-accent-red">{error}</p>}
    </Panel>
  );
}

/** Where the box is: the wider region with the spot marked, or the box itself
 * on a street map. Both are pictures saved with the pair, so no internet. */
function LocationMap({ pair }: { pair: SatellitePair }) {
  const [zoom, setZoom] = useState<"map_region" | "map_local">("map_region");
  const [w, s, e, n] = pair.bbox;
  return (
    <Panel
      title="Where this is"
      actions={
        <div className="flex rounded border border-border-1 text-[12px]">
          {([["map_region", "Region"], ["map_local", "Close-up"]] as const).map(([key, label]) => (
            <button key={key} onClick={() => setZoom(key)}
              className={`px-2.5 py-1 ${zoom === key ? "bg-bg-3 text-text-primary" : "text-text-tertiary"}`}>
              {label}
            </button>
          ))}
        </div>
      }
    >
      <img src={satelliteApi.imageUrl(pair.id, zoom)} alt={`Map of ${pair.name}`}
        className="block w-full rounded border border-border-1" style={{ aspectRatio: "640 / 400" }} />
      <p className="mt-2 flex items-start gap-1.5 text-[11.5px] leading-relaxed text-text-tertiary">
        <MapPin size={13} className="mt-0.5 shrink-0" />
        {((n + s) / 2).toFixed(4)}° N, {((w + e) / 2).toFixed(4)}° E · the red {zoom === "map_region" ? "circle" : "box"} is
        the area compared below.
      </p>
    </Panel>
  );
}

export function SatelliteCheck() {
  const [pairs, setPairs] = useState<SatellitePair[] | null>(null);
  const [pairId, setPairId] = useState<string>("");
  const [result, setResult] = useState<SatelliteResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"slider" | "side">("slider");
  const [split, setSplit] = useState(50);
  const [selected, setSelected] = useState<number | null>(null);

  function loadPairs(select?: string) {
    satelliteApi.pairs()
      .then((r) => { setPairs(r.pairs); setPairId(select ?? r.pairs[0]?.id ?? ""); })
      .catch((err) => { setPairs([]); setError(err instanceof ApiError ? err.message : "Could not load saved pairs."); });
  }

  useEffect(() => loadPairs(), []);

  useEffect(() => {
    if (!pairId) return;
    setResult(null);
    setSelected(null);
    setError(null);
    satelliteApi.lastCheck(pairId).then(setResult).catch(() => void run(pairId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pairId]);

  async function run(id: string) {
    setBusy(true);
    setError(null);
    try {
      setResult(await satelliteApi.check(id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "The check failed.");
    } finally {
      setBusy(false);
    }
  }

  const pair = pairs?.find((p) => p.id === pairId);
  const ready = result && result.pair_id === pairId ? result : null;
  const before = pair ? satelliteApi.imageUrl(pair.id, "before") : "";
  const after = pair ? satelliteApi.imageUrl(pair.id, "after") : "";
  const imgClass = "block h-full w-full select-none [image-rendering:pixelated]";
  const aspect = ready ? `${ready.width} / ${ready.height}` : "1 / 1";

  return (
    <div className="space-y-5">
      <div>
        <h1 className="flex items-center gap-2 text-[20px] font-bold text-text-primary">
          <Satellite size={20} /> Satellite Check
        </h1>
        <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-text-tertiary">
          Compares two free Sentinel-2 satellite images of the same stretch behind the border and marks what changed
          on the ground: new paths and tracks, cleared vegetation, new structures. Cameras cannot see this far.
          Search any place and pick two dates to fetch new images (needs internet), or open a saved pair, which is
          checked here on this machine with no internet.
        </p>
      </div>

      <div className="flex items-start gap-2 rounded border border-accent-amber/40 bg-accent-amber/10 px-4 py-3 text-[12.5px] text-accent-amber">
        <CalendarClock size={15} className="mt-0.5 shrink-0" />
        <span>
          <b>Weekly overview, not live.</b> Each picture is days old. One pixel is {pair?.pixel_m ?? 10} m across, so it
          shows paths and clearings, never people. Clouds block it.
        </span>
      </div>

      <PlaceSearch onFetched={(id) => loadPairs(id)} />

      {pairs && pairs.length === 0 && !error && (
        <div className="flex items-start gap-2 rounded border border-border-1 px-4 py-3 text-[12.5px] text-text-tertiary">
          <Info size={15} className="mt-0.5 shrink-0" /> No saved pairs. Run scripts/fetch_sentinel.py at sector HQ.
        </div>
      )}

      {pairs && pairs.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          {pairs.map((p) => (
            <button key={p.id} onClick={() => setPairId(p.id)}
              className={`rounded border px-3 py-1.5 text-[12.5px] ${p.id === pairId
                ? "border-accent-blue bg-accent-blue/15 text-text-primary" : "border-border-1 text-text-tertiary hover:text-text-primary"}`}>
              {p.name}
            </button>
          ))}
          {pair && (
            <button onClick={() => void run(pair.id)} disabled={busy}
              className="ml-auto flex items-center gap-1.5 rounded border border-border-1 px-3 py-1.5 text-[12.5px] text-text-primary disabled:opacity-50">
              {busy && <Loader2 size={13} className="animate-spin" />} Run check again
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="rounded border border-accent-red/40 bg-accent-red/10 px-4 py-3 text-[12.5px] text-accent-red">{error}</div>
      )}

      {pair && (
        <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
          <Panel
            title={`${pair.before_date} → ${pair.after_date} (${days(pair.before_date, pair.after_date)} days apart)`}
            meta={pair.has_ndvi ? "colour + vegetation index (NDVI)" : "colour only"}
            actions={
              <div className="flex rounded border border-border-1 text-[12px]">
                {(["slider", "side"] as const).map((m) => (
                  <button key={m} onClick={() => setMode(m)}
                    className={`px-2.5 py-1 ${mode === m ? "bg-bg-1 text-text-primary" : "text-text-tertiary"}`}>
                    {m === "slider" ? "Slider" : "Side by side"}
                  </button>
                ))}
              </div>
            }
          >
            {mode === "slider" ? (
              <div>
                <div className="relative mx-auto w-full max-w-[820px] overflow-hidden rounded bg-bg-1" style={{ aspectRatio: aspect }}>
                  <img src={after} alt="after" className={`absolute inset-0 ${imgClass}`} draggable={false} />
                  <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
                    <img src={before} alt="before" className={imgClass} draggable={false} />
                  </div>
                  <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `${split}%` }} />
                  <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white">Before {pair.before_date}</span>
                  <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white">After {pair.after_date}</span>
                  {ready && <Overlay result={ready} selected={selected} onSelect={setSelected} />}
                </div>
                <input type="range" min={0} max={100} value={split} onChange={(e) => setSplit(Number(e.target.value))}
                  className="mx-auto mt-3 block w-full max-w-[820px]" aria-label="Before / after slider" />
              </div>
            ) : (
              <div className="grid gap-3 md:grid-cols-2">
                {([["Before", before, pair.before_date], ["After", after, pair.after_date]] as const).map(([label, src, date]) => (
                  <div key={label}>
                    <div className="mb-1 text-[12px] text-text-tertiary">{label} · {date}</div>
                    <div className="relative overflow-hidden rounded bg-bg-1" style={{ aspectRatio: aspect }}>
                      <img src={src} alt={label} className={imgClass} draggable={false} />
                      {ready && label === "After" && <Overlay result={ready} selected={selected} onSelect={setSelected} />}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <p className="mt-3 text-[11.5px] leading-relaxed text-text-tertiary">
              {pair.source}. Scenes {pair.before_scene} and {pair.after_scene}.
              {ready && ` Aligned by ${ready.shift_px[0]}, ${ready.shift_px[1]} px; checked in ${ready.seconds} s.`}
            </p>
          </Panel>

          <div className="space-y-5">
          {pair.has_map && <LocationMap key={pair.id} pair={pair} />}
          <Panel title="What changed" meta={ready ? `${ready.total_changes} areas · ${(ready.changed_area_m2 / 10000).toFixed(1)} ha` : undefined}>
            {pair.note && <p className="mb-3 text-[12px] leading-relaxed text-text-tertiary">{pair.note}</p>}
            {busy && !ready && (
              <div className="flex items-center gap-2 text-[12.5px] text-text-tertiary"><Loader2 size={14} className="animate-spin" /> Comparing…</div>
            )}
            {ready && ready.changes.length === 0 && <p className="text-[12.5px] text-text-tertiary">No change found.</p>}
            {ready && (
              <ol className="max-h-[560px] space-y-1 overflow-y-auto pr-1">
                {ready.changes.map((c) => (
                  <li key={c.id}>
                    <button onClick={() => setSelected(selected === c.id ? null : c.id)}
                      className={`flex w-full items-start gap-2 rounded px-2 py-1.5 text-left text-[12.5px] ${selected === c.id ? "bg-bg-1" : "hover:bg-bg-1"}`}>
                      <span className="mt-0.5 w-5 shrink-0 text-right font-bold" style={{ color: KIND_COLOUR[c.kind] }}>{c.id}</span>
                      <span className="text-text-primary">{c.reason}</span>
                    </button>
                  </li>
                ))}
              </ol>
            )}
            {ready && ready.total_changes > ready.changes.length && (
              <p className="mt-2 text-[11.5px] text-text-tertiary">Showing the {ready.changes.length} largest of {ready.total_changes}.</p>
            )}
            <p className="mt-3 border-t border-border-1 pt-3 text-[11.5px] leading-relaxed text-text-tertiary">
              A change is something to send a patrol to look at, not proof of a crossing. Harvests, floods and drifting
              sand also change the ground.
            </p>
          </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
