import { useState } from "react";
import { Camera, Check, Loader2, Plus, Radar } from "lucide-react";
import { ApiError } from "../../api/client";
import { liveApi, type LiveCamera, type OnvifCamera, type OnvifProfile } from "../../api/live";

const input =
  "rounded border border-border-2 bg-bg-3 px-3 py-2 text-[12.5px] text-text-primary placeholder:text-text-disabled focus:border-accent-blue focus:outline-none";

function message(err: unknown, fallback: string) {
  return err instanceof ApiError ? err.message : fallback;
}

/**
 * Adds an IP camera of any brand without typing an RTSP URL.
 *
 * Scan asks every ONVIF camera on the network to answer; the operator picks
 * one and logs in, and the camera itself reports its stream address. That
 * address (with the login in it) stays on the server.
 */
export function OnvifScanPanel({ onAdded }: { onAdded: (camera?: LiveCamera) => void }) {
  const [scanning, setScanning] = useState(false);
  const [scanned, setScanned] = useState<{ interfaces: string[]; cameras: OnvifCamera[] } | null>(null);
  const [picked, setPicked] = useState<OnvifCamera | null>(null);
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [profiles, setProfiles] = useState<OnvifProfile[] | null>(null);
  const [profile, setProfile] = useState<string>("");
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function scan() {
    setScanning(true);
    setError(null);
    setPicked(null);
    setProfiles(null);
    try {
      setScanned(await liveApi.onvifScan());
    } catch (err) {
      setError(message(err, "The scan failed."));
    } finally {
      setScanning(false);
    }
  }

  function pick(camera: OnvifCamera) {
    setPicked(camera);
    setProfiles(null);
    setProfile("");
    setName(camera.name ?? `Camera ${camera.address}`);
    setError(null);
  }

  async function login() {
    if (!picked) return;
    setBusy(true);
    setError(null);
    try {
      const found = await liveApi.onvifProfiles({ xaddr: picked.xaddr, username, password });
      setProfiles(found);
      setProfile(found[0]?.token ?? "");
    } catch (err) {
      setError(message(err, "Could not log in to the camera."));
    } finally {
      setBusy(false);
    }
  }

  async function add() {
    if (!picked || !profile || !name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const camera = await liveApi.onvifAdd({
        xaddr: picked.xaddr, username, password, profile_token: profile,
        name: name.trim(), location: location.trim() || null,
      });
      onAdded(camera);
      setPicked(null);
      setProfiles(null);
      setPassword("");
    } catch (err) {
      setError(message(err, "Could not add the camera."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded border border-border-1 bg-bg-1 p-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-[12.5px] font-semibold text-text-primary">Find cameras automatically</div>
          <div className="text-[11.5px] text-text-tertiary">Any brand that supports ONVIF, on this network.</div>
        </div>
        <button
          onClick={() => void scan()}
          disabled={scanning}
          className="inline-flex shrink-0 items-center gap-1.5 rounded border border-border-2 bg-bg-3 px-3 py-1.5 text-[12px] font-semibold text-text-primary hover:bg-bg-4 disabled:opacity-40"
        >
          {scanning ? <Loader2 size={13} className="animate-spin" /> : <Radar size={13} />}
          {scanning ? "Scanning…" : "Scan network"}
        </button>
      </div>

      {scanned && scanned.cameras.length === 0 && (
        <div className="text-[11.5px] leading-relaxed text-text-tertiary">
          No camera answered on {scanned.interfaces.join(", ") || "this network"}. Check that the camera is
          powered and on the same network, and that ONVIF is switched on in its settings (some brands turn it
          off by default). Windows Firewall can also block the replies. You can still enter its RTSP address below.
        </div>
      )}

      {scanned && scanned.cameras.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {scanned.cameras.map((c) => (
            <button
              key={c.xaddr}
              onClick={() => pick(c)}
              className={`flex items-center gap-2 rounded border px-3 py-2 text-left text-[12px] ${
                picked?.xaddr === c.xaddr ? "border-accent-blue bg-accent-blue/10" : "border-border-2 hover:bg-bg-3"
              }`}
            >
              <Camera size={14} className="shrink-0 text-text-tertiary" />
              <span className="min-w-0 flex-1">
                <span className="block truncate font-semibold text-text-primary">{c.name ?? "ONVIF camera"}</span>
                <span className="block truncate font-mono text-[11px] text-text-tertiary">
                  {c.address}{c.hardware ? ` · ${c.hardware}` : ""}
                </span>
              </span>
              {picked?.xaddr === c.xaddr && <Check size={14} className="text-accent-blue" />}
            </button>
          ))}
        </div>
      )}

      {picked && !profiles && (
        <div className="flex flex-col gap-2">
          <div className="text-[11.5px] text-text-tertiary">Log in with the camera's own username and password.</div>
          <div className="flex gap-2">
            <input id="onvif-user" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" autoComplete="off" className={`${input} min-w-0 flex-1`} />
            <input id="onvif-pass" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" autoComplete="off" className={`${input} min-w-0 flex-1`} />
          </div>
          <button
            onClick={() => void login()}
            disabled={busy || !username}
            className="inline-flex items-center justify-center gap-1.5 rounded border border-border-2 bg-bg-3 px-3 py-2 text-[12px] font-semibold text-text-primary hover:bg-bg-4 disabled:opacity-40"
          >
            {busy && <Loader2 size={13} className="animate-spin" />} Find its video streams
          </button>
        </div>
      )}

      {picked && profiles && (
        <div className="flex flex-col gap-2">
          <div className="text-[11.5px] text-text-tertiary">Pick a stream. The first is the camera's main, highest-resolution stream, best for reading plates.</div>
          {profiles.map((p) => (
            <label key={p.token} className="flex items-center gap-2 text-[12px] text-text-secondary">
              <input type="radio" name="onvif-profile" checked={profile === p.token} onChange={() => setProfile(p.token)} />
              <span className="font-semibold text-text-primary">{p.name}</span>
              <span className="font-mono text-[11px] text-text-tertiary">
                {p.width && p.height ? `${p.width}×${p.height}` : "size unknown"}{p.encoding ? ` · ${p.encoding}` : ""}
              </span>
            </label>
          ))}
          <input id="onvif-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="Camera name" className={input} />
          <input id="onvif-location" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Location (optional, e.g. Gate 3 - East Fence)" className={input} />
          <button
            onClick={() => void add()}
            disabled={busy || !profile || !name.trim()}
            className="flex items-center justify-center gap-2 rounded bg-accent-blue px-4 py-2.5 text-[12.5px] font-semibold text-on-accent disabled:opacity-40"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />} Add this camera
          </button>
        </div>
      )}

      {error && (
        <div className="rounded border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">{error}</div>
      )}
    </div>
  );
}
