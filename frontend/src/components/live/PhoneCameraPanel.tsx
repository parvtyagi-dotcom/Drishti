import { useCallback, useEffect, useState } from "react";
import { Copy, Loader2, ShieldAlert, Smartphone } from "lucide-react";
import QRCode from "qrcode";
import { ApiError, api } from "../../api/client";
import type { LiveCamera } from "../../api/live";

interface PhoneCameraResponse extends LiveCamera {
  join_path: string;
}

/**
 * Pairs a phone as a live camera.
 *
 * The phone opens a page that captures its camera and pushes frames to the
 * backend, so no app install is needed. Browsers only grant camera access on a
 * secure origin, though, and a LAN address over plain HTTP is not one - so the
 * panel is explicit about needing https rather than letting the operator hit a
 * silent permission failure on the phone.
 */
export function PhoneCameraPanel({ onAdded }: { onAdded: () => void }) {
  const [name, setName] = useState("Phone camera");
  const [location, setLocation] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [joinUrl, setJoinUrl] = useState<string | null>(null);
  const [qrDataUrl, setQrDataUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const isLoopback = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
  // Camera access needs a secure origin. Loopback counts as secure, so opening
  // the dashboard on localhost hides the problem right up until a phone tries
  // to join - which is why both conditions are surfaced separately below.
  const insecure = window.location.protocol !== "https:" && !isLoopback;

  useEffect(() => {
    if (!joinUrl) {
      setQrDataUrl(null);
      return;
    }
    void QRCode.toDataURL(joinUrl, { width: 320, margin: 1, color: { dark: "#0a0f16", light: "#ffffff" } })
      .then(setQrDataUrl)
      .catch(() => setQrDataUrl(null));
  }, [joinUrl]);

  const create = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const camera = await api.post<PhoneCameraResponse>("/api/live/phone-cameras", {
        name: name.trim() || "Phone camera",
        location: location.trim() || null,
      });
      // Built from the address the operator is already using, so the phone is
      // told to dial the same host rather than a hardcoded one.
      setJoinUrl(new URL(camera.join_path, window.location.origin).toString());
      onAdded();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not create phone camera");
    } finally {
      setBusy(false);
    }
  }, [name, location, onAdded]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start gap-2 rounded border border-border-1 bg-bg-1 px-3 py-2 text-[11.5px] text-text-tertiary">
        <Smartphone size={14} className="mt-[1px] shrink-0" />
        <span>
          Turns a phone into a border camera with no app install — it opens a page and streams its
          camera straight into the analysis pipeline.
        </span>
      </div>

      {isLoopback && (
        <div className="flex items-start gap-2 rounded border border-accent-amber/40 bg-accent-amber/10 px-3 py-2 text-[11.5px] text-accent-amber">
          <ShieldAlert size={14} className="mt-[1px] shrink-0" />
          <span>
            You are viewing this on <b>{window.location.hostname}</b>, which a phone cannot reach.
            Reopen the dashboard using this machine's network address (e.g.{" "}
            <code className="font-mono">https://192.168.x.x:8443</code>) before pairing, or the QR
            code will point the phone back at itself.
          </span>
        </div>
      )}

      {insecure && (
        <div className="flex items-start gap-2 rounded border border-accent-amber/40 bg-accent-amber/10 px-3 py-2 text-[11.5px] text-accent-amber">
          <ShieldAlert size={14} className="mt-[1px] shrink-0" />
          <span>
            You are on <b>http</b>. Phones refuse camera access unless the page is served over
            <b> https</b>. Run <code className="font-mono">python scripts/make_dev_cert.py</code> and
            restart the API with TLS, then reopen the dashboard over https — or use an IP-camera app
            instead, which needs no HTTPS.
          </span>
        </div>
      )}

      {!joinUrl ? (
        <>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Camera name"
            className="rounded border border-border-2 bg-bg-3 px-3 py-2 text-[12.5px] text-text-primary placeholder:text-text-disabled focus:border-accent-blue focus:outline-none"
          />
          <input
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="Location (optional)"
            className="rounded border border-border-2 bg-bg-3 px-3 py-2 text-[12.5px] text-text-primary placeholder:text-text-disabled focus:border-accent-blue focus:outline-none"
          />
          {error && (
            <div className="rounded border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">
              {error}
            </div>
          )}
          <button
            onClick={() => void create()}
            disabled={busy}
            className="flex items-center justify-center gap-2 rounded bg-accent-blue px-4 py-2.5 text-[12.5px] font-semibold text-on-accent disabled:opacity-40"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Smartphone size={14} />}
            {busy ? "Creating…" : "Create phone camera"}
          </button>
        </>
      ) : (
        <div className="flex flex-col items-center gap-3">
          <div className="text-center text-[12px] text-text-secondary">
            Scan with the phone, then tap <b>Start streaming</b>.
          </div>
          {qrDataUrl && (
            <img src={qrDataUrl} alt="QR code to open the phone camera page" className="w-44 rounded bg-white p-2" />
          )}
          <div className="flex w-full items-center gap-2">
            <code className="min-w-0 flex-1 truncate rounded border border-border-1 bg-bg-1 px-2 py-1.5 font-mono text-[10.5px] text-text-tertiary">
              {joinUrl}
            </code>
            <button
              onClick={() => {
                void navigator.clipboard.writeText(joinUrl).then(() => {
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1500);
                });
              }}
              className="shrink-0 rounded border border-border-2 p-1.5 text-text-secondary hover:bg-bg-3 hover:text-text-primary"
              title="Copy link"
            >
              <Copy size={13} />
            </button>
          </div>
          {copied && <div className="text-[11px] text-accent-green">Link copied</div>}
          <button
            onClick={() => setJoinUrl(null)}
            className="w-full rounded border border-border-2 px-3 py-2 text-[12px] text-text-secondary hover:bg-bg-3"
          >
            Pair another phone
          </button>
        </div>
      )}
    </div>
  );
}
