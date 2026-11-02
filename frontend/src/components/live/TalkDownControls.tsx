import { useCallback, useEffect, useState } from "react";
import { Megaphone, Speaker, X } from "lucide-react";
import QRCode from "qrcode";
import { api, ApiError } from "../../api/client";
import { liveApi, type LiveCamera } from "../../api/live";

const POLL_MS = 4000;

/** Warn-now button, automatic talk-down switch, and the link that turns a
 * phone near the camera into its speaker. */
export function TalkDownControls({ camera }: { camera: LiveCamera }) {
  const [speakers, setSpeakers] = useState(0);
  const [auto, setAuto] = useState(true);
  const [note, setNote] = useState<string | null>(null);
  const [showLink, setShowLink] = useState(false);
  const [base, setBase] = useState<string | null>(null);
  const [qr, setQr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await liveApi.talkdownStatus();
      setSpeakers(s.speakers[camera.name] ?? 0);
      setAuto(s.auto);
    } catch {
      /* the status line just stays as it was */
    }
  }, [camera.name]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!showLink) return;
    api.get<{ base_url: string | null }>("/api/live/connect-info")
      .then((info) => setBase(info.base_url ?? window.location.origin))
      .catch(() => setBase(window.location.origin));
  }, [showLink]);

  const url = base ? `${base}/api/live/speaker?camera=${encodeURIComponent(camera.name)}` : null;
  useEffect(() => {
    if (!url) return;
    void QRCode.toDataURL(url, { width: 300, margin: 1 }).then(setQr).catch(() => setQr(null));
  }, [url]);

  async function warn() {
    try {
      const r = await liveApi.warnNow(camera.camera_id);
      setNote(`Warning played on ${r.speakers} speaker${r.speakers === 1 ? "" : "s"}.`);
    } catch (err) {
      setNote(err instanceof ApiError ? err.message : "Could not send the warning.");
    }
    window.setTimeout(() => setNote(null), 6000);
  }

  return (
    <div className="flex flex-col gap-2 rounded border border-border-1 bg-bg-1 px-3 py-2 text-[12px]">
      <div className="flex flex-wrap items-center gap-2">
        <Megaphone size={14} className="text-accent-amber" />
        <span className="text-text-secondary">
          Talk-down: {speakers ? `${speakers} speaker${speakers === 1 ? "" : "s"} connected` : "no speaker connected"}
        </span>
        <label className="ml-2 flex items-center gap-1.5 text-text-tertiary">
          <input
            type="checkbox"
            checked={auto}
            onChange={(e) => void liveApi.setTalkdownAuto(e.target.checked).then((s) => setAuto(s.auto))}
          />
          warn automatically on approach or loitering
        </label>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setShowLink((v) => !v)}
            className="flex items-center gap-1.5 rounded border border-border-2 px-2.5 py-1 text-text-secondary hover:text-text-primary"
          >
            {showLink ? <X size={12} /> : <Speaker size={12} />} Speaker link
          </button>
          <button
            onClick={() => void warn()}
            disabled={!speakers}
            title={speakers ? "Play the spoken warning now" : "Connect a speaker first"}
            className="flex items-center gap-1.5 rounded bg-accent-amber px-3 py-1 font-semibold text-black disabled:opacity-40"
          >
            <Megaphone size={12} /> Warn now
          </button>
        </div>
      </div>
      {note && <div className="text-text-tertiary">{note}</div>}
      {showLink && url && (
        <div className="flex items-center gap-3">
          {qr && <img src={qr} alt="Speaker page QR code" className="w-28 rounded bg-white p-1" />}
          <div className="min-w-0 text-text-tertiary">
            Open on a phone or PC with a loudspeaker placed near this camera, then tap Start speaker.
            <code className="mt-1 block truncate font-mono text-[10.5px]">{url}</code>
          </div>
        </div>
      )}
    </div>
  );
}
