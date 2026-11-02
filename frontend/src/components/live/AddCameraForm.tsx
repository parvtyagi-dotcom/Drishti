import { useState } from "react";
import { Loader2, Plus, Smartphone } from "lucide-react";
import { ApiError } from "../../api/client";
import { api } from "../../api/client";
import type { LiveCamera } from "../../api/live";
import { OnvifScanPanel } from "./OnvifScanPanel";
import { PhoneCameraPanel } from "./PhoneCameraPanel";
import { WebcamScanPanel } from "./WebcamScanPanel";

type Preset = "webcam" | "phone" | "rtsp" | "app" | "file";

const PRESETS: { key: Preset; label: string; placeholder: string; hint: string }[] = [
  {
    key: "webcam",
    label: "Webcam",
    placeholder: "",
    hint: "",
  },
  {
    key: "phone",
    label: "Phone",
    placeholder: "",
    hint: "",
  },
  {
    key: "rtsp",
    label: "IP / CCTV",
    placeholder: "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101",
    hint: "Standard RTSP URL from an existing CCTV camera or NVR.",
  },
  {
    key: "app",
    label: "Camera app",
    placeholder: "http://192.168.1.42:8080/video",
    hint: "Install the IP Webcam app on Android, tap 'Start server', and enter the URL it shows with /video on the end. Needs no HTTPS, so this is the most reliable phone option. Both devices must be on the same WiFi.",
  },
  {
    key: "file",
    label: "File",
    placeholder: "uploads/smoke_person_id.mp4",
    hint: "A local clip looped to behave like a continuous camera, for testing without hardware.",
  },
];

/**
 * Registers a new live camera.
 *
 * Any source OpenCV can open works - RTSP from real CCTV, an MJPEG URL from a
 * phone running an IP-camera app, or a local file looped as a stand-in. The
 * analysis pipeline cannot tell them apart, which is the point: nothing
 * downstream changes when a real border camera replaces a test source.
 */
export function AddCameraForm({ onAdded }: { onAdded: (camera?: LiveCamera) => void }) {
  const [preset, setPreset] = useState<Preset>("webcam");
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [location, setLocation] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const active = PRESETS.find((p) => p.key === preset)!;

  async function submit() {
    if (!name.trim() || !source.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const camera = await api.post<LiveCamera>("/api/live/cameras", {
        name: name.trim(),
        source: source.trim(),
        location: location.trim() || null,
        // Only a file needs looping; a real camera is already continuous.
        loop: preset === "file",
        start: true,
      });
      onAdded(camera);
      setName("");
      setSource("");
      setLocation("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not add camera");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-1 rounded-md border border-border-2 bg-bg-3 p-1">
        {PRESETS.map((p) => (
          <button
            key={p.key}
            onClick={() => {
              setPreset(p.key);
              setSource("");
            }}
            className={`flex-1 rounded px-2 py-1.5 text-[11.5px] font-semibold transition-colors ${
              preset === p.key ? "bg-accent-blue text-on-accent" : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {preset === "phone" ? (
        // Pairing stays open after creating the camera so the QR code remains
        // on screen for the phone to scan.
        <PhoneCameraPanel onAdded={() => onAdded()} />
      ) : (
        <>
      {preset === "rtsp" && <OnvifScanPanel onAdded={onAdded} />}
      {preset === "webcam" ? (
        <WebcamScanPanel
          selected={source}
          onPick={(index) => {
            setSource(String(index));
            if (!name.trim()) setName(`WEBCAM-${index}`);
          }}
        />
      ) : (
        <div className="flex items-start gap-2 rounded border border-border-1 bg-bg-1 px-3 py-2 text-[11.5px] text-text-tertiary">
          {preset === "app" && <Smartphone size={14} className="mt-[1px] shrink-0" />}
          <span>{preset === "rtsp" ? `Or enter the address yourself. ${active.hint}` : active.hint}</span>
        </div>
      )}

      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Camera name (e.g. CAM-03)"
        className="rounded border border-border-2 bg-bg-3 px-3 py-2 text-[12.5px] text-text-primary placeholder:text-text-disabled focus:border-accent-blue focus:outline-none"
      />
      {preset !== "webcam" && (
        <input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder={active.placeholder}
          spellCheck={false}
          className="rounded border border-border-2 bg-bg-3 px-3 py-2 font-mono text-[12px] text-text-primary placeholder:font-sans placeholder:text-text-disabled focus:border-accent-blue focus:outline-none"
        />
      )}
      <input
        value={location}
        onChange={(e) => setLocation(e.target.value)}
        placeholder="Location (optional, e.g. Gate 3 - East Fence)"
        className="rounded border border-border-2 bg-bg-3 px-3 py-2 text-[12.5px] text-text-primary placeholder:text-text-disabled focus:border-accent-blue focus:outline-none"
      />

      {error && (
        <div className="rounded border border-accent-red/35 bg-accent-red/10 px-3 py-2 text-[12px] text-accent-red">
          {error}
        </div>
      )}

      <button
        onClick={() => void submit()}
        disabled={busy || !name.trim() || !source.trim()}
        className="flex items-center justify-center gap-2 rounded bg-accent-blue px-4 py-2.5 text-[12.5px] font-semibold text-on-accent disabled:opacity-40"
      >
        {busy ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
        {busy ? "Connecting…" : "Add camera"}
      </button>
        </>
      )}
    </div>
  );
}
