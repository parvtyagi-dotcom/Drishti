import { useCallback, useEffect, useState } from "react";
import { Camera, Copy, Download, RefreshCw, ShieldAlert, Siren, Smartphone, WifiOff } from "lucide-react";
import QRCode from "qrcode";
import { api, ApiError } from "../api/client";
import { Panel } from "../components/common/Panel";

interface ConnectInfo {
  base_url: string | null;
  // Where a phone should go for anything needing camera access. May differ
  // from base_url: the server usually listens on both http and https, and
  // only the https one can ask a phone for its camera.
  secure_base_url: string | null;
  apk_url: string | null;
  addresses: string[];
  scheme: string;
  port: number;
  siren_url: string | null;
  camera_capable: boolean;
  camera_warning: string | null;
  on_network: boolean;
}

interface PhoneCamera {
  camera_id: string;
  name: string;
  join_path: string;
}

function QrCard({
  title, description, url, icon: Icon, disabled, disabledReason,
}: {
  title: string;
  description: string;
  url: string | null;
  icon: typeof Siren;
  disabled?: boolean;
  disabledReason?: string | null;
}) {
  const [dataUrl, setDataUrl] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!url || disabled) {
      setDataUrl(null);
      return;
    }
    void QRCode.toDataURL(url, { width: 420, margin: 1, color: { dark: "#0a0f16", light: "#ffffff" } })
      .then(setDataUrl)
      .catch(() => setDataUrl(null));
  }, [url, disabled]);

  return (
    <div className="flex flex-col rounded-lg border border-border-1 bg-bg-2 p-5">
      <div className="mb-1 flex items-center gap-2 text-[14px] font-bold text-text-primary">
        <Icon size={16} className="text-accent-blue" /> {title}
      </div>
      <p className="mb-4 text-[12px] leading-relaxed text-text-tertiary">{description}</p>

      {disabled || !url ? (
        <div className="flex flex-1 items-center gap-2 rounded border border-accent-amber/40 bg-accent-amber/10 px-3 py-3 text-[12px] text-accent-amber">
          <ShieldAlert size={15} className="shrink-0" />
          <span>{disabledReason ?? "Not available."}</span>
        </div>
      ) : (
        <>
          <div className="mb-3 flex justify-center rounded-lg bg-white p-3">
            {dataUrl ? (
              <img src={dataUrl} alt={`QR code for ${title}`} className="w-full max-w-[230px]" />
            ) : (
              <div className="flex h-[230px] w-full items-center justify-center text-[12px] text-black/50">
                generating…
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <code className="min-w-0 flex-1 truncate rounded border border-border-1 bg-bg-1 px-2 py-1.5 font-mono text-[10.5px] text-text-tertiary">
              {url}
            </code>
            <button
              onClick={() => {
                void navigator.clipboard.writeText(url).then(() => {
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
          {copied && <div className="mt-1 text-[11px] text-accent-green">Copied</div>}
        </>
      )}
    </div>
  );
}

/**
 * One place to connect a phone, without typing an address.
 *
 * Links are built from the address the *server* is reachable at, not from the
 * browser's URL - an operator working on localhost would otherwise generate a
 * QR code that sends the phone back to itself.
 */
export function Devices() {
  const [info, setInfo] = useState<ConnectInfo | null>(null);
  const [phoneCamera, setPhoneCamera] = useState<PhoneCamera | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setInfo(await api.get<ConnectInfo>("/api/live/connect-info"));
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not read network details");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function createPhoneCamera() {
    setBusy(true);
    try {
      const created = await api.post<PhoneCamera>("/api/live/phone-cameras", {
        name: `Phone camera ${new Date().toLocaleTimeString([], { hour12: false })}`,
      });
      setPhoneCamera(created);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not create a phone camera");
    } finally {
      setBusy(false);
    }
  }

  const cameraUrl =
    info?.secure_base_url && phoneCamera
      ? `${info.secure_base_url}${phoneCamera.join_path}`
      : null;

  return (
    <div className="flex flex-col gap-4 p-6 max-[1366px]:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-text-primary">Connect a Device</h1>
          <p className="mt-1 text-[12.5px] text-text-tertiary">
            Scan with a phone on the same network. No address to type in.
          </p>
        </div>
        <button
          onClick={() => void load()}
          className="flex items-center gap-1.5 rounded border border-border-2 px-3 py-1.5 text-[12px] text-text-secondary hover:bg-bg-3 hover:text-text-primary"
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {error && (
        <div className="rounded border border-accent-red/35 bg-accent-red/10 px-4 py-3 text-[12.5px] text-accent-red">
          {error}
        </div>
      )}

      {info && !info.on_network && (
        <div className="flex items-center gap-2 rounded border border-accent-amber/40 bg-accent-amber/10 px-4 py-3 text-[12.5px] text-accent-amber">
          <WifiOff size={15} />
          This machine has no network address, so no other device can reach it. Connect it to WiFi
          and refresh.
        </div>
      )}

      {info?.on_network && (
        <Panel title="This server">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-[12.5px]">
            <div>
              <span className="text-text-tertiary">Reachable at </span>
              <code className="font-mono text-text-primary">{info.base_url}</code>
            </div>
            <div>
              <span className="text-text-tertiary">Connection </span>
              <span className={info.scheme === "https" ? "text-accent-green" : "text-accent-amber"}>
                {info.scheme.toUpperCase()}
              </span>
            </div>
            {info.secure_base_url && info.secure_base_url !== info.base_url && (
              <div>
                <span className="text-text-tertiary">Secure (phone cameras) </span>
                <code className="font-mono text-accent-green">{info.secure_base_url}</code>
              </div>
            )}
            {info.addresses.length > 1 && (
              <div className="text-text-tertiary">
                Other addresses: {info.addresses.slice(1).join(", ")}
              </div>
            )}
          </div>
        </Panel>
      )}

      <div className="grid grid-cols-2 gap-4 max-[1000px]:grid-cols-1">
        <QrCard
          title="Post Siren"
          icon={Siren}
          description="Turns a phone into the alarm at the post. Scan, tap Arm, then add it to the home screen to keep it one tap away. No sign-in needed."
          url={info?.siren_url ?? null}
          disabled={!info?.on_network}
          disabledReason="This machine is not on a network."
        />

        <div className="flex flex-col rounded-lg border border-border-1 bg-bg-2 p-5">
          <div className="mb-1 flex items-center gap-2 text-[14px] font-bold text-text-primary">
            <Smartphone size={16} className="text-accent-blue" /> Phone as a Camera
          </div>
          <p className="mb-4 text-[12px] leading-relaxed text-text-tertiary">
            Streams the phone's camera into the analysis pipeline. Create a camera, scan, then tap
            Start streaming.
          </p>

          {!info?.camera_capable ? (
            <div className="flex flex-1 items-start gap-2 rounded border border-accent-amber/40 bg-accent-amber/10 px-3 py-3 text-[12px] text-accent-amber">
              <ShieldAlert size={15} className="mt-[1px] shrink-0" />
              <span>{info?.camera_warning ?? "Phone cameras require https."}</span>
            </div>
          ) : !phoneCamera ? (
            <button
              onClick={() => void createPhoneCamera()}
              disabled={busy || !info?.on_network}
              className="mt-auto flex items-center justify-center gap-2 rounded bg-accent-blue px-4 py-2.5 text-[12.5px] font-semibold text-on-accent disabled:opacity-40"
            >
              <Camera size={14} /> {busy ? "Creating…" : "Create phone camera"}
            </button>
          ) : (
            <QrCard
              title={phoneCamera.name}
              icon={Camera}
              description="Scan this, then tap Start streaming on the phone."
              url={cameraUrl}
            />
          )}
        </div>
      </div>

      {info?.on_network && (
        <QrCard
          title="Android Siren App (APK)"
          icon={Download}
          description="Scan to download and install the siren as a real app. Unlike the browser version it keeps the screen awake, shows over the lock screen, and vibrates - built for a phone left at a post."
          url={info.apk_url}
        />
      )}

      <Panel title="First time on a device">
        <ol className="flex list-decimal flex-col gap-1.5 pl-4 text-[12.5px] leading-relaxed text-text-secondary">
          <li>Make sure the phone is on the same WiFi as this machine.</li>
          <li>
            Scan the code. The browser will warn that the certificate is not trusted — that is
            expected for a local server. Choose <b>Advanced → Proceed</b>. Once per device.
          </li>
          <li>
            For the siren, tap <b>Arm siren</b>, then use the browser menu to{" "}
            <b>Add to Home Screen</b> so it opens like an app.
          </li>
          <li>Sessions last 30 days, so a device left in place will not ask you to sign in again.</li>
        </ol>
      </Panel>
    </div>
  );
}
