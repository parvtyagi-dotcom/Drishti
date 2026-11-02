/**
 * Audible alert for the operator's own screen.
 *
 * Synthesised with the Web Audio API rather than loaded from an audio file:
 * there is no asset to ship or fail to download, and a swept two-tone wail
 * carries across a room better than most notification sounds.
 *
 * Browsers refuse to play audio until the user has interacted with the page,
 * so this can fail through no fault of the caller. It resolves either way and
 * reports whether sound actually started, because a caller that believes it
 * alerted someone when it did not is worse than one that knows it failed.
 */

const DEFAULT_SECONDS = 2.5;
const SWEEP_STEP_SEC = 0.55;
const LOW_HZ = 600;
const HIGH_HZ = 1100;

let context: AudioContext | null = null;

function getContext(): AudioContext | null {
  if (context) return context;
  const Ctor = window.AudioContext ?? (window as unknown as {
    webkitAudioContext?: typeof AudioContext;
  }).webkitAudioContext;
  if (!Ctor) return null;
  try {
    context = new Ctor();
    return context;
  } catch {
    return null;
  }
}

/**
 * Sound the alert. Returns false when the browser blocked audio - typically
 * because the user has not interacted with the page yet.
 */
export async function playSiren(durationSec: number = DEFAULT_SECONDS): Promise<boolean> {
  const audio = getContext();
  if (!audio) return false;

  try {
    if (audio.state === "suspended") await audio.resume();
    if (audio.state !== "running") return false;

    const oscillator = audio.createOscillator();
    const gain = audio.createGain();
    oscillator.type = "sawtooth";
    oscillator.connect(gain).connect(audio.destination);

    const start = audio.currentTime;
    // Fade in and out rather than switching on: an abrupt start and stop
    // produces an audible click on most speakers.
    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.5, start + 0.04);
    gain.gain.setValueAtTime(0.5, start + durationSec - 0.12);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + durationSec);

    // Schedule the whole sweep up front so it keeps wailing even if the tab's
    // timers are throttled.
    oscillator.frequency.setValueAtTime(LOW_HZ, start);
    const steps = Math.ceil(durationSec / SWEEP_STEP_SEC);
    for (let i = 0; i < steps; i++) {
      const at = start + (i + 1) * SWEEP_STEP_SEC;
      oscillator.frequency.linearRampToValueAtTime(i % 2 === 0 ? HIGH_HZ : LOW_HZ, at);
    }

    oscillator.start(start);
    oscillator.stop(start + durationSec);
    return true;
  } catch {
    return false;
  }
}

/**
 * Let later alarms play: browsers only allow audio after a user gesture, so
 * resume the audio context on the operator's first click or key press
 * anywhere, rather than waiting for them to find the "enable sound" banner.
 */
export function unlockAudioOnFirstGesture(): void {
  const unlock = () => {
    const audio = getContext();
    if (audio && audio.state === "suspended") void audio.resume();
    window.removeEventListener("pointerdown", unlock);
    window.removeEventListener("keydown", unlock);
  };
  window.addEventListener("pointerdown", unlock);
  window.addEventListener("keydown", unlock);
}

/** Short double-buzz where one is available. Silent on desktop, by design. */
export function vibrateAlert(): void {
  try {
    navigator.vibrate?.([300, 120, 300]);
  } catch {
    /* unsupported; the sound is the primary signal */
  }
}

/**
 * A camera going off is a fault, not an intrusion, so it gets its own short
 * sound: three quick falling beeps, nothing like the wailing siren.
 */
export async function playOfflineTone(): Promise<boolean> {
  const audio = getContext();
  if (!audio) return false;
  try {
    if (audio.state === "suspended") await audio.resume();
    if (audio.state !== "running") return false;
    const start = audio.currentTime;
    [880, 660, 440].forEach((hz, i) => {
      const at = start + i * 0.18;
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      osc.type = "square";
      osc.frequency.setValueAtTime(hz, at);
      osc.connect(gain).connect(audio.destination);
      gain.gain.setValueAtTime(0.0001, at);
      gain.gain.exponentialRampToValueAtTime(0.3, at + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, at + 0.14);
      osc.start(at);
      osc.stop(at + 0.15);
    });
    return true;
  } catch {
    return false;
  }
}

/** The right sound for an alarm: the short tone for a camera going off, the siren otherwise. */
export function playAlarmSound(type: string | undefined): Promise<boolean> {
  return type === "CAMERA_OFFLINE" ? playOfflineTone() : playSiren();
}
