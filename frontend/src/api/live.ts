import { api } from "./client";

export type CameraState = "connecting" | "online" | "reconnecting" | "offline" | "stopped";
export type EventSeverity = "INFO" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface LiveCamera {
  camera_id: string;
  name: string;
  source_kind: "rtsp" | "file" | "webcam" | "phone";
  loop: boolean;
  rotate?: number;
  target_fps: number;
  max_width: number;
  location: string | null;
  state: CameraState;
  frames_read: number;
  frames_dropped: number;
  reconnect_count: number;
  frame_age_sec: number | null;
  measured_fps: number;
  width: number;
  height: number;
  source_fps: number;
  error: string | null;
  degraded: boolean;
  running: boolean;
  uptime_sec: number | null;
  modules: string[];
  reference_set_id: string | null;
  target_plate: string | null;
  zone_ids: string[];
  analysis_error: string | null;
  /** Frames per second the analysis gets through; null when no module is on. */
  analysis_fps?: number | null;
  event_count: number;
}

/** A rectangle on the ground with its real size: lets behaviour analytics
 * measure in metres. Points are normalized, in order around the rectangle. */
export interface GroundCalibration {
  calibrated: boolean;
  points?: [number, number][];
  width_m?: number;
  depth_m?: number;
}

export interface LiveEvent {
  event_id?: string;
  type: string;
  track_id: number | null;
  frame_index: number;
  timestamp_sec: number;
  zone?: string;
  camera_id: string;
  camera_name: string;
  severity: EventSeverity;
  reason: string;
  alarm: boolean;
  wall_time: number;
  data?: Record<string, unknown>;
  plate?: string;
  confidence?: number;
  /** Set when the alarm matched a pattern guards marked false before. */
  likely_false?: { marks: number; around: string; note: string | null } | null;
}

export interface AnalysisConfig {
  person_id: boolean;
  reference_set_id?: string | null;
  anpr: boolean;
  target_plate?: string | null;
  zone_ids: string[];
  behavior: boolean;
  lowlight: boolean;
  fog?: boolean;
}

export interface EvidenceBundle {
  bundle_id: string;
  camera_id: string;
  created_at: number;
  event_type: string | null;
  severity: EventSeverity | null;
  reason: string | null;
  camera_name: string | null;
  has_snapshot: boolean;
  has_clip: boolean;
  /** "unavailable" means assembly finished with nothing to record - not that it is still running. */
  clip_state: "ready" | "pending" | "unavailable";
  clip_size_bytes: number;
}

/** Evidence assets are served as files, used directly as <img>/<video> src. */
export function evidenceAssetUrl(bundle: EvidenceBundle, asset: "snapshot.jpg" | "clip.mp4"): string {
  return `/api/live/evidence/${bundle.camera_id}/${bundle.bundle_id}/${asset}`;
}

export const liveApi = {
  listEvidence: (limit = 50) => api.get<EvidenceBundle[]>(`/api/live/evidence?limit=${limit}`),
  listCameras: () => api.get<LiveCamera[]>("/api/live/cameras"),
  getCamera: (id: string) => api.get<LiveCamera>(`/api/live/cameras/${id}`),
  start: (id: string) => api.post<LiveCamera>(`/api/live/cameras/${id}/start`),
  stop: (id: string) => api.post<LiveCamera>(`/api/live/cameras/${id}/stop`),
  setQuality: (id: string, mode: "accuracy" | "balanced" | "speed") =>
    api.post<LiveCamera>(`/api/live/cameras/${id}/quality`, { mode }),
  rotate: (id: string, degrees: number) => api.post<LiveCamera>(`/api/live/cameras/${id}/rotate`, { degrees }),
  configureAnalysis: (id: string, config: AnalysisConfig) =>
    api.put<LiveCamera>(`/api/live/cameras/${id}/analysis`, config),
  createZone: (id: string, zone: { label: string; shape: "rectangle" | "polygon"; points: number[][] }) =>
    api.post<{ zone_id: string; label: string }>(`/api/live/cameras/${id}/zones`, zone),
  getCalibration: (id: string) => api.get<GroundCalibration>(`/api/live/cameras/${id}/calibration`),
  setCalibration: (id: string, c: { points: [number, number][]; width_m: number; depth_m: number }) =>
    api.put<GroundCalibration>(`/api/live/cameras/${id}/calibration`, c),
  clearCalibration: (id: string) => api.del<GroundCalibration>(`/api/live/cameras/${id}/calibration`),
  talkdownStatus: () => api.get<{ auto: boolean; speakers: Record<string, number> }>("/api/live/talkdown"),
  setTalkdownAuto: (auto: boolean) =>
    api.put<{ auto: boolean; speakers: Record<string, number> }>("/api/live/talkdown", { auto }),
  warnNow: (id: string) => api.post<{ reason: string; speakers: number }>(`/api/live/cameras/${id}/talkdown`),
  verdict: (eventId: string, verdict: "false" | "real", note?: string) =>
    api.post<{ verdict: string; pattern_marks: number; forgotten: number }>(
      `/api/live/events/${eventId}/verdict`, { verdict, note: note || null }),
  recentEvents: (limit = 100) => api.get<LiveEvent[]>(`/api/live/events?limit=${limit}`),
  onvifScan: () => api.post<OnvifScanResult>("/api/live/onvif/scan"),
  onvifProfiles: (login: OnvifLogin) => api.post<OnvifProfile[]>("/api/live/onvif/profiles", login),
  onvifAdd: (req: OnvifLogin & { profile_token: string; name: string; location: string | null }) =>
    api.post<LiveCamera>("/api/live/onvif/add", req),
  webcamScan: () => api.get<{ webcams: Webcam[] }>("/api/live/webcams"),
};

export interface Webcam {
  index: number;
  width: number;
  height: number;
  /** data: URL of one frame, so the operator can see which camera this is. */
  snapshot: string | null;
  /** Name of the camera already using this webcam, if any. */
  in_use_by: string | null;
  /** Opens only through Media Foundation: a phone-webcam app such as DroidCam, not a USB webcam. */
  virtual: boolean;
}

export interface OnvifCamera {
  address: string;
  xaddr: string;
  name: string | null;
  hardware: string | null;
}

export interface OnvifScanResult {
  interfaces: string[];
  cameras: OnvifCamera[];
}

export interface OnvifLogin {
  xaddr: string;
  username: string;
  password: string;
}

export interface OnvifProfile {
  token: string;
  name: string;
  width: number | null;
  height: number | null;
  encoding: string | null;
}

/** MJPEG feed URL. Used directly as an <img> src - the browser renders a
 * multipart stream natively, so no player library is involved. The cache
 * buster keeps a remounted <img> from reusing a dead connection. */
export function streamUrl(cameraId: string, fps = 12, nonce?: number): string {
  const bust = nonce === undefined ? "" : `&t=${nonce}`;
  return `/api/live/cameras/${cameraId}/stream?fps=${fps}${bust}`;
}

export function snapshotUrl(cameraId: string, nonce?: number): string {
  return `/api/live/cameras/${cameraId}/snapshot${nonce === undefined ? "" : `?t=${nonce}`}`;
}
