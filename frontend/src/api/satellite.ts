import { api } from "./client";

export interface SatellitePair {
  id: string;
  name: string;
  note?: string;
  bbox: number[];
  pixel_m: number;
  source: string;
  before_date: string;
  after_date: string;
  before_scene?: string;
  after_scene?: string;
  has_ndvi: boolean;
  has_map?: boolean;
  checked: boolean;
}

export interface SatelliteChange {
  id: number;
  kind: "vegetation_cleared" | "vegetation_growth" | "track" | "bare_ground" | "structure";
  reason: string;
  area_m2: number;
  box: [number, number, number, number];
  polygon: [number, number][];
  strength: number;
}

export interface SatelliteResult {
  pair_id: string;
  width: number;
  height: number;
  pixel_m: number;
  method: string;
  shift_px: [number, number];
  changed_area_m2: number;
  changed_fraction: number;
  total_changes: number;
  changes: SatelliteChange[];
  seconds: number;
  before_date: string;
  after_date: string;
}

export interface Place {
  name: string;
  lat: number;
  lon: number;
  kind: string;
}

export interface FetchJob {
  job_id: string;
  status: "running" | "done" | "failed";
  pair_id: string;
  log: string[];
  error: string | null;
}

export const satelliteApi = {
  places: (q: string) => api.get<{ places: Place[] }>(`/api/satellite/places?q=${encodeURIComponent(q)}`),
  fetch: (req: { name: string; lat: number; lon: number; before: string; after: string; size_km: number }) =>
    api.post<FetchJob>("/api/satellite/fetch", req),
  fetchStatus: (jobId: string) => api.get<FetchJob>(`/api/satellite/fetch/${encodeURIComponent(jobId)}`),
  pairs: () => api.get<{ pairs: SatellitePair[] }>("/api/satellite/pairs"),
  lastCheck: (id: string) => api.get<SatelliteResult>(`/api/satellite/pairs/${encodeURIComponent(id)}/check`),
  check: (id: string) => api.post<SatelliteResult>(`/api/satellite/pairs/${encodeURIComponent(id)}/check`),
  imageUrl: (id: string, which: "before" | "after" | "map_region" | "map_local") =>
    `/api/satellite/pairs/${encodeURIComponent(id)}/${which}.png`,
};
