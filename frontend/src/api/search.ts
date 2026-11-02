import { api } from "./client";

export interface AppearanceMatch {
  id: number;
  camera_id: string;
  camera_name: string;
  track_id: number;
  kind: string;
  wall_time: number;
  score: number;
}

export interface AppearanceStatus {
  available: boolean;
  message?: string;
  sightings?: number;
  since?: number | null;
}

export const searchApi = {
  status: () => api.get<AppearanceStatus>("/api/search/appearance/status"),
  search: (q: string, kind: string, camera: string, hours: number) => {
    const params = new URLSearchParams({ q, hours: String(hours), limit: "36" });
    if (kind) params.set("kind", kind);
    if (camera) params.set("camera", camera);
    return api.get<{ query: string; results: AppearanceMatch[] }>(`/api/search/appearance?${params}`);
  },
  cropUrl: (id: number) => `/api/search/appearance/${id}/crop.jpg`,
};
