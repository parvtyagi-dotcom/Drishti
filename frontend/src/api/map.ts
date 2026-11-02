import { api } from "./client";

export interface SiteSummary {
  id: string;
  name: string;
  has_image: boolean;
}

export interface SiteMap {
  site_id: string;
  site_name: string;
  has_image: boolean;
  image_version: number;
  /** Camera name → [x, y] as fractions of the plan's width and height. */
  positions: Record<string, [number, number]>;
  sites: SiteSummary[];
}

/** Every call takes the site to act on; without one the backend uses its default site. */
const q = (site?: string | null) => (site ? `site=${encodeURIComponent(site)}` : "");

export interface PathPrediction {
  from: [string, string];
  trail: string[];
  heading_deg: number;
  last_seen: number;
  basis: string;
  next_camera: string | null;
  /** Where to send the patrol, as fractions of the plan. */
  point: [number, number];
  eta_sec?: number;
  advice: string;
}

export const mapApi = {
  prediction: (site?: string | null) =>
    api.get<{ prediction: PathPrediction | null }>(`/api/live/map/prediction${site ? `?${q(site)}` : ""}`),
  get: (site?: string | null) => api.get<SiteMap>(`/api/live/map${site ? `?${q(site)}` : ""}`),
  place: (site: string | null, name: string, x: number | null, y: number | null) =>
    api.put<{ positions: SiteMap["positions"] }>(`/api/live/map/positions${site ? `?${q(site)}` : ""}`, { name, x, y }),
  uploadImage: (site: string | null, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return api.postForm<{ image_version: number }>(`/api/live/map/image${site ? `?${q(site)}` : ""}`, form);
  },
  imageUrl: (site: string, version: number) => `/api/live/map/image?${q(site)}&v=${version}`,
  addSite: (name: string) => api.post<{ id: string; sites: SiteSummary[] }>("/api/live/map/sites", { name }),
  renameSite: (id: string, name: string) => api.patch<{ sites: SiteSummary[] }>(`/api/live/map/sites/${id}`, { name }),
  removeSite: (id: string) => api.del<{ sites: SiteSummary[] }>(`/api/live/map/sites/${id}`),
};
