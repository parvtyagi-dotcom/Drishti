import { api } from "./client";

export interface Sensor {
  sensor_id: string;
  name: string;
  kind: string;
  label: string;
  /** Linked camera, by name. */
  camera: string | null;
  /** False for kinds a ground camera cannot confirm (acoustic drone). */
  checkable: boolean;
  created: number;
  /** Only returned to admins. */
  key?: string;
}

export interface SensorKind {
  kind: string;
  label: string;
}

export const sensorsApi = {
  list: () => api.get<{ sensors: Sensor[]; kinds: SensorKind[] }>("/api/sensors"),
  add: (name: string, kind: string, camera: string | null) =>
    api.post<Sensor>("/api/sensors", { name, kind, camera }),
  remove: (id: string) => api.del<{ removed: string }>(`/api/sensors/${id}`),
  test: (id: string) => api.post<{ status: string; event_type: string | null; reason: string | null }>(
    `/api/sensors/${id}/test`),
};

/** Event types the sensor fusion raises. */
export const SENSOR_EVENT_TYPES = new Set([
  "SENSOR_TRIGGERED", "SENSOR_ALARM_CONFIRMED", "SENSOR_ALARM_UNVERIFIED",
  "SENSOR_ALARM_UNCONFIRMED", "DRONE_SUSPECTED", "DIGGING_SUSPECTED",
]);
