import { api } from "./client";

export type Role = "ADMIN" | "OPERATOR" | "AUDITOR";

export interface SessionUser {
  username: string;
  role: Role;
  /** Present for Google sign-ins only. */
  name?: string;
  email?: string;
  picture?: string;
}

export interface SessionState {
  /** False before any account exists - the system needs an admin created. */
  configured: boolean;
  authenticated: boolean;
  user: SessionUser | null;
}

export interface AuditVerification {
  total_events: number;
  valid: boolean;
  broken_at_id: number | null;
  broken_event_id: string | null;
  detail: string | null;
  summary: string;
  anchor: { count: number; last_id: number; head_hash: string; updated_at: number } | null;
  guarantees: { detects: string[]; does_not_detect: string[]; note: string };
}

export const authApi = {
  me: () => api.get<SessionState>("/api/auth/me"),
  login: (username: string, password: string) =>
    api.post<{ token: string; user: SessionUser }>("/api/auth/login", { username, password }),
  google: (credential: string) =>
    api.post<{ token: string; user: SessionUser }>("/api/auth/google", { credential }),
  bootstrap: (username: string, password: string) =>
    api.post<{ token: string; user: SessionUser }>("/api/auth/bootstrap", { username, password }),
  logout: () => api.post<{ ok: boolean }>("/api/auth/logout"),
  listUsers: () => api.get<SessionUser[]>("/api/auth/users"),
  addUser: (username: string, password: string, role: Role) =>
    api.post<SessionUser>("/api/auth/users", { username, password, role }),
};

export const auditApi = {
  verify: () => api.get<AuditVerification>("/api/live/audit/verify"),
  log: (limit = 100, alarmsOnly = false) =>
    api.get<Record<string, unknown>[]>(
      `/api/live/audit?limit=${limit}${alarmsOnly ? "&alarms_only=true" : ""}`,
    ),
};
