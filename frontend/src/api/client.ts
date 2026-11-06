// Centralized fetch wrapper. Every API call in the app goes through this
// file (directly or via the resource modules alongside it) - no endpoint
// URLs are hardcoded elsewhere, no mock data is ever returned from here.

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    try {
      const text = await res.text();
      return text || res.statusText;
    } catch {
      return res.statusText || `HTTP ${res.status}`;
    }
  }
}

/** Fired on window when a non-auth endpoint answers 401 (session expired). */
export const UNAUTHORIZED_EVENT = "drishti:unauthorized";

// Endpoints where a 401 is an expected answer (wrong password, signed out).
const AUTH_PATHS = ["/api/auth/me", "/api/auth/login", "/api/auth/google", "/api/auth/bootstrap"];

type Options = RequestInit & { signal?: AbortSignal };

/** Resolves to the parsed JSON body, or undefined when the response has none. */
async function request<T>(path: string, init?: Options): Promise<T | undefined> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "Cannot reach the Drishti backend. Is the API running?");
  }
  if (!res.ok) {
    if (res.status === 401 && !AUTH_PATHS.some((p) => path.startsWith(p))) {
      window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
    }
    throw new ApiError(res.status, await parseErrorDetail(res));
  }
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return (await res.json()) as T;
  }
  return undefined;
}

/** For endpoints that must return a JSON body. */
async function requestJson<T>(path: string, init?: Options): Promise<T> {
  const body = await request<T>(path, init);
  if (body === undefined) throw new ApiError(502, "The server returned an empty response.");
  return body;
}

const jsonInit = (method: string, body: unknown, signal?: AbortSignal): Options => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
  signal,
});

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => requestJson<T>(path, { signal }),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    requestJson<T>(path, {
      method: "POST",
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    }),
  put: <T>(path: string, body?: unknown, signal?: AbortSignal) => requestJson<T>(path, jsonInit("PUT", body, signal)),
  patch: <T>(path: string, body: unknown, signal?: AbortSignal) => requestJson<T>(path, jsonInit("PATCH", body, signal)),
  postForm: <T>(path: string, form: FormData, signal?: AbortSignal) =>
    requestJson<T>(path, { method: "POST", body: form, signal }),
  del: <T>(path: string, signal?: AbortSignal) => requestJson<T>(path, { method: "DELETE", signal }),
  /** DELETE for endpoints that answer 204 No Content. */
  delVoid: async (path: string, signal?: AbortSignal): Promise<void> => {
    await request<unknown>(path, { method: "DELETE", signal });
  },
};

/** Absolute-ish URL for a media resource (video/image) served by the API -
 * used directly as an <img>/<video> src, never fetched+parsed as JSON. */
export function mediaUrl(path: string): string {
  return path;
}
