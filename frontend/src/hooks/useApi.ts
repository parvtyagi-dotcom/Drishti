import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export type AsyncState<T> =
  | { status: "loading"; data?: undefined; error?: undefined }
  | { status: "error"; data?: undefined; error: string }
  | { status: "success"; data: T; error?: undefined };

/** One-shot data fetch with loading/error/success states and a manual
 * refetch - used for pages that load once (or on demand), as opposed to
 * useJobPolling's continuous polling for an in-progress job. */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = []): AsyncState<T> & { refetch: () => void } {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });
  const [tick, setTick] = useState(0);

  // The latest fetcher lives in a ref so the effect need not list it (callers
  // pass a fresh closure every render); `deps` decide when to refetch.
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  // Collapse the caller's deps array into a version number that only changes
  // when an element changes, giving the effect a static dependency list.
  const [tracked, setTracked] = useState({ deps, version: 0 });
  if (tracked.deps.length !== deps.length || tracked.deps.some((d, i) => !Object.is(d, deps[i]))) {
    setTracked({ deps, version: tracked.version + 1 });
  }
  const depsVersion = tracked.version;

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fetcherRef.current()
      .then((data) => {
        if (!cancelled) setState({ status: "success", data });
      })
      .catch((err) => {
        if (!cancelled) {
          setState({ status: "error", error: err instanceof ApiError ? err.message : "Something went wrong." });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [depsVersion, tick]);

  const refetch = useCallback(() => setTick((t) => t + 1), []);

  return { ...state, refetch };
}
