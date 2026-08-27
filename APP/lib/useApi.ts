"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";

export interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  /** True until the first attempt (success or failure) has completed. */
  firstLoad: boolean;
  reload: () => void;
}

/**
 * Small fetch-on-mount + optional-poll hook.
 *
 * Deliberately hand-rolled rather than pulling in SWR: the console only needs
 * one pattern (load, poll, expose an ApiError so an offline backend can be
 * named rather than silently rendered as empty data).
 *
 * `deps` must be a stable-length array of primitives - it is the dependency
 * list that re-triggers the fetch.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: ReadonlyArray<unknown> = [],
  pollMs = 0,
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [firstLoad, setFirstLoad] = useState(true);
  const [tick, setTick] = useState(0);

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const reload = useCallback(() => setTick((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);

    fetcherRef
      .current(controller.signal)
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled || controller.signal.aborted) return;
        setError(
          cause instanceof ApiError
            ? cause
            : new ApiError(String((cause as Error)?.message ?? cause), 0, "cctv", "?"),
        );
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
        setFirstLoad(false);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  useEffect(() => {
    if (pollMs <= 0) return;
    const timer = setInterval(reload, pollMs);
    return () => clearInterval(timer);
  }, [pollMs, reload]);

  return { data, error, loading, firstLoad, reload };
}
