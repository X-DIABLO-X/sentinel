"use client";

import { CCTV_API, DRONE_API, getHealth } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import type { Backend } from "@/lib/types";

function Dot({ ok, pending }: { ok: boolean; pending: boolean }) {
  const colour = pending ? "bg-ink-3" : ok ? "bg-sev-low" : "bg-sev-critical";
  return <span className={`inline-block h-1.5 w-1.5 rounded-full ${colour}`} />;
}

function Indicator({ backend, url }: { backend: Backend; url: string }) {
  // 15s poll: enough to notice a backend dying mid-demo, cheap enough that it
  // never competes with the page's own data fetches.
  const { data, error, firstLoad } = useApi(
    (signal) => getHealth(backend, signal),
    [backend],
    15000,
  );
  const ok = Boolean(data && !error);

  return (
    <span
      title={
        ok
          ? `${url} — ${data?.cameras ?? 0} cameras, ${data?.road_graph_edges ?? 0} road-graph edges`
          : `${url} — ${error?.message ?? "checking"}`
      }
      className="inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wide text-ink-3"
    >
      <Dot ok={ok} pending={firstLoad} />
      {backend}
      <span className={ok ? "text-ink-2" : "text-sev-critical"}>
        {firstLoad ? "…" : ok ? "up" : "down"}
      </span>
    </span>
  );
}

/**
 * Live reachability of both FastAPI processes. Two separate processes by
 * design, so two separate lights - a green CCTV light must never imply the
 * drone backend is answering.
 */
export default function BackendStatus() {
  return (
    <div className="ml-auto flex items-center gap-4">
      <Indicator backend="cctv" url={CCTV_API} />
      <Indicator backend="drone" url={DRONE_API} />
    </div>
  );
}
