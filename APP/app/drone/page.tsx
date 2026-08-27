"use client";

import { DRONE_API, getDashboard, getHealth } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { DRONE_STATUS } from "@/lib/types";
import EmptyState from "@/components/EmptyState";
import KpiCard from "@/components/KpiCard";
import IncidentCard from "@/components/IncidentCard";

/**
 * Same shape as the CCTV page, pointed at the DRONE backend (port 8011 by
 * default). FINAL/DRONE is currently an empty skeleton with no api.py, so
 * every panel here is expected to render its offline state at the jury demo
 * until a real drone backend answers this same route contract. The detector
 * has not been fine-tuned on VisDrone — that fact is shown up front, always,
 * regardless of whether the backend is reachable, because it is a model-
 * readiness fact, not a connectivity fact.
 */
export default function DronePage() {
  const { data: health, error: healthError, firstLoad: healthLoading } = useApi(
    (signal) => getHealth("drone", signal),
    [],
    15000,
  );
  const { data: dashboard, error: dashError, firstLoad: dashLoading } = useApi(
    (signal) => getDashboard("drone", signal),
    [],
    15000,
  );

  const reachable = Boolean(health && !healthError);

  return (
    <div className="mx-auto max-w-[1300px] space-y-5 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[18px] font-semibold text-ink-0">Drone</h1>
        <span className="pill">{DRONE_API}</span>
      </div>

      <EmptyState
        title={`Detector not yet fine-tuned on ${DRONE_STATUS.finetuneTarget}`}
        tone="stub"
        detail={DRONE_STATUS.note}
      />

      <div className="note">
        Hover-based escalation, not continuous patrol: endurance is 20–40 min and DGCA regulation
        does not approve BVLOS for traffic monitoring, so the drone is dispatched to an
        already-confirmed CCTV incident to verify and hold position — see the CCTV{" "}
        incident it responds to for the original detection.
      </div>

      {healthLoading ? (
        <p className="text-[13px] text-ink-3">Checking DRONE backend…</p>
      ) : !reachable ? (
        <EmptyState
          title="DRONE backend not reachable"
          tone="warn"
          detail={
            <>
              GET {DRONE_API}/api/health failed
              {healthError ? `: ${healthError.message}` : ""}. FINAL/DRONE currently ships only{" "}
              <code className="font-mono">config/</code>, <code className="font-mono">demo/</code>,{" "}
              <code className="font-mono">models/detector/</code>,{" "}
              <code className="font-mono">results/</code> and <code className="font-mono">scripts/</code>{" "}
              — no <code className="font-mono">api.py</code> yet. Once a DRONE FastAPI process
              mirrors the CCTV route contract on this port, this page starts rendering live.
            </>
          }
        />
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <KpiCard label="Cameras / feeds" value={health?.cameras ?? 0} />
          <KpiCard label="Road-graph edges" value={health?.road_graph_edges ?? 0} />
          <KpiCard label="Status" value={health?.status ?? "—"} tone="accent" />
        </div>
      )}

      {reachable ? (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Escalation incidents</span>
          </div>
          <div className="space-y-2 p-3">
            {dashLoading ? (
              <p className="text-[12.5px] text-ink-3">Loading…</p>
            ) : dashError ? (
              <EmptyState title="Dashboard unavailable" tone="warn" detail={dashError.message} />
            ) : dashboard?.incidents.length ? (
              dashboard.incidents.map((incident) => (
                <IncidentCard key={incident.id} incident={incident} />
              ))
            ) : (
              <EmptyState title="No drone escalations recorded yet" />
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
