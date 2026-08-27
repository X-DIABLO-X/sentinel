"use client";

import { useState } from "react";
import Link from "next/link";
import { getDashboard, getGraph, getRoute } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import type { Incident, RouteResponse } from "@/lib/types";
import MapView, { RouteLegend } from "@/components/MapView";
import IncidentCard from "@/components/IncidentCard";
import EvidencePanel from "@/components/EvidencePanel";
import SeverityBadge from "@/components/SeverityBadge";
import EmptyState from "@/components/EmptyState";
import { eventLabel, incidentRef } from "@/lib/format";
import { electronicCityDemo } from "@/lib/electronicCityDemo";

/**
 * 3-column operator console: incident feed / centre map + run controls /
 * right detail panel. Concept carried over from TEST/app/frontend/index.html,
 * rebuilt in React against the real dashboard + graph + route endpoints.
 *
 * Route convention (kept in one place — see components/MapView RouteLegend):
 *   solid RED  = diversion route for other traffic
 *   dashed BLUE = simulated responder access route (NOT available — no
 *                 backend endpoint computes one; see lib/api.ts MISSING_ENDPOINTS)
 */
export default function MapPage() {
  const { data: dashboard, error: dashError, firstLoad: dashLoading } = useApi(
    (signal) => getDashboard("cctv", signal),
    [],
    15000,
  );
  const { data: graph } = useApi((signal) => getGraph("cctv", signal), [], 30000);

  const [selected, setSelected] = useState<Incident | null>(null);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [routeError, setRouteError] = useState<string | null>(null);
  const [routing, setRouting] = useState(false);
  const [showDemo, setShowDemo] = useState(true);
  const [congestionActive, setCongestionActive] = useState(false);

  const demo = showDemo ? electronicCityDemo(congestionActive) : null;

  async function computeRoute() {
    if (!source.trim() || !target.trim()) return;
    setRouting(true);
    setRouteError(null);
    try {
      const result = await getRoute(source.trim(), target.trim());
      setRoute(result);
      if (result.error) setRouteError(result.error);
    } catch (cause) {
      setRoute(null);
      setRouteError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setRouting(false);
    }
  }

  const incidents = dashboard?.incidents ?? [];
  const cameras = dashboard?.cameras ?? [];

  return (
    <div className="grid h-[calc(100vh-53px)] grid-cols-1 lg:grid-cols-[320px_1fr_360px]">
      {/* left: incident feed */}
      <div className="flex min-h-0 flex-col border-r border-line">
        <div className="border-b border-line px-4 py-3">
          <span className="panel-title">Incidents ({incidents.length})</span>
        </div>
        <div className="flex-1 overflow-y-auto p-2.5">
          {dashLoading ? (
            <p className="p-2 text-[12.5px] text-ink-3">Loading…</p>
          ) : dashError ? (
            <EmptyState
              title="Backend unavailable"
              tone="warn"
              detail={dashError.message}
            />
          ) : incidents.length === 0 ? (
            <EmptyState title="No incidents yet" />
          ) : (
            <div className="space-y-2">
              {incidents.map((incident) => (
                <IncidentCard
                  key={incident.id}
                  incident={incident}
                  selected={selected?.id === incident.id}
                  onSelect={setSelected}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* centre: map + run controls */}
      <div className="flex min-h-0 flex-col">
        <div className="min-h-0 flex-1">
          <MapView
            cameras={cameras}
            incidents={incidents}
            graph={graph}
            diversion={route?.current ?? null}
            selectedIncidentId={selected?.id ?? null}
            onSelectIncident={setSelected}
            demo={demo}
          />
        </div>
        <div className="border-t border-line bg-panel-1 p-4">
          <div className="rounded-md border border-accent/30 bg-accent/[0.06] p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="panel-title text-accent">Electronic City Phase 1 showcase</div>
                <p className="mt-1 max-w-2xl text-[12px] leading-relaxed text-ink-2">
                  Synthetic CCTV sites + nearby reference hospitals. The blue line is the shortest
                  reachable path in an illustrative local road graph, not live navigation or dispatch.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <button type="button" className="btn" onClick={() => setShowDemo((value) => !value)}>
                  {showDemo ? "Hide showcase" : "Show showcase"}
                </button>
                {showDemo ? (
                  <button
                    type="button"
                    className={congestionActive ? "btn btn-danger" : "btn btn-primary"}
                    onClick={() => setCongestionActive((value) => !value)}
                  >
                    {congestionActive ? "Clear congestion" : "Simulate congestion"}
                  </button>
                ) : null}
              </div>
            </div>
            {demo ? (
              <div className="mt-3 grid gap-2 text-[11.5px] text-ink-2 sm:grid-cols-3">
                <div><strong className="text-ink-1">1. Detect</strong><br />{demo.incident.id} at Neeladri Junction</div>
                <div><strong className="text-ink-1">2. Constrain</strong><br />{demo.incident.congestionActive ? "Congested approach excluded" : "No closure applied yet"}</div>
                <div><strong className="text-ink-1">3. Route</strong><br />{demo.route.facility.name} · {demo.route.distanceM.toLocaleString()} m · ~{demo.route.etaMinutes} min</div>
              </div>
            ) : null}
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <label className="min-w-[160px]">
              <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">
                Source node
              </span>
              <input
                className="field font-mono"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder="road-graph node id"
              />
            </label>
            <label className="min-w-[160px]">
              <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">
                Target node
              </span>
              <input
                className="field font-mono"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="road-graph node id"
              />
            </label>
            <button
              type="button"
              className="btn btn-primary"
              disabled={routing || !source.trim() || !target.trim()}
              onClick={computeRoute}
            >
              {routing ? "Routing…" : "Compute route"}
            </button>
            {route?.diverted ? (
              <span className="pill border-sev-medium/50 bg-sev-medium/10 text-sev-medium">
                diverted +{route.extra_seconds}s / +{route.extra_metres}m
              </span>
            ) : null}
            {routeError ? <span className="text-[12px] text-sev-critical">{routeError}</span> : null}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-ink-3">
            Advanced backend route: GET <code className="font-mono">/api/route</code> needs two
            road-graph node IDs. This remains separate from the Phase 1 showcase until a production
            road graph maps each incident to a real edge.
          </p>
          <div className="mt-2">
            <RouteLegend accessRouteAvailable={false} />
          </div>
        </div>
      </div>

      {/* right: detail panel */}
      <div className="flex min-h-0 flex-col border-l border-line">
        <div className="border-b border-line px-4 py-3">
          <span className="panel-title">Detail</span>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {selected ? (
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-[15px] font-semibold text-ink-0">
                    {eventLabel(selected.event_type)}
                  </h2>
                  <SeverityBadge severity={selected.severity_label} score={selected.severity} />
                </div>
                <div className="mt-1 font-mono text-[11.5px] text-ink-3">
                  {incidentRef(selected.id)} · {selected.camera_id}
                </div>
                <Link
                  href={`/incidents/${selected.id}`}
                  className="mt-2 inline-block text-[12.5px] text-accent hover:underline"
                >
                  Open full detail →
                </Link>
              </div>
              <EvidencePanel incident={selected} />
            </div>
          ) : (
            demo ? (
              <div className="space-y-3">
                <div>
                  <div className="panel-title">Showcase decision</div>
                  <h2 className="mt-1 text-[15px] font-semibold text-ink-0">{demo.incident.label}</h2>
                </div>
                <div className="panel px-3 py-3 text-[12px] leading-relaxed text-ink-2">
                  <p><strong className="text-ink-1">Recommended reference facility</strong><br />{demo.route.facility.name}</p>
                  <p className="mt-2"><strong className="text-ink-1">Route state</strong><br />{demo.incident.congestionActive ? "Congested edge removed; alternate path selected." : "Direct path selected before congestion."}</p>
                  <p className="mt-2 text-ink-3">This route is illustrative. Facility availability and dispatch are not inferred.</p>
                </div>
              </div>
            ) : (
              <EmptyState
                title="No incident selected"
                detail="Click an incident card or a map pin to see its evidence and detail here."
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}
