"use client";

import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getIncidents } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import type { EventType, Incident, IncidentStatus } from "@/lib/types";
import { ALL_STATUSES } from "@/lib/types";
import IncidentCard from "@/components/IncidentCard";
import EmptyState from "@/components/EmptyState";
import { eventLabel } from "@/lib/format";

/** Tabs mirror the shipped NETRA UI's event-type tabs, plus an "All" and an
 *  "Uploads" tab (source_kind === "upload", not an event type). */
const EVENT_TABS: { key: EventType | "all" | "upload"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "queue", label: "Queues" },
  { key: "collision_candidate", label: "Accidents" },
  { key: "wrong_way", label: "Wrong-side" },
  { key: "blockage", label: "Blockage" },
  { key: "upload", label: "Uploads" },
];

function IncidentsInner() {
  const searchParams = useSearchParams();
  const [tab, setTab] = useState<(typeof EVENT_TABS)[number]["key"]>("all");
  const [status, setStatus] = useState<IncidentStatus | "">(
    (searchParams.get("status") as IncidentStatus | null) ?? "",
  );
  const [cameraId, setCameraId] = useState("");

  const { data, error, firstLoad, reload } = useApi(
    (signal) =>
      getIncidents(
        {
          status: status || undefined,
          event_type: tab !== "all" && tab !== "upload" ? tab : undefined,
          camera_id: cameraId || undefined,
          limit: 500,
        },
        "cctv",
        signal,
      ),
    [status, tab, cameraId],
    15000,
  );

  const incidents = data ?? [];
  const shown = useMemo(
    () => (tab === "upload" ? incidents.filter((i) => i.source_kind === "upload") : incidents),
    [incidents, tab],
  );

  const cameraOptions = useMemo(() => {
    const ids = new Set(incidents.map((i) => i.camera_id));
    return Array.from(ids).sort();
  }, [incidents]);

  return (
    <div className="mx-auto max-w-[1200px] space-y-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[18px] font-semibold text-ink-0">Incidents</h1>
        <button type="button" className="btn" onClick={reload}>
          Refresh
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5 border-b border-line pb-3">
        {EVENT_TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={[
              "rounded-md px-3 py-1.5 text-[13px] transition-colors",
              tab === t.key
                ? "bg-accent/15 text-accent"
                : "text-ink-2 hover:bg-panel-2 hover:text-ink-0",
            ].join(" ")}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="min-w-[160px]">
          <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">Status</span>
          <select
            className="field"
            value={status}
            onChange={(event) => setStatus(event.target.value as IncidentStatus | "")}
          >
            <option value="">All statuses</option>
            {ALL_STATUSES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className="min-w-[180px]">
          <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">Camera</span>
          <select className="field" value={cameraId} onChange={(event) => setCameraId(event.target.value)}>
            <option value="">All cameras</option>
            {cameraOptions.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <span className="pb-1.5 text-[12.5px] text-ink-3">{shown.length} shown</span>
      </div>

      {firstLoad ? (
        <p className="text-[13px] text-ink-3">Loading incidents…</p>
      ) : error ? (
        <EmptyState
          title="Incident feed unavailable"
          tone="warn"
          detail={<>GET /api/incidents failed: {error.message}</>}
        />
      ) : shown.length === 0 ? (
        <EmptyState
          title={`No incidents match ${eventLabel(tab === "all" ? undefined : (tab as EventType))}${
            status ? ` · ${status}` : ""
          }`}
          detail="Adjust the tab or filters above, or run the pipeline over more footage."
        />
      ) : (
        <div className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
          {shown.map((incident: Incident) => (
            <IncidentCard key={incident.id} incident={incident} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function IncidentsPage() {
  return (
    <Suspense fallback={<div className="p-5 text-[13px] text-ink-3">Loading…</div>}>
      <IncidentsInner />
    </Suspense>
  );
}
