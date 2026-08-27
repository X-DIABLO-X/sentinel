"use client";

import { use } from "react";
import Link from "next/link";
import { getIncident } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import EvidencePanel from "@/components/EvidencePanel";
import StatusWorkflow from "@/components/StatusWorkflow";
import SeverityBadge from "@/components/SeverityBadge";
import EmptyState from "@/components/EmptyState";
import {
  clock,
  eventLabel,
  incidentRef,
  incidentTitle,
  kvRows,
  num,
  relative,
  secs,
} from "@/lib/format";

export default function IncidentDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const incidentId = Number(id);

  const { data: incident, error, firstLoad, reload } = useApi(
    (signal) => getIncident(incidentId, "cctv", signal),
    [incidentId],
  );

  if (firstLoad) {
    return <div className="p-5 text-[13px] text-ink-3">Loading incident…</div>;
  }

  if (error || !incident) {
    return (
      <div className="p-5">
        <EmptyState
          title={`Incident ${incidentRef(incidentId)} not found`}
          tone="warn"
          detail={<>GET /api/incidents/{incidentId} failed: {error?.message ?? "not found"}</>}
        >
          <Link href="/incidents" className="btn">
            Back to incident feed
          </Link>
        </EmptyState>
      </div>
    );
  }

  const loc = incident.location ?? {};
  const measurementRows: [string, string][] = [
    ["Camera", incident.camera_id],
    ["Corridor", incident.corridor_id ?? "—"],
    ["Road", loc.road_name ?? "—"],
    ["Zone", loc.zone ?? "—"],
    ["Location precision", loc.precision ?? "—"],
    ["Road edge (routing graph)", loc.road_edge_id ?? "not mapped"],
    ["Onset method", incident.onset_method ?? "—"],
    ["Onset recovered", secs(incident.onset_recovered_s)],
    ["Started t", secs(incident.started_t)],
    ["Detected t", secs(incident.detected_t)],
    ["Ended t", secs(incident.ended_t)],
    ["Duration", secs(incident.duration)],
    ["Detection delay", secs(incident.detection_delay)],
    ["Confidence", num(incident.confidence)],
    ["Severity score", num(incident.severity)],
    ["Priority", incident.priority !== undefined && incident.priority !== null ? String(incident.priority) : "—"],
    ["Track IDs", incident.track_ids?.length ? incident.track_ids.join(", ") : "—"],
    ["Source", incident.source_kind ?? "—"],
    ["Created", clock(incident.created_at)],
    ["Updated", clock(incident.updated_at)],
  ];

  const triggerRows = kvRows(incident.triggers);
  const severityRows = kvRows(incident.severity_parts as Record<string, unknown> | undefined);

  return (
    <div className="mx-auto max-w-[1400px] space-y-5 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Link href="/incidents" className="text-[12.5px] text-accent hover:underline">
            ← Incident feed
          </Link>
          <h1 className="mt-1 text-[20px] font-semibold text-ink-0">{incidentTitle(incident)}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 font-mono text-[12px] text-ink-3">
            <span>{incidentRef(incident.id)}</span>
            <span>·</span>
            <span>{eventLabel(incident.event_type)}</span>
            <span>·</span>
            <span>detected {relative(incident.created_at)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <SeverityBadge severity={incident.severity_label} score={incident.severity} />
          <span className="pill">{incident.status}</span>
        </div>
      </div>

      {incident.needs_verification ? (
        <div className="warn-note">
          Needs human verification.
          {incident.event_type === "wrong_way" && incident.triggers?.direction_source
            ? ` ${String(incident.triggers.direction_source)}.`
            : ""}
        </div>
      ) : null}

      {incident.explanation ? <p className="note">{incident.explanation}</p> : null}
      {incident.recommended_action ? (
        <div className="panel px-4 py-3">
          <div className="panel-title mb-1">Recommended action</div>
          <p className="text-[13.5px] text-ink-1">{incident.recommended_action}</p>
        </div>
      ) : null}

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Evidence</span>
            </div>
            <div className="p-4">
              <EvidencePanel incident={incident} />
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Measurements</span>
            </div>
            <div className="overflow-x-auto p-4">
              <table className="kv-table">
                <tbody>
                  {measurementRows.map(([k, v]) => (
                    <tr key={k}>
                      <th>{k}</th>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {triggerRows.length ? (
            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Triggers</span>
              </div>
              <div className="overflow-x-auto p-4">
                <table className="kv-table">
                  <tbody>
                    {triggerRows.map(([k, v]) => (
                      <tr key={k}>
                        <th>{k}</th>
                        <td>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          {severityRows.length ? (
            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Severity components</span>
              </div>
              <div className="overflow-x-auto p-4">
                <table className="kv-table">
                  <tbody>
                    {severityRows.map(([k, v]) => (
                      <tr key={k}>
                        <th>{k}</th>
                        <td>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          {incident.history?.length ? (
            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Status history</span>
              </div>
              <div className="overflow-x-auto p-4">
                <table className="kv-table">
                  <tbody>
                    {incident.history.map((h, i) => (
                      <tr key={h.id ?? i}>
                        <th>{clock(h.changed_at)}</th>
                        <td>
                          {h.old_status ?? "—"} → {h.new_status} · {h.actor ?? "—"}
                          {h.reason ? ` · ${h.reason}` : ""}
                          {h.comment ? ` · ${h.comment}` : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>

        <div className="panel h-fit">
          <div className="panel-head">
            <span className="panel-title">Actions</span>
          </div>
          <div className="p-4">
            <StatusWorkflow incident={incident} onChanged={reload} />
          </div>
        </div>
      </div>
    </div>
  );
}
