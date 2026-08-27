"use client";

import Link from "next/link";
import type { Incident } from "@/lib/types";
import SeverityBadge from "./SeverityBadge";
import { eventLabel, incidentRef, num, relative, secs } from "@/lib/format";

const SOURCE_LABELS: Record<string, string> = {
  problem_set: "review set",
  upload: "uploaded clip",
  legacy_accident: "legacy accident set",
  traffic: "fixed camera",
};

export default function IncidentCard({
  incident,
  selected = false,
  /** When set, the card calls back instead of navigating (3-column console). */
  onSelect,
}: {
  incident: Incident;
  selected?: boolean;
  onSelect?: (incident: Incident) => void;
}) {
  const loc = incident.location ?? {};
  const where = loc.road_name || loc.zone || incident.camera_id;

  const inner = (
    <>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-[14px] font-semibold text-ink-0">
            {eventLabel(incident.event_type)}
          </div>
          <div className="mt-0.5 truncate font-mono text-[11px] text-ink-3">
            {incidentRef(incident.id)} · {incident.camera_id}
            {incident.corridor_id ? ` · ${incident.corridor_id}` : ""}
          </div>
        </div>
        <SeverityBadge severity={incident.severity_label} score={incident.severity} />
      </div>

      <div className="mt-2 truncate text-[12.5px] text-ink-2">{where}</div>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <span className="pill">{incident.status}</span>
        <span className="pill" title="Detector/engine confidence, never folded into severity">
          conf {num(incident.confidence)}
        </span>
        {incident.duration ? <span className="pill">dur {secs(incident.duration)}</span> : null}
        {incident.source_kind ? (
          <span className="pill">{SOURCE_LABELS[incident.source_kind] ?? incident.source_kind}</span>
        ) : null}
        {incident.needs_verification ? (
          <span className="pill border-sev-medium/50 bg-sev-medium/10 text-sev-medium">
            needs human verification
          </span>
        ) : null}
      </div>

      <div className="mt-2 font-mono text-[10.5px] text-ink-3">
        detected {relative(incident.created_at)} · t+{secs(incident.detected_t)} into clip
      </div>
    </>
  );

  const className = [
    "block w-full rounded-lg border px-3.5 py-3 text-left transition-colors",
    selected
      ? "border-accent bg-accent/[0.08]"
      : "border-line bg-panel-1 hover:border-line/80 hover:bg-panel-2",
  ].join(" ");

  if (onSelect) {
    return (
      <button type="button" className={className} onClick={() => onSelect(incident)}>
        {inner}
      </button>
    );
  }

  return (
    <Link href={`/incidents/${incident.id}`} className={className}>
      {inner}
    </Link>
  );
}
