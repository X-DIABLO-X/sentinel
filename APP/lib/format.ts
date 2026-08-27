import type { EventType, Incident, Severity } from "./types";
import { normaliseSeverity } from "./types";

/** Seconds into the analysed clip -> "12.4s". Null-safe, never invents a 0. */
export function secs(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(digits)}s`;
}

/** A 0..1 score -> "0.72". Null-safe. */
export function num(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

/** Percent from a 0..1 score, clamped. */
export function pct(value: number | null | undefined): number {
  if (value === null || value === undefined || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(100, Number(value) * 100));
}

/** Unix epoch seconds (SQLite time.time()) -> local time string. */
export function clock(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "—";
  return new Date(epochSeconds * 1000).toLocaleString();
}

export function relative(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "—";
  const delta = Date.now() / 1000 - epochSeconds;
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

export function incidentRef(id: number): string {
  return `INC-${String(id).padStart(5, "0")}`;
}

const EVENT_LABELS: Record<string, string> = {
  queue: "Queue build-up",
  wrong_way: "Wrong-side movement",
  blockage: "Carriageway blockage",
  collision_candidate: "Suspected collision",
  stationary: "Stationary vehicle",
};

export function eventLabel(eventType: EventType | null | undefined): string {
  if (!eventType) return "Incident";
  return EVENT_LABELS[eventType] ?? String(eventType).replace(/_/g, " ");
}

export function severityOf(incident: Incident): Severity | null {
  return normaliseSeverity(incident.severity_label);
}

/** Human title for one incident row. */
export function incidentTitle(incident: Incident): string {
  return incident.label?.trim() || eventLabel(incident.event_type);
}

/** Flatten a triggers/severity_parts object into printable rows. */
export function kvRows(source: Record<string, unknown> | undefined | null): [string, string][] {
  if (!source) return [];
  return Object.entries(source)
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([key, value]) => {
      const label = key.replace(/_/g, " ");
      if (typeof value === "number") return [label, Number.isInteger(value) ? String(value) : value.toFixed(3)];
      if (typeof value === "boolean") return [label, value ? "yes" : "no"];
      if (typeof value === "string") return [label, value];
      return [label, JSON.stringify(value)];
    });
}
