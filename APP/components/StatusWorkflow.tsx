"use client";

import { useState } from "react";
import {
  assignIncident,
  closeRoad,
  rejectIncident,
  reopenRoad,
  setIncidentStatus,
  verifyIncident,
} from "@/lib/api";
import type { Backend, Incident, IncidentStatus } from "@/lib/types";
import { REJECTION_REASONS, WORKFLOW } from "@/lib/types";

/**
 * The operator workflow, exactly as the backend models it:
 *   detected -> verified -> assigned -> responding -> resolved -> closed
 * with `rejected` as a terminal side branch that carries a REASON, because a
 * rejection is labelled data for the next threshold iteration, not a delete.
 *
 * Routes used (all real):
 *   POST  /api/incidents/{id}/verify
 *   POST  /api/incidents/{id}/assign        {owner, team}
 *   POST  /api/incidents/{id}/reject        {reason, actor, comment}
 *   PATCH /api/incidents/{id}/status        {status, actor, reason, comment}
 *   POST  /api/incidents/{id}/close_road | /reopen_road
 */
export default function StatusWorkflow({
  incident,
  backend = "cctv",
  onChanged,
}: {
  incident: Incident;
  backend?: Backend;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [owner, setOwner] = useState("");
  const [team, setTeam] = useState("");
  const [reason, setReason] = useState<string>(REJECTION_REASONS[0]);
  const [comment, setComment] = useState("");

  const rejected = incident.status === "rejected";
  const currentIndex = WORKFLOW.indexOf(incident.status as IncidentStatus);
  const nextStatus: IncidentStatus | null =
    !rejected && currentIndex >= 0 && currentIndex < WORKFLOW.length - 1
      ? WORKFLOW[currentIndex + 1]
      : null;

  const hasMappedRoad = Boolean(incident.location?.road_edge_id);

  async function run(key: string, action: () => Promise<unknown>, success?: string) {
    setBusy(key);
    setError(null);
    setNotice(null);
    try {
      await action();
      if (success) setNotice(success);
      onChanged();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="space-y-4">
      {/* --- the ladder ------------------------------------------------- */}
      <ol className="flex flex-wrap items-center gap-1">
        {WORKFLOW.map((step, index) => {
          const done = !rejected && currentIndex >= index;
          const isCurrent = !rejected && currentIndex === index;
          return (
            <li key={step} className="flex items-center gap-1">
              <span
                className={[
                  "rounded px-2 py-1 font-mono text-[11px] uppercase tracking-wide",
                  isCurrent
                    ? "bg-accent/20 text-accent"
                    : done
                      ? "bg-panel-3 text-ink-1"
                      : "bg-panel-2 text-ink-3",
                ].join(" ")}
              >
                {step}
              </span>
              {index < WORKFLOW.length - 1 ? <span className="text-ink-3">›</span> : null}
            </li>
          );
        })}
        {rejected ? (
          <li className="ml-2 rounded bg-sev-critical/15 px-2 py-1 font-mono text-[11px] uppercase tracking-wide text-sev-critical">
            rejected
          </li>
        ) : null}
      </ol>

      {/* --- transitions ------------------------------------------------ */}
      {backend === "drone" ? (
        <p className="text-[11.5px] leading-relaxed text-ink-3">
          Drone findings are recomputed from results JSON on every request, not stored in a
          database — there is no status to change here. Verify, assign and reject only exist on
          the CCTV backend, which owns the incident record this escalation responds to.
        </p>
      ) : (
      <>
      <div className="flex flex-wrap items-center gap-2">
        {incident.status === "detected" ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy !== null}
            onClick={() => run("verify", () => verifyIncident(incident.id, "operator", backend), "Marked verified.")}
          >
            {busy === "verify" ? "Verifying…" : "Verify"}
          </button>
        ) : null}

        {nextStatus && nextStatus !== "verified" ? (
          <button
            type="button"
            className="btn"
            disabled={busy !== null}
            onClick={() =>
              run(
                "advance",
                () => setIncidentStatus(incident.id, nextStatus, { reason: "operator advanced" }, backend),
                `Moved to ${nextStatus}.`,
              )
            }
          >
            {busy === "advance" ? "Working…" : `Move to ${nextStatus}`}
          </button>
        ) : null}

        {hasMappedRoad ? (
          <>
            <button
              type="button"
              className="btn"
              disabled={busy !== null}
              onClick={() =>
                run("close", () => closeRoad(incident.id, backend), "Carriageway closed in the road graph — routing now diverts around it.")
              }
              title="Removes this incident's road edge from the routing graph"
            >
              {busy === "close" ? "Closing…" : "Confirm road closed"}
            </button>
            <button
              type="button"
              className="btn"
              disabled={busy !== null}
              onClick={() => run("reopen", () => reopenRoad(incident.id, backend), "Road edge reopened.")}
            >
              {busy === "reopen" ? "Reopening…" : "Reopen road"}
            </button>
          </>
        ) : (
          <span className="text-[11.5px] text-ink-3">
            No mapped road edge on this incident — road closure is unavailable.
          </span>
        )}
      </div>

      {/* --- assign ----------------------------------------------------- */}
      <div className="panel px-3.5 py-3">
        <div className="panel-title mb-2">Assign</div>
        <div className="flex flex-wrap items-end gap-2">
          <label className="min-w-[150px] flex-1">
            <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">Owner</span>
            <input
              className="field"
              value={owner}
              onChange={(event) => setOwner(event.target.value)}
              placeholder="operator name"
            />
          </label>
          <label className="min-w-[150px] flex-1">
            <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">Team</span>
            <input
              className="field"
              value={team}
              onChange={(event) => setTeam(event.target.value)}
              placeholder="traffic police / ambulance / towing"
            />
          </label>
          <button
            type="button"
            className="btn"
            disabled={busy !== null || !owner.trim()}
            onClick={() =>
              run(
                "assign",
                () => assignIncident(incident.id, owner.trim(), team.trim(), backend),
                `Assigned to ${owner.trim()}.`,
              )
            }
          >
            {busy === "assign" ? "Assigning…" : "Assign"}
          </button>
        </div>
      </div>

      {/* --- reject ----------------------------------------------------- */}
      <div className="panel px-3.5 py-3">
        <div className="panel-title mb-2">Reject as a false alarm</div>
        <p className="mb-2 text-[11.5px] leading-relaxed text-ink-3">
          A rejection is not a deletion. The reason is stored as labelled data and is what the
          next iteration of the detection thresholds is tuned against.
        </p>
        <div className="flex flex-wrap items-end gap-2">
          <label className="min-w-[170px]">
            <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">Reason</span>
            <select className="field" value={reason} onChange={(event) => setReason(event.target.value)}>
              {REJECTION_REASONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="min-w-[180px] flex-1">
            <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">Comment</span>
            <input
              className="field"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              placeholder="optional"
            />
          </label>
          <button
            type="button"
            className="btn btn-danger"
            disabled={busy !== null}
            onClick={() =>
              run(
                "reject",
                () => rejectIncident(incident.id, reason, comment, "operator", backend),
                `Rejected: ${reason}.`,
              )
            }
          >
            {busy === "reject" ? "Rejecting…" : "Reject"}
          </button>
        </div>
      </div>
      </>
      )}

      {notice ? <div className="note text-sev-low">{notice}</div> : null}
      {error ? <div className="stub-note">Action failed: {error}</div> : null}
    </div>
  );
}
