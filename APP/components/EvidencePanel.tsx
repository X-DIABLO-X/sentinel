"use client";

import { useState } from "react";
import { evidenceUrl } from "@/lib/api";
import type { Backend, Incident } from "@/lib/types";
import EmptyState from "./EmptyState";
import { secs } from "@/lib/format";

/**
 * Visual evidence for one incident.
 *
 * Both media items come from GET /api/incidents/{id}/evidence/{name}, where
 * {name} is a bare filename recorded in the incident's evidence manifest
 * (evidence.py writes `annotated_frame` and `clip`). The backend transcodes
 * legacy mp4v clips to VP8/WebM on first request, so the first play of an old
 * clip can take a few seconds.
 */
export default function EvidencePanel({
  incident,
  backend = "cctv",
}: {
  incident: Incident;
  backend?: Backend;
}) {
  const evidence = incident.evidence ?? {};
  const frame = typeof evidence.annotated_frame === "string" ? evidence.annotated_frame : null;
  const clip = typeof evidence.clip === "string" ? evidence.clip : null;
  const span = Array.isArray(evidence.clip_span_s) ? evidence.clip_span_s : null;

  const [frameFailed, setFrameFailed] = useState(false);
  const [clipFailed, setClipFailed] = useState(false);

  if (!frame && !clip) {
    return (
      <EmptyState
        title="No visual evidence recorded"
        detail={
          <>
            This incident has no <code className="font-mono">annotated_frame</code> or{" "}
            <code className="font-mono">clip</code> in its evidence manifest. Evidence is written
            during analysis; incidents produced by an older run, or by a run with rendering
            disabled, carry only their numeric triggers.
            {evidence.dir ? (
              <>
                {" "}
                Manifest directory on the analysis host:{" "}
                <code className="font-mono text-ink-3">{String(evidence.dir)}</code>
              </>
            ) : null}
          </>
        }
      />
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {frame ? (
        <figure className="panel overflow-hidden">
          {frameFailed ? (
            <div className="px-4 py-6 text-[12.5px] text-ink-3">
              Evidence frame <code className="font-mono">{frame}</code> could not be loaded from
              the {backend.toUpperCase()} backend.
            </div>
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={evidenceUrl(incident.id, frame, backend)}
              alt={`Annotated evidence frame for incident ${incident.id}`}
              className="block w-full bg-panel-0"
              onError={() => setFrameFailed(true)}
            />
          )}
          <figcaption className="border-t border-line px-3 py-2 text-[11.5px] text-ink-3">
            Annotated frame — implicated tracks and motion at detection.
          </figcaption>
        </figure>
      ) : null}

      {clip ? (
        <figure className="panel overflow-hidden">
          {clipFailed ? (
            <div className="px-4 py-6 text-[12.5px] text-ink-3">
              Evidence clip <code className="font-mono">{clip}</code> could not be loaded. Legacy
              clips are transcoded to WebM on first request; retry once.
            </div>
          ) : (
            <video
              src={evidenceUrl(incident.id, clip, backend)}
              controls
              muted
              loop
              playsInline
              className="block w-full bg-panel-0"
              onError={() => setClipFailed(true)}
            />
          )}
          <figcaption className="border-t border-line px-3 py-2 text-[11.5px] text-ink-3">
            Evidence clip — spans the recovered onset
            {span ? ` (${secs(span[0])} → ${secs(span[1])})` : ""}.
          </figcaption>
        </figure>
      ) : null}
    </div>
  );
}
