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
  const [reloadKey, setReloadKey] = useState(0);

  const retry = () => {
    setFrameFailed(false);
    setClipFailed(false);
    setReloadKey((n) => n + 1);
  };

  /**
   * Evidence URLs carry a stable version token, not a bare path.
   *
   * These files were 404ing for a window before the evidence tree was
   * deployed, and browsers heuristically cache a 404 that carries no
   * Cache-Control. Any client that loaded a detail page during that window
   * holds a poisoned entry for the bare URL, and a media element will not
   * re-fetch a src it has already failed on -- which is exactly the reported
   * symptom: the clip fails on load, then works on Retry, because Retry was
   * the only thing changing the URL.
   *
   * `v` is derived from the incident's own run, so it is IDENTICAL on every
   * load (stays cacheable and fast -- unlike a random token, which would
   * defeat caching entirely) while being a different cache key from the
   * poisoned bare URL. `r` is appended only after an explicit Retry.
   */
  const version = String(incident.run_id ?? incident.updated_at ?? incident.id);
  const mediaUrl = (name: string) => {
    const base = `${evidenceUrl(incident.id, name, backend)}?v=${encodeURIComponent(version)}`;
    return reloadKey ? `${base}&r=${reloadKey}` : base;
  };

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
            <div className="space-y-2 px-4 py-6 text-[12.5px] text-ink-3">
              <p>
                Evidence frame <code className="font-mono">{frame}</code> could not be loaded from
                the {backend.toUpperCase()} backend.
              </p>
              <button type="button" className="btn" onClick={retry}>
                Retry
              </button>
            </div>
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={`frame-${reloadKey}`}
              src={mediaUrl(frame)}
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
            <div className="space-y-2 px-4 py-6 text-[12.5px] text-ink-3">
              <p>
                Evidence clip <code className="font-mono">{clip}</code> could not be loaded.
              </p>
              <button type="button" className="btn" onClick={retry}>
                Retry
              </button>
            </div>
          ) : (
            <video
              key={`clip-${reloadKey}`}
              src={mediaUrl(clip)}
              controls
              muted
              loop
              playsInline
              preload="metadata"
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
