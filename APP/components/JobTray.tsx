"use client";

import { getJobs, jobPosterUrl, jobVideoUrl } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import type { Backend, Job } from "@/lib/types";
import EmptyState from "./EmptyState";
import { relative } from "@/lib/format";

const STATUS_TONE: Record<string, string> = {
  uploading: "text-accent",
  queued: "text-ink-2",
  running: "text-accent",
  complete: "text-sev-low",
  failed: "text-sev-critical",
};

function ProgressBar({ percent, status }: { percent: number; status: string }) {
  const colour =
    status === "failed" ? "bg-sev-critical" : status === "complete" ? "bg-sev-low" : "bg-accent";
  return (
    <div className="h-1 w-full overflow-hidden rounded-full bg-panel-3">
      <div
        className={`h-full transition-all duration-300 ${colour}`}
        style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
      />
    </div>
  );
}

export function JobRow({ job, backend = "cctv" }: { job: Job; backend?: Backend }) {
  const finished = job.status === "complete";
  const result = job.result ?? null;

  return (
    <div className="panel px-3.5 py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-[13.5px] font-semibold text-ink-0">{job.filename}</div>
          <div className="mt-0.5 font-mono text-[11px] text-ink-3">
            {job.id}
            {job.camera_id ? ` · ${job.camera_id}` : ""} · {relative(job.updated_at)}
          </div>
        </div>
        <span className={`pill ${STATUS_TONE[job.status] ?? "text-ink-2"}`}>
          {job.phase || job.status} {job.percent}%
        </span>
      </div>

      <div className="mt-2">
        <ProgressBar percent={job.percent} status={job.status} />
      </div>

      {job.message ? <p className="mt-2 text-[12px] text-ink-2">{job.message}</p> : null}
      {job.error ? <p className="mt-2 text-[12px] text-sev-critical">{job.error}</p> : null}

      {finished && result ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          <span className="pill">{result.events_total ?? 0} incidents</span>
          {result.corridors !== undefined ? (
            <span className="pill">{result.corridors} corridors</span>
          ) : null}
          {result.calibrated ? <span className="pill">auto-calibrated</span> : null}
          {result.wall_seconds !== undefined ? (
            <span className="pill">{result.wall_seconds}s wall</span>
          ) : null}
          {result.collision_candidates_dropped ? (
            <span className="pill">{result.collision_candidates_dropped} candidates dropped</span>
          ) : null}
        </div>
      ) : null}

      {finished && job.annotated_video ? (
        <video
          className="mt-3 w-full rounded-md border border-line bg-panel-0"
          src={jobVideoUrl(job.id, backend)}
          poster={jobPosterUrl(job.id, backend)}
          controls
          muted
          playsInline
        />
      ) : null}
    </div>
  );
}

/**
 * Live job tray, polling GET /api/jobs.
 *
 * Jobs are held in memory by VideoJobManager and re-hydrated from
 * uploads/results on backend start, so restarting the backend mid-upload loses
 * the in-flight job but keeps completed ones.
 */
export default function JobTray({
  backend = "cctv",
  pollMs = 2000,
  limit,
}: {
  backend?: Backend;
  pollMs?: number;
  limit?: number;
}) {
  const { data, error, firstLoad } = useApi((signal) => getJobs(backend, signal), [backend], pollMs);

  if (firstLoad) {
    return <p className="text-[13px] text-ink-3">Loading analysis jobs…</p>;
  }

  if (error) {
    return (
      <EmptyState
        title="Job list unavailable"
        tone="warn"
        detail={<>GET /api/jobs failed: {error.message}</>}
      />
    );
  }

  const jobs = data ?? [];
  if (!jobs.length) {
    return (
      <EmptyState
        title="No analysis jobs yet"
        detail="Upload a clip to run the full pipeline over it. Jobs appear here with live progress."
      />
    );
  }

  const shown = limit ? jobs.slice(0, limit) : jobs;

  return (
    <div className="space-y-2.5">
      {shown.map((job) => (
        <JobRow key={job.id} job={job} backend={backend} />
      ))}
    </div>
  );
}
