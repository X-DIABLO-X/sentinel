"use client";

import { useRef, useState } from "react";
import { uploadVideo } from "@/lib/api";
import type { Job } from "@/lib/types";
import JobTray from "@/components/JobTray";

/**
 * Upload a clip, then run the full NETRA pipeline over it.
 *
 * POST /api/jobs streams the RAW file body (not multipart) with the filename
 * in an `x-filename` header — see lib/api.ts uploadVideo(). The 2GB cap and
 * accepted extensions are enforced server-side (jobs.py safe_video_name); this
 * page only pre-filters by extension for a faster error, it does not
 * duplicate the server's real validation.
 */
const ACCEPTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"];

export default function UploadPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [state, setState] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [message, setMessage] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);

  function pick(f: File | null) {
    setFile(f);
    setState("idle");
    setMessage(null);
    setProgress(0);
  }

  function start() {
    if (!file) return;
    setState("uploading");
    setMessage(null);
    const { promise, abort } = uploadVideo(file, setProgress);
    abortRef.current = abort;
    promise
      .then((job: Job) => {
        setState("done");
        setMessage(`Queued as ${job.id} — see the job tray below for live progress.`);
      })
      .catch((cause: Error) => {
        setState("error");
        setMessage(cause.message);
      });
  }

  function cancel() {
    abortRef.current?.();
    setState("idle");
    setProgress(0);
  }

  const extensionOk = file
    ? ACCEPTED_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))
    : true;

  return (
    <div className="mx-auto max-w-[900px] space-y-6 p-5">
      <h1 className="text-[18px] font-semibold text-ink-0">Upload a clip</h1>
      <p className="text-[13px] leading-relaxed text-ink-2">
        Uploads run the same detection → tracking → event → severity pipeline as the fixed
        cameras. The backend auto-calibrates a scene model for the clip if it does not already
        recognise the source, so a first run on a brand-new clip takes longer.
      </p>

      <div
        className="panel flex flex-col items-center justify-center gap-3 border-dashed px-6 py-10 text-center"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const dropped = e.dataTransfer.files?.[0] ?? null;
          if (dropped) pick(dropped);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <>
            <div className="text-[14px] font-medium text-ink-0">{file.name}</div>
            <div className="font-mono text-[12px] text-ink-3">
              {(file.size / (1024 * 1024)).toFixed(1)} MB
            </div>
            {!extensionOk ? (
              <p className="text-[12px] text-sev-medium">
                Unusual extension for a video file — the backend may reject it. Accepted:{" "}
                {ACCEPTED_EXTENSIONS.join(", ")}.
              </p>
            ) : null}
          </>
        ) : (
          <>
            <div className="text-[14px] text-ink-1">Drop a video here, or</div>
          </>
        )}
        <div className="flex flex-wrap items-center justify-center gap-2">
          <button type="button" className="btn" onClick={() => inputRef.current?.click()}>
            Choose file
          </button>
          {file ? (
            <button
              type="button"
              className="btn btn-primary"
              disabled={state === "uploading"}
              onClick={start}
            >
              {state === "uploading" ? `Uploading… ${progress}%` : "Upload & analyse"}
            </button>
          ) : null}
          {state === "uploading" ? (
            <button type="button" className="btn btn-danger" onClick={cancel}>
              Cancel
            </button>
          ) : null}
        </div>
        {state === "uploading" ? (
          <div className="h-1 w-full max-w-sm overflow-hidden rounded-full bg-panel-3">
            <div
              className="h-full bg-accent transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        ) : null}
        {message ? (
          <p className={`text-[12.5px] ${state === "error" ? "text-sev-critical" : "text-sev-low"}`}>
            {message}
          </p>
        ) : null}
        <p className="max-w-md text-[11px] leading-relaxed text-ink-3">
          Up to 2 GB. The upload streams straight to disk; the analysis job starts once the file
          finishes writing.
        </p>
      </div>

      <div>
        <h2 className="mb-2 text-[13px] font-semibold uppercase tracking-wider text-ink-2">
          Job tray
        </h2>
        <JobTray pollMs={2000} />
      </div>
    </div>
  );
}
