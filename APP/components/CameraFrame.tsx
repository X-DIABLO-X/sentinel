"use client";

import { useState } from "react";
import { cameraFrameUrl } from "@/lib/api";
import type { Backend } from "@/lib/types";

/**
 * GET /api/cameras/{id}/frame - the calibration still, used on /cctv and
 * /calibrate. The backend reads the camera's configured source video and
 * 404s with a real reason ("could not read a frame from the source") when
 * that file is not present on this host - which, for a checkout that does
 * not carry the full raw-footage dataset, is the common case, not an edge
 * case. A bare <img> would just show the browser's broken-image icon with
 * no explanation; this renders the actual backend reason instead.
 *
 * Pass `key={cameraId}` from the caller when the same mounted spot cycles
 * through different cameras, so the failed-state resets per camera.
 */
export default function CameraFrame({
  cameraId,
  alt,
  className = "",
  backend = "cctv",
}: {
  cameraId: string;
  alt: string;
  className?: string;
  backend?: Backend;
}) {
  const [failed, setFailed] = useState<string | null>(null);

  if (failed) {
    return (
      <div
        className={`flex min-h-[220px] flex-col items-center justify-center gap-1.5 bg-panel-0 px-5 py-8 text-center ${className}`}
      >
        <p className="text-[12.5px] font-medium text-ink-2">First frame unavailable</p>
        <p className="max-w-sm text-[11.5px] leading-relaxed text-ink-3">
          GET {cameraFrameUrl(cameraId, backend)} failed: {failed}. The source video for{" "}
          <code className="font-mono">{cameraId}</code> is not present on this backend host.
        </p>
      </div>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={cameraFrameUrl(cameraId, backend)}
      alt={alt}
      className={className}
      onError={async () => {
        try {
          const res = await fetch(cameraFrameUrl(cameraId, backend));
          const body = await res.json().catch(() => null);
          setFailed((body && body.detail) || `${res.status} ${res.statusText}`);
        } catch {
          setFailed("backend not reachable");
        }
      }}
    />
  );
}
