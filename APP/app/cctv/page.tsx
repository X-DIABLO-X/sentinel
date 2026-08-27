"use client";

import { useState } from "react";
import Link from "next/link";
import {
  demoPosterUrl,
  demoVideoUrl,
  getCameras,
  getDemoVideos,
  getIncidents,
  getProblemVideos,
  problemPosterUrl,
  problemVideoUrl,
} from "@/lib/api";
import { useApi } from "@/lib/useApi";
import IncidentCard from "@/components/IncidentCard";
import EmptyState from "@/components/EmptyState";
import CameraFrame from "@/components/CameraFrame";

/**
 * Pick a camera, see its calibration still + its recent event feed. The
 * pipeline is offline/batch (see lib/api.ts MISSING_ENDPOINTS) — there is no
 * live MJPEG/WebRTC stream, so "pipeline output" here means the calibration
 * still plus the incidents that camera has produced, which is what the
 * backend genuinely has. The fixed ProblemSet review clips (if the results
 * directory is present) are shown underneath as pre-rendered pipeline output.
 */
export default function CctvPage() {
  const [cameraId, setCameraId] = useState<string>("");

  const { data: cameras, error: camerasError, firstLoad: camerasLoading } = useApi(
    (signal) => getCameras("cctv", signal),
    [],
  );

  const active = cameraId || cameras?.[0]?.camera_id || "";

  const { data: incidents, firstLoad: incidentsLoading } = useApi(
    (signal) => (active ? getIncidents({ camera_id: active, limit: 100 }, "cctv", signal) : Promise.resolve([])),
    [active],
    active ? 15000 : 0,
  );

  const { data: problemVideos, firstLoad: problemLoading } = useApi(
    (signal) => getProblemVideos("cctv", signal),
    [],
  );

  const { data: demoVideos, firstLoad: demoLoading } = useApi(
    (signal) => getDemoVideos("cctv", signal),
    [],
  );

  const camera = cameras?.find((c) => c.camera_id === active) ?? null;

  /**
   * The physics render for the selected camera, when one exists.
   *
   * Renders are named after the source clip (ACCIDENTS_11 -> 11.mp4 ->
   * 11_physics.mp4), so match on the camera id's trailing segment rather than
   * hardcoding a table — a newly rendered clip then shows up on its camera
   * with no frontend change. Only the clips actually returned by
   * /api/demo-videos are considered, so this can never point at a file that
   * isn't really there.
   */
  const cameraClip = (() => {
    if (!camera || !demoVideos?.length) return null;
    const tail = camera.camera_id.split("_").pop();
    if (!tail) return null;
    return demoVideos.find((v) => v.stem === `${tail}_physics`) ?? null;
  })();

  return (
    <div className="mx-auto max-w-[1300px] space-y-5 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[18px] font-semibold text-ink-0">CCTV</h1>
      </div>

      {camerasLoading ? (
        <p className="text-[13px] text-ink-3">Loading cameras…</p>
      ) : camerasError ? (
        <EmptyState title="Cameras unavailable" tone="warn" detail={camerasError.message} />
      ) : !cameras?.length ? (
        <EmptyState title="No cameras configured" detail="Run calibration to register a camera." />
      ) : (
        <div className="grid gap-5 lg:grid-cols-3">
          <div className="panel lg:col-span-1">
            <div className="panel-head">
              <span className="panel-title">Cameras ({cameras.length})</span>
            </div>
            <div className="max-h-[560px] overflow-y-auto p-2.5">
              <div className="space-y-1.5">
                {cameras.map((c) => (
                  <button
                    key={c.camera_id}
                    type="button"
                    onClick={() => setCameraId(c.camera_id)}
                    className={[
                      "block w-full rounded-md border px-3 py-2 text-left text-[12.5px] transition-colors",
                      c.camera_id === active
                        ? "border-accent bg-accent/[0.08] text-ink-0"
                        : "border-line bg-panel-1 text-ink-2 hover:border-line/80 hover:bg-panel-2",
                    ].join(" ")}
                  >
                    <div className="truncate font-medium">{c.name || c.camera_id}</div>
                    <div className="mt-0.5 truncate font-mono text-[10.5px] text-ink-3">
                      {c.road_name || c.zone || "no location"} · {c.corridors?.length ?? 0} corridors ·{" "}
                      {c.incidents ?? 0} incidents
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-5 lg:col-span-2">
            {camera ? (
              <>
                <div className="panel overflow-hidden">
                  <div className="panel-head">
                    <span className="panel-title">{camera.name || camera.camera_id}</span>
                    <span className="pill">
                      {camera.corridors?.length ?? 0} corridors · {camera.zones?.length ?? 0} zones
                    </span>
                  </div>
                  {cameraClip ? (
                    <video
                      key={cameraClip.stem}
                      className="block w-full bg-panel-0"
                      src={demoVideoUrl(cameraClip.stem)}
                      poster={demoPosterUrl(cameraClip.stem)}
                      controls
                      muted
                      playsInline
                      preload="metadata"
                    />
                  ) : (
                    <CameraFrame
                      key={camera.camera_id}
                      cameraId={camera.camera_id}
                      alt={`First frame from ${camera.camera_id}`}
                      className="block w-full bg-panel-0"
                    />
                  )}
                  <div className="border-t border-line px-4 py-2.5 text-[11.5px] text-ink-3">
                    {cameraClip ? (
                      <>
                        Physics-annotated analysis of this camera&apos;s clip — per-vehicle speed,
                        acceleration, momentum and trajectory, drawn by{" "}
                        <code className="font-mono">rotation_gate.py</code>. Playback of a
                        pre-rendered analysis, not a live stream.
                      </>
                    ) : (
                      <>
                        First frame from the source. No live stream, and no physics render exists
                        for this camera yet — the analysed clips are listed below.
                      </>
                    )}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-head">
                    <span className="panel-title">
                      Event feed{incidentsLoading ? "" : ` (${incidents?.length ?? 0})`}
                    </span>
                  </div>
                  <div className="max-h-[420px] space-y-2 overflow-y-auto p-3">
                    {incidentsLoading ? (
                      <p className="text-[12.5px] text-ink-3">Loading…</p>
                    ) : incidents?.length ? (
                      incidents.map((incident) => (
                        <IncidentCard key={incident.id} incident={incident} />
                      ))
                    ) : (
                      <EmptyState title="No incidents recorded for this camera" />
                    )}
                  </div>
                </div>
              </>
            ) : (
              <EmptyState title="Select a camera" />
            )}
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">
            Physics demo — rotation-gate engine{demoLoading ? "" : ` (${demoVideos?.length ?? 0})`}
          </span>
        </div>
        <div className="p-4">
          {demoLoading ? (
            <p className="text-[12.5px] text-ink-3">Loading…</p>
          ) : !demoVideos?.length ? (
            <EmptyState
              title="No physics-render demo clips available"
              detail={
                <>
                  GET /api/demo-videos returned nothing — <code className="font-mono">CCTV/demo/</code>{" "}
                  has no <code className="font-mono">*_physics_result.json</code> files in this port of
                  the backend.
                </>
              }
            />
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {demoVideos.map((row) => (
                <div key={row.stem} className="panel overflow-hidden">
                  <video
                    className="block w-full bg-panel-0"
                    src={demoVideoUrl(row.stem)}
                    poster={demoPosterUrl(row.stem)}
                    controls
                    muted
                    playsInline
                  />
                  <div className="px-3 py-2 text-[11.5px] text-ink-2">
                    <div className="truncate font-mono">{row.file}</div>
                    <div className="text-ink-3">
                      {row.collision_score != null
                        ? `score ${row.collision_score.toFixed(3)} · ${row.interaction ?? "?"} ·
                           ${row.collision_confident ? "CONFIDENT" : "below threshold"}`
                        : "no collision candidate"}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Fixed review set (ProblemSet)</span>
        </div>
        <div className="p-4">
          {problemLoading ? (
            <p className="text-[12.5px] text-ink-3">Loading…</p>
          ) : !problemVideos?.length ? (
            <EmptyState
              title="No ProblemSet results available"
              detail={
                <>
                  GET /api/problem-videos returned nothing.{" "}
                  <code className="font-mono">CCTV/ProblemSet/Results_release_candidate/</code> is
                  not present in this port of the backend, so this panel is empty rather than
                  showing fabricated review clips.
                </>
              }
            />
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {problemVideos.map((row) => {
                const stem = row.file ? row.file.replace(/\.[^/.]+$/, "") : "";
                return (
                  <div key={stem} className="panel overflow-hidden">
                    <video
                      className="block w-full bg-panel-0"
                      src={problemVideoUrl(stem)}
                      poster={problemPosterUrl(stem)}
                      controls
                      muted
                      playsInline
                    />
                    <div className="px-3 py-2 text-[11.5px] text-ink-2">
                      <div className="truncate font-mono">{row.file}</div>
                      <div className="text-ink-3">{row.events_total ?? 0} events</div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
