"use client";

import { useMemo, useState } from "react";
import { cameraFrameUrl, getCameras } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import type { Camera, Corridor } from "@/lib/types";
import EmptyState from "@/components/EmptyState";

/** True once the scene has learned corridors AND a human has cleared the
 *  "DRAFT" note — mirrors CCTV/ui/calibrate.js `reviewed()` exactly. */
function reviewed(camera: Camera): boolean {
  return (camera.corridors?.length ?? 0) > 0 && !(camera.notes ?? "").toUpperCase().includes("DRAFT");
}

function flowArrow(corridor: Corridor, width: number, height: number) {
  const points = corridor.polygon ?? [];
  if (!points.length) return null;
  const cx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const cy = points.reduce((s, p) => s + p[1], 0) / points.length;
  const [dx, dy] = corridor.direction ?? [0, 0];
  const scale = Math.max(35, Math.min(width, height) * 0.08);
  return (
    <line
      x1={cx - dx * scale}
      y1={cy - dy * scale}
      x2={cx + dx * scale}
      y2={cy + dy * scale}
      stroke="#4da3ff"
      strokeWidth={2.5}
      markerEnd="url(#cal-arrow)"
    />
  );
}

const STREAM_COLOURS = ["#4da3ff", "#f0883e", "#3fb950", "#e5484d"];

/**
 * Camera onboarding review — the ~90-second calibration flow from
 * CCTV/ui/calibrate.html rebuilt in React. The backend does auto-calibration
 * as a CLI step (scripts/autocalibrate.py); this screen reviews and confirms
 * the geometry it produced, it does not trigger a fresh run over HTTP — see
 * lib/api.ts MISSING_ENDPOINTS.
 */
export default function CalibratePage() {
  const { data: cameras, error, firstLoad } = useApi((signal) => getCameras("cctv", signal), []);
  const [selectedId, setSelectedId] = useState<string>("");

  const camera = useMemo(
    () => cameras?.find((c) => c.camera_id === (selectedId || cameras[0]?.camera_id)) ?? null,
    [cameras, selectedId],
  );

  if (firstLoad) return <div className="p-5 text-[13px] text-ink-3">Loading cameras…</div>;

  if (error) {
    return (
      <div className="p-5">
        <EmptyState title="Cameras unavailable" tone="warn" detail={error.message} />
      </div>
    );
  }

  if (!cameras?.length) {
    return (
      <div className="p-5">
        <EmptyState
          title="No camera configurations found"
          detail={
            <>
              Run <code className="font-mono">python run.py calibrate --camera &lt;id&gt;</code> from{" "}
              <code className="font-mono">FINAL/CCTV</code> to auto-calibrate a source, then
              refresh this page to review it.
            </>
          }
        />
      </div>
    );
  }

  const [width, height] = camera?.frame_size ?? [1280, 720];
  const corridors = camera?.corridors ?? [];
  const isReviewed = camera ? reviewed(camera) : false;

  return (
    <div className="mx-auto max-w-[1300px] space-y-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-[18px] font-semibold text-ink-0">Camera &amp; lane geometry</h1>
          <p className="mt-1 text-[12.5px] text-ink-2">
            Observed motion streams support tracking. Legal direction remains draft until a human
            reviews the road geometry.
          </p>
        </div>
        <label>
          <span className="mb-1 block text-[11px] uppercase tracking-wider text-ink-3">Camera</span>
          <select
            className="field min-w-[240px]"
            value={camera?.camera_id ?? ""}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {cameras.map((c) => (
              <option key={c.camera_id} value={c.camera_id}>
                {c.name || c.camera_id} · {c.corridors?.length ?? 0} streams
              </option>
            ))}
          </select>
        </label>
      </div>

      {camera ? (
        <>
          <div className="flex flex-wrap gap-2">
            <span className={`pill ${corridors.length ? "border-sev-low/50 text-sev-low" : "border-sev-critical/50 text-sev-critical"}`}>
              {corridors.length} observed traffic streams
            </span>
            <span
              className={`pill ${isReviewed ? "border-sev-low/50 text-sev-low" : "border-sev-medium/50 text-sev-medium"}`}
            >
              {isReviewed ? "REVIEWED legal direction" : "DRAFT — legal direction not reviewed"}
            </span>
            <span className="pill">{width}×{height}</span>
            <span className="pill">{camera.incidents ?? 0} recorded incidents</span>
          </div>

          <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
            <div className="panel overflow-hidden">
              <div className="relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={cameraFrameUrl(camera.camera_id)}
                  alt={`First frame from ${camera.camera_id}`}
                  className="block w-full bg-panel-0"
                />
                <svg
                  viewBox={`0 0 ${width} ${height}`}
                  className="absolute inset-0 h-full w-full"
                  role="img"
                  aria-label="Observed traffic stream overlay"
                >
                  <defs>
                    <marker id="cal-arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L8,3 z" fill="#4da3ff" />
                    </marker>
                  </defs>
                  {corridors.map((c, i) => (
                    <g key={c.id}>
                      <polygon
                        points={(c.polygon ?? []).map((p) => p.join(",")).join(" ")}
                        fill={STREAM_COLOURS[i % STREAM_COLOURS.length]}
                        fillOpacity={0.18}
                        stroke={STREAM_COLOURS[i % STREAM_COLOURS.length]}
                        strokeWidth={2}
                      />
                      {flowArrow(c, width, height)}
                      <text
                        x={c.polygon?.[0]?.[0] ?? 8}
                        y={c.polygon?.[0]?.[1] ?? 18}
                        fill="#f2f5f8"
                        fontSize={14}
                        fontFamily="monospace"
                      >
                        {c.id}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
              <p className="note m-3">
                Polygon outlines are learned traffic streams, not pixel-perfect painted lanes.
                Arrows show observed majority flow.
              </p>
            </div>

            <aside className="space-y-4">
              <div className="panel px-4 py-3">
                <div className="panel-title mb-2">Geometry register</div>
                <table className="kv-table">
                  <tbody>
                    <tr><th>Camera ID</th><td>{camera.camera_id}</td></tr>
                    <tr><th>Zone</th><td>{camera.zone ?? "—"}</td></tr>
                    <tr><th>Road</th><td>{camera.road_name ?? "—"}</td></tr>
                    <tr><th>Analysis rate</th><td>{camera.analysis_fps ?? "—"} fps</td></tr>
                    <tr><th>Calibration</th><td>{isReviewed ? "reviewed" : "draft"}</td></tr>
                  </tbody>
                </table>
                {camera.notes ? <p className="note mt-2">{camera.notes}</p> : null}
              </div>

              <div className="panel overflow-hidden">
                <div className="panel-head"><span className="panel-title">Streams</span></div>
                <div className="overflow-x-auto">
                  <table className="kv-table w-full">
                    <thead>
                      <tr className="text-[11px] uppercase tracking-wide text-ink-3">
                        <th className="px-3 py-1.5 text-left">Stream</th>
                        <th className="px-3 py-1.5 text-left">Direction</th>
                        <th className="px-3 py-1.5 text-left">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {corridors.length ? (
                        corridors.map((c) => {
                          const deg = (Math.atan2(c.direction?.[1] ?? 0, c.direction?.[0] ?? 0) * 180) / Math.PI;
                          return (
                            <tr key={c.id}>
                              <th className="font-mono">{c.id}</th>
                              <td>{deg.toFixed(0)}°</td>
                              <td>{isReviewed ? "legal" : "observed"}</td>
                            </tr>
                          );
                        })
                      ) : (
                        <tr>
                          <td colSpan={3} className="px-3 py-3 text-ink-3">
                            No usable motion corridor learned.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="warn-note">
                <strong>Wrong-way assurance.</strong> NETRA can flag movement opposite to a stream
                only after its direction is reviewed as the legal direction. Draft auto-calibration
                alone is not legal evidence.
              </div>
            </aside>
          </div>
        </>
      ) : null}
    </div>
  );
}
