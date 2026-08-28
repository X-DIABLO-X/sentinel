"use client";

import {
  DRONE_API,
  droneClipPosterUrl,
  droneClipVideoUrl,
  getDashboard,
  getDroneClips,
  getHealth,
} from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { DRONE_STATUS } from "@/lib/types";
import EmptyState from "@/components/EmptyState";
import KpiCard from "@/components/KpiCard";
import IncidentCard from "@/components/IncidentCard";
import { SkeletonKpis, SkeletonPanel } from "@/components/Skeleton";

/**
 * Same shape as the CCTV page, pointed at the DRONE backend (port 8011 by
 * default). A real api.py now answers this route contract — DRONE_STATUS
 * (lib/types.ts) is shown up front, always, regardless of whether the
 * backend happens to be reachable at page-load time, because it is a
 * model-readiness fact, not a connectivity fact.
 */
export default function DronePage() {
  const { data: health, error: healthError, firstLoad: healthLoading } = useApi(
    (signal) => getHealth("drone", signal),
    [],
    15000,
  );
  const { data: dashboard, error: dashError, firstLoad: dashLoading } = useApi(
    (signal) => getDashboard("drone", signal),
    [],
    15000,
  );

  const { data: clips, error: clipsError, firstLoad: clipsLoading } = useApi(
    (signal) => getDroneClips(signal),
    [],
  );

  const reachable = Boolean(health && !healthError);

  return (
    <div className="mx-auto max-w-[1300px] space-y-5 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[18px] font-semibold text-ink-0">Drone</h1>
        <span className="pill">{DRONE_API}</span>
      </div>

      <EmptyState
        title={
          DRONE_STATUS.detectorFineTuned
            ? `Detector fine-tuned on ${DRONE_STATUS.finetuneTarget} — real batch results below`
            : `Detector not yet fine-tuned on ${DRONE_STATUS.finetuneTarget}`
        }
        tone={DRONE_STATUS.detectorFineTuned ? "ok" : "stub"}
        detail={DRONE_STATUS.note}
      />

      <div className="note">
        Hover-based escalation, not continuous patrol: endurance is 20–40 min and DGCA regulation
        does not approve BVLOS for traffic monitoring, so the drone is dispatched to an
        already-confirmed CCTV incident to verify and hold position — see the CCTV{" "}
        incident it responds to for the original detection.
      </div>

      {healthLoading ? (
        <SkeletonKpis />
      ) : !reachable ? (
        <EmptyState
          title="DRONE backend not reachable"
          tone="warn"
          detail={
            <>
              GET {DRONE_API}/api/health failed
              {healthError ? `: ${healthError.message}` : ""}. The DRONE FastAPI process (
              <code className="font-mono">FINAL/DRONE/scripts/api.py</code>) is not answering on
              this port — start it with{" "}
              <code className="font-mono">python scripts/api.py</code> from{" "}
              <code className="font-mono">FINAL/DRONE</code>. Once it is up, this page starts
              rendering live.
            </>
          }
        />
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <KpiCard label="Status" value={health?.status ?? "—"} tone="accent" />
          <KpiCard label="Mode" value={health?.mode ?? "—"} />
          <KpiCard
            label="Detector"
            value={health?.detector_finetuned ? "fine-tuned" : "placeholder"}
            tone={health?.detector_finetuned ? "low" : "medium"}
          />
          <KpiCard
            label="GMC"
            value={health?.gmc_enabled ? "enabled" : "disabled"}
            tone={health?.gmc_enabled ? "low" : "neutral"}
          />
        </div>
      )}

      {reachable ? (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Escalation incidents</span>
          </div>
          <div className="space-y-2 p-3">
            {dashLoading ? (
              <SkeletonPanel rows={4} />
            ) : dashError ? (
              <EmptyState title="Dashboard unavailable" tone="warn" detail={dashError.message} />
            ) : dashboard?.incidents.length ? (
              dashboard.incidents.map((incident) => (
                <IncidentCard key={incident.id} incident={incident} backend="drone" />
              ))
            ) : (
              <EmptyState title="No drone escalations recorded yet" />
            )}
          </div>
        </div>
      ) : null}

      {reachable ? (
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">
              Analysed segments{clipsLoading ? "" : ` (${clips?.length ?? 0})`}
            </span>
          </div>
          <div className="p-4">
            {clipsLoading ? (
              <SkeletonPanel rows={3} />
            ) : clipsError ? (
              <EmptyState title="Segments unavailable" tone="warn" detail={clipsError.message} />
            ) : clips?.length ? (
              <>
                <p className="mb-3 text-[11.5px] text-ink-3">
                  Every segment the pipeline processed, including the ones that produced no
                  event — a clean segment is a real result, not a gap.
                </p>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {clips.map((clip) => (
                    <div key={clip.stem} className="panel overflow-hidden">
                      <video
                        className="block w-full bg-panel-0"
                        src={droneClipVideoUrl(clip.stem)}
                        poster={droneClipPosterUrl(clip.stem)}
                        controls
                        muted
                        playsInline
                        preload="none"
                      />
                      <div className="px-3 py-2 text-[11.5px] text-ink-2">
                        <div className="truncate font-mono">{clip.source_video ?? clip.file}</div>
                        <div className="text-ink-3">
                          {clip.track_count ?? 0} tracks ·{" "}
                          {clip.queue_events ? `${clip.queue_events} queue` : "no queue"} ·{" "}
                          {clip.blockage_events ? `${clip.blockage_events} blockage` : "no blockage"}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <EmptyState
                title="No analysed segments on this backend"
                detail={
                  <>
                    GET /api/clips returned nothing —{" "}
                    <code className="font-mono">DRONE/results/</code> holds no results JSON on
                    this host.
                  </>
                }
              />
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
