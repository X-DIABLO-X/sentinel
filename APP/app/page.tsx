"use client";

import Link from "next/link";
import { getDashboard } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import KpiCard from "@/components/KpiCard";
import EmptyState from "@/components/EmptyState";
import IncidentCard from "@/components/IncidentCard";
import JobTray from "@/components/JobTray";
import { eventLabel } from "@/lib/format";

/**
 * Overview — one consistent snapshot from GET /api/dashboard (summary +
 * incidents + cameras in a single scan, per api.py's own comment on that
 * route). Everything below reads from that one response; nothing here is
 * computed from a second, possibly-inconsistent fetch.
 */
export default function OverviewPage() {
  const { data, error, firstLoad, reload } = useApi(
    (signal) => getDashboard("cctv", signal),
    [],
    10000,
  );

  if (firstLoad) {
    return (
      <div className="p-5">
        <p className="text-[13px] text-ink-3">Loading dashboard…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-5">
        <EmptyState
          title="CCTV backend not reachable"
          tone="warn"
          detail={
            <>
              GET /api/dashboard failed: {error.message}. Start the backend with{" "}
              <code className="font-mono">python run.py serve</code> from{" "}
              <code className="font-mono">FINAL/CCTV</code>, or check{" "}
              <code className="font-mono">NEXT_PUBLIC_CCTV_API</code>.
            </>
          }
        >
          <button type="button" className="btn" onClick={reload}>
            Retry
          </button>
        </EmptyState>
      </div>
    );
  }

  const summary = data!.summary;
  const incidents = data!.incidents;
  const recent = [...incidents]
    .sort((a, b) => (Number(b.created_at) || 0) - (Number(a.created_at) || 0))
    .slice(0, 6);

  const bySeverity = summary.by_severity ?? {};
  const byType = summary.by_type ?? {};

  return (
    <div className="mx-auto max-w-[1400px] space-y-6 p-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <KpiCard label="Total incidents" value={summary.total_incidents} href="/incidents" />
        <KpiCard
          label="Open"
          value={summary.open_incidents}
          tone="accent"
          href="/incidents?status=detected"
          hint="not resolved/closed/rejected"
        />
        <KpiCard
          label="Awaiting verification"
          value={summary.awaiting_verification}
          tone="medium"
          href="/incidents?status=detected"
        />
        <KpiCard label="Low" value={bySeverity.Low ?? 0} tone="low" />
        <KpiCard label="Medium" value={bySeverity.Medium ?? 0} tone="medium" />
        <KpiCard label="High" value={bySeverity.High ?? 0} tone="high" />
      </div>

      <p className="note">
        {summary.severity_disclaimer} · {summary.cameras} camera{summary.cameras === 1 ? "" : "s"}{" "}
        configured.
      </p>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="panel lg:col-span-2">
          <div className="panel-head">
            <span className="panel-title">Recent incidents</span>
            <Link href="/incidents" className="text-[12.5px] text-accent hover:underline">
              View all →
            </Link>
          </div>
          <div className="space-y-2 p-3">
            {recent.length ? (
              recent.map((incident) => <IncidentCard key={incident.id} incident={incident} />)
            ) : (
              <EmptyState
                title="No incidents recorded yet"
                detail="Run the CCTV pipeline over a camera or upload a clip to populate this feed."
              />
            )}
          </div>
        </div>

        <div className="space-y-5">
          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">By event type</span>
            </div>
            <div className="space-y-1.5 p-3">
              {Object.keys(byType).length ? (
                Object.entries(byType)
                  .sort((a, b) => b[1] - a[1])
                  .map(([type, count]) => (
                    <div key={type} className="flex items-center justify-between text-[13px]">
                      <span className="text-ink-1">{eventLabel(type)}</span>
                      <span className="font-mono text-ink-2">{count}</span>
                    </div>
                  ))
              ) : (
                <p className="text-[12.5px] text-ink-3">No incidents yet.</p>
              )}
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Analysis jobs</span>
              <Link href="/upload" className="text-[12.5px] text-accent hover:underline">
                Upload →
              </Link>
            </div>
            <div className="p-3">
              <JobTray limit={3} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
