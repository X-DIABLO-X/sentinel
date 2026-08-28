/**
 * Loading placeholders.
 *
 * These exist so a navigation never looks frozen. The API itself answers in
 * roughly 2-30ms, but the console is a client-rendered app talking to a
 * backend over the public internet, so there is always a window between the
 * click and the first byte of data. Without a placeholder the old page just
 * sits there and reads as a hang.
 *
 * Shapes deliberately mirror the real layout of the page they stand in for,
 * so the content does not jump when it arrives.
 */

export function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-panel-3/60 ${className}`} />;
}

/** One panel with a header strip and a few body rows. */
export function SkeletonPanel({
  rows = 3,
  className = "",
}: {
  rows?: number;
  className?: string;
}) {
  return (
    <div className={`panel overflow-hidden ${className}`}>
      <div className="panel-head">
        <SkeletonBlock className="h-3.5 w-32" />
      </div>
      <div className="space-y-2.5 p-3">
        {Array.from({ length: rows }).map((_, i) => (
          <SkeletonBlock key={i} className="h-14 w-full" />
        ))}
      </div>
    </div>
  );
}

/** A row of KPI tiles, as on the overview. */
export function SkeletonKpis({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="panel p-4">
          <SkeletonBlock className="mb-2.5 h-2.5 w-16" />
          <SkeletonBlock className="h-6 w-20" />
        </div>
      ))}
    </div>
  );
}

/** The page title strip every route opens with. */
export function SkeletonHeader() {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <SkeletonBlock className="h-5 w-40" />
      <SkeletonBlock className="h-5 w-28" />
    </div>
  );
}

/** A large media/canvas area - camera frame, map, evidence still. */
export function SkeletonCanvas({ className = "h-[360px]" }: { className?: string }) {
  return (
    <div className="panel overflow-hidden">
      <div className="panel-head">
        <SkeletonBlock className="h-3.5 w-40" />
      </div>
      <SkeletonBlock className={`w-full rounded-none ${className}`} />
    </div>
  );
}
