import { SkeletonBlock, SkeletonPanel } from "@/components/Skeleton";

/** Incident detail placeholder - title block, evidence pair, side actions. */
export default function Loading() {
  return (
    <div className="mx-auto max-w-[1400px] space-y-5 p-5">
      <div className="space-y-2">
        <SkeletonBlock className="h-3 w-28" />
        <SkeletonBlock className="h-6 w-96 max-w-full" />
        <SkeletonBlock className="h-3 w-72 max-w-full" />
      </div>
      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <div className="panel overflow-hidden">
            <div className="panel-head">
              <SkeletonBlock className="h-3.5 w-24" />
            </div>
            <div className="grid gap-4 p-4 lg:grid-cols-2">
              <SkeletonBlock className="h-56 w-full" />
              <SkeletonBlock className="h-56 w-full" />
            </div>
          </div>
          <SkeletonPanel rows={5} />
        </div>
        <SkeletonPanel rows={4} className="h-fit" />
      </div>
    </div>
  );
}
