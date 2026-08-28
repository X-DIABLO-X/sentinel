import { SkeletonBlock, SkeletonHeader, SkeletonKpis, SkeletonPanel } from "@/components/Skeleton";

/** Drone placeholder - status banner, KPI row, escalation list. */
export default function Loading() {
  return (
    <div className="mx-auto max-w-[1300px] space-y-5 p-5">
      <SkeletonHeader />
      <SkeletonBlock className="h-24 w-full" />
      <SkeletonKpis />
      <SkeletonPanel rows={4} />
    </div>
  );
}
