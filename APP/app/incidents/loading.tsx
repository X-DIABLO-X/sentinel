import { SkeletonBlock, SkeletonHeader, SkeletonPanel } from "@/components/Skeleton";

/** Incident feed placeholder - tab strip, filter row, then cards. */
export default function Loading() {
  return (
    <div className="mx-auto max-w-[1200px] space-y-4 p-5">
      <SkeletonHeader />
      <div className="flex flex-wrap gap-1.5 border-b border-line pb-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <SkeletonBlock key={i} className="h-7 w-20" />
        ))}
      </div>
      <SkeletonPanel rows={6} />
    </div>
  );
}
