import { SkeletonCanvas, SkeletonHeader, SkeletonPanel } from "@/components/Skeleton";

/** CCTV placeholder - camera list beside the frame/clip panel. */
export default function Loading() {
  return (
    <div className="mx-auto max-w-[1300px] space-y-5 p-5">
      <SkeletonHeader />
      <div className="grid gap-5 lg:grid-cols-3">
        <SkeletonPanel rows={7} className="lg:col-span-1" />
        <div className="space-y-5 lg:col-span-2">
          <SkeletonCanvas className="h-[420px]" />
          <SkeletonPanel rows={2} />
        </div>
      </div>
    </div>
  );
}
