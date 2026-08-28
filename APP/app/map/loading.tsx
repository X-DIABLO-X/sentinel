import { SkeletonBlock, SkeletonHeader } from "@/components/Skeleton";

/** Map placeholder. The Leaflet canvas is the tall element here, so the
 *  placeholder matches its footprint to avoid a layout jump on arrival. */
export default function Loading() {
  return (
    <div className="mx-auto max-w-[1400px] space-y-4 p-5">
      <SkeletonHeader />
      <div className="grid gap-4 lg:grid-cols-3">
        <SkeletonBlock className="h-[560px] w-full lg:col-span-2" />
        <SkeletonBlock className="h-[560px] w-full" />
      </div>
    </div>
  );
}
