import { SkeletonHeader, SkeletonKpis, SkeletonPanel } from "@/components/Skeleton";

/** Overview placeholder. Rendered by the App Router the instant a navigation
 *  to "/" starts, so the previous page never sits there looking frozen. */
export default function Loading() {
  return (
    <div className="mx-auto max-w-[1300px] space-y-5 p-5">
      <SkeletonHeader />
      <SkeletonKpis />
      <div className="grid gap-5 lg:grid-cols-2">
        <SkeletonPanel rows={4} />
        <SkeletonPanel rows={4} />
      </div>
    </div>
  );
}
