"use client";

import dynamic from "next/dynamic";
import type { MapCanvasProps } from "./MapCanvas";

/**
 * SSR-safe entry point for the map.
 *
 * react-leaflet v4 and leaflet itself reference `window` during module
 * evaluation, so under the App Router they must be loaded only in the browser.
 * `next/dynamic` with `ssr: false` is only legal inside a Client Component -
 * hence the "use client" directive at the top of THIS file, with the actual
 * leaflet code isolated in MapCanvas.tsx.
 *
 * Pages should import this component, never MapCanvas.
 */
const MapCanvas = dynamic(() => import("./MapCanvas"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center bg-panel-0 text-[13px] text-ink-3">
      Loading map…
    </div>
  ),
});

export default function MapView(props: MapCanvasProps) {
  return <MapCanvas {...props} />;
}

/**
 * The route-colour convention, kept in one place so the legend and the
 * polylines cannot drift apart.
 *
 *   SOLID RED   = diversion route other traffic should take
 *   DASHED BLUE = simulated responder access route
 *
 * They point in opposite directions and must never be confused on screen.
 */
export function RouteLegend({
  accessRouteAvailable = false,
}: {
  accessRouteAvailable?: boolean;
}) {
  return (
    <div className="space-y-1.5 text-[11.5px] text-ink-2">
      <div className="flex items-center gap-2">
        <svg width="34" height="6" aria-hidden="true">
          <line x1="0" y1="3" x2="34" y2="3" stroke="#e5484d" strokeWidth="4" />
        </svg>
        <span>
          <strong className="text-ink-1">Solid red</strong> — diversion route for other traffic
        </span>
      </div>
      <div className="flex items-center gap-2">
        <svg width="34" height="6" aria-hidden="true">
          <line
            x1="0"
            y1="3"
            x2="34"
            y2="3"
            stroke="#4da3ff"
            strokeWidth="3"
            strokeDasharray="7,6"
          />
        </svg>
        <span>
          <strong className="text-ink-1">Dashed blue</strong> — simulated responder access route
          {accessRouteAvailable ? null : (
            <em className="ml-1 not-italic text-sev-critical">
              (NOT AVAILABLE — no access-route endpoint exists in the backend)
            </em>
          )}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <svg width="34" height="6" aria-hidden="true">
          <line
            x1="0"
            y1="3"
            x2="34"
            y2="3"
            stroke="#e5484d"
            strokeWidth="4"
            strokeDasharray="2,6"
          />
        </svg>
        <span>
          <strong className="text-ink-1">Dotted red</strong> — carriageway an operator has
          confirmed closed
        </span>
      </div>
    </div>
  );
}
