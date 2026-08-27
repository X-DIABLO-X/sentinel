"use client";

import { useMemo } from "react";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
} from "react-leaflet";
import type { Camera, Incident, RoadGraphGeoJSON, RouteGeometry } from "@/lib/types";
import type { ElectronicCityDemoState } from "@/lib/electronicCityDemo";
import { normaliseSeverity } from "@/lib/types";
import { SEVERITY_HEX, UNKNOWN_SEVERITY_HEX } from "./SeverityBadge";
import { eventLabel, incidentRef } from "@/lib/format";

/**
 * Leaflet implementation. NEVER import this module directly from a page - it
 * touches `window` at module scope through leaflet. Import components/MapView
 * instead, which wraps it in next/dynamic with ssr:false.
 *
 * Two deliberate choices:
 *  - CircleMarker, not Marker. Leaflet's default Marker pulls PNG icons by
 *    relative URL, which breaks under Next's asset pipeline; a vector circle
 *    also colours cleanly by severity tier.
 *  - Coordinate order. GeoJSON from /api/graph is [lon, lat]; RouteGeometry
 *    `coords` from /api/route is already [lat, lon]. Leaflet wants [lat, lon],
 *    so only the graph is flipped. Getting this backwards puts India in Somalia.
 */

/** Fallback view when nothing on screen has coordinates: Electronic City
 *  Phase 1, Bengaluru — the deployment's default operating area. */
const FALLBACK_CENTRE: [number, number] = [12.8452, 77.6602];

export interface MapCanvasProps {
  cameras?: Camera[];
  incidents?: Incident[];
  graph?: RoadGraphGeoJSON | null;
  /** Solid RED — the diversion other traffic should take. */
  diversion?: RouteGeometry | null;
  /** Dashed BLUE — simulated responder access route. Usually null; see api.ts. */
  accessRoute?: RouteGeometry | null;
  selectedIncidentId?: number | null;
  onSelectIncident?: (incident: Incident) => void;
  /** Clearly-labelled showcase layer; it never changes backend data. */
  demo?: ElectronicCityDemoState | null;
  className?: string;
}

export default function MapCanvas({
  cameras = [],
  incidents = [],
  graph = null,
  diversion = null,
  accessRoute = null,
  selectedIncidentId = null,
  onSelectIncident,
  demo = null,
  className = "",
}: MapCanvasProps) {
  /** Cameras carry the only real coordinates in the system. */
  const locatedCameras = useMemo(
    () =>
      cameras.filter(
        (camera) =>
          typeof camera.latitude === "number" && typeof camera.longitude === "number",
      ),
    [cameras],
  );

  const cameraPosition = useMemo(() => {
    const index = new Map<string, [number, number]>();
    for (const camera of locatedCameras) {
      index.set(camera.camera_id, [camera.latitude as number, camera.longitude as number]);
    }
    return index;
  }, [locatedCameras]);

  /**
   * An incident is pinned at its CAMERA's position - never at a guessed
   * vehicle position. location.py is explicit about this and the popup says so.
   */
  const pinnedIncidents = useMemo(
    () =>
      incidents
        .map((incident) => {
          const fromLocation =
            typeof incident.location?.latitude === "number" &&
            typeof incident.location?.longitude === "number"
              ? ([incident.location.latitude, incident.location.longitude] as [number, number])
              : null;
          const position = fromLocation ?? cameraPosition.get(incident.camera_id) ?? null;
          return position ? { incident, position } : null;
        })
        .filter((entry): entry is { incident: Incident; position: [number, number] } =>
          entry !== null,
        ),
    [incidents, cameraPosition],
  );

  const graphLines = useMemo(() => {
    if (!graph?.features?.length) return [];
    return graph.features.map((feature) => ({
      id: feature.properties.id,
      name: feature.properties.name,
      closed: feature.properties.closed,
      penalty: feature.properties.penalty,
      // GeoJSON [lon, lat] -> Leaflet [lat, lon]
      positions: feature.geometry.coordinates.map(
        ([lon, lat]) => [lat, lon] as [number, number],
      ),
    }));
  }, [graph]);

  const centre = useMemo<[number, number]>(() => {
    if (demo) return demo.centre;
    const points = [
      ...pinnedIncidents.map((entry) => entry.position),
      ...locatedCameras.map(
        (camera) => [camera.latitude as number, camera.longitude as number] as [number, number],
      ),
      ...graphLines.flatMap((line) => line.positions),
    ];
    if (!points.length) return FALLBACK_CENTRE;
    const lat = points.reduce((sum, point) => sum + point[0], 0) / points.length;
    const lon = points.reduce((sum, point) => sum + point[1], 0) / points.length;
    return [lat, lon];
  }, [demo, pinnedIncidents, locatedCameras, graphLines]);

  const hasAnything = Boolean(demo) || pinnedIncidents.length + locatedCameras.length + graphLines.length > 0;

  return (
    <div className={`relative h-full w-full ${className}`}>
      <MapContainer
        center={centre}
        zoom={demo ? 15 : hasAnything ? 13 : 11}
        scrollWheelZoom
        className="h-full w-full"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* road graph */}
        {graphLines.map((line, index) => (
          <Polyline
            key={`${line.id}-${index}`}
            positions={line.positions}
            pathOptions={{
              color: line.closed ? "#e5484d" : "#5d6775",
              weight: line.closed ? 4 : 2,
              opacity: line.closed ? 0.9 : 0.55,
              dashArray: line.closed ? "2,6" : undefined,
            }}
          >
            <Tooltip sticky>
              {line.name || line.id}
              {line.closed ? " — CLOSED by operator" : ""}
              {line.penalty ? ` — penalty ${line.penalty.toFixed(2)}` : ""}
            </Tooltip>
          </Polyline>
        ))}

        {/* SOLID RED: diversion route for other traffic */}
        {diversion?.coords?.length ? (
          <Polyline
            positions={diversion.coords}
            pathOptions={{ color: "#e5484d", weight: 5, opacity: 0.95 }}
          >
            <Popup>
              <b>Diversion route (other traffic)</b>
              <br />
              {diversion.length_m} m · {diversion.cost_s} s
              <br />
              {diversion.edge_ids.join(" → ")}
            </Popup>
          </Polyline>
        ) : null}

        {/* DASHED BLUE: simulated responder access route */}
        {accessRoute?.coords?.length ? (
          <Polyline
            positions={accessRoute.coords}
            pathOptions={{ color: "#4da3ff", weight: 4, opacity: 0.9, dashArray: "7,6" }}
          >
            <Popup>
              <b>SIMULATED responder access route</b>
              <br />
              {accessRoute.length_m} m · {accessRoute.cost_s} s
            </Popup>
          </Polyline>
        ) : null}

        {/* Electronic City Phase 1 showcase. All of these marks are supplied
            by APP/lib/electronicCityDemo.ts and intentionally stay outside
            the backend incident/camera records. */}
        {demo ? (
          <>
            {demo.incident.closedEdgeCoords ? (
              <Polyline
                positions={demo.incident.closedEdgeCoords}
                pathOptions={{ color: "#e5484d", weight: 6, opacity: 0.95, dashArray: "2,8" }}
              >
                <Popup>
                  <b>SIMULATED congested approach</b>
                  <br />
                  Excluded from the Phase 1 demo route model.
                </Popup>
              </Polyline>
            ) : null}
            {demo.route.coords.length ? (
              <Polyline
                positions={demo.route.coords}
                pathOptions={{ color: "#4da3ff", weight: 5, opacity: 0.95 }}
              >
                <Popup>
                  <b>SIMULATED hospital access route</b>
                  <br />
                  {demo.route.facility.name}
                  <br />
                  {demo.route.distanceM.toLocaleString()} m · ~{demo.route.etaMinutes} min
                  <br />
                  Demo road model only; no dispatch is issued.
                </Popup>
              </Polyline>
            ) : null}

            {demo.cameras.map((camera) => (
              <CircleMarker
                key={`demo-camera-${camera.id}`}
                center={camera.position}
                radius={camera.status === "incident" ? 8 : 6}
                pathOptions={{
                  color: camera.status === "incident" ? "#e5a73f" : "#3fb9a6",
                  fillColor: camera.status === "incident" ? "#e5a73f" : "#3fb9a6",
                  fillOpacity: 0.75,
                  weight: 2,
                }}
              >
                <Tooltip direction="top" offset={[0, -8]}>
                  <b>{camera.id}</b> · {camera.name}
                </Tooltip>
                <Popup>
                  <b>{camera.name}</b>
                  <br />
                  {camera.road}
                  <br />
                  <span style={{ color: "#8b95a5" }}>SYNTHETIC CCTV showcase location</span>
                </Popup>
              </CircleMarker>
            ))}

            {demo.hospitals.map((hospital) => (
              <CircleMarker
                key={`demo-hospital-${hospital.id}`}
                center={hospital.position}
                radius={hospital.id === demo.route.facility.id ? 9 : 6}
                pathOptions={{
                  color: "#4da3ff",
                  fillColor: "#4da3ff",
                  fillOpacity: hospital.id === demo.route.facility.id ? 0.85 : 0.45,
                  weight: hospital.id === demo.route.facility.id ? 3 : 2,
                }}
              >
                <Tooltip direction="top" offset={[0, -8]}>
                  <b>H</b> · {hospital.name}
                </Tooltip>
                <Popup>
                  <b>{hospital.name}</b>
                  <br />
                  <span style={{ color: "#8b95a5" }}>{hospital.source}</span>
                </Popup>
              </CircleMarker>
            ))}

            <CircleMarker
              center={demo.incident.position}
              radius={12}
              pathOptions={{ color: "#e5484d", fillColor: "#e5484d", fillOpacity: 0.75, weight: 3 }}
            >
              <Tooltip direction="top" offset={[0, -10]}>
                <b>{demo.incident.id}</b> · congestion scenario
              </Tooltip>
              <Popup>
                <b>{demo.incident.label}</b>
                <br />
                {demo.incident.congestionActive
                  ? `Congestion active: ${demo.incident.closedEdgeLabel} is excluded.`
                  : "No demo road closure active."}
                <br />
                <span style={{ color: "#8b95a5" }}>SYNTHETIC scenario; not a backend incident.</span>
              </Popup>
            </CircleMarker>
          </>
        ) : null}

        {/* cameras */}
        {locatedCameras.map((camera) => (
          <CircleMarker
            key={`cam-${camera.camera_id}`}
            center={[camera.latitude as number, camera.longitude as number]}
            radius={4}
            pathOptions={{ color: "#5d6775", weight: 1, fillOpacity: 0, dashArray: "2,3" }}
          >
            <Popup>
              <b>{camera.name || camera.camera_id}</b>
              <br />
              {camera.road_name || "road unnamed"}
              {camera.zone ? ` · ${camera.zone}` : ""}
              <br />
              {(camera.corridors?.length ?? 0)} calibrated corridors
            </Popup>
          </CircleMarker>
        ))}

        {/* incidents */}
        {pinnedIncidents.map(({ incident, position }) => {
          const tier = normaliseSeverity(incident.severity_label);
          const colour = tier ? SEVERITY_HEX[tier] : UNKNOWN_SEVERITY_HEX;
          const selected = selectedIncidentId === incident.id;
          return (
            <CircleMarker
              key={`inc-${incident.id}`}
              center={position}
              radius={selected ? 11 : 7}
              pathOptions={{
                color: colour,
                fillColor: colour,
                fillOpacity: selected ? 0.75 : 0.45,
                weight: selected ? 3 : 2,
              }}
              eventHandlers={
                onSelectIncident ? { click: () => onSelectIncident(incident) } : undefined
              }
            >
              <Popup>
                <b>{eventLabel(incident.event_type)}</b>
                <br />
                {incidentRef(incident.id)} · {incident.camera_id}
                <br />
                severity {incident.severity_label ?? "unscored"} · status {incident.status}
                <br />
                <span style={{ color: "#8b95a5" }}>
                  Pin is the camera&rsquo;s position, not the vehicle&rsquo;s.
                </span>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {!hasAnything ? (
        <div className="pointer-events-none absolute inset-0 z-[400] flex items-center justify-center p-6">
          <div className="pointer-events-auto max-w-md rounded-lg border border-dashed border-line bg-panel-1/95 px-5 py-4 text-center">
            <p className="text-[14px] font-semibold text-ink-1">Nothing geo-located to draw</p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-2">
              No camera in <code className="font-mono">config/cameras/</code> has a latitude and
              longitude, and the road graph is empty. Cameras are the only source of coordinates
              in this system, so the map has nothing to place. Add{" "}
              <code className="font-mono">latitude</code>/<code className="font-mono">longitude</code>{" "}
              to a camera JSON, or point{" "}
              <code className="font-mono">paths.road_graph</code> at a road graph file.
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
