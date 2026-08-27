/**
 * Electronic City Phase 1 showcase scenario.
 *
 * This is intentionally a SMALL, local routing model for a presentation—not
 * a navigation service. CCTV positions are synthetic. Hospital coordinates
 * mirror the public OSM-tagged facilities already recorded in
 * CCTV/config/facilities.json, but availability, clinical capability and
 * dispatch status are unknown. The Dijkstra result is therefore labelled a
 * "demo route" everywhere it is rendered.
 */

export type LatLon = [number, number];

export interface DemoCamera {
  id: string;
  name: string;
  road: string;
  position: LatLon;
  status: "monitoring" | "incident";
}

export interface DemoHospital {
  id: string;
  name: string;
  position: LatLon;
  source: string;
}

export interface DemoRoute {
  facility: DemoHospital;
  coords: LatLon[];
  distanceM: number;
  etaMinutes: number;
  baselineDistanceM: number;
  baselineEtaMinutes: number;
}

export interface ElectronicCityDemoState {
  centre: LatLon;
  cameras: DemoCamera[];
  hospitals: DemoHospital[];
  incident: {
    id: string;
    label: string;
    position: LatLon;
    congestionActive: boolean;
    closedEdgeLabel: string | null;
  };
  route: DemoRoute;
}

interface GraphNode {
  id: string;
  position: LatLon;
}

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  distanceM: number;
}

const NODES: GraphNode[] = [
  { id: "kauvery", position: [12.855088, 77.66316] },
  { id: "best-e-city", position: [12.850377, 77.656706] },
  { id: "ramakrishna", position: [12.850065, 77.662471] },
  { id: "north-link", position: [12.8524, 77.6619] },
  { id: "neeladri-junction", position: [12.849, 77.6608] },
  { id: "wipro-link", position: [12.8472, 77.6577] },
  { id: "south-link", position: [12.8446, 77.6585] },
  { id: "incident", position: [12.8461, 77.6617] },
];

// Edge lengths are illustrative and only support an explainable Dijkstra
// demonstration. They are not turn-by-turn road distances.
const EDGES: GraphEdge[] = [
  { id: "kauvery-north", from: "kauvery", to: "north-link", distanceM: 400 },
  { id: "north-neeladri", from: "north-link", to: "neeladri-junction", distanceM: 500 },
  { id: "neeladri-incident", from: "neeladri-junction", to: "incident", distanceM: 450 },
  { id: "north-wipro", from: "north-link", to: "wipro-link", distanceM: 700 },
  { id: "wipro-south", from: "wipro-link", to: "south-link", distanceM: 450 },
  { id: "south-incident", from: "south-link", to: "incident", distanceM: 600 },
  { id: "best-wipro", from: "best-e-city", to: "wipro-link", distanceM: 300 },
  { id: "ramakrishna-neeladri", from: "ramakrishna", to: "neeladri-junction", distanceM: 250 },
];

const NODE_BY_ID = new Map(NODES.map((node) => [node.id, node]));

const HOSPITALS: DemoHospital[] = [
  {
    id: "kauvery",
    name: "Kauvery Hospital, Electronic City",
    position: [12.855088, 77.66316],
    source: "Public OSM-tagged facility; availability is not verified.",
  },
  {
    id: "best-e-city",
    name: "Best E City Hospital",
    position: [12.850377, 77.656706],
    source: "Public OSM-tagged facility; availability is not verified.",
  },
  {
    id: "ramakrishna",
    name: "Ramakrishna Hospital",
    position: [12.850065, 77.662471],
    source: "Public OSM-tagged facility; availability is not verified.",
  },
];

const CAMERAS: DemoCamera[] = [
  {
    id: "EC-P1-CCTV-01",
    name: "Phase 1 Gate",
    road: "Hosur Road approach",
    position: [12.8449, 77.664],
    status: "monitoring",
  },
  {
    id: "EC-P1-CCTV-02",
    name: "Neeladri Junction",
    road: "Neeladri Road corridor",
    position: [12.8461, 77.6617],
    status: "incident",
  },
  {
    id: "EC-P1-CCTV-03",
    name: "Wipro Link",
    road: "Phase 1 western connector",
    position: [12.8472, 77.6577],
    status: "monitoring",
  },
  {
    id: "EC-P1-CCTV-04",
    name: "Velankani Link",
    road: "Southern Phase 1 approach",
    position: [12.8427, 77.6589],
    status: "monitoring",
  },
];

function routeFrom(start: string, blockedEdgeIds: Set<string>): { nodeIds: string[]; distanceM: number } | null {
  const distance = new Map<string, number>(NODES.map((node) => [node.id, Infinity]));
  const previous = new Map<string, string>();
  const unsettled = new Set(NODES.map((node) => node.id));
  distance.set(start, 0);

  while (unsettled.size) {
    let current: string | null = null;
    for (const nodeId of Array.from(unsettled)) {
      if (current === null || (distance.get(nodeId) ?? Infinity) < (distance.get(current) ?? Infinity)) {
        current = nodeId;
      }
    }
    if (!current || distance.get(current) === Infinity) break;
    if (current === "incident") break;
    unsettled.delete(current);

    for (const edge of EDGES) {
      if (blockedEdgeIds.has(edge.id)) continue;
      const neighbour = edge.from === current ? edge.to : edge.to === current ? edge.from : null;
      if (!neighbour || !unsettled.has(neighbour)) continue;
      const candidate = (distance.get(current) ?? Infinity) + edge.distanceM;
      if (candidate < (distance.get(neighbour) ?? Infinity)) {
        distance.set(neighbour, candidate);
        previous.set(neighbour, current);
      }
    }
  }

  const finalDistance = distance.get("incident") ?? Infinity;
  if (!Number.isFinite(finalDistance)) return null;
  const nodeIds = ["incident"];
  while (nodeIds[0] !== start) {
    const parent = previous.get(nodeIds[0]);
    if (!parent) return null;
    nodeIds.unshift(parent);
  }
  return { nodeIds, distanceM: finalDistance };
}

function bestRoute(blockedEdgeIds: Set<string>) {
  return HOSPITALS.flatMap((facility) => {
    const path = routeFrom(facility.id, blockedEdgeIds);
    return path ? [{ facility, path }] : [];
  }).sort((a, b) => a.path.distanceM - b.path.distanceM)[0];
}

export function electronicCityDemo(congestionActive: boolean): ElectronicCityDemoState {
  const baseline = bestRoute(new Set());
  const blocked = congestionActive ? new Set(["neeladri-incident"]) : new Set<string>();
  const recommended = bestRoute(blocked);
  if (!baseline || !recommended) throw new Error("Electronic City demo graph is disconnected");

  return {
    centre: [12.8482, 77.6603],
    cameras: CAMERAS,
    hospitals: HOSPITALS,
    incident: {
      id: "EC-P1-INC-01",
      label: "Heavy congestion near Neeladri Junction",
      position: NODE_BY_ID.get("incident")!.position,
      congestionActive,
      closedEdgeLabel: congestionActive ? "Neeladri Junction → incident approach" : null,
    },
    route: {
      facility: recommended.facility,
      coords: recommended.path.nodeIds.map((id) => NODE_BY_ID.get(id)!.position),
      distanceM: recommended.path.distanceM,
      etaMinutes: Number((recommended.path.distanceM / 1000 / 22 * 60).toFixed(1)),
      baselineDistanceM: baseline.path.distanceM,
      baselineEtaMinutes: Number((baseline.path.distanceM / 1000 / 22 * 60).toFixed(1)),
    },
  };
}
