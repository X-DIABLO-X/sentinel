"""Location intelligence and incident-aware diversion.

Two claims are carefully separated here, because conflating them is one of the
easiest ways to overstate what a camera knows.

**What we can say:** the camera is at a known place. It is a fixed geospatial
sensor; a human recorded its position and the road segment it watches. So
"incident on the westbound carriageway of X Road, Zone Y, camera CAM_04" is
true and useful.

**What we cannot say:** where the *vehicle* is in world coordinates. Projecting
an image point to the ground needs camera pose, intrinsics and a ground model.
Without a homography NETRA reports camera-associated location and labels its
precision as such. It never invents per-vehicle GPS.

The diversion engine is a road graph, not a language model. An incident raises
the traversal cost of the affected edge in proportion to its severity -- or
removes the edge entirely once an operator confirms a closure -- and the route
is recomputed with Dijkstra. That is *incident-aware routing*, and calling it
anything more (live traffic-aware routing, say) would require live network
speeds we do not have.

The graph is bundled as JSON rather than fetched from OSM at runtime. The final
demonstration may have no internet, and a system that needs a network call to
draw a diversion is a system that fails on stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx

# Precision vocabulary used on the dashboard so nobody misreads a pin.
PRECISION_CAMERA = "camera-associated"     # we know the camera, not the object
PRECISION_GROUND = "ground-plane-projected"  # homography available
PRECISION_ZONE = "zone-only"               # no coordinates at all


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between two (lat, lon) pairs."""
    R = 6371000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(h)))


class RoadGraph:
    """A small directed road network with incident-aware edge costs."""

    def __init__(self, path: str | Path | None = None):
        self.G = nx.DiGraph()
        self.nodes: dict[str, dict] = {}
        self.penalties: dict[str, float] = {}
        self.closed: set[str] = set()
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.load(self.path)

    # -- construction ------------------------------------------------------
    def load(self, path: str | Path) -> "RoadGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for n in data.get("nodes", []):
            self.nodes[n["id"]] = n
            self.G.add_node(n["id"], **n)
        for e in data.get("edges", []):
            u, v = e["from"], e["to"]
            length = e.get("length_m")
            if length is None and u in self.nodes and v in self.nodes:
                length = haversine_m(
                    (self.nodes[u]["lat"], self.nodes[u]["lon"]),
                    (self.nodes[v]["lat"], self.nodes[v]["lon"]),
                )
            length = float(length or 100.0)
            speed = float(e.get("speed_kph", 30.0))
            base_cost = length / max(speed * 1000.0 / 3600.0, 1e-6)   # seconds
            self.G.add_edge(u, v, id=e["id"], name=e.get("name", ""),
                            length_m=length, speed_kph=speed,
                            base_cost=base_cost, oneway=e.get("oneway", False))
            if not e.get("oneway", False):
                self.G.add_edge(v, u, id=e["id"] + "_rev", name=e.get("name", ""),
                                length_m=length, speed_kph=speed,
                                base_cost=base_cost, oneway=False)
        return self

    # -- incident effects --------------------------------------------------
    def apply_incident(self, edge_id: str, severity: float, confirmed_closed: bool = False):
        """Penalise (or close) an edge.

        The multiplier is ``1 + lambda * S`` with lambda = 9, so a
        maximum-severity incident makes the edge ten times as expensive to
        traverse -- enough to divert around it without making the graph
        disconnected, which an infinite cost can do.
        """
        if confirmed_closed:
            self.closed.add(edge_id)
        else:
            self.penalties[edge_id] = max(self.penalties.get(edge_id, 0.0), float(severity))

    def clear_incident(self, edge_id: str) -> None:
        self.penalties.pop(edge_id, None)
        self.closed.discard(edge_id)

    def _weight(self, u, v, data) -> float | None:
        eid = data.get("id", "")
        root = eid.replace("_rev", "")
        if root in self.closed:
            return None                      # networkx treats None as "no edge"
        pen = self.penalties.get(root, 0.0)
        return float(data["base_cost"]) * (1.0 + 9.0 * pen)

    # -- routing -----------------------------------------------------------
    def route(self, src: str, dst: str) -> dict:
        """Shortest path under current incident costs, plus the clean baseline."""
        out: dict = {"source": src, "target": dst}
        try:
            base_nodes = nx.shortest_path(self.G, src, dst, weight="base_cost")
            out["baseline"] = self._describe(base_nodes, use_penalty=False)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            out["baseline"] = None
            out["error"] = str(exc)
            return out
        try:
            nodes = nx.shortest_path(self.G, src, dst, weight=self._weight)
            out["current"] = self._describe(nodes, use_penalty=True)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            out["current"] = None
            out["note"] = "no route available with current closures"
            return out

        b, c = out["baseline"], out["current"]
        out["diverted"] = b["edge_ids"] != c["edge_ids"]
        out["extra_seconds"] = round(c["cost_s"] - b["cost_s"], 1)
        out["extra_metres"] = round(c["length_m"] - b["length_m"], 1)
        return out

    def _describe(self, nodes: list[str], use_penalty: bool) -> dict:
        edges, ids, cost, length = [], [], 0.0, 0.0
        for u, v in zip(nodes, nodes[1:]):
            data = self.G.edges[u, v]
            w = self._weight(u, v, data) if use_penalty else data["base_cost"]
            cost += float(w or data["base_cost"])
            length += float(data["length_m"])
            ids.append(data["id"].replace("_rev", ""))
            edges.append({"from": u, "to": v, "id": data["id"], "name": data.get("name", "")})
        return {
            "nodes": nodes,
            "edges": edges,
            "edge_ids": ids,
            "cost_s": round(cost, 1),
            "length_m": round(length, 1),
            "coords": [[self.nodes[n]["lat"], self.nodes[n]["lon"]]
                       for n in nodes if n in self.nodes],
        }

    def to_geojson(self) -> dict:
        feats = []
        for u, v, data in self.G.edges(data=True):
            if u not in self.nodes or v not in self.nodes:
                continue
            root = data["id"].replace("_rev", "")
            feats.append({
                "type": "Feature",
                "properties": {
                    "id": root, "name": data.get("name", ""),
                    "penalty": self.penalties.get(root, 0.0),
                    "closed": root in self.closed,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [self.nodes[u]["lon"], self.nodes[u]["lat"]],
                        [self.nodes[v]["lon"], self.nodes[v]["lat"]],
                    ],
                },
            })
        return {"type": "FeatureCollection", "features": feats}


def describe_location(scene, event=None) -> dict:
    """Build the location block that goes on every alert."""
    if scene.latitude is not None and scene.longitude is not None:
        precision = PRECISION_GROUND if scene.has_metric_scale else PRECISION_CAMERA
    else:
        precision = PRECISION_ZONE
    return {
        "camera_id": scene.camera_id,
        "camera_name": scene.name,
        "zone": scene.zone,
        "road_name": scene.road_name,
        "road_edge_id": scene.road_edge_id,
        "corridor_id": event.corridor_id if event else None,
        "latitude": scene.latitude,
        "longitude": scene.longitude,
        "precision": precision,
        "note": (
            "Position is the camera's, not the vehicle's. Per-object geolocation "
            "would require camera pose, intrinsics and a ground model."
        ),
    }
