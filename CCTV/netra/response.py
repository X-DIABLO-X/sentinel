"""Diversion routing and (simulated) responder access routing.

NETRA already answers *what happened, where the camera is, and how disruptive it
is*. It does not answer the two questions a control room asks next:

    1. "Where do I send the rest of the traffic?"        -> DIVERSION route
    2. "How would a responder reach this point?"         -> ACCESS route

This module answers both over a real OpenStreetMap drive graph (OSMnx +
NetworkX), cached to disk on first use so the demonstration does not depend on
network access at demo time.

Two routes, deliberately never the same object
----------------------------------------------
The **diversion** route is for OTHER TRAFFIC and is computed on a copy of the
graph with the incident's own node REMOVED. That removal is the whole point: it
is what makes "this route avoids the incident" a provable property of the
computation rather than an eyeball judgement about a line that happens to pass
nearby.

The **access** route runs the other way -- from the nearest facility TOWARD the
incident -- and is computed on the FULL graph, because reaching the incident is
its entire purpose. It is **SIMULATED**. Nobody is contacted, no dispatch system
is touched, and its ETA comes from an explicitly stated assumed mean speed
(``ACCESS_ASSUMED_MEAN_SPEED_KMH``), not from live traffic, which we do not
have. ``access_simulated`` is carried on the *data*, not merely in this
docstring, so it survives JSON serialisation into the DB and the API response
and every consumer must check it.

Frontend convention (kept in the returned data so a UI cannot confuse them):

    diversion -> solid  RED  line   (audience: other traffic)
    access    -> dashed BLUE line   (audience: responders, SIMULATED)

Degradation is explicit at every step
-------------------------------------
If osmnx is not installed, or the graph cannot be fetched, or the camera has no
coordinates, or no path exists, the plan still returns its operational actions
and says *why* there is no route, in ``degraded_reason`` /
``access_degraded_reason``. It never fabricates a route and never draws a line
that was not computed.

Interface (deliberately decoupled from NETRA's and any other schema)
--------------------------------------------------------------------
This module depends only on plain primitives, so it can be wired into
``netra/api.py``, a scene object, a DB row or a test without importing any
event/severity class:

    plan = ResponseRecommender().recommend(
        event_type="collision_candidate",   # netra.events.base type constant
        severity_label="High",              # "Low"|"Medium"|"High" (or "Critical")
        latitude=20.4625, longitude=85.883, # camera coords, or None
        road_name="NH-16 link road",        # optional, headline text only
        zone="Cuttack",                     # optional, headline text only
        camera_id="CUTTACK_LINK_01",        # optional, echoed back
        base_action="...",                  # optional; e.g. netra.severity.recommend(event)
    )
    payload = plan.as_dict()

``severity_label`` is matched case-insensitively and accepts NETRA's three-band
vocabulary ("Low"/"Medium"/"High") as well as a "Critical" tier if a caller ever
introduces one. ``event_type`` is matched against the string constants in
``netra.events.base`` (imported here only for those constants; no event objects
cross this boundary).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# Only the string constants -- no event object ever crosses this boundary.
from .events.base import (
    BLOCKAGE,
    COLLISION,
    LANE_VIOLATION,
    PEDESTRIAN,
    QUEUE,
    STOPPED,
    WRONG_WAY,
)

#: repo root: .../CCTV  (this file is .../CCTV/netra/response.py)
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Assumed mean speed used for the SIMULATED responder ETA. Stated as an
#: assumption everywhere it is reported. There is no live traffic feed behind
#: this number and it must never be presented as a measured or promised ETA.
ACCESS_ASSUMED_MEAN_SPEED_KMH = 30.0

#: Rendering contract shared with the frontend. Carried on the data so the two
#: route kinds cannot be confused by a consumer that only looks at coordinates.
DIVERSION_STYLE: dict[str, Any] = {
    "kind": "diversion",
    "color": "#d62828",
    "dashed": False,
    "audience": "other_traffic",
}
ACCESS_STYLE: dict[str, Any] = {
    "kind": "access",
    "color": "#1d4ed8",
    "dashed": True,
    "audience": "responders",
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


#: kept under its ported name as well, for callers that expect the private one
_haversine_m = haversine_m


# --------------------------------------------------------------------------
# policy tables
# --------------------------------------------------------------------------

#: Canonical tier vocabulary. NETRA's severity module bands into
#: Low/Medium/High; CRITICAL is accepted for forward compatibility only.
_TIERS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

#: Tiers at which activating a diversion is a RECOMMENDED ACTION. A low-tier
#: queue does not warrant a VMS advisory even though showing the way around it
#: is still informative -- see `_ALWAYS_ROUTE_TYPES`.
_DIVERSION_TIERS = {"HIGH", "CRITICAL"}

#: Event types for which the diversion route is computed regardless of tier.
#:
#: Computing a route is free and read-only -- it does not claim anything
#: happened, it only asks the road graph "what is the way around this point".
#: Gating the arithmetic behind severity means the one incident a reviewer is
#: most likely to click on can show no route at all: NETRA's severity rewards
#: SUSTAINED disruption, so the impact instant of a collision can band MEDIUM
#: while a later wreckage/blockage event bands HIGH purely on dwell time.
#: The tier still gates the recommended ACTION below; only the arithmetic is
#: unconditional.
_ALWAYS_ROUTE_TYPES = {COLLISION}

#: Event types that get a SIMULATED responder access route regardless of tier.
#: Everything else gets one only at HIGH/CRITICAL.
_ALWAYS_ACCESS_TYPES = {COLLISION}

_ESCALATION: dict[str, list[str]] = {
    "LOW": ["Log for corridor analytics"],
    "MEDIUM": [
        "Notify the traffic management centre",
        "Continue automated monitoring",
    ],
    "HIGH": [
        "Alert the nearest traffic patrol",
        "Publish a variable-message-sign advisory",
        "Notify the traffic management centre supervisor",
    ],
    "CRITICAL": [
        "Dispatch the nearest patrol immediately",
        "Notify emergency services and recovery",
        "Publish a variable-message-sign advisory",
        "Escalate to the traffic management centre supervisor",
    ],
}

_TYPE_ACTIONS: dict[str, str] = {
    QUEUE: "Review signal timing for the affected corridor",
    BLOCKAGE: "Arrange vehicle recovery or removal",
    WRONG_WAY: "Warn oncoming traffic before attempting interception",
    LANE_VIOLATION: "Refer to enforcement review",
    STOPPED: "Check whether the stop is legitimate before dispatching",
    PEDESTRIAN: "Protect the pedestrian; slow the affected corridor",
    COLLISION: "Human verification of the evidence clip is required before dispatch",
}

#: Which facility kinds are preferred for each event type. Falls back to every
#: facility in the registry when none of the preferred kinds yields a route.
_PREFERRED_FACILITY_KINDS: dict[str, tuple[str, ...]] = {
    COLLISION: ("hospital", "police", "fire"),
    PEDESTRIAN: ("hospital", "police"),
    BLOCKAGE: ("police", "fire"),
    STOPPED: ("police",),
    WRONG_WAY: ("police",),
    LANE_VIOLATION: ("police",),
    QUEUE: ("police",),
}


def _normalise_tier(severity_label: Optional[str]) -> str:
    if not severity_label:
        return "LOW"
    tier = str(severity_label).strip().upper()
    return tier if tier in _TIERS else "LOW"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class ResponseConfig:
    """Routing configuration.

    Defaults are deliberately place-agnostic: NETRA's demo cameras are not all
    in one city, so the primary graph strategy is a radius fetch around the
    incident, cached per rounded coordinate. Set ``osm_place`` if a deployment
    really is confined to one named area and you would rather cache that.
    """

    enable_osmnx: bool = True
    #: optional named place, e.g. "Cuttack, Odisha, India". None -> radius fetch
    osm_place: Optional[str] = None
    osm_network_type: str = "drive"
    osm_radius_m: int = 2500
    #: directory for cached GraphML. Relative paths resolve against CCTV root.
    graph_cache_dir: str = "results/cache/roadgraph"
    #: how far upstream/downstream of the incident the diversion endpoints sit
    diversion_offset_m: float = 700.0
    #: how many facilities to try before giving up on an access route
    max_facility_candidates: int = 5
    #: registry path. Relative paths resolve against CCTV root.
    facilities_path: str = "config/facilities.json"

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "ResponseConfig":
        """Build from a config.yaml ``response:`` block, ignoring unknown keys."""
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}          # noqa: F821
        return cls(**{k: v for k, v in data.items() if k in known})


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------
# road graph
# --------------------------------------------------------------------------


class OSMRoadGraph:
    """Lazily-loaded OSM drive graph, cached to disk as GraphML.

    Named ``OSMRoadGraph`` rather than ``RoadGraph`` because
    ``netra.location.RoadGraph`` already exists and is a different thing: that
    one is a small hand-authored JSON network with incident-aware edge costs;
    this one is the real OSM drive network. They are complementary, not rivals.

    One graph is held per rounded coordinate key, because NETRA's cameras span
    several cities and a single global graph would be wrong for all but one of
    them.
    """

    def __init__(self, config: Optional[ResponseConfig] = None):
        self.config = config or ResponseConfig()
        self._graphs: dict[str, Any] = {}
        self._failed: set[str] = set()
        self.status = "not loaded"
        #: True once at least one graph has been obtained
        self.available = False

    # -- cache keys --------------------------------------------------------

    @staticmethod
    def _point_key(lat: float, lon: float) -> str:
        # ~1.1 km cells: nearby cameras share one cached graph instead of each
        # fetching its own.
        return f"pt_{lat:.2f}_{lon:.2f}".replace("-", "m").replace(".", "p")

    @staticmethod
    def _place_key(place: str) -> str:
        return "place_" + re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")

    def cache_path(self, key: str) -> Path:
        return _resolve(self.config.graph_cache_dir) / f"{key}.graphml"

    # -- injection (tests / preloaded graphs) ------------------------------

    def set_graph(self, graph: Any, lat: float, lon: float, status: str = "graph supplied by caller") -> None:
        """Install a graph directly, bypassing OSMnx entirely.

        Used by tests and by any deployment that ships its own GraphML.
        """
        self._graphs[self._point_key(lat, lon)] = graph
        if self.config.osm_place:
            self._graphs[self._place_key(self.config.osm_place)] = graph
        self.available = True
        self.status = status

    # -- loading -----------------------------------------------------------

    def graph_for(self, lat: float, lon: float) -> Optional[Any]:
        """Return a drive graph covering (lat, lon), or None with a reason in
        ``self.status``."""
        keys = []
        if self.config.osm_place:
            keys.append(self._place_key(self.config.osm_place))
        keys.append(self._point_key(lat, lon))

        for key in keys:
            g = self._graphs.get(key)
            if g is not None:
                self.available = True
                self.status = f"using in-memory graph '{key}'"
                return g

        if not self.config.enable_osmnx:
            self.status = (
                "diversion routing disabled in configuration "
                "(response.enable_osmnx = false)"
            )
            return None

        try:
            import osmnx as ox
        except ImportError:
            self.status = (
                "osmnx is not installed; routing unavailable "
                "(pip install osmnx). Response degrades to text-only advice."
            )
            return None

        # 1. disk cache
        for key in keys:
            cache = self.cache_path(key)
            if not cache.exists():
                continue
            try:
                g = ox.load_graphml(str(cache))
            except Exception as exc:  # noqa: BLE001
                self.status = (
                    f"cached graph {cache.name} unreadable "
                    f"({type(exc).__name__}); refetching"
                )
                continue
            self._graphs[key] = g
            self.available = True
            self.status = f"loaded cached graph from {cache.name}"
            return g

        # 2. named place, if one is configured
        if self.config.osm_place:
            key = self._place_key(self.config.osm_place)
            if key not in self._failed:
                try:
                    g = ox.graph_from_place(
                        self.config.osm_place,
                        network_type=self.config.osm_network_type,
                    )
                except Exception as exc:  # noqa: BLE001
                    g = None
                    self.status = (
                        f"could not download the road graph for "
                        f"'{self.config.osm_place}' ({type(exc).__name__}). "
                        "Falling back to a radius fetch."
                    )
                if g is not None:
                    self._graphs[key] = g
                    self._save(g, key)
                    self.available = True
                    self.status = f"downloaded and cached graph for {self.config.osm_place}"
                    return g
                self._failed.add(key)

        # 3. radius fetch around the incident
        key = self._point_key(lat, lon)
        if key in self._failed:
            return None
        try:
            g = ox.graph_from_point(
                (lat, lon),
                dist=self.config.osm_radius_m,
                network_type=self.config.osm_network_type,
            )
        except Exception as exc:  # noqa: BLE001
            self._failed.add(key)
            self.status = (
                f"could not download a road graph within "
                f"{self.config.osm_radius_m} m of ({lat:.4f}, {lon:.4f}): "
                f"{type(exc).__name__}. Response degrades to text-only advice."
            )
            return None

        if g is None:
            self._failed.add(key)
            self.status = (
                f"road graph fetch returned nothing for ({lat:.4f}, {lon:.4f}). "
                "Response degrades to text-only advice."
            )
            return None

        self._graphs[key] = g
        self._save(g, key)
        self.available = True
        self.status = (
            f"downloaded and cached graph within {self.config.osm_radius_m} m "
            f"of ({lat:.4f}, {lon:.4f})"
        )
        return g

    def _save(self, graph: Any, key: str) -> None:
        """Best-effort disk cache. A failure here must not fail the routing."""
        try:
            import osmnx as ox

            cache = self.cache_path(key)
            cache.parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(graph, str(cache))
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# graph helpers (osmnx where available, pure-python fallback otherwise)
# --------------------------------------------------------------------------


def _nearest_node(graph: Any, lat: float, lon: float) -> Any:
    """Nearest graph node to a coordinate.

    Prefers ``osmnx.distance.nearest_nodes``; falls back to a linear haversine
    scan so a graph that did not come from OSMnx (a test fixture, a shipped
    GraphML) still routes without pulling osmnx into the hot path.
    """
    try:
        import osmnx as ox

        return ox.distance.nearest_nodes(graph, lon, lat)
    except Exception:  # noqa: BLE001
        best, best_d = None, float("inf")
        for n, data in graph.nodes(data=True):
            if "x" not in data or "y" not in data:
                continue
            d = haversine_m(lat, lon, float(data["y"]), float(data["x"]))
            if d < best_d:
                best, best_d = n, d
        if best is None:
            raise ValueError("graph has no nodes with x/y coordinates")
        return best


def _edge_length_m(graph: Any, u: Any, v: Any) -> float:
    """Length of the shortest parallel edge u->v, for multi- and simple graphs."""
    data = graph.get_edge_data(u, v)
    if not data:
        return 0.0
    if isinstance(data, dict) and all(isinstance(d, dict) for d in data.values()):
        lengths = [float(d.get("length", 0.0) or 0.0) for d in data.values()]
        return min(lengths) if lengths else 0.0
    return float(data.get("length", 0.0) or 0.0)


def _path_coords(graph: Any, path: Iterable[Any]) -> list[tuple[float, float]]:
    return [
        (float(graph.nodes[n]["y"]), float(graph.nodes[n]["x"]))
        for n in path
        if "y" in graph.nodes[n] and "x" in graph.nodes[n]
    ]


def _path_length_m(graph: Any, path: list[Any]) -> float:
    return sum(_edge_length_m(graph, u, v) for u, v in zip(path[:-1], path[1:]))


# --------------------------------------------------------------------------
# facilities
# --------------------------------------------------------------------------


def load_facilities(path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Read the facility registry. Missing/unreadable registry -> empty list."""
    p = _resolve(path or ResponseConfig.facilities_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for f in data.get("facilities", []):
        if f.get("latitude") is None or f.get("longitude") is None:
            continue
        out.append(dict(f))
    return out


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


@dataclass
class ResponsePlan:
    """Recommended operator response, with routes when they were computed.

    ``route_latlon`` / ``access_route_latlon`` may be empty. Degraded mode is
    explicit and carries its reason; it never carries a silently faked route.
    """

    headline: str
    actions: list[str] = field(default_factory=list)
    camera_id: Optional[str] = None
    event_type: str = ""
    severity_tier: str = "LOW"

    # -- diversion: for OTHER TRAFFIC, computed with the incident node removed
    diversion_available: bool = False
    diversion_summary: str = ""
    diversion_length_m: Optional[float] = None
    route_latlon: list[tuple[float, float]] = field(default_factory=list)
    degraded_reason: str = ""

    # -- access: SIMULATED responder route, nearest facility -> incident.
    # `access_simulated` lives on the value, not just in a docstring, so it
    # survives serialisation and every consumer must check it rather than infer
    # intent from the field name.
    access_facility: str = ""
    access_facility_kind: str = ""
    access_route_latlon: list[tuple[float, float]] = field(default_factory=list)
    access_route_summary: str = ""
    access_length_m: Optional[float] = None
    access_eta_minutes: Optional[float] = None
    access_degraded_reason: str = ""
    access_simulated: bool = True

    graph_status: str = ""

    # -- derived -----------------------------------------------------------

    def routes(self) -> list[dict[str, Any]]:
        """Both routes as separately-tagged records.

        Every record carries its kind, its rendering style and its
        ``simulated`` flag, so a UI or downstream consumer cannot mistake the
        responder access route for the public diversion route.
        """
        out: list[dict[str, Any]] = []
        if self.route_latlon:
            out.append({
                **DIVERSION_STYLE,
                "simulated": False,
                "latlon": [list(p) for p in self.route_latlon],
                "summary": self.diversion_summary,
                "length_m": self.diversion_length_m,
                "label": "Diversion for other traffic",
            })
        if self.access_route_latlon:
            out.append({
                **ACCESS_STYLE,
                "simulated": True,          # always; see class docstring
                "latlon": [list(p) for p in self.access_route_latlon],
                "summary": self.access_route_summary,
                "length_m": self.access_length_m,
                "eta_minutes": self.access_eta_minutes,
                "facility": self.access_facility,
                "label": "SIMULATED responder access route",
            })
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "actions": list(self.actions),
            "camera_id": self.camera_id,
            "event_type": self.event_type,
            "severity_tier": self.severity_tier,
            "diversion_available": self.diversion_available,
            "diversion_summary": self.diversion_summary,
            "diversion_length_m": self.diversion_length_m,
            "route_latlon": [list(p) for p in self.route_latlon],
            "degraded_reason": self.degraded_reason,
            "access_facility": self.access_facility,
            "access_facility_kind": self.access_facility_kind,
            "access_route_latlon": [list(p) for p in self.access_route_latlon],
            "access_route_summary": self.access_route_summary,
            "access_length_m": self.access_length_m,
            "access_eta_minutes": self.access_eta_minutes,
            "access_degraded_reason": self.access_degraded_reason,
            "access_simulated": self.access_simulated,
            "access_speed_assumption_kmh": ACCESS_ASSUMED_MEAN_SPEED_KMH,
            "routes": self.routes(),
            "graph_status": self.graph_status,
        }


# --------------------------------------------------------------------------
# recommender
# --------------------------------------------------------------------------


class ResponseRecommender:
    """Turns an assessed incident into an operator response plan."""

    def __init__(
        self,
        config: Optional[ResponseConfig] = None,
        road_graph: Optional[OSMRoadGraph] = None,
        facilities: Optional[list[dict[str, Any]]] = None,
    ):
        self.config = config or ResponseConfig()
        self.road_graph = road_graph or OSMRoadGraph(self.config)
        self._facilities = (
            list(facilities)
            if facilities is not None
            else load_facilities(self.config.facilities_path)
        )

    # -- main --------------------------------------------------------------

    def recommend(
        self,
        event_type: str,
        severity_label: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        road_name: Optional[str] = None,
        zone: Optional[str] = None,
        camera_id: Optional[str] = None,
        base_action: Optional[str] = None,
    ) -> ResponsePlan:
        tier = _normalise_tier(severity_label)

        actions: list[str] = []
        if base_action:
            actions.append(base_action)
        actions.extend(_ESCALATION.get(tier, []))
        type_action = _TYPE_ACTIONS.get(event_type)
        if type_action:
            actions.append(type_action)
        seen: set[str] = set()
        actions = [a for a in actions if not (a in seen or seen.add(a))]

        parts = [p for p in (road_name, zone) if p and p.lower() != "unknown"]
        where = ", ".join(parts)
        label = str(event_type or "incident").replace("_", " ").lower()
        headline = f"{tier.capitalize()} {label}" + (f" on {where}" if where else "")

        plan = ResponsePlan(
            headline=headline,
            actions=actions,
            camera_id=camera_id,
            event_type=event_type,
            severity_tier=tier,
        )

        self._attach_diversion(plan, event_type, tier, latitude, longitude)
        if event_type in _ALWAYS_ACCESS_TYPES or tier in _DIVERSION_TIERS:
            self._attach_access_route(plan, event_type, latitude, longitude)
        else:
            plan.access_degraded_reason = (
                f"No access route computed: a {tier.lower()} {label} does not "
                "warrant a responder dispatch route."
            )
        plan.graph_status = self.road_graph.status
        return plan

    # -- diversion ---------------------------------------------------------

    def _attach_diversion(
        self,
        plan: ResponsePlan,
        event_type: str,
        tier: str,
        lat: Optional[float],
        lon: Optional[float],
    ) -> None:
        if tier not in _DIVERSION_TIERS and event_type not in _ALWAYS_ROUTE_TYPES:
            plan.degraded_reason = (
                f"No diversion computed: severity {tier.capitalize()} does not "
                "warrant one."
            )
            return

        if lat is None or lon is None:
            plan.degraded_reason = (
                "No diversion computed: this camera has no coordinates. Add "
                "latitude/longitude to its config/cameras/<id>.json."
            )
            plan.actions.append(
                "Route diversion manually -- this camera is not geolocated"
            )
            return

        graph = self.road_graph.graph_for(lat, lon)
        if graph is None:
            plan.degraded_reason = f"No diversion computed: {self.road_graph.status}"
            plan.actions.append(
                "Route diversion manually -- automatic routing is unavailable"
            )
            return

        try:
            route = self._compute_diversion(graph, lat, lon)
        except Exception as exc:  # noqa: BLE001
            plan.degraded_reason = (
                f"No diversion computed: {type(exc).__name__}: {exc}"
            )
            plan.actions.append(
                "Route diversion manually -- automatic routing is unavailable"
            )
            return

        if not route:
            plan.degraded_reason = (
                "No diversion computed: no alternative path exists around the "
                "incident in the road graph."
            )
            return

        coords, length_m = route
        plan.diversion_available = True
        plan.route_latlon = coords
        plan.diversion_length_m = round(length_m, 1)
        plan.diversion_summary = (
            f"Alternative route around the incident: {len(coords)} nodes, "
            f"{length_m / 1000.0:.2f} km, computed on a road graph with the "
            "incident's own junction removed."
        )
        # The route is shown regardless of tier (see _ALWAYS_ROUTE_TYPES); only
        # the recommended ACTION of activating it is gated on severity, since
        # publishing a VMS advisory for a medium-tier event is not warranted
        # even when showing the path around it is informative.
        if tier in _DIVERSION_TIERS:
            plan.actions.append(
                "Activate the proposed diversion and publish it to VMS"
            )

    def _compute_diversion(
        self, graph: Any, lat: float, lon: float
    ) -> Optional[tuple[list[tuple[float, float]], float]]:
        """Route from upstream of the incident to downstream, with the
        incident's OWN NODE REMOVED so the path genuinely avoids it.

        Removing the node -- rather than penalising it, or routing between two
        points and hoping -- is what makes "this route avoids the incident" a
        property of the computation instead of an eyeball judgement.
        """
        import networkx as nx

        if graph is None:
            return None

        offset_deg = self.config.diversion_offset_m / 111_320.0
        origin = _nearest_node(graph, lat - offset_deg, lon)
        destination = _nearest_node(graph, lat + offset_deg, lon)
        incident = _nearest_node(graph, lat, lon)
        if origin == destination:
            return None

        pruned = graph.copy()
        if incident in pruned:
            pruned.remove_node(incident)

        try:
            path = nx.shortest_path(pruned, origin, destination, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        return _path_coords(pruned, path), _path_length_m(pruned, path)

    # -- SIMULATED responder access route ----------------------------------
    #
    # Distinct from the diversion above on purpose: opposite direction,
    # different graph query (the FULL graph -- reaching the incident is the
    # point), different audience, and a rendering style that keeps them apart.
    #
    # It NEVER contacts anyone. It is a routing computation on an
    # offline-cached OSM graph, reported with access_simulated=True carried on
    # the data itself.

    def facilities_by_distance(
        self, lat: float, lon: float, kinds: Optional[tuple[str, ...]] = None
    ) -> list[dict[str, Any]]:
        """Known facilities, nearest first by straight-line distance."""
        pool = self._facilities
        if kinds:
            preferred = [f for f in pool if str(f.get("kind", "")).lower() in kinds]
            others = [f for f in pool if str(f.get("kind", "")).lower() not in kinds]
        else:
            preferred, others = list(pool), []

        def decorate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            out = [
                {**f, "straight_line_m": haversine_m(
                    lat, lon, float(f["latitude"]), float(f["longitude"])
                )}
                for f in items
            ]
            out.sort(key=lambda f: f["straight_line_m"])
            return out

        return decorate(preferred) + decorate(others)

    def _compute_access_route(
        self, graph: Any, lat: float, lon: float, facility: dict[str, Any]
    ) -> Optional[tuple[list[tuple[float, float]], float]]:
        """Shortest path from a facility TOWARD the incident, on the FULL graph."""
        import networkx as nx

        if graph is None:
            return None

        origin = _nearest_node(
            graph, float(facility["latitude"]), float(facility["longitude"])
        )
        destination = _nearest_node(graph, lat, lon)
        if origin == destination:
            return None

        try:
            path = nx.shortest_path(graph, origin, destination, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        return _path_coords(graph, path), _path_length_m(graph, path)

    def _attach_access_route(
        self,
        plan: ResponsePlan,
        event_type: str,
        lat: Optional[float],
        lon: Optional[float],
    ) -> None:
        plan.access_simulated = True     # unconditionally, before anything else

        if lat is None or lon is None:
            plan.access_degraded_reason = (
                "No access route computed: this camera has no coordinates."
            )
            return

        kinds = _PREFERRED_FACILITY_KINDS.get(event_type)
        candidates = self.facilities_by_distance(lat, lon, kinds)
        if not candidates:
            plan.access_degraded_reason = (
                "No access route computed: no facility registry at "
                f"{self.config.facilities_path}."
            )
            return

        graph = self.road_graph.graph_for(lat, lon)
        if graph is None:
            plan.access_degraded_reason = (
                f"No access route computed: {self.road_graph.status}"
            )
            return

        # Nearest-first, but not nearest-only: the single nearest facility
        # sometimes snaps to the SAME graph node as the incident, which is an
        # uninteresting zero-length "route" rather than a failure. Move on to
        # the next-nearest instead of reporting no route when a perfectly good
        # one exists a little further out.
        n_tried = max(1, int(self.config.max_facility_candidates))
        facility = route = None
        for cand in candidates[:n_tried]:
            try:
                r = self._compute_access_route(graph, lat, lon, cand)
            except Exception as exc:  # noqa: BLE001
                plan.access_degraded_reason = (
                    f"No access route computed: {type(exc).__name__}: {exc}"
                )
                return
            if r and len(r[0]) > 1:
                facility, route = cand, r
                break

        if not route or facility is None:
            plan.access_degraded_reason = (
                "No access route computed: no path exists to the nearest "
                f"{len(candidates[:n_tried])} facilities in the road graph."
            )
            return

        coords, length_m = route
        assumed = ACCESS_ASSUMED_MEAN_SPEED_KMH
        eta_min = (length_m / 1000.0) / assumed * 60.0

        plan.access_facility = str(facility.get("name", "unnamed facility"))
        plan.access_facility_kind = str(facility.get("kind", "unknown"))
        plan.access_route_latlon = coords
        plan.access_length_m = round(length_m, 1)
        plan.access_eta_minutes = round(eta_min, 1)
        plan.access_route_summary = (
            f"{plan.access_facility}, {length_m / 1000.0:.2f} km by road, "
            f"~{eta_min:.0f} min at an assumed {assumed:.0f} km/h mean speed "
            "(no live traffic data). SIMULATED: no facility was contacted and "
            "no dispatch was issued."
        )

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "osmnx_enabled": self.config.enable_osmnx,
            "osmnx_installed": _osmnx_installed(),
            "graph_status": self.road_graph.status,
            "graph_available": self.road_graph.available,
            "place": self.config.osm_place,
            "cache_dir": str(_resolve(self.config.graph_cache_dir)),
            "facilities_loaded": len(self._facilities),
            "facilities_path": str(_resolve(self.config.facilities_path)),
            "access_speed_assumption_kmh": ACCESS_ASSUMED_MEAN_SPEED_KMH,
            "access_routes_are_simulated": True,
        }


def _osmnx_installed() -> bool:
    try:
        import osmnx  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


__all__ = [
    "ACCESS_ASSUMED_MEAN_SPEED_KMH",
    "ACCESS_STYLE",
    "DIVERSION_STYLE",
    "OSMRoadGraph",
    "ResponseConfig",
    "ResponsePlan",
    "ResponseRecommender",
    "haversine_m",
    "load_facilities",
]
