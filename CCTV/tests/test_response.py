"""Tests for netra/response.py -- diversion and simulated access routing.

Every test here runs WITHOUT NETWORK ACCESS. Nothing downloads an OSM graph:
routing tests inject a synthetic lattice graph via `OSMRoadGraph.set_graph`,
and degradation tests force each failure mode explicitly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import pytest

from netra.events.base import BLOCKAGE, COLLISION, QUEUE
from netra.response import (
    ACCESS_ASSUMED_MEAN_SPEED_KMH,
    OSMRoadGraph,
    ResponseConfig,
    ResponsePlan,
    ResponseRecommender,
    haversine_m,
    load_facilities,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# an arbitrary but fixed incident point (Cuttack link road demo camera)
LAT, LON = 20.4625, 85.8830


# --------------------------------------------------------------------------
# fixtures: a synthetic road lattice, so no OSM download ever happens
# --------------------------------------------------------------------------


def _lattice(lat: float = LAT, lon: float = LON, n: int = 9, step_deg: float = 0.004):
    """A MultiDiGraph lattice shaped like what OSMnx returns.

    Integer node ids, x/y node attributes, `length` edge attributes, edges in
    both directions. Centred on (lat, lon) so the centre node is the incident.
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "epsg:4326"
    half = n // 2
    for i in range(n):          # rows: north/south
        for j in range(n):      # cols: east/west
            g.add_node(
                i * 100 + j,
                y=lat + (i - half) * step_deg,
                x=lon + (j - half) * step_deg,
            )
    for i in range(n):
        for j in range(n):
            here = i * 100 + j
            for di, dj in ((0, 1), (1, 0)):
                oi, oj = i + di, j + dj
                if oi >= n or oj >= n:
                    continue
                there = oi * 100 + oj
                length = haversine_m(
                    g.nodes[here]["y"], g.nodes[here]["x"],
                    g.nodes[there]["y"], g.nodes[there]["x"],
                )
                g.add_edge(here, there, length=length)
                g.add_edge(there, here, length=length)
    return g


FACILITIES = [
    {
        "name": "Test Hospital (placeholder)",
        "kind": "hospital",
        "latitude": LAT + 0.005,
        "longitude": LON + 0.003,
        "coordinate_source": "illustrative_placeholder",
    },
    {
        "name": "Test Police Station (placeholder)",
        "kind": "police",
        "latitude": LAT - 0.006,
        "longitude": LON + 0.004,
        "coordinate_source": "illustrative_placeholder",
    },
]


@pytest.fixture()
def offline_recommender(tmp_path):
    """A recommender with a pre-installed graph and no ability to reach OSM."""
    cfg = ResponseConfig(
        enable_osmnx=False,                      # belt and braces: no fetching
        graph_cache_dir=str(tmp_path / "cache"),
        diversion_offset_m=700.0,
    )
    graph = OSMRoadGraph(cfg)
    graph.set_graph(_lattice(), LAT, LON, status="synthetic test lattice")
    return ResponseRecommender(config=cfg, road_graph=graph, facilities=FACILITIES)


# --------------------------------------------------------------------------
# haversine
# --------------------------------------------------------------------------


def test_haversine_one_degree_of_longitude_at_the_equator():
    """One degree of longitude at the equator is ~111.19 km on a sphere."""
    d = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert 111_100 < d < 111_400, d


def test_haversine_known_city_pair():
    """Bengaluru (12.9716, 77.5946) -> Delhi (28.6139, 77.2090): ~1740 km."""
    d = haversine_m(12.9716, 77.5946, 28.6139, 77.2090)
    assert 1_730_000 < d < 1_755_000, d


def test_haversine_is_zero_and_symmetric():
    assert haversine_m(LAT, LON, LAT, LON) == pytest.approx(0.0, abs=1e-6)
    assert haversine_m(LAT, LON, 12.97, 77.59) == pytest.approx(
        haversine_m(12.97, 77.59, LAT, LON), rel=1e-12
    )


# --------------------------------------------------------------------------
# graceful, explicit degradation
# --------------------------------------------------------------------------


def test_degrades_explicitly_when_camera_has_no_coordinates(offline_recommender):
    plan = offline_recommender.recommend(
        event_type=COLLISION, severity_label="High",
        latitude=None, longitude=None, camera_id="ACCIDENTS_1",
    )
    assert plan.actions, "operational actions must survive degradation"
    assert plan.degraded_reason
    assert "coordinates" in plan.degraded_reason.lower()
    assert plan.route_latlon == []
    assert plan.diversion_available is False
    assert plan.access_route_latlon == []
    assert plan.access_degraded_reason
    assert plan.routes() == [], "no route was computed, so none may be drawn"


def test_degrades_explicitly_when_osmnx_is_disabled_or_missing(tmp_path):
    """enable_osmnx=False exercises the same code path as osmnx not installed:
    graph_for() returns None with a human-readable reason."""
    cfg = ResponseConfig(
        enable_osmnx=False, graph_cache_dir=str(tmp_path / "cache")
    )
    rec = ResponseRecommender(
        config=cfg, road_graph=OSMRoadGraph(cfg), facilities=FACILITIES
    )
    plan = rec.recommend(
        event_type=COLLISION, severity_label="High",
        latitude=LAT, longitude=LON, camera_id="CUTTACK_LINK_01",
    )
    assert plan.actions
    assert plan.degraded_reason.startswith("No diversion computed:")
    assert "enable_osmnx" in plan.degraded_reason
    assert plan.route_latlon == []
    assert plan.access_degraded_reason.startswith("No access route computed:")
    assert plan.access_route_latlon == []
    assert any("manually" in a for a in plan.actions), plan.actions


def test_degrades_explicitly_when_graph_download_fails(tmp_path, monkeypatch):
    """A failed fetch must report why, not silently produce nothing."""
    cfg = ResponseConfig(enable_osmnx=True, graph_cache_dir=str(tmp_path / "cache"))
    graph = OSMRoadGraph(cfg)

    class _Boom:
        @staticmethod
        def graph_from_point(*_a, **_k):
            raise OSError("simulated: no network at demo time")

        @staticmethod
        def load_graphml(*_a, **_k):
            raise OSError("simulated")

        @staticmethod
        def save_graphml(*_a, **_k):
            raise OSError("simulated")

    monkeypatch.setitem(__import__("sys").modules, "osmnx", _Boom)

    rec = ResponseRecommender(config=cfg, road_graph=graph, facilities=FACILITIES)
    plan = rec.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    assert plan.actions
    assert plan.degraded_reason.startswith("No diversion computed:")
    assert "OSError" in plan.degraded_reason
    assert plan.route_latlon == []
    assert plan.access_route_latlon == []
    assert plan.access_degraded_reason


def test_degrades_explicitly_when_no_facility_registry(tmp_path):
    cfg = ResponseConfig(
        enable_osmnx=False, graph_cache_dir=str(tmp_path / "cache")
    )
    graph = OSMRoadGraph(cfg)
    graph.set_graph(_lattice(), LAT, LON)
    rec = ResponseRecommender(config=cfg, road_graph=graph, facilities=[])
    plan = rec.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    assert plan.access_route_latlon == []
    assert "facility registry" in plan.access_degraded_reason
    # the diversion is independent of the facility registry and still computes
    assert plan.diversion_available is True


def test_low_severity_queue_states_why_no_diversion(offline_recommender):
    plan = offline_recommender.recommend(
        event_type=QUEUE, severity_label="Low", latitude=LAT, longitude=LON
    )
    assert plan.route_latlon == []
    assert "does not warrant" in plan.degraded_reason
    assert plan.actions


# --------------------------------------------------------------------------
# the diversion route actually avoids the incident
# --------------------------------------------------------------------------


def test_diversion_route_omits_the_incident_node(offline_recommender):
    plan = offline_recommender.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    assert plan.diversion_available is True
    assert len(plan.route_latlon) > 2
    # the incident's own junction is the lattice centre; it must not be on the
    # returned path, because the route was computed with that node removed
    for plat, plon in plan.route_latlon:
        assert haversine_m(plat, plon, LAT, LON) > 1.0, (plat, plon)
    assert plan.diversion_length_m and plan.diversion_length_m > 0
    assert "removed" in plan.diversion_summary


def test_high_severity_adds_the_vms_action_but_medium_does_not(offline_recommender):
    high = offline_recommender.recommend(
        event_type=BLOCKAGE, severity_label="High", latitude=LAT, longitude=LON
    )
    medium = offline_recommender.recommend(
        event_type=COLLISION, severity_label="Medium", latitude=LAT, longitude=LON
    )
    assert any("VMS" in a for a in high.actions)
    # a medium collision still gets the ROUTE (arithmetic is unconditional)...
    assert medium.diversion_available is True
    # ...but not the recommendation to activate it
    assert not any("VMS" in a for a in medium.actions)


# --------------------------------------------------------------------------
# the access route is always, unconditionally, flagged as simulated
# --------------------------------------------------------------------------


def test_access_simulated_is_true_on_a_computed_route(offline_recommender):
    plan = offline_recommender.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    assert plan.access_simulated is True
    assert plan.access_route_latlon
    assert plan.access_facility
    assert "SIMULATED" in plan.access_route_summary
    assert "assumed" in plan.access_route_summary
    assert str(int(ACCESS_ASSUMED_MEAN_SPEED_KMH)) in plan.access_route_summary
    assert plan.as_dict()["access_simulated"] is True
    assert plan.as_dict()["access_speed_assumption_kmh"] == ACCESS_ASSUMED_MEAN_SPEED_KMH


@pytest.mark.parametrize(
    "event_type,severity,lat,lon",
    [
        (COLLISION, "High", LAT, LON),          # computed
        (COLLISION, "High", None, None),        # degraded: no coordinates
        (QUEUE, "Low", LAT, LON),               # degraded: tier too low
        (BLOCKAGE, "High", LAT, LON),           # computed
    ],
)
def test_access_simulated_is_true_in_every_path(
    offline_recommender, event_type, severity, lat, lon
):
    """access_simulated must never be False -- computed, degraded or skipped."""
    plan = offline_recommender.recommend(
        event_type=event_type, severity_label=severity, latitude=lat, longitude=lon
    )
    assert plan.access_simulated is True
    assert plan.as_dict()["access_simulated"] is True
    for route in plan.routes():
        if route["kind"] == "access":
            assert route["simulated"] is True


def test_a_bare_plan_defaults_to_simulated():
    assert ResponsePlan(headline="x").access_simulated is True


def test_access_eta_uses_the_stated_speed_assumption(offline_recommender):
    plan = offline_recommender.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    expected = (plan.access_length_m / 1000.0) / ACCESS_ASSUMED_MEAN_SPEED_KMH * 60.0
    assert plan.access_eta_minutes == pytest.approx(expected, abs=0.1)


# --------------------------------------------------------------------------
# the two routes stay distinguishable
# --------------------------------------------------------------------------


def test_diversion_and_access_routes_are_separate_records(offline_recommender):
    plan = offline_recommender.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    routes = plan.routes()
    assert len(routes) == 2
    kinds = [r["kind"] for r in routes]
    assert sorted(kinds) == ["access", "diversion"]

    div = next(r for r in routes if r["kind"] == "diversion")
    acc = next(r for r in routes if r["kind"] == "access")

    # solid red for other traffic, dashed blue for responders
    assert div["dashed"] is False and acc["dashed"] is True
    assert div["color"] != acc["color"]
    assert div["audience"] == "other_traffic"
    assert acc["audience"] == "responders"
    assert div["simulated"] is False and acc["simulated"] is True
    assert div["latlon"] != acc["latlon"]
    assert "SIMULATED" in acc["label"]

    # and they stay separate through serialisation
    payload = json.loads(json.dumps(plan.as_dict()))
    assert payload["route_latlon"] != payload["access_route_latlon"]
    assert len(payload["routes"]) == 2


def test_access_route_ends_at_the_incident_and_diversion_does_not(offline_recommender):
    """The two routes point in opposite directions on purpose: the access route
    reaches the incident (full graph); the diversion avoids it (node removed)."""
    plan = offline_recommender.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    last_lat, last_lon = plan.access_route_latlon[-1]
    assert haversine_m(last_lat, last_lon, LAT, LON) < 1.0
    assert all(
        haversine_m(p[0], p[1], LAT, LON) > 1.0 for p in plan.route_latlon
    )


# --------------------------------------------------------------------------
# the shipped registry
# --------------------------------------------------------------------------


def test_shipped_facilities_registry_is_honest_and_loadable():
    path = REPO_ROOT / "config" / "facilities.json"
    assert path.exists(), path
    raw = json.loads(path.read_text(encoding="utf-8"))
    about = raw["_about"]
    assert "PLACEHOLDER" in about["STATUS"].upper()
    facilities = load_facilities(path)
    assert facilities
    for f in facilities:
        assert f["kind"] in {"hospital", "police", "fire"}, f
        assert f["coordinate_source"] in {
            "illustrative_placeholder",
            "openstreetmap_amenity_tag",
        }, f
        assert -90.0 <= float(f["latitude"]) <= 90.0
        assert -180.0 <= float(f["longitude"]) <= 180.0
        if f["coordinate_source"] == "illustrative_placeholder":
            assert "(placeholder)" in f["name"], f["name"]


def test_load_facilities_on_a_missing_registry_returns_empty(tmp_path):
    assert load_facilities(tmp_path / "nope.json") == []


def test_status_reports_the_simulation_flag(offline_recommender):
    st = offline_recommender.status()
    assert st["access_routes_are_simulated"] is True
    assert st["access_speed_assumption_kmh"] == ACCESS_ASSUMED_MEAN_SPEED_KMH
    assert st["facilities_loaded"] == len(FACILITIES)


def test_facility_kind_preference_beats_raw_proximity(offline_recommender):
    """The hospital is the nearest facility, but a queue is a police matter, so
    a queue must route from the police station and a collision -- which accepts
    any of hospital/police/fire -- from the nearest, i.e. the hospital."""
    police = next(f for f in FACILITIES if f["kind"] == "police")
    hospital = next(f for f in FACILITIES if f["kind"] == "hospital")
    assert haversine_m(LAT, LON, hospital["latitude"], hospital["longitude"]) < \
        haversine_m(LAT, LON, police["latitude"], police["longitude"])

    queue = offline_recommender.recommend(
        event_type=QUEUE, severity_label="High", latitude=LAT, longitude=LON
    )
    assert queue.access_facility_kind == "police"

    collision = offline_recommender.recommend(
        event_type=COLLISION, severity_label="High", latitude=LAT, longitude=LON
    )
    assert collision.access_facility_kind == "hospital"


def test_no_network_was_needed(offline_recommender):
    """Sanity: everything above ran against an injected graph. If a test ever
    starts hitting the network, this asserts the fixture is still offline."""
    assert offline_recommender.config.enable_osmnx is False
    assert math.isfinite(haversine_m(LAT, LON, LAT + 1, LON + 1))
