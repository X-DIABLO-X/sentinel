"""Per-camera scene model: what the camera knows about its own road.

This is the highest-leverage component in NETRA and the answer to the hardest
question the project faces -- *"you were given no footage, so how do you know it
works on our cameras?"*

A fixed traffic camera looks at the same road for years. Asking a segmentation
network to rediscover where the road is, and which way traffic may legally
travel, 30 times a second is both expensive and unreliable. Worse, no network
can tell you the *legal* direction of a lane: that is a fact about the world,
not about the pixels. So a human states it once, in about ninety seconds, and
the system uses it forever.

The scene model holds:

* **corridors** -- directional travel polygons, each with the unit vector
  traffic is permitted to move along, plus flags for the boundaries it shares
  with its neighbours (solid = crossing forbidden, dashed = permitted)
* **exclusion zones** -- junction boxes, turning areas, parking. Direction
  logic is suspended inside these, because a legal turn looks exactly like a
  wrong-way violation to a dot product
* **no-stop zones** -- where a stationary vehicle is an incident rather than a
  parked car
* an optional **homography**, and only with one do we ever speak in metres

Everything is JSON. A camera is a config file, not a code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .geometry import (
    Homography,
    as_contour,
    cosine,
    heading_degrees,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    unit,
)

Point = tuple[float, float]


@dataclass
class Corridor:
    """A directional stretch of carriageway."""

    id: str
    polygon: list[Point]
    direction: np.ndarray                 # unit vector, image space
    name: str = ""
    lanes: int = 1
    solid_boundary_with: list[str] = field(default_factory=list)
    baseline_speed_px: float | None = None    # learned, see SceneModel.learn_baseline

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def centroid(self) -> Point:
        return polygon_centroid(self.polygon)

    @property
    def heading(self) -> float:
        return heading_degrees(self.direction)

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def alignment(self, direction: Sequence[float]) -> float:
        """Cosine between a track's heading and the legal one. -1 = fully wrong-way."""
        return cosine(direction, self.direction)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon],
            "direction": [round(float(self.direction[0]), 4),
                          round(float(self.direction[1]), 4)],
            "lanes": self.lanes,
            "solid_boundary_with": list(self.solid_boundary_with),
            "baseline_speed_px": self.baseline_speed_px,
        }


@dataclass
class Zone:
    """A named polygon with no direction -- exclusion, no-stop, footway, etc."""

    id: str
    polygon: list[Point]
    kind: str = "exclusion"
    name: str = ""

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in self.polygon],
        }


class SceneModel:
    """Everything one camera knows about its own view."""

    def __init__(self,
                 camera_id: str,
                 source: str = "",
                 name: str = "",
                 zone: str = "",
                 road_name: str = "",
                 latitude: float | None = None,
                 longitude: float | None = None,
                 frame_size: tuple[int, int] | None = None,
                 analysis_fps: float = 8.0,
                 corridors: list[Corridor] | None = None,
                 zones: list[Zone] | None = None,
                 homography: Homography | None = None,
                 road_edge_id: str | None = None,
                 notes: str = "") -> None:
        self.camera_id = camera_id
        self.source = source
        self.name = name or camera_id
        self.zone = zone
        self.road_name = road_name
        self.latitude = latitude
        self.longitude = longitude
        self.frame_size = frame_size          # (width, height)
        self.analysis_fps = analysis_fps
        self.corridors = corridors or []
        self.zones = zones or []
        self.homography = homography
        self.road_edge_id = road_edge_id
        self.notes = notes

    def scaled_to(self, frame_size: tuple[int, int]) -> "SceneModel":
        """Return this calibration expressed in another analysis resolution.

        Saved camera polygons are pixel coordinates. Reusing a 1280x720
        calibration during a 1920x1080 run without scaling silently moves every
        corridor and produces wrong-way/lane errors even though both views have
        the same aspect ratio.
        """
        target = tuple(map(int, frame_size))
        if not self.frame_size or tuple(self.frame_size) == target:
            return self
        old_w, old_h = self.frame_size
        new_w, new_h = target
        sx, sy = new_w / max(old_w, 1), new_h / max(old_h, 1)

        corridors = []
        for c in self.corridors:
            direction = unit([float(c.direction[0]) * sx,
                              float(c.direction[1]) * sy])
            corridors.append(Corridor(
                id=c.id, name=c.name,
                polygon=[(float(x) * sx, float(y) * sy) for x, y in c.polygon],
                direction=direction, lanes=c.lanes,
                solid_boundary_with=list(c.solid_boundary_with),
                baseline_speed_px=(None if c.baseline_speed_px is None else
                                   float(c.baseline_speed_px) * 0.5 * (sx + sy)),
            ))
        zones = [Zone(id=z.id, kind=z.kind, name=z.name,
                      polygon=[(float(x) * sx, float(y) * sy)
                               for x, y in z.polygon])
                 for z in self.zones]
        hom = self.homography
        if hom is not None:
            scaled_hom = object.__new__(Homography)
            scaled_hom.matrix = np.asarray(hom.matrix, dtype=np.float64) @ np.array(
                [[1.0 / sx, 0.0, 0.0],
                 [0.0, 1.0 / sy, 0.0],
                 [0.0, 0.0, 1.0]], dtype=np.float64)
            hom = scaled_hom
        return SceneModel(
            camera_id=self.camera_id, source=self.source, name=self.name,
            zone=self.zone, road_name=self.road_name,
            road_edge_id=self.road_edge_id, latitude=self.latitude,
            longitude=self.longitude, frame_size=target,
            analysis_fps=self.analysis_fps, corridors=corridors, zones=zones,
            homography=hom, notes=self.notes,
        )

    # -- lookups -----------------------------------------------------------
    def corridor_at(self, point: Point) -> Corridor | None:
        """Which corridor a ground point falls in, or None.

        Corridors *should* be disjoint -- a point on the road belongs to exactly
        one lane -- and the calibration tool enforces that. But hand-drawn
        polygons overlap, and a naive "first match wins" turns that overlap into
        phantom wrong-way alerts: a vehicle in the eastbound stream whose ground
        point strays into the westbound polygon is instantly a violator.

        So when polygons do overlap, the point is awarded to the corridor it is
        furthest *inside* -- the one whose boundary is most distant. That is a
        real geometric criterion rather than list order, and it degrades
        gracefully instead of silently.
        """
        import cv2

        best, best_depth = None, -1e18
        for c in self.corridors:
            if len(c.polygon) < 3:
                continue
            depth = cv2.pointPolygonTest(
                as_contour(c.polygon), (float(point[0]), float(point[1])), True
            )
            if depth >= 0 and depth > best_depth:
                best, best_depth = c, depth
        return best

    def corridor_by_id(self, cid: str) -> Corridor | None:
        for c in self.corridors:
            if c.id == cid:
                return c
        return None

    def in_exclusion(self, point: Point) -> bool:
        """True inside a junction/turning box, where direction logic is suspended.

        Without this, every legal right turn at an intersection fires the
        wrong-way engine. It is the single most important false-positive guard
        in the system.
        """
        return any(z.contains(point) for z in self.zones if z.kind == "exclusion")

    def in_no_stop(self, point: Point) -> bool:
        return any(z.contains(point) for z in self.zones if z.kind == "no_stop")

    def zones_of(self, kind: str) -> list[Zone]:
        return [z for z in self.zones if z.kind == kind]

    @property
    def has_metric_scale(self) -> bool:
        """Whether this camera may report metres and km/h at all."""
        return self.homography is not None

    @property
    def legal_direction_reviewed(self) -> bool:
        """Whether corridor directions were confirmed as legal, not inferred.

        Auto-calibration deliberately writes ``DRAFT`` into the notes.  That
        marker is a machine-readable safety boundary: majority traffic flow is
        an observation, while legal direction is external road metadata.
        """
        note = (self.notes or "").upper()
        return bool(self.corridors) and "DRAFT" not in note and "UNCALIBRATED" not in note

    @property
    def road_polygon(self) -> list[Point]:
        """Union hull of all corridors, used for whole-carriageway occupancy."""
        pts: list[Point] = []
        for c in self.corridors:
            pts.extend(c.polygon)
        if len(pts) < 3:
            return []
        import cv2
        hull = cv2.convexHull(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2))
        return [(float(p[0][0]), float(p[0][1])) for p in hull]

    # -- boundary semantics ------------------------------------------------
    def boundary_is_solid(self, a: str, b: str) -> bool:
        """Is the marking between corridors ``a`` and ``b`` a solid line?

        Crossing a solid line is a violation; crossing a dashed one is an
        ordinary lane change. Encoding this per-camera is what lets NETRA
        report *wrong lane crossing* rather than merely *lane change*.
        """
        ca, cb = self.corridor_by_id(a), self.corridor_by_id(b)
        if ca and b in ca.solid_boundary_with:
            return True
        if cb and a in cb.solid_boundary_with:
            return True
        return False

    def is_opposing(self, a: str, b: str, tol: float = -0.5) -> bool:
        """True when two corridors carry traffic in broadly opposite directions."""
        ca, cb = self.corridor_by_id(a), self.corridor_by_id(b)
        if not ca or not cb:
            return False
        return cosine(ca.direction, cb.direction) < tol

    # -- learned baselines -------------------------------------------------
    def learn_baseline(self, corridor_id: str, speeds: Sequence[float]) -> float | None:
        """Record a corridor's free-flow speed from observed nominal traffic.

        Severity's flow-loss term needs something to be a loss *relative to*.
        We take a high percentile rather than the mean so that a corridor that
        spent part of the warm-up congested still yields a sensible free-flow
        reference.
        """
        vals = [s for s in speeds if s > 0.5]
        if len(vals) < 20:
            return None
        baseline = float(np.percentile(vals, 85))
        c = self.corridor_by_id(corridor_id)
        if c is not None:
            c.baseline_speed_px = baseline
        return baseline

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": self.source,
            "zone": self.zone,
            "road_name": self.road_name,
            "road_edge_id": self.road_edge_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "analysis_fps": self.analysis_fps,
            "corridors": [c.to_dict() for c in self.corridors],
            "zones": [z.to_dict() for z in self.zones],
            "notes": self.notes,
        }
        if self.homography is not None:
            d["homography"] = self.homography.to_dict()
        return d

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "SceneModel":
        corridors = []
        for c in d.get("corridors", []):
            corridors.append(Corridor(
                id=c["id"],
                name=c.get("name", ""),
                polygon=[tuple(map(float, p)) for p in c["polygon"]],
                direction=unit(c["direction"]),
                lanes=int(c.get("lanes", 1)),
                solid_boundary_with=list(c.get("solid_boundary_with", [])),
                baseline_speed_px=c.get("baseline_speed_px"),
            ))
        zones = [
            Zone(id=z["id"], kind=z.get("kind", "exclusion"), name=z.get("name", ""),
                 polygon=[tuple(map(float, p)) for p in z["polygon"]])
            for z in d.get("zones", [])
        ]
        hom = None
        if d.get("homography") and d["homography"].get("matrix"):
            hom = object.__new__(Homography)
            hom.matrix = np.asarray(d["homography"]["matrix"], dtype=np.float64)
        fs = d.get("frame_size")
        return cls(
            camera_id=d["camera_id"],
            source=d.get("source", ""),
            name=d.get("name", ""),
            zone=d.get("zone", ""),
            road_name=d.get("road_name", ""),
            road_edge_id=d.get("road_edge_id"),
            latitude=d.get("latitude"),
            longitude=d.get("longitude"),
            frame_size=tuple(fs) if fs else None,
            analysis_fps=float(d.get("analysis_fps", 8.0)),
            corridors=corridors,
            zones=zones,
            homography=hom,
            notes=d.get("notes", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SceneModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self) -> str:
        return (f"<SceneModel {self.camera_id} corridors={len(self.corridors)} "
                f"zones={len(self.zones)} metric={self.has_metric_scale}>")


def load_all(directory: str | Path) -> dict[str, SceneModel]:
    """Load every camera config in a directory, keyed by camera id."""
    out: dict[str, SceneModel] = {}
    d = Path(directory)
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            m = SceneModel.load(p)
            out[m.camera_id] = m
        except Exception as exc:                       # pragma: no cover
            print(f"[scene] skipped {p.name}: {exc}")
    return out
