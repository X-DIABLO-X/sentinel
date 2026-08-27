"""Planar geometry helpers.

Everything the event engines need to reason about *where* a road user is and
*which way* it is going. Deliberately dependency-light: numpy + cv2 only, no
shapely, so the install stays small and reproducible.

Conventions used throughout NETRA
---------------------------------
* Image coordinates are (x, y) in pixels, origin top-left, y increasing down.
* A road user's *ground point* is the bottom-centre of its bounding box. That
  approximates the point where the vehicle contacts the road, which is what we
  want for corridor membership -- the box centroid floats above the road and
  drifts with vehicle height.
* A corridor's ``direction`` is a unit vector in image space pointing the way
  traffic is legally allowed to travel.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]


# --------------------------------------------------------------------------
# basic vector maths
# --------------------------------------------------------------------------

def unit(v: Sequence[float]) -> np.ndarray:
    """Return ``v`` scaled to unit length; a zero vector is returned unchanged."""
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return np.zeros_like(a)
    return a / n


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two 2-D vectors, clipped to [-1, 1].

    This is the single most important number in the wrong-way engine:
    ``+1`` means travelling exactly with the corridor, ``-1`` exactly against.
    """
    ua, ub = unit(a), unit(b)
    if not ua.any() or not ub.any():
        return 0.0
    return float(np.clip(np.dot(ua, ub), -1.0, 1.0))


def heading_degrees(v: Sequence[float]) -> float:
    """Compass-style heading in degrees for a screen-space vector.

    0 deg = up the frame, 90 deg = right. Used only for human-readable
    evidence; nothing decides on it.
    """
    x, y = float(v[0]), float(v[1])
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return 0.0
    return (math.degrees(math.atan2(x, -y)) + 360.0) % 360.0


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Unsigned angle between two vectors, in degrees (0..180)."""
    return math.degrees(math.acos(cosine(a, b)))


# --------------------------------------------------------------------------
# polygons
# --------------------------------------------------------------------------

def as_contour(polygon: Iterable[Point]) -> np.ndarray:
    """Convert a list of points to the int32 contour shape cv2 expects."""
    return np.asarray(list(polygon), dtype=np.float32).reshape(-1, 1, 2)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """True when ``point`` lies inside (or on the edge of) ``polygon``."""
    if polygon is None or len(polygon) < 3:
        return False
    contour = as_contour(polygon)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def polygon_area(polygon: Sequence[Point]) -> float:
    """Absolute area of a polygon in square pixels."""
    if polygon is None or len(polygon) < 3:
        return 0.0
    return abs(float(cv2.contourArea(as_contour(polygon))))


def polygon_mask(polygon: Sequence[Point], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon to a uint8 mask of ``shape`` = (height, width)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if polygon is not None and len(polygon) >= 3:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask


def polygon_centroid(polygon: Sequence[Point]) -> Point:
    """Area-weighted centroid, falling back to the mean vertex if degenerate."""
    pts = np.asarray(polygon, dtype=np.float64)
    if len(pts) == 0:
        return (0.0, 0.0)
    m = cv2.moments(as_contour(polygon))
    if abs(m["m00"]) > 1e-9:
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


# --------------------------------------------------------------------------
# boxes
# --------------------------------------------------------------------------

def box_ground_point(box: Sequence[float]) -> Point:
    """Bottom-centre of an ``(x1, y1, x2, y2)`` box -- our road-contact proxy."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def box_centre(box: Sequence[float]) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two arrays of boxes, shape (Na, 4) x (Nb, 4)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def boxes_occupancy(boxes: Sequence[Sequence[float]],
                    polygon: Sequence[Point],
                    shape: tuple[int, int]) -> float:
    """Fraction of ``polygon`` covered by the union of ``boxes``.

    Rasterised rather than computed analytically because boxes overlap heavily
    in dense traffic and we want *coverage*, not summed area -- summing areas
    double-counts and can exceed 1.0, which would silently corrupt severity.
    """
    if polygon is None or len(polygon) < 3:
        return 0.0
    road = polygon_mask(polygon, shape)
    road_px = int(np.count_nonzero(road))
    if road_px == 0:
        return 0.0
    veh = np.zeros(shape[:2], dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1 = max(0, min(shape[1] - 1, x1))
        x2 = max(0, min(shape[1], x2))
        y1 = max(0, min(shape[0] - 1, y1))
        y2 = max(0, min(shape[0], y2))
        if x2 > x1 and y2 > y1:
            veh[y1:y2, x1:x2] = 255
    covered = int(np.count_nonzero(cv2.bitwise_and(road, veh)))
    return float(covered / road_px)


# --------------------------------------------------------------------------
# homography (optional -- only used when a camera has been calibrated)
# --------------------------------------------------------------------------

class Homography:
    """Maps image points onto a metric ground plane.

    Only constructed when a camera config supplies >= 4 image/world point
    pairs. Without it NETRA reports speeds in px/s and lengths in vehicles,
    never in km/h or metres -- see LIMITATIONS.md.
    """

    def __init__(self, image_pts: Sequence[Point], world_pts: Sequence[Point]):
        if len(image_pts) < 4 or len(world_pts) < 4:
            raise ValueError("homography needs at least 4 point correspondences")
        src = np.asarray(image_pts, dtype=np.float32)
        dst = np.asarray(world_pts, dtype=np.float32)
        self.matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if self.matrix is None:
            raise ValueError("could not solve homography from the given points")

    def to_world(self, point: Point) -> Point:
        p = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        w = cv2.perspectiveTransform(p, self.matrix)[0][0]
        return (float(w[0]), float(w[1]))

    def distance(self, a: Point, b: Point) -> float:
        """Ground-plane distance between two image points, in world units."""
        wa, wb = self.to_world(a), self.to_world(b)
        return float(math.hypot(wb[0] - wa[0], wb[1] - wa[1]))

    def to_dict(self) -> dict:
        return {"matrix": self.matrix.tolist()}


def robust_direction(points: Sequence[Point], min_span: float = 4.0) -> np.ndarray | None:
    """Estimate a stable direction of travel from a short trajectory.

    Averaging the first and last thirds of the window rather than differencing
    the two endpoint samples -- endpoint differencing amplifies detector jitter,
    which is exactly what produces phantom wrong-way alerts. This mirrors the
    half-window averaging used by Ghahremannezhad et al. (IEEE IST 2022).

    Returns ``None`` when the track has not moved far enough for the direction
    to mean anything.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 4:
        return None
    k = max(1, len(pts) // 3)
    head = pts[:k].mean(axis=0)
    tail = pts[-k:].mean(axis=0)
    delta = tail - head
    if float(np.linalg.norm(delta)) < min_span:
        return None
    return unit(delta)
