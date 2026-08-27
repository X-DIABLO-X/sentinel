"""Ground-plane footprints: the 2-D patch of road a vehicle actually occupies.

The mistake this module corrects
--------------------------------
Every proximity test in this system used to compare bounding boxes, and a
bounding box spans a vehicle's *height*. A collision does not happen in the air;
it happens on the road surface. Under perspective those are very different
things: a bus in the background and a car in the foreground can have boxes that
nearly touch while sitting many metres apart on the ground. Measuring the wrong
quantity and then tuning thresholds against it is why proximity kept firing on
ordinary traffic.

The correct primitive is the vehicle's **footprint** -- the patch of road it
stands on. The bottom edge of a detection box approximates the contact line, so
the footprint is modelled as a small ellipse centred at the bottom-centre point,
and two vehicles have collided only when those ellipses overlap or come within a
whisker of it.

Why an ellipse, and why these proportions
-----------------------------------------
Seen from a traffic camera the road recedes, so a vehicle's footprint projects
to something much shallower than it is wide. Its width in the image is the box
width; its depth is a fraction of that, set by how obliquely the camera views the
road. ``depth_ratio`` carries that, defaulting to 0.35 for a typical elevated
view -- lower for a shallow, near-horizontal camera, higher for a near-overhead
drone shot.

The happy consequence is that this scales itself. Image size is proportional to
1/depth, so a footprint automatically shrinks with distance and one threshold
holds across the whole frame, at any resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Footprint:
    """The elliptical patch of road under one vehicle."""

    cx: float
    cy: float
    a: float          # semi-axis across the road (image x)
    b: float          # semi-axis along viewing depth (image y)

    @classmethod
    def from_box(cls, box, depth_ratio: float = 0.35, inflate: float = 1.0) -> "Footprint":
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(1e-6, x2 - x1)
        a = 0.5 * w * inflate
        b = max(2.0, 0.5 * w * depth_ratio * inflate)
        return cls(cx=(x1 + x2) / 2.0, cy=y2, a=a, b=b)

    def as_tuple(self):
        return (self.cx, self.cy, self.a, self.b)


def separation(f1: Footprint, f2: Footprint) -> float:
    """Normalised gap between two footprints.

    Returns the distance between centres expressed in units of the summed
    semi-axes, along each direction independently:

        0.0  -- concentric
        <1.0 -- the ellipses OVERLAP; the vehicles are sharing road surface
        1.0  -- exactly touching
        >1.0 -- a clear gap, in multiples of a vehicle footprint

    Independent normalisation per axis matters because the footprint is
    deliberately shallow: two vehicles side by side and two vehicles nose-to-tail
    are separated by very different pixel counts but by comparable multiples of
    their own footprints.
    """
    dx = (f1.cx - f2.cx) / max(f1.a + f2.a, 1e-6)
    dy = (f1.cy - f2.cy) / max(f1.b + f2.b, 1e-6)
    return float(np.hypot(dx, dy))


def overlap(f1: Footprint, f2: Footprint, tolerance: float = 0.0) -> bool:
    """True when the two vehicles share road surface (within a tolerance)."""
    return separation(f1, f2) <= (1.0 + tolerance)


def contact_score(f1: Footprint, f2: Footprint, near: float = 1.35) -> float:
    """Continuous 0..1 measure of how close two vehicles are to sharing road.

    1.0 when the footprints fully overlap, falling to 0 at ``near`` footprint
    widths apart. Continuous rather than a hard test so it can be weighted
    against other evidence instead of vetoing it.
    """
    s = separation(f1, f2)
    if s <= 1.0:
        return 1.0
    if s >= near:
        return 0.0
    return float((near - s) / max(near - 1.0, 1e-6))


def box_separation(box_a, box_b, depth_ratio: float = 0.35) -> float:
    """Convenience: normalised footprint separation straight from two boxes."""
    return separation(Footprint.from_box(box_a, depth_ratio),
                      Footprint.from_box(box_b, depth_ratio))


def estimate_depth_ratio(boxes, default: float = 0.35) -> float:
    """Guess how obliquely this camera views the road, from vehicle shapes.

    A near-overhead view makes vehicles look almost as deep as they are wide; a
    shallow roadside view flattens them. Median box aspect over many detections
    is a serviceable proxy, and it lets a camera self-tune instead of every site
    needing a hand-set constant.
    """
    if boxes is None or len(boxes) < 8:
        return default
    ratios = []
    for b in boxes:
        w = max(1e-6, float(b[2]) - float(b[0]))
        h = max(1e-6, float(b[3]) - float(b[1]))
        ratios.append(h / w)
    med = float(np.median(ratios))
    # h/w near 0.5 => flat, oblique view; near 1.2 => steep, near-overhead
    return float(np.clip(0.22 + 0.30 * med, 0.18, 0.75))


def draw(img, f: Footprint, colour=(90, 200, 255), thickness: int = 2, scale: float = 1.0):
    """Render a footprint, so a reviewer can see the geometry being reasoned over."""
    import cv2
    centre = (int(f.cx * scale), int(f.cy * scale))
    axes = (max(1, int(f.a * scale)), max(1, int(f.b * scale)))
    cv2.ellipse(img, centre, axes, 0, 0, 360, colour, thickness, cv2.LINE_AA)
    return img
