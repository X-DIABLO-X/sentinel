"""Where the road actually is, learned from the traffic that drives on it.

This module exists because of a specific, measured failure. Detecting stopped
vehicles on the background image -- the method every winning AI City entry uses
-- found 22 "stationary vehicles" on a night-time intersection clip. All 22 were
**parked cars in a car park** at the top of the frame, and because they were
parked next to each other, each one counted the others as "companion stopped
vehicles" and scored as a probable crash. The actual collision, in the middle of
the junction, was ignored.

Zhao et al. hit exactly this and say so plainly: traffic anomalies happen on
vehicles driving the main road, so static vehicles on side roads and in parking
lots must be filtered out. Their solution, and this one, is a **road mask** --
and the elegant part is that you do not need a segmentation model to build it.
The traffic tells you where the road is: accumulate the ground points of
everything that has *moved*, dilate, and that is the drivable surface.

A car park never generates moving trajectories through it at speed, so it never
enters the mask, so nothing parked there can ever raise an incident.
"""

from __future__ import annotations

import cv2
import numpy as np


class RoadMask:
    """Drivable surface, accumulated online from moving trajectories."""

    def __init__(self,
                 frame_shape: tuple[int, int],
                 scale: float = 0.25,
                 min_speed_px: float = 9.0,
                 dilate_px: float = 16.0,
                 min_samples: int = 60) -> None:
        self.scale = scale
        self.h = max(1, int(frame_shape[0] * scale))
        self.w = max(1, int(frame_shape[1] * scale))
        self.min_speed_px = min_speed_px
        self.dilate_px = dilate_px
        self.min_samples = min_samples

        self._acc = np.zeros((self.h, self.w), np.float32)
        self._mask: np.ndarray | None = None
        self.samples = 0
        self._dirty = False

    # ------------------------------------------------------------------
    def observe(self, tracks) -> None:
        """Record where road users are *while moving*.

        Speed is the whole filter. A vehicle manoeuvring in a car park moves
        slowly and briefly; a vehicle using the carriageway moves consistently.
        Only the latter paints the mask.
        """
        for tr in tracks:
            if tr.speed_px(1.0) < self.min_speed_px:
                continue
            x, y = tr.ground_point
            px = int(x * self.scale)
            py = int(y * self.scale)
            if 0 <= px < self.w and 0 <= py < self.h:
                cv2.circle(self._acc, (px, py), 2, 1.0, -1)
                self.samples += 1
                self._dirty = True

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.samples >= self.min_samples

    def _build(self) -> np.ndarray:
        m = (self._acc > 0).astype(np.uint8) * 255
        k = max(3, int(self.dilate_px * self.scale) * 2 + 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return m

    @property
    def mask(self) -> np.ndarray | None:
        if not self.ready:
            return None
        if self._mask is None or self._dirty:
            self._mask = self._build()
            self._dirty = False
        return self._mask

    # ------------------------------------------------------------------
    def contains(self, point) -> bool:
        """Is this image point on the drivable surface?

        Returns ``True`` while the mask is still forming -- refusing to judge is
        better than rejecting everything before enough traffic has been seen.
        """
        m = self.mask
        if m is None:
            return True
        px = int(point[0] * self.scale)
        py = int(point[1] * self.scale)
        if not (0 <= px < self.w and 0 <= py < self.h):
            return False
        return bool(m[py, px] > 0)

    def coverage(self, box) -> float:
        """Fraction of a box that lies on the drivable surface."""
        m = self.mask
        if m is None:
            return 1.0
        x1 = int(max(0, box[0] * self.scale))
        y1 = int(max(0, box[1] * self.scale))
        x2 = int(min(self.w, box[2] * self.scale))
        y2 = int(min(self.h, box[3] * self.scale))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        patch = m[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(np.count_nonzero(patch) / patch.size)

    def on_road(self, box, min_coverage: float = 0.25) -> bool:
        """A stopped vehicle counts only if it is genuinely on the carriageway.

        The ground point (bottom-centre) is checked first because that is where
        the vehicle touches the road; box coverage is the fallback for objects
        whose base is occluded.
        """
        ground = ((box[0] + box[2]) / 2.0, box[3])
        if self.contains(ground):
            return True
        return self.coverage(box) >= min_coverage

    def overlay(self, img, colour=(60, 120, 60), alpha: float = 0.12):
        """Tint the learned drivable surface, for the annotated video."""
        m = self.mask
        if m is None:
            return img
        full = cv2.resize(m, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(img)
        tint[full > 0] = colour
        return cv2.addWeighted(tint, alpha, img, 1.0 - alpha, 0)

    def stats(self) -> dict:
        m = self.mask
        return {
            "ready": self.ready,
            "samples": self.samples,
            "road_fraction": (float(np.count_nonzero(m) / m.size) if m is not None else None),
        }
