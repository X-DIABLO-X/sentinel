"""Multi-object tracking on GMC-compensated coordinates.

Tracking here happens in **reference-frame** coordinates — i.e. after every
box has already been passed through :func:`gmc.apply_gmc_boxes` with the
chained ego-motion homography from ``GMCEstimator``. That ordering is the
whole point: in reference-frame coordinates a stationary vehicle has a
constant position and zero velocity regardless of how much the drone itself
drifted during hover, and a moving vehicle's displacement is its own. Track
this module's ``Track.speed_px`` the same way you would a fixed CCTV camera's
and the ego-motion has already been removed upstream.

Relationship to the CCTV tracker
---------------------------------
``CCTV/netra/track.py`` implements a full ByteTrack (Zhang et al., ECCV 2022)
with a constant-velocity Kalman filter and Hungarian (``scipy``) assignment —
read to match its update-loop shape: two-stage association where low-score
detections get a second chance against tracks the high-score pass missed,
which is ByteTrack's actual contribution and the reason a partially-occluded
vehicle doesn't fracture into two track ids.

This module is a **thin local reimplementation**, not an import of that file,
for two reasons:

1. ``netra.track`` is reached through relative imports inside the ``netra``
   package (``from .geometry import ...``) and pulls in the rest of that
   package's import graph (db, jobs, ...) to use in isolation. DRONE has zero
   dependency on CCTV; the task brief explicitly asks for that isolation, and
   the requirements list here (``ultralytics, opencv-python, numpy, pyyaml,
   fastapi, uvicorn``) intentionally has no ``scipy``.
2. Optimal (Hungarian) vs. greedy assignment is a real accuracy trade, not
   free — greedy is a legitimate, honest choice for a single hovering
   incident with a handful of tracks, not a silent downgrade. If precise
   parity with the CCTV tracker ever matters, swap ``_greedy_match`` below for
   ``scipy.optimize.linear_sum_assignment`` and add the dependency; nothing
   else in this file would need to change.

No appearance/ReID model, same reasoning as the CCTV side: over a short
hover-based incident window the motion cue is sufficient.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["Track", "DroneTracker", "iou_matrix", "ground_point"]

NEW = "new"
TRACKED = "tracked"
LOST = "lost"
REMOVED = "removed"


# --------------------------------------------------------------------------
# geometry helpers (self-contained — no CCTV import)
# --------------------------------------------------------------------------

def ground_point(box: Sequence[float]) -> tuple[float, float]:
    """Bottom-centre of an xyxy box — the point taken to touch the ground."""
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    return (float(x1 + x2) / 2.0, float(y2))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) and (M,4) xyxy box arrays -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _greedy_match(cost: np.ndarray, max_cost: float):
    """Greedy minimum-cost matching: repeatedly take the best remaining pair.

    Not optimal like Hungarian, but for a handful of tracks against a handful
    of detections (a single hovering incident, not 82 CCTV feeds) the cases
    where greedy and optimal diverge are rare and low-stakes. See module
    docstring for the honest trade-off statement.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    matches: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()

    flat = [(cost[i, j], i, j) for i in range(n) for j in range(m) if cost[i, j] <= max_cost]
    flat.sort(key=lambda t: t[0])
    for c, i, j in flat:
        if i in used_r or j in used_c:
            continue
        matches.append((i, j))
        used_r.add(i)
        used_c.add(j)

    ur = [i for i in range(n) if i not in used_r]
    uc = [j for j in range(m) if j not in used_c]
    return matches, ur, uc


# --------------------------------------------------------------------------
# Track
# --------------------------------------------------------------------------

@dataclass
class Track:
    """One road user, followed across frames in reference-frame coordinates.

    ``box`` and ``history`` are in the GMC reference frame (post ego-motion
    compensation), NOT raw pixel coordinates — that is what makes
    ``speed_px`` meaningful for a moving-camera platform. ``px_box`` is kept
    alongside purely for optional overlay/debug rendering against the raw
    frame; nothing kinematic should ever read it.
    """

    track_id: int
    cls: int
    score: float
    box: np.ndarray                      # reference-frame x1,y1,x2,y2
    px_box: np.ndarray                   # current-frame pixel x1,y1,x2,y2 (display only)
    frame_idx: int
    t: float

    state: str = TRACKED
    hits: int = 1
    time_since_update: int = 0

    first_t: float = 0.0
    last_t: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=600))  # (t, x, y) ref-frame
    score_history: deque = field(default_factory=lambda: deque(maxlen=200))
    class_votes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float64)
        self.px_box = np.asarray(self.px_box, dtype=np.float64)
        self.first_t = self.t
        self.last_t = self.t
        self._push_history()

    def _push_history(self) -> None:
        gx, gy = ground_point(self.box)
        self.history.append((self.t, gx, gy))
        self.score_history.append(self.score)
        self.class_votes[self.cls] = self.class_votes.get(self.cls, 0) + 1

    @property
    def ground_point_ref(self) -> tuple[float, float]:
        return ground_point(self.box)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_t - self.first_t)

    @property
    def majority_cls(self) -> int:
        """Most frequently observed class over the track's life.

        A single-frame flicker (placeholder detector especially) should not
        define the whole track's reported class.
        """
        if not self.class_votes:
            return self.cls
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    def points(self, seconds: float | None = None) -> list[tuple[float, float]]:
        if seconds is None:
            return [(x, y) for _, x, y in self.history]
        cutoff = self.last_t - seconds
        return [(x, y) for t, x, y in self.history if t >= cutoff]

    def speed_px(self, seconds: float = 1.0) -> float:
        """Reference-frame speed in px/s, arc length over the trailing window.

        Already ego-motion compensated: this is the vehicle's own motion, not
        a mixture of vehicle motion and drone drift.
        """
        pts = [(t, x, y) for t, x, y in self.history if t >= self.last_t - seconds]
        if len(pts) < 2:
            return 0.0
        dt = pts[-1][0] - pts[0][0]
        if dt <= 1e-6:
            return 0.0
        dist = 0.0
        for i in range(1, len(pts)):
            dist += float(np.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]))
        return dist / dt

    def displacement(self, seconds: float = 1.0) -> float:
        pts = self.points(seconds)
        if len(pts) < 2:
            return 0.0
        return float(np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))

    def update(self, box: Sequence[float], px_box: Sequence[float], score: float,
               cls: int, frame_idx: int, t: float) -> None:
        self.box = np.asarray(box, dtype=np.float64)
        self.px_box = np.asarray(px_box, dtype=np.float64)
        self.score = float(score)
        self.cls = int(cls)
        self.frame_idx = frame_idx
        self.t = t
        self.last_t = t
        self.hits += 1
        self.time_since_update = 0
        self.state = TRACKED
        self._push_history()

    def mark_lost(self) -> None:
        self.state = LOST

    def mark_removed(self) -> None:
        self.state = REMOVED

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "cls": int(self.majority_cls),
            "score": round(float(self.score), 4),
            "ref_box": [round(float(v), 1) for v in self.box],
            "px_box": [round(float(v), 1) for v in self.px_box],
            "ground_point_ref": [round(v, 1) for v in self.ground_point_ref],
            "hits": self.hits,
            "first_t": round(self.first_t, 3),
            "last_t": round(self.last_t, 3),
            "duration": round(self.duration, 3),
            "speed_px_s": round(self.speed_px(), 2),
        }


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------

class DroneTracker:
    """ByteTrack-shaped two-stage association, greedy IoU matching.

    ``update(detections, px_boxes, t)`` mirrors ``ByteTracker.update`` in
    ``CCTV/netra/track.py``: ``detections`` is ``(N, 6)``
    ``[x1, y1, x2, y2, score, cls]`` in **reference-frame** coordinates.
    ``px_boxes`` is the matching ``(N, 4)`` array in current-frame pixel
    coordinates, kept only for display/debug on the ``Track`` objects.
    """

    def __init__(self, high_thresh: float = 0.35, low_thresh: float = 0.10,
                 match_thresh: float = 0.80, second_match_thresh: float = 0.50,
                 max_time_lost: int = 30, min_hits: int = 3) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh          # IoU-distance (1 - IoU)
        self.second_match_thresh = second_match_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        self.tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self._next_id = 1
        self.frame_idx = 0

    @classmethod
    def from_config(cls, tracker_cfg) -> "DroneTracker":
        return cls(
            high_thresh=tracker_cfg.high_thresh,
            low_thresh=tracker_cfg.low_thresh,
            match_thresh=tracker_cfg.match_thresh,
            max_time_lost=tracker_cfg.max_time_lost,
            min_hits=tracker_cfg.min_hits,
        )

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def reset(self) -> None:
        self.tracks.clear()
        self.lost_tracks.clear()

    def update(self, detections: np.ndarray, px_boxes: np.ndarray, t: float) -> list[Track]:
        self.frame_idx += 1
        dets = np.asarray(detections, dtype=np.float64).reshape(-1, 6)
        px = np.asarray(px_boxes, dtype=np.float64).reshape(-1, 4)
        if px.shape[0] != dets.shape[0]:
            raise ValueError("px_boxes must have the same length as detections")

        keep = dets[:, 4] >= self.low_thresh
        dets, px = dets[keep], px[keep]
        high_mask = dets[:, 4] >= self.high_thresh
        high, high_px = dets[high_mask], px[high_mask]
        low, low_px = dets[~high_mask], px[~high_mask]

        for tr in self.tracks:
            tr.time_since_update += 1
        for tr in self.lost_tracks:
            tr.time_since_update += 1

        confirmed = [tr for tr in self.tracks if tr.hits >= self.min_hits] + self.lost_tracks
        unconfirmed = [tr for tr in self.tracks if tr.hits < self.min_hits]

        # ---- pass 1: confirmed tracks vs high-score detections ------------
        matches, u_track, u_det = self._associate(confirmed, high, self.match_thresh)
        matched_confirmed_ids = set()
        for ti, di in matches:
            tr = confirmed[ti]
            tr.update(high[di, :4], high_px[di], high[di, 4], int(high[di, 5]),
                      self.frame_idx, t)
            matched_confirmed_ids.add(id(tr))
            if tr in self.lost_tracks:
                self.lost_tracks.remove(tr)
                if tr not in self.tracks:
                    self.tracks.append(tr)

        # ---- pass 2: remaining tracked (not lost) vs low-score detections -
        # ByteTrack's actual contribution: a weak box from partial occlusion
        # still gets a chance to extend a track instead of only birthing a
        # fresh one.
        remaining = [confirmed[i] for i in u_track if confirmed[i].state == TRACKED]
        m2, u_track2, _ = self._associate(remaining, low, self.second_match_thresh)
        matched_second_ids = set()
        for ti, di in m2:
            tr = remaining[ti]
            tr.update(low[di, :4], low_px[di], low[di, 4], int(low[di, 5]),
                      self.frame_idx, t)
            matched_second_ids.add(id(tr))

        for i in u_track:
            tr = confirmed[i]
            if id(tr) in matched_confirmed_ids or id(tr) in matched_second_ids:
                continue
            if tr.state != LOST:
                tr.mark_lost()
                if tr in self.tracks:
                    self.tracks.remove(tr)
                if tr not in self.lost_tracks:
                    self.lost_tracks.append(tr)

        # ---- pass 3: unconfirmed tracks vs leftover high-score detections -
        leftover = high[u_det] if len(u_det) else np.empty((0, 6))
        leftover_px = high_px[u_det] if len(u_det) else np.empty((0, 4))
        m3, u_unconf, u_det3 = self._associate(unconfirmed, leftover, self.match_thresh)
        for ti, di in m3:
            tr = unconfirmed[ti]
            tr.update(leftover[di, :4], leftover_px[di], leftover[di, 4],
                      int(leftover[di, 5]), self.frame_idx, t)
        for i in u_unconf:
            tr = unconfirmed[i]
            tr.mark_removed()
            if tr in self.tracks:
                self.tracks.remove(tr)

        # ---- birth ----------------------------------------------------
        for di in u_det3:
            d = leftover[di]
            if d[4] < self.high_thresh:
                continue
            tr = Track(track_id=self._new_id(), cls=int(d[5]), score=float(d[4]),
                       box=d[:4], px_box=leftover_px[di], frame_idx=self.frame_idx, t=t)
            self.tracks.append(tr)

        # ---- death ------------------------------------------------------
        still_lost = []
        for tr in self.lost_tracks:
            if tr.time_since_update > self.max_time_lost:
                tr.mark_removed()
            else:
                still_lost.append(tr)
        self.lost_tracks = still_lost

        return [tr for tr in self.tracks if tr.state == TRACKED and tr.hits >= self.min_hits]

    def _associate(self, tracks: list[Track], dets: np.ndarray, thresh: float):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tboxes = np.stack([tr.box for tr in tracks])
        dboxes = dets[:, :4]
        cost = 1.0 - iou_matrix(tboxes, dboxes)
        return _greedy_match(cost, thresh)

    @property
    def all_tracks(self) -> list[Track]:
        return self.tracks + self.lost_tracks


if __name__ == "__main__":   # pragma: no cover - manual smoke check
    tr = DroneTracker()
    # a "vehicle" sitting still in reference-frame coords across 5 frames
    for i in range(5):
        det = np.array([[100.0, 100.0, 140.0, 130.0, 0.9, 3.0]])
        px = np.array([[100.0 + i, 100.0, 140.0 + i, 130.0]])  # pixel drifts, ref doesn't
        live = tr.update(det, px, t=i * 0.1)
    print("tracks:", [t.to_dict() for t in live])
    print("speed_px (should be ~0, ref-frame static):", live[0].speed_px())
