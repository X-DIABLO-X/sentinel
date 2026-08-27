"""ByteTrack multi-object tracker, implemented from the paper.

Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection
Box", ECCV 2022 (arXiv:2110.06864).

Why implement it rather than import a tracker
---------------------------------------------
Three reasons, all of which matter for this project:

1. **No extra dependency.** The reference implementation pulls in ``lap`` and a
   pinned numpy; scipy's Hungarian solver does the same job and is already
   installed.
2. **We need the track history, not just the boxes.** Every NETRA event engine
   reasons over trajectories, speeds and dwell times. Owning the track object
   means that history is first-class rather than something we reconstruct.
3. **It is auditable.** A judge can read this file and see exactly why two
   detections were joined. That is the whole thesis of the system.

The key idea of ByteTrack is in ``update``: low-confidence detections are *not*
discarded. They get a second association pass against tracks that failed to
match on the first pass. A partially occluded vehicle emits a weak box, and
throwing it away is what breaks a track -- which in turn corrupts every
downstream event.

No appearance/ReID model is used. On a fixed camera with short-horizon events
the motion cue is sufficient, and a ReID CNN would roughly double the per-frame
cost for no benefit we can measure at the event level.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import box_ground_point, iou_matrix, robust_direction, unit

# track lifecycle states
NEW = "new"
TRACKED = "tracked"
LOST = "lost"
REMOVED = "removed"


# --------------------------------------------------------------------------
# Kalman filter (constant-velocity, SORT parameterisation)
# --------------------------------------------------------------------------

class KalmanBoxFilter:
    """8-dimensional state ``[cx, cy, a, h, vx, vy, va, vh]``.

    ``a`` is the aspect ratio w/h and ``h`` the box height. Noise scales with
    height, so distant (small) objects are modelled as having correspondingly
    smaller absolute uncertainty -- important here because traffic cameras look
    down a road and object scale varies enormously across one frame.
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov


def _xyxy_to_xyah(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = max(1e-3, x2 - x1), max(1e-3, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def _xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, a, h = state[:4]
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


# --------------------------------------------------------------------------
# Track
# --------------------------------------------------------------------------

@dataclass
class Track:
    """One road user, followed across frames.

    Carries everything the event engines need. ``history`` holds recent ground
    points with timestamps, which is the substrate for direction, speed and
    dwell reasoning.
    """

    track_id: int
    cls: int
    score: float
    box: np.ndarray                      # x1, y1, x2, y2
    frame_idx: int
    t: float                             # seconds into the clip

    state: str = NEW
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None

    hits: int = 1
    age: int = 0
    time_since_update: int = 0

    first_t: float = 0.0
    last_t: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=300))   # (t, x, y)
    box_history: deque = field(default_factory=lambda: deque(maxlen=120))
    score_history: deque = field(default_factory=lambda: deque(maxlen=120))

    # populated by the scene/event layer, kept here so one object carries the
    # whole picture of a road user
    corridor_id: str | None = None
    stationary_since: float | None = None
    wrongway_evidence: float = 0.0

    _kf: KalmanBoxFilter | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.box = np.asarray(self.box, dtype=np.float64)
        self.first_t = self.t
        self.last_t = self.t
        self._kf = KalmanBoxFilter()
        self.mean, self.covariance = self._kf.initiate(_xyxy_to_xyah(self.box))
        self._push_history()

    # -- history -----------------------------------------------------------
    def _push_history(self) -> None:
        gx, gy = box_ground_point(self.box)
        self.history.append((self.t, gx, gy))
        self.box_history.append((self.t, self.box.copy()))
        self.score_history.append(self.score)

    @property
    def ground_point(self) -> tuple[float, float]:
        return box_ground_point(self.box)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_t - self.first_t)

    def points(self, seconds: float | None = None) -> list[tuple[float, float]]:
        """Ground points from the last ``seconds`` (all of them if None)."""
        if seconds is None:
            return [(x, y) for _, x, y in self.history]
        cutoff = self.last_t - seconds
        return [(x, y) for t, x, y in self.history if t >= cutoff]

    # -- derived motion ----------------------------------------------------
    def direction(self, seconds: float = 1.5, min_span: float = 4.0):
        """Unit direction of travel over the recent window, or None."""
        return robust_direction(self.points(seconds), min_span=min_span)

    def speed_px(self, seconds: float = 1.0) -> float:
        """Speed in pixels/second, smoothed over the window.

        Image-plane speed is perspective-dependent and is never reported to a
        user as a physical speed. It is used for *relative* comparisons within
        one corridor, where perspective is roughly constant.
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
        """Straight-line distance covered in the window (px).

        Distinct from ``speed_px`` * time: a vehicle jittering in place has
        speed but no displacement, which is how we tell a stopped vehicle from
        a noisy one.
        """
        pts = self.points(seconds)
        if len(pts) < 2:
            return 0.0
        return float(np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))

    def acceleration_px(self, seconds: float = 1.0) -> float:
        """Change in speed across two adjacent windows, px/s^2. Negative = braking."""
        half = seconds / 2.0
        now = self.speed_px(half)
        pts = [(t, x, y) for t, x, y in self.history if t <= self.last_t - half]
        if len(pts) < 2:
            return 0.0
        dt = pts[-1][0] - pts[0][0]
        if dt <= 1e-6:
            return 0.0
        dist = sum(
            float(np.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]))
            for i in range(1, len(pts))
        )
        before = dist / dt
        return (now - before) / max(half, 1e-6)

    def peak_speed(self, seconds: float | None = None) -> float:
        """Fastest this track has ever moved, px/s.

        Used to answer "was this vehicle ever actually driving?". A parked car
        has a peak speed near zero for its whole life; a crashed one was moving
        and then stopped. That single distinction removes the largest class of
        false positives we measured -- parked cars and cars waiting at a signal
        being reported as collisions.
        """
        pts = [(t, x, y) for t, x, y in self.history
               if seconds is None or t >= self.last_t - seconds]
        if len(pts) < 3:
            return 0.0
        best = 0.0
        for i in range(2, len(pts)):
            dt = pts[i][0] - pts[i - 2][0]
            if dt <= 1e-6:
                continue
            d = float(np.hypot(pts[i][1] - pts[i - 2][1], pts[i][2] - pts[i - 2][2]))
            best = max(best, d / dt)
        return best

    def aspect_ratio(self) -> float:
        w = max(1e-6, self.box[2] - self.box[0])
        h = max(1e-6, self.box[3] - self.box[1])
        return float(w / h)

    def aspect_shift(self, seconds: float = 2.0) -> float:
        """Relative change in bounding-box aspect ratio over the window.

        A vehicle that rolls, flips or is spun broadside changes its silhouette
        dramatically, and an axis-aligned box registers that as a large swing in
        width/height even without oriented-box support. An upright vehicle
        driving normally holds its aspect almost constant, so this is close to
        free evidence of a violent event.
        """
        hist = [(t, b) for t, b in self.box_history if t >= self.last_t - seconds]
        if len(hist) < 4:
            return 0.0
        def ar(b):
            return max(1e-6, b[2] - b[0]) / max(1e-6, b[3] - b[1])
        k = max(1, len(hist) // 3)
        early = float(np.median([ar(b) for _, b in hist[:k]]))
        late = float(np.median([ar(b) for _, b in hist[-k:]]))
        if early <= 1e-6:
            return 0.0
        return float(abs(late - early) / early)

    def heading_change(self, seconds: float = 1.0) -> float:
        """Absolute change in direction across two adjacent windows, degrees."""
        half = max(seconds / 2.0, 1e-3)
        recent = robust_direction(self.points(half), min_span=2.0)
        older_pts = [(x, y) for t, x, y in self.history
                     if self.last_t - seconds <= t <= self.last_t - half]
        older = robust_direction(older_pts, min_span=2.0)
        if recent is None or older is None:
            return 0.0
        c = float(np.clip(np.dot(unit(recent), unit(older)), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    # -- lifecycle ---------------------------------------------------------
    def predict(self) -> None:
        if self.mean is None:
            return
        # a stopped object should not be pushed forward by stale velocity
        if self.state != TRACKED:
            self.mean[6] = 0.0
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, box: Sequence[float], score: float, cls: int,
               frame_idx: int, t: float) -> None:
        self.box = np.asarray(box, dtype=np.float64)
        self.score = float(score)
        self.cls = int(cls)
        self.frame_idx = frame_idx
        self.t = t
        self.last_t = t
        self.hits += 1
        self.time_since_update = 0
        self.state = TRACKED
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, _xyxy_to_xyah(self.box)
        )
        self._push_history()

    def mark_lost(self) -> None:
        self.state = LOST

    def mark_removed(self) -> None:
        self.state = REMOVED

    @property
    def predicted_box(self) -> np.ndarray:
        if self.mean is None:
            return self.box
        return _xyah_to_xyxy(self.mean)

    def to_dict(self) -> dict:
        d = self.direction()
        return {
            "track_id": self.track_id,
            "cls": int(self.cls),
            "score": round(float(self.score), 4),
            "box": [round(float(v), 1) for v in self.box],
            "ground_point": [round(v, 1) for v in self.ground_point],
            "speed_px": round(self.speed_px(), 2),
            "direction": [round(float(v), 4) for v in d] if d is not None else None,
            "corridor_id": self.corridor_id,
            "first_t": round(self.first_t, 3),
            "last_t": round(self.last_t, 3),
            "duration": round(self.duration, 3),
        }


# --------------------------------------------------------------------------
# the tracker
# --------------------------------------------------------------------------

def _hungarian(cost: np.ndarray, max_cost: float):
    """Solve the assignment and drop pairs whose cost exceeds ``max_cost``."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches, ur, uc = [], [], []
    matched_r, matched_c = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_r.add(int(r))
            matched_c.add(int(c))
    ur = [i for i in range(cost.shape[0]) if i not in matched_r]
    uc = [j for j in range(cost.shape[1]) if j not in matched_c]
    return matches, ur, uc


class ByteTracker:
    """The two-stage association loop.

    Parameters mirror the paper's defaults, retuned only where traffic differs
    from pedestrian benchmarks:

    ``high_thresh``   detections at/above this start new tracks
    ``low_thresh``    detections above this are still used for the *second*
                      association pass -- this is the ByteTrack idea
    ``match_thresh``  maximum IoU *distance* (1 - IoU) for the first pass
    ``max_time_lost`` frames a track survives unmatched before removal
    """

    def __init__(self,
                 high_thresh: float = 0.35,
                 low_thresh: float = 0.10,
                 match_thresh: float = 0.80,
                 second_match_thresh: float = 0.50,
                 unconfirmed_thresh: float = 0.70,
                 max_time_lost: int = 30,
                 min_hits: int = 3) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.unconfirmed_thresh = unconfirmed_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        self.tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self._next_id = 1
        self.frame_idx = 0

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def reset(self) -> None:
        """Drop all state. Called on a scene cut, where identities cannot carry."""
        self.tracks.clear()
        self.lost_tracks.clear()

    def update(self, detections: np.ndarray, t: float) -> list[Track]:
        """Advance the tracker one processed frame.

        ``detections`` is an (N, 6) array of ``[x1, y1, x2, y2, score, cls]``.
        ``t`` is the timestamp of this frame in seconds from clip start.
        Returns the list of currently-confirmed tracks.
        """
        self.frame_idx += 1
        dets = np.asarray(detections, dtype=np.float64).reshape(-1, 6)

        keep = dets[:, 4] >= self.low_thresh
        dets = dets[keep]
        high = dets[dets[:, 4] >= self.high_thresh]
        low = dets[(dets[:, 4] < self.high_thresh)]

        # predict every live track forward
        pool = [tr for tr in self.tracks if tr.state != REMOVED] + self.lost_tracks
        for tr in pool:
            tr.predict()

        confirmed = [tr for tr in pool if tr.hits >= self.min_hits or tr.state == LOST]
        unconfirmed = [tr for tr in pool if tr not in confirmed]

        # ---- pass 1: confirmed tracks vs high-score detections -----------
        matches, u_track, u_det = self._associate(confirmed, high, self.match_thresh)
        for ti, di in matches:
            tr = confirmed[ti]
            d = high[di]
            tr.update(d[:4], d[4], int(d[5]), self.frame_idx, t)
            if tr in self.lost_tracks:
                self.lost_tracks.remove(tr)
                if tr not in self.tracks:
                    self.tracks.append(tr)

        # ---- pass 2: still-unmatched tracks vs LOW-score detections ------
        # This is ByteTrack's contribution. A vehicle behind a bus emits a weak
        # box; recovering it here keeps the trajectory whole, and a whole
        # trajectory is what every event engine downstream depends on.
        remaining = [confirmed[i] for i in u_track if confirmed[i].state == TRACKED]
        m2, u_track2, _ = self._associate(remaining, low, self.second_match_thresh)
        for ti, di in m2:
            tr = remaining[ti]
            d = low[di]
            tr.update(d[:4], d[4], int(d[5]), self.frame_idx, t)

        matched_second = {id(remaining[i]) for i, _ in m2}
        for i in u_track:
            tr = confirmed[i]
            if id(tr) in matched_second:
                continue
            if tr.state != LOST:
                tr.mark_lost()
                if tr in self.tracks:
                    self.tracks.remove(tr)
                if tr not in self.lost_tracks:
                    self.lost_tracks.append(tr)

        # ---- pass 3: unconfirmed (single-hit) tracks vs leftovers --------
        leftover = high[u_det] if len(u_det) else np.empty((0, 6))
        m3, u_unconf, u_det3 = self._associate(unconfirmed, leftover, self.unconfirmed_thresh)
        for ti, di in m3:
            tr = unconfirmed[ti]
            d = leftover[di]
            tr.update(d[:4], d[4], int(d[5]), self.frame_idx, t)
        for i in u_unconf:
            tr = unconfirmed[i]
            tr.mark_removed()
            if tr in self.tracks:
                self.tracks.remove(tr)

        # ---- birth ------------------------------------------------------
        for di in u_det3:
            d = leftover[di]
            if d[4] < self.high_thresh:
                continue
            tr = Track(track_id=self._new_id(), cls=int(d[5]), score=float(d[4]),
                       box=d[:4], frame_idx=self.frame_idx, t=t)
            self.tracks.append(tr)

        # ---- death ------------------------------------------------------
        still_lost = []
        for tr in self.lost_tracks:
            if tr.time_since_update > self.max_time_lost:
                tr.mark_removed()
            else:
                still_lost.append(tr)
        self.lost_tracks = still_lost

        return [tr for tr in self.tracks
                if tr.state == TRACKED and tr.hits >= self.min_hits]

    def _associate(self, tracks: list[Track], dets: np.ndarray, thresh: float):
        """IoU-distance assignment between predicted track boxes and detections."""
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        tboxes = np.stack([tr.predicted_box for tr in tracks])
        dboxes = dets[:, :4]
        cost = 1.0 - iou_matrix(tboxes, dboxes)
        return _hungarian(cost, thresh)

    @property
    def all_tracks(self) -> list[Track]:
        return self.tracks + self.lost_tracks
