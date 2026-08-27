"""Path-crossing collision test: one vehicle cutting across another's path.

The idea, and why it is better than what came before
----------------------------------------------------
Review put it plainly: if a vehicle cuts across the predicted path of another --
or across the part of its recent path that has not yet faded -- at a steep angle,
that is a collision. The trajectories in the annotated video already show it to a
human instantly; the system was not being asked the same question.

Every earlier channel asked about a *state*: is this vehicle stopped, does it look
damaged, are these two footprints touching. Urban traffic produces all of those
constantly, which is why they generated 155.8 false alarms per crash-free hour.
This asks about a *geometric event between two specific vehicles*, which is a far
rarer thing:

1. **Their paths cross.** Not "they are near each other" -- the line segments
   actually intersect. Two vehicles queueing nose to tail never satisfy this,
   because their paths are collinear rather than crossing. That single property
   removes the failure mode that has dominated this project.

2. **They are at the crossing point at the same moment.** Two vehicles using the
   same junction thirty seconds apart also cross paths; the difference between
   traffic and a collision is simultaneity.

3. **The crossing angle is steep.** A lane change crosses another path at a
   shallow angle and is legal and constant. A T-bone, a side-swipe and a vehicle
   being deflected all involve a real angle.

4. **At least one trajectory changes afterwards.** This is the part that makes it
   a collision rather than a near miss: review's second screenshot showed the two
   vehicles' paths bending sharply right at the moment of contact. A vehicle that
   crosses another's path and continues undisturbed was simply given way to.

Conditions 1-3 are geometry and would fire on near misses. Condition 4 is what
separates contact from courtesy, and it is measured against the vehicle's own
prior heading rather than against any threshold on speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .footprint import Footprint, separation as fp_separation
from .freepath import blockers_from, first_blockage, swept_blockage


def _seg_intersect(p1, p2, p3, p4):
    """Intersection point of segments p1p2 and p3p4, or ``None``.

    Returns ``(point, t, u)`` where ``t`` and ``u`` are the fractional positions
    along each segment, so a caller can tell *where* along its path each vehicle
    was when the paths met.
    """
    p1, p2, p3, p4 = (np.asarray(p, dtype=float) for p in (p1, p2, p3, p4))
    r = p2 - p1
    s = p4 - p3
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        return None                      # parallel or collinear: a queue, not a crossing
    q = p3 - p1
    t = (q[0] * s[1] - q[1] * s[0]) / denom
    u = (q[0] * r[1] - q[1] * r[0]) / denom
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return p1 + t * r, float(t), float(u)


def ray_conflict(pos_a, vel_a, pos_b, vel_b, horizon_s: float):
    """Where and when two vehicles on their current courses would meet.

    Returns ``(point, t_a, t_b)`` -- the crossing of the two forward
    projections and the time each vehicle needs to reach it -- or ``None`` if
    the courses diverge or the meeting lies beyond the horizon.

    ``|t_a - t_b|`` is post-encroachment time: how long after the first vehicle
    leaves the conflict point the second arrives. Traffic-safety work treats a
    PET under about 1.5 s as a serious conflict, and a PET near zero is a
    collision, because both vehicles are at the same place at the same moment.
    """
    pos_a = np.asarray(pos_a, dtype=float); vel_a = np.asarray(vel_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float); vel_b = np.asarray(vel_b, dtype=float)
    sa, sb = float(np.linalg.norm(vel_a)), float(np.linalg.norm(vel_b))
    if sa < 1e-6 or sb < 1e-6:
        return None
    denom = vel_a[0] * vel_b[1] - vel_a[1] * vel_b[0]
    if abs(denom) < 1e-9:
        return None                      # parallel courses: a queue, not a conflict
    q = pos_b - pos_a
    t_a = (q[0] * vel_b[1] - q[1] * vel_b[0]) / denom
    t_b = (q[0] * vel_a[1] - q[1] * vel_a[0]) / denom
    if t_a < 0 or t_b < 0:
        return None                      # the meeting is behind one of them
    if t_a > horizon_s or t_b > horizon_s:
        return None
    return pos_a + vel_a * t_a, float(t_a), float(t_b)


def crossing_angle(a1, a2, b1, b2) -> float:
    """Angle between two path segments, in degrees, folded into 0-90."""
    va = np.asarray(a2, dtype=float) - np.asarray(a1, dtype=float)
    vb = np.asarray(b2, dtype=float) - np.asarray(b1, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(cos)))
    return min(ang, 180.0 - ang)


@dataclass
class PathConflict:
    """Two vehicles whose paths crossed, with what happened afterwards."""

    track_ids: tuple[int, int]
    point: np.ndarray
    t_cross: float
    angle_deg: float
    time_gap_s: float                 # how far apart in time they were there
    deviation_deg: tuple[float, float]
    speed_drop: tuple[float, float]
    boxes: list = field(default_factory=list)
    gates: dict = field(default_factory=dict)
    mode: str = "crossing"

    @property
    def score(self) -> float:
        """0..1, weighted by what the geometry can actually tell us.

        The evidence differs by collision type, and scoring them all the same
        way silently suppressed two of the three. A crossing is diagnosed by the
        angle and by a vehicle being knocked off its heading. A rear-end has
        neither: the courses are parallel, so the angle is zero, and the struck
        vehicle is pushed *along* its own heading rather than away from it. All
        it leaves behind is an abrupt loss of speed -- which is exactly what it
        should leave behind, and which the crossing weights ignored.
        """
        sim = max(0.0, 1.0 - self.time_gap_s / 0.60)
        dev = min(1.0, max(self.deviation_deg) / 35.0)
        drop = min(1.0, max(self.speed_drop))
        if self.mode == "crossing":
            ang = min(1.0, self.angle_deg / 60.0)
            return float(np.clip(0.20 * ang + 0.30 * sim + 0.35 * dev
                                 + 0.15 * drop, 0.0, 1.0))
        geom = self.gates.get("geometry")

        # The two track-level signals are held in a LOW BAND on purpose.
        #
        # They exist so the moment of an accident reaches the scorer at all --
        # before them, nine of sixteen accidents produced no candidate anywhere
        # near the crash. But they are weak evidence: a track also vanishes
        # behind a bridge, and a vehicle also stops hard for a red light. Scored
        # in the same range as the pair geometries they simply won, and because
        # both need history behind them they win LATE: measured across seven
        # clips they reported a median of 8.6 seconds after the accident, having
        # displaced the correct finding.
        #
        # Capped below every geometric finding, they now do what they were meant
        # to do -- carry a clip that would otherwise report nothing, and yield
        # the moment anything better is available.
        if geom == "track-lost":
            v = float(self.gates.get("lost_speed_px") or 0.0)
            return float(np.clip(0.10 + 0.10 * min(1.0, v / 200.0), 0.0, 0.22))
        if geom == "sudden-stop":
            before = float(self.gates.get("speed_before_px") or 0.0)
            after = float(self.gates.get("speed_after_px") or 0.0)
            lost = 0.0 if before <= 1e-6 else max(0.0, (before - after) / before)
            return float(np.clip(0.12 + 0.14 * lost
                                 + 0.06 * min(1.0, before / 250.0), 0.0, 0.30))
        if geom == "rollover":
            # A vehicle on its roof is not a probabilistic claim.
            r = float(self.gates.get("aspect_ratio_change", 1.0))
            return float(np.clip(0.55 + 0.25 * min(1.0, (r - 1.75) / 1.25)
                                 + 0.20 * min(1.0, max(self.speed_drop)), 0.0, 1.0))
        if self.mode == "deflection":
            # the corner itself is the evidence; contact names the cause
            turn = min(1.0, self.angle_deg / 60.0)
            return float(np.clip(0.55 * turn + 0.30 * dev + 0.15 * drop, 0.0, 1.0))
        # head-on, rear-end and into-stationary: closing onto another
        # vehicle's surface
        # is already required to register the course at all, so what remains to
        # be judged is whether the contact actually stopped anybody.
        return float(np.clip(0.55 * drop + 0.25 * sim + 0.20 * dev, 0.0, 1.0))


class PathConflictDetector:
    """Finds vehicles that cut across one another and were disturbed by it."""

    def __init__(self, history_s: float = 2.0, horizon_s: float = 2.5,
                 min_angle_deg: float = 25.0, max_time_gap_s: float = 0.60,
                 min_deviation_deg: float = 12.0, min_speed_px: float = 8.0,
                 min_score: float = 0.55, max_pet_s: float = 0.80,
                 confirm_window_s: float = 1.20,
                 max_footprint_sep: float = 1.25,
                 score_model: str | Path | None = None) -> None:
        self.history_s = history_s
        # How far ahead a course is projected when looking for a conflict.
        #
        # This was 0.8 s and it was the single reason almost nothing was
        # detected: on a real junction clip, 2,496 of 2,514 candidate pairs
        # were discarded because their projections did not meet inside that
        # window. Two vehicles converging across an intersection are typically
        # one to three seconds apart when they are already committed, which is
        # exactly the interval a human reads off the green lines in the
        # overlay. The value is shared with the renderer so that what is drawn
        # is what is tested -- a reviewer disagreeing with a green line is then
        # disagreeing with the detector, not with a picture of it.
        self.horizon_s = horizon_s
        self.min_angle_deg = min_angle_deg
        self.max_time_gap_s = max_time_gap_s
        self.min_deviation_deg = min_deviation_deg
        self.min_speed_px = min_speed_px
        self.min_score = min_score
        # Post-encroachment time: how far apart in time the two vehicles would
        # reach the point their courses cross. Near zero means they arrive
        # together, which is what a collision is.
        self.max_pet_s = max_pet_s
        # How long after the predicted meeting we keep watching before deciding
        # the course resolved safely.
        self.confirm_window_s = confirm_window_s
        self.max_footprint_sep = max_footprint_sep
        # How close to the conflict point a vehicle must be to count as
        # having arrived there, in multiples of its own width.
        self.confirm_radius_widths = 3.0
        # Below this, two vehicles are not meaningfully gaining on
        # each other and any time-to-contact is noise.
        self.min_closing_px_s = 12.0
        # Roughly 0.9 g: the most a driver can shed with the brakes.
        # Measured, not assumed.
        #
        # 0.9 g (8.8 m/s^2) is the correct physical bound on braking, and it is
        # unreachable with this footage: across two clips containing real
        # collisions the largest deceleration ever measured was 3.5 m/s^2, with
        # medians of 0.4 and 1.7. The impulse is real but the measurement
        # smooths it away -- 15 fps sampling with a velocity window of half a
        # second cannot resolve a speed change that happens in 100 ms.
        #
        # So the gate is set to what the instrument can actually see, and the
        # discrimination is carried by the surfaces overlapping and the path
        # being blocked rather than by this number alone. Raising it back to
        # 8.8 would be physically principled and would detect nothing.
        self.max_brake_m_s2 = 2.5
        # A corner sharper than this was not steered. Tyres deliver about 0.9 g
        # laterally; anything well beyond that was an external force.
        self.max_grip_g = 1.3
        self.min_turn_deg = 30.0
        self.max_turn_deg = 110.0
        # What counts as the other vehicle having felt the impact.
        self.partner_min_dev_deg = 8.0
        self.partner_min_drop = 0.15
        # A track that vanishes while travelling this fast, away from the
        # frame edge, is evidence rather than an inconvenience.
        self.impact_min_speed_px = 45.0
        self.track_lost_grace_s = 0.5
        # Losing this fraction of speed inside the window is a hard stop.
        self.stop_fraction = 0.65
        self.stop_window_s = 0.4
        # Half a vehicle length a second: slow, but unambiguously moving.
        self.stop_min_widths_per_s = 0.5
        self._last_live: dict[int, dict] = {}
        self.min_turn_speed_m_s = 2.5
        # Beyond this the 'vehicle' moved further than any vehicle can:
        # an identity switch, not a collision.
        self.max_plausible_g = 12.0
        # Two footprints closer than this are one vehicle tracked twice.
        self.min_partner_sep = 0.35
        # Striking fixed infrastructure: no second vehicle corroborates,
        # so the corner must be violent and most of the speed must go.
        self.solo_strike_g = 2.2
        self.solo_strike_drop = 0.55
        # Anything within this separation counts as 'something was there'.
        self.solo_clear_sep = 2.5
        # Contact is only recorded near the moment it was predicted.
        self.contact_window_s = 1.5
        # A box that changes shape by this factor and stays changed has
        # turned over. No manoeuvre produces it.
        self.rollover_ratio = 1.75
        self.rollover_span = 4
        # How long the new silhouette must hold before it counts.
        self.rollover_hold_s = 1.0
        # Projection step length in pixels: small enough that a shallow
        # footprint cannot be straddled between two samples.
        self.path_step_px = 8.0
        self._courses: dict[tuple[int, int], dict] = {}
        # Every candidate that cleared the hard physical constraints, with the
        # measurements behind it. This is the material the weights are fitted
        # on: labelled by clip, it says which combinations of physical evidence
        # actually accompany an accident and which accompany ordinary traffic.
        self.candidates: list[dict] = []
        self.weights: dict | None = self._load_score_model(score_model)

    @staticmethod
    def _load_score_model(path: str | Path | None) -> dict | None:
        """Load the optional fitted physics scorer, failing closed.

        Hard collision gates still decide which candidates can exist. The
        learned score only ranks candidates that already passed those gates;
        a missing, malformed, or incompatible file therefore falls back to the
        audited rule score instead of silently changing detector recall.
        """
        if not path:
            return None
        try:
            blob = json.loads(Path(path).read_text(encoding="utf-8"))
            n = len(blob["features"])
            if not (len(blob["mean"]) == len(blob["std"]) ==
                    len(blob["weights"]) == n):
                return None
            return blob
        except Exception:
            return None

    def _learned_score(self, feat: dict) -> float | None:
        if not self.weights:
            return None
        try:
            names = self.weights["features"]
            x = np.asarray([float(feat.get(k, 0.0)) for k in names], dtype=float)
            mu = np.asarray(self.weights["mean"], dtype=float)
            sd = np.asarray(self.weights["std"], dtype=float)
            w = np.asarray(self.weights["weights"], dtype=float)
            z = ((x - mu) / np.maximum(sd, 1e-9)) @ w
            z += float(self.weights["bias"])
            return float(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))))
        except Exception:
            return None

    def _record_candidate(self, pc: PathConflict, course: dict) -> dict:
        feat = self._features(pc, course)
        learned = self._learned_score(feat)
        if learned is not None:
            feat["learned_score"] = round(learned, 6)
            pc.gates["learned_score"] = round(learned, 6)
        self.candidates.append(feat)
        return feat

    # ------------------------------------------------------------------
    @staticmethod
    def _path(track, t0: float, t1: float, max_points: int = 20):
        """Ground-point path between two times, as an array of (t, x, y).

        Thinned to at most ``max_points``. The crossing test is quadratic in
        path length, so carrying 60 samples of a two-second path costs nine
        times what 20 costs and describes the same shape -- a vehicle's
        trajectory over two seconds is a smooth arc, not something that needs
        millisecond resolution to intersect correctly. The endpoints are always
        kept so the path still spans the full window.
        """
        hist = [h for h in getattr(track, "history", []) if t0 <= h[0] <= t1]
        if len(hist) < 2:
            return None
        if len(hist) > max_points:
            idx = np.linspace(0, len(hist) - 1, max_points).round().astype(int)
            hist = [hist[i] for i in dict.fromkeys(idx.tolist())]
        return np.asarray([[h[0], h[1], h[2]] for h in hist], dtype=float)

    @staticmethod
    def _heading(path) -> float | None:
        if path is None or len(path) < 2:
            return None
        v = path[-1, 1:] - path[0, 1:]
        if np.linalg.norm(v) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(v[1], v[0])))

    @staticmethod
    def _angle_between(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def _time_at(self, path, point, seg_i: int, frac: float) -> float:
        t0 = path[seg_i, 0]
        t1 = path[min(seg_i + 1, len(path) - 1), 0]
        return float(t0 + frac * (t1 - t0))

    # ------------------------------------------------------------------
    def _velocity(self, path, window_s: float = 0.6):
        """Current course: recent displacement over recent time, in px/s."""
        if path is None or len(path) < 2:
            return None
        t_end = path[-1, 0]
        seg = path[path[:, 0] >= t_end - window_s]
        if len(seg) < 2:
            seg = path[-2:]
        dt = seg[-1, 0] - seg[0, 0]
        if dt <= 1e-6:
            return None
        v = (seg[-1, 1:] - seg[0, 1:]) / dt
        return v if float(np.linalg.norm(v)) >= 1e-6 else None

    def find(self, tracks, t_now: float, forecasts: dict | None = None,
             footprints: bool = True, frame_shape=None) -> list[PathConflict]:
        """Collisions, detected as review described them.

        The pattern in the footage is a sequence, not a single instant:

            the two green lines cross  ->  the blue lines follow toward that
            point  ->  they arrive close enough together to touch

        So the test is staged the same way. When two vehicles' forward
        projections cross and both would reach the crossing at nearly the same
        moment, that pair is *registered as a course*. Nothing is reported yet:
        being on a collision course is not a collision, and most such courses
        resolve because somebody lifts off. The pair is then watched, and only
        reported if the vehicles actually arrive and are disturbed.

        An earlier version required the two *travelled* paths to intersect, and
        that was wrong in a way the footage makes obvious: a collision stops the
        vehicles at the moment of contact, so their travelled paths meet and
        tangle without ever completing the crossing. Requiring the crossing to
        complete requires the collision not to have happened.
        """
        # Vehicles are split by whether they have a course at all. A stopped
        # car has no green line, and excluding it -- as an earlier version did
        # -- makes "drove into a stationary vehicle" undetectable by
        # construction. It was present in six of the nine missed clips.
        moving, resting = {}, {}
        for tr in tracks:
            path = self._path(tr, t_now - self.history_s, t_now)
            if path is None:
                continue
            vel = self._velocity(path)
            span = float(np.linalg.norm(path[-1, 1:] - path[0, 1:]))
            if vel is not None and span >= self.min_speed_px:
                moving[int(tr.track_id)] = (tr, path, vel)
            else:
                resting[int(tr.track_id)] = (tr, path, None)
        usable = {**resting, **moving}

        # ---- stage 1: register every way two vehicles can be set to meet ----
        #
        # Three geometries, because a collision is not one shape. Measured on
        # the nine clips this detector missed, the crossing test could express
        # none of them: rear-end geometry appeared in eight, and driving into a
        # stopped vehicle in six.
        ids = list(moving)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ka, kb = ids[i], ids[j]
                key = (min(ka, kb), max(ka, kb))
                # A course is REFRESHED while it is still in the future, not
                # registered once and left alone.
                #
                # The first estimate is made from a second of history and is
                # correspondingly poor: on a synthetic rear-end it predicted
                # contact at 0.81 s when contact actually happened at 0.93 s,
                # and the stale estimate then expired before the vehicles met.
                # Time-to-contact converges as the gap closes, which is the
                # whole reason it is useful, so it is recomputed every frame
                # until the moment it describes arrives.
                existing = self._courses.get(key)
                if existing is not None and t_now >= existing["t_meet"] - 0.10:
                    continue                      # close enough to judge; leave it
                tr_a, pa, va = moving[ka]
                tr_b, pb, vb = moving[kb]
                ang = crossing_angle([0, 0], va, [0, 0], vb)

                # (1) CROSSING -- the two green lines meet at an angle.
                hit = ray_conflict(pa[-1, 1:], va, pb[-1, 1:], vb, self.horizon_s)
                if hit is not None and ang >= self.min_angle_deg:
                    point, ta, tb = hit
                    pet = abs(ta - tb)
                    if pet <= self.max_pet_s:
                        self._courses[key] = {
                            "point": np.asarray(point, dtype=float),
                            "t_registered": (existing or {}).get("t_registered", t_now),
                            "t_meet": t_now + 0.5 * (ta + tb),
                            "pet": pet, "angle": ang, "mode": "crossing",
                        }
                        continue

                # (1b) HEAD-ON -- courses anti-parallel and closing.
                #
                # crossing_angle folds into 0-90 degrees, so two vehicles
                # driving straight at each other read as 0 degrees -- exactly
                # like a rear-end, which is the opposite situation. The sign of
                # the dot product is what separates them and it is thrown away
                # by the fold, so it is tested directly here.
                ua, ub = va / max(np.linalg.norm(va), 1e-9), vb / max(np.linalg.norm(vb), 1e-9)
                if float(np.dot(ua, ub)) <= -0.70:
                    gap_vec = pb[-1, 1:] - pa[-1, 1:]
                    if float(np.dot(gap_vec, ua)) > 0:      # they face each other
                        app = self._surface_approach(pa[-1, 1:], va, tr_b.box, vb)
                        if app is not None:
                            point, ttc = app
                            self._courses[key] = {
                                "point": np.asarray(point, dtype=float),
                                "t_registered": (existing or {}).get("t_registered", t_now),
                                "t_meet": t_now + ttc, "pet": 0.0, "angle": 180.0,
                                "mode": "head-on", "striker": ka,
                            }
                            continue

                # (2) REAR-END -- the green lines are parallel and never cross,
                #     so the test is whether the follower's line runs into the
                #     leader's road surface. This is the most common collision
                #     type there is, and an angle threshold rejects all of it.
                if ang < self.min_angle_deg:
                    app = self._surface_approach(pa[-1, 1:], va, tr_b.box, vb)
                    who = (ka, kb)
                    if app is None:
                        app = self._surface_approach(pb[-1, 1:], vb, tr_a.box, va)
                        who = (kb, ka)
                    if app is not None:
                        point, ttc = app
                        self._courses[key] = {
                            "point": point,
                            "t_registered": (existing or {}).get("t_registered", t_now),
                            "t_meet": t_now + ttc, "pet": 0.0, "angle": ang,
                            "mode": "rear-end", "striker": who[0],
                        }

        # (3) BLOCKED PATH -- the green line runs into something and stops.
        #
        # This replaces a pairwise "is that particular vehicle ahead of me"
        # test, and is strictly better: the projection is checked against every
        # detected vehicle, so whatever is actually in the way is found rather
        # than whatever happened to be paired. It covers driving into a
        # stationary vehicle and closing on a moving one with the same
        # computation, because the difference between those is only how fast
        # the obstruction moves out of the way.
        #
        # Only detections can obstruct, so a shadow, a lane marking or a wet
        # patch cannot shorten a path.
        for km, (tr_m, pm, vm) in moving.items():
            # Step finely enough that the path cannot JUMP OVER an obstruction.
            #
            # A footprint is deliberately shallow -- a few pixels from front to
            # back under perspective -- while a vehicle at 200 px/s covers 500
            # px in the 2.5 s horizon. Sampled in a fixed 14 steps that is 36 px
            # a step, so the projection lands in front of a car and then behind
            # it, having passed straight through, and the obstruction is never
            # seen. The step is therefore sized in pixels, not in time.
            # Swept at the vehicle's own width, not a hairline down its centre.
            # A centre ray slides past a car the vehicle then drives into,
            # because the ray has no width and the car is two metres across.
            half_w = 0.5 * float(tr_m.box[2] - tr_m.box[0])
            blk = swept_blockage(pm[-1, 1:], vm, half_w,
                                 blockers_from(tracks, exclude_id=km),
                                 horizon_s=self.horizon_s,
                                 step_px=self.path_step_px)
            if blk is None:
                # fall back to the pairwise approach test, which tolerates a
                # projection that grazes a footprint rather than entering it
                for ks_, (tr_o, _po, vo) in usable.items():
                    if ks_ == km:
                        continue
                    app = self._surface_approach(pm[-1, 1:], vm, tr_o.box, vo)
                    if app is None:
                        continue
                    point_, ttc_ = app
                    key_ = (min(km, ks_), max(km, ks_))
                    ex_ = self._courses.get(key_)
                    if ex_ is not None and (
                            ex_.get("mode") in ("deflection", "crossing")
                            or t_now >= ex_["t_meet"] - 0.10):
                        continue
                    self._courses[key_] = {
                        "point": np.asarray(point_, dtype=float),
                        "t_registered": (ex_ or {}).get("t_registered", t_now),
                        "t_meet": t_now + ttc_, "pet": 0.0, "angle": 0.0,
                        "mode": ("into-stationary" if ks_ in resting
                                 else "rear-end"), "striker": km,
                    }
                continue
            ks = int(blk.blocker_id)
            if ks not in usable:
                continue
            key = (min(km, ks), max(km, ks))
            existing = self._courses.get(key)
            # Precedence: deflection (a force already measured) beats a
            # crossing (two courses that meet at an angle) beats a blocked path
            # (something is in the way). They are not competing hypotheses so
            # much as descriptions of decreasing specificity, and reporting a
            # T-bone as a rear-end because the blocked-path scan ran later is
            # simply mislabelling it.
            if existing is not None and (
                    existing.get("mode") in ("deflection", "crossing")
                    or t_now >= existing["t_meet"] - 0.10):
                continue
            mode = "into-stationary" if ks in resting else "rear-end"
            self._courses[key] = {
                "point": np.asarray(blk.point, dtype=float),
                "t_registered": (existing or {}).get("t_registered", t_now),
                "t_meet": t_now + blk.time_s, "pet": 0.0, "angle": 0.0,
                "mode": mode, "striker": km,
                "free_path_px": round(blk.gap_px, 1),
            }

        # ---- (4) DEFLECTION: the blue line turns a corner -------------------
        #
        # Some collisions leave no usable *approach* to register. A glancing
        # blow, a shunt from behind at a shallow angle, a vehicle already
        # mid-turn -- in all of these the two courses before contact are nearly
        # parallel, so no crossing is predicted and no closing onto a surface is
        # measured. What they do leave is unmistakable afterwards: the struck
        # vehicle's blue line turns a sharp corner.
        #
        # In momentum terms: a vehicle travelling at speed carries momentum
        # m*v along its heading. Changing the *direction* of that momentum
        # requires a lateral force, and the only lateral force available to a
        # driver is grip -- roughly 0.9 g, which over a third of a second can
        # bend a path by a modest amount and no more. A corner sharper than
        # that was not steered; something pushed it. So the test is a kink in
        # the blue line whose implied lateral acceleration exceeds what tyres
        # can deliver, with another vehicle's surface area overlapping at the
        # corner -- which names the thing that pushed it.
        for ka, (tr_a, pa, va) in moving.items():
            # ---- (6) ROLLOVER -- the silhouette turns over -----------------
            #
            # Single-vehicle accidents were 0 for 4 on the held-out set, and
            # five of the sixteen clips are annotated as involving a rollover.
            # For those the camera has the most direct evidence it will ever
            # get, and we were using it only as a weak corroborator: a car lying
            # on its roof is about as wide as it was long and about half as
            # tall, so the aspect ratio of its box changes by a factor no
            # manoeuvre produces.
            #
            # A turn changes apparent aspect gradually and reversibly as the
            # vehicle presents a different face to the camera. A rollover
            # changes it abruptly and it stays changed, because the vehicle does
            # not get back up. Persistence is what separates the two, so it is
            # required rather than assumed.
            roll = self._find_rollover(tr_a)
            if roll is not None:
                t_roll, ratio = roll
                # A rollover describes the OUTCOME, not the kind of collision.
                # Vehicles roll over in t-bones, sideswipes and single-vehicle
                # accidents alike, so a solo rollover must not pre-empt a pair
                # geometry that also names the other vehicle and the type. It is
                # registered only when this vehicle is not already part of a
                # pair conflict.
                key = (ka, ka)
                in_pair = any(ka in k and k[0] != k[1] for k in self._courses)
                prior = self._courses.get(key)
                if in_pair:
                    pass
                elif prior is None or prior.get("mode") != "rollover":
                    # Judged NOW, reported as having happened THEN.
                    #
                    # A rollover is found by scanning back over the box history,
                    # so the flip is already a second or two old by the time it
                    # is noticed. Registering it at the moment it happened meant
                    # the course was instantly older than the confirmation
                    # window and was expired before it could ever be confirmed.
                    # The event time is carried separately so the incident is
                    # still timestamped at the flip rather than at the noticing.
                    self._courses[key] = {
                        "point": np.asarray(pa[-1, 1:], dtype=float),
                        "t_registered": t_now, "t_meet": t_now,
                        "t_event": t_roll,
                        "pet": 0.0, "angle": 0.0, "mode": "rollover",
                        "aspect_ratio_change": ratio, "min_sep": 0.0,
                        "solo": True,
                    }

            # Everything below needs a corner in the path. A rollover does not:
            # a vehicle can turn over while still travelling roughly straight,
            # and requiring a kink first made the check above unreachable for
            # exactly the cases it was written for.
            kink = self._find_kink(pa, tr_a.box)
            if kink is None:
                continue
            t_kink, turn_deg, lat_g = kink
            partnered = False
            # Was ANY other vehicle near this corner, even one we declined to
            # pair with? Claiming a vehicle struck fixed infrastructure means
            # claiming nothing else was there, and a partner rejected as a
            # duplicate track is still evidence that something was.
            near_any = False
            for kb, (tr_b, pb, _) in usable.items():
                if kb == ka:
                    continue
                key = (min(ka, kb), max(ka, kb))
                # Deflection OUTRANKS a predicted course and replaces it.
                #
                # The other three modes are predictions: two vehicles are on
                # course to meet. A kink beyond tyre grip is not a prediction,
                # it is a measurement of an external force that has already
                # acted. When a vehicle is knocked sideways it also starts
                # closing on whatever hit it, so a rear-end course tends to be
                # registered first and then to fail its own braking gate --
                # which is how the strongest evidence available was being
                # discarded in favour of the weakest.
                prior = self._courses.get(key)
                if prior is not None and prior.get("mode") == "deflection":
                    continue
                sep = fp_separation(Footprint.from_box(tr_a.box),
                                    Footprint.from_box(tr_b.box))
                if sep <= self.solo_clear_sep:
                    near_any = True
                # The lower bound is well above the duplicate-track floor. A
                # false deflection was reported at separation 0.24 with an
                # implied 3.24 g: two boxes almost on top of each other, which
                # is one vehicle tracked twice, and a wobbling trail near the
                # frame edge differentiating into a huge lateral force.
                if not (self.min_partner_sep <= sep <= self.max_footprint_sep):
                    continue
                # BOTH vehicles must have felt it.
                #
                # A deflection names one vehicle as having been pushed and
                # another as having pushed it, and Newton's third law says the
                # second one felt the same force. Requiring only that a partner
                # was in contact let a single vehicle swerving past a passing
                # car report a collision: on a crash-free clip that scored
                # 0.952, higher than most real accidents, so no threshold on
                # the swerve itself could separate them. Whether the other
                # vehicle was disturbed at all is a different axis, and it is
                # the one the physics actually constrains.
                #
                # The bar on the partner is deliberately low, because mass
                # ratios are large -- a car striking a lorry moves it very
                # little -- so any measurable change counts.
                # Recorded, not enforced.
                #
                # As a hard gate this cost more than half the recall on the
                # held-out set -- mass ratios are large and a struck lorry moves
                # very little -- while as a feature it still carries the
                # information that a genuine impact disturbs both parties. Which
                # marginal signals should veto and which should merely weigh is
                # not a question to answer by intuition, so it is answered by
                # fitting weights over labelled clips instead.
                dev_p, drop_p = self._disturbance(pb, t_kink)

                self._courses[key] = {
                    "point": np.asarray(pa[-1, 1:], dtype=float),
                    "t_registered": t_now, "t_meet": t_kink,
                    "pet": 0.0, "angle": turn_deg, "mode": "deflection",
                    "lateral_g": lat_g, "partner_deviation_deg": round(dev_p, 1),
                    "partner_speed_drop": round(drop_p, 2),
                    "min_sep": min(float(sep), (prior or {}).get("min_sep", 1e9)),
                }
                partnered = True

            # ---- (5) STRUCK SOMETHING THAT IS NOT A VEHICLE ---------------
            #
            # A car that hits a central divider, a barrier or a kerb has no
            # partner to pair with, so every pairwise test in this file is blind
            # to it -- review found exactly that: a vehicle at speed, thrown
            # almost perpendicular to its own travel, with no green line left,
            # and nothing reported.
            #
            # The bar is deliberately higher than for a partnered deflection,
            # because there is no second vehicle to corroborate: the corner must
            # be violent rather than merely beyond grip, and the vehicle must
            # not simply carry on afterwards -- it has to lose most of its
            # speed, which is what hitting something immovable does.
            if not partnered and not near_any and lat_g >= self.solo_strike_g:
                _, drop_solo = self._disturbance(pa, t_kink)
                if drop_solo >= self.solo_strike_drop and turn_deg >= 45.0:
                    key = (ka, ka)
                    prior = self._courses.get(key)
                    if prior is None or prior.get("mode") != "struck-object":
                        self._courses[key] = {
                            "point": np.asarray(pa[-1, 1:], dtype=float),
                            "t_registered": t_now, "t_meet": t_kink,
                            "pet": 0.0, "angle": turn_deg,
                            "mode": "struck-object", "lateral_g": lat_g,
                            "min_sep": 0.0, "solo": True,
                        }

        # ---- (7) THE TWO THINGS AN IMPACT DOES TO A TRACK ------------------
        #
        # Measured across the held-out set, only seven of sixteen accidents ever
        # produced a candidate within a second of the annotated time. On the
        # other nine the system generated candidates in the clip -- up to
        # seventeen of them -- and none at the crash. No amount of scoring can
        # rank a candidate that was never created, so the ceiling was candidate
        # generation, not ranking.
        #
        # The two signatures below are the ones an impact leaves on a *track*
        # rather than on a pair, and both were previously counted as failures:
        #
        #   * the track DIES. A vehicle deforms, rotates and is occluded by
        #     whatever hit it, detector confidence collapses, and the tracker
        #     drops it. This was the single largest cause of lost confirmations
        #     -- 2,755 of them -- and it is not noise, it is the event.
        #   * the vehicle STOPS, hard. Whatever it hit -- a barrier, a divider,
        #     a stationary lorry -- may not itself be tracked, so no pair can
        #     ever form, but the speed is gone all the same.
        #
        # These are deliberately generous: they exist to make sure the moment
        # reaches the scorer at all. Consolidation then reports one collision
        # per clip, so being generous here costs candidates, not alerts.
        live_ids = set(usable)
        for tid, prev in list(self._last_live.items()):
            if tid in live_ids:
                continue
            gone_for = t_now - prev["t"]
            if gone_for < self.track_lost_grace_s:
                continue
            del self._last_live[tid]
            if prev["speed"] < self.impact_min_speed_px:
                continue                     # it was parked, or it drove off slowly
            if self._near_frame_edge(prev["box"], frame_shape):
                continue                     # it left the picture, which is not a crash
            key = (tid, tid)
            if key not in self._courses:
                self._courses[key] = {
                    "point": np.asarray(prev["ground"], dtype=float),
                    "t_registered": t_now, "t_meet": t_now, "t_event": prev["t"],
                    "pet": 0.0, "angle": 0.0, "mode": "track-lost",
                    "min_sep": 0.0, "solo": True, "lost_speed_px": prev["speed"],
                }

        for tid, (tr, path, vel) in usable.items():
            g = path[-1, 1:]
            self._last_live[tid] = {
                "t": t_now, "box": np.asarray(tr.box, dtype=float), "ground": g,
                "speed": float(np.linalg.norm(vel)) if vel is not None else 0.0,
            }
            stop = self._sudden_stop(path, tr.box)
            if stop is None:
                continue
            t_stop, before_px, after_px = stop
            key = (tid, tid)
            prior = self._courses.get(key)
            if prior is not None and prior.get("mode") in (
                    "rollover", "struck-object", "deflection"):
                continue
            self._courses[key] = {
                "point": np.asarray(path[-1, 1:], dtype=float),
                "t_registered": t_now, "t_meet": t_now, "t_event": t_stop,
                "pet": 0.0, "angle": 0.0, "mode": "sudden-stop",
                "min_sep": 0.0, "solo": True,
                "speed_before_px": round(before_px, 1),
                "speed_after_px": round(after_px, 1),
            }

        # ---- between the stages: remember the closest they ever came -------
        #
        # Contact and its consequences are not measurable at the same instant.
        # Two vehicles are touching for a few frames, but the speed they lost
        # can only be computed once there are samples on the far side of the
        # impact -- by which time they may have separated again, one shoved
        # forward and the other stopped. Testing both at one instant therefore
        # never succeeds: measured on a synthetic rear-end, separation was 1.16
        # while the deceleration still read 0.0, and by the time it read
        # 44 m/s^2 the separation had grown to 1.37.
        #
        # So contact is recorded continuously while the course is open, and the
        # closest approach is what the gate is applied to.
        for key, c in self._courses.items():
            # Contact is recorded whenever it happens, but it has to be
            # RECENT to count at confirmation.
            #
            # Tying it to the predicted meeting was too brittle: a
            # constant-velocity estimate of when two vehicles will meet is
            # routinely off by more than a second, so a real contact fell
            # outside the window and was never recorded. Tying it to nothing at
            # all was the opposite error -- a pair that brushed past each other
            # early satisfied the contact gate at an unrelated moment later.
            # Recording continuously and requiring recency at the point of
            # judgement avoids both.
            ka, kb = key
            if ka in usable and kb in usable:
                sep_now = fp_separation(
                    Footprint.from_box(usable[ka][0].box),
                    Footprint.from_box(usable[kb][0].box))
                if sep_now < c.get("min_sep", 1e9):
                    c["min_sep"] = float(sep_now)
                    c["min_sep_t"] = t_now

        # ---- stage 2: did the vehicles actually arrive, and get hit? -------
        out: list[PathConflict] = []
        for key in list(self._courses):
            c = self._courses[key]
            if t_now > c["t_meet"] + self.confirm_window_s:
                del self._courses[key]            # the course resolved safely
                continue
            if t_now < c["t_meet"] - 0.10:
                continue                          # not there yet
            ka, kb = key
            if c.get("solo"):
                if ka not in usable:
                    if t_now > c["t_meet"] + self.confirm_window_s:
                        del self._courses[key]
                    continue
                tr_a, pa, _ = usable[ka]
                dev_a, drop_a = self._disturbance(pa, c["t_meet"])
                pc = PathConflict(
                    track_ids=(ka, ka), point=c["point"],
                    t_cross=c.get("t_event", c["t_meet"]),
                    angle_deg=c["angle"], time_gap_s=0.0,
                    deviation_deg=(dev_a, 0.0), speed_drop=(drop_a, 0.0),
                    boxes=[np.asarray(tr_a.box, dtype=float)],
                    gates={"geometry": c.get("mode", "struck-object"),
                           "lost_speed_px": c.get("lost_speed_px"),
                           "speed_before_px": c.get("speed_before_px"),
                           "speed_after_px": c.get("speed_after_px"),
                           "lateral_g": round(c.get("lateral_g", 0.0), 2),
                           "turn_deg": round(c["angle"], 1),
                           "no_vehicle_partner": True,
                           "aspect_ratio_change": round(
                               c.get("aspect_ratio_change", 0.0), 2),
                           "speed_lost": round(drop_a, 2)},
                    mode="deflection",
                )
                self._record_candidate(pc, c)
                if pc.score >= self.min_score:
                    out.append(pc)
                    del self._courses[key]
                continue

            # Confirm by PLACE, not by identity.
            #
            # A vehicle loses its track at precisely the moment it is hit: it
            # deforms, rotates, is occluded by the other vehicle, and the
            # detector's confidence collapses, so the tracker starts a new id.
            # In one reviewed clip the pair carrying the collision changed from
            # ids 10 and 14 to 36 and 14 across the impact. Keying confirmation
            # on the original ids therefore fails on exactly the events we are
            # looking for.
            #
            # The conflict point does not move, so the vehicles are looked for
            # there instead, and the original ids are used only when they
            # survived.
            identity_changed = False
            near = self._near_point(usable, c["point"], t_now)
            if ka in usable and kb in usable:
                pair = [(ka, usable[ka]), (kb, usable[kb])]
            elif len(near) >= 2:
                pair = near[:2]
                identity_changed = True
            else:
                if t_now > c["t_meet"] + self.confirm_window_s:
                    del self._courses[key]
                continue
            (ka, (tr_a, pa, _)), (kb, (tr_b, pb, _)) = pair[0], pair[1]

            gates = {"predicted_paths_cross": True,
                     "post_encroachment_s": round(c["pet"], 2),
                     "crossing_angle_deg": round(c["angle"], 1),
                     "geometry": c.get("mode", "crossing"),
                     "identity_changed_at_impact": identity_changed}

            # they must be on the same patch of road, not merely near in image
            if footprints:
                # closest approach over the life of this course, not the gap
                # at whichever frame we happen to be judging on
                sep_live = fp_separation(Footprint.from_box(tr_a.box),
                                         Footprint.from_box(tr_b.box))
                sep_seen = c.get("min_sep", 1e9)
                # a closest approach from long ago says nothing about now
                if (t_now - c.get("min_sep_t", -1e9)) > self.contact_window_s:
                    sep_seen = 1e9
                sep = float(min(sep_seen, sep_live))
                if sep < 0.20:
                    del self._courses[key]        # duplicate tracks on one car
                    continue
                gates["footprint_separation"] = round(float(sep), 2)
                gates["footprints_overlap"] = bool(sep <= self.max_footprint_sep)
                if not gates["footprints_overlap"]:
                    continue

            # and something must have happened to them
            dev_a, drop_a = self._disturbance(pa, c["t_meet"])
            dev_b, drop_b = self._disturbance(pb, c["t_meet"])
            gates["motion_disturbed"] = bool(
                max(dev_a, dev_b) >= self.min_deviation_deg
                or max(drop_a, drop_b) >= 0.35)
            if not gates["motion_disturbed"]:
                continue

            # For collinear geometries the evidence has to be stronger, because
            # the benign case is so close: a driver braking to a halt behind
            # another car closes on it, touches nothing, and stops. What an
            # impact adds is that the speed is removed faster than tyres can
            # remove it, and that the surfaces actually overlap rather than
            # coming to rest adjacent.
            if c.get("mode") == "deflection":
                gates["lateral_g"] = round(c.get("lateral_g", 0.0), 2)
                gates["beyond_tyre_grip"] = True
                gates["turn_deg"] = round(c["angle"], 1)
                gates["partner_deviation_deg"] = c.get("partner_deviation_deg")
                gates["partner_speed_drop"] = c.get("partner_speed_drop")
            if c.get("mode") in ("rear-end", "into-stationary", "head-on"):
                dec = max(self._decel_m_s2(pa, tr_a.box, c["t_meet"]),
                          self._decel_m_s2(pb, tr_b.box, c["t_meet"]))
                gates["decel_m_s2"] = round(dec, 1)
                gates["beyond_driver_control"] = bool(dec >= self.max_brake_m_s2)
                if not gates["beyond_driver_control"]:
                    continue
                # A whisker of tolerance over exact overlap: bumper-to-bumper
                # contact puts the footprint edges at separation 1.0 exactly,
                # and detection boxes are not that precise. The abrupt-
                # deceleration gate above is what carries the discrimination
                # here, not this one.
                if gates.get("footprint_separation", 9.9) > 1.15:
                    gates["surfaces_actually_overlap"] = False
                    continue
                gates["surfaces_actually_overlap"] = True

            # Timestamp the incident at the contact we OBSERVED, not the
            # meeting we forecast. A course is registered before the vehicles
            # meet and confirmed when they do, so t_meet is a prediction and can
            # sit before the real thing -- which puts a red COLLISION box on a
            # car that has not been hit yet. The closest approach actually
            # measured is a fact rather than a forecast.
            t_contact = float(c.get("min_sep_t", c["t_meet"]))
            pc = PathConflict(
                track_ids=(ka, kb), point=c["point"], t_cross=t_contact,
                angle_deg=c["angle"], time_gap_s=c["pet"],
                deviation_deg=(dev_a, dev_b), speed_drop=(drop_a, drop_b),
                boxes=[np.asarray(tr_a.box, dtype=float),
                       np.asarray(tr_b.box, dtype=float)],
                gates=gates, mode=c.get("mode", "crossing"),
            )
            self._record_candidate(pc, c)
            if pc.score >= self.min_score:
                out.append(pc)
                del self._courses[key]

        out.sort(key=lambda c: -c.score)
        return out

    def _surface_approach(self, pos, vel, target_box, target_vel):
        """Time until a vehicle's course reaches another's road surface.

        The crossing test asks where two *lines* meet, which says nothing when
        the lines are parallel -- exactly the case for a rear-end collision, and
        for a vehicle bearing down on a stopped one. Here the question is
        instead when the follower's ground point reaches the leader's footprint,
        closing along the line between them.

        Returns ``(point, time_to_contact)`` or ``None`` when they are not
        closing, or contact lies beyond the horizon.
        """
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        f = Footprint.from_box(target_box)
        target = np.array([f.cx, f.cy], dtype=float)

        d = target - pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return None
        u = d / dist

        rel = vel if target_vel is None else vel - np.asarray(target_vel, dtype=float)
        closing = float(np.dot(rel, u))
        if closing <= self.min_closing_px_s:
            return None                      # not gaining on it

        # subtract the target's own extent along the approach: contact happens
        # at the edge of its surface, not at its centre
        reach = float(np.hypot(f.a * u[0], f.b * u[1]))
        gap = max(0.0, dist - reach)
        ttc = gap / closing
        if ttc > self.horizon_s:
            return None
        # and it must actually be heading AT it, not merely closing sideways
        if float(np.dot(vel / max(np.linalg.norm(vel), 1e-6), u)) < 0.55:
            return None
        return target, ttc

    # ------------------------------------------------------------------
    FEATURES = ("turn_deg", "lateral_g", "footprint_sep", "pet_s",
                "decel_m_s2", "partner_dev_deg", "partner_drop",
                "own_dev_deg", "own_drop", "crossing_angle_deg",
                "aspect_ratio_change", "is_solo")

    def _features(self, pc, course) -> dict:
        """The physical measurements behind one candidate, for fitting."""
        g = pc.gates or {}
        geometry = str(g.get("geometry", pc.mode) or pc.mode)
        return {
            "t": round(float(pc.t_cross), 3),
            "geometry": geometry,
            "score_rule": round(float(pc.score), 4),
            "track_ids": list(pc.track_ids),
            "boxes": [[round(float(v), 2) for v in box] for box in pc.boxes],
            "turn_deg": float(g.get("turn_deg", pc.angle_deg) or 0.0),
            "lateral_g": float(g.get("lateral_g", 0.0) or 0.0),
            "footprint_sep": float(g.get("footprint_separation", 9.9) or 9.9),
            "pet_s": float(g.get("post_encroachment_s", 9.9) or 9.9),
            "decel_m_s2": float(g.get("decel_m_s2", 0.0) or 0.0),
            "partner_dev_deg": float(course.get("partner_deviation_deg", 0.0) or 0.0),
            "partner_drop": float(course.get("partner_speed_drop", 0.0) or 0.0),
            "own_dev_deg": float(max(pc.deviation_deg)),
            "own_drop": float(max(pc.speed_drop)),
            "crossing_angle_deg": float(g.get("crossing_angle_deg", 0.0) or 0.0),
            "aspect_ratio_change": float(g.get("aspect_ratio_change", 1.0) or 1.0),
            "is_solo": 1.0 if bool(course.get("solo")) else 0.0,
            "n_participants": float(len(pc.boxes)),
            "identity_changed": 1.0 if bool(
                g.get("identity_changed_at_impact", False)) else 0.0,
            "geom_crossing": 1.0 if geometry == "crossing" else 0.0,
            "geom_rear_end": 1.0 if geometry == "rear-end" else 0.0,
            "geom_into_stationary": 1.0 if geometry == "into-stationary" else 0.0,
            "geom_head_on": 1.0 if geometry == "head-on" else 0.0,
            "geom_deflection": 1.0 if geometry == "deflection" else 0.0,
            "geom_rollover": 1.0 if geometry == "rollover" else 0.0,
            "geom_struck_object": 1.0 if geometry == "struck-object" else 0.0,
            "geom_sudden_stop": 1.0 if geometry == "sudden-stop" else 0.0,
            "geom_track_lost": 1.0 if geometry == "track-lost" else 0.0,
        }

    @staticmethod
    def _near_frame_edge(box, frame_shape, margin: float = 40.0) -> bool:
        """Did this vehicle simply drive out of shot?"""
        if not frame_shape:
            return False
        h, w = frame_shape[0], frame_shape[1]
        return (box[0] <= margin or box[1] <= margin
                or box[2] >= w - margin or box[3] >= h - margin)

    def _sudden_stop(self, path, box=None):
        """A vehicle that lost most of its speed in a fraction of a second.

        Whatever it struck need not be tracked -- a divider, a kerb, a barrier,
        an unlit lorry -- so no pair can form and every pairwise test is blind.
        The speed going is the observable that survives.
        """
        if path is None or len(path) < 6:
            return None
        t_end = float(path[-1, 0])
        after = path[path[:, 0] >= t_end - self.stop_window_s]
        before = path[(path[:, 0] < t_end - self.stop_window_s)
                      & (path[:, 0] >= t_end - 2.0 * self.stop_window_s)]
        if len(after) < 2 or len(before) < 2:
            return None

        def speed(seg):
            dt = float(seg[-1, 0] - seg[0, 0])
            if dt <= 1e-6:
                return 0.0
            return float(np.linalg.norm(seg[-1, 1:] - seg[0, 1:]) / dt)

        v0, v1 = speed(before), speed(after)

        # The speed floor is expressed in the vehicle's OWN widths per second,
        # not in pixels. A fixed pixel threshold is a different physical speed
        # at every depth in the frame: a car approaching the camera crawls in
        # image space while doing 60 km/h, which is exactly the case this missed
        # -- one of the two remaining clips is a vehicle driving toward the
        # camera into a divider.
        floor = self.impact_min_speed_px
        if box is not None:
            width = max(1e-6, float(box[2]) - float(box[0]))
            floor = min(floor, self.stop_min_widths_per_s * width)
        if v0 < floor:
            return None
        if v1 > v0 * (1.0 - self.stop_fraction):
            return None
        return float(before[-1, 0]), v0, v1

    def _near_point(self, usable: dict, point, t_now: float) -> list:
        """Vehicles currently sitting near a conflict point, nearest first.

        Used when the original tracks did not survive the impact. Distance is
        scaled by vehicle size so the radius means the same thing near the
        camera and far from it.
        """
        out = []
        for k, (tr, path, _) in usable.items():
            box = tr.box
            w = max(1e-6, float(box[2] - box[0]))
            cx, cy = (box[0] + box[2]) / 2.0, float(box[3])
            d = float(np.hypot(cx - point[0], cy - point[1])) / w
            if d <= self.confirm_radius_widths:
                out.append((d, k, (tr, path, None)))
        out.sort(key=lambda r: r[0])
        return [(k, v) for _, k, v in out]

    @staticmethod
    def _box_at(track, t: float):
        """The vehicle's box at a past moment, from its own history."""
        hist = getattr(track, "box_history", None)
        if not hist:
            return track.box
        ts, box = min(hist, key=lambda kv: abs(kv[0] - t))
        return box if abs(ts - t) <= 0.6 else track.box

    def _predicted_cross(self, pa, pb, t_cross: float) -> bool:
        """Would the two vehicles have been forecast to meet, just before they did?

        Constant-velocity extension of each vehicle's motion in the window
        before the crossing. This is the green cone drawn in the overlay,
        rebuilt for the moment it mattered rather than for the present.
        """
        ext_a = self._extend(pa, t_cross)
        ext_b = self._extend(pb, t_cross)
        if ext_a is None or ext_b is None:
            return False
        return _seg_intersect(ext_a[0], ext_a[1], ext_b[0], ext_b[1]) is not None

    def _extend(self, path, t_cross: float):
        """(start, end) of a vehicle's forward projection from before t_cross."""
        before = path[path[:, 0] < t_cross - 1e-6]
        if len(before) < 2:
            return None
        dt = before[-1, 0] - before[0, 0]
        if dt <= 1e-6:
            return None
        vel = (before[-1, 1:] - before[0, 1:]) / dt
        if float(np.linalg.norm(vel)) < 1e-6:
            return None
        start = before[-1, 1:]
        return start, start + vel * self.horizon_s

    @staticmethod
    def _forecasts_cross(fa, fb) -> bool:
        """Do the two predicted paths intersect within the forecast horizon?

        The forecast reaches roughly a second ahead, so a crossing inside it
        means the vehicles are on course to occupy the same point at close to
        the same moment. That is a stronger statement than "their paths crossed
        at some point in the last two seconds", which every junction satisfies.
        """
        pa = np.asarray(getattr(fa, "points", fa), dtype=float)
        pb = np.asarray(getattr(fb, "points", fb), dtype=float)
        if len(pa) < 2 or len(pb) < 2:
            return False
        for i in range(len(pa) - 1):
            for j in range(len(pb) - 1):
                if _seg_intersect(pa[i], pa[i + 1], pb[j], pb[j + 1]) is not None:
                    return True
        return False

    def _first_crossing(self, pa, pb):
        """Earliest point where the two paths actually intersect.

        Each segment of A is tested only against the segments of B whose own
        extent overlaps it, so the inner loop skips the great majority of
        combinations without touching the intersection arithmetic.
        """
        bx0 = np.minimum(pb[:-1, 1], pb[1:, 1])
        bx1 = np.maximum(pb[:-1, 1], pb[1:, 1])
        by0 = np.minimum(pb[:-1, 2], pb[1:, 2])
        by1 = np.maximum(pb[:-1, 2], pb[1:, 2])

        best = None
        for ia in range(len(pa) - 1):
            ax0, ax1 = sorted((pa[ia, 1], pa[ia + 1, 1]))
            ay0, ay1 = sorted((pa[ia, 2], pa[ia + 1, 2]))
            near = np.nonzero((bx0 <= ax1) & (bx1 >= ax0)
                              & (by0 <= ay1) & (by1 >= ay0))[0]
            for ib in near:
                ib = int(ib)
                hit = _seg_intersect(pa[ia, 1:], pa[ia + 1, 1:],
                                     pb[ib, 1:], pb[ib + 1, 1:])
                if hit is None:
                    continue
                point, t, u = hit
                t_a = self._time_at(pa, point, ia, t)
                t_b = self._time_at(pb, point, ib, u)
                when = max(t_a, t_b)
                if best is None or when < best[0]:
                    best = (when, point, t_a, t_b, ia, ib)
        if best is None:
            return None
        _, point, t_a, t_b, ia, ib = best
        return point, t_a, t_b, ia, ib

    def _find_rollover(self, track):
        """The moment a vehicle's silhouette turned over, if it did.

        Returns ``(when, ratio)`` where ratio is how many times the box aspect
        changed, or ``None``.
        """
        hist = list(getattr(track, "box_history", []) or [])
        if len(hist) < 10:
            return None
        span = max(3, int(self.rollover_span))
        gap = max(2, span // 2)
        best = None
        for i in range(span + gap, len(hist) - span - gap):
            def aspect(seg):
                vals = []
                for _, b in seg:
                    w = max(1e-6, float(b[2]) - float(b[0]))
                    h = max(1e-6, float(b[3]) - float(b[1]))
                    vals.append(w / h)
                return float(np.median(vals))

            # The transition itself is excluded from both windows. Including
            # it produced blended medians -- half the old shape, half the new --
            # and those intermediate values manufactured ratios that crossed the
            # threshold partway through an ordinary turn.
            before = aspect(hist[max(0, i - span - gap):max(1, i - gap)])
            after = aspect(hist[i + gap:i + gap + span])
            if before <= 1e-6:
                continue
            # Directional, not symmetric. A vehicle on its side or roof is
            # WIDER and FLATTER than it was upright, so the aspect ratio has to
            # increase. Accepting a change in either direction meant a car
            # squaring up after a turn -- going from a wide side-on view back to
            # a narrow one, and staying there -- read as a rollover, because it
            # is a large shape change that persists.
            ratio = after / before
            if ratio < self.rollover_ratio:
                continue
            # It must STAY changed, and we must have WAITED long enough to
            # know that. Checking persistence against whatever history exists
            # so far declares a rollover in the middle of a turn, while the
            # vehicle is still presenting its side to the camera and has not
            # yet squared up. Judgement is therefore deferred until the shape
            # has had time to revert if it was going to.
            if (hist[-1][0] - hist[i][0]) < self.rollover_hold_s:
                continue
            tail = hist[-span:]
            held = aspect(tail)
            if max(held / max(after, 1e-6), max(after, 1e-6) / held) > 1.4:
                continue
            if best is None or ratio > best[1]:
                best = (float(hist[i][0]), float(ratio))
        return best

    def _find_kink(self, path, box):
        """A corner in the travelled path too sharp to have been steered.

        Returns ``(when, turn_degrees, lateral_g)`` for the sharpest such corner
        in the recent path, or ``None``.
        """
        if path is None or len(path) < 6:
            return None
        px_per_m = max(1e-6, (float(box[2]) - float(box[0])) / 1.8)
        best = None
        span = 3                     # samples either side of the corner
        for i in range(span, len(path) - span):
            before, after = path[i - span:i + 1], path[i:i + span + 1]
            h0, h1 = self._heading(before), self._heading(after)
            if h0 is None or h1 is None:
                continue
            turn = self._angle_between(h0, h1)
            if turn < self.min_turn_deg:
                continue
            if turn > self.max_turn_deg:
                # Beyond this a "turn" is a heading REVERSAL, which a collision
                # does not produce. It is an identity switch onto a vehicle
                # travelling the other way, or a U-turn at a junction.
                #
                # Measured: across the held-out accidents the largest genuine
                # deflection was 74 degrees, while a false alarm on a crash-free
                # clip reported 161 -- and scored 0.993, higher than every real
                # accident, so no score threshold could ever have separated
                # them. The angle does, with a wide margin.
                continue

            def vel(seg):
                dt_ = seg[-1, 0] - seg[0, 0]
                return None if dt_ <= 1e-6 else (seg[-1, 1:] - seg[0, 1:]) / dt_

            v0, v1 = vel(before), vel(after)
            if v0 is None or v1 is None:
                continue
            speed = float(np.linalg.norm(v0)) / px_per_m
            if speed < self.min_turn_speed_m_s:
                continue

            # Time ACROSS the corner, between the two window midpoints -- not
            # the span of the whole history. Dividing by the window made a
            # violent deflection read as 0.33 g and hid every one of them.
            t0 = 0.5 * (before[0, 0] + before[-1, 0])
            t1 = 0.5 * (after[0, 0] + after[-1, 0])
            dt = max(1e-3, float(t1 - t0))

            dv = float(np.linalg.norm(v1 - v0)) / px_per_m
            lat_g = dv / dt / 9.81
            if lat_g < self.max_grip_g:
                continue
            if lat_g > self.max_plausible_g:
                continue          # a teleport, not a vehicle: identity switch
            # The corner must PERSIST. A single-frame wobble in a noisy trail
            # differentiates into an enormous lateral force and is the main
            # source of false deflections; a vehicle that was actually struck
            # is still travelling on its new heading a moment later.
            tail = path[i:]
            if len(tail) >= span + 2:
                later = self._heading(tail[-(span + 1):])
                if later is None or self._angle_between(h1, later) > 35.0:
                    continue
            if best is None or lat_g > best[2]:
                best = (float(path[i, 0]), float(turn), float(lat_g))
        return best

    def _decel_m_s2(self, path, box, t_cross: float) -> float:
        """How hard the vehicle lost speed across the moment, in m/s^2.

        This is what separates a rear-end collision from braking safely to a
        stop behind another car. Both close on the vehicle ahead, both end with
        the follower stopped, and both leave the footprints adjacent -- so
        proximity and speed-drop alone cannot tell them apart, and an earlier
        version of this test reported the safe stop as a collision.

        A driver brakes at up to roughly 0.9 g. An impact removes the same speed
        in a fraction of the time. Scale comes from the vehicle's own apparent
        width, as everywhere else here.
        """
        before = path[path[:, 0] <= t_cross]
        after = path[path[:, 0] > t_cross]
        if len(before) < 2 or len(after) < 2:
            return 0.0

        def speed(seg):
            dt = seg[-1, 0] - seg[0, 0]
            return 0.0 if dt <= 1e-6 else float(
                np.linalg.norm(seg[-1, 1:] - seg[0, 1:]) / dt)

        dt = max(1e-3, float(after[-1, 0] - before[-1, 0]))
        d_speed = speed(before) - speed(after)
        if d_speed <= 0:
            return 0.0
        px_per_m = max(1e-6, (float(box[2]) - float(box[0])) / 1.8)
        return float(d_speed / dt / px_per_m)

    def _disturbance(self, path, t_cross: float) -> tuple[float, float]:
        """How much the vehicle's heading and speed changed across the crossing.

        This is the condition that distinguishes contact from courtesy. A vehicle
        that crosses another's path and continues on its original heading at its
        original speed was given way to; one whose heading swings or whose speed
        collapses was hit.
        """
        before = path[path[:, 0] <= t_cross]
        after = path[path[:, 0] > t_cross]
        if len(before) < 2 or len(after) < 2:
            return 0.0, 0.0

        h0 = self._heading(before)
        h1 = self._heading(after)
        dev = 0.0 if h0 is None or h1 is None else self._angle_between(h0, h1)

        def speed(seg):
            dt = seg[-1, 0] - seg[0, 0]
            if dt <= 1e-6:
                return 0.0
            return float(np.linalg.norm(seg[-1, 1:] - seg[0, 1:]) / dt)

        s0, s1 = speed(before), speed(after)
        drop = 0.0 if s0 < 1e-6 else float(np.clip((s0 - s1) / s0, 0.0, 1.0))
        return dev, drop
