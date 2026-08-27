"""Forward motion prediction, residuals, and the momentum-exchange test.

Why this channel exists
-----------------------
Every other collision cue in this system measures the *aftermath*: a vehicle at
rest, a vehicle that looks damaged, two footprints overlapping, something left
behind in the background image. All of them confuse with queues and parking, for
the simple reason that a queue is stopped vehicles with overlapping footprints.
No amount of threshold tuning fixes a cue that is measuring the wrong thing.

This module measures the *event*. A collision is an impulse -- an external force
the driver did not apply -- and a motion model is a model of driver-controlled
motion. So the one thing a motion model structurally cannot predict is exactly
the thing we are looking for.

**Newton's third law separates a crash from a queue.** This is the part that
matters, because geometry cannot do it: a rear-end collision and a queue are the
same shape -- vehicles nose to tail, footprints touching -- and this system has
mistaken one for the other repeatedly. They are kinematically opposite. In a
queue both vehicles decelerate, so their momentum changes point the same way and
*add*. In a collision momentum transfers, so the changes oppose and *cancel*.
:func:`momentum_exchange` measures that cancellation and is dimensionless, so it
needs no calibration and mass enters only as a ratio.

What the first version got wrong
--------------------------------
Both errors were caught by measuring on crash-free footage before looking at any
accident clip, which is the only reason they were caught at all.

**The momentum test never ran.** ``velocity_change`` needs samples on both sides
of the moment of interest, and it was being asked about the current instant,
where no later samples exist yet. It returned ``None`` every single time, so no
pair was ever formed -- across 299,499 residuals from sixteen clips, zero pairs
were evaluated. Everything is therefore assessed at a deliberate **lag**: the
question "did these two vehicles exchange momentum" can only be answered a
fraction of a second after the fact, and pretending otherwise silently disabled
the channel.

**Per-frame impulse was noise, not physics.** Scored frame to frame, clean
traffic produced a median of 0.271 and a 99.9th percentile of 0.986 -- ordinary
driving saturating a measure meant to mean "beyond what tyres can do". The
culprit is differentiating a filtered estimate over a 33 ms step, where the
Kalman correction for measurement noise is indistinguishable from real
acceleration. Velocity change is now taken across a window of a few hundred
milliseconds, which is both the physically meaningful interval for an impact and
long enough for detector jitter to average out.

Raw jerk is deliberately never computed. It is the third derivative of a noisy
measurement, and it is what once turned an identity switch into 1,164 px/s and
-1,452 px/s^2 of apparent impact.

A note on NIS
-------------
Normalised innovation squared is retained as a diagnostic but is **not** used as
a gate. In theory it is chi-square(2) distributed and would give a calibrated
test; measured on clean footage it has a median of 0.52 against an expected 1.39
and a 99th percentile of 95.7 against an expected 9.2. The tracker's covariance
is far too confident in the tail, mostly because identity switches and box
jitter on large vehicles are not in its noise model. A test is only calibrated
if the measurement says so, and here it does not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Approximate physical width of each class, metres. Only the ratio to the box
# width matters, so rough values are fine -- they turn pixels into a scale that
# shrinks correctly with distance, which is the whole point.
COCO_WIDTH_M = {1: 0.6, 2: 1.8, 3: 0.8, 5: 2.5, 7: 2.4}
COCO_MASS_KG = {1: 100.0, 2: 1500.0, 3: 200.0, 5: 12000.0, 7: 8000.0}

# UVH-26's Indian taxonomy, resolved by name so it survives index changes.
UVH_WIDTH_M = {
    "two-wheeler": 0.7, "motorcycle": 0.7, "bicycle": 0.6, "cycle": 0.6,
    "auto-rickshaw": 1.4, "auto": 1.4, "e-rickshaw": 1.3,
    "car": 1.8, "suv": 1.9, "van": 2.0, "tempo": 2.0,
    "bus": 2.6, "truck": 2.5, "lcv": 2.2, "tractor": 2.0,
}
UVH_MASS_KG = {
    "two-wheeler": 150.0, "motorcycle": 150.0, "bicycle": 90.0, "cycle": 90.0,
    "auto-rickshaw": 450.0, "auto": 450.0, "e-rickshaw": 450.0,
    "car": 1400.0, "suv": 1900.0, "van": 2200.0, "tempo": 2400.0,
    "bus": 12000.0, "truck": 9000.0, "lcv": 3500.0, "tractor": 4000.0,
}

DEFAULT_WIDTH_M = 1.8
DEFAULT_MASS_KG = 1400.0

# Roughly 0.9 g. Emergency braking reaches it; ordinary driving does not.
MAX_DRIVER_DECEL_M_S2 = 8.8

# The interval an impact resolves over. Long enough that detector jitter
# averages out, short enough that ordinary braking does not fill it.
DEFAULT_WINDOW_S = 0.30

# How far behind the present the impulse question is asked. A momentum exchange
# is only observable once the "after" side of it has been seen.
DEFAULT_LAG_S = 0.40


def _class_lookup(cls, names, table_by_name, table_by_id, default):
    """Resolve a per-class constant under either taxonomy."""
    if names:
        label = str(names.get(int(cls), "")).lower().replace("_", "-")
        if label in table_by_name:
            return table_by_name[label]
        for key, val in table_by_name.items():
            if key in label:
                return val
    return table_by_id.get(int(cls), default)


def vehicle_width_m(cls, names=None) -> float:
    return _class_lookup(cls, names, UVH_WIDTH_M, COCO_WIDTH_M, DEFAULT_WIDTH_M)


def vehicle_mass_kg(cls, names=None) -> float:
    return _class_lookup(cls, names, UVH_MASS_KG, COCO_MASS_KG, DEFAULT_MASS_KG)


def ground_point(box) -> np.ndarray:
    """Bottom-centre of the box: where the vehicle meets the road.

    The tracker's own filter runs on box centres, which sit at half a vehicle's
    height and therefore move whenever the box grows -- a vehicle approaching
    the camera appears to accelerate even at constant speed. The contact point
    does not have that problem, so prediction is done here instead.
    """
    x1, _, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) / 2.0, y2], dtype=float)


def pixels_per_metre(box, cls, names=None) -> float:
    """Local image scale, from the vehicle's own apparent width."""
    x1, _, x2, _ = [float(v) for v in box]
    return max(1e-6, (x2 - x1)) / max(1e-6, vehicle_width_m(cls, names))


@dataclass
class Forecast:
    """Where a vehicle is expected to go, and how confidently."""

    points: np.ndarray            # (N, 2) predicted ground points, pixels
    sigma: np.ndarray             # (N,) 1-sigma radius at each step, pixels
    horizon_s: float
    valid: bool = True

    def envelope(self, k: float = 2.0):
        """The +/-k sigma cone, as two polylines suitable for drawing."""
        if len(self.points) < 2:
            return np.empty((0, 2)), np.empty((0, 2))
        d = np.gradient(self.points, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, np.maximum(norm, 1e-6))
        off = n * (k * self.sigma[:, None])
        return self.points + off, self.points - off


@dataclass
class Residual:
    """How far the vehicle was from its own prediction, for one frame.

    Diagnostic only. Nothing gates on a single frame: see the module docstring
    on why per-frame differentiation of a filtered estimate measures the filter
    rather than the vehicle.
    """

    t: float
    nis: float = 0.0
    lateral_px: float = 0.0
    longitudinal_px: float = 0.0
    speed_m_s: float = 0.0
    trusted: bool = False
    reason: str = ""


@dataclass
class Impulse:
    """A velocity change measured across a window, in metric units."""

    t: float
    dv: np.ndarray                # metres per second, image-plane components
    accel_m_s2: float
    lateral_fraction: float       # 0 = pure along-track, 1 = pure sideways
    speed_m_s: float

    @property
    def score(self) -> float:
        """0..1: how far past driver control authority this was.

        Lateral change is weighted up. A vehicle that slows abruptly may be
        braking hard, which is legal and common; a vehicle shoved sideways at
        speed has been acted on by something. Rating both equally is how a
        traffic light becomes an incident.
        """
        over = min(1.0, self.accel_m_s2 / MAX_DRIVER_DECEL_M_S2)
        return float(np.clip(0.65 * over + 0.35 * self.lateral_fraction, 0.0, 1.0))


class MotionPredictor:
    """Constant-velocity filter on one vehicle's ground contact point.

    Deliberately separate from the tracker's Kalman filter. That one estimates
    box geometry in ``(x, y, aspect, height)`` for association, and rewriting it
    to serve a second purpose would put a working component at risk. This one
    answers a different question -- where on the road will this vehicle be, and
    how surprised should we be if it is not there -- in the coordinates where
    the question makes sense.
    """

    def __init__(self, q_accel_m_s2: float = 2.0, r_px: float = 3.0,
                 history: int = 120) -> None:
        self.q_accel = float(q_accel_m_s2)
        self.r_px = float(r_px)
        self.x: np.ndarray | None = None     # [px, py, vx, vy], px and px/s
        self.P: np.ndarray | None = None
        self.last_t: float | None = None
        self.scale: float = 50.0             # px per metre, most recent
        self.residuals: deque[Residual] = deque(maxlen=history)
        self.vel_history: deque[tuple[float, np.ndarray, float]] = deque(maxlen=history)
        self.box_history: deque[tuple[float, np.ndarray]] = deque(maxlen=history)
        self._gap_frames = 0
        self._bad_until: float = -1e9        # suppressed after an occlusion

    # ------------------------------------------------------------------
    def observe(self, t: float, box, cls, names=None,
                dt_nominal: float = 1 / 30.0, occluded: bool = False) -> Residual:
        """Fold in one observation and report how surprising it was."""
        z = ground_point(box)
        self.scale = pixels_per_metre(box, cls, names)
        self.box_history.append((t, np.asarray(box, dtype=float)))

        if self.x is None or self.last_t is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
            self.P = np.diag([self.r_px ** 2, self.r_px ** 2,
                              (20.0 * self.r_px) ** 2, (20.0 * self.r_px) ** 2])
            self.last_t = t
            res = Residual(t=t, reason="initialising")
            self.residuals.append(res)
            return res

        dt = max(1e-3, float(t - self.last_t))
        self.last_t = t

        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        q_px = self.q_accel * self.scale
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = G @ (np.eye(2) * q_px ** 2) @ G.T
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        R = np.eye(2) * self.r_px ** 2
        nu = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        try:
            nis = float(nu @ np.linalg.solve(S, nu))
        except np.linalg.LinAlgError:
            nis = 0.0

        v_prev = self.x[2:4].copy()
        speed_px = float(np.linalg.norm(v_prev))
        if speed_px > 1e-6:
            u = v_prev / speed_px
            lon = float(nu @ u)
            lat = float(abs(nu[0] * -u[1] + nu[1] * u[0]))
        else:
            lon, lat = 0.0, float(np.linalg.norm(nu))

        K = P_pred @ H.T @ np.linalg.pinv(S)
        self.x = x_pred + K @ nu
        self.P = (np.eye(4) - K @ H) @ P_pred
        self.vel_history.append((t, self.x[2:4].copy(), self.scale))

        res = Residual(t=t, nis=nis, lateral_px=lat, longitudinal_px=lon,
                       speed_m_s=speed_px / max(1e-6, self.scale))

        # Gates. Each closes a false-positive source this project has produced.
        if occluded or self._gap_frames > 0:
            res.reason = "occluded or re-acquired: displacement is not motion"
            self._bad_until = t + 1.0
        elif dt > 3.5 * dt_nominal:
            res.reason = f"frame gap {dt:.2f}s: impact would be unobservable"
            self._bad_until = t + 1.0
        elif len(self.residuals) < 5:
            res.reason = "filter not yet settled"
        elif res.speed_m_s < 1.5:
            res.reason = "too slow for residual direction to mean anything"
        else:
            res.trusted = True

        self.residuals.append(res)
        return res

    def miss(self) -> None:
        self._gap_frames += 1

    def hit(self) -> None:
        self._gap_frames = 0

    # ------------------------------------------------------------------
    def forecast(self, horizon_s: float = 1.2, steps: int = 12) -> Forecast:
        """Propagate the current state forward, carrying its uncertainty.

        Returned as a widening cone rather than a line because that is what the
        filter actually knows. Drawing a bare predicted path would overstate the
        system's confidence to anyone watching it.
        """
        if self.x is None or self.P is None:
            return Forecast(np.empty((0, 2)), np.empty(0), horizon_s, valid=False)
        pts, sig = [], []
        x, P = self.x.copy(), self.P.copy()
        dt = horizon_s / max(1, steps)

        # How fast the vehicle's apparent size is changing, which is how fast
        # it is approaching or receding.
        #
        # Without this the forecast runs off into the sky. Image speed is
        # proportional to apparent size -- both scale as 1/depth -- so a vehicle
        # driving away from the camera slows down in the image even at constant
        # road speed. Extrapolating its *current* image velocity in a straight
        # line therefore overshoots badly, and since receding vehicles travel up
        # the frame, the overshoot points at the horizon and past it, into
        # buildings and sky. Shrinking the projected velocity with the apparent
        # size keeps the line on the road and converging toward the vanishing
        # point, which is where the vehicle is actually going.
        scale_rate = 0.0
        if len(self.vel_history) >= 6:
            (t0, _, s0), (t1, _, s1) = self.vel_history[-6], self.vel_history[-1]
            if t1 - t0 > 1e-6:
                scale_rate = (s1 - s0) / (t1 - t0)
        s_now = max(1e-6, self.scale)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
                     dtype=float)
        q_px = self.q_accel * self.scale
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = G @ (np.eye(2) * q_px ** 2) @ G.T
        for k in range(steps):
            x = F @ x
            P = F @ P @ F.T + Q
            # re-scale the velocity for the depth the vehicle will have reached
            s_k = s_now + scale_rate * (k + 1) * dt
            if s_k <= 0.05 * s_now:
                break                      # it would have reached the horizon
            x[2:4] *= float(np.clip(s_k / s_now, 0.05, 1.0)) ** (1.0 / max(steps, 1))
            pts.append(x[:2].copy())
            sig.append(float(np.sqrt(max(P[0, 0], P[1, 1]))))
        if not pts:
            return Forecast(np.empty((0, 2)), np.empty(0), horizon_s, valid=False)
        return Forecast(np.asarray(pts), np.asarray(sig), horizon_s, valid=True)

    # ------------------------------------------------------------------
    def impulse_at(self, t: float, window_s: float = DEFAULT_WINDOW_S):
        """Velocity change across ``t``, as an :class:`Impulse`, or ``None``.

        Requires observations on *both* sides of ``t``, which is why every
        caller has to ask about a moment already past. The first version of this
        module asked about the present instant, always found nothing after it,
        and silently returned ``None`` on every call -- disabling the channel
        entirely while appearing to work.
        """
        if t <= self._bad_until:
            return None
        before = [(v, s) for (ts, v, s) in self.vel_history
                  if t - window_s <= ts <= t]
        after = [(v, s) for (ts, v, s) in self.vel_history
                 if t < ts <= t + window_s]
        if len(before) < 2 or len(after) < 2:
            return None

        scale = float(np.mean([s for (_, s) in before] + [s for (_, s) in after]))
        scale = max(1e-6, scale)
        v0 = np.mean([v for (v, _) in before], axis=0) / scale
        v1 = np.mean([v for (v, _) in after], axis=0) / scale
        dv = v1 - v0
        mag = float(np.linalg.norm(dv))

        # the two window means are centred about window_s apart
        accel = mag / max(1e-6, window_s)

        speed = float(np.linalg.norm(v0))
        if speed > 1e-6 and mag > 1e-9:
            u = v0 / speed
            lat = abs(float(dv[0] * -u[1] + dv[1] * u[0]))
            lat_frac = float(np.clip(lat / mag, 0.0, 1.0))
        else:
            lat_frac = 0.0

        return Impulse(t=t, dv=dv, accel_m_s2=accel,
                       lateral_fraction=lat_frac, speed_m_s=speed)

    def velocity_change(self, t: float, window_s: float = DEFAULT_WINDOW_S):
        imp = self.impulse_at(t, window_s)
        return None if imp is None else imp.dv

    def box_at(self, t: float):
        """The observed box nearest a past moment."""
        if not self.box_history:
            return None
        ts, box = min(self.box_history, key=lambda kv: abs(kv[0] - t))
        return box if abs(ts - t) <= 0.5 else None


def momentum_exchange(dv_a, m_a: float, dv_b, m_b: float) -> float:
    """How much of the two velocity changes cancelled: 0..1.

    This is the test geometry could never provide. A rear-end collision and a
    queue are the same shape -- nose to tail, footprints touching -- and this
    system has mistaken one for the other repeatedly. They are kinematically
    opposite:

    * **queue** -- both vehicles decelerate, so the momentum changes point the
      same way, add rather than cancel, and this returns near 0;
    * **collision** -- momentum transfers from striker to struck, so the changes
      oppose, largely cancel, and this returns near 1.

    Formally ``1 - |dp_a + dp_b| / (|dp_a| + |dp_b|)``: the fraction of the
    total momentum change that was mutual exchange rather than common-mode
    braking. Dimensionless, so it needs no calibration, and mass enters only as
    a ratio, so rough per-class values suffice.
    """
    if dv_a is None or dv_b is None:
        return 0.0
    dp_a = np.asarray(dv_a, dtype=float) * float(m_a)
    dp_b = np.asarray(dv_b, dtype=float) * float(m_b)
    denom = float(np.linalg.norm(dp_a) + np.linalg.norm(dp_b))
    if denom < 1e-9:
        return 0.0
    return float(np.clip(1.0 - np.linalg.norm(dp_a + dp_b) / denom, 0.0, 1.0))


class ResidualMonitor:
    """Holds one predictor per track and answers questions about the set."""

    def __init__(self, fps: float = 30.0, min_fps: float = 5.0,
                 lag_s: float = DEFAULT_LAG_S,
                 window_s: float = DEFAULT_WINDOW_S,
                 min_samples: int = 3) -> None:
        self.fps = float(fps)
        self.min_fps = float(min_fps)
        self.min_samples = int(min_samples)
        # The window ADAPTS to the frame rate instead of the channel switching
        # itself off. What a velocity estimate actually needs is a few samples
        # either side of the moment, not a particular frame rate -- so the
        # window widens on slow footage until it contains them.
        #
        # The previous fixed 15 fps floor was set by reasoning rather than
        # measurement and was wrong at the boundary in the worst possible way:
        # a held-out clip reporting 14.954 fps was refused by 0.046 fps, which
        # disabled prediction on every vehicle in it and printed the words
        # "15.0 fps is below 15".
        self.window_s = float(max(window_s, self.min_samples / max(fps, 1e-6)))
        self.lag_s = float(max(lag_s, self.window_s + 0.05))
        self.pred: dict[int, MotionPredictor] = {}
        self.names = None

    @property
    def available(self) -> bool:
        """Can a velocity change be measured at all on this footage?

        The requirement is samples, not frame rate, and the window above widens
        to supply them. Only genuinely degenerate footage -- a few frames per
        second, where a vehicle teleports between observations -- is refused.
        """
        return self.fps >= self.min_fps

    def observe(self, tracks, t: float) -> dict[int, Residual]:
        if not self.available:
            return {}
        seen = set()
        out: dict[int, Residual] = {}
        for tr in tracks:
            tid = int(tr.track_id)
            seen.add(tid)
            p = self.pred.get(tid)
            if p is None:
                p = self.pred[tid] = MotionPredictor()
            p.hit()
            out[tid] = p.observe(t, tr.box, tr.cls, self.names,
                                 dt_nominal=1.0 / max(1e-6, self.fps))
        for tid, p in self.pred.items():
            if tid not in seen:
                p.miss()
        return out

    def forecasts(self, tracks, horizon_s: float = 1.2) -> dict[int, Forecast]:
        out = {}
        for tr in tracks:
            p = self.pred.get(int(tr.track_id))
            if p is not None:
                f = p.forecast(horizon_s)
                if f.valid:
                    out[int(tr.track_id)] = f
        return out

    def impulse_pairs(self, tracks, t: float,
                      # Both gates sit AT the 99.9th percentile of crash-free
                      # traffic, measured before any accident clip was looked
                      # at: impulse 0.984, exchange 0.949 over 642,536
                      # residuals and 140,937 adjacent pairs from sixteen
                      # confirmed-clean clips. Requiring BOTH simultaneously is
                      # far stricter than either tail alone.
                      min_impulse: float = 0.85, min_exchange: float = 0.90,
                      max_separation: float = 1.25,
                      coincidence_s: float = 0.20) -> list[dict]:
        """Pairs whose velocity changes look like a momentum exchange.

        Evaluated at ``t - lag_s``, never at ``t``: whether two vehicles
        exchanged momentum is only answerable once the far side of the exchange
        has been observed.

        Three conditions, all required, each removing a failure this system has
        actually produced:

        * both vehicles past driver control authority -- one alone is far more
          likely to be an occlusion or an identity switch than a collision;
        * their footprints in contact at that moment, using the boxes from then
          rather than from now;
        * their momentum changes largely cancelling, which is what separates a
          collision from two vehicles braking in the same queue.
        """
        if not self.available:
            return []
        from .footprint import Footprint, separation

        t_eval = t - self.lag_s
        hot = []
        for tr in tracks:
            p = self.pred.get(int(tr.track_id))
            if p is None:
                continue
            imp = p.impulse_at(t_eval, self.window_s)
            if imp is None or imp.score < min_impulse:
                continue
            box = p.box_at(t_eval)
            if box is None:
                continue
            hot.append((tr, p, imp, box))

        out = []
        for i in range(len(hot)):
            for j in range(i + 1, len(hot)):
                tr_a, pa, ia, box_a = hot[i]
                tr_b, pb, ib, box_b = hot[j]
                sep = separation(Footprint.from_box(box_a),
                                 Footprint.from_box(box_b))
                if sep > max_separation:
                    continue
                ex = momentum_exchange(ia.dv, vehicle_mass_kg(tr_a.cls, self.names),
                                       ib.dv, vehicle_mass_kg(tr_b.cls, self.names))
                if ex < min_exchange:
                    continue
                out.append({
                    "track_ids": [int(tr_a.track_id), int(tr_b.track_id)],
                    "boxes": [np.asarray(box_a, dtype=float),
                              np.asarray(box_b, dtype=float)],
                    "impulse": [round(ia.score, 3), round(ib.score, 3)],
                    "accel_m_s2": [round(ia.accel_m_s2, 1), round(ib.accel_m_s2, 1)],
                    "onset_t": round(t_eval, 2),
                    "momentum_exchange": round(ex, 3),
                    "footprint_separation": round(sep, 2),
                    "score": float(np.clip(0.45 * min(ia.score, ib.score)
                                           + 0.55 * ex, 0.0, 1.0)),
                })
        out.sort(key=lambda d: -d["score"])
        return out

    def drop(self, keep_ids) -> None:
        keep = {int(i) for i in keep_ids}
        for tid in [k for k in self.pred if k not in keep]:
            self.pred.pop(tid, None)
