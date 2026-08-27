"""Per-track physics for the DRONE subsystem — trajectory, speed, acceleration,
momentum.

Everything here operates on **GMC-compensated reference-frame** coordinates
(see ``gmc.py``), never on raw pixels. Raw pixels mix the vehicle's own motion
with the drone's; that mixture is exactly what ``GMCEstimator`` exists to
remove, and every physics quantity below would be meaningless (or actively
misleading) computed before that removal.

Speed
-----
``speed_px_s`` — reference-frame arc length over a trailing time window,
divided by that window. Always computable, always honest: it is a real
displacement in a real coordinate system, just not yet a metric one.

``speed_kmh_estimate`` — the same convention CCTV already uses (see
``CCTV/netra/predict.py``'s ``pixels_per_metre`` and
``CCTV/scripts/render_physics_demo.py``'s on-screen label): a **monocular,
class-width-based scale estimate**, not a calibrated measurement. It divides
the vehicle's own apparent box width by an assumed physical width for its
class to get a local pixels-per-metre figure, then converts speed through
that. This is reported becaues *some* number is more useful than none for a
demo, but it is never presented as metric ground truth — every field it
appears in is suffixed ``_estimate`` or carries an explicit disclaimer string,
and ``config/drone_config.yaml``'s ``road_plane.homography: null`` is checked
first: if/when a real road-plane calibration exists, ``speed_kmh_calibrated``
(computed exactly like ``pipeline_drone._track_kinematics``) is the number to
trust instead, and this module still reports the estimate alongside it so the
two can be compared.

Acceleration
------------
**Never a raw frame-to-frame second derivative.** CCTV's ``netra/predict.py``
module docstring records the specific lesson this module follows: "per-frame
impulse was noise, not physics" — differentiating an already-noisy per-frame
position estimate over a ~33ms step amplifies detector/tracker jitter into
fake acceleration (one identity switch there produced 1,164 px/s and
-1,452 px/s^2 of apparent impact out of nothing). The fix used there, and
here, is the same: compute speed itself over a real trailing window first
(``kinematics.speed_window_s``), then take the discrete derivative of *that*
already-smoothed speed across a second real time gap
(``physics.accel_window_s``), using two non-overlapping windows so the two
speed samples are independent measurements, not the same data read twice.

Momentum
--------
``momentum = mass_kg * speed_m_s_estimate``, using the same
class-width-derived speed estimate above (so it inherits that estimate's
uncertainty — it is not a stronger number than the speed it is built from)
and a per-class mass prior. The mass table below is **not invented for this
module** — it reuses ``CCTV/netra/predict.py``'s ``UVH_MASS_KG`` /
``UVH_WIDTH_M`` values verbatim wherever a VisDrone class has a defensible
correspondence (car, van/tempo, truck, bus, motor/two-wheeler, bicycle,
tricycle/awning-tricycle -> auto-rickshaw), the same reuse discipline this
project applies everywhere else.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from track_drone import ground_point  # noqa: E402  (reuse, don't redefine)

__all__ = [
    "TrackSample",
    "vehicle_width_m",
    "vehicle_mass_kg",
    "windowed_speed",
    "position_at",
    "pixels_per_metre_estimate",
    "compute_acceleration",
    "compute_track_physics",
    "VISDRONE_WIDTH_M",
    "VISDRONE_MASS_KG",
]

# --------------------------------------------------------------------------
# Class-width / mass priors — VisDrone ids: 0 pedestrian, 1 people, 2 bicycle,
# 3 car, 4 van, 5 truck, 6 tricycle, 7 awning-tricycle, 8 bus, 9 motor.
#
# Values are CCTV/netra/predict.py's UVH_WIDTH_M / UVH_MASS_KG table, reused
# verbatim by matching name, not re-derived:
#   car->car, van->tempo(2.0m/2400kg is the closest UVH "van-ish" class; we
#   use UVH's own "van" entry, 2.0m/2200kg), truck->truck, bus->bus,
#   motor->two-wheeler, bicycle->bicycle/cycle,
#   tricycle & awning-tricycle -> auto-rickshaw (closest physical analogue;
#   the awning variant is the same chassis with a sun cover, not a different
#   mass class, so it gets the identical prior rather than an invented one).
# Pedestrian/people (0, 1) are not vehicles; no mass/width prior is defined
# for them and momentum/width fields are simply not computed for those
# classes (checked via cls not in VISDRONE_WIDTH_M below).
# --------------------------------------------------------------------------
VISDRONE_WIDTH_M: dict[int, float] = {
    2: 0.6,    # bicycle    <- UVH bicycle/cycle
    3: 1.8,    # car        <- UVH car
    4: 2.0,    # van        <- UVH van
    5: 2.5,    # truck      <- UVH truck
    6: 1.4,    # tricycle           <- UVH auto-rickshaw
    7: 1.4,    # awning-tricycle    <- UVH auto-rickshaw (same chassis)
    8: 2.6,    # bus        <- UVH bus
    9: 0.7,    # motor      <- UVH two-wheeler/motorcycle
}
VISDRONE_MASS_KG: dict[int, float] = {
    2: 90.0,      # bicycle
    3: 1400.0,    # car
    4: 2200.0,    # van
    5: 9000.0,    # truck
    6: 450.0,     # tricycle
    7: 450.0,     # awning-tricycle
    8: 12000.0,   # bus
    9: 150.0,     # motor
}
DEFAULT_WIDTH_M = 1.8
DEFAULT_MASS_KG = 1400.0


def vehicle_width_m(cls: int) -> float:
    return VISDRONE_WIDTH_M.get(int(cls), DEFAULT_WIDTH_M)


def vehicle_mass_kg(cls: int) -> float:
    return VISDRONE_MASS_KG.get(int(cls), DEFAULT_MASS_KG)


def is_vru_class(cls: int) -> bool:
    """Pedestrian/people — VisDrone 0/1. Excluded from vehicle mass/momentum."""
    return int(cls) in (0, 1)


# --------------------------------------------------------------------------
# Per-frame sample — what the orchestrator records for every live track on
# every processed frame. This is deliberately richer than track_drone.Track's
# own ``history`` (which only keeps a ground point), because the box WIDTH at
# each frame is what the class-width scale estimate needs, and Track does not
# retain a box-size history — only its current box. Keeping this as a
# separate, explicit record (rather than reaching into Track internals) keeps
# physics_drone.py decoupled from track_drone.py's storage choices.
# --------------------------------------------------------------------------

@dataclass
class TrackSample:
    t: float
    ref_box: tuple[float, float, float, float]   # reference-frame x1,y1,x2,y2
    px_box: tuple[float, float, float, float]     # current-frame pixel box (display only)
    score: float
    cls: int


@dataclass
class _Arr:
    """Numpy-backed view of a track's samples for fast windowed queries."""
    t: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    w: np.ndarray          # ref box width, per sample
    cls: np.ndarray
    score: np.ndarray

    @classmethod
    def from_samples(cls_, samples: Sequence[TrackSample]) -> "_Arr":
        t = np.array([s.t for s in samples], dtype=np.float64)
        order = np.argsort(t)
        t = t[order]
        gp = np.array([ground_point(s.ref_box) for s in samples], dtype=np.float64)[order]
        w = np.array([max(0.0, s.ref_box[2] - s.ref_box[0]) for s in samples], dtype=np.float64)[order]
        c = np.array([s.cls for s in samples], dtype=np.int64)[order]
        sc = np.array([s.score for s in samples], dtype=np.float64)[order]
        return _Arr(t=t, gx=gp[:, 0], gy=gp[:, 1], w=w, cls=c, score=sc)

    def window(self, t_end: float, window_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Indices (as t,x,y arrays) with ``t_end - window_s <= t <= t_end``."""
        lo = np.searchsorted(self.t, t_end - window_s, side="left")
        hi = np.searchsorted(self.t, t_end, side="right")
        return self.t[lo:hi], self.gx[lo:hi], self.gy[lo:hi]


def windowed_speed(arr: "_Arr", t_end: float, window_s: float) -> float | None:
    """Reference-frame arc-length speed (px/s) over a trailing causal window.

    Same method as ``track_drone.Track.speed_px``, generalised to an
    arbitrary end time so it can be evaluated at any point in a track's
    history, not just its most recent frame — needed for acceleration (two
    windows at different times) and for the queue/blockage engines (a speed
    at each grid-sample time). Returns ``None`` when fewer than two samples
    fall in the window; the caller must treat that as "no measurement", not
    zero.
    """
    t, x, y = arr.window(t_end, window_s)
    if len(t) < 2:
        return None
    dt = float(t[-1] - t[0])
    if dt <= 1e-6:
        return None
    dist = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return dist / dt


def position_at(arr: "_Arr", t: float, tol: float) -> tuple[float, float] | None:
    """Nearest sample's ground point to ``t``, within ``tol`` seconds, else None."""
    if len(arr.t) == 0:
        return None
    idx = int(np.searchsorted(arr.t, t))
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(arr.t)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(arr.t[i] - t))
    if abs(arr.t[best] - t) > tol:
        return None
    return float(arr.gx[best]), float(arr.gy[best])


def width_at(arr: "_Arr", t: float, tol: float) -> float | None:
    if len(arr.t) == 0:
        return None
    idx = int(np.searchsorted(arr.t, t))
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(arr.t)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(arr.t[i] - t))
    if abs(arr.t[best] - t) > tol:
        return None
    return float(arr.w[best])


def pixels_per_metre_estimate(box_width_px: float, cls: int) -> float:
    """Local monocular scale from the vehicle's own apparent width. ESTIMATE."""
    return max(1e-6, float(box_width_px)) / max(1e-6, vehicle_width_m(cls))


# --------------------------------------------------------------------------
# Acceleration
# --------------------------------------------------------------------------

def compute_acceleration(arr: "_Arr", t_end: float, speed_window_s: float,
                         accel_window_s: float) -> dict[str, Any]:
    """(speed_now - speed_before) / accel_window_s, from two independent,
    non-overlapping trailing-window speed samples. See module docstring.
    """
    speed_now = windowed_speed(arr, t_end, speed_window_s)
    speed_before = windowed_speed(arr, t_end - accel_window_s, speed_window_s)

    if speed_now is None or speed_before is None:
        return {
            "accel_px_s2": None,
            "speed_now_px_s": speed_now,
            "speed_before_px_s": speed_before,
            "accel_reason": "insufficient_history_for_two_windows",
        }

    accel = (speed_now - speed_before) / max(accel_window_s, 1e-6)
    return {
        "accel_px_s2": round(accel, 3),
        "speed_now_px_s": round(speed_now, 3),
        "speed_before_px_s": round(speed_before, 3),
        "accel_reason": "ok",
    }


# --------------------------------------------------------------------------
# Per-track summary (end-of-track / results-JSON entry)
# --------------------------------------------------------------------------

def compute_track_physics(track_id: int, samples: Sequence[TrackSample],
                          cls_name: str, kcfg, pcfg, road_plane, gmc_apply) -> dict[str, Any]:
    """Full physics summary for one completed (or in-progress) track.

    ``gmc_apply`` is ``gmc.apply_gmc`` (passed in rather than imported, to
    avoid a hard import cycle and to make this function pure/testable) — used
    only when a calibrated ``road_plane`` homography exists.
    """
    if not samples:
        return {"physics_reason": "no_samples"}

    arr = _Arr.from_samples(samples)
    last = samples[-1]
    cls = int(last.cls)
    t_end = float(arr.t[-1])
    first_t = float(arr.t[0])
    duration = max(0.0, t_end - first_t)
    sufficient = duration >= kcfg.min_track_seconds

    trajectory = [
        [round(float(t), 3), round(float(x), 1), round(float(y), 1)]
        for t, x, y in zip(arr.t, arr.gx, arr.gy)
    ]

    out: dict[str, Any] = {
        "cls_name": cls_name,
        "duration_s": round(duration, 3),
        "n_samples": len(samples),
        "sufficient_duration": sufficient,
        "trajectory_ref": trajectory,
        "is_vru": is_vru_class(cls),
    }

    if not sufficient:
        out["speed_px_s"] = None
        out["stationary"] = None
        out["metric_reason"] = "track_too_short"
        out["speed_kmh_estimate"] = None
        out["accel_px_s2"] = None
        out["accel_reason"] = "track_too_short"
        out["momentum_kgms_estimate"] = None
        return out

    speed_px_s = windowed_speed(arr, t_end, kcfg.speed_window_s)
    out["speed_px_s"] = round(speed_px_s, 2) if speed_px_s is not None else None
    out["stationary"] = (speed_px_s is not None and speed_px_s <= kcfg.stationary_speed_px_s)

    # -- class-width based estimate (always attemptable for a real vehicle) --
    box_w = width_at(arr, t_end, tol=max(kcfg.speed_window_s, 0.5)) or 0.0
    if is_vru_class(cls) or speed_px_s is None or box_w <= 0:
        out["speed_kmh_estimate"] = None
        out["momentum_kgms_estimate"] = None
        out["mass_kg_assumed"] = None
        out["width_m_assumed"] = None
    else:
        ppm = pixels_per_metre_estimate(box_w, cls)
        speed_m_s_est = speed_px_s / ppm
        out["speed_kmh_estimate"] = round(speed_m_s_est * 3.6, 2)
        out["speed_m_s_estimate"] = round(speed_m_s_est, 3)
        mass = vehicle_mass_kg(cls)
        out["mass_kg_assumed"] = mass
        out["width_m_assumed"] = vehicle_width_m(cls)
        out["momentum_kgms_estimate"] = round(mass * speed_m_s_est, 1)
    out["speed_estimate_note"] = (
        "speed_kmh_estimate / momentum_kgms_estimate are a monocular class-width "
        "scale estimate (box width / assumed physical width for the class), the "
        "same convention CCTV uses (see CCTV/netra/predict.py pixels_per_metre "
        "and CCTV/scripts/render_physics_demo.py). NOT a calibrated measurement."
    )

    # -- acceleration --------------------------------------------------
    if duration >= pcfg.min_track_seconds_accel:
        accel = compute_acceleration(arr, t_end, kcfg.speed_window_s, pcfg.accel_window_s)
        out.update(accel)
    else:
        out["accel_px_s2"] = None
        out["speed_now_px_s"] = None
        out["speed_before_px_s"] = None
        out["accel_reason"] = "track_too_short_for_two_windows"

    if out.get("accel_px_s2") is not None and not (is_vru_class(cls) or box_w <= 0):
        ppm = pixels_per_metre_estimate(box_w, cls)
        out["accel_mps2_estimate"] = round(out["accel_px_s2"] / ppm, 3)
    else:
        out["accel_mps2_estimate"] = None

    # -- calibrated metric speed, only if a real road-plane homography exists
    out["speed_kmh"] = None
    out["speed_m_s"] = None
    if not road_plane.available:
        out["metric_reason"] = "no_road_plane_homography"
    else:
        H = road_plane.matrix()
        t, x, y = arr.window(t_end, kcfg.speed_window_s)
        if len(t) < 2:
            out["metric_reason"] = "insufficient_points_in_window"
        else:
            pts = np.stack([x, y], axis=1)
            metric_pts = gmc_apply(pts, H)
            if np.isnan(metric_pts).any():
                out["metric_reason"] = "road_plane_projection_failed"
            else:
                dt = float(t[-1] - t[0])
                dist_m = float(np.sum(np.hypot(np.diff(metric_pts[:, 0]), np.diff(metric_pts[:, 1]))))
                if dt <= 1e-6:
                    out["metric_reason"] = "zero_time_window"
                else:
                    speed_m_s = dist_m / dt
                    out["speed_m_s"] = round(speed_m_s, 3)
                    out["speed_kmh"] = round(speed_m_s * 3.6, 2)
                    out["metric_reason"] = "ok"

    return out


if __name__ == "__main__":   # pragma: no cover - manual smoke check
    # Synthetic track: a "car" (cls=3) moving at a constant 20 px/frame @ 10fps
    # for 3s, then decelerating hard for 1s. No footage needed.
    samples = []
    t = 0.0
    x = 100.0
    for i in range(30):
        samples.append(TrackSample(t=t, ref_box=(x, 200.0, x + 40.0, 240.0),
                                    px_box=(x, 200.0, x + 40.0, 240.0), score=0.8, cls=3))
        t += 0.1
        x += 20.0
    for i in range(10):
        samples.append(TrackSample(t=t, ref_box=(x, 200.0, x + 40.0, 240.0),
                                    px_box=(x, 200.0, x + 40.0, 240.0), score=0.8, cls=3))
        t += 0.1
        x += 2.0

    import config as drone_config
    cfg = drone_config.load_config()
    result = compute_track_physics(1, samples, "car", cfg.kinematics, cfg.physics,
                                    cfg.road_plane, None)
    import json
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory_ref"}, indent=2))
    print("trajectory points:", len(result["trajectory_ref"]))
