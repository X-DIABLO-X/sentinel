"""Rotation-gated collision pair scoring.

Ported into NETRA from the COMBINED project's second inference
(``IDEAS/COMBINED/src/inference2.py`` + the ``inference2`` section of its
``config.yaml``). The measured result there was **4/4 correct top-1 on the four
labelled ground-truth videos**, up from 1/4 under a purely additive score.

READ THIS BEFORE QUOTING THE NUMBER
-----------------------------------
4/4 is **four videos**. Each is worth 25 percentage points, so 4/4 is not
statistically distinguishable from 3/4 and **must never be quoted as "100%
accuracy"**. Worse, the thresholds below were selected *using those same four
videos* -- ``oncoming_min_deg`` 135 -> 160 and ``prior_unknown`` 0.75 -> 0.60
each moved one labelled video from wrong to correct. **There is no held-out
set.** These are in-sample numbers. All four confirmed collisions were
``crossing``, so ``prior_crossing`` (1.00) versus ``prior_following`` (0.55) is
doing real work and has *never* been tested against a true rear-end; this
module will under-rank one. That is a known, deliberate bias.

Nothing in this module has been re-measured on NETRA footage. The port is
faithful; the *accuracy claim does not travel with it*.

The physics
-----------
1. **Contact search at full sample rate.** Both tracks' boxes and centres are
   linearly interpolated across detector dropouts, the normalised gap
   ``(centre_dist - mean_bbox_diag) / mean_bbox_diag`` is evaluated at every
   step of the overlapping span, and the arg-min is refined *sub-sample* by a
   parabolic fit through the three samples bracketing it. At 30 fps and 15 m/s
   a vehicle moves ~0.5 m per frame -- the same order as the gap being
   resolved, so this is not academic.

2. **Rotation is a multiplicative GATE, not an additive term.**

       Braking decelerates you along your own axis. Being struck rotates you.

   Measured on video 13, a vehicle standing on the brakes to avoid the crash
   ahead scored ``decel 0.93, momentum 0.99, yaw 0.00, aspect 0.00`` and
   outranked the real collision. A vehicle with no rotation now keeps only
   ``rotation_floor`` (0.20) of its kinematic score:
   ``gate = rotation_floor + (1 - rotation_floor) * rot``.

   Two independent rotation measures, because each fails where the other works:
   *yaw/heading shock* is precise while moving but undefined at zero speed --
   which is exactly the state a struck car ends up in; *bbox aspect shock* is a
   shape measurement so it survives the vehicle stopping, but fails under
   occlusion. **Both are gated on the vehicle having been STABLE beforehand.**
   That stability gate is the entire difference between a car mid-turn at a
   junction and a car that was punted sideways.

3. **A track that dies at contact is evidence, not missing evidence.** Video
   13's true participant #1287 has its last frame at the contact instant to
   three decimals. It has no "after", so its rotation is unmeasurable and it
   scored near zero -- backwards, because the tracker lost it *precisely
   because* the vehicle deformed and rotated. ``break_implies_rotation`` (0.70)
   substitutes for the rotation that could not be observed. Tightened to
   *termination only*, and only after the track was established >= 1 s: an
   earlier version that also counted tracks *starting* near contact fired 1.0
   on every vehicle in the scene, and a feature that is 1.0 for everything
   carries no information.

4. **Interaction geometry is a prior on how much evidence to demand, never a
   veto.**

   ===========  ==============  ==========================================
   type         rel. heading    what it usually is
   ===========  ==============  ==========================================
   crossing     45-160 deg      T-bone, the classic intersection collision
   following    0-45 deg        queuing / brake-to-avoid (rarely a rear-end)
   oncoming     160-180 deg     two vehicles passing safely on opposite
                                sides, whose 2-D boxes overlap only because
                                a box cannot encode depth (rarely head-on)
   ===========  ==============  ==========================================

   The oncoming threshold moved 135 -> 160 because video 11's *real* 138 deg
   collision was being demoted as "passing traffic". Genuine passing traffic is
   near-exactly antiparallel (video 13's false positive: 178.4 deg).

5. **Pair score = geometric_mean(evidence_a, evidence_b) x contact x prior.**
   *Geometric* specifically: an arithmetic mean let one violently anomalous
   vehicle drag a placid bystander into the top pair, which is how video 14
   paired #1805 with #1902 instead of #1896 with #1902.

What changed in the port (and why)
----------------------------------
The original read pre-computed offline tracking JSON from disk, with per-sample
``speed_kmh``, ``heading_deg``, ``accel_mps2`` and ``momentum_kgms`` already in
the file. This module takes **live NETRA** :class:`netra.track.Track` objects as
arguments and derives all of that in-process. No file paths, no disk reads.

* **Samples** come from ``Track.box_history`` -- ``(t, box)`` pairs pushed on
  every tracker update. Detector dropouts are exactly the holes in it, which is
  what the interpolation is for.
* **Metric units are recovered per-track** with :func:`netra.predict.pixels_per_metre`
  (the vehicle's own apparent width against its class's physical width) and
  :func:`netra.predict.vehicle_mass_kg`. This is the same monocular per-track
  scale the original used, so ``ref_decel``, ``ref_speed_drop_kmh`` and
  ``ref_momentum`` keep their COMBINED values and their meaning. It is a
  *monocular* scale: single-sample acceleration is genuinely noisy, which is
  why ``ref_decel`` is 30.0 and not 8.0 (see the config comments).
* **Motion (speed / heading / acceleration) is measured on the ground point**,
  the bottom-centre of the box, per NETRA convention -- box centres sit at half
  a vehicle's height and drift upward as the box grows, so a vehicle
  approaching the camera appears to accelerate at constant speed. **The contact
  gap is measured on box centres**, verbatim from the original, because that is
  the geometry the -0.007 / -0.482 / -0.427 versus +0.36 / +0.86 / +3.51
  separation was measured on. (``netra.footprint.separation`` is the
  calibrated alternative; swapping it in would be a change worth measuring, not
  a free improvement.)
* **``track_break`` needs a liveness test that the offline version did not.**
  Offline, every track has genuinely ended by the time it is scored. Live,
  *every* track's ``last_t`` is "now", so the naive test would fire the break
  bonus on the entire scene. A break is therefore only counted when the tracker
  has actually stopped updating the track -- ``state`` is LOST/REMOVED, or
  ``last_t`` is older than ``now_t`` by more than ``track_break_window_s``.
  See :func:`build_windows`.
* **An unavailable channel is dropped, not scored as zero.** NETRA ships with
  the crash-appearance classifier off by default (measured anti-correlated
  in-domain -- see ``events/collision.py``), so ``appearance`` is normally
  ``None``. ``None`` removes the channel from *both* the numerator and the
  denominator of the kinematic sum. Passing ``0.0`` instead means "the
  classifier looked and saw nothing", which is a different claim.

Integration surface (a later step wires this into ``collision.py``)
-------------------------------------------------------------------
::

    from netra.events.rotation_gate import RotationGateConfig, score_pairs

    cfg = RotationGateConfig.from_config(config.get("collision", {}))
    pairs = score_pairs(ctx.tracks, cfg, now_t=ctx.t,
                        classes=MOTORISED_CLASSES | VULNERABLE_CLASSES)
    if pairs:
        best = pairs[0]                 # PairResult, highest score first
        best.track_ids                  # (a, b)
        best.score                      # 0..1
        best.interaction                # crossing / following / oncoming / unknown
        best.as_dict()                  # every number, for the triggers blob
        best.explain()                  # one human-readable sentence

``score_pairs`` is read-only with respect to the tracks: it mutates nothing and
raises no events. It returns an empty list when no pair came within
``pair_max_gap`` vehicle-diagonals of each other, which is the honest answer.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..geometry import heading_degrees
from ..predict import ground_point, pixels_per_metre, vehicle_mass_kg

__all__ = [
    "RotationGateConfig",
    "TrackSample",
    "TrackWindow",
    "ContactEvent",
    "ImpactEvidence",
    "PairResult",
    "CROSSING",
    "FOLLOWING",
    "ONCOMING",
    "UNKNOWN",
    "ang_diff",
    "circular_mean_deg",
    "circular_std_deg",
    "saturate",
    "parabolic_vertex",
    "samples_from_track",
    "build_windows",
    "find_contact",
    "contact_strength",
    "rotation_gate",
    "impact_evidence",
    "relative_heading",
    "interaction_prior",
    "score_pair",
    "score_pairs",
]

Box = tuple[float, float, float, float]

CROSSING = "crossing"
FOLLOWING = "following"
ONCOMING = "oncoming"
UNKNOWN = "unknown"

# Tracker lifecycle states that mean "this track is no longer being updated".
# Mirrors netra.track.LOST / REMOVED without importing them, so this module
# stays importable in isolation.
_DEAD_STATES = frozenset({"lost", "removed"})


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RotationGateConfig:
    """The tuned constants, defaults verbatim from COMBINED ``config.yaml``.

    Every default below was copied from the ``inference2:`` section of
    ``D:/HARSHIT/ELCIA/IDEAS/COMBINED/config.yaml`` and is pinned by
    ``tests/test_rotation_gate.py::test_defaults_match_combined_config``. They
    were tuned on four labelled videos with no held-out set -- see the module
    docstring. Change one and you are re-tuning, not configuring.
    """

    # -- candidate gates ---------------------------------------------------
    # A track that never travels this many of its OWN body diagonals is parked,
    # or is street furniture the detector called a car.
    min_track_samples: int = 8            # config key: min_track_frames
    min_travel_diagonals: float = 1.0
    min_max_speed_kmh: float = 5.0

    # -- windows -----------------------------------------------------------
    # Anomalies are evaluated around the CONTACT instant, not over the whole
    # track. Measured on video 14: both true participants are tracked ~520
    # frames and their global peak anomalies fall nowhere near the collision.
    contact_window_s: float = 1.0
    heading_lookback_s: float = 1.0

    # -- stability gates ---------------------------------------------------
    min_speed_for_heading_kmh: float = 6.0    # direction of travel is
                                              # meaningless below this
    stable_heading_std_deg: float = 25.0      # the gate that separates a car
                                              # mid-turn from a car struck
    stable_aspect_rel_std: float = 0.30       # same idea for the silhouette

    # -- saturating references (0 at 0, ~0.63 at ref, -> 1 beyond) ---------
    ref_yaw_deg: float = 45.0
    ref_aspect: float = 0.60
    # ref_decel was 8.0, which saturated EVERY vehicle to 0.85-1.00 and made
    # the term carry no information: a monocular per-track scale makes
    # single-sample acceleration very noisy (peaks of 17-67 m/s^2 are routine).
    ref_decel: float = 30.0                   # m/s^2
    ref_speed_drop_kmh: float = 20.0
    ref_momentum: float = 12000.0             # kg m/s

    # -- the rotation gate -------------------------------------------------
    rotation_floor: float = 0.20
    break_implies_rotation: float = 0.70
    break_floor_score: float = 0.55
    track_break_window_s: float = 0.4
    track_break_min_life_s: float = 1.0

    # -- contact -----------------------------------------------------------
    pair_ref_gap: float = 0.5                 # contact = 1/(1 + max(0,gap)/ref)
    pair_max_gap: float = 1.5                 # never came within this many
                                              # vehicle-sizes -> not a candidate

    # -- interaction geometry priors (evidence demanded, never a veto) -----
    following_max_deg: float = 45.0
    oncoming_min_deg: float = 160.0           # 135 demoted a real 138 deg crash
    prior_crossing: float = 1.00
    prior_following: float = 0.55
    prior_oncoming: float = 0.50
    prior_unknown: float = 0.60               # absence of evidence should rank
                                              # below verified crossing

    # -- weights -----------------------------------------------------------
    w_rotation: float = 0.45                  # the gate also carries the
                                              # largest single share
    w_decel: float = 0.15
    w_speed_drop: float = 0.15
    w_momentum: float = 0.10
    w_appearance: float = 0.35
    w_track_break: float = 0.25

    # -- NETRA-side cost control (not from COMBINED) -----------------------
    max_pairs: int = 400
    # Resampling floor/ceiling for the contact search grid, seconds. Purely a
    # numerical guard against a degenerate timestamp sequence.
    min_step_s: float = 1.0 / 240.0
    max_step_s: float = 0.5

    # -- provenance --------------------------------------------------------
    source: str = "COMBINED/config.yaml :: inference2"

    @classmethod
    def from_config(cls, section: Mapping[str, Any] | None) -> "RotationGateConfig":
        """Build from a NETRA config dict, accepting the COMBINED key names.

        Unknown keys are ignored rather than raising, so a config file can carry
        settings for the rest of the collision engine alongside these.
        """
        if not section:
            return cls()
        sub = section.get("rotation_gate", section)
        if not isinstance(sub, Mapping):
            return cls()

        aliases = {
            "min_track_frames": "min_track_samples",
            "min_speed_for_heading": "min_speed_for_heading_kmh",
        }
        flat: dict[str, Any] = {}
        for key, val in sub.items():
            name = aliases.get(str(key), str(key))
            flat[name] = val

        # ``weights: {rotation: 0.45, ...}`` -> ``w_rotation`` etc.
        weights = sub.get("weights")
        if isinstance(weights, Mapping):
            for key, val in weights.items():
                flat[f"w_{key}"] = val

        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in flat.items() if k in fields})

    # -- derived -----------------------------------------------------------
    @property
    def kinematic_weights(self) -> dict[str, float]:
        """Weights of the ungated (speed-loss side) channels."""
        return {
            "decel": self.w_decel,
            "speed_drop": self.w_speed_drop,
            "momentum": self.w_momentum,
            "appearance": self.w_appearance,
            "track_break": self.w_track_break,
        }


DEFAULT_CONFIG = RotationGateConfig()


# ---------------------------------------------------------------------------
# angle / normalisation helpers
#
# Headings live on a circle, so plain arithmetic is wrong: 179 and -179 are two
# degrees apart, not 358.
# ---------------------------------------------------------------------------

def ang_diff(a: float, b: float) -> float:
    """Smallest signed difference ``a - b`` in degrees, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles: Sequence[float]) -> float:
    """Mean direction of a set of headings, in degrees."""
    if len(angles) == 0:
        return 0.0
    r = np.radians(np.asarray(angles, dtype=float))
    return float(np.degrees(math.atan2(float(np.mean(np.sin(r))),
                                       float(np.mean(np.cos(r))))))


def circular_std_deg(angles: Sequence[float]) -> float:
    """Circular standard deviation in degrees; 0 for fewer than two samples."""
    if len(angles) < 2:
        return 0.0
    r = np.radians(np.asarray(angles, dtype=float))
    R = math.hypot(float(np.mean(np.cos(r))), float(np.mean(np.sin(r))))
    if R >= 1.0:
        return 0.0
    if R <= 1e-9:
        return 180.0
    return float(np.degrees(math.sqrt(-2.0 * math.log(R))))


def saturate(x: float, ref: float) -> float:
    """Saturating normaliser: 0 at 0, ~0.63 at ``ref``, asymptotic to 1.

    Keeps one enormous outlier from dominating a weighted sum -- which matters
    here because monocular acceleration estimates routinely spike.
    """
    if ref <= 0:
        return 0.0
    return float(1.0 - math.exp(-max(0.0, x) / ref))


def parabolic_vertex(y0: float, y1: float, y2: float) -> float:
    """Offset of the parabola's minimum from the middle sample, in samples.

    Fits ``y`` through three equally spaced samples and returns the vertex
    offset. When ``y1`` is the smallest of the three the result lies in
    [-0.5, +0.5], so the refined instant is always strictly bracketed by the two
    neighbouring samples. Returns 0.0 for a degenerate (flat or linear) fit
    rather than dividing by a near-zero denominator.
    """
    den = y0 - 2.0 * y1 + y2
    if abs(den) <= 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))


# ---------------------------------------------------------------------------
# samples: NETRA track history -> the per-sample record the algorithm wants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackSample:
    """One observation of one vehicle, with the kinematics derived in-process.

    The original read these fields from an offline tracking JSON. Here they are
    computed from ``Track.box_history`` plus the class-based monocular scale.
    """

    t: float
    box: tuple[float, float, float, float]
    cx: float                  # box centre -- used for the contact gap
    cy: float
    gx: float                  # ground point (bottom-centre) -- used for motion
    gy: float
    diag: float                # box diagonal, px: the local size normaliser
    aspect: float              # box width / height
    speed_kmh: float
    heading_deg: float         # 0 = up the frame, 90 = right (netra.geometry)
    accel_mps2: float          # negative = braking
    momentum_kgms: float
    px_per_m: float


def _central_pair(i: int, n: int, span: int) -> tuple[int, int]:
    """Indices bracketing ``i`` for a centred difference, clipped to the ends."""
    return max(0, i - span), min(n - 1, i + span)


def samples_from_track(track: Any, names: Mapping[int, str] | None = None,
                       *, smooth_span: int = 2) -> list[TrackSample]:
    """Derive the per-sample kinematic record from a NETRA :class:`Track`.

    Reads ``track.box_history`` -- a deque of ``(t, box)`` pushed on every
    tracker update. Nothing is read from disk and the track is not mutated.

    ``smooth_span`` is the half-width, in samples, of the centred difference
    used for velocity. Differencing adjacent samples amplifies detector jitter
    into phantom heading swings, which is the failure mode
    :func:`netra.geometry.robust_direction` exists to avoid; +/-2 samples is
    the same idea at per-sample resolution.
    """
    hist = list(getattr(track, "box_history", ()) or ())
    if len(hist) < 2:
        return []

    hist.sort(key=lambda item: float(item[0]))
    ts = np.asarray([float(t) for t, _ in hist], dtype=float)
    boxes = np.asarray([np.asarray(b, dtype=float).reshape(4) for _, b in hist],
                       dtype=float)
    n = len(ts)

    cls = int(getattr(track, "cls", 2))
    mass = float(vehicle_mass_kg(cls, names))

    gpts = np.asarray([ground_point(b) for b in boxes], dtype=float)
    ppm = np.asarray([pixels_per_metre(b, cls, names) for b in boxes], dtype=float)

    # -- speed and heading, from the ground point -------------------------
    speeds_mps = np.zeros(n, dtype=float)
    headings = np.zeros(n, dtype=float)
    last_heading = 0.0
    for i in range(n):
        j0, j1 = _central_pair(i, n, smooth_span)
        dt = ts[j1] - ts[j0]
        if dt <= 1e-9 or j0 == j1:
            headings[i] = last_heading
            continue
        d = gpts[j1] - gpts[j0]
        dist_px = float(math.hypot(d[0], d[1]))
        # local scale averaged over the differencing interval
        scale = float(np.mean(ppm[j0:j1 + 1])) if j1 > j0 else float(ppm[i])
        speeds_mps[i] = (dist_px / max(scale, 1e-6)) / dt
        if dist_px > 1e-6:
            last_heading = float(heading_degrees((float(d[0]), float(d[1]))))
        headings[i] = last_heading

    # -- acceleration, from the speed series ------------------------------
    accels = np.zeros(n, dtype=float)
    for i in range(n):
        j0, j1 = _central_pair(i, n, 1)
        dt = ts[j1] - ts[j0]
        if dt > 1e-9 and j1 > j0:
            accels[i] = (speeds_mps[j1] - speeds_mps[j0]) / dt

    out: list[TrackSample] = []
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        w, h = max(1e-3, x2 - x1), max(1e-3, y2 - y1)
        out.append(TrackSample(
            t=float(ts[i]),
            box=(x1, y1, x2, y2),
            cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0,
            gx=float(gpts[i][0]), gy=float(gpts[i][1]),
            diag=float(math.hypot(w, h)),
            aspect=float(w / h),
            speed_kmh=float(speeds_mps[i] * 3.6),
            heading_deg=float(headings[i]),
            accel_mps2=float(accels[i]),
            momentum_kgms=float(mass * speeds_mps[i]),
            px_per_m=float(ppm[i]),
        ))
    return out


# ---------------------------------------------------------------------------
# full-sample-rate track access with gap interpolation
# ---------------------------------------------------------------------------

class TrackWindow:
    """Position / box / kinematics for one track at ANY instant in its span.

    Detector dropouts leave holes in a track. A collision lasts a handful of
    frames, so a hole can swallow the true closest approach. Linear
    interpolation across the holes means the contact search sees a continuous
    trajectory instead of one with the critical moment missing.

    ``first_t`` / ``last_t`` are the track's *lifetime* bounds (from the Track
    object when available), which can be wider than the sample span because
    ``box_history`` is a bounded deque. ``ended`` says whether the tracker has
    actually stopped updating this track -- see the module docstring on why the
    offline version did not need it.
    """

    __slots__ = ("track_id", "cls", "samples", "first_t", "last_t", "ended",
                 "_t", "_cx", "_cy", "_bb", "_diag", "_max_speed_kmh",
                 "_travel_diagonals", "track")

    def __init__(self, samples: Sequence[TrackSample], *, track_id: int, cls: int,
                 first_t: float | None = None, last_t: float | None = None,
                 ended: bool = False, track: Any = None) -> None:
        if len(samples) < 2:
            raise ValueError("a TrackWindow needs at least two samples")
        self.samples = list(samples)
        self.track_id = int(track_id)
        self.cls = int(cls)
        self.track = track
        self.ended = bool(ended)

        self._t = np.asarray([s.t for s in self.samples], dtype=float)
        self._cx = np.asarray([s.cx for s in self.samples], dtype=float)
        self._cy = np.asarray([s.cy for s in self.samples], dtype=float)
        self._bb = np.asarray([s.box for s in self.samples], dtype=float)
        self._diag = np.asarray([s.diag for s in self.samples], dtype=float)

        self.first_t = float(self._t[0] if first_t is None else first_t)
        self.last_t = float(self._t[-1] if last_t is None else last_t)

        self._max_speed_kmh = float(max(s.speed_kmh for s in self.samples))
        travel = 0.0
        for a, b in zip(self.samples, self.samples[1:]):
            travel += math.hypot(b.gx - a.gx, b.gy - a.gy)
        mean_diag = float(np.mean(self._diag)) if len(self._diag) else 1.0
        self._travel_diagonals = travel / max(1e-6, mean_diag)

    # -- span --------------------------------------------------------------
    @property
    def t0(self) -> float:
        """First sampled instant (not the same as ``first_t``)."""
        return float(self._t[0])

    @property
    def t1(self) -> float:
        return float(self._t[-1])

    @property
    def max_speed_kmh(self) -> float:
        return self._max_speed_kmh

    @property
    def travel_diagonals(self) -> float:
        """Distance travelled in units of the vehicle's own body diagonal."""
        return self._travel_diagonals

    @property
    def median_step_s(self) -> float:
        if len(self._t) < 2:
            return 1.0 / 30.0
        d = np.diff(self._t)
        d = d[d > 1e-9]
        return float(np.median(d)) if len(d) else 1.0 / 30.0

    def covers(self, t: float) -> bool:
        return self.t0 <= t <= self.t1

    # -- interpolated access ----------------------------------------------
    def centre(self, t: float) -> tuple[float, float]:
        """Box centre at ``t``, interpolated across detector dropouts."""
        return (float(np.interp(t, self._t, self._cx)),
                float(np.interp(t, self._t, self._cy)))

    def box(self, t: float) -> Box:
        return tuple(float(np.interp(t, self._t, self._bb[:, i])) for i in range(4))

    def diag(self, t: float) -> float:
        return float(np.interp(t, self._t, self._diag))

    def window(self, t_centre: float, half: float) -> list[TrackSample]:
        """Real (uninterpolated) samples within ``half`` seconds of ``t_centre``."""
        return [s for s in self.samples if abs(s.t - t_centre) <= half]

    def __repr__(self) -> str:                                  # pragma: no cover
        return (f"TrackWindow(id={self.track_id}, n={len(self.samples)}, "
                f"t={self.t0:.2f}..{self.t1:.2f}, ended={self.ended})")


def build_windows(tracks: Iterable[Any], cfg: RotationGateConfig | None = None,
                  *, now_t: float | None = None,
                  classes: Iterable[int] | None = None,
                  names: Mapping[int, str] | None = None) -> list[TrackWindow]:
    """Turn NETRA tracks into scoreable windows, applying the candidate gates.

    Three gates, all from COMBINED's ``inference2`` section, all there to keep
    street furniture out of the ranking:

    * ``min_track_samples`` -- too few observations to say anything;
    * ``min_travel_diagonals`` -- never moved a body-length, so it is parked or
      it is a signboard the detector called a car;
    * ``min_max_speed_kmh`` -- never actually drove.

    ``now_t`` is the current pipeline time. It is what makes ``track_break``
    honest in a live system: without it every live track looks like it "ended
    at contact", because live tracks always end at the present moment.
    """
    cfg = cfg or DEFAULT_CONFIG
    allowed = set(classes) if classes is not None else None

    out: list[TrackWindow] = []
    for tr in tracks:
        cls = int(getattr(tr, "cls", -1))
        if allowed is not None and cls not in allowed:
            continue

        samples = samples_from_track(tr, names)
        if len(samples) < max(2, int(cfg.min_track_samples)):
            continue

        first_t = float(getattr(tr, "first_t", samples[0].t))
        last_t = float(getattr(tr, "last_t", samples[-1].t))

        state = str(getattr(tr, "state", "") or "").lower()
        if state in _DEAD_STATES:
            ended = True
        elif now_t is None:
            # No clock supplied. Refuse to guess: without a reference "now",
            # "this track ended" is unmeasurable, and asserting it would hand
            # break_implies_rotation to every vehicle in the scene.
            ended = False
        else:
            ended = (float(now_t) - last_t) > cfg.track_break_window_s

        try:
            win = TrackWindow(samples, track_id=int(getattr(tr, "track_id", -1)),
                              cls=cls, first_t=first_t, last_t=last_t,
                              ended=ended, track=tr)
        except ValueError:
            continue

        if win.travel_diagonals < cfg.min_travel_diagonals:
            continue
        if win.max_speed_kmh < cfg.min_max_speed_kmh:
            continue
        out.append(win)
    return out


# ---------------------------------------------------------------------------
# contact search
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactEvent:
    """Closest approach between two tracks, refined sub-sample."""

    t: float                 # refined contact instant, seconds
    t_grid: float            # the discrete grid instant before refinement
    gap: float               # normalised: <=0 boxes overlap, ~1 one vehicle apart
    dist_px: float           # raw centre distance at the refined instant
    refined: bool            # whether the parabolic fit could be applied
    n_steps: int

    def as_dict(self) -> dict[str, Any]:
        return {"t": round(self.t, 4), "t_grid": round(self.t_grid, 4),
                "gap": round(self.gap, 4), "dist_px": round(self.dist_px, 1),
                "refined": self.refined, "n_steps": self.n_steps}


def find_contact(a: TrackWindow, b: TrackWindow,
                 cfg: RotationGateConfig | None = None) -> ContactEvent | None:
    """Closest approach at full sample rate, refined sub-sample.

    The gap is normalised by the two vehicles' own mean bbox diagonal, so it
    means the same thing near and far from the camera::

        gap <= 0    boxes overlap -- touching, in image terms
        gap ~= 1    about one vehicle apart

    Measured on the confirmed pairs this was the single cleanest discriminator
    available: true pairs at -0.007 / -0.482 / -0.427 against the best *wrong*
    pairs at +0.36 / +0.86 / +3.51. No kinematic feature came close.

    Returns ``None`` when the two tracks were never on screen together.
    """
    cfg = cfg or DEFAULT_CONFIG
    lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
    if hi < lo:
        return None

    step = float(np.clip(min(a.median_step_s, b.median_step_s),
                         cfg.min_step_s, cfg.max_step_s))
    if hi - lo <= 1e-9:
        ts = np.asarray([lo], dtype=float)
    else:
        n = int(math.floor((hi - lo) / step)) + 1
        ts = lo + step * np.arange(n, dtype=float)
        if ts[-1] < hi - 1e-9:
            ts = np.append(ts, hi)

    gaps = np.empty(len(ts), dtype=float)
    for i, t in enumerate(ts):
        (ax, ay), (bx, by) = a.centre(t), b.centre(t)
        d = math.hypot(ax - bx, ay - by)
        r = 0.5 * (a.diag(t) + b.diag(t))
        gaps[i] = (d - r) / max(1e-6, r)

    k = int(np.argmin(gaps))
    t_grid = float(ts[k])
    t_star, refined = t_grid, False
    # Parabolic sub-sample refinement. At 30 fps and 15 m/s a vehicle covers
    # ~0.5 m per frame, comparable to the gap being resolved, so the discrete
    # minimum is not good enough.
    if 0 < k < len(ts) - 1:
        delta = parabolic_vertex(float(gaps[k - 1]), float(gaps[k]), float(gaps[k + 1]))
        if delta != 0.0:
            local_step = (ts[k + 1] - ts[k - 1]) / 2.0
            t_star = float(np.clip(t_grid + delta * local_step,
                                   float(ts[k - 1]), float(ts[k + 1])))
            refined = True

    (ax, ay), (bx, by) = a.centre(t_star), b.centre(t_star)
    return ContactEvent(t=t_star, t_grid=t_grid, gap=float(gaps[k]),
                        dist_px=math.hypot(ax - bx, ay - by),
                        refined=refined, n_steps=len(ts))


def contact_strength(gap: float, cfg: RotationGateConfig | None = None) -> float:
    """Contact as a GATE, not a bonus: 1.0 when overlapping, decaying beyond.

    A pair that never touched is not a collision however anomalous each vehicle
    looked on its own -- so this multiplies the pair score rather than nudging
    it. It is necessary and nowhere near sufficient: in dense traffic most pairs
    of boxes overlap at some point through occlusion alone, which is why the
    evidence below is evaluated *at this instant* rather than over whole tracks.
    """
    cfg = cfg or DEFAULT_CONFIG
    return 1.0 / (1.0 + max(0.0, gap) / max(1e-9, cfg.pair_ref_gap))


# ---------------------------------------------------------------------------
# per-vehicle impact evidence, with rotation as a gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImpactEvidence:
    """Everything measured about one vehicle at one contact instant.

    Kept as raw-ish numbers so a caller (or a jury) can see *why* the score is
    what it is rather than being handed one opaque float.
    """

    track_id: int
    rotation: float          # 0..1 -- the gate
    gate: float              # rotation_floor + (1-floor)*rotation
    yaw_deg: float           # raw heading shock, degrees
    aspect: float            # raw relative silhouette shock
    decel: float             # normalised 0..1
    speed_drop: float        # normalised 0..1
    momentum_drop: float     # normalised 0..1
    track_break: float       # 0 or 1
    appearance: float | None # None = channel unavailable, not "saw nothing"
    kinematic: float         # the speed-loss side, ungated
    score: float             # final, rotation-gated
    n_window: int
    channels_used: tuple[str, ...] = ()
    yaw_gate_passed: bool = False
    aspect_gate_passed: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key, val in list(d.items()):
            if isinstance(val, float):
                d[key] = round(val, 4)
        d["channels_used"] = list(self.channels_used)
        return d


def rotation_gate(rot: float, cfg: RotationGateConfig | None = None) -> float:
    """``rotation_floor + (1 - rotation_floor) * rot`` -- the multiplicative gate.

    Braking decelerates you along your own axis; being struck rotates you. A
    vehicle with no rotation keeps only ``rotation_floor`` (0.20) of whatever
    its kinematics claimed. That single change is what demoted the
    brake-to-avoid bystanders on video 13 from outranking the real collision to
    0.03-0.05 against the true pair's 0.39.
    """
    cfg = cfg or DEFAULT_CONFIG
    floor = cfg.rotation_floor
    return floor + (1.0 - floor) * float(np.clip(rot, 0.0, 1.0))


def _yaw_shock(pre: Sequence[TrackSample], post: Sequence[TrackSample],
               cfg: RotationGateConfig) -> tuple[float, bool]:
    """Heading shock across the contact instant, degrees, and whether it counted.

    Precise while the vehicle is moving; MEANINGLESS once it stops, because
    velocity direction is undefined at zero speed -- which is exactly the state
    a struck vehicle ends up in. Hence the aspect measure below.

    Gated on the heading having been STABLE beforehand. Without that gate a car
    turning through a junction scores identically to a car punted sideways.
    """
    ms = cfg.min_speed_for_heading_kmh
    pre_moving = [s for s in pre if s.speed_kmh >= ms]
    if len(pre_moving) < 3:
        return 0.0, False
    heads = [s.heading_deg for s in pre_moving]
    if circular_std_deg(heads) > cfg.stable_heading_std_deg:
        return 0.0, False
    base = circular_mean_deg(heads)
    post_moving = [s for s in post if s.speed_kmh >= ms]
    if not post_moving:
        return 0.0, True
    return max(abs(ang_diff(s.heading_deg, base)) for s in post_moving), True


def _aspect_shock(pre: Sequence[TrackSample], post: Sequence[TrackSample],
                  cfg: RotationGateConfig) -> tuple[float, bool]:
    """Relative bbox aspect shock, and whether the stability gate passed.

    A shape measurement, not a motion one, so it keeps working once the vehicle
    has stopped. A box whose aspect was already thrashing (partial occlusion,
    detector jitter) cannot testify to a sudden rotation, which is what the
    stability gate rejects.
    """
    if len(pre) < 3 or not post:
        return 0.0, False
    pa = [s.aspect for s in pre]
    base = float(np.median(pa))
    if base <= 1e-3:
        return 0.0, False
    if float(np.std(pa)) / base > cfg.stable_aspect_rel_std:
        return 0.0, False
    return max(abs(s.aspect - base) / base for s in post), True


def _running_drop(values: Sequence[float]) -> float:
    """Largest fall from a running peak -- how much was shed, not the range."""
    drop = peak = 0.0
    for v in values:
        peak = max(peak, v)
        drop = max(drop, peak - v)
    return drop


def impact_evidence(win: TrackWindow, t_contact: float,
                    cfg: RotationGateConfig | None = None,
                    appearance: float | None = None) -> ImpactEvidence:
    """Rotation-gated impact evidence for ONE vehicle at ONE instant.

    ``appearance`` is the crash-classifier's attribution for this vehicle, or
    ``None`` when no classifier ran. ``None`` removes the channel from both
    sides of the weighted mean; ``0.0`` asserts the classifier looked and saw
    nothing. They are different claims and are scored differently.
    """
    cfg = cfg or DEFAULT_CONFIG
    win_samples = win.window(t_contact, cfg.contact_window_s)

    # -- track break -------------------------------------------------------
    # Specifically TERMINATION, and only for a track established beforehand.
    # An earlier version also counted tracks that STARTED near contact and it
    # fired 1.0 on every vehicle in the scene, including one that began at
    # t=6.32 and then ran for another 24 seconds -- a tracker pick-up, not a
    # vehicle being destroyed.
    was_moving = win.max_speed_kmh >= cfg.min_speed_for_heading_kmh
    established = (t_contact - win.first_t) >= cfg.track_break_min_life_s
    ends_here = abs(win.last_t - t_contact) <= cfg.track_break_window_s
    track_break = 1.0 if (win.ended and was_moving and established and ends_here) else 0.0

    if len(win_samples) < 4:
        # No 'after' to measure rotation in. That absence is itself evidence --
        # the tracker lost it because the vehicle deformed and rotated -- so
        # track_break substitutes for the rotation it could not observe.
        # Otherwise the true participant that gets destroyed scores lowest of
        # all, which is backwards.
        rot = cfg.break_implies_rotation * track_break
        return ImpactEvidence(
            track_id=win.track_id, rotation=rot, gate=rotation_gate(rot, cfg),
            yaw_deg=0.0, aspect=0.0, decel=0.0, speed_drop=0.0,
            momentum_drop=0.0, track_break=track_break, appearance=appearance,
            kinematic=0.0, score=rot * cfg.break_floor_score,
            n_window=len(win_samples), channels_used=("track_break",),
            note=("too few samples around contact to measure rotation; "
                  "track-break substitutes for it" if track_break
                  else "too few samples around contact and no track break"))

    pre = [s for s in win_samples if s.t < t_contact]
    post = [s for s in win_samples if s.t >= t_contact]

    yaw, yaw_ok = _yaw_shock(pre, post, cfg)
    aspect, aspect_ok = _aspect_shock(pre, post, cfg)

    decel = max(0.0, -min(s.accel_mps2 for s in win_samples))
    speed_drop = _running_drop([s.speed_kmh for s in win_samples])
    mom_drop = _running_drop([s.momentum_kgms for s in win_samples])

    yaw_n = saturate(yaw, cfg.ref_yaw_deg)
    asp_n = saturate(aspect, cfg.ref_aspect)
    # The gate. Either measurement counts: heading works while moving,
    # silhouette works once stopped, a dead track stands in for both.
    rot = max(yaw_n, asp_n, cfg.break_implies_rotation * track_break)

    dec_n = saturate(decel, cfg.ref_decel)
    drop_n = saturate(speed_drop, cfg.ref_speed_drop_kmh)
    mom_n = saturate(mom_drop, cfg.ref_momentum)

    channels: dict[str, float] = {
        "decel": dec_n,
        "speed_drop": drop_n,
        "momentum": mom_n,
        "track_break": track_break,
    }
    if appearance is not None:
        channels["appearance"] = float(min(1.0, max(0.0, appearance)))

    weights = cfg.kinematic_weights
    denom = sum(weights[c] for c in channels)
    kinematic = (sum(weights[c] * v for c, v in channels.items())
                 / max(1e-9, denom))

    gate = rotation_gate(rot, cfg)
    score = (cfg.w_rotation * rot + (1.0 - cfg.w_rotation) * kinematic) * gate

    if rot <= 1e-9:
        note = ("no rotation measured: decelerating along its own axis is a "
                "driver braking, not a vehicle being struck")
    elif track_break and rot == cfg.break_implies_rotation * track_break:
        note = "rotation inferred from the track terminating at contact"
    elif yaw_n >= asp_n:
        note = f"heading shock of {yaw:.0f} deg after a stable approach"
    else:
        note = f"silhouette changed by {aspect * 100:.0f}% after a stable approach"

    return ImpactEvidence(
        track_id=win.track_id, rotation=rot, gate=gate, yaw_deg=yaw,
        aspect=aspect, decel=dec_n, speed_drop=drop_n, momentum_drop=mom_n,
        track_break=track_break,
        appearance=None if appearance is None else float(min(1.0, appearance)),
        kinematic=kinematic, score=score, n_window=len(win_samples),
        channels_used=tuple(sorted(channels)), yaw_gate_passed=yaw_ok,
        aspect_gate_passed=aspect_ok, note=note)


# ---------------------------------------------------------------------------
# interaction geometry
# ---------------------------------------------------------------------------

def relative_heading(a: TrackWindow, b: TrackWindow, t_contact: float,
                     cfg: RotationGateConfig | None = None) -> float | None:
    """Unsigned angle between the two approach headings, 0..180 degrees.

    Measured over ``heading_lookback_s`` *before* contact, because after it the
    headings are exactly what the impact scrambled. Returns ``None`` when either
    vehicle was too slow or too briefly observed for its direction of travel to
    mean anything -- absence of evidence, reported as such.
    """
    cfg = cfg or DEFAULT_CONFIG
    lookback, ms = cfg.heading_lookback_s, cfg.min_speed_for_heading_kmh

    def approach(win: TrackWindow) -> float | None:
        ss = [s for s in win.samples
              if t_contact - lookback <= s.t < t_contact and s.speed_kmh >= ms]
        if len(ss) < 3:
            return None
        return circular_mean_deg([s.heading_deg for s in ss])

    ha, hb = approach(a), approach(b)
    if ha is None or hb is None:
        return None
    return abs(ang_diff(ha, hb))


def interaction_prior(rel_deg: float | None,
                      cfg: RotationGateConfig | None = None) -> tuple[float, str]:
    """How much to trust a contact given the two vehicles' relative heading.

    A prior on *how much evidence to demand*, never a veto -- real rear-end and
    head-on collisions exist and stay reachable when the rotation evidence is
    there.

    The ``oncoming`` threshold is 160, not 135. At 135 video 11's *real* 138 deg
    collision was classified as passing traffic and demoted. Two vehicles
    genuinely passing in opposing lanes are near-exactly antiparallel -- video
    13's false positive measured 178.4 deg -- while 138 deg is a converging
    angled impact.
    """
    cfg = cfg or DEFAULT_CONFIG
    if rel_deg is None:
        # The heading could NOT be measured (too slow, track too short). That is
        # absence of evidence, which should rank below verified crossing rather
        # than nearly matching it.
        return cfg.prior_unknown, UNKNOWN
    if rel_deg <= cfg.following_max_deg:
        return cfg.prior_following, FOLLOWING
    if rel_deg >= cfg.oncoming_min_deg:
        return cfg.prior_oncoming, ONCOMING
    return cfg.prior_crossing, CROSSING


# ---------------------------------------------------------------------------
# pair scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairResult:
    """One candidate collision pair with the full evidence breakdown."""

    a: int
    b: int
    score: float
    contact: float
    gap: float
    contact_t: float
    contact_refined: bool
    interaction: str
    rel_heading_deg: float | None
    prior: float
    evidence_a: ImpactEvidence
    evidence_b: ImpactEvidence
    geometric_mean: float

    @property
    def track_ids(self) -> tuple[int, int]:
        return (self.a, self.b)

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_ids": [self.a, self.b],
            "score": round(self.score, 4),
            "contact": round(self.contact, 4),
            "gap": round(self.gap, 4),
            "contact_t": round(self.contact_t, 3),
            "contact_refined": self.contact_refined,
            "interaction": self.interaction,
            "relative_heading_deg": (None if self.rel_heading_deg is None
                                     else round(self.rel_heading_deg, 1)),
            "interaction_prior": round(self.prior, 3),
            "geometric_mean_evidence": round(self.geometric_mean, 4),
            "evidence_a": self.evidence_a.as_dict(),
            "evidence_b": self.evidence_b.as_dict(),
        }

    def explain(self) -> str:
        """One sentence an operator can read. No number is invented here."""
        rel = ("relative heading unmeasurable" if self.rel_heading_deg is None
               else f"{self.rel_heading_deg:.0f} deg relative heading")
        touch = ("boxes overlapping" if self.gap <= 0
                 else f"{self.gap:.2f} vehicle-widths apart")
        return (f"#{self.a} and #{self.b} came closest at t={self.contact_t:.2f}s "
                f"({touch}, {rel} -> {self.interaction}); "
                f"#{self.a}: {self.evidence_a.note}; "
                f"#{self.b}: {self.evidence_b.note}; score {self.score:.3f}")


def score_pair(a: TrackWindow, b: TrackWindow,
               cfg: RotationGateConfig | None = None,
               *, appearance_a: float | None = None,
               appearance_b: float | None = None) -> PairResult | None:
    """Score one candidate pair. Returns ``None`` when it is not a candidate.

    ``score = geometric_mean(evidence_a, evidence_b) * contact * prior``

    The mean is GEOMETRIC on purpose: **both** vehicles must look struck. An
    arithmetic mean let one violently anomalous vehicle drag a placid bystander
    into the top pair, which is how video 14 paired #1805 with #1902 instead of
    #1896 with #1902.
    """
    cfg = cfg or DEFAULT_CONFIG
    contact = find_contact(a, b, cfg)
    if contact is None:
        return None                       # never on screen together
    if contact.gap > cfg.pair_max_gap:
        return None                       # never came close: not a candidate

    strength = contact_strength(contact.gap, cfg)
    ev_a = impact_evidence(a, contact.t, cfg, appearance_a)
    ev_b = impact_evidence(b, contact.t, cfg, appearance_b)

    rel = relative_heading(a, b, contact.t, cfg)
    prior, kind = interaction_prior(rel, cfg)

    geo = math.sqrt(max(0.0, ev_a.score) * max(0.0, ev_b.score))
    return PairResult(
        a=a.track_id, b=b.track_id, score=geo * strength * prior,
        contact=strength, gap=contact.gap, contact_t=contact.t,
        contact_refined=contact.refined, interaction=kind,
        rel_heading_deg=rel, prior=prior,
        evidence_a=ev_a, evidence_b=ev_b, geometric_mean=geo)


def score_pairs(tracks: Iterable[Any], cfg: RotationGateConfig | None = None,
                *, now_t: float | None = None,
                appearance: Mapping[int, float] | None = None,
                classes: Iterable[int] | None = None,
                names: Mapping[int, str] | None = None,
                top_k: int | None = None) -> list[PairResult]:
    """Rank every candidate collision pair among ``tracks``, best first.

    The main entry point. Takes live NETRA :class:`netra.track.Track` objects
    and mutates none of them; performs no I/O.

    Pairs are formed over **all** gated vehicles, not a top-K shortlist by
    individual score: on video 13 the true pair ranked 8th and 10th
    individually, so a top-8 shortlist discarded the right answer before
    pairing began.

    Parameters
    ----------
    tracks
        NETRA tracks (anything with ``track_id``, ``cls``, ``box_history``,
        ``first_t``, ``last_t``, and optionally ``state``).
    cfg
        :class:`RotationGateConfig`; the COMBINED defaults if omitted.
    now_t
        Current pipeline time. Required for ``track_break`` to mean anything in
        a live system -- omit it and the break channel simply stays at 0 for
        tracks the tracker has not explicitly marked lost.
    appearance
        Optional ``{track_id: 0..1}`` crash-classifier attribution. A track
        absent from the mapping has its appearance channel dropped from the
        weighted mean rather than scored as zero.
    classes
        Optional class-id whitelist, e.g. ``MOTORISED_CLASSES | VULNERABLE_CLASSES``.
    top_k
        Truncate the returned ranking.

    Returns
    -------
    list[PairResult]
        Sorted by ``score`` descending. Empty when nothing came within
        ``pair_max_gap`` -- which is the honest answer, not a failure.
    """
    cfg = cfg or DEFAULT_CONFIG
    windows = build_windows(tracks, cfg, now_t=now_t, classes=classes, names=names)
    if len(windows) < 2:
        return []

    app = dict(appearance or {})
    results: list[PairResult] = []
    considered = 0
    for wa, wb in itertools.combinations(windows, 2):
        if considered >= cfg.max_pairs:
            break
        considered += 1
        res = score_pair(wa, wb, cfg,
                         appearance_a=app.get(wa.track_id),
                         appearance_b=app.get(wb.track_id))
        if res is not None:
            results.append(res)

    results.sort(key=lambda r: r.score, reverse=True)
    return results if top_k is None else results[:top_k]
