"""Cheap, always-on scene signals.

Three ideas taken directly from the literature, none of which needs a second
neural network:

**Dual-window median background** -- Aboah et al., CVPRW 2021.
Estimate the background as the median of a random sample of recent frames, over
*two* window lengths. A vehicle queued at a red light dissolves into a
five-minute median; a genuinely stalled vehicle does not. That difference is
how NETRA separates an ordinary signal queue from a blockage, and it is the
cheapest useful trick in this whole codebase.

**Change-point detection on global motion** -- the OF and OSD baselines from
the ACCIDENT benchmark (Picek et al., CVPRW 2026). Two scalars per frame:
mean dense-optical-flow magnitude, and summed detection-box area. Z-score them,
smooth, and look for a peak. On that benchmark the ensemble of these two beat
both evaluated 7B vision-language models at relaxed temporal tolerance -- for a
few milliseconds of CPU instead of GPU-hours.

**CUSUM sequential detection** -- Doshi & Yilmaz, CVPRW 2021.
A single-shot threshold fires on every gust of wind and exposure change. An
accumulator demands *sustained* evidence, which collapses the false-alarm rate
and gives one tunable knob trading precision against detection delay. Every
NETRA event engine runs its evidence through this rather than an ad-hoc counter.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------
# CUSUM
# --------------------------------------------------------------------------

@dataclass
class Cusum:
    """One-sided cumulative-sum change detector.

        s_t = max(0, s_{t-1} + e_t - beta)      alarm when s_t >= h

    ``beta`` is the nominal drift we expect to see even when nothing is
    happening -- fit it from quiet footage rather than guessing. ``h`` is the
    precision/delay dial: raise it for fewer, later, surer alarms.
    """

    beta: float = 0.15
    h: float = 2.0
    decay: float = 0.0                  # optional leak toward zero per update

    s: float = 0.0
    alarmed: bool = False
    alarm_t: float | None = None
    first_evidence_t: float | None = None
    peak: float = 0.0

    def update(self, evidence: float, t: float) -> bool:
        """Feed one evidence sample; returns True on the *rising edge* of an alarm."""
        if evidence > self.beta and self.first_evidence_t is None:
            self.first_evidence_t = t
        self.s = max(0.0, self.s + (evidence - self.beta))
        if self.decay:
            self.s = max(0.0, self.s - self.decay)
        self.peak = max(self.peak, self.s)

        if self.s <= 0.0:
            # evidence has fully decayed: the episode is over
            self.first_evidence_t = None
            if self.alarmed:
                self.alarmed = False
                self.alarm_t = None

        rising = False
        if not self.alarmed and self.s >= self.h:
            self.alarmed = True
            self.alarm_t = t
            rising = True
        return rising

    @property
    def confidence(self) -> float:
        """How far past threshold we are, squashed to [0, 1] for reporting."""
        if self.h <= 0:
            return 0.0
        return float(np.clip(self.s / (2.0 * self.h), 0.0, 1.0))

    def reset(self) -> None:
        self.s = 0.0
        self.alarmed = False
        self.alarm_t = None
        self.first_evidence_t = None
        self.peak = 0.0


# --------------------------------------------------------------------------
# dual-window median background
# --------------------------------------------------------------------------

class DualWindowBackground:
    """Median-of-random-sample background at two timescales.

    Following Aboah et al.: sample frames within a window, take the median of a
    random subset. Random sampling plus a median suppresses transient junk --
    compression bursts, a lorry filling the frame, a camera auto-exposing --
    far better than a running average, and it costs nothing but memory.

    The *short* window (default 30 s) shows anything currently stationary.
    The *long* window (default 5 min) shows only things stationary for a long
    time. A vehicle present in both is a blockage candidate; a vehicle present
    only in the short one is very likely a signal queue.
    """

    def __init__(self,
                 short_seconds: float = 30.0,
                 long_seconds: float = 300.0,
                 sample_hz: float = 1.0,
                 sample_fraction: float = 0.25,
                 max_samples: int = 60,
                 scale: float = 0.5,
                 seed: int = 0) -> None:
        self.short_seconds = short_seconds
        self.long_seconds = long_seconds
        self.sample_interval = 1.0 / max(sample_hz, 1e-6)
        self.sample_fraction = sample_fraction
        self.max_samples = max_samples
        self.scale = scale
        self._rng = random.Random(seed)

        self._short: deque = deque(maxlen=int(short_seconds * sample_hz) + 2)
        self._long: deque = deque(maxlen=int(long_seconds * sample_hz) + 2)
        self._last_sample_t = -1e9

        self.short_bg: np.ndarray | None = None
        self.long_bg: np.ndarray | None = None
        self._last_build_t = -1e9

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        return frame

    def update(self, frame: np.ndarray, t: float, rebuild_every: float = 5.0) -> None:
        if t - self._last_sample_t < self.sample_interval:
            return
        self._last_sample_t = t
        small = self._prep(frame)
        self._short.append(small)
        self._long.append(small)

        if t - self._last_build_t >= rebuild_every:
            self._last_build_t = t
            self.short_bg = self._median(self._short)
            self.long_bg = self._median(self._long)

    def _median(self, buf: deque) -> np.ndarray | None:
        n = len(buf)
        if n < 5:
            return None
        k = max(5, min(self.max_samples, int(n * self.sample_fraction)))
        idx = self._rng.sample(range(n), k) if k < n else list(range(n))
        stack = np.stack([buf[i] for i in idx], axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    def persistence(self, box: tuple[float, float, float, float]) -> dict:
        """How strongly a box's contents appear in each background.

        Returns similarity in [0, 1] against short and long backgrounds. A high
        ``long`` score means the object has been sitting there long enough to
        become part of the scene -- i.e. it is not merely waiting for a light.
        """
        out = {"short": 0.0, "long": 0.0, "ready": False}
        if self.short_bg is None:
            return out
        s = self.scale
        x1, y1, x2, y2 = [int(round(v * s)) for v in box]
        h, w = self.short_bg.shape[:2]
        x1, x2 = max(0, min(w - 1, x1)), max(1, min(w, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(1, min(h, y2))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return out

        def edge_energy(img):
            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                return 0.0
            g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
            return float(cv2.Laplacian(g, cv2.CV_32F).var())

        out["short"] = edge_energy(self.short_bg)
        if self.long_bg is not None:
            out["long"] = edge_energy(self.long_bg)
            out["ready"] = True
        return out

    def is_persistent(self, box, ratio: float = 0.55) -> bool:
        """True when the object is visible in the *long* background too."""
        p = self.persistence(box)
        if not p["ready"] or p["short"] <= 1e-6:
            return False
        return (p["long"] / max(p["short"], 1e-6)) >= ratio


# --------------------------------------------------------------------------
# change-point detection on global motion
# --------------------------------------------------------------------------

@dataclass
class ChangePointDetector:
    """Z-scored peak detection over two global scene signals.

    Channel 1 -- ``M_t``: mean dense optical-flow magnitude (ACCIDENT "OF").
    Channel 2 -- ``A_t``: summed detection-box area (ACCIDENT "OSD").

    A collision produces a simultaneous excursion in both: violent local motion,
    and a sudden change in the apparent size/number of tracked objects as
    vehicles rotate, stop and cluster. Requiring agreement across two cheap
    channels is what keeps swaying vegetation and exposure changes out.

    Critically, this fires *without needing both vehicles to be tracked through
    the impact* -- which is exactly the moment tracking is most likely to fail.
    """

    window: int = 5                    # rolling-mean smoothing, samples
    z_threshold: float = 1.5
    history: int = 240                 # ~30 s at 8 Hz
    flow_scale: float = 0.25           # downscale before computing flow

    _flow_hist: deque = field(default_factory=lambda: deque(maxlen=240))
    _mag_hist: deque = field(default_factory=lambda: deque(maxlen=24))
    _area_hist: deque = field(default_factory=lambda: deque(maxlen=240))
    _t_hist: deque = field(default_factory=lambda: deque(maxlen=240))
    _prev_gray: np.ndarray | None = field(default=None, repr=False)

    last_flow: float = 0.0
    last_area: float = 0.0
    last_z_flow: float = 0.0
    last_z_area: float = 0.0
    last_score: float = 0.0

    def _small_gray(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.flow_scale != 1.0:
            g = cv2.resize(g, None, fx=self.flow_scale, fy=self.flow_scale,
                           interpolation=cv2.INTER_AREA)
        return g

    def update(self, frame: np.ndarray, boxes, t: float) -> float:
        """Push one sample, return the combined anomaly score in z units."""
        gray = self._small_gray(frame)

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            # Farneback measured 51 ms per frame at the previous working
            # size. The signal wanted here is where global motion energy
            # concentrates, which survives a coarser field intact; the
            # parameters are reduced together so the flow stays smooth rather
            # than merely cheaper.
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=13,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            self.last_flow = float(mag.mean())
            # Keep the magnitude field, not just its mean. The mean answers
            # "did something violent happen"; the field answers "where" -- and
            # without the second answer we cannot put a box around the vehicles
            # that collided, only announce that a collision occurred somewhere.
            self._mag_hist.append(mag.astype(np.float32))
        else:
            self.last_flow = 0.0
        self._prev_gray = gray

        area = 0.0
        for b in boxes:
            area += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        self.last_area = float(area)

        self._flow_hist.append(self.last_flow)
        self._area_hist.append(self.last_area)
        self._t_hist.append(t)

        self.last_z_flow = self._z(self._flow_hist)
        self.last_z_area = self._z(self._area_hist)

        # both channels must agree, and we care about *sudden* change in either
        # direction for area (vehicles clustering or vanishing), but only about
        # excess motion for flow
        self.last_score = min(self.last_z_flow, abs(self.last_z_area))
        return self.last_score

    def _z(self, hist: deque) -> float:
        n = len(hist)
        if n < max(12, self.window * 2):
            return 0.0
        a = np.asarray(hist, dtype=np.float64)
        k = min(self.window, n)
        smoothed = np.convolve(a, np.ones(k) / k, mode="valid")
        if smoothed.size < 6:
            return 0.0
        ref = smoothed[:-1]
        mu, sd = float(ref.mean()), float(ref.std())
        if sd < 1e-8:
            return 0.0
        return float((smoothed[-1] - mu) / sd)

    @property
    def triggered(self) -> bool:
        return self.last_score >= self.z_threshold

    def impact_point(self, frame_shape, accumulate: int = 12,
                     percentile: float = 90.0):
        """Where the violent motion was, in full-frame pixel coordinates.

        This is the spatial-localisation baseline from the 2026 ACCIDENT
        benchmark: accumulate dense optical-flow magnitude over a short window,
        keep only the top decile so ordinary traffic motion drops out, and take
        the intensity-weighted centroid of what remains.

        On that benchmark this heuristic scores 0.273 against a human ceiling of
        0.995 -- weak on its own. But it is not being asked to work on its own
        here: it only has to point close enough to the impact for the tracker to
        say *which vehicles* were there, and a track identity is a far more
        robust output than a raw coordinate.

        Returns ``(x, y, confidence)`` or ``None``.
        """
        if not self._mag_hist:
            return None
        window = list(self._mag_hist)[-max(1, accumulate):]
        acc = np.sum(np.stack(window, axis=0), axis=0)
        if not np.isfinite(acc).all() or acc.max() <= 1e-6:
            return None

        thresh = float(np.percentile(acc, percentile))
        mask = acc >= thresh
        if mask.sum() < 4:
            return None
        weights = np.where(mask, acc, 0.0)
        total = float(weights.sum())
        if total <= 1e-6:
            return None

        ys, xs = np.nonzero(mask)
        w = weights[ys, xs]
        cx = float((xs * w).sum() / total)
        cy = float((ys * w).sum() / total)

        # how concentrated the hot region is: a tight blob is a real impact,
        # a diffuse one is the whole scene moving
        spread = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean()))
        diag = float(np.hypot(*acc.shape))
        conf = float(np.clip(1.0 - (spread / max(diag * 0.35, 1e-6)), 0.0, 1.0))

        H, W = frame_shape[:2]
        sy = H / acc.shape[0]
        sx = W / acc.shape[1]
        return (cx * sx, cy * sy, conf)

    def flow_energy_in_boxes(self, boxes, frame_shape, accumulate: int = 12) -> list[float]:
        """Accumulated optical-flow energy inside each box, per unit area.

        This is the attribution step, and it is deliberately object-centric.
        A raw flow centroid answers "where were the hottest pixels", which at
        night is often a headlight flare or a swaying tree. Asking instead
        "which *tracked vehicle* contains anomalous motion" grounds the answer
        in an object the tracker is already following -- so the box that gets
        drawn stays locked to the right car as it moves, and a reviewer can
        check the claim frame by frame.

        Returns one value per box, normalised by box area so a lorry does not
        outrank a motorcycle purely by being large.
        """
        if not len(boxes) or not self._mag_hist:
            return [0.0] * len(boxes)
        window = list(self._mag_hist)[-max(1, accumulate):]
        acc = np.sum(np.stack(window, axis=0), axis=0)
        H, W = frame_shape[:2]
        sy = acc.shape[0] / max(H, 1)
        sx = acc.shape[1] / max(W, 1)

        out = []
        for b in boxes:
            x1 = int(np.clip(b[0] * sx, 0, acc.shape[1] - 1))
            x2 = int(np.clip(b[2] * sx, 1, acc.shape[1]))
            y1 = int(np.clip(b[1] * sy, 0, acc.shape[0] - 1))
            y2 = int(np.clip(b[3] * sy, 1, acc.shape[0]))
            if x2 <= x1 or y2 <= y1:
                out.append(0.0)
                continue
            patch = acc[y1:y2, x1:x2]
            out.append(float(patch.mean()))
        return out

    def snapshot(self) -> dict:
        return {
            "flow": round(self.last_flow, 4),
            "box_area": round(self.last_area, 1),
            "z_flow": round(self.last_z_flow, 3),
            "z_area": round(self.last_z_area, 3),
            "score": round(self.last_score, 3),
            "threshold": self.z_threshold,
        }


# --------------------------------------------------------------------------
# camera health
# --------------------------------------------------------------------------

class CameraHealth:
    """Detects the two conditions that silently invalidate every geometric event.

    * **Corrupted / black frames.** Doshi & Yilmaz filter these by frame mean;
      the AI City test set is full of them and they generate phantom alarms.
    * **Camera movement.** All of NETRA's geometry is expressed in image
      coordinates tied to a hand-drawn corridor. If the camera pans or is
      knocked, those polygons now describe the wrong piece of road. Detecting it
      and *refusing to emit geometric events* is far better than confidently
      reporting nonsense.
    """

    def __init__(self, shift_tolerance: float = 12.0, scale: float = 0.25) -> None:
        self.shift_tolerance = shift_tolerance
        self.scale = scale
        self._ref: np.ndarray | None = None
        self.shift_px = 0.0
        self.geometry_valid = True
        self.last_reason = ""

    def _prep(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_AREA)

    @staticmethod
    def is_corrupt(frame: np.ndarray, stride: int = 8) -> bool:
        """Blank / corrupt frame test, on a strided subsample.

        Measured at 65 ms per frame on 4K when computed over every pixel --
        more than half the cost of the detector, to answer a question that a
        few thousand samples settle just as well. A frame that is uniformly
        black or blown out is uniform everywhere, so sampling every eighth
        pixel in each axis gives the same verdict for 1/64th of the work.
        """
        if frame is None or frame.size == 0:
            return True
        sub = frame[::stride, ::stride]
        m = float(sub.mean())
        if m < 3.0 or m > 252.0:
            return True
        return float(sub.std()) < 3.0

    def update(self, frame: np.ndarray) -> dict:
        if self.is_corrupt(frame):
            self.geometry_valid = False
            self.last_reason = "corrupt or blank frame"
            return self.snapshot()

        small = self._prep(frame)
        if self._ref is None:
            self._ref = small
            self.geometry_valid = True
            self.last_reason = ""
            return self.snapshot()

        try:
            shift, _ = cv2.phaseCorrelate(self._ref.astype(np.float32),
                                          small.astype(np.float32))
            self.shift_px = float(np.hypot(shift[0], shift[1]) / max(self.scale, 1e-6))
        except cv2.error:
            self.shift_px = 0.0

        if self.shift_px > self.shift_tolerance:
            self.geometry_valid = False
            self.last_reason = f"camera moved ~{self.shift_px:.0f}px; recalibration required"
        else:
            self.geometry_valid = True
            self.last_reason = ""
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "geometry_valid": self.geometry_valid,
            "shift_px": round(self.shift_px, 2),
            "reason": self.last_reason,
        }
