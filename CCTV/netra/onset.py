"""Recovering when an incident actually started.

This module exists because of one number in the evidence review. On the AI City
2021 Track 4 leaderboard, four systems all *found* roughly the same anomalies --
their F1 scores span 0.857 to 0.952 -- but their final ranking was decided
almost entirely by timing error, which spans **3.40 s to 101.0 s**. Aboah et al.
detect 86% of anomalies and still place fifth, because they are on average
nearly two minutes late.

The reason is structural. Background modelling only reveals a vehicle *after*
it has been stationary long enough to melt into the background, and a crashed
vehicle keeps moving for some seconds after impact. Zhao et al. document
examples where the vehicle comes to rest 9 s and 20 s after the collision.

So the moment you detect is not the moment that matters. You have to walk back.

The method here follows Chen et al. (CVPRW 2021), who achieved the best timing
error in that competition:

1. seed feature points inside the stopped object's box
2. track them **backwards** in time with sparse Lucas-Kanade optical flow
3. reject drifted points with a kNN consistency filter
4. find where the point cloud's speed departs sharply from its recent band --
   that departure is the impact or the start of the manoeuvre
5. confirm with a vote across PSNR, SSIM and normalised Euclidean distance
   between the patch then and now

All of it is classical CV on a handful of points, triggered once per incident.
It costs milliseconds and buys tens of seconds of accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------
# patch similarity measures (the confirmation vote)
# --------------------------------------------------------------------------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return 0.0
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-9:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM, no external dependency.

    Implemented directly rather than importing scikit-image: it is a dozen
    lines, and it keeps the install to numpy/cv2/scipy.
    """
    if a.shape != b.shape or a.size == 0:
        return 0.0
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    k1, k2, L = 0.01, 0.03, 255.0
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(np.clip(np.mean(num / np.maximum(den, 1e-12)), -1.0, 1.0))


def normalised_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """1 - normalised L2 distance, so higher means more similar."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    d = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64))
    dmax = np.sqrt(a.size) * 255.0
    return float(1.0 - min(1.0, d / max(dmax, 1e-9)))


@dataclass
class OnsetResult:
    onset_t: float
    stop_t: float
    method: str
    recovered_seconds: float
    confidence: float
    detail: dict

    @property
    def improved(self) -> bool:
        return self.recovered_seconds > 0.25


class OnsetRecovery:
    """Backward optical-flow search for the true start of an incident."""

    def __init__(self,
                 max_backtrack_s: float = 15.0,
                 max_points: int = 60,
                 knn_k: int = 6,
                 band_scale: float = 2.5,
                 psnr_thresh: float = 13.0,
                 ssim_thresh: float = 0.40,
                 euclid_thresh: float = 0.70) -> None:
        self.max_backtrack_s = max_backtrack_s
        self.max_points = max_points
        self.knn_k = knn_k
        self.band_scale = band_scale
        self.psnr_thresh = psnr_thresh
        self.ssim_thresh = ssim_thresh
        self.euclid_thresh = euclid_thresh

    # -- main entry --------------------------------------------------------
    def recover(self,
                frames: list[tuple[float, np.ndarray]],
                box: tuple[float, float, float, float],
                stop_t: float) -> OnsetResult:
        """Walk backwards from ``stop_t`` looking for the motion discontinuity.

        ``frames`` is the evidence ring buffer as ``(t, frame)``, oldest first.
        ``box`` is where the object came to rest.
        """
        detail: dict = {}
        window = [(t, f) for t, f in frames
                  if stop_t - self.max_backtrack_s <= t <= stop_t]
        if len(window) < 6:
            return OnsetResult(stop_t, stop_t, "insufficient-buffer", 0.0, 0.0,
                               {"frames_available": len(window)})

        window.sort(key=lambda x: x[0], reverse=True)     # newest first
        t_series, speeds = self._backward_flow_speeds(window, box)
        detail["samples"] = len(speeds)

        if len(speeds) < 6:
            return OnsetResult(stop_t, stop_t, "insufficient-flow", 0.0, 0.0, detail)

        idx = self._departure_index(speeds)
        if idx is None:
            return OnsetResult(stop_t, stop_t, "no-departure", 0.0, 0.0, detail)

        onset_t = float(t_series[idx])
        recovered = max(0.0, stop_t - onset_t)

        vote, votes = self._confirm(window, box, onset_t, stop_t)
        detail["votes"] = votes
        detail["speed_profile"] = [round(float(s), 3) for s in speeds[:40]]

        if vote < 2:
            # the appearance evidence disagrees with the motion evidence; keep
            # the finding but report it as low confidence rather than silently
            # trusting one signal
            return OnsetResult(onset_t, stop_t, "flow-unconfirmed", recovered,
                               0.35, detail)

        conf = float(np.clip(0.5 + 0.15 * vote, 0.0, 0.95))
        return OnsetResult(onset_t, stop_t, "flow-confirmed", recovered, conf, detail)

    # -- steps -------------------------------------------------------------
    def _backward_flow_speeds(self, window, box):
        """Sparse LK backwards from the resting box; returns (times, speeds)."""
        t0, f0 = window[0]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = f0.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return [], []

        prev_gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(prev_gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=self.max_points,
                                      qualityLevel=0.01, minDistance=4, mask=mask)
        if pts is None or len(pts) < 5:
            return [], []

        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        times, speeds = [], []
        cur_pts = pts
        for i in range(1, len(window)):
            t_i, f_i = window[i]
            dt = abs(window[i - 1][0] - t_i)
            if dt <= 1e-6:
                continue
            gray = cv2.cvtColor(f_i, cv2.COLOR_BGR2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, cur_pts, None, **lk)
            if nxt is None or st is None:
                break
            good_new = nxt[st.flatten() == 1]
            good_old = cur_pts[st.flatten() == 1]
            if len(good_new) < 4:
                break

            disp = np.linalg.norm(good_new.reshape(-1, 2) - good_old.reshape(-1, 2), axis=1)
            kept = self._knn_filter(good_new.reshape(-1, 2), disp)
            if kept.size == 0:
                break

            times.append(t_i)
            speeds.append(float(np.mean(disp[kept]) / dt))

            prev_gray = gray
            cur_pts = good_new.reshape(-1, 1, 2)

        return times, speeds

    def _knn_filter(self, points: np.ndarray, disp: np.ndarray) -> np.ndarray:
        """Drop points whose displacement disagrees with their k neighbours.

        A feature that has slid onto the background moves with the camera, not
        the vehicle. Requiring local agreement removes those without needing a
        model of either.
        """
        n = len(points)
        if n <= self.knn_k:
            return np.arange(n)
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, : self.knn_k]
        local = disp[nn].mean(axis=1)
        resid = np.abs(disp - local)
        tol = np.median(resid) + 2.0 * (np.std(resid) + 1e-6)
        return np.where(resid <= tol)[0]

    def _departure_index(self, speeds: list[float]) -> int | None:
        """First index (walking back in time) where speed leaves its recent band.

        Band follows Chen et al.:  2 * (MAE + scale * sigma) about the mean.
        """
        a = np.asarray(speeds, dtype=np.float64)
        if a.size < 6:
            return None
        for i in range(4, a.size):
            ref = a[:i]
            mu = float(ref.mean())
            mae = float(np.mean(np.abs(ref - mu)))
            sd = float(ref.std())
            band = 2.0 * (mae + self.band_scale * sd)
            if abs(a[i] - mu) > max(band, 1.5):
                return i
        return None

    def _confirm(self, window, box, onset_t, stop_t) -> tuple[int, dict]:
        """Three-measure vote comparing the patch at onset against at rest."""
        def patch_at(target_t):
            best = min(window, key=lambda x: abs(x[0] - target_t))
            f = best[1]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            h, w = f.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return None
            return f[y1:y2, x1:x2]

        p_on, p_stop = patch_at(onset_t), patch_at(stop_t)
        if p_on is None or p_stop is None:
            return 0, {}
        if p_on.shape != p_stop.shape:
            p_on = cv2.resize(p_on, (p_stop.shape[1], p_stop.shape[0]))

        v_psnr = psnr(p_on, p_stop)
        v_ssim = ssim(p_on, p_stop)
        v_eucl = normalised_euclidean(p_on, p_stop)

        # the scene *should* differ between onset and rest -- a vehicle arrived,
        # rotated or crumpled. So a vote is cast when similarity is BELOW the
        # threshold, i.e. genuine change occurred.
        votes = {
            "psnr": {"value": round(v_psnr, 2), "threshold": self.psnr_thresh,
                     "changed": v_psnr < self.psnr_thresh},
            "ssim": {"value": round(v_ssim, 3), "threshold": self.ssim_thresh,
                     "changed": v_ssim < self.ssim_thresh},
            "euclidean": {"value": round(v_eucl, 3), "threshold": self.euclid_thresh,
                          "changed": v_eucl < self.euclid_thresh},
        }
        return sum(1 for v in votes.values() if v["changed"]), votes
