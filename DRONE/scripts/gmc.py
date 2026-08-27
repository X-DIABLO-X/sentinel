"""Global Motion Compensation (GMC) — background-homography ego-motion removal.

This is the core module of the drone subsystem. Everything downstream that
claims to measure motion depends on it being right.

The problem
-----------
A vehicle's speed from a *fixed* camera is a pixel displacement over time. From
a *moving* camera it is not: the observed displacement is the sum of the
vehicle's real motion and the camera's own motion, and those two are not
separable from a single frame pair without a model.

Even in hover the drone moves. Wind, GPS position-hold error and gimbal
micro-corrections produce continuous small drift. A parked car under a drifting
drone appears to move, and a queue of stopped vehicles looks like slow-moving
traffic. Uncompensated, every downstream measurement — speed, queue length,
stationarity, blockage duration — is wrong in a way that is not visible in the
output.

The method
----------
Chained background homography, the standard solution:

1. Detect vehicles in the frame.
2. **Mask those boxes out**, so features are only taken from the static
   background — road markings, kerbs, building corners, poles.
3. Detect and match features (ORB by default) between the previous frame's
   background and this frame's background.
4. Estimate a frame-to-frame homography with RANSAC.
5. Chain that homography back to a reference frame, so any frame's pixels can
   be mapped into one common coordinate system.

Vehicle positions then pass through *both* the ego-motion homography (drone
motion removed) and, when calibrated, the road-plane homography (pixels to
metres).

Why masking the vehicles is not optional: features on a moving vehicle are
"background" as far as RANSAC is concerned. In dense traffic, vehicles can be
the *majority* of the trackable texture, and RANSAC will happily fit the
dominant motion — which is then the traffic, not the drone. The homography
comes out confidently wrong, and every vehicle is measured relative to the
traffic stream instead of the ground.

Reference pattern: BoT-SORT's CMC (camera motion compensation) option; and
github.com/Thamkench/uav-speedlab (YOLOv11 + BoT-SORT + homography GMC).

Accuracy expectations — do not overstate these
----------------------------------------------
Published homography-based UAV speed estimation reports RMSE roughly
**0.53–16 km/h** depending on altitude, resolution, and calibration quality;
calibration-free monocular approaches report **9.7–15% MAE**. That is the band
this approach lives in. Nothing in this repository has been measured against
ground truth, because no drone footage exists yet. When it does, the number
that goes in the submission is the measured one, whatever it turns out to be.

Telemetry-assisted direct georeferencing (per-frame drone GPS/IMU/gimbal pose)
is the stronger alternative where available, because it does not accumulate
chain drift. GPS alone at ~10 Hz is too coarse to interpolate per frame and IMU
alone drifts; real systems fuse both. See ``telemetry_ingest.py`` — not
implemented yet, signature frozen.

Failure policy
--------------
When an estimate cannot be trusted — too few matches, too few RANSAC inliers,
a degenerate or physically implausible transform — this module returns the
**identity matrix with ``ok=False``**. It never returns a low-confidence
homography dressed up as a good one. The caller is expected to check ``ok`` and
skip measurement for that frame. Silently returning garbage here would corrupt
speeds without any visible symptom, which is the worst possible failure mode
for this system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "GMCResult",
    "GMCEstimator",
    "IDENTITY",
    "build_dynamic_mask",
    "estimate_frame_homography",
    "chain_homographies",
    "apply_gmc",
    "compensate_track_positions",
    "invert_homography",
    "decompose_translation_scale",
]

IDENTITY = np.eye(3, dtype=np.float64)


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------

@dataclass
class GMCResult:
    """Outcome of one frame-to-frame estimate.

    ``H`` maps **previous-frame pixels to current-frame pixels**. It is the
    identity whenever ``ok`` is False, and ``reason`` says why.
    """
    H: np.ndarray = field(default_factory=lambda: IDENTITY.copy())
    n_inliers: int = 0
    n_matches: int = 0
    n_keypoints_prev: int = 0
    n_keypoints_cur: int = 0
    ok: bool = False
    reason: str = "not_run"

    @property
    def inlier_ratio(self) -> float:
        return (self.n_inliers / self.n_matches) if self.n_matches else 0.0

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "reason": self.reason,
            "n_matches": int(self.n_matches),
            "n_inliers": int(self.n_inliers),
            "inlier_ratio": round(self.inlier_ratio, 4),
            "n_keypoints_prev": int(self.n_keypoints_prev),
            "n_keypoints_cur": int(self.n_keypoints_cur),
        }


# --------------------------------------------------------------------------
# Feature backends
# --------------------------------------------------------------------------

def _make_detector(feature_type: str, max_features: int):
    """Build the keypoint detector. ORB by default.

    ORB is chosen over SIFT as the default for a practical reason rather than
    an accuracy one: it is binary-descriptor, patent-free, and roughly an order
    of magnitude cheaper, and the frame-to-frame baseline in hover is tiny, so
    the extra invariance SIFT buys is not needed. SIFT stays selectable for
    difficult scenes (low texture, strong illumination change) where ORB's
    corner response collapses.
    """
    ft = (feature_type or "orb").lower()
    if ft == "orb":
        return cv2.ORB_create(nfeatures=int(max_features)), cv2.NORM_HAMMING
    if ft == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if ft == "sift":
        create = getattr(cv2, "SIFT_create", None)
        if create is None:      # very old opencv builds
            raise RuntimeError(
                "SIFT is not available in this OpenCV build; use feature_type: orb"
            )
        return create(nfeatures=int(max_features)), cv2.NORM_L2
    raise ValueError(f"unknown gmc.feature_type {feature_type!r} (orb|sift|akaze)")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"cannot convert frame of shape {frame.shape} to grayscale")


# --------------------------------------------------------------------------
# Dynamic masking
# --------------------------------------------------------------------------

def build_dynamic_mask(shape: tuple[int, int],
                       boxes: Iterable[Sequence[float]] | None,
                       dilate_px: int = 12,
                       border_frac: float = 0.02) -> np.ndarray:
    """Mask of pixels that may be used for background feature detection.

    255 = usable static background, 0 = excluded.

    Excluded regions:

    * every detected vehicle box, grown by ``dilate_px``. The dilation matters:
      a tight box clips the vehicle's own edge, and that edge is exactly the
      kind of high-contrast corner a feature detector loves. Those corners move
      with the vehicle, so leaving them in reintroduces the contamination that
      masking exists to remove.
    * a thin border of the frame. Features there fall out of view on the next
      frame and produce one-sided, biased matches.

    ``boxes`` are ``[x1, y1, x2, y2, ...]`` in pixels; extra columns (score,
    class) are ignored, so a detector's raw (N, 6) array can be passed straight
    in.
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.full((h, w), 255, dtype=np.uint8)

    b = max(1, int(round(min(h, w) * float(border_frac))))
    if b > 0:
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0

    if boxes is None:
        return mask

    d = int(max(0, dilate_px))
    for box in boxes:
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        xa = int(np.clip(np.floor(min(x1, x2)) - d, 0, w))
        ya = int(np.clip(np.floor(min(y1, y2)) - d, 0, h))
        xb = int(np.clip(np.ceil(max(x1, x2)) + d, 0, w))
        yb = int(np.clip(np.ceil(max(y1, y2)) + d, 0, h))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 0

    return mask


def mask_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame still usable as background. Diagnostic."""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


# --------------------------------------------------------------------------
# Homography sanity
# --------------------------------------------------------------------------

def decompose_translation_scale(H: np.ndarray) -> tuple[float, float]:
    """Crude (translation magnitude in px, isotropic scale) from a homography.

    Not a full decomposition — just enough to reject an obviously broken
    estimate. Scale comes from sqrt(|det|) of the affine 2x2 block.
    """
    A = np.asarray(H, dtype=np.float64)[:2, :2]
    t = np.asarray(H, dtype=np.float64)[:2, 2]
    det = float(abs(np.linalg.det(A)))
    scale = float(np.sqrt(det)) if det > 0 else 0.0
    return float(np.hypot(t[0], t[1])), scale


def _is_plausible(H: np.ndarray,
                  frame_shape: tuple[int, int],
                  max_scale_change: float,
                  max_translation_frac: float) -> tuple[bool, str]:
    """Reject physically impossible frame-to-frame transforms.

    A drone in hover cannot, between two adjacent frames, double the scale of
    the scene or translate a quarter of the frame. If RANSAC produced such a
    transform it locked onto the wrong thing — most often a large vehicle that
    slipped past the mask, or a repeating texture (lane markings, zebra
    crossings, tiled paving) that matched to the wrong period.
    """
    if H is None:
        return False, "no_homography"
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False, "non_finite_homography"

    if abs(H[2, 2]) < 1e-9:
        return False, "degenerate_h22"

    det = float(np.linalg.det(H))
    if not np.isfinite(det) or abs(det) < 1e-9:
        return False, "singular_homography"

    tmag, scale = decompose_translation_scale(H)
    if scale <= 0:
        return False, "degenerate_scale"

    s = float(max_scale_change)
    if s > 1.0 and not (1.0 / s <= scale <= s):
        return False, f"implausible_scale_{scale:.3f}"

    diag = float(np.hypot(frame_shape[0], frame_shape[1]))
    if max_translation_frac > 0 and tmag > max_translation_frac * diag:
        return False, f"implausible_translation_{tmag:.1f}px"

    return True, "ok"


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------

def estimate_frame_homography(prev_gray: np.ndarray,
                              cur_gray: np.ndarray,
                              vehicle_boxes: Iterable[Sequence[float]] | None = None,
                              *,
                              feature_type: str = "orb",
                              max_features: int = 2000,
                              ransac_reproj_thresh: float = 3.0,
                              min_matches: int = 12,
                              min_inliers: int = 8,
                              mask_dynamic_boxes: bool = True,
                              box_dilate_px: int = 12,
                              ratio_test: float = 0.75,
                              max_scale_change: float = 1.25,
                              max_translation_frac: float = 0.25,
                              prev_boxes: Iterable[Sequence[float]] | None = None,
                              detail: bool = False):
    """Estimate the background homography from ``prev_gray`` to ``cur_gray``.

    Returns ``(H, n_inliers, ok)`` where ``H`` is 3x3 float64 mapping
    **previous-frame pixels to current-frame pixels**.

    Pass ``detail=True`` to get a :class:`GMCResult` instead, which carries the
    match counts and the failure reason.

    On any failure ``H`` is the identity and ``ok`` is False. It never returns
    a low-confidence estimate as though it were good.

    ``vehicle_boxes`` are the detections in the **current** frame.
    ``prev_boxes`` are the previous frame's detections; if omitted, the current
    boxes are reused for the previous frame's mask, which is a good
    approximation at video frame rates and a conservative one (it masks
    slightly more than necessary).
    """
    res = GMCResult()

    if prev_gray is None or cur_gray is None:
        res.reason = "missing_frame"
        return res if detail else (res.H, res.n_inliers, res.ok)

    prev_gray = _to_gray(prev_gray)
    cur_gray = _to_gray(cur_gray)

    if prev_gray.shape != cur_gray.shape:
        res.reason = "frame_shape_mismatch"
        return res if detail else (res.H, res.n_inliers, res.ok)

    h, w = prev_gray.shape[:2]
    if h < 32 or w < 32:
        res.reason = "frame_too_small"
        return res if detail else (res.H, res.n_inliers, res.ok)

    # -- masks ------------------------------------------------------------
    if mask_dynamic_boxes:
        cur_boxes = list(vehicle_boxes) if vehicle_boxes is not None else []
        pb = list(prev_boxes) if prev_boxes is not None else cur_boxes
        mask_prev = build_dynamic_mask((h, w), pb, box_dilate_px)
        mask_cur = build_dynamic_mask((h, w), cur_boxes, box_dilate_px)
    else:
        mask_prev = build_dynamic_mask((h, w), None, 0)
        mask_cur = build_dynamic_mask((h, w), None, 0)

    # If vehicles cover nearly everything there is no static background left to
    # measure against. Say so rather than fitting to whatever slivers remain.
    if mask_coverage(mask_cur) < 0.05 or mask_coverage(mask_prev) < 0.05:
        res.reason = "insufficient_background_area"
        return res if detail else (res.H, res.n_inliers, res.ok)

    # -- features ---------------------------------------------------------
    try:
        det, norm = _make_detector(feature_type, max_features)
    except Exception as exc:                      # noqa: BLE001
        res.reason = f"detector_unavailable:{exc}"
        return res if detail else (res.H, res.n_inliers, res.ok)

    kp1, des1 = det.detectAndCompute(prev_gray, mask_prev)
    kp2, des2 = det.detectAndCompute(cur_gray, mask_cur)
    res.n_keypoints_prev = len(kp1) if kp1 else 0
    res.n_keypoints_cur = len(kp2) if kp2 else 0

    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        res.reason = "too_few_keypoints"
        return res if detail else (res.H, res.n_inliers, res.ok)

    # -- matching ---------------------------------------------------------
    # Lowe's ratio test. A second-nearest neighbour almost as close as the
    # nearest means the descriptor is ambiguous, and road scenes are full of
    # ambiguity: lane dashes, paving, railings and zebra stripes all repeat.
    # Those are exactly the matches that produce a confident wrong homography.
    try:
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        knn = matcher.knnMatch(des1, des2, k=2)
    except cv2.error as exc:
        res.reason = f"match_failed:{exc}"
        return res if detail else (res.H, res.n_inliers, res.ok)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < float(ratio_test) * n.distance:
            good.append(m)

    res.n_matches = len(good)
    if len(good) < int(min_matches):
        res.reason = f"too_few_matches({len(good)}<{min_matches})"
        return res if detail else (res.H, res.n_inliers, res.ok)

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # -- RANSAC -----------------------------------------------------------
    try:
        H, inlier_mask = cv2.findHomography(
            src, dst,
            method=cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_thresh),
            maxIters=2000,
            confidence=0.995,
        )
    except cv2.error as exc:
        res.reason = f"ransac_failed:{exc}"
        return res if detail else (res.H, res.n_inliers, res.ok)

    if H is None or inlier_mask is None:
        res.reason = "ransac_no_model"
        return res if detail else (res.H, res.n_inliers, res.ok)

    n_in = int(inlier_mask.sum())
    res.n_inliers = n_in
    if n_in < int(min_inliers):
        res.reason = f"too_few_inliers({n_in}<{min_inliers})"
        res.H = IDENTITY.copy()
        return res if detail else (res.H, res.n_inliers, res.ok)

    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]                     # normalise the scale ambiguity

    plausible, why = _is_plausible(H, (h, w), max_scale_change, max_translation_frac)
    if not plausible:
        res.reason = why
        res.H = IDENTITY.copy()
        return res if detail else (res.H, res.n_inliers, res.ok)

    res.H = H
    res.ok = True
    res.reason = "ok"
    return res if detail else (res.H, res.n_inliers, res.ok)


# --------------------------------------------------------------------------
# Chaining and application
# --------------------------------------------------------------------------

def invert_homography(H: np.ndarray) -> np.ndarray:
    """Inverse of a homography, falling back to identity if singular."""
    H = np.asarray(H, dtype=np.float64)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return IDENTITY.copy()
    if not np.all(np.isfinite(Hi)):
        return IDENTITY.copy()
    if abs(Hi[2, 2]) > 1e-12:
        Hi = Hi / Hi[2, 2]
    return Hi


def chain_homographies(homographies: Sequence[np.ndarray]) -> np.ndarray:
    """Compose a sequence of frame-to-frame homographies into one transform.

    Given ``[H_1, H_2, ..., H_n]`` where each ``H_i`` maps frame ``i-1`` into
    frame ``i``, the composition mapping frame 0 into frame ``n`` is the matrix
    product in reverse order::

        H_n0 = H_n @ H_{n-1} @ ... @ H_1

    Composition is the whole point of the homography representation, and also
    where the error budget lives: the chain accumulates. Small per-frame errors
    compound over hundreds of frames, which is why the reference frame is reset
    periodically in :class:`GMCEstimator`, and why per-frame telemetry is the
    better long-horizon answer when it exists.
    """
    out = IDENTITY.copy()
    for H in homographies:
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            continue
        out = H @ out
    if abs(out[2, 2]) > 1e-12:
        out = out / out[2, 2]
    return out


def apply_gmc(points: np.ndarray | Sequence[Sequence[float]],
              H: np.ndarray) -> np.ndarray:
    """Map (N, 2) pixel points through homography ``H``.

    Returns (N, 2) float64. Points whose homogeneous w collapses to ~0 (mapped
    to or past the horizon) come back as NaN rather than a huge finite number,
    so a caller cannot mistake a projection failure for a real position.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = pts.reshape(-1, 2)

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return pts.copy()

    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    proj = homo @ H.T
    w = proj[:, 2:3]
    bad = np.abs(w[:, 0]) < 1e-9
    w = np.where(np.abs(w) < 1e-9, 1.0, w)
    out = proj[:, :2] / w
    out[bad] = np.nan
    return out


def apply_gmc_boxes(boxes: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Map (N, 4) xyxy boxes through ``H`` via their four corners.

    The axis-aligned bounding box of the four transformed corners is returned.
    Under rotation this is slightly conservative (larger than the true extent),
    which is the right direction to err for masking and association.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if b.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    out = np.empty_like(b)
    for i, (x1, y1, x2, y2) in enumerate(b):
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        t = apply_gmc(corners, H)
        if np.isnan(t).any():
            out[i] = [np.nan, np.nan, np.nan, np.nan]
            continue
        out[i] = [t[:, 0].min(), t[:, 1].min(), t[:, 0].max(), t[:, 1].max()]
    return out


def compensate_track_positions(positions: np.ndarray | Sequence[Sequence[float]],
                               H_chain: np.ndarray) -> np.ndarray:
    """Map current-frame pixel positions into **reference-frame** coordinates.

    ``H_chain`` is the accumulated transform from the current frame back to the
    reference frame (``GMCEstimator.H_ref_from_cur``). This is the function
    that makes drone-borne kinematics meaningful: after this call, a stationary
    vehicle has a constant position and zero velocity regardless of how the
    drone drifted, and a moving vehicle's displacement is its own.

    Returns (N, 2) float64 in reference-frame pixels. NaN marks a point that
    could not be projected — a caller must treat NaN as "no measurement", never
    as a coordinate.
    """
    return apply_gmc(positions, H_chain)


# --------------------------------------------------------------------------
# Stateful chain
# --------------------------------------------------------------------------

class GMCEstimator:
    """Frame-to-frame GMC with an accumulated transform back to a reference.

    Typical use::

        gmc = GMCEstimator(cfg.gmc)
        for frame, boxes in stream:
            res = gmc.update(frame, boxes)          # res.ok tells you if it worked
            ref_pts = gmc.to_reference(pixel_points)

    Invariants this class maintains:

    * ``H_ref_from_cur`` always maps *current-frame* pixels into the reference
      frame. It starts as the identity on the first frame, which therefore
      **is** the reference frame.
    * On a failed estimate the chain is **not** advanced. Freezing is the least
      wrong option: composing an identity as though the drone had not moved
      would inject a real error into every subsequent frame, whereas freezing
      confines the damage to the frames that failed, which are flagged.
    * After ``max_failure_streak`` consecutive failures the chain is reset and
      the current frame becomes a new reference. Continuing to accumulate on
      top of a chain that has lost lock produces confident nonsense; a reset at
      least makes the discontinuity explicit, and ``reference_epoch`` counts
      them so downstream code can refuse to compare positions across a reset.
    """

    def __init__(self, cfg=None, *, max_failure_streak: int = 15) -> None:
        from dataclasses import asdict as _asdict

        if cfg is None:
            params: dict = {}
        elif isinstance(cfg, dict):
            params = dict(cfg)
        else:
            params = _asdict(cfg)

        self.enabled: bool = bool(params.get("enabled", True))
        self.feature_type: str = str(params.get("feature_type", "orb"))
        self.max_features: int = int(params.get("max_features", 2000))
        self.ransac_reproj_thresh: float = float(params.get("ransac_reproj_thresh", 3.0))
        self.mask_dynamic_boxes: bool = bool(params.get("mask_dynamic_boxes", True))
        self.box_dilate_px: int = int(params.get("box_dilate_px", 12))
        self.min_matches: int = int(params.get("min_matches", 12))
        self.min_inliers: int = int(params.get("min_inliers", 8))
        self.ratio_test: float = float(params.get("ratio_test", 0.75))
        self.max_scale_change: float = float(params.get("max_scale_change", 1.25))
        self.max_translation_frac: float = float(params.get("max_translation_frac", 0.25))
        self.downscale: float = float(params.get("downscale", 1.0) or 1.0)
        self.max_failure_streak: int = int(
            params.get("max_gmc_failure_streak", max_failure_streak)
        )

        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        """Drop the chain. The next frame becomes the reference frame."""
        self.H_ref_from_cur: np.ndarray = IDENTITY.copy()
        self._prev_gray: np.ndarray | None = None
        self._prev_boxes: list | None = None
        self.frame_count: int = 0
        self.failure_streak: int = 0
        self.n_ok: int = 0
        self.n_failed: int = 0
        self.reference_epoch: int = getattr(self, "reference_epoch", -1) + 1
        self.last: GMCResult = GMCResult(reason="not_run")
        self.history: list[np.ndarray] = []

    # -- main step --------------------------------------------------------
    def update(self, frame, vehicle_boxes=None) -> GMCResult:
        """Ingest one frame and advance the chain. Returns the frame's result."""
        gray = _to_gray(frame)

        if self.downscale and self.downscale > 1.0:
            gray = cv2.resize(
                gray,
                (max(1, int(gray.shape[1] / self.downscale)),
                 max(1, int(gray.shape[0] / self.downscale))),
                interpolation=cv2.INTER_AREA,
            )
            boxes = None if vehicle_boxes is None else [
                [c / self.downscale for c in b[:4]] for b in vehicle_boxes
            ]
        else:
            boxes = None if vehicle_boxes is None else [list(b[:4]) for b in vehicle_boxes]

        if not self.enabled:
            self._prev_gray, self._prev_boxes = gray, boxes
            self.frame_count += 1
            self.last = GMCResult(reason="gmc_disabled", ok=False)
            return self.last

        if self._prev_gray is None:
            # first frame defines the reference
            self._prev_gray, self._prev_boxes = gray, boxes
            self.frame_count += 1
            self.H_ref_from_cur = IDENTITY.copy()
            self.last = GMCResult(H=IDENTITY.copy(), ok=True, reason="reference_frame")
            self.n_ok += 1
            return self.last

        res: GMCResult = estimate_frame_homography(
            self._prev_gray, gray, boxes,
            feature_type=self.feature_type,
            max_features=self.max_features,
            ransac_reproj_thresh=self.ransac_reproj_thresh,
            min_matches=self.min_matches,
            min_inliers=self.min_inliers,
            mask_dynamic_boxes=self.mask_dynamic_boxes,
            box_dilate_px=self.box_dilate_px,
            ratio_test=self.ratio_test,
            max_scale_change=self.max_scale_change,
            max_translation_frac=self.max_translation_frac,
            prev_boxes=self._prev_boxes,
            detail=True,
        )

        if res.ok:
            # H maps prev -> cur, so cur -> prev is its inverse, and
            #   H_ref_from_cur = H_ref_from_prev @ H_prev_from_cur
            H_prev_from_cur = invert_homography(res.H)
            if self.downscale and self.downscale > 1.0:
                H_prev_from_cur = self._rescale_homography(H_prev_from_cur, self.downscale)
            self.H_ref_from_cur = self.H_ref_from_cur @ H_prev_from_cur
            if abs(self.H_ref_from_cur[2, 2]) > 1e-12:
                self.H_ref_from_cur = self.H_ref_from_cur / self.H_ref_from_cur[2, 2]
            self.history.append(res.H.copy())
            self.failure_streak = 0
            self.n_ok += 1
        else:
            # chain frozen — see class docstring
            self.failure_streak += 1
            self.n_failed += 1
            if self.max_failure_streak and self.failure_streak >= self.max_failure_streak:
                keep = self.reference_epoch
                self.reset()
                self.reference_epoch = keep + 1
                self._prev_gray, self._prev_boxes = gray, boxes
                self.frame_count += 1
                res.reason = f"{res.reason}|chain_reset"
                self.last = res
                return self.last

        self._prev_gray, self._prev_boxes = gray, boxes
        self.frame_count += 1
        self.last = res
        return res

    @staticmethod
    def _rescale_homography(H: np.ndarray, s: float) -> np.ndarray:
        """Lift a homography estimated on a downscaled image to full scale."""
        S = np.array([[1.0 / s, 0, 0], [0, 1.0 / s, 0], [0, 0, 1.0]], dtype=np.float64)
        Sinv = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1.0]], dtype=np.float64)
        return Sinv @ H @ S

    # -- projection helpers ----------------------------------------------
    def to_reference(self, points) -> np.ndarray:
        """Current-frame pixels -> reference-frame pixels."""
        return compensate_track_positions(points, self.H_ref_from_cur)

    def from_reference(self, points) -> np.ndarray:
        """Reference-frame pixels -> current-frame pixels (for drawing)."""
        return apply_gmc(points, invert_homography(self.H_ref_from_cur))

    def boxes_to_reference(self, boxes) -> np.ndarray:
        return apply_gmc_boxes(boxes, self.H_ref_from_cur)

    # -- reporting --------------------------------------------------------
    @property
    def health(self) -> float:
        total = self.n_ok + self.n_failed
        return (self.n_ok / total) if total else 0.0

    def stats(self) -> dict:
        return {
            "frames": self.frame_count,
            "ok": self.n_ok,
            "failed": self.n_failed,
            "health": round(self.health, 4),
            "failure_streak": self.failure_streak,
            "reference_epoch": self.reference_epoch,
            "feature_type": self.feature_type,
            "enabled": self.enabled,
            "last": self.last.to_dict(),
        }


# --------------------------------------------------------------------------
# Self-check: synthetic pan, no footage needed.
# --------------------------------------------------------------------------
if __name__ == "__main__":   # pragma: no cover - manual smoke check
    rng = np.random.default_rng(0)
    base = (rng.random((480, 640)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    # add hard corners so ORB has something to lock onto
    for _ in range(120):
        x, y = int(rng.integers(20, 620)), int(rng.integers(20, 460))
        cv2.rectangle(base, (x, y), (x + 8, y + 8), int(rng.integers(0, 255)), -1)

    shift = np.float32([[1, 0, 7], [0, 1, -4]])
    moved = cv2.warpAffine(base, shift, (640, 480))

    H, n_in, ok = estimate_frame_homography(base, moved, vehicle_boxes=[])
    tmag, scale = decompose_translation_scale(H)
    print(f"ok={ok} inliers={n_in} translation={H[0,2]:+.2f},{H[1,2]:+.2f} "
          f"(expected +7.00,-4.00) scale={scale:.4f}")

    est = GMCEstimator({"feature_type": "orb"})
    est.update(base, [])
    est.update(moved, [])
    print("H_ref_from_cur translation:", est.H_ref_from_cur[:2, 2].round(2),
          "(expected -7, +4)")
    print("stats:", est.stats())
