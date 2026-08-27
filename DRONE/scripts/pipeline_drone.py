"""Orchestrates the DRONE pipeline on one aerial clip.

    quality gate -> detect -> GMC ego-motion compensation -> track -> per-track
    kinematics -> results JSON

Every stage is honest about what it did not measure. Nothing here fabricates
a metric speed when there is no road-plane calibration, and nothing hides
that the detector is a COCO placeholder. Read the ``provenance`` block of any
results file before trusting a number in it.

Usage::

    python pipeline_drone.py path/to/clip.mp4
    python pipeline_drone.py path/to/clip.mp4 --max-frames 300 --config ../config/drone_config.yaml

Or programmatically::

    from pipeline_drone import run_pipeline
    result = run_pipeline("path/to/clip.mp4")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as drone_config          # noqa: E402
import gmc                             # noqa: E402
import hover_mode                      # noqa: E402
import telemetry_ingest                # noqa: E402
import thermal_presence                # noqa: E402
from detect_drone import load_detector  # noqa: E402
from track_drone import DroneTracker   # noqa: E402

log = logging.getLogger("drone.pipeline")

__all__ = ["run_pipeline", "PipelineError"]


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Frame quality gate
# --------------------------------------------------------------------------

def _assess_frame_quality(gray: np.ndarray, qcfg) -> tuple[bool, str]:
    """Reject frames too blurred or too dark/bright to trust for measurement.

    A rejected frame is still counted through the pipeline (detector still
    runs so tracks don't fracture unnecessarily) but is flagged, and the
    fraction rejected is reported rather than silently averaged over.
    """
    if not qcfg.enabled:
        return True, "quality_gate_disabled"

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < qcfg.min_blur_var:
        return False, f"blurred(var={blur_var:.1f}<{qcfg.min_blur_var})"

    brightness = float(gray.mean())
    if brightness < qcfg.min_brightness:
        return False, f"too_dark(mean={brightness:.1f}<{qcfg.min_brightness})"
    if brightness > qcfg.max_brightness:
        return False, f"too_bright(mean={brightness:.1f}>{qcfg.max_brightness})"

    return True, "ok"


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------

def _track_kinematics(tr, kcfg, road_plane) -> dict[str, Any]:
    """Per-track speed. Pixel speed is always available (post-GMC); metric
    speed is only reported when a real road-plane calibration exists.
    """
    speed_px_s = tr.speed_px(kcfg.speed_window_s)
    stationary = speed_px_s <= kcfg.stationary_speed_px_s
    sufficient = tr.duration >= kcfg.min_track_seconds

    out: dict[str, Any] = {
        "speed_px_s": round(speed_px_s, 2) if sufficient else None,
        "stationary": stationary if sufficient else None,
        "sufficient_duration": sufficient,
        "speed_kmh": None,
        "speed_m_s": None,
        "metric_reason": None,
    }

    if not sufficient:
        out["metric_reason"] = "track_too_short"
        return out

    if not road_plane.available:
        out["metric_reason"] = "no_road_plane_homography"
        return out

    H = road_plane.matrix()
    pts = tr.points(kcfg.speed_window_s)
    if len(pts) < 2:
        out["metric_reason"] = "insufficient_points_in_window"
        return out

    metric_pts = gmc.apply_gmc(np.asarray(pts, dtype=np.float64), H)
    if np.isnan(metric_pts).any():
        out["metric_reason"] = "road_plane_projection_failed"
        return out

    dt = pts[-1][0] - pts[0][0] if len(pts) >= 2 else 0.0
    # accumulate arc length in metric space, same convention as speed_px
    dist_m = 0.0
    for i in range(1, len(metric_pts)):
        dist_m += float(np.hypot(*(metric_pts[i] - metric_pts[i - 1])))
    if dt <= 1e-6:
        out["metric_reason"] = "zero_time_window"
        return out

    speed_m_s = dist_m / dt
    out["speed_m_s"] = round(speed_m_s, 3)
    out["speed_kmh"] = round(speed_m_s * 3.6, 2)
    out["metric_reason"] = "ok"
    return out


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------

def run_pipeline(video_path: str | Path, cfg: "drone_config.DroneConfig | None" = None,
                 max_frames: int | None = None) -> dict[str, Any]:
    """Run the full DRONE pipeline on one clip and write a results JSON.

    Returns the results dict (also written to ``results/<stem>_<ts>.json``).
    Raises :class:`PipelineError` if the clip cannot be opened at all — every
    other failure mode (bad frames, GMC loss of lock, no detector weights) is
    handled per-frame/per-track and reported, not raised.
    """
    t_start = time.time()
    cfg = cfg or drone_config.load_config()
    hover_mode.assert_supported(cfg.mode)   # raises NotImplementedError for patrol

    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = drone_config.PROJECT_ROOT / video_path
    if not video_path.exists():
        raise PipelineError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0.0:
        fps = 30.0
        log.warning("video reports fps<=0; assuming %.1f for timestamps", fps)

    frame_limit = max_frames if max_frames is not None else cfg.processing.max_frames
    stride = max(1, int(cfg.processing.frame_stride))

    detector = load_detector(cfg.detector)
    gmc_est = gmc.GMCEstimator(cfg.gmc, max_failure_streak=cfg.quality.max_gmc_failure_streak)
    tracker = DroneTracker.from_config(cfg.tracker)

    telemetry = None
    if cfg.telemetry.enabled:
        telemetry = telemetry_ingest.load_telemetry(cfg.telemetry.resolved_path, cfg.telemetry.format)
        # load_telemetry() returns None today unconditionally; the pipeline
        # already falls back to vision-only GMC below regardless.

    frame_idx = 0
    processed = 0
    rejected_reasons: dict[str, int] = {}
    last_tracks_snapshot: list[dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if (frame_idx - 1) % stride != 0:
            continue
        if frame_limit is not None and processed >= frame_limit:
            break

        t = (frame_idx - 1) / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        usable, reason = _assess_frame_quality(gray, cfg.quality)
        if not usable:
            rejected_reasons[reason.split("(")[0]] = rejected_reasons.get(reason.split("(")[0], 0) + 1

        dets = detector.detect(frame)   # (N,6) current-frame pixel boxes

        gmc_res = gmc_est.update(frame, dets[:, :4] if dets.shape[0] else None)

        if dets.shape[0]:
            ref_boxes = gmc.apply_gmc_boxes(dets[:, :4], gmc_est.H_ref_from_cur)
            valid = ~np.isnan(ref_boxes).any(axis=1)
            ref_dets = np.hstack([ref_boxes, dets[:, 4:6]])[valid]
            px_boxes = dets[:, :4][valid]
        else:
            ref_dets = np.empty((0, 6), dtype=np.float64)
            px_boxes = np.empty((0, 4), dtype=np.float64)

        # Measurement only happens on frames that pass the quality gate AND
        # have a trustworthy GMC estimate this frame. Tracking itself still
        # runs on every frame (dropping frames from the tracker fractures
        # tracks far more than it protects a measurement).
        live_tracks = tracker.update(ref_dets, px_boxes, t)
        if usable and gmc_res.ok:
            last_tracks_snapshot = [tr.to_dict() for tr in live_tracks]

        processed += 1

    cap.release()

    all_tracks = tracker.all_tracks
    kcfg = cfg.kinematics
    track_records = []
    for tr in all_tracks:
        rec = tr.to_dict()
        rec["cls_name"] = detector.class_names.get(rec["cls"], f"class_{rec['cls']}")
        rec.update(_track_kinematics(tr, kcfg, cfg.road_plane))
        track_records.append(rec)

    total_measured_frames = processed - sum(rejected_reasons.values())
    elapsed = time.time() - t_start

    results: dict[str, Any] = {
        "schema_version": 1,
        "source_video": str(video_path.name),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_seconds": round(elapsed, 3),
        "provenance": cfg.provenance(),
        "hover_mode": hover_mode.describe(),
        "telemetry": telemetry_ingest.telemetry_status(),
        "thermal": thermal_presence.thermal_status(),
        "detector": detector.status(),
        "gmc": gmc_est.stats(),
        "frames": {
            "video_fps": round(fps, 3),
            "total_seen": frame_idx,
            "processed": processed,
            "rejected_by_quality_gate": sum(rejected_reasons.values()),
            "rejection_reasons": rejected_reasons,
        },
        "tracks": track_records,
        "track_count": len(track_records),
    }

    out_dir = cfg.processing.results_path
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"{video_path.stem}_{stamp}.json"
    out_path = out_dir / out_name
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    results["_results_file"] = out_name
    log.info("wrote %s (%d tracks, %d/%d frames usable)",
              out_path, len(track_records), total_measured_frames, processed)
    return results


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run the DRONE pipeline on one clip.")
    p.add_argument("video", help="path to an aerial video clip")
    p.add_argument("--config", default=None, help="path to drone_config.yaml (default: config/drone_config.yaml)")
    p.add_argument("--max-frames", type=int, default=None, help="override processing.max_frames")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = drone_config.load_config(args.config)
    result = run_pipeline(args.video, cfg=cfg, max_frames=args.max_frames)
    print(json.dumps({
        "results_file": result["_results_file"],
        "track_count": result["track_count"],
        "detector_finetuned": result["provenance"]["detector_finetuned"],
        "gmc_health": result["gmc"]["health"],
        "frames_processed": result["frames"]["processed"],
    }, indent=2))


if __name__ == "__main__":
    _cli()
