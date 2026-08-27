#!/usr/bin/env python3
"""Full DRONE analysis + HUD render, end to end, on one real aerial clip.

    quality gate -> detect -> GMC ego-motion compensation -> track ->
    physics (physics_drone.py) -> queue/blockage (queue_blockage_drone.py) ->
    annotated video + results JSON

This is a new top-level script rather than a modification of
``pipeline_drone.py`` — that module is already self-verified and is left
untouched. This script reuses its frame-quality gate
(``pipeline_drone._assess_frame_quality``) and every core building block
(``config``, ``gmc``, ``detect_drone``, ``track_drone``, ``hover_mode``,
``telemetry_ingest``, ``thermal_presence``) directly, and additionally caches
the per-frame data ``pipeline_drone.run_pipeline`` does not keep (per-frame
pixel boxes for rendering, per-frame reference boxes for physics) so it can
render a demonstrable HUD the way ``CCTV/scripts/render_physics_demo.py``
does for the CCTV side.

Two passes over the clip, same reasoning as the CCTV renderer:

1. Detect + GMC + track every frame. Record ``frame_records[frame_idx]`` (for
   drawing, current-frame pixel coordinates) and
   ``track_samples[track_id]`` (``physics_drone.TrackSample`` list, reference-
   frame coordinates, appended only on frames that passed the quality gate
   AND had a trustworthy GMC estimate — the same "measurement only on trusted
   frames" rule ``pipeline_drone.py`` already applies).
2. Physics (per track) and queue/blockage (over the whole clip) are computed
   once from ``track_samples``. Re-open the clip and render every frame using
   ``frame_records`` for boxes/trails and the physics/queue/blockage results
   for the HUD text and highlight colour.

Usage::

    python scripts/run_drone_analysis.py path/to/clip.mp4 --out-dir ../inference

Honesty
-------
Every number in the HUD and the JSON is traceable to one of: a real detector
box (VisDrone-fine-tuned YOLOv8x when ``detector.weights`` is configured —
``detector_finetuned: true`` — or the generic COCO placeholder with a printed
banner when it is not), a real GMC homography (ok/failed shown per-frame), a
real windowed speed, or an explicitly labelled ESTIMATE (class-width km/h,
momentum). Nothing here invents a metric measurement. A clip that produces no
queue and no blockage is reported as exactly that — not forced into a
marginal finding. Track identity is native Ultralytics BoT-SORT
(``tracker.use_native_botsort: true``, the default for the real-footage
config) rather than this project's earlier hand-rolled greedy-IoU
``track_drone.DroneTracker`` — see ``track_drone.NativeTrackRegistry`` for
the bridge and ``models/tracker/botsort_drone.yaml`` for the tuned config.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as drone_config              # noqa: E402
import gmc                                  # noqa: E402
import hover_mode                           # noqa: E402
import telemetry_ingest                     # noqa: E402
import thermal_presence                     # noqa: E402
from detect_drone import load_detector      # noqa: E402
from track_drone import DroneTracker, NativeTrackRegistry  # noqa: E402
from pipeline_drone import _assess_frame_quality  # noqa: E402
from physics_drone import TrackSample, compute_track_physics  # noqa: E402
from queue_blockage_drone import detect_queues, detect_blockages  # noqa: E402

log = logging.getLogger("drone.run_analysis")

# ---------------------------------------------------------------------------
# visual style — matches CCTV/scripts/render_physics_demo.py's convention
# ---------------------------------------------------------------------------
RED = (0, 0, 235)        # blockage candidate
AMBER = (0, 165, 255)    # queue member
GREY = (140, 140, 140)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CLASS_COLOR = {
    "car": (78, 161, 255),
    "van": (78, 161, 200),
    "truck": (255, 138, 76),
    "bus": (167, 139, 250),
    "motor": (78, 201, 126),
    "bicycle": (245, 196, 81),
    "tricycle": (200, 160, 90),
    "awning-tricycle": (200, 160, 90),
    "pedestrian": (255, 255, 255),
    "people": (255, 255, 255),
}


def _label(frame: np.ndarray, x: int, y_top: int, lines: list[str], color) -> None:
    ty = y_top - 4
    for line in reversed(lines):
        (tw, th), _ = cv2.getTextSize(line, FONT, 0.42, 1)
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 6, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, line, (x + 3, ty - 2), FONT, 0.42, color, 1, cv2.LINE_AA)
        ty -= th + 8


def _draw_trail(frame: np.ndarray, pts: list[tuple[float, float]], color) -> None:
    n = len(pts)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / n
        thickness = 1 if alpha < 0.6 else 2
        p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p1 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# pass 1: detect + GMC + track, caching per-frame data
# ---------------------------------------------------------------------------

def run_tracking(video_path: Path, cfg: "drone_config.DroneConfig", max_frames: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0.0:
        fps = 30.0
        log.warning("video reports fps<=0; assuming %.1f for timestamps", fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_limit = max_frames if max_frames is not None else cfg.processing.max_frames
    stride = max(1, int(cfg.processing.frame_stride))

    detector = load_detector(cfg.detector)
    gmc_est = gmc.GMCEstimator(cfg.gmc, max_failure_streak=cfg.quality.max_gmc_failure_streak)

    use_native = bool(getattr(cfg.tracker, "use_native_botsort", False))
    if use_native:
        tracker = NativeTrackRegistry(min_hits=cfg.tracker.min_hits,
                                      max_time_lost=cfg.tracker.max_time_lost)
        botsort_yaml_path = cfg.tracker.botsort_yaml_path
        botsort_yaml_arg = str(botsort_yaml_path) if botsort_yaml_path and botsort_yaml_path.exists() else "botsort.yaml"
        log.info("tracking backend: native Ultralytics BoT-SORT (%s)", botsort_yaml_arg)
    else:
        tracker = DroneTracker.from_config(cfg.tracker)
        botsort_yaml_arg = None
        log.info("tracking backend: hand-rolled DroneTracker (legacy)")

    frame_records: dict[int, dict[int, dict]] = {}
    track_samples: dict[int, list[TrackSample]] = {}
    gmc_per_frame: dict[int, dict] = {}       # frame_idx -> {"ok":..., "reason":...}
    quality_per_frame: dict[int, dict] = {}   # frame_idx -> {"usable":..., "reason":...}

    frame_idx = 0
    processed = 0
    rejected_reasons: dict[str, int] = {}

    t0 = time.perf_counter()
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
            key = reason.split("(")[0]
            rejected_reasons[key] = rejected_reasons.get(key, 0) + 1
        quality_per_frame[frame_idx - 1] = {"usable": usable, "reason": reason}

        if use_native:
            dets7 = detector.track(frame, tracker_yaml=botsort_yaml_arg)
            dets = dets7[:, :6] if dets7.shape[0] else np.empty((0, 6), dtype=np.float64)
            track_ids_all = dets7[:, 6] if dets7.shape[0] else np.empty((0,), dtype=np.float64)
        else:
            dets = detector.detect(frame)
            track_ids_all = None

        gmc_res = gmc_est.update(frame, dets[:, :4] if dets.shape[0] else None)
        gmc_per_frame[frame_idx - 1] = {"ok": bool(gmc_res.ok), "reason": gmc_res.reason}

        if dets.shape[0]:
            ref_boxes = gmc.apply_gmc_boxes(dets[:, :4], gmc_est.H_ref_from_cur)
            valid = ~np.isnan(ref_boxes).any(axis=1)
            ref_dets = np.hstack([ref_boxes, dets[:, 4:6]])[valid]
            px_boxes = dets[:, :4][valid]
            if use_native:
                track_ids = track_ids_all[valid]
        else:
            ref_dets = np.empty((0, 6), dtype=np.float64)
            px_boxes = np.empty((0, 4), dtype=np.float64)
            if use_native:
                track_ids = np.empty((0,), dtype=np.float64)

        if use_native:
            live_tracks = tracker.update(ref_dets, px_boxes, track_ids, t)
        else:
            live_tracks = tracker.update(ref_dets, px_boxes, t)

        snap: dict[int, dict] = {}
        for tr in live_tracks:
            snap[tr.track_id] = {
                "px_box": [float(v) for v in tr.px_box],
                "cls": int(tr.majority_cls),
                "score": float(tr.score),
            }
            if usable and gmc_res.ok:
                track_samples.setdefault(tr.track_id, []).append(
                    TrackSample(t=t, ref_box=tuple(float(v) for v in tr.box),
                               px_box=tuple(float(v) for v in tr.px_box),
                               score=float(tr.score), cls=int(tr.majority_cls))
                )
        frame_records[frame_idx - 1] = snap

        processed += 1

    cap.release()
    elapsed = time.perf_counter() - t0

    return {
        "fps": fps, "w": w, "h": h,
        "n_frames_total": frame_idx,
        "n_frames_processed": processed,
        "frame_records": frame_records,
        "track_samples": track_samples,
        "gmc_per_frame": gmc_per_frame,
        "quality_per_frame": quality_per_frame,
        "rejected_reasons": rejected_reasons,
        "detector": detector,
        "gmc_est": gmc_est,
        "tracker": tracker,
        "use_native_tracker": use_native,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# pass 2: render
# ---------------------------------------------------------------------------

def render(video_path: Path, out_path: Path, run_info: dict, track_physics: dict[int, dict],
          queue_result: dict, blockage_result: dict, trail_seconds: float,
          use_ffmpeg: bool, frame_stride: int = 1) -> None:
    fps, w, h = run_info["fps"], run_info["w"], run_info["h"]
    frame_records = run_info["frame_records"]
    gmc_per_frame = run_info["gmc_per_frame"]

    # When frame_stride > 1 (real-footage batch: every 2nd raw frame is
    # actually detected/tracked, see config/drone_config_real_footage.yaml's
    # processing.frame_stride note), frame_records only has an entry for the
    # RAW frame indices that were processed -- every other raw frame has no
    # detection to draw at all. Writing those blank raw frames into the
    # output would flash boxes on/off every other frame, which looks broken
    # even though nothing is actually wrong. Instead, only the processed
    # frames are written, at a proportionally reduced output fps, so the
    # rendered clip's wall-clock duration still matches the source and every
    # written frame has real (or honestly absent) evidence drawn on it.
    out_fps = fps / max(1, int(frame_stride))

    # -- per-track flags active at a given time --------------------------
    queue_member_of: dict[int, list[dict]] = {}
    for ev in queue_result["events"]:
        for tid in ev["track_ids"]:
            queue_member_of.setdefault(tid, []).append(ev)

    blockage_of: dict[int, list[dict]] = {}
    for ev in blockage_result["events"]:
        blockage_of.setdefault(ev["track_id"], []).append(ev)

    trail_frames = max(1, int(round(trail_seconds * fps)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not re-open video for rendering: {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"_tmp_{out_path.name}")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    n_queue = len(queue_result["events"])
    n_blockage_candidates = sum(1 for e in blockage_result["events"] if e["classification"] == "blockage_candidate")

    fidx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        if fidx not in frame_records:
            continue    # not a processed frame at this stride -- skip, see note above
        snap = frame_records.get(fidx, {})
        t = fidx / fps

        for tid, info in snap.items():
            cls = info["cls"]
            phys = track_physics.get(tid, {})
            cls_name = phys.get("cls_name", str(cls))

            active_blockage = [e for e in blockage_of.get(tid, [])
                               if e["classification"] == "blockage_candidate" and e["start_t"] - 1.0 <= t]
            active_queue = [e for e in queue_member_of.get(tid, [])
                            if e["start_t"] - 1.0 <= t <= e["end_t"] + 1.0]

            if active_blockage:
                col = RED
            elif active_queue:
                col = AMBER
            else:
                col = CLASS_COLOR.get(cls_name, GREY)

            x1, y1, x2, y2 = [int(v) for v in info["px_box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if (active_blockage or active_queue) else 1)

            pts = []
            for j in range(max(0, fidx - trail_frames), fidx + 1):
                s = frame_records.get(j, {}).get(tid)
                if s is not None:
                    bx = s["px_box"]
                    pts.append(((bx[0] + bx[2]) / 2.0, bx[3]))
            _draw_trail(frame, pts, col)

            speed_px_s = phys.get("speed_px_s")
            speed_kmh_est = phys.get("speed_kmh_estimate")
            accel = phys.get("accel_mps2_estimate")
            momentum = phys.get("momentum_kgms_estimate")

            lines = [f"#{tid} {cls_name}"]
            if speed_px_s is not None:
                kmh_txt = f" (~{speed_kmh_est:.0f} km/h est.)" if speed_kmh_est is not None else ""
                lines.append(f"{speed_px_s:.0f} px/s{kmh_txt}")
            if accel is not None:
                lines.append(f"accel ~{accel:+.1f} m/s2 est.")
            if momentum is not None:
                lines.append(f"p ~{momentum:.0f} kg m/s est.")
            if active_blockage:
                ev = active_blockage[0]
                lines.append(f"BLOCKAGE cand.: stationary {ev['stationary_s']:.1f}s, "
                             f"{len(ev['evidence'])} neighbour(s) slowed")
            if active_queue:
                ev = active_queue[0]
                lines.append(f"QUEUE member: {ev['peak_vehicle_count']} vehicles, {ev['duration_s']:.1f}s")

            _label(frame, x1, y1, lines, col)

        gmc_info = gmc_per_frame.get(fidx, {"ok": None, "reason": "n/a"})
        gmc_txt = "GMC ok" if gmc_info["ok"] else f"GMC FAILED ({gmc_info['reason']})"
        gmc_col = (140, 235, 140) if gmc_info["ok"] else (60, 60, 235)

        head = (f"{video_path.name}  t={t:.2f}s  tracked={len(snap)}  "
               f"queue_events={n_queue}  blockage_candidates={n_blockage_candidates}")
        cv2.rectangle(frame, (0, 0), (min(w, 1250), 44), (0, 0, 0), -1)
        cv2.putText(frame, head, (6, 18), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, gmc_txt, (6, 38), FONT, 0.5, gmc_col, 1, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    if use_ffmpeg and shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
            check=True,
        )
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(out_path)


# ---------------------------------------------------------------------------
# results JSON
# ---------------------------------------------------------------------------

def build_result_json(video_path: Path, cfg: "drone_config.DroneConfig", run_info: dict,
                      track_physics: dict[int, dict], queue_result: dict,
                      blockage_result: dict, out_video_name: str, elapsed_total: float) -> dict:
    detector = run_info["detector"]
    gmc_est = run_info["gmc_est"]
    tracker = run_info["tracker"]
    use_native = bool(run_info.get("use_native_tracker"))

    tracks_out = []
    for tr in tracker.all_tracks:
        rec = tr.to_dict()
        phys = track_physics.get(tr.track_id, {})
        rec.update(phys)
        tracks_out.append(rec)

    total_measured = run_info["n_frames_processed"] - sum(run_info["rejected_reasons"].values())

    return {
        "schema_version": 1,
        "source_video": str(video_path.name),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_seconds": round(elapsed_total, 3),
        "provenance": cfg.provenance(),
        "hover_mode": hover_mode.describe(),
        "telemetry": telemetry_ingest.telemetry_status(),
        "thermal": thermal_presence.thermal_status(),
        "detector": detector.status(),
        "tracker": {
            "backend": "native_botsort" if use_native else "hand_rolled_greedy_iou",
            "botsort_yaml": str(cfg.tracker.botsort_yaml) if use_native else None,
            "with_reid": True if use_native else None,
            "min_hits": cfg.tracker.min_hits,
            "max_time_lost_frames": cfg.tracker.max_time_lost,
        },
        "gmc": gmc_est.stats(),
        "frames": {
            "video_fps": round(run_info["fps"], 3),
            "total_seen": run_info["n_frames_total"],
            "processed": run_info["n_frames_processed"],
            "measured_usable": total_measured,
            "rejected_by_quality_gate": sum(run_info["rejected_reasons"].values()),
            "rejection_reasons": run_info["rejected_reasons"],
        },
        "tracks": tracks_out,
        "track_count": len(tracks_out),
        "queue": queue_result,
        "blockage": blockage_result,
        "annotated_video": out_video_name,
        "notes": (
            (
                "detector_finetuned=true: YOLOv8x fine-tuned on VisDrone2019-DET "
                "(dronefreak/visdrone-yolov8x, mAP50 36.8/mAP50-95 21.5 on the "
                "VisDrone test set per that checkpoint's own model card -- NOT "
                "measured against this project's own footage, no ground truth "
                "exists for that). Chosen over Ultralytics' official DOTA-pretrained "
                "OBB weights after a side-by-side check on real frames from this "
                "footage: the OBB model detected almost nothing (0-2 boxes/frame on "
                "a scene with 10+ visible motorbikes) while this VisDrone checkpoint "
                "found essentially all of them. "
                if not detector.status().get("is_placeholder") else
                "detector_finetuned=false: placeholder COCO-pretrained detector, never "
                "trained on an aerial viewing angle -- box/track quality on this footage "
                "is a first mechanical test, not a validated accuracy number. "
            ) +
            (
                "tracker=native Ultralytics BoT-SORT (with_reid=true, own Kalman + "
                "appearance ReID + camera-motion compensation), replacing this "
                "project's earlier hand-rolled greedy-IoU tracker for the real-footage "
                "run -- chosen for nadir footage's frequent top-down occlusion between "
                "passing vehicles, which a motion/IoU-only associator is more prone to "
                "losing identity through. "
                if use_native else
                "tracker=hand-rolled greedy-IoU DroneTracker (legacy path). "
            ) +
            "speed_kmh is null unless road_plane.homography is calibrated (it is not); "
            "speed_kmh_estimate is a monocular class-width ESTIMATE, not a "
            "measurement. queue/blockage events are a corridor-free heuristic with "
            "no ground truth to validate precision/recall against -- see "
            "queue_blockage_drone.py module docstring. This project's drone platform "
            "is near-static (hover, small residual jitter, not continuous patrol); "
            "gmc.py's own homography correction is expected to be small on this "
            "footage and gmc.health reports how often a trustworthy estimate was "
            "produced regardless of its magnitude."
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def analyze_one(video_path: Path, cfg: "drone_config.DroneConfig", out_dir: Path,
                max_frames: int | None, trail_seconds: float = 2.0,
                use_ffmpeg: bool = True) -> dict[str, Any]:
    t_start = time.time()
    hover_mode.assert_supported(cfg.mode)

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    print(f"[{video_path.name}] pass 1/2: detect + GMC + track ...")
    run_info = run_tracking(video_path, cfg, max_frames)
    print(f"[{video_path.name}] pass 1 done: {run_info['n_frames_processed']} frames in "
          f"{run_info['elapsed_s']:.1f}s ({run_info['n_frames_processed'] / max(run_info['elapsed_s'], 1e-6):.1f} fps), "
          f"{len(run_info['tracker'].all_tracks)} tracks, gmc_health={run_info['gmc_est'].health:.3f}")

    class_names = run_info["detector"].class_names
    vehicle_classes = set(cfg.detector.vehicle_classes)

    track_physics: dict[int, dict] = {}
    for tid, samples in run_info["track_samples"].items():
        cls_id = samples[-1].cls if samples else -1
        cls_name = class_names.get(cls_id, str(cls_id))
        track_physics[tid] = compute_track_physics(
            tid, samples, cls_name, cfg.kinematics, cfg.physics, cfg.road_plane, gmc.apply_gmc
        )

    queue_result = detect_queues(run_info["track_samples"], class_names, vehicle_classes,
                                 cfg.kinematics, cfg.queue)
    blockage_result = detect_blockages(run_info["track_samples"], class_names, vehicle_classes,
                                       cfg.kinematics, cfg.blockage)
    n_blockage_candidates = sum(1 for e in blockage_result["events"] if e["classification"] == "blockage_candidate")
    print(f"[{video_path.name}] queue candidates: {len(queue_result['events'])}  "
         f"blockage candidates: {n_blockage_candidates} "
         f"(stationary events total: {len(blockage_result['events'])})")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    out_video = out_dir / f"{stem}_annotated.mp4"
    out_json = out_dir / f"{stem}_results.json"

    print(f"[{video_path.name}] pass 2/2: rendering HUD -> {out_video.name}")
    render(video_path, out_video, run_info, track_physics, queue_result, blockage_result,
          trail_seconds, use_ffmpeg, frame_stride=int(cfg.processing.frame_stride))

    elapsed_total = time.time() - t_start
    result = build_result_json(video_path, cfg, run_info, track_physics, queue_result,
                               blockage_result, out_video.name, elapsed_total)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"[{video_path.name}] wrote {out_video}")
    print(f"[{video_path.name}] wrote {out_json}")
    print(f"[{video_path.name}] total {elapsed_total:.1f}s")

    result["_out_video"] = str(out_video)
    result["_out_json"] = str(out_json)
    return result


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="path to an aerial video clip")
    p.add_argument("--out-dir", required=True, help="output directory for the annotated video + JSON")
    p.add_argument("--config", default=None, help="path to drone_config.yaml")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--trail-seconds", type=float, default=2.0)
    p.add_argument("--no-ffmpeg", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = drone_config.load_config(args.config)
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)

    result = analyze_one(video_path, cfg, out_dir, args.max_frames, args.trail_seconds,
                         use_ffmpeg=not args.no_ffmpeg)
    print(json.dumps({
        "track_count": result["track_count"],
        "queue_events": len(result["queue"]["events"]),
        "blockage_events": len(result["blockage"]["events"]),
        "gmc_health": result["gmc"]["health"],
        "detector_finetuned": result["provenance"]["detector_finetuned"],
    }, indent=2))


if __name__ == "__main__":
    _cli()
