#!/usr/bin/env python3
"""
scripts/render_physics_demo.py

Standalone demo renderer for NETRA's physics-based collision-analysis
pipeline: detection (netra.detect.Detector) + tracking
(netra.track.ByteTracker) + the rotation-gated pair scorer
(netra.events.rotation_gate.score_pairs, ported from the COMBINED project's
inference2, see that module's docstring for the physics and the honesty
caveat on its 4/4-videos number).

Run on one real video clip end to end:

    python scripts/render_physics_demo.py --video demo/4.mp4 --out demo/4_physics.mp4

Two passes over the clip:

1. Detect + track every frame, GPU by default. After every tracker update
   the CURRENT live tracks (``tracker.all_tracks`` -- confirmed and recently
   lost) are handed to ``rotation_gate.score_pairs`` with ``now_t`` advancing.
   This is deliberate, not merely "run it once at the end": ``Track.box_history``
   is a bounded deque (maxlen=120 -- a few seconds at native fps), so a
   contact that happened early in a clip that keeps running scrolls out of
   that window long before the clip ends. Scoring continuously, and keeping
   the single best-ever ``PairResult`` (plus, per vehicle, the best score it
   ever achieved in any pair), is what makes an end-of-clip report honest.
2. Re-open the clip and render every frame using the per-frame track
   snapshots recorded in pass 1 -- boxes, a short trajectory trail, live
   speed, and (for anything gated by rotation_gate) its evidence score. If
   the best pair ever seen clears ``--threshold``, its two boxes go red for
   the whole clip, its score/interaction/heading appear in a HUD bar, and a
   "<< CONTACT" marker appears within +/-0.5s of the sub-frame-refined
   contact instant. If nothing clears the threshold, the HUD says so in
   plain words -- no collision indicator is ever drawn without a pair
   actually clearing it.

Honesty notes (see the project's LIMITATIONS.md and rotation_gate.py):
* These cameras carry no image-to-ground calibration (checked their
  config/cameras/*.json: no homography). The speed drawn on screen is
  therefore px/s, labelled as such -- the same convention as
  ``Track.speed_px()``'s own docstring ("never reported to a user as a
  physical speed"). A second, explicitly-labelled "class-width estimate"
  km/h number is also shown, computed the same way rotation_gate computes
  its *own* internal km/h (a monocular per-track scale from the vehicle's
  apparent width against an assumed physical width for its class) -- it is
  an estimate, not a calibrated measurement, and is labelled as one both on
  screen and in the JSON.
* A collision indicator is only ever drawn when score_pairs actually
  returned a pair whose score clears ``--threshold``. A weak or empty result
  is reported as exactly that.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from netra.config import load_config  # noqa: E402
from netra.detect import (  # noqa: E402
    Detector,
    MOTORISED_CLASSES,
    ROAD_USER_CLASSES,
    VULNERABLE_CLASSES,
)
from netra.events.rotation_gate import (  # noqa: E402
    PairResult,
    RotationGateConfig,
    score_pairs,
)
from netra.predict import pixels_per_metre  # noqa: E402
from netra.track import ByteTracker  # noqa: E402

# ---------------------------------------------------------------------------
# visual style -- matches IDEAS/COMBINED/render_inference2.py's convention
# ---------------------------------------------------------------------------

RED = (0, 0, 235)      # top / confident pair
AMBER = (0, 165, 255)  # a vehicle that scored meaningfully in SOME pair
GREY = (140, 140, 140) # fallback for an unclassed track
FONT = cv2.FONT_HERSHEY_SIMPLEX

CLASS_COLOR = {
    "car": (78, 161, 255),
    "truck": (255, 138, 76),
    "bus": (167, 139, 250),
    "motorcycle": (78, 201, 126),
    "bicycle": (245, 196, 81),
    "person": (255, 255, 255),
}

GATED_CLASSES = MOTORISED_CLASSES | VULNERABLE_CLASSES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="input clip (relative to repo root or absolute)")
    ap.add_argument("--out", required=True, help="output annotated mp4")
    ap.add_argument("--json", default=None, help="output JSON path (default: <out>_result.json next to --out)")
    ap.add_argument("--config", default=str(REPO / "config" / "config.yaml"))
    ap.add_argument("--weights", default=str(REPO / "models" / "yolo26m.pt"))
    ap.add_argument("--device", default="auto", help="'auto' | 'cpu' | 'cuda:0'")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--aux-imgsz", type=int, default=512, help="0 disables the close-range second pass")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="minimum PairResult.score to render as a confident collision candidate")
    ap.add_argument("--score-every", type=int, default=1,
                    help="call rotation_gate.score_pairs every N tracked frames (1 = every frame)")
    ap.add_argument("--trail-seconds", type=float, default=1.5,
                    help="length of the drawn trajectory trail")
    ap.add_argument("--no-ffmpeg", action="store_true", help="skip the H264 re-encode pass")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# small drawing helpers
# ---------------------------------------------------------------------------

def _label(frame: np.ndarray, x: int, y_top: int, lines: list[str],
          color: tuple[int, int, int]) -> None:
    """Stack label lines upward from (x, y_top), each on its own filled tag."""
    ty = y_top - 4
    for line in reversed(lines):
        (tw, th), _ = cv2.getTextSize(line, FONT, 0.45, 1)
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 6, ty + 2), (0, 0, 0), -1)
        cv2.putText(frame, line, (x + 3, ty - 2), FONT, 0.45, color, 1, cv2.LINE_AA)
        ty -= th + 8


def _draw_trail(frame: np.ndarray, pts: list[tuple[float, float]],
                color: tuple[int, int, int]) -> None:
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
# pass 1: detect + track + continuously score
# ---------------------------------------------------------------------------

def run_tracking(video_path: Path, detector: Detector, tracker: ByteTracker,
                 rg_cfg: RotationGateConfig, score_every: int) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector.warmup()

    frame_records: dict[int, dict[int, dict]] = {}
    track_summary: dict[int, dict] = {}
    best_of: dict[int, float] = {}
    best_pair: PairResult | None = None
    n_score_calls = 0
    n_pairs_ever_considered = 0

    t0 = time.perf_counter()
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t = frame_idx / native_fps

        dets = detector.detect_array(frame)
        tracks = tracker.update(dets, t)

        snap: dict[int, dict] = {}
        for tr in tracks:
            gx, gy = tr.ground_point
            snap[tr.track_id] = {
                "box": [float(v) for v in tr.box],
                "cls": int(tr.cls),
                "speed_px_s": float(tr.speed_px()),
                "ground_point": [float(gx), float(gy)],
            }
            ts = track_summary.setdefault(tr.track_id, {
                "cls": int(tr.cls),
                "first_t": float(tr.first_t),
                "max_speed_px_s": 0.0,
                "n_frames": 0,
            })
            ts["last_t"] = float(tr.last_t)
            ts["max_speed_px_s"] = max(ts["max_speed_px_s"], float(tr.speed_px()))
            ts["n_frames"] += 1
        frame_records[frame_idx] = snap

        if score_every <= 1 or frame_idx % score_every == 0:
            pairs = score_pairs(tracker.all_tracks, rg_cfg, now_t=t, classes=GATED_CLASSES)
            n_score_calls += 1
            n_pairs_ever_considered += len(pairs)
            for p in pairs:
                best_of[p.a] = max(best_of.get(p.a, 0.0), p.score)
                best_of[p.b] = max(best_of.get(p.b, 0.0), p.score)
            if pairs and (best_pair is None or pairs[0].score > best_pair.score):
                best_pair = pairs[0]

    # Final call so the very last window (which the stride above may have
    # skipped) is not silently missed.
    final_t = frame_idx / native_fps if frame_idx >= 0 else 0.0
    pairs = score_pairs(tracker.all_tracks, rg_cfg, now_t=final_t, classes=GATED_CLASSES)
    n_score_calls += 1
    n_pairs_ever_considered += len(pairs)
    for p in pairs:
        best_of[p.a] = max(best_of.get(p.a, 0.0), p.score)
        best_of[p.b] = max(best_of.get(p.b, 0.0), p.score)
    if pairs and (best_pair is None or pairs[0].score > best_pair.score):
        best_pair = pairs[0]

    elapsed = time.perf_counter() - t0
    cap.release()

    return {
        "native_fps": native_fps,
        "w": w,
        "h": h,
        "n_frames": frame_idx + 1,
        "frame_records": frame_records,
        "track_summary": track_summary,
        "best_of": best_of,
        "best_pair": best_pair,
        "n_score_calls": n_score_calls,
        "n_pairs_ever_considered": n_pairs_ever_considered,
        "detect_track_seconds": elapsed,
        "detector_latency": detector.latency_stats(),
    }


# ---------------------------------------------------------------------------
# pass 2: render
# ---------------------------------------------------------------------------

def render(video_path: Path, out_path: Path, run_info: dict, threshold: float,
          trail_seconds: float, use_ffmpeg: bool) -> None:
    native_fps = run_info["native_fps"]
    w, h = run_info["w"], run_info["h"]
    frame_records = run_info["frame_records"]
    best_of = run_info["best_of"]
    best_pair: PairResult | None = run_info["best_pair"]

    confident = best_pair is not None and best_pair.score >= threshold
    pair_ids = set(best_pair.track_ids) if confident else set()

    trail_frames = max(1, int(round(trail_seconds * native_fps)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not re-open video for rendering: {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"_tmp_{out_path.name}")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             native_fps, (w, h))

    fidx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fidx += 1
        snap = frame_records.get(fidx, {})

        for tid, info in snap.items():
            cls_name = ROAD_USER_CLASSES.get(info["cls"], str(info["cls"]))
            sc = best_of.get(tid, 0.0)
            in_pair = tid in pair_ids
            if in_pair:
                col = RED
            elif sc > 0.1:
                col = AMBER
            else:
                col = CLASS_COLOR.get(cls_name, GREY)

            x1, y1, x2, y2 = [int(v) for v in info["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if in_pair else 1)

            pts = []
            for j in range(max(0, fidx - trail_frames), fidx + 1):
                s = frame_records.get(j, {}).get(tid)
                if s is not None:
                    pts.append(s["ground_point"])
            _draw_trail(frame, pts, col)

            speed_px_s = info["speed_px_s"]
            ppm = pixels_per_metre(info["box"], info["cls"])
            speed_kmh_est = (speed_px_s / ppm) * 3.6 if ppm > 1e-9 else 0.0
            lines = [
                f"#{tid} {cls_name}",
                f"{speed_px_s:.0f} px/s (~{speed_kmh_est:.0f} km/h est.)",
            ]
            if sc > 0.0:
                lines.append(f"evidence {sc:.2f}")
            _label(frame, x1, y1, lines, col)

        t = fidx / native_fps
        head = f"{video_path.name}  t={t:.2f}s  tracked={len(snap)}"
        if confident:
            rel = ("n/a" if best_pair.rel_heading_deg is None
                   else f"{best_pair.rel_heading_deg:.0f}")
            head += (f"   TOP #{best_pair.a}<->#{best_pair.b}  score={best_pair.score:.3f}"
                     f"  {best_pair.interaction}  rel={rel}deg  gap={best_pair.gap:.2f}")
        elif best_pair is not None:
            head += f"   no confident collision candidate (best {best_pair.score:.3f} < {threshold:.2f})"
        else:
            head += "   no confident collision candidate (no pair ever came within range)"

        cv2.rectangle(frame, (0, 0), (min(w, 1180), 26), (0, 0, 0), -1)
        cv2.putText(frame, head, (6, 18), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        if confident and abs(t - best_pair.contact_t) <= 0.5:
            (tw, th), _ = cv2.getTextSize("<< CONTACT", FONT, 0.8, 2)
            cx = w // 2 - tw // 2
            cv2.putText(frame, "<< CONTACT", (cx, 60), FONT, 0.8, RED, 2, cv2.LINE_AA)

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
# JSON result
# ---------------------------------------------------------------------------

def build_result_json(video_path: Path, run_info: dict, threshold: float,
                      rg_cfg: RotationGateConfig, detector: Detector,
                      tracker_cfg: dict) -> dict:
    native_fps = run_info["native_fps"]
    best_pair: PairResult | None = run_info["best_pair"]
    confident = best_pair is not None and best_pair.score >= threshold

    tracks_out = {}
    for tid, ts in run_info["track_summary"].items():
        cls_name = ROAD_USER_CLASSES.get(ts["cls"], str(ts["cls"]))
        duration = max(0.0, ts["last_t"] - ts["first_t"])
        tracks_out[str(tid)] = {
            "cls": ts["cls"],
            "cls_name": cls_name,
            "first_t": round(ts["first_t"], 3),
            "last_t": round(ts["last_t"], 3),
            "duration_s": round(duration, 3),
            "n_frames_confirmed": ts["n_frames"],
            "max_speed_px_s": round(ts["max_speed_px_s"], 2),
            "best_rotation_gate_score": round(run_info["best_of"].get(tid, 0.0), 4),
        }

    if best_pair is None:
        collision = {
            "confident": False,
            "reason": ("no candidate pair ever came within "
                       f"{rg_cfg.pair_max_gap} vehicle-diagonals of each other "
                       "(rotation_gate.score_pairs returned nothing)"),
        }
    elif confident:
        collision = {
            "confident": True,
            "threshold": threshold,
            **best_pair.as_dict(),
            "explain": best_pair.explain(),
        }
    else:
        collision = {
            "confident": False,
            "threshold": threshold,
            "reason": (f"top pair score {best_pair.score:.4f} is below the "
                       f"{threshold:.2f} confidence threshold -- reporting as "
                       "an unconfirmed candidate, not a collision"),
            "top_candidate": best_pair.as_dict(),
        }

    return {
        "video": str(video_path),
        "resolution": [run_info["w"], run_info["h"]],
        "native_fps": round(native_fps, 3),
        "frame_count": run_info["n_frames"],
        "duration_s": round(run_info["n_frames"] / native_fps, 3),
        "processing": {
            "detector_weights": str(detector.weights),
            "detector_backend": detector.backend,
            "device": str(detector.device),
            "tracker": tracker_cfg,
            "frames_processed": run_info["n_frames"],
            "detect_track_seconds": round(run_info["detect_track_seconds"], 2),
            "fps_processed": round(run_info["n_frames"] / max(run_info["detect_track_seconds"], 1e-6), 2),
            "detector_latency_ms": detector.latency_stats(),
            "rotation_gate_score_calls": run_info["n_score_calls"],
            "rotation_gate_pairs_ever_considered": run_info["n_pairs_ever_considered"],
            "speed_units_note": ("px/s is the honest reported speed -- these cameras have no "
                                 "image-to-ground calibration (checked config/cameras/*.json). "
                                 "km/h figures are a class-width monocular ESTIMATE (same method "
                                 "rotation_gate uses internally), not a calibrated measurement."),
        },
        "tracks": tracks_out,
        "collision": collision,
        "rotation_gate_config_source": rg_cfg.source,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (REPO / video_path).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (REPO / out_path).resolve()
    if args.json:
        json_path = Path(args.json)
        if not json_path.is_absolute():
            json_path = (REPO / json_path).resolve()
    else:
        stem = out_path.stem
        json_path = out_path.with_name(f"{stem}_result.json")

    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    rg_cfg = RotationGateConfig.from_config(cfg.get("collision", {}))

    tracker_cfg = dict(cfg.get("tracker", {}))
    tracker = ByteTracker(**tracker_cfg)

    aux_imgsz = args.aux_imgsz if args.aux_imgsz > 0 else None
    detector = Detector(weights=args.weights, device=args.device, imgsz=args.imgsz,
                        aux_imgsz=aux_imgsz, conf=args.conf, iou=args.iou)

    print(f"[{video_path.name}] detector={detector.describe()}")
    run_info = run_tracking(video_path, detector, tracker, rg_cfg, args.score_every)
    print(f"[{video_path.name}] tracked {run_info['n_frames']} frames in "
          f"{run_info['detect_track_seconds']:.1f}s "
          f"({run_info['n_frames'] / max(run_info['detect_track_seconds'], 1e-6):.1f} fps), "
          f"{len(run_info['track_summary'])} tracks")

    best_pair = run_info["best_pair"]
    if best_pair is None:
        print(f"[{video_path.name}] rotation_gate: no candidate pair ever found")
    else:
        conf = "CONFIDENT" if best_pair.score >= args.threshold else "below threshold"
        print(f"[{video_path.name}] rotation_gate top pair: #{best_pair.a}<->#{best_pair.b} "
              f"score={best_pair.score:.4f} ({conf}) interaction={best_pair.interaction} "
              f"contact_t={best_pair.contact_t:.2f}s")

    render(video_path, out_path, run_info, args.threshold, args.trail_seconds,
          use_ffmpeg=not args.no_ffmpeg)
    print(f"[{video_path.name}] wrote {out_path}")

    result = build_result_json(video_path, run_info, args.threshold, rg_cfg, detector, tracker_cfg)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{video_path.name}] wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
