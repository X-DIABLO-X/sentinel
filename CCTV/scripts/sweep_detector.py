"""Detector operating-point sweep.

Answers one question with measurements rather than intuition: for dense,
small-object traffic footage, do you get more usable detections from a *bigger
model* or from a *bigger input*?

The evidence review predicted the answer -- raise resolution before changing
architecture -- and this script exists so that claim is backed by numbers taken
on the actual target hardware rather than quoted from a paper.

Usage:
    python scripts/sweep_detector.py --video data/raw/india_cuttack_linkroad.webm
    python scripts/sweep_detector.py --video ... --device cpu --backend openvino
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netra.detect import Detector  # noqa: E402

THRESHOLDS = (0.10, 0.25, 0.35, 0.50)


def sample_frames(video: str, k: int = 12) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if n > 10:
        for j in np.linspace(0, n - 5, k).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    else:                                    # streams without a frame count
        while len(frames) < k:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {video}")
    return frames


def run(video: str, models: list[str], sizes: list[int],
        device: str, backend: str, samples: int) -> dict:
    frames = sample_frames(video, samples)
    h, w = frames[0].shape[:2]
    rows = []

    for weights in models:
        for imgsz in sizes:
            try:
                det = Detector(weights=weights, imgsz=imgsz, conf=min(THRESHOLDS),
                               device=device, backend=backend)
                det.warmup((imgsz, imgsz))
                counts = {t: 0 for t in THRESHOLDS}
                t0 = time.perf_counter()
                for f in frames:
                    d = det.detect_array(f)
                    for t in THRESHOLDS:
                        counts[t] += int((d[:, 4] >= t).sum())
                elapsed = time.perf_counter() - t0
                k = len(frames)
                rows.append({
                    "weights": weights,
                    "imgsz": imgsz,
                    "backend": det.backend,
                    "device": str(det.device),
                    "det_per_frame": {str(t): round(counts[t] / k, 2) for t in THRESHOLDS},
                    "ms_per_frame": round(elapsed / k * 1000.0, 1),
                    "implied_fps": round(k / max(elapsed, 1e-9), 2),
                    "latency": det.latency_stats(),
                })
                print(f"  {weights:12s} imgsz={imgsz:<5d} "
                      f"conf>=0.35: {counts[0.35] / k:6.1f}/frame   "
                      f"{elapsed / k * 1000:6.1f} ms")
            except Exception as exc:
                rows.append({"weights": weights, "imgsz": imgsz,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"  {weights:12s} imgsz={imgsz:<5d} FAILED: {exc}")

    return {
        "video": video,
        "frame_size": [w, h],
        "samples": len(frames),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--models", nargs="+", default=["yolo12n.pt", "yolo12s.pt"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 960, 1280])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out", default="reports/detector_sweep.json")
    args = ap.parse_args()

    print(f"sweeping {args.video} on device={args.device} backend={args.backend}")
    result = run(args.video, args.models, args.sizes,
                 args.device, args.backend, args.samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")

    ok = [r for r in result["rows"] if "error" not in r]
    if ok:
        best = max(ok, key=lambda r: r["det_per_frame"]["0.35"])
        print(f"most confident detections: {best['weights']} @ {best['imgsz']} "
              f"-> {best['det_per_frame']['0.35']}/frame at {best['ms_per_frame']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
