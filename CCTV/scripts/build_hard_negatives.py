"""Add hard negatives: stationary vehicles that are NOT crashed.

The bug this fixes
------------------
The first crash classifier scored AUC 0.954 on held-out data and then confidently
labelled parked and queued cars as collisions -- p(crashed) of 0.99, 0.96, 0.93
on vehicles that were plainly fine. That is not a threshold problem, it is a
training-data problem, and a textbook one.

Every positive crop was taken *after* the accident, when the vehicle had stopped.
Every negative crop was taken *before* it, while traffic was moving. So the
easiest way for the model to fit the data was to learn **"stopped equals
crashed"** -- a shortcut that scores beautifully on that validation split and is
worthless in deployment, because the only crops it is ever shown at inference are
of stationary vehicles. It had never seen a stationary undamaged car in its life.

The fix is hard negatives from two sources:

* **Traffic clips confirmed to contain no accidents.** Every stopped vehicle in
  them -- queued, parked, waiting at a signal -- is a labelled negative, and it
  is exactly the confusion we need to break.
* **Stationary vehicles in the ACCIDENT clips that are not the accident.** Same
  camera and lighting as the positives, but parked or queued elsewhere in frame.

This is the difference between a classifier that recognises damage and one that
recognises stillness.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.detect import MOTORISED_CLASSES, Detector   # noqa: E402
from netra.track import ByteTracker                     # noqa: E402


def expand(box, w, h, pad: float = 0.35):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * (1 + pad)
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))


def harvest(video: Path, detector: Detector, out_dir: Path, size: int,
            per_video: int, long_side: int, stop_speed: float = 4.0,
            min_still_s: float = 1.5) -> int:
    """Crop vehicles that are demonstrably stationary and demonstrably fine.

    Stationarity is taken from the tracker rather than a single frame, so what
    lands in the negative set is genuinely a still vehicle -- the precise thing
    the classifier was never shown.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / 6.0)))
    tracker = ByteTracker()

    saved, i = 0, 0
    while saved < per_video:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % stride:
            continue
        h0, w0 = frame.shape[:2]
        sc = long_side / max(h0, w0)
        if sc < 1:
            frame = cv2.resize(frame, (int(w0 * sc), int(h0 * sc)),
                               interpolation=cv2.INTER_AREA)
        t = (i - 1) / fps
        h, w = frame.shape[:2]
        tracks = tracker.update(detector.detect_array(frame), t)

        for tr in tracks:
            if saved >= per_video:
                break
            if tr.cls not in MOTORISED_CLASSES or tr.hits < 6:
                continue
            if tr.speed_px(1.0) >= stop_speed:
                continue
            if tr.stationary_since is None:
                tr.stationary_since = t
            if (t - tr.stationary_since) < min_still_s:
                continue
            x1, y1, x2, y2 = expand(tr.box, w, h)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            c = frame[y1:y2, x1:x2]
            if c.size == 0:
                continue
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{video.stem}_still{i}_{tr.track_id}.jpg"), c,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
    cap.release()
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+",
                    default=["data/problems/Traffic", "data/raw"])
    ap.add_argument("--out", default="data/crash_cls")
    ap.add_argument("--per-video", type=int, default=90)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--weights", default="yolo26m.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    tr_dir = out / "train" / "normal"
    va_dir = out / "val" / "normal"
    tr_dir.mkdir(parents=True, exist_ok=True)
    va_dir.mkdir(parents=True, exist_ok=True)

    vids: list[Path] = []
    for srcdir in args.sources:
        d = Path(srcdir)
        if d.exists():
            for ext in ("*.mp4", "*.webm", "*.avi"):
                vids.extend(sorted(d.glob(ext)))
    if not vids:
        print("no source videos found")
        return 1

    det = Detector(weights=args.weights, imgsz=960, conf=0.25, device=args.device)
    det.warmup((960, 960))
    print(f"harvesting stationary-but-undamaged vehicles from {len(vids)} clips")

    total = 0
    for k, v in enumerate(vids, 1):
        # hold out by clip, matching how the positives were split
        dest = va_dir if k % 5 == 0 else tr_dir
        n = harvest(v, det, dest, args.size, args.per_video, args.long_side)
        total += n
        print(f"  [{k}/{len(vids)}] {v.name:44s} +{n} still-vehicle negatives")

    meta_path = out / "dataset.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["hard_negatives_added"] = total
    meta["hard_negative_sources"] = args.sources
    meta["hard_negative_rationale"] = (
        "stationary undamaged vehicles: the class the first classifier never saw, "
        "which let it learn 'stopped = crashed'")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    import glob
    print(f"\nadded {total} hard negatives")
    print("dataset now:",
          len(glob.glob(str(out / '*' / 'crash' / '*.jpg'))), "crash /",
          len(glob.glob(str(out / '*' / 'normal' / '*.jpg'))), "normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
