"""Harvest stationary undamaged vehicles from the ACCIDENT clips themselves.

Why a second harvest was needed
-------------------------------
The first hard-negative pass took stationary vehicles from crash-free traffic
clips. It looked like a complete success -- crash-vs-still-negative AUC 0.993,
zero still negatives above threshold -- and in deployment it barely moved.
Re-scoring the vehicles review had called wrong, inside the accident clips, the
parked truck in clip 1 went from 0.993 to 0.995. Up.

The reason is that every negative came from a camera no positive ever appeared
on, so "which scene is this" separates the classes just as well as "is this
vehicle wrecked", and gradient descent has no reason to prefer the harder
feature. The validation split shared the confound, so it certified the shortcut
instead of catching it. This is the same mistake as the first dataset -- a
nuisance variable correlated with the label -- wearing different clothes.

The fix is negatives that differ from the positives in nothing but the label:
same clip, same camera, same weather, same hour, same compression. Every
ACCIDENT clip has exactly that, because a road with a collision on it is also a
road with ordinary vehicles parked and queueing along it. The annotation gives
the accident box, so anything stationary that is demonstrably *not* that box is
a labelled negative drawn from inside the positive distribution.

Selection rules
---------------
* tracker-confirmed stationary, not a single-frame guess;
* no overlap with the annotated accident box, with a wide margin, so a
  participant is never harvested as a negative;
* sampled across the whole clip, before the accident frame as well as after, so
  "late in the clip" cannot become the next shortcut;
* split by clip, under the same rule as the positives, so the same vehicle
  cannot appear on both sides of validation.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.detect import MOTORISED_CLASSES, Detector   # noqa: E402
from netra.geometry import iou                          # noqa: E402
from netra.track import ByteTracker                     # noqa: E402

DATASET = "picekl/accident"


def kaggle_api():
    import os
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def fetch_one(api, remote: str, dest_dir: Path):
    """Download one clip; the client names its output unpredictably, so isolate it."""
    tmp = dest_dir / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_file(DATASET, remote, path=str(tmp), force=True, quiet=True)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    got = sorted(tmp.glob("*"))
    if not got:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    src = got[0]
    if src.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        src.unlink()
        inner = [p for p in tmp.rglob("*") if p.is_file()]
        if not inner:
            shutil.rmtree(tmp, ignore_errors=True)
            return None
        src = inner[0]
    out = dest_dir / Path(remote).name
    shutil.move(str(src), str(out))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def expand(box, w, h, pad: float = 0.35):
    """Pad outward, matching exactly how the positive crops were cut."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * (1 + pad)
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))


def harvest(video: Path, acc_box, det: Detector, out_dir: Path, stem: str,
            size: int, per_video: int, long_side: int,
            stop_speed: float = 4.0, min_still_s: float = 1.2) -> int:
    """Crop stationary vehicles that are provably not the accident."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / 5.0)))
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
        else:
            sc = 1.0
        h, w = frame.shape[:2]
        t = (i - 1) / fps
        # the accident box, brought into the resized frame's coordinates
        ab = None if acc_box is None else [v * sc for v in acc_box]

        for tr in tracker.update(det.detect_array(frame), t):
            if saved >= per_video:
                break
            if tr.cls not in MOTORISED_CLASSES or tr.hits < 6:
                continue
            if tr.speed_px(1.0) >= stop_speed:
                continue
            if getattr(tr, "stationary_since", None) is None:
                tr.stationary_since = t
            if (t - tr.stationary_since) < min_still_s:
                continue
            # Never harvest anything touching the collision. A participant
            # mislabelled as a negative is far worse than one negative fewer.
            if ab is not None and iou(tr.box, ab) > 0.02:
                continue
            x1, y1, x2, y2 = expand(tr.box, w, h)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{stem}_samecam{i}_{tr.track_id}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
    cap.release()
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    ap.add_argument("--out", default="data/crash_cls")
    ap.add_argument("--videos", type=int, default=260)
    ap.add_argument("--per-video", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--weights", default="yolo26m.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    with open(args.metadata, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("path")]
    rows = rows[: args.videos]
    if not rows:
        print("no metadata rows")
        return 1

    out = Path(args.out)
    tr_dir, va_dir = out / "train" / "normal", out / "val" / "normal"
    tr_dir.mkdir(parents=True, exist_ok=True)
    va_dir.mkdir(parents=True, exist_ok=True)
    staging = Path("data/accident/_videos")
    staging.mkdir(parents=True, exist_ok=True)

    api = kaggle_api()
    det = Detector(weights=args.weights, imgsz=960, conf=0.25, device=args.device)
    det.warmup((960, 960))
    print(f"harvesting same-camera still negatives from {len(rows)} accident clips")

    total = 0
    for k, r in enumerate(rows, 1):
        remote = r["path"]
        stem = Path(remote).stem
        vid = fetch_one(api, remote, staging)
        if vid is None:
            print(f"  [{k}/{len(rows)}] {stem:38s} download failed", flush=True)
            continue
        try:
            acc = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            if not all(np.isfinite(acc)) or acc[2] <= acc[0]:
                acc = None
        except Exception:
            acc = None
        dest = va_dir if k % 5 == 0 else tr_dir      # same split rule as the positives
        n = 0
        try:
            n = harvest(vid, acc, det, dest, stem, args.size,
                        args.per_video, args.long_side)
        except Exception as exc:
            print(f"  {stem}: {type(exc).__name__} {exc}", flush=True)
        finally:
            vid.unlink(missing_ok=True)              # 32 GB free: keep nothing
        total += n
        print(f"  [{k}/{len(rows)}] {stem:38s} +{n}  (total {total})", flush=True)

    import glob
    print(f"\nadded {total} same-camera still negatives")
    print("dataset now:",
          len(glob.glob(str(out / "*" / "crash" / "*.jpg"))), "crash /",
          len(glob.glob(str(out / "*" / "normal" / "*.jpg"))), "normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
