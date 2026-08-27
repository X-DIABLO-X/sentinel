"""Build a crashed-vehicle vs normal-vehicle crop dataset from ACCIDENT.

Why this exists
---------------
Every discriminator we tried at bounding-box level failed to separate a crashed
vehicle from an ordinary stopped one, and the measurements were unambiguous: on
a clean clip versus a real crash, separation was 0.85 vs 0.91 vehicle-lengths
and the conflict score was 0.82 vs 0.795 -- the clean clip scored *higher*.
Proximity, approach angle, deceleration, stop persistence, companion vehicles
and nearby pedestrians all fire on normal traffic.

The thing that actually distinguishes the two is **what the vehicle looks like**
-- crumpled, skewed, resting at an impossible angle. That is a perception
question, and it needs a model trained to see it rather than another threshold.

The ACCIDENT benchmark supplies exactly the supervision required: 2,027 real
CCTV clips, each annotated with the accident frame and the accident bounding
box. Positives are crops of that box after impact. Negatives are crops of
ordinary vehicles taken from the *same clips* before the accident -- same
camera, same weather, same compression -- which forces the classifier to learn
crash appearance rather than camera characteristics.

Disk discipline
---------------
The full corpus is far larger than the free space here, so videos are processed
one at a time: download, extract crops, delete. Peak usage is one video.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.detect import MOTORISED_CLASSES  # noqa: E402

os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
DATASET = "picekl/accident"


def kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def fetch_one(api, remote: str, dest_dir: Path) -> Path | None:
    """Download a single file; the client names it unpredictably, so isolate it."""
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
    """Pad the accident box outward.

    The annotation is tight on the collision itself; a classifier needs some
    surrounding road to judge whether the vehicle is resting abnormally, so the
    crop is deliberately widened.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * (1 + pad)
    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(w, cx + side / 2))
    ny2 = int(min(h, cy + side / 2))
    return nx1, ny1, nx2, ny2


def crop_ok(c, min_px: int = 24) -> bool:
    return c is not None and c.size > 0 and c.shape[0] >= min_px and c.shape[1] >= min_px


def positives_from(video: Path, row, out_dir: Path, n: int, size: int) -> int:
    """Crops of the accident box, from impact onward."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return 0
    af = int(row["accident_frame"])
    # sample from the impact through the following ~3 s: the aftermath is what
    # a stationary-object classifier will actually be shown at inference time
    idx = np.linspace(af, min(total - 1, af + int(3 * fps)), n).astype(int)
    saved = 0
    for i in sorted(set(idx.tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        box = (row["x1"] * w, row["y1"] * h, row["x2"] * w, row["y2"] * h)
        x1, y1, x2, y2 = expand(box, w, h)
        c = f[y1:y2, x1:x2]
        if not crop_ok(c):
            continue
        c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{video.stem}_p{i}.jpg"), c,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
    cap.release()
    return saved


def negatives_from(video: Path, row, detector, out_dir: Path, n: int, size: int) -> int:
    """Crops of ordinary vehicles from the same clip, before the accident.

    Same camera, same lighting, same compression as the positives -- so the
    classifier cannot cheat by learning the camera instead of the crash. The
    accident region is excluded so a pre-impact view of the doomed vehicle does
    not leak in as a negative.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    af = int(row["accident_frame"])
    hi = max(1, af - int(2.0 * fps))
    if hi < 5:
        cap.release()
        return 0
    idx = np.linspace(0, hi, max(2, n // 2)).astype(int)
    saved = 0
    for i in sorted(set(idx.tolist())):
        if saved >= n:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        ax1, ay1 = row["x1"] * w, row["y1"] * h
        ax2, ay2 = row["x2"] * w, row["y2"] * h
        for d in detector.detect_array(f):
            if saved >= n:
                break
            if d[4] < 0.25:
                continue
            # vehicles only: the classifier is asked "is this vehicle crashed?",
            # so a pedestrian in the negative set teaches it nothing useful and
            # makes the decision boundary about object class instead of state
            if int(d[5]) not in MOTORISED_CLASSES:
                continue
            bx1, by1, bx2, by2 = d[:4]
            # skip anything overlapping where the accident will happen
            if not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2):
                continue
            x1, y1, x2, y2 = expand((bx1, by1, bx2, by2), w, h)
            c = f[y1:y2, x1:x2]
            if not crop_ok(c):
                continue
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{video.stem}_n{i}_{saved}.jpg"), c,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
    cap.release()
    return saved


def stratified(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample evenly across collision type, lighting and scene layout."""
    rng = random.Random(seed)
    df = df.copy()
    df["_k"] = (df["type"].astype(str) + "|" + df["day_time"].astype(str)
                + "|" + df["scene_layout"].astype(str))
    groups = {k: g.index.tolist() for k, g in df.groupby("_k")}
    for v in groups.values():
        rng.shuffle(v)
    picked, keys = [], list(groups)
    while len(picked) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return df.loc[picked]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    ap.add_argument("--out", default="data/crash_cls")
    ap.add_argument("--videos", type=int, default=400, help="clips to process")
    ap.add_argument("--pos-per-video", type=int, default=6)
    ap.add_argument("--neg-per-video", type=int, default=6)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--weights", default="yolo26m.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    args = ap.parse_args()

    from netra.detect import Detector

    df = pd.read_csv(args.metadata)
    if args.split != "all":
        df = df[df["split_in_distribution"] == args.split]
    print(f"{len(df)} clips in split '{args.split}'")

    sel = stratified(df, min(args.videos, len(df)))
    print(f"selected {len(sel)} clips, stratified by type/lighting/layout")
    print(sel["type"].value_counts().to_string())

    out = Path(args.out)
    for sub in ("train/crash", "train/normal", "val/crash", "val/normal"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    staging = Path("data/accident/_videos")
    staging.mkdir(parents=True, exist_ok=True)

    det = Detector(weights=args.weights, imgsz=960, conf=0.20, device=args.device)
    det.warmup((960, 960))
    api = kaggle_api()

    n_pos = n_neg = n_clip = 0
    for k, (_, row) in enumerate(sel.iterrows(), 1):
        free_gb = shutil.disk_usage(".").free / 1e9
        if free_gb < args.min_free_gb:
            print(f"stopping: only {free_gb:.1f} GB free")
            break

        # hold out every 5th clip for validation, by clip not by crop, so the
        # same vehicle can never appear in both splits
        part = "val" if k % 5 == 0 else "train"
        vid = fetch_one(api, row["path"], staging)
        if vid is None:
            continue
        try:
            p = positives_from(vid, row, out / part / "crash",
                               args.pos_per_video, args.size)
            n = negatives_from(vid, row, det, out / part / "normal",
                               args.neg_per_video, args.size)
            n_pos += p
            n_neg += n
            n_clip += 1
        except Exception as exc:
            print(f"  {Path(row['path']).name}: {type(exc).__name__} {exc}")
        finally:
            vid.unlink(missing_ok=True)

        if k % 10 == 0 or k == len(sel):
            print(f"[{k}/{len(sel)}] clips={n_clip} crash={n_pos} normal={n_neg} "
                  f"free={shutil.disk_usage('.').free / 1e9:.1f}GB")

    shutil.rmtree(staging, ignore_errors=True)
    meta = {"clips_used": n_clip, "crash_crops": n_pos, "normal_crops": n_neg,
            "crop_size": args.size, "source": DATASET, "split": args.split}
    (out / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n{json.dumps(meta, indent=2)}")
    print(f"\ntrain a classifier with:\n"
          f"  yolo classify train data={out} model=yolo11n-cls.pt epochs=30 imgsz={args.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
