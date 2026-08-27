"""Measure the residual and momentum-exchange distributions on given clips.

Run this on crash-free footage FIRST, and set the operating point from its tail
before ever looking at the accident clips. That ordering is not pedantry; it is
the specific discipline this project has twice failed to keep.

The crash classifier scored 0.954 AUC and turned out to have learned "stopped
equals crashed". Rebuilt with hard negatives it scored 0.993 and turned out to
have learned "this is the camera that has a crash on it". Both numbers were
measured against negatives chosen after the fact, and both flattered a shortcut
rather than exposing it.

A threshold taken from the clean-clip tail cannot flatter anything, because the
clean clips contain no positives to fit to. If the crash values then fail to
separate from that tail, the channel is worthless and the honest response is to
say so and drop it -- not to nudge the threshold until the pictures look right.

Reported per clip and in aggregate:

* **impulse** -- the per-vehicle score for exceeding driver control authority;
* **NIS** -- normalised innovation squared, which should sit near chi-square(2)
  if the motion model is honest about its own uncertainty;
* **momentum exchange** -- computed for every footprint-adjacent pair, which on
  crash-free footage means queues, junctions and lane changes. This is the
  distribution that matters most: it is exactly the confusion the channel claims
  to resolve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.detect import Detector                                   # noqa: E402
from netra.footprint import Footprint, separation                   # noqa: E402
from netra.predict import ResidualMonitor, momentum_exchange, vehicle_mass_kg  # noqa: E402
from netra.track import ByteTracker                                 # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def scan(video: Path, det: Detector, long_side: int, max_seconds: float | None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = ByteTracker()
    mon = ResidualMonitor(fps=fps)
    mon.names = getattr(det, "names", None)

    impulses, nis_vals, exchanges, joint = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if max_seconds and t > max_seconds:
            break
        h0, w0 = frame.shape[:2]
        sc = long_side / max(h0, w0)
        if sc < 1:
            frame = cv2.resize(frame, (int(w0 * sc), int(h0 * sc)),
                               interpolation=cv2.INTER_AREA)
        tracks = tracker.update(det.detect_array(frame), t)
        mon.observe(tracks, t)

        # Everything is asked about a moment already past, exactly as the
        # deployed channel does. Asking about the present is what silently
        # disabled the pair test in the first version.
        t_eval = t - mon.lag_s
        imps = {}
        for tr in tracks:
            pr = mon.pred.get(int(tr.track_id))
            if pr is None:
                continue
            r = pr.residuals[-1] if pr.residuals else None
            if r is not None and r.trusted:
                nis_vals.append(r.nis)
            im = pr.impulse_at(t_eval)
            if im is None:
                continue
            imps[int(tr.track_id)] = (pr, im)
            impulses.append(im.score)

        # every footprint-adjacent pair, whether or not anything fired: on clean
        # footage these are the queues and junctions the channel must not fire on
        ids = list(imps)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pa, ia = imps[ids[a]]
                pb, ib = imps[ids[b]]
                box_a, box_b = pa.box_at(t_eval), pb.box_at(t_eval)
                if box_a is None or box_b is None:
                    continue
                if separation(Footprint.from_box(box_a),
                              Footprint.from_box(box_b)) > 1.25:
                    continue
                ca = next((x.cls for x in tracks if int(x.track_id) == ids[a]), 2)
                cb = next((x.cls for x in tracks if int(x.track_id) == ids[b]), 2)
                ex = momentum_exchange(
                    ia.dv, vehicle_mass_kg(ca, mon.names),
                    ib.dv, vehicle_mass_kg(cb, mon.names))
                exchanges.append(ex)
                # The joint pair is what the channel actually gates on.
                # Marginal tails say nothing about how often BOTH
                # conditions hold at once, which is the false-alarm rate.
                joint.append((min(ia.score, ib.score), ex))
    cap.release()
    return {"fps": fps, "available": mon.available, "impulse": impulses,
            "nis": nis_vals, "exchange": exchanges, "joint": joint,
            "seconds": i / max(fps, 1e-6)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/problems/Traffic")
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--weights", default="yolo26m.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vids = sorted(p for ext in ("*.mp4", "*.webm", "*.avi")
                  for p in Path(args.dir).glob(ext))
    if not vids:
        print(f"no clips in {args.dir}")
        return 1

    det = Detector(weights=args.weights, imgsz=args.imgsz, conf=0.25,
                   device=args.device)
    det.warmup()

    allimp, allnis, allex, alljoint = [], [], [], []
    total_seconds = 0.0
    skipped = []
    print(f"{'clip':30s} {'fps':>6s} {'imp p99':>8s} {'imp max':>8s} "
          f"{'NIS p99':>8s} {'exch p99':>9s} {'exch max':>9s}")
    print("-" * 82)
    for v in vids:
        r = scan(v, det, args.long_side, args.max_seconds)
        if r is None:
            continue
        if not r["available"]:
            skipped.append((v.name, r["fps"]))
            print(f"{v.name[:30]:30s} {r['fps']:>6.1f}   (below the frame-rate floor)")
            continue
        allimp += r["impulse"]
        allnis += r["nis"]
        allex += r["exchange"]
        alljoint += r["joint"]
        total_seconds += r["seconds"]
        print(f"{v.name[:30]:30s} {r['fps']:>6.1f} {pct(r['impulse'],99):>8.3f} "
              f"{(max(r['impulse']) if r['impulse'] else float('nan')):>8.3f} "
              f"{pct(r['nis'],99):>8.2f} {pct(r['exchange'],99):>9.3f} "
              f"{(max(r['exchange']) if r['exchange'] else float('nan')):>9.3f}")

    print("-" * 82)
    print(f"\nsamples: {len(allimp)} trusted residuals, {len(allex)} adjacent pairs")
    if skipped:
        print(f"{len(skipped)} clip(s) below the frame-rate floor, excluded")

    for name, arr in (("impulse", allimp), ("NIS", allnis),
                      ("momentum exchange", allex)):
        if not arr:
            continue
        a = np.asarray(arr, dtype=float)
        print(f"\n{name}: n={len(a)}  median={np.median(a):.3f}  "
              f"p95={pct(a,95):.3f}  p99={pct(a,99):.3f}  "
              f"p99.9={pct(a,99.9):.3f}  max={a.max():.3f}")

    if allimp and allex:
        print("\nsuggested operating point, from the crash-free tail alone:")
        print(f"  min_impulse  >= {pct(allimp, 99.9):.2f}")
        print(f"  min_exchange >= {pct(allex, 99.9):.2f}")
        print("\nThese are upper bounds on what clean traffic produces. If the")
        print("accident clips do not clear them, the channel does not work and")
        print("should be dropped rather than tuned.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"impulse": allimp, "nis": allnis, "exchange": allex,
             "joint": alljoint, "hours": total_seconds / 3600.0,
             "skipped_low_fps": skipped}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
