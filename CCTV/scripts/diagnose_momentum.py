"""Why did the momentum channel not fire? Report what it actually saw.

The channel raised nothing on fifteen clips that all contain collisions. That is
either a threshold set too high or a precondition never met, and those call for
opposite responses, so this measures which.

For every clip it reports the best values the channel could have seen with all
gates removed, and counts how many candidate pairs each precondition rejected:

* neither vehicle tracked through the impact -- nothing to compute;
* the frame rate below the floor, where an impact is a single frame;
* no footprint contact at the moment of the impulse;
* impulse below driver-control authority;
* momentum changes that did not cancel.

If the best achievable exchange on real collisions is far below the gate, the
gate is wrong. If no pair is ever formed, the gate is irrelevant and tracking is
the binding constraint -- the same conclusion this project reached about
detection recall, which would make it a detector problem wearing a new hat.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.detect import Detector                                    # noqa: E402
from netra.footprint import Footprint, separation                    # noqa: E402
from netra.predict import (ResidualMonitor, momentum_exchange,        # noqa: E402
                           vehicle_mass_kg)
from netra.track import ByteTracker                                  # noqa: E402


def scan(video: Path, det: Detector, long_side: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = ByteTracker()
    mon = ResidualMonitor(fps=fps)
    mon.names = getattr(det, "names", None)
    rej = collections.Counter()
    best = {"impulse": 0.0, "exchange": 0.0, "score": 0.0, "t": None,
            "pair": None, "sep": None}
    max_tracks = 0

    if not mon.available:
        cap.release()
        return {"fps": fps, "available": False, "rej": rej, "best": best,
                "max_tracks": 0}

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        h0, w0 = frame.shape[:2]
        sc = long_side / max(h0, w0)
        if sc < 1:
            frame = cv2.resize(frame, (int(w0 * sc), int(h0 * sc)),
                               interpolation=cv2.INTER_AREA)
        tracks = tracker.update(det.detect_array(frame), t)
        mon.observe(tracks, t)
        max_tracks = max(max_tracks, len(tracks))

        t_eval = t - mon.lag_s
        live = []
        for tr in tracks:
            p = mon.pred.get(int(tr.track_id))
            if p is None:
                continue
            imp = p.impulse_at(t_eval, mon.window_s)
            if imp is None:
                rej["no impulse measurable (short history / occluded)"] += 1
                continue
            box = p.box_at(t_eval)
            if box is None:
                rej["no box at the evaluated moment"] += 1
                continue
            live.append((tr, p, imp, box))

        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                tr_a, pa, ia, box_a = live[a]
                tr_b, pb, ib, box_b = live[b]
                sep = separation(Footprint.from_box(box_a),
                                 Footprint.from_box(box_b))
                ex = momentum_exchange(
                    ia.dv, vehicle_mass_kg(tr_a.cls, mon.names),
                    ib.dv, vehicle_mass_kg(tr_b.cls, mon.names))
                score = float(np.clip(0.45 * min(ia.score, ib.score) + 0.55 * ex,
                                      0.0, 1.0))
                if sep > 1.25:
                    rej["footprints not in contact"] += 1
                elif min(ia.score, ib.score) < 0.85:
                    rej["impulse below driver-control authority"] += 1
                elif ex < 0.90:
                    rej["momentum did not cancel"] += 1
                else:
                    rej["PASSED all gates"] += 1

                # best seen with contact required but the value gates removed
                if sep <= 1.25 and score > best["score"]:
                    best.update(score=score, exchange=ex,
                                impulse=min(ia.score, ib.score),
                                t=round(t_eval, 2), sep=round(sep, 2),
                                pair=(int(tr_a.track_id), int(tr_b.track_id)))
    cap.release()
    return {"fps": fps, "available": True, "rej": rej, "best": best,
            "max_tracks": max_tracks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/problems/Accidents")
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--weights", default="yolo26m.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    vids = sorted(Path(args.dir).glob("*.mp4"),
                  key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    det = Detector(weights=args.weights, imgsz=args.imgsz, conf=0.25,
                   device=args.device)
    det.warmup()

    total = collections.Counter()
    print(f"{'clip':6s} {'fps':>5s} {'trk':>4s} {'best impulse':>13s} "
          f"{'best exch':>10s} {'best score':>11s} {'at':>7s}")
    print("-" * 64)
    for v in vids:
        r = scan(v, det, args.long_side)
        if r is None:
            continue
        if not r["available"]:
            print(f"{v.stem:6s} {r['fps']:>5.1f}    -   below the frame-rate floor")
            continue
        total.update(r["rej"])
        b = r["best"]
        print(f"{v.stem:6s} {r['fps']:>5.1f} {r['max_tracks']:>4d} "
              f"{b['impulse']:>13.3f} {b['exchange']:>10.3f} "
              f"{b['score']:>11.3f} {str(b['t']):>7s}", flush=True)

    print("\nwhy candidate pairs were rejected, across all clips")
    for k, v in total.most_common():
        print(f"  {v:>8d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
