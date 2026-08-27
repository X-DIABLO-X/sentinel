"""Why each clip was missed: how far the staged detector got, and what else was there.

Detection is staged -- a converging course is registered, then confirmed -- so a
miss can happen at four different places, and the remedy differs at each. This
reports where each clip stopped, and simultaneously measures three collision
geometries the current test cannot express at all:

* **rear-end**, where the two courses are collinear and therefore never cross;
* **into a stationary vehicle**, where one participant has no velocity and is
  excluded before any pair is formed;
* **single-vehicle**, where there is no second party to pair with.
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from netra.detect import Detector, MOTORISED_CLASSES          # noqa: E402
from netra.footprint import Footprint, separation as fp_sep    # noqa: E402
from netra.pathconflict import PathConflictDetector, ray_conflict, crossing_angle  # noqa: E402
from netra.track import ByteTracker                            # noqa: E402


def analyse(video: Path, det: Detector, long_side: int) -> dict:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    trk, d = ByteTracker(), PathConflictDetector()
    st = collections.Counter()
    max_tracks = 0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps; i += 1
        h, w = fr.shape[:2]; sc = long_side / max(h, w)
        if sc < 1:
            fr = cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        tracks = trk.update(det.detect_array(fr), t)
        max_tracks = max(max_tracks, len(tracks))
        st["confirmed"] += len(d.find(tracks, t))
        st["courses_open"] = max(st["courses_open"], len(d._courses))

        # ---- what the current test cannot see -------------------------
        moving, still = [], []
        for tr in tracks:
            if tr.cls not in MOTORISED_CLASSES:
                continue
            p = d._path(tr, t - d.history_s, t)
            if p is None:
                continue
            v = d._velocity(p)
            span = float(np.linalg.norm(p[-1, 1:] - p[0, 1:]))
            (moving if (v is not None and span >= d.min_speed_px) else still).append((tr, p, v))

        for a in range(len(moving)):
            for b in range(a + 1, len(moving)):
                tr_a, pa, va = moving[a]; tr_b, pb, vb = moving[b]
                ang = crossing_angle([0, 0], va, [0, 0], vb)
                sep = fp_sep(Footprint.from_box(tr_a.box), Footprint.from_box(tr_b.box))
                closing = float(np.dot(np.asarray(pb[-1, 1:]) - np.asarray(pa[-1, 1:]),
                                       np.asarray(va) - np.asarray(vb)))
                if ang < 20 and sep <= 1.6 and closing > 0:
                    st["REAR-END geometry (collinear, closing, touching)"] += 1
        for tr_m, pm, vm in moving:
            for tr_s, ps, _ in still:
                nose = np.asarray(pm[-1, 1:]) + np.asarray(vm) * 1.5
                f_s = Footprint.from_box(tr_s.box)
                if abs(nose[0] - f_s.cx) <= f_s.a * 1.5 and abs(nose[1] - f_s.cy) <= f_s.b * 3.0:
                    st["INTO-STATIONARY geometry (projection enters a stopped car)"] += 1
        for tr, p, v in moving:
            if getattr(tr, "aspect_shift", None) and tr.aspect_shift(2.0) >= 0.45:
                st["SINGLE-VEHICLE cue (large silhouette change)"] += 1
    cap.release()
    return {"fps": round(fps, 1), "max_tracks": max_tracks, **st}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ProblemSet")
    ap.add_argument("--results", default="ProblemSet/Results")
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    det = Detector(weights="yolo26m.pt", imgsz=args.long_side, conf=0.1, device=args.device)
    det.warmup()
    hit = set()
    for f in Path(args.results).glob("*.json"):
        if f.name == "summary.json":
            continue
        if json.loads(f.read_text(encoding="utf-8")).get("events"):
            hit.add(f.stem)

    print(f"{'clip':22s} {'fps':>5s} {'trk':>4s} {'conf':>4s} {'rear':>6s} {'->still':>7s} {'solo':>5s}")
    print("-" * 66)
    for v in sorted(Path(args.dir).glob("*.mp4")):
        if v.stem in hit:
            continue
        r = analyse(v, det, args.long_side)
        print(f"{v.stem[:22]:22s} {r['fps']:>5.1f} {r['max_tracks']:>4d} "
              f"{r.get('confirmed',0):>4d} "
              f"{r.get('REAR-END geometry (collinear, closing, touching)',0):>6d} "
              f"{r.get('INTO-STATIONARY geometry (projection enters a stopped car)',0):>7d} "
              f"{r.get('SINGLE-VEHICLE cue (large silhouette change)',0):>5d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
