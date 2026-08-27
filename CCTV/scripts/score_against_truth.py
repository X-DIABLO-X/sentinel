"""Score detections against the ACCIDENT dataset's annotated accident times.

Until now every judgement here has been "did the clip fire at all", which is a
weak question: a clip can fire on a stopped bus twenty seconds after the
collision and still be counted. The dataset annotates the moment of each
accident, so the real questions are answerable -- did we fire, did we fire at
the right moment, and if not, how wrong were we.

Temporal error is reported signed, because early and late are different
failures. Firing early means something that was not the accident satisfied the
gates. Firing late means the accident itself was missed and something later was
picked up instead. They call for opposite fixes.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / max(ua, 1e-9))


def named_right_vehicle(trig, gt, proc=1920.0) -> bool:
    """Did any vehicle we boxed overlap the annotated accident box?

    "Did the clip fire" says nothing about whether we accused the right car,
    which is the thing an operator actually acts on. The dataset annotates the
    accident's own box, so this is directly checkable rather than a matter of
    opinion.
    """
    boxes = trig.get("participant_boxes") or []
    if not boxes:
        return False
    w, h = float(gt["width"]), float(gt["height"])
    factor = min(1.0, proc / max(w, h))
    gt_box = [float(gt["x1"]) * w, float(gt["y1"]) * h,
              float(gt["x2"]) * w, float(gt["y2"]) * h]
    for b in boxes:
        ours = [b[0] / factor, b[1] / factor, b[2] / factor, b[3] / factor]
        if _iou(ours, gt_box) > 0.0:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="ProblemSet/Results")
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--time-field", choices=("started_t", "detected_t"),
                    default="detected_t",
                    help="event timestamp to score; detected_t preserves the "
                         "published project benchmark, while started_t audits "
                         "the separate onset-recovery estimate")
    ap.add_argument("--times", default="data/labels/accident_times.json",
                    help="team-reviewed accident times; these take precedence "
                         "over the dataset annotation where both exist")
    args = ap.parse_args()

    meta = {}
    with open(args.metadata, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[os.path.basename(r["path"]).rsplit(".", 1)[0]] = r

    # The team watched every clip and recorded the accident second. Those
    # readings agree with the dataset annotations to a median of 0.27 s, which
    # is itself a useful measure of how precisely an accident time can be
    # defined; where they differ, the reading from someone who watched the clip
    # for this purpose is the one used.
    team = {}
    if os.path.exists(args.times):
        team = {k: v for k, v in json.loads(
            open(args.times, encoding="utf-8").read()).items()
            if not k.startswith("_")}

    files = [f for f in glob.glob(os.path.join(args.results, "**", "*.json"),
                                  recursive=True)
             if os.path.basename(f) != "summary.json"]
    rows, errs = [], []
    for f in sorted(files):
        stem = os.path.basename(f)[:-5]
        gt = meta.get(stem)
        if gt is None:
            continue
        d = json.load(open(f))
        evs = d.get("events", [])
        t_gt = float(team.get(stem, gt["accident_time"]))
        if not evs:
            rows.append((stem, gt["type"], t_gt, None, None, "MISSED", "-", False))
            continue
        # the event closest to the truth, so a clip is not penalised for also
        # reporting something else
        best = min(evs, key=lambda e: abs(float(e.get(args.time_field, 0)) - t_gt))
        t_det = float(best.get(args.time_field, 0))
        err = t_det - t_gt
        g = ((best["triggers"].get("path_conflict_channel") or {}).get("gates")
             or {}).get("geometry", best["triggers"].get("detector", "-"))
        verdict = "HIT" if abs(err) <= args.tolerance else (
            "early" if err < 0 else "late")
        rows.append((stem, gt["type"], t_gt, t_det, err, verdict, str(g)[:16],
                     named_right_vehicle(best["triggers"], gt)))
        errs.append(err)

    print(f"timestamp scored: {args.time_field}")
    print(f"{'clip':20s} {'type':11s} {'truth':>6s} {'ours':>6s} {'error':>7s}  "
          f"{'verdict':8s} {'car':>4s} geometry")
    print("-" * 92)
    for r in rows:
        t_det = "-" if r[3] is None else f"{r[3]:6.2f}"
        err = "-" if r[4] is None else f"{r[4]:+7.2f}"
        car = "OK" if r[7] else "--"
        print(f"{r[0][:20]:20s} {r[1][:11]:11s} {r[2]:6.2f} {t_det:>6s} {err:>7s}  "
              f"{r[5]:8s} {car:>4s} {r[6]}")
    print("-" * 92)

    n = len(rows)
    hits = sum(1 for r in rows if r[5] == "HIT")
    fired = sum(1 for r in rows if r[3] is not None)
    print(f"clips                      : {n}")
    print(f"fired at all               : {fired}/{n}")
    right_car = sum(1 for r in rows if r[7])
    both = sum(1 for r in rows if r[5] == "HIT" and r[7])
    print(f"within +/-{args.tolerance:.1f}s of the accident : {hits}/{n}")
    print(f"named a vehicle in the accident   : {right_car}/{n}")
    print(f"RIGHT TIME AND RIGHT VEHICLE      : {both}/{n}"
          f"   <- the number that matters")
    if errs:
        a = np.abs(errs)
        print(f"absolute temporal error    : median {np.median(a):.2f}s  "
              f"mean {a.mean():.2f}s")
        print(f"fired early / late         : {sum(1 for e in errs if e < -args.tolerance)}"
              f" / {sum(1 for e in errs if e > args.tolerance)}")

    # what the accident types are, so misses can be read by category
    by_type = {}
    for r in rows:
        by_type.setdefault(r[1], [0, 0])
        by_type[r[1]][1] += 1
        if r[5] == "HIT":
            by_type[r[1]][0] += 1
    print("\nby accident type (hit / total):")
    for k, (h, t) in sorted(by_type.items()):
        print(f"   {k:12s} {h}/{t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
