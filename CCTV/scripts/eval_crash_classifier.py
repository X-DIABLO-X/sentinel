"""Score the crash classifier where it actually failed: on stationary vehicles.

Aggregate AUC is what hid the last bug. The first classifier measured 0.954 and
was worthless, because its validation split had the same flaw as its training
split -- every positive stopped, every negative moving -- so "detect stillness"
scored as well as "detect damage".

This script therefore reports the aggregate number and then ignores it, breaking
the negatives into two groups:

    moving negatives  -- ordinary traffic, the easy case
    STILL negatives   -- parked, queued, waiting at a signal

Only the second group matters. At inference the classifier is shown stationary
vehicles and nothing else, so its score on still negatives *is* its false-alarm
rate in deployment. The separation to look at is positives vs still negatives.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    from ultralytics import YOLO
    w = "runs/classify/models/crash_cls_yolo/weights/best.pt"
    model = YOLO(w)
    names = model.names
    crash_idx = [i for i, n in names.items() if str(n).lower().startswith("crash")][0]

    root = Path("data/crash_cls/val")
    groups = {"crash": [], "moving-negative": [], "STILL-negative": []}
    for p in sorted((root / "crash").glob("*.jpg")):
        groups["crash"].append(p)
    for p in sorted((root / "normal").glob("*.jpg")):
        key = "STILL-negative" if "_still" in p.name else "moving-negative"
        groups[key].append(p)

    scored = {}
    for g, paths in groups.items():
        if not paths:
            continue
        out = []
        for i in range(0, len(paths), 64):
            batch = [str(x) for x in paths[i:i + 64]]
            for r in model.predict(batch, verbose=False, device=0):
                out.append(float(r.probs.data[crash_idx]))
        scored[g] = np.array(out)

    print(f"weights: {w}\n")
    print(f"{'group':18s} {'n':>5s} {'mean p':>8s} {'median':>8s} {'p>0.72':>8s} {'p>0.90':>8s}")
    print("-" * 60)
    for g in ("crash", "moving-negative", "STILL-negative"):
        s = scored.get(g)
        if s is None or not len(s):
            continue
        print(f"{g:18s} {len(s):>5d} {s.mean():>8.3f} {np.median(s):>8.3f} "
              f"{(s > 0.718).mean():>8.1%} {(s > 0.90).mean():>8.1%}")

    pos = scored.get("crash", np.array([]))
    still = scored.get("STILL-negative", np.array([]))
    mov = scored.get("moving-negative", np.array([]))

    if len(pos) and len(still):
        y = np.r_[np.ones(len(pos)), np.zeros(len(still))]
        s = np.r_[pos, still]
        print(f"\nAUC, crash vs STILL negatives : {auc(y, s):.3f}   <- the number that matters")
    if len(pos) and len(mov):
        y = np.r_[np.ones(len(pos)), np.zeros(len(mov))]
        s = np.r_[pos, mov]
        print(f"AUC, crash vs moving negatives: {auc(y, s):.3f}   (the easy case)")

    # operating point chosen against still negatives, since those are the only
    # negatives the deployed system ever presents
    if len(pos) and len(still):
        best = None
        for t in np.arange(0.30, 0.99, 0.01):
            tp = (pos >= t).mean()
            fp = (still >= t).mean()
            if fp <= 0.10 and (best is None or tp > best[1]):
                best = (t, tp, fp)
        if best:
            print(f"\nthreshold for <=10% false alarms on still negatives: "
                  f"{best[0]:.2f}  (recall {best[1]:.1%}, FA {best[2]:.1%})")
        else:
            print("\nno threshold reaches 10% false alarms on still negatives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
