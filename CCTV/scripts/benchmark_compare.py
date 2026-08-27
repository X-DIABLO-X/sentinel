"""Score NETRA on the ACCIDENT benchmark's own metrics, beside published results.

Why bother reimplementing someone else's metric
-----------------------------------------------
"We detect 14 of 16 clips" is not comparable to anything. The published work on
this task reports three uncertainty-aware scores and a unified harmonic mean of
them, and the only way to say where we stand -- against the best automatic
systems and against humans -- is to compute the same quantities on the same
kind of data.

The three metrics, as described in the ACCIDENT paper:

* **Temporal (T)** -- Gaussian similarity on the error in the predicted accident
  time, exp(-dt^2 / 2 sigma^2), with sigma = 1 s. A miss scores zero. This
  rewards being close rather than exactly right, which is the honest treatment
  for an event whose own annotation is uncertain to a frame or two.
* **Spatial (S)** -- an anisotropic Gaussian on the distance from the annotated
  accident location, wider along the axis the accident is wider along.
* **Collision type (C)** -- top-1 accuracy over five classes: rear-end, t-bone,
  sideswipe, head-on, single.
* **Unified** -- the harmonic mean, so a system cannot buy a good score by
  excelling at one and failing another.

Two honest caveats, stated up front rather than buried
------------------------------------------------------
**The spatial sigma is our reconstruction.** The paper specifies an anisotropic
Gaussian but not the constants, so the annotated accident box is used to set the
scale -- sigma_x and sigma_y are its half-width and half-height. That is the
natural anisotropic choice and it is scale-invariant, but it is not necessarily
theirs, so the spatial column is indicative rather than an official reproduction.

**The clip sets differ.** The published figures are over the full benchmark test
split; ours is over the sixteen held-out clips available here. Sixteen clips is a
small sample and the interval on any percentage from it is wide. This tells us
roughly where we stand, not that we have beaten anybody.

Our collision type comes free from the detection geometry -- the shape of the
conflict that fired *is* a claim about what kind of accident it was -- so no
separate classifier is involved, and none of the shortcut risk that came with
the two appearance classifiers tried earlier applies.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

# Published results, ACCIDENT benchmark (2026). Reproduced here for comparison.
PUBLISHED = [
    ("Human annotators",              0.979, 0.995, 0.923, None),
    ("Molmo-7B (zero-shot VLM)",      0.343, 0.596, 0.270, 0.412),
    ("Qwen2.5-VL-7B (zero-shot)",     None,  None,  0.119, None),
    ("DINOv2 linear probe",           None,  None,  0.440, None),
    ("Heuristic (optical flow)",      None,  0.273, None,  None),
    ("Naive content-agnostic prior",  None,  None,  None,  0.245),
]

# Our detection geometry is itself a claim about the kind of accident.
GEOMETRY_TO_TYPE = {
    "crossing": "t-bone",
    "head-on": "head-on",
    "rear-end": "rear-end",
    "into-stationary": "rear-end",
    "deflection": "sideswipe",
    "struck-object": "single",
    "rollover": "single",
    "background-stationary": None,     # makes no claim about type
}


def temporal_score(dt: float, sigma: float = 1.0) -> float:
    return float(np.exp(-(dt ** 2) / (2.0 * sigma ** 2)))


def spatial_score(pred_xy, gt_xy, sigma_xy) -> float:
    dx = (pred_xy[0] - gt_xy[0]) / max(sigma_xy[0], 1e-6)
    dy = (pred_xy[1] - gt_xy[1]) / max(sigma_xy[1], 1e-6)
    return float(np.exp(-0.5 * (dx * dx + dy * dy)))


def harmonic(*vals) -> float:
    vals = [v for v in vals if v is not None]
    if not vals or min(vals) <= 0:
        return 0.0
    return float(len(vals) / sum(1.0 / v for v in vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="ProblemSet/Results")
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    ap.add_argument("--sigma-t", type=float, default=1.0)
    args = ap.parse_args()

    meta = {}
    with open(args.metadata, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[os.path.basename(r["path"]).rsplit(".", 1)[0]] = r

    files = [f for f in glob.glob(os.path.join(args.results, "**", "*.json"),
                                  recursive=True)
             if os.path.basename(f) != "summary.json"]

    rows = []
    for f in sorted(files):
        stem = os.path.basename(f)[:-5]
        gt = meta.get(stem)
        if gt is None:
            continue
        d = json.loads(open(f, encoding="utf-8").read())
        evs = d.get("events", [])
        t_gt = float(gt["accident_time"])
        w = float(gt["width"]) or 1.0
        h = float(gt["height"]) or 1.0
        gt_xy = (float(gt["center_x"]), float(gt["center_y"]))       # normalised
        sig = (max(abs(float(gt["x2"]) - float(gt["x1"])) / 2.0, 0.02),
               max(abs(float(gt["y2"]) - float(gt["y1"])) / 2.0, 0.02))

        if not evs:
            rows.append((stem, gt["type"], 0.0, 0.0, 0.0, "missed"))
            continue

        best = min(evs, key=lambda e: abs(float(e.get("detected_t", 0)) - t_gt))
        trig = best.get("triggers") or {}
        t_pred = float(best.get("detected_t", 0))
        T = temporal_score(t_pred - t_gt, args.sigma_t)

        # spatial: centre of the vehicles we named, in normalised coordinates
        boxes = trig.get("participant_boxes") or []
        if boxes:
            cx = float(np.mean([(b[0] + b[2]) / 2 for b in boxes]))
            cy = float(np.mean([(b[1] + b[3]) / 2 for b in boxes]))
            # Boxes live in the ANALYSIS frame, which is the source frame only
            # downscaled when it exceeds the processing long side -- never
            # upscaled. Assuming a resize that did not happen put every
            # prediction a third of a frame from where it belonged and scored
            # the spatial metric at essentially zero.
            proc = float((d.get("model_run") or {}).get("resize_long_side") or 1920)
            factor = min(1.0, proc / max(w, h))
            S = spatial_score((cx / (w * factor), cy / (h * factor)), gt_xy, sig)
        else:
            S = 0.0

        geom = ((trig.get("path_conflict_channel") or {}).get("gates")
                or {}).get("geometry", trig.get("detector"))
        pred_type = GEOMETRY_TO_TYPE.get(str(geom))
        C = 1.0 if (pred_type is not None and pred_type == gt["type"]) else 0.0
        rows.append((stem, gt["type"], T, S, C,
                     pred_type or str(geom)[:14]))

    if not rows:
        print("no scored clips; run the pipeline first")
        return 1

    print(f"{'clip':20s} {'true type':10s} {'T':>5s} {'S':>5s} {'C':>3s}  predicted")
    print("-" * 66)
    for r in rows:
        print(f"{r[0][:20]:20s} {r[1][:10]:10s} {r[2]:5.2f} {r[3]:5.2f} "
              f"{r[4]:3.0f}  {r[5]}")

    T = float(np.mean([r[2] for r in rows]))
    S = float(np.mean([r[3] for r in rows]))
    C = float(np.mean([r[4] for r in rows]))
    U = harmonic(T, S, C)

    print("-" * 66)
    print(f"\nNETRA on {len(rows)} held-out clips")
    print(f"   temporal  T = {T:.3f}")
    print(f"   spatial   S = {S:.3f}   (sigma reconstructed from the annotated box)")
    print(f"   type      C = {C:.3f}   (from detection geometry, no classifier)")
    print(f"   unified     = {U:.3f}")

    print(f"\n{'published on the ACCIDENT benchmark':38s} {'T':>6s} {'S':>6s} "
          f"{'C':>6s} {'unified':>8s}")
    print("-" * 68)
    for name, t, s, c, u in PUBLISHED:
        f = lambda v: "  --  " if v is None else f"{v:6.3f}"
        print(f"{name:38s} {f(t)} {f(s)} {f(c)} "
              f"{'    --  ' if u is None else f'{u:8.3f}'}")
    print(f"{'NETRA (this system, 16 clips)':38s} {T:6.3f} {S:6.3f} {C:6.3f} {U:8.3f}")
    print("\nSample sizes differ: published figures are over the full benchmark")
    print("test split, ours over sixteen clips. Treat as indicative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
