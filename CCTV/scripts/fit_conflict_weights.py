"""Fit the collision score from labelled clips, instead of choosing thresholds by hand.

Why fit rather than tune
------------------------
Every gate in this system was set by reasoning about physics, and most of them
were right in principle and wrong in practice. The clearest example: requiring
the struck vehicle to be visibly disturbed is exactly what Newton's third law
says should happen, and as a hard gate it removed more than half the recall,
because mass ratios are large and a struck lorry barely moves. Meanwhile a false
alarm on a crash-free clip scored 0.993 -- higher than every real accident -- so
no threshold on the rule-based score could have separated them either.

Tuning one gate at a time also cannot see interactions. A sharp turn is weak
evidence alone and strong evidence when the other vehicle was also disturbed and
the footprints overlapped; a hand-set threshold on the turn cannot express that.

So the weights are fitted. What is being fitted matters as much as that it is:

* the inputs are **physical measurements** -- turn angle, lateral g, footprint
  separation, post-encroachment time, deceleration, how much each vehicle was
  disturbed. Not pixels. Two appearance classifiers were tried earlier in this
  project and both collapsed into shortcuts -- "stopped equals crashed", then
  "this is the camera with the crash on it" -- because pixels carry the scene
  along with the subject. A turn angle carries nothing but the turn.
* the labels come from **clip-level truth**: sixteen clips with an annotated
  accident second, sixteen with no accident at all. A candidate on an accident
  clip within a second of the annotated time is positive; every candidate on a
  crash-free clip is negative, and so is every candidate far from the truth on
  an accident clip.
* generalisation is measured by **leave-one-clip-out** cross-validation, never
  on the training fit. With thirty-two clips the fitted numbers would otherwise
  say nothing at all.

The result is a small logistic model over a dozen interpretable numbers, which
can be read, argued with, and overridden -- the hard physical constraints stay
in front of it and no weight can overturn them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

FEATURES = ["score_rule", "turn_deg", "lateral_g", "footprint_sep", "pet_s",
            "decel_m_s2", "partner_dev_deg", "partner_drop", "own_dev_deg",
            "own_drop", "crossing_angle_deg", "aspect_ratio_change", "is_solo",
            "n_participants", "identity_changed", "geom_crossing",
            "geom_rear_end", "geom_into_stationary", "geom_head_on",
            "geom_deflection", "geom_rollover", "geom_struck_object",
            "geom_sudden_stop", "geom_track_lost"]


def load_truth(metadata: str, extra: str | None) -> dict:
    truth = {}
    if os.path.exists(metadata):
        with open(metadata, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                truth[os.path.basename(r["path"]).rsplit(".", 1)[0]] = {
                    "t": float(r["accident_time"]),
                    "width": float(r["width"]), "height": float(r["height"]),
                    "box": [float(r["x1"]), float(r["y1"]),
                            float(r["x2"]), float(r["y2"])],
                }
    if extra and os.path.exists(extra):
        for k, v in json.loads(open(extra, encoding="utf-8").read()).items():
            if k.startswith("_"):      # provenance notes, not labels
                continue
            if k in truth:
                truth[k]["t"] = float(v)
    return truth


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / max(union, 1e-9))


def build(dumps: list[str], truth: dict, tol: float,
          min_rule_score: float | None = None):
    X, y, groups, meta = [], [], [], []
    for path in dumps:
        stem = os.path.basename(path).replace("_candidates.json", "")
        blob = json.loads(open(path, encoding="utf-8").read())
        cands = blob.get("candidates", [])
        gt = truth.get(stem)
        crash_free = blob.get("crash_free", gt is None)
        video = blob.get("video") or {}
        width = float(video.get("width") or (gt or {}).get("width") or 1.0)
        height = float(video.get("height") or (gt or {}).get("height") or 1.0)
        proc = float(blob.get("analysis_long_side") or max(width, height))
        factor = min(1.0, proc / max(width, height))
        gt_box = None
        if gt is not None:
            x1, y1, x2, y2 = gt["box"]
            gt_box = [x1 * width, y1 * height, x2 * width, y2 * height]
        for c in cands:
            if (min_rule_score is not None and
                    float(c.get("score_rule", 0.0)) < min_rule_score):
                continue
            if crash_free:
                label = 0
                spatial_hit = False
            elif gt is None:
                continue
            else:
                boxes = [[v / max(factor, 1e-9) for v in b]
                         for b in (c.get("boxes") or [])]
                best_iou = max((_iou(b, gt_box) for b in boxes), default=0.0)
                spatial_hit = best_iou > 0.0
                label = int(abs(float(c["t"]) - gt["t"]) <= tol and spatial_hit)
            if crash_free or gt is None:
                boxes = []
                best_iou = 0.0
            X.append([float(c.get(f, 0.0)) for f in FEATURES])
            y.append(label)
            groups.append(stem)
            meta.append({"clip": stem, "t": round(float(c["t"]), 2),
                         "geometry": c.get("geometry"),
                         "spatial_hit": bool(spatial_hit),
                         "best_iou": round(float(best_iou), 4),
                         "boxes_source": boxes,
                         "gt_t": None if gt is None else float(gt["t"]),
                         "rule_score": float(c.get("score_rule", 0.0))})
    return np.asarray(X, float), np.asarray(y, int), np.asarray(groups), meta


def standardise(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd, mu, sd


def fit_logistic(X, y, l2: float = 1.0, iters: int = 700, lr: float = 0.12):
    """Plain gradient descent, so there is no dependency to install or explain."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    pos = max(1, int(y.sum()))
    neg = max(1, int((1 - y).sum()))
    # positives are rare; weight them so the fit does not simply predict "no"
    sw = np.where(y == 1, neg / pos, 1.0)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * sw
        w -= lr * ((X.T @ g) / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def bounded_training_indices(y, groups, meta, mask, max_neg_per_clip=120):
    """Keep all positives and the hardest rule-scored negatives per clip."""
    chosen = []
    for clip in sorted(set(groups[mask])):
        idx = np.flatnonzero(mask & (groups == clip))
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        neg = sorted(neg, key=lambda i: meta[i]["rule_score"], reverse=True)
        chosen.extend(pos.tolist())
        chosen.extend(neg[:max_neg_per_clip])
    return np.asarray(sorted(set(chosen)), dtype=int)


def predict(X, w, b):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", nargs="+",
                    default=["ProblemSet/Results/**/*_candidates.json",
                             "results/Traffic/**/*_candidates.json"])
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    ap.add_argument("--extra-truth", default="data/labels/accident_times.json")
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--min-rule-score", type=float, default=None,
                    help="only fit/rank candidates that the current hard gate "
                         "can emit (deployment default is 0.55)")
    ap.add_argument("--out", default="models/conflict_weights.json")
    ap.add_argument("--cv-report", default=None,
                    help="optional JSON report of held-out per-clip rankings")
    args = ap.parse_args()

    files = []
    for pat in args.dumps:
        files += glob.glob(pat, recursive=True)
    if not files:
        print("no candidate dumps found; run with --dump-candidates first")
        return 1

    truth = load_truth(args.metadata, args.extra_truth)
    X, y, groups, meta = build(sorted(files), truth, args.tolerance,
                               args.min_rule_score)
    if len(X) == 0:
        print("no candidates to fit on")
        return 1

    print(f"candidates: {len(X)}   positive: {int(y.sum())}   "
          f"negative: {int((1 - y).sum())}   clips: {len(set(groups))}")
    truth_in_dumps = sorted({os.path.basename(p).replace("_candidates.json", "")
                             for p in files} & set(truth))
    positive_clips = sorted(set(groups[y == 1]))
    print(f"accident clips with a usable positive candidate: "
          f"{len(positive_clips)}/{len(truth_in_dumps)}")
    missing_positive = sorted(set(truth_in_dumps) - set(positive_clips))
    if missing_positive:
        print("no usable positive candidate:", ", ".join(missing_positive))
    if y.sum() == 0:
        print("no positives -- nothing to learn from")
        return 1

    # ---- leave one accident clip out --------------------------------------
    # Crash-free clips remain training negatives in every fold; the held-out
    # unit is the accident camera whose answer must not influence its score.
    accident_clips = sorted({g for g, lab in zip(groups, y) if lab == 1})
    oof = np.zeros(len(y))
    evaluated = np.zeros(len(y), dtype=bool)
    for c in accident_clips:
        te = groups == c
        tr_mask = ~te
        tr = bounded_training_indices(y, groups, meta, tr_mask)
        if len(tr) == 0 or y[tr].sum() == 0:
            continue
        Xtr, mu_fold, sd_fold = standardise(X[tr])
        w, b = fit_logistic(Xtr, y[tr])
        oof[te] = predict((X[te] - mu_fold) / sd_fold, w, b)
        evaluated[te] = True

    top1, top3, cv_rows = [], [], []
    for c in accident_clips:
        idx = np.flatnonzero((groups == c) & evaluated)
        ranked = sorted(idx, key=lambda i: oof[i], reverse=True)
        top1.append(bool(ranked and y[ranked[0]] == 1))
        top3.append(any(y[i] == 1 for i in ranked[:3]))
        best = ranked[0] if ranked else None
        positive_rank = next((rank + 1 for rank, i in enumerate(ranked)
                              if y[i] == 1), None)
        row = {
            "clip": c,
            "correct_top1": bool(ranked and y[ranked[0]] == 1),
            "correct_top3": any(y[i] == 1 for i in ranked[:3]),
            "positive_rank": positive_rank,
            "selected": None if best is None else {
                **meta[best], "learned_score": round(float(oof[best]), 6),
                "is_positive": bool(y[best]),
            },
        }
        cv_rows.append(row)

    print("\nleave-one-accident-clip-out spatial + temporal ranking")
    print(f"   correct candidate ranked first  : {sum(top1)}/{len(top1)}")
    print(f"   correct candidate in top three  : {sum(top3)}/{len(top3)}")
    for row in cv_rows:
        chosen = row["selected"] or {}
        marker = "OK" if row["correct_top1"] else "MISS"
        print(f"   {marker:4s} {row['clip']:24s} "
              f"t={chosen.get('t', '-')} geometry={chosen.get('geometry', '-')} "
              f"positive-rank={row['positive_rank']}")

    # ---- final fit on everything, for deployment ----
    final_idx = bounded_training_indices(y, groups, meta,
                                         np.ones(len(y), dtype=bool))
    Xs, mu, sd = standardise(X[final_idx])
    w, b = fit_logistic(Xs, y[final_idx])
    order = np.argsort(-np.abs(w))
    print("\nfitted weights (standardised, largest first)")
    for i in order:
        print(f"   {FEATURES[i]:20s} {w[i]:+7.3f}")

    out = {
        "features": FEATURES,
        "mean": mu.tolist(), "std": sd.tolist(),
        "weights": w.tolist(), "bias": float(b),
        "threshold": None,
        "fitted_on": {"clips": len(set(groups)),
                      "candidates_available": int(len(X)),
                      "candidates_used": int(len(final_idx)),
                      "positives": int(y.sum())},
        "candidate_filter": {"min_rule_score": args.min_rule_score,
                             "accident_clips_present": len(truth_in_dumps),
                             "accident_clips_with_positive": len(positive_clips),
                             "accident_clips_without_positive": missing_positive},
        "cv": {"scheme": "leave-one-accident-clip-out",
               "label": "within time tolerance AND overlaps accident region",
               "top1_spatiotemporal": int(sum(top1)),
               "top3_spatiotemporal": int(sum(top3)),
               "accident_clips_total": len(accident_clips)},
        "note": ("Fitted over physical measurements only -- angles, forces, "
                 "separations, speed changes, and geometry type. Ground-truth "
                 "location is used only to create the training label and is not "
                 "an inference feature. The hard physical constraints run in "
                 "front of this and no weight can create a candidate."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    if args.cv_report:
        report_dir = os.path.dirname(args.cv_report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(args.cv_report, "w", encoding="utf-8") as fh:
            json.dump({"summary": out["cv"], "clips": cv_rows}, fh, indent=2)
        print(f"wrote {args.cv_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
