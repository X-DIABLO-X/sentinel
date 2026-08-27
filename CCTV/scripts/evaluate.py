"""Clip-level evaluation against the labelled problem set.

The two folders are ground truth:

    data/problems/Accidents/  -- every clip contains a collision  (positives)
    data/problems/Traffic/    -- no clip contains a collision     (negatives)

That makes a real confusion matrix possible, which is worth far more than any
single accuracy number. Two quantities matter operationally and are reported
first-class:

* **Recall** -- of the crashes that happened, how many did we raise?
* **False alarms per hour of clean footage** -- the number that decides whether
  a control room keeps the system switched on or mutes it. A detector with
  perfect recall and an alarm every four minutes is worse than useless, because
  operators stop looking.

Detection latency is measured too, but note the honest caveat: we have clip-level
labels, not frame-level ones, so latency here is measured from clip start rather
than from the true impact instant. It is a lower bound on how quickly we would
respond, not an onset-accuracy metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLLISION = "collision_candidate"


def load_group(results: Path, group: str) -> list[dict]:
    out = []
    d = results / group
    if not d.exists():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def collision_events(rep: dict) -> list[dict]:
    return [e for e in rep.get("events", []) if e.get("type") == COLLISION]


def evaluate(results: Path, positives: str, negatives: str) -> dict:
    pos = load_group(results, positives)
    neg = load_group(results, negatives)

    tp = [r for r in pos if collision_events(r)]
    fn = [r for r in pos if not collision_events(r)]
    fp = [r for r in neg if collision_events(r)]
    tn = [r for r in neg if not collision_events(r)]

    n_tp, n_fn, n_fp, n_tn = len(tp), len(fn), len(fp), len(tn)
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = n_tn / (n_tn + n_fp) if (n_tn + n_fp) else 0.0

    # false alarms per hour on footage known to be clean
    neg_seconds = sum(r.get("stats", {}).get("video_seconds", 0.0) for r in neg)
    neg_alarms = sum(len(collision_events(r)) for r in neg)
    fa_per_hour = neg_alarms / (neg_seconds / 3600.0) if neg_seconds > 0 else 0.0

    # attribution quality: did we name the vehicles, and did we avoid naming
    # the wrong ones on clean footage?
    attribution = {}
    for r in pos + neg:
        for e in collision_events(r):
            a = (e.get("triggers") or {}).get("attribution", "unknown")
            attribution[a] = attribution.get(a, 0) + 1

    delays = sorted(e["detected_t"] for r in tp for e in collision_events(r))
    median_delay = delays[len(delays) // 2] if delays else None

    detectors = {}
    for r in pos + neg:
        for e in collision_events(r):
            d = (e.get("triggers") or {}).get("detector", "unknown")
            detectors[d] = detectors.get(d, 0) + 1

    return {
        "positives_group": positives,
        "negatives_group": negatives,
        "counts": {"TP": n_tp, "FN": n_fn, "FP": n_fp, "TN": n_tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "false_alarms_per_clean_hour": round(fa_per_hour, 2),
        "clean_footage_minutes": round(neg_seconds / 60.0, 1),
        "median_first_alert_t_s": median_delay,
        "attribution": attribution,
        "trigger_channel": detectors,
        "missed_clips": [Path(r.get("source", "?")).name for r in fn],
        "false_alarm_clips": [Path(r.get("source", "?")).name for r in fp],
    }


def render(m: dict) -> str:
    c = m["counts"]
    L = []
    L.append("=" * 62)
    L.append("  CLIP-LEVEL COLLISION DETECTION  (labelled problem set)")
    L.append("=" * 62)
    L.append("")
    L.append(f"  positives : {m['positives_group']}  (every clip contains a crash)")
    L.append(f"  negatives : {m['negatives_group']}  ({m['clean_footage_minutes']} min, no crashes)")
    L.append("")
    L.append("                  flagged     not flagged")
    L.append(f"    crash        {c['TP']:>7d}     {c['FN']:>11d}")
    L.append(f"    no crash     {c['FP']:>7d}     {c['TN']:>11d}")
    L.append("")
    L.append(f"  recall              {m['recall']:.3f}   (crashes we caught)")
    L.append(f"  precision           {m['precision']:.3f}   (flags that were real)")
    L.append(f"  F1                  {m['f1']:.3f}")
    L.append(f"  specificity         {m['specificity']:.3f}   (clean clips left alone)")
    L.append(f"  false alarms/hour   {m['false_alarms_per_clean_hour']:.2f}  on clean footage")
    if m["median_first_alert_t_s"] is not None:
        L.append(f"  median first alert  {m['median_first_alert_t_s']:.2f}s into clip")
    L.append("")
    L.append(f"  trigger channel : {m['trigger_channel']}")
    L.append(f"  attribution     : {m['attribution']}")
    if m["missed_clips"]:
        L.append(f"  missed          : {', '.join(m['missed_clips'])}")
    if m["false_alarm_clips"]:
        L.append(f"  false alarms on : {', '.join(m['false_alarm_clips'])}")
    L.append("=" * 62)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--positives", default="Accidents")
    ap.add_argument("--negatives", default="Traffic")
    ap.add_argument("--out", default="reports/evaluation.json")
    args = ap.parse_args()

    m = evaluate(Path(args.results), args.positives, args.negatives)
    print(render(m))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
