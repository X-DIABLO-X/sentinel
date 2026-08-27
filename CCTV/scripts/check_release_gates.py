"""Fail closed when a deadline iteration regresses accepted outputs.

Gates:
* every user-approved ProblemSet anchor remains right-time/right-vehicle;
* no ProblemSet clip reports more than one collision;
* the confirmed crash-free ElciaDataSet reports zero collisions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLLISION = "collision_candidate"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = ((a[2]-a[0])*(a[3]-a[1]) +
             (b[2]-b[0])*(b[3]-b[1]) - inter)
    return float(inter / max(union, 1e-9))


def load_metadata(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as fh:
        return {Path(r["path"]).stem: r for r in csv.DictReader(fh)}


def participant_hit(event: dict, gt: dict, analysis_long_side=1920.0) -> bool:
    boxes = (event.get("triggers") or {}).get("participant_boxes") or []
    width, height = float(gt["width"]), float(gt["height"])
    factor = min(1.0, analysis_long_side / max(width, height))
    gt_box = [float(gt["x1"])*width, float(gt["y1"])*height,
              float(gt["x2"])*width, float(gt["y2"])*height]
    return any(iou([v/factor for v in box], gt_box) > 0 for box in boxes)


def reports(path: Path):
    out = []
    for p in sorted(path.glob("*.json")):
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Identify reports by contract, not by a growing filename denylist.
        if blob.get("camera_id") and isinstance(blob.get("events"), list):
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-results",
                    default="ProblemSet/Results_localization_baseline")
    ap.add_argument("--traffic-results", default=None)
    ap.add_argument("--traffic-source", default="results/ElciaDataSet")
    ap.add_argument("--anchors", default="data/labels/regression_anchors.json")
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    args = ap.parse_args()

    anchors = {k: v for k, v in json.loads(
        Path(args.anchors).read_text(encoding="utf-8")).items()
        if not k.startswith("_")}
    metadata = load_metadata(Path(args.metadata))
    failures = []

    problem_dir = Path(args.problem_results)
    problem_reports = reports(problem_dir)
    for report_path in problem_reports:
        blob = json.loads(report_path.read_text(encoding="utf-8"))
        collision_events = [e for e in blob.get("events", [])
                            if e.get("type") == COLLISION]
        if len(collision_events) > 1:
            failures.append(f"{report_path.stem}: {len(collision_events)} collision alerts")

    for clip, anchor in anchors.items():
        path = problem_dir / f"{clip}.json"
        if not path.exists():
            failures.append(f"{clip}: anchor report missing")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        events = [e for e in blob.get("events", []) if e.get("type") == COLLISION]
        if len(events) != 1:
            failures.append(f"{clip}: expected one collision, got {len(events)}")
            continue
        event = events[0]
        error = abs(float(event["detected_t"])-float(anchor["reviewed_time_s"]))
        if error > 1.0:
            failures.append(f"{clip}: timing regressed by {error:.2f}s")
        if not participant_hit(event, metadata[clip]):
            failures.append(f"{clip}: accepted participant location regressed")

    traffic_collisions = 0
    traffic_clips = 0
    if args.traffic_results:
        traffic_reports = reports(Path(args.traffic_results))
        expected = {p.stem for p in Path(args.traffic_source).glob("*.mp4")}
        actual = {p.stem for p in traffic_reports}
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            failures.append(f"ElciaDataSet report coverage mismatch; missing={missing}, extra={extra}")
        for path in traffic_reports:
            traffic_clips += 1
            blob = json.loads(path.read_text(encoding="utf-8"))
            n = sum(e.get("type") == COLLISION for e in blob.get("events", []))
            traffic_collisions += n
            if n:
                failures.append(f"{path.stem}: {n} false collision alert(s)")

    print(f"anchors: {len(anchors)}  ProblemSet reports: {len(problem_reports)}")
    if args.traffic_results:
        print(f"ElciaDataSet: {traffic_clips} clips, {traffic_collisions} collisions")
    if failures:
        print("RELEASE GATE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RELEASE GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
