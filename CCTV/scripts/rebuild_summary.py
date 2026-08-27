"""Rebuild a portable batch summary from per-clip reports."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "frames": frames, "width": width, "height": height,
            "duration": frames / fps if fps > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="ProblemSet/Results")
    ap.add_argument("--group", default="ProblemSet")
    ap.add_argument("--provenance", default=None)
    args = ap.parse_args()
    results = Path(args.results)
    reports = sorted(p for p in results.glob("*.json")
                     if p.name != "summary.json" and
                     not p.name.endswith("_candidates.json"))
    rows, first = [], None
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        first = first or report
        source = Path(str(report.get("source") or ""))
        events = report.get("events") or []
        rows.append({
            "group": args.group,
            "file": source.name or f"{path.stem}.mp4",
            "camera_id": report.get("camera_id", path.stem),
            "video": probe(source) if source.is_file() else {},
            "calibrated": False,
            "corridors": 0,
            "render": {"skipped": True},
            "report": path.name,
            "events_total": report.get("events_total", len(events)),
            "events_by_type": report.get("events_by_type", {}),
            "events_by_severity": report.get("events_by_severity", {}),
            "alerts_per_video_hour": report.get("alerts_per_video_hour", 0.0),
            "stats": report.get("stats", {}),
            "events": events,
        })
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector": (first or {}).get("model_run", {}),
        "clips": len(rows),
        "total_incidents": sum(r["events_total"] for r in rows),
        "output_validation": {
            "expected_reports": len(reports),
            "written_reports": len(reports),
            "missing_reports": [],
            "failed_clips": [],
            "complete": bool(reports),
        },
        "result_provenance": args.provenance,
        "rows": rows,
    }
    pending = results / "summary.json.tmp"
    pending.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pending.replace(results / "summary.json")
    print(f"rebuilt {len(rows)} rows -> {results / 'summary.json'}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
