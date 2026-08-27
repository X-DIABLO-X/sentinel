"""Bundle each reported event's short evidence clip into a portable result set."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def readable_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    return bool(ok)


def write_json(path: Path, payload: dict) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pending.replace(path)


def render_review_clip(report: dict, event: dict, target: Path,
                       seconds_each_side: float = 4.0) -> bool:
    """Render a portable fallback when historical evidence is missing.

    The boxes are explicitly the alert-time localization, not re-tracked
    trajectories. This keeps an old benchmark result reviewable without
    pretending that a new inference run reproduced it.
    """
    source = Path(str(report.get("source") or ""))
    if not source.is_file():
        return False
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return False
    detected = float(event.get("detected_t", 0.0) or 0.0)
    start = max(0.0, detected - seconds_each_side)
    end = detected + seconds_each_side
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False

    analysis_long_side = float(
        (report.get("model_run") or {}).get("resize_long_side") or 1920.0)
    analysis_scale = min(1.0, analysis_long_side / max(width, height))
    boxes = (event.get("triggers") or {}).get("participant_boxes") or []
    label = str(event.get("label") or event.get("type") or "event")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if t > end:
            break
        if abs(t - detected) <= 1.25:
            for box in boxes:
                x1, y1, x2, y2 = [int(round(v / analysis_scale)) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 235), 4)
        cv2.rectangle(frame, (0, 0), (width, 72), (18, 18, 18), -1)
        cv2.putText(frame, f"{label.upper()} | alert t={detected:.2f}s",
                    (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Alert-time localization; human verification required",
                    (18, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (90, 190, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cap.release()
    return readable_video(target)


def bundle(results: Path, force_render: bool = False) -> dict:
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_dir = results / "evidence"
    video_dir.mkdir(parents=True, exist_ok=True)

    bundled, missing = 0, []
    for row in summary.get("rows", []):
        report_rel = row.get("report")
        if not report_rel:
            missing.append(f"{row.get('file')}: report path absent")
            continue
        report_path = results / report_rel
        if not report_path.is_file():
            missing.append(f"{row.get('file')}: report missing")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        events = report.get("events") or []
        if not events:
            continue
        evidence = events[0].get("evidence") or {}
        clip_name = evidence.get("clip")
        evidence_dir = evidence.get("dir")
        target = video_dir / f"{Path(row['file']).stem}_evidence.mp4"
        source = (Path(evidence_dir) / clip_name
                  if clip_name and evidence_dir else None)
        if not force_render and source is not None and readable_video(source):
            shutil.copy2(source, target)
        elif not render_review_clip(report, events[0], target):
            missing.append(f"{row.get('file')}: evidence source/render invalid")
            continue
        if not readable_video(target):
            missing.append(f"{row.get('file')}: bundled clip invalid")
            continue
        row["evidence_video"] = target.relative_to(results).as_posix()
        bundled += 1

    summary["evidence_bundle"] = {
        "bundled": bundled,
        "missing_or_invalid": missing,
        "complete_for_reported_events": not missing,
    }
    write_json(summary_path, summary)
    return summary["evidence_bundle"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="ProblemSet/Results")
    ap.add_argument("--force-render", action="store_true",
                    help="regenerate clips from source/report timestamps even "
                         "when historical evidence exists")
    args = ap.parse_args()
    status = bundle(Path(args.results), force_render=args.force_render)
    print(json.dumps(status, indent=2))
    return 0 if status["complete_for_reported_events"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
