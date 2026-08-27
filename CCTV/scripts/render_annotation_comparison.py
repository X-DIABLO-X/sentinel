"""Render full videos comparing hand annotations with NETRA predictions.

Yellow is human/dataset ground truth. Red is the model output stored in the
per-clip report. The renderer does not rerun inference and says so in every
video: boxes are alert-time localizations, not reconstructed trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SCREENSHOT_MAP = {
    "0eoMti_njew_00": "Screenshot 2026-08-26 151854.png",
    "0ThWw_efieo_01": "Screenshot 2026-08-26 152343.png",
    "2W7S_-7F6S8_00": "Screenshot 2026-08-26 152519.png",
    "-6SQSDj8cYU_00": "Screenshot 2026-08-26 152714.png",
    "-7-vQ4obVwQ_00": "Screenshot 2026-08-26 152817.png",
    "29O6I-sITyw_00": "Screenshot 2026-08-26 152926.png",
    "-AztVDZ6cEE_00": "Screenshot 2026-08-26 153024.png",
    "-dmYsQc-odI_00": "Screenshot 2026-08-26 153126.png",
    "-i9bRJWMtTo_00": "Screenshot 2026-08-26 153229.png",
    "-NgnSm_oEB4_00": "Screenshot 2026-08-26 153339.png",
    "-PpBteU0p3Q_00": "Screenshot 2026-08-26 153436.png",
    "-PpjzmhI_PE_00": "Screenshot 2026-08-26 153530.png",
    "-Qt5bDJNT84_00": "Screenshot 2026-08-26 153747.png",
    "-RE3XseZINA_00": "Screenshot 2026-08-26 153848.png",
    "-RrDtLjWsT4_00": "Screenshot 2026-08-26 154822.png",
}


def load_metadata(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {Path(r["path"]).stem: r for r in csv.DictReader(fh)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = ((a[2] - a[0]) * (a[3] - a[1]) +
            (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / max(area, 1e-9))


def readable_video(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    return bool(ok and path.is_file() and path.stat().st_size > 0)


def fit_text(text: str, max_chars: int = 105) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 3] + "..."


def render_one(source: Path, report: dict, gt: dict, hand: dict | None,
               screenshot: str | None, target: Path, long_side: int) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_scale = min(1.0, long_side / max(width, height))
    ow, oh = int(round(width * out_scale)), int(round(height * out_scale))
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(pending), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (ow, oh))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open writer for {target}")

    events = report.get("events") or []
    t_human = float((hand or {}).get("t", gt["accident_time"]))
    # Evaluate the closest of the model's emitted alerts, exactly as the scorer
    # does, but draw EVERY emitted alert in the video. Ground truth selects no
    # box and changes no prediction; it only determines the reported error.
    event = (min(events, key=lambda e: abs(float(e["detected_t"]) - t_human))
             if events else None)
    t_model = float(event["detected_t"]) if event else None
    error = t_model - t_human if t_model is not None else None
    time_hit = error is not None and abs(error) <= 1.0

    # The team-drawn centre is the primary location. The benchmark box supplies
    # scale and the exact fallback when no screenshot was drawn.
    cx = float((hand or {}).get("cx")
               if (hand or {}).get("cx") is not None else gt["center_x"])
    cy = float((hand or {}).get("cy")
               if (hand or {}).get("cy") is not None else gt["center_y"])
    half_w = max(abs(float(gt["x2"]) - float(gt["x1"])) * 0.65, 0.045)
    half_h = max(abs(float(gt["y2"]) - float(gt["y1"])) * 0.65, 0.045)
    gt_box = [float(gt["x1"]) * width, float(gt["y1"]) * height,
              float(gt["x2"]) * width, float(gt["y2"]) * height]

    boxes = ((event or {}).get("triggers") or {}).get("participant_boxes") or []
    analysis_long = float((report.get("model_run") or {}).get(
        "resize_long_side") or 1920.0)
    analysis_scale = min(1.0, analysis_long / max(width, height))
    boxes_source = [[v / analysis_scale for v in box] for box in boxes]
    overlaps = [iou(box, gt_box) for box in boxes_source]
    vehicle_hit = bool(overlaps and max(overlaps) > 0.0)
    if event is None:
        verdict = "MODEL MISS"
    elif time_hit and vehicle_hit:
        verdict = "RIGHT TIME + ANNOTATED VEHICLE"
    elif time_hit:
        verdict = "RIGHT TIME; VEHICLE NOT MATCHED"
    elif vehicle_hit:
        verdict = "VEHICLE MATCH; WRONG TIME"
    else:
        verdict = "WRONG TIME; VEHICLE NOT MATCHED"

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        if out_scale != 1.0:
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)

        # Human annotation: include the collision instant and immediate
        # aftermath because several yellow screenshots mark post-impact state.
        if t_human - 0.5 <= t <= t_human + 1.5:
            centre = (int(cx * ow), int(cy * oh))
            axes = (max(18, int(half_w * ow)), max(18, int(half_h * oh)))
            cv2.ellipse(frame, centre, axes, 0, 0, 360, (0, 220, 255), 5,
                        cv2.LINE_AA)
            cv2.putText(frame, "HUMAN YELLOW EVENT REGION", (max(8, centre[0] - axes[0]),
                        max(90, centre[1] - axes[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2,
                        cv2.LINE_AA)

        # Model boxes are a frozen snapshot at the stored alert time, not a
        # trajectory. Keep them to a narrow alert-frame window; drawing the
        # same coordinates for seconds while a vehicle moves creates an
        # artificial spatial error that the model did not actually report.
        for model_event in events:
            model_event_t = float(model_event["detected_t"])
            if abs(t - model_event_t) > max(0.12, 2.0 / fps):
                continue
            event_boxes = ((model_event.get("triggers") or {})
                           .get("participant_boxes") or [])
            for raw_box in event_boxes:
                box = [v / analysis_scale for v in raw_box]
                x1, y1, x2, y2 = [int(round(v * out_scale)) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 235), 4,
                              cv2.LINE_AA)
                cv2.putText(frame, f"NETRA ALERT {model_event_t:.2f}s",
                            (max(8, x1), max(90, y1 - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 235), 2,
                            cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (ow, 82), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        human_source = "HAND YELLOW" if screenshot else "DATASET GT (NO HAND SCREENSHOT)"
        model_text = ("none" if t_model is None else
                      f"closest {t_model:.2f}s ({error:+.2f}s), {len(events)} alert(s)")
        cv2.putText(frame, fit_text(f"{source.stem} | {verdict}"), (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, fit_text(
            f"{human_source}: {t_human:.2f}s | NETRA alert: {model_text}"),
            (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 220, 225), 1,
            cv2.LINE_AA)
        cv2.putText(frame, "YELLOW=human event region   RED=model alert-time boxes",
                    (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (0, 220, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    if not readable_video(pending):
        raise RuntimeError(f"rendered video failed read-back: {pending}")
    pending.replace(target)
    return {
        "clip": source.stem,
        "hand_screenshot": screenshot,
        "human_time_s": round(t_human, 3),
        "human_kind": (hand or {}).get("kind"),
        "model_time_s": round(t_model, 3) if t_model is not None else None,
        "model_alerts": len(events),
        "temporal_error_s": round(error, 3) if error is not None else None,
        "within_1s": time_hit,
        "participant_boxes": len(boxes),
        "max_vehicle_iou": round(max(overlaps), 4) if overlaps else 0.0,
        "annotated_vehicle_matched": vehicle_hit,
        "verdict": verdict,
        "video": target.name,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="ProblemSet")
    ap.add_argument("--results", default="ProblemSet/Results")
    ap.add_argument("--hand", default="data/labels/hand_annotations.json")
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    ap.add_argument("--long-side", type=int, default=1280)
    args = ap.parse_args()

    videos, results = Path(args.videos), Path(args.results)
    hand_blob = json.loads(Path(args.hand).read_text(encoding="utf-8"))
    hand = {k: v for k, v in hand_blob.items() if not k.startswith("_")}
    metadata = load_metadata(Path(args.metadata))
    out_dir = results / "annotated_videos"
    comparison = []
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_stem = {Path(r.get("file", "")).stem: r for r in summary.get("rows", [])}

    for i, source in enumerate(sorted(videos.glob("*.mp4")), 1):
        report_path = results / f"{source.stem}.json"
        if not report_path.is_file() or source.stem not in metadata:
            print(f"[{i}] skip {source.name}: report/metadata missing")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        screenshot = SCREENSHOT_MAP.get(source.stem)
        if screenshot and not (videos / "MyAnnotations" / screenshot).is_file():
            raise FileNotFoundError(f"mapped hand screenshot missing: {screenshot}")
        target = out_dir / f"{source.stem}_annotated.mp4"
        row = render_one(source, report, metadata[source.stem], hand.get(source.stem),
                         screenshot, target, args.long_side)
        comparison.append(row)
        if source.stem in by_stem:
            by_stem[source.stem]["annotated_video"] = target.relative_to(results).as_posix()
            by_stem[source.stem]["annotation_comparison"] = row
        print(f"[{i}/16] {source.name}: {row['verdict']}")

    out = {
        "note": ("Yellow regions use the team-drawn centre and benchmark box scale. "
                 "Red boxes are the stored NETRA alert-time participant boxes. "
                 "Rendering does not rerun inference."),
        "screenshots_inspected": len(SCREENSHOT_MAP),
        "videos_rendered": len(comparison),
        "rows": comparison,
    }
    (results / "annotation_comparison.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    summary["annotation_comparison"] = {
        "screenshots_inspected": len(SCREENSHOT_MAP),
        "videos_rendered": len(comparison),
        "report": "annotation_comparison.json",
    }
    pending = summary_path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pending.replace(summary_path)
    print(f"wrote {len(comparison)} full annotated videos -> {out_dir}")
    return 0 if len(comparison) == 16 else 2


if __name__ == "__main__":
    raise SystemExit(main())
