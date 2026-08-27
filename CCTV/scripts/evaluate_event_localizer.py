"""Audit whether appearance can re-localise an already-timed collision event.

This is deliberately an evaluator, not an inference shortcut.  It compares the
existing lightweight crashed-vehicle verifier against exact ACCIDENT boxes at
both the pipeline timestamp and the annotated timestamp.  The latter is an
oracle diagnostic: if it fails there, appearance cannot repair localisation;
if only the pipeline-time result fails, the remaining problem is temporal.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from netra.crashcls import CrashClassifier


VEHICLES = {"car", "motorcycle", "bus", "truck"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = ((a[2] - a[0]) * (a[3] - a[1]) +
            (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / max(area, 1e-9))


def frame_at(path: Path, t: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def flow_acceleration(prev, cur, nxt, long_side: int = 640):
    """Camera-motion-compensated change in optical-flow velocity."""
    h, w = cur.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    size = (max(2, int(round(w * scale))), max(2, int(round(h * scale))))
    gray = [cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_BGR2GRAY)
            for f in (prev, cur, nxt)]
    flow1 = cv2.calcOpticalFlowFarneback(gray[0], gray[1], None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
    flow2 = cv2.calcOpticalFlowFarneback(gray[1], gray[2], None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)
    # Median translation is the dashcam/camera component.  Subtracting it
    # leaves local acceleration, turn, impact and deformation evidence.
    f1 = flow1 - np.median(flow1.reshape(-1, 2), axis=0)
    f2 = flow2 - np.median(flow2.reshape(-1, 2), axis=0)
    acc = np.linalg.norm(f2 - f1, axis=2)
    acc = cv2.GaussianBlur(acc, (0, 0), 2.0)
    return acc, scale


def box_flow_score(acc, scale: float, box) -> float:
    h, w = acc.shape
    x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    roi = acc[y1:y2, x1:x2]
    if roi.size < 16:
        return 0.0
    # High percentile captures a contact edge without rewarding a large box
    # merely because it covers more moving road.
    return float(0.65 * np.percentile(roi, 90) + 0.35 * np.mean(roi))


def detect_vehicles(detector, frame, imgsz: int):
    pred = detector.predict(frame, imgsz=imgsz, conf=0.05,
                            device=0, verbose=False)[0]
    names = pred.names
    return [xyxy.tolist() for xyxy, cls in zip(
        pred.boxes.xyxy.cpu().numpy(), pred.boxes.cls.cpu().numpy())
        if str(names[int(cls)]).lower() in VEHICLES]


def centre_distance(a, b) -> float:
    ac = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    bc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    diag = max(8.0, float(np.hypot(a[2] - a[0], a[3] - a[1])))
    return float(np.linalg.norm(ac - bc) / diag)


def new_stop_scores(current, before, after):
    """Place-based arrival then persistence; no track identity is required."""
    scores = []
    for box in current:
        pre_d = min((centre_distance(box, b) for b in before), default=4.0)
        post_d = min((centre_distance(box, b) for b in after), default=4.0)
        # High when a vehicle was not already at this place but remains there
        # after the event.  A missing post detection is not persistence.
        persistence = max(0.0, 1.5 - post_d) if after else 0.0
        arrival = min(2.0, pre_d)
        # Contact/queue interaction: a nearby road user is corroboration, but
        # the arrival-to-stop transition remains the dominant term.
        others = [centre_distance(box, b) for b in current if b is not box]
        proximity = max(0.0, 1.2 - min(others, default=4.0))
        scores.append(1.3 * persistence + 0.6 * arrival + 0.25 * proximity)
    return scores


def closest_pair(boxes):
    """Pair with the smallest surface gap, normalised by vehicle size."""
    best = None
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
            dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
            scale = max(8.0, min(np.hypot(a[2]-a[0], a[3]-a[1]),
                                 np.hypot(b[2]-b[0], b[3]-b[1])))
            gap = float(np.hypot(dx, dy) / scale)
            if best is None or gap < best[0]:
                best = (gap, i, j)
    return best


def contact_scores(boxes):
    out = []
    for i, a in enumerate(boxes):
        gaps = []
        for j, b in enumerate(boxes):
            if i == j:
                continue
            dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
            dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
            scale = max(8.0, min(np.hypot(a[2]-a[0], a[3]-a[1]),
                                 np.hypot(b[2]-b[0], b[3]-b[1])))
            gaps.append(float(np.hypot(dx, dy) / scale))
        out.append(1.0 / (1.0 + min(gaps, default=8.0)))
    return out


LOCALIZER_FEATURES = ("appearance", "flow_acceleration", "new_stop",
                      "contact", "log_area_fraction", "aspect_ratio")


def fit_logistic(X, y, iters=900, lr=0.08, l2=1.0):
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    pos, neg = max(1, int(y.sum())), max(1, int((1-y).sum()))
    sw = np.where(y == 1, neg / pos, 1.0)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        g = (p - y) * sw
        w -= lr * ((X.T @ g) / len(X) + l2 * w / len(X))
        b -= lr * g.mean()
    return w, b


def train_localizer(rows):
    X, y, groups = [], [], []
    for row in rows:
        for feat in row["at_truth_time"].get("box_features", []):
            X.append([feat[k] for k in LOCALIZER_FEATURES])
            y.append(int(feat["iou"] > 0))
            groups.append(row["clip"])
    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    valid_groups = sorted({g for g in groups if y[groups == g].sum() > 0})
    hits, cv_rows = 0, []
    for clip in valid_groups:
        tr, te = groups != clip, groups == clip
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        w, b = fit_logistic((X[tr]-mu)/sd, y[tr])
        p = 1.0 / (1.0 + np.exp(-np.clip(((X[te]-mu)/sd) @ w + b, -30, 30)))
        chosen = np.flatnonzero(te)[int(np.argmax(p))]
        hit = bool(y[chosen])
        hits += int(hit)
        cv_rows.append({"clip": clip, "hit": hit,
                        "selected_probability": round(float(np.max(p)), 5)})
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    w, b = fit_logistic((X-mu)/sd, y)
    return {
        "features": list(LOCALIZER_FEATURES), "mean": mu.tolist(),
        "std": sd.tolist(), "weights": w.tolist(), "bias": float(b),
        "training": {"boxes": len(X), "positive_boxes": int(y.sum()),
                     "clips": len(set(groups))},
        "cv": {"scheme": "leave-one-video-out at annotated event time",
               "top1": hits, "detectable_clips": len(valid_groups),
               "all_clips": len(rows), "clips": cv_rows},
        "scope": ("triggered participant localisation only; timestamp and "
                  "event decision remain independent"),
    }


def load_truth(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as fh:
        return {Path(r["path"]).stem: r for r in csv.DictReader(fh)}


def audit_at(detector, verifier, video: Path, t: float, gt_box, imgsz: int):
    frame = frame_at(video, t)
    prev = frame_at(video, t - 0.4)
    nxt = frame_at(video, t + 0.4)
    before = frame_at(video, t - 1.2)
    after = frame_at(video, t + 1.5)
    if frame is None:
        return {"t": t, "detections": 0, "oracle_iou": 0.0,
                "selected_iou": 0.0, "selected_score": 0.0}
    boxes = detect_vehicles(detector, frame, imgsz)
    before_boxes = detect_vehicles(detector, before, imgsz) if before is not None else []
    after_boxes = detect_vehicles(detector, after, imgsz) if after is not None else []
    scores = verifier.score_boxes(frame, boxes)
    if prev is not None and nxt is not None:
        acc, flow_scale = flow_acceleration(prev, frame, nxt)
        flow_scores = [box_flow_score(acc, flow_scale, b) for b in boxes]
    else:
        flow_scores = [0.0] * len(boxes)
    stop_scores = new_stop_scores(boxes, before_boxes, after_boxes)
    contacts = contact_scores(boxes)
    oracle = max((iou(b, gt_box) for b in boxes), default=0.0)
    best = max(range(len(boxes)), key=lambda i: scores[i]) if boxes else None
    flow_best = (max(range(len(boxes)), key=lambda i: flow_scores[i])
                 if boxes else None)
    stop_best = (max(range(len(boxes)), key=lambda i: stop_scores[i])
                 if boxes else None)
    pair = closest_pair(boxes)
    fh, fw = frame.shape[:2]
    box_features = []
    for b, ap, fl, st, ct in zip(boxes, scores, flow_scores,
                                 stop_scores, contacts):
        bw, bh = max(1.0, b[2]-b[0]), max(1.0, b[3]-b[1])
        box_features.append({
            "box": b, "iou": round(float(iou(b, gt_box)), 4),
            "appearance": float(ap), "flow_acceleration": float(fl),
            "new_stop": float(st), "contact": float(ct),
            "log_area_fraction": float(np.log(max(1e-8, bw*bh/(fw*fh)))),
            "aspect_ratio": float(bw/bh),
        })
    return {
        "t": round(float(t), 3), "detections": len(boxes),
        "oracle_iou": round(float(oracle), 4),
        "selected_iou": round(float(iou(boxes[best], gt_box)), 4)
                        if best is not None else 0.0,
        "selected_score": round(float(scores[best]), 4)
                          if best is not None else 0.0,
        "selected_box": boxes[best] if best is not None else None,
        "flow_selected_iou": round(float(iou(boxes[flow_best], gt_box)), 4)
                             if flow_best is not None else 0.0,
        "flow_score": round(float(flow_scores[flow_best]), 4)
                      if flow_best is not None else 0.0,
        "flow_selected_box": boxes[flow_best] if flow_best is not None else None,
        "stop_selected_iou": round(float(iou(boxes[stop_best], gt_box)), 4)
                             if stop_best is not None else 0.0,
        "stop_score": round(float(stop_scores[stop_best]), 4)
                      if stop_best is not None else 0.0,
        "stop_selected_box": boxes[stop_best] if stop_best is not None else None,
        "pair_selected_iou": round(float(max(iou(boxes[pair[1]], gt_box),
                                                   iou(boxes[pair[2]], gt_box))), 4)
                             if pair is not None else 0.0,
        "pair_gap": round(float(pair[0]), 4) if pair is not None else None,
        "pair_selected_boxes": ([boxes[pair[1]], boxes[pair[2]]]
                                if pair is not None else []),
        "box_features": box_features,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="ProblemSet")
    ap.add_argument("--results", default="ProblemSet/Results_localization_baseline")
    ap.add_argument("--metadata", default="data/accident/metadata-real.csv")
    ap.add_argument("--times", default="data/labels/accident_times.json")
    ap.add_argument("--detector", default="yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="ProblemSet/Results_localization_baseline/appearance_localizer_audit.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    detector = YOLO(args.detector)
    verifier = CrashClassifier(device="cuda:0")
    if not verifier.available:
        raise RuntimeError("crashed-vehicle verifier weights were not found")

    truth = load_truth(Path(args.metadata))
    team = json.loads(Path(args.times).read_text(encoding="utf-8"))
    rows = []
    for video in sorted(Path(args.videos).glob("*.mp4")):
        gt = truth[video.stem]
        width, height = float(gt["width"]), float(gt["height"])
        gt_box = [float(gt["x1"]) * width, float(gt["y1"]) * height,
                  float(gt["x2"]) * width, float(gt["y2"]) * height]
        t_gt = float(team.get(video.stem, gt["accident_time"]))
        report_path = Path(args.results) / f"{video.stem}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        events = report.get("events") or []
        t_model = float(events[0]["detected_t"]) if events else None
        oracle_time = audit_at(detector, verifier, video, t_gt, gt_box, args.imgsz)
        model_time = (audit_at(detector, verifier, video, t_model, gt_box,
                               args.imgsz) if t_model is not None else None)
        row = {"clip": video.stem, "truth_t": t_gt, "model_t": t_model,
               "at_truth_time": oracle_time, "at_model_time": model_time}
        rows.append(row)
        model_iou = None if model_time is None else model_time["selected_iou"]
        flow_iou = None if model_time is None else model_time["flow_selected_iou"]
        stop_iou = None if model_time is None else model_time["stop_selected_iou"]
        pair_iou = None if model_time is None else model_time["pair_selected_iou"]
        print(f"{video.stem:24s} t={t_model!s:>6s} "
              f"appearance={model_iou}/{oracle_time['selected_iou']} "
              f"flow={flow_iou}/{oracle_time['flow_selected_iou']} "
              f"new-stop={stop_iou}/{oracle_time['stop_selected_iou']} "
              f"pair={pair_iou}/{oracle_time['pair_selected_iou']} "
              f"detector ceiling@truth={oracle_time['oracle_iou']}")

    summary = {
        "clips": len(rows),
        "appearance_hit_at_model_time": sum(
            1 for r in rows if r["at_model_time"] and
            r["at_model_time"]["selected_iou"] > 0),
        "appearance_hit_at_truth_time": sum(
            1 for r in rows if r["at_truth_time"]["selected_iou"] > 0),
        "detector_has_box_at_truth_time": sum(
            1 for r in rows if r["at_truth_time"]["oracle_iou"] > 0),
        "flow_hit_at_model_time": sum(
            1 for r in rows if r["at_model_time"] and
            r["at_model_time"]["flow_selected_iou"] > 0),
        "flow_hit_at_truth_time": sum(
            1 for r in rows if r["at_truth_time"]["flow_selected_iou"] > 0),
        "new_stop_hit_at_model_time": sum(
            1 for r in rows if r["at_model_time"] and
            r["at_model_time"]["stop_selected_iou"] > 0),
        "new_stop_hit_at_truth_time": sum(
            1 for r in rows if r["at_truth_time"]["stop_selected_iou"] > 0),
        "closest_pair_hit_at_model_time": sum(
            1 for r in rows if r["at_model_time"] and
            r["at_model_time"]["pair_selected_iou"] > 0),
        "closest_pair_hit_at_truth_time": sum(
            1 for r in rows if r["at_truth_time"]["pair_selected_iou"] > 0),
    }
    learned = train_localizer(rows)
    summary["learned_localizer_loco_top1"] = learned["cv"]["top1"]
    summary["learned_localizer_detectable_clips"] = learned["cv"]["detectable_clips"]
    model_path = Path("models/event_localizer.json")
    model_path.write_text(json.dumps(learned, indent=2), encoding="utf-8")
    payload = {"summary": summary, "clips": rows,
               "warning": "truth-time results are diagnostic oracle results, not deployable performance"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {model_path}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
