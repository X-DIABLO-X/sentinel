"""Does this vehicle look crashed?

The one question no amount of geometry could answer. Every trajectory-level
discriminator we tried -- proximity, approach angle, deceleration, stop
persistence, companion vehicles, nearby pedestrians -- scored a clean-traffic
clip at or above a real collision. That is not a tuning failure; a vehicle
braking hard beside another and a vehicle that has just been hit look the same
in terms of position and velocity. The difference is in the vehicle's
*appearance*: crumpled, skewed, resting at an angle it should not be at.

So this is a small classifier trained on that difference, supervised for free by
the ACCIDENT benchmark's annotated accident boxes: 2,693 crash crops and 1,695
normal-vehicle crops taken from the same clips before the accident, so the model
cannot cheat by learning the camera.

Model choice was measured rather than assumed. A frozen DINOv3 backbone with a
linear probe was the favourite going in -- the ACCIDENT paper found exactly that
family beat 7B vision-language models fourfold on their classification task --
but on a like-for-like comparison a fine-tuned yolo11n-cls matched it
(AUC 0.954 vs 0.951) with fourteen times fewer parameters and 6 ms per crop.
At ~3,500 training samples there is enough data to fine-tune a small CNN
properly, which is the regime where the data-efficiency argument for frozen
features stops applying.

Where it runs: only on *candidate* vehicles for an already-raised incident, a
handful per event. It is triggered verification, never a per-frame cost.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Operating point. The dominant complaint in review was over-tagging -- boxing
# four vehicles when two collided, including cars that had merely queued behind
# the crash. Precision therefore matters more than recall here, because a wrong
# box is an accusation against an uninvolved vehicle. Measured on held-out
# validation: this threshold yields ~0.95 precision at ~0.85 recall.
DEFAULT_THRESHOLD = 0.72

SEARCH = [
    "runs/classify/models/crash_cls_yolo/weights/best.pt",
    "models/crash_cls_yolo/weights/best.pt",
    "models/crash_cls/yolo/best.pt",
]


class CrashClassifier:
    """Binary crashed-vs-normal vehicle classifier over cropped detections."""

    def __init__(self, weights: str | None = None, threshold: float = DEFAULT_THRESHOLD,
                 device: str = "auto", pad: float = 0.35, imgsz: int = 128) -> None:
        self.threshold = float(threshold)
        self.pad = float(pad)
        self.imgsz = int(imgsz)
        self.available = False
        self.model = None
        self.crash_index = 0
        self.weights = weights or self._find()
        self.device = device
        self.calls = 0

        if self.weights:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights)
                names = getattr(self.model, "names", {0: "crash", 1: "normal"})
                idx = [k for k, v in names.items() if str(v).lower().startswith("crash")]
                self.crash_index = idx[0] if idx else 0
                self.available = True
            except Exception:
                self.available = False

    @staticmethod
    def _find() -> str | None:
        for c in SEARCH:
            if Path(c).exists():
                return c
        hits = sorted(Path(".").glob("**/crash_cls_yolo/weights/best.pt"))
        return str(hits[0]) if hits else None

    # ------------------------------------------------------------------
    def _crop(self, frame: np.ndarray, box) -> np.ndarray | None:
        """Pad the box outward before cropping.

        The classifier needs a little surrounding road to judge whether a
        vehicle is resting abnormally -- a tight crop of a car body looks much
        the same crashed or not. The padding matches what training crops used.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2, y1 + bh / 2
        side = max(bw, bh) * (1 + self.pad)
        nx1, ny1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        nx2, ny2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if nx2 - nx1 < 16 or ny2 - ny1 < 16:
            return None
        c = frame[ny1:ny2, nx1:nx2]
        if c.size == 0:
            return None
        return cv2.resize(c, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)

    def score_boxes(self, frame: np.ndarray, boxes) -> list[float]:
        """p(crashed) for each box. Returns 0.0 for all if unavailable.

        A missing model must not silently become evidence, so the neutral
        return is zero and callers treat the classifier as absent rather than
        as a vote of no-confidence.
        """
        if not self.available or frame is None or not len(boxes):
            return [0.0] * len(boxes)
        crops, keep = [], []
        for i, b in enumerate(boxes):
            c = self._crop(frame, b)
            if c is not None:
                crops.append(c)
                keep.append(i)
        if not crops:
            return [0.0] * len(boxes)

        out = [0.0] * len(boxes)
        try:
            dev = 0 if str(self.device).startswith("cuda") else (
                None if self.device == "auto" else "cpu")
            res = self.model.predict(crops, verbose=False,
                                     **({"device": dev} if dev is not None else {}))
            self.calls += len(crops)
            for i, r in zip(keep, res):
                out[i] = float(r.probs.data[self.crash_index])
        except Exception:
            return [0.0] * len(boxes)
        return out

    def describe(self) -> dict:
        return {
            "available": self.available,
            "weights": self.weights,
            "threshold": self.threshold,
            "crops_scored": self.calls,
            "operating_point": "~0.95 precision / ~0.85 recall on held-out validation",
        }
