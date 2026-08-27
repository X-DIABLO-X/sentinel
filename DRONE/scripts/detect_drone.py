"""Vehicle/VRU detector for the DRONE subsystem — Ultralytics YOLO wrapper.

=============================================================================
STATUS TODAY: PLACEHOLDER. ``models/detector/visdrone_config.yaml`` has
``weights: null`` because no drone footage and no fine-tuned checkpoint
exist yet. This module loads a generic COCO-pretrained YOLO instead, so the
pipeline is mechanically runnable end to end right now, and prints a loud
multi-line banner on every run saying so. See models/detector/README.md for
the full reasoning (aerial view removes vehicle side-profile; VisDrone is the
correct fine-tune target, not the CCTV side's eye-level datasets).
=============================================================================

Two operating modes, selected automatically from ``cfg.detector.weights``:

* **weights is None (placeholder, today)** — a small generic COCO model is
  loaded. COCO's classes are a rough, *lossy* stand-in for the VisDrone
  taxonomy the rest of this project reasons in (see ``COCO_TO_VISDRONE``
  below). Detections that have no sane VisDrone counterpart (COCO has no
  concept of "van" vs "car", no "tricycle", no "awning-tricycle") are
  dropped rather than mapped to something misleading. Boxes are real
  (whatever a COCO detector finds in an aerial frame — likely little, since
  it was never trained on this viewing angle), but the class semantics and
  any downstream count/severity number are not meaningful. That is the
  entire point of ``is_placeholder``.
* **weights is a path (future, once fine-tuned)** — the checkpoint already
  outputs VisDrone class ids directly and is used as-is, no remapping.

Both modes return the same array shape so nothing downstream needs to know
which one is active — it only needs to read ``DroneDetector.is_placeholder``
and ``DroneDetector.class_names`` before trusting a class label.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as drone_config  # noqa: E402

log = logging.getLogger("drone.detect")

__all__ = [
    "DroneDetector",
    "load_detector",
    "COCO_TO_VISDRONE",
    "PLACEHOLDER_MODEL_NAME",
]

# A small generic Ultralytics checkpoint, resolved by name via the Ultralytics
# hub (downloaded once, cached) when no fine-tuned weights are configured.
# Deliberately the smallest sensible model: it is standing in for a network
# that has never seen this viewing angle, so spending compute on a bigger
# placeholder buys nothing.
PLACEHOLDER_MODEL_NAME = "yolo11n.pt"

# COCO id -> VisDrone id. Only classes with a defensible correspondence are
# mapped; everything else is dropped rather than guessed. See the module
# docstring — a COCO detector has no notion of van / tricycle / awning-tricycle,
# and forcing car-shaped boxes into those buckets would fabricate a class the
# model never actually distinguished.
#   VisDrone: 0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck,
#             6 tricycle, 7 awning-tricycle, 8 bus, 9 motor
COCO_TO_VISDRONE: dict[int, int] = {
    0: 0,   # person       -> pedestrian
    1: 2,   # bicycle      -> bicycle
    2: 3,   # car          -> car
    3: 9,   # motorcycle   -> motor
    5: 8,   # bus          -> bus
    7: 5,   # truck        -> truck
}

VISDRONE_NAMES: dict[int, str] = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}

_BANNER_LOCK = threading.Lock()


def _print_placeholder_banner(model_name: str) -> None:
    lines = [
        "",
        "=" * 78,
        "  PLACEHOLDER DETECTOR — NOT FINE-TUNED ON VISDRONE",
        "  RESULTS ARE NOT MEANINGFUL",
        "=" * 78,
        f"  Loaded a generic COCO-pretrained model ({model_name}) because",
        "  models/detector/visdrone_config.yaml has weights: null.",
        "  No drone footage and no VisDrone fine-tune have been run.",
        "  This model has never seen a nadir/oblique aerial view. Detections",
        "  from it are for pipeline plumbing only — do NOT read counts,",
        "  classes, or confidence out of this as a system capability.",
        "  detector_finetuned=false is stamped on every downstream result.",
        "  See DRONE/models/detector/README.md.",
        "=" * 78,
        "",
    ]
    banner = "\n".join(lines)
    with _BANNER_LOCK:
        print(banner)
    log.warning("PLACEHOLDER DETECTOR ACTIVE — not fine-tuned on VisDrone; results not meaningful")


class DroneDetector:
    """Thin wrapper around Ultralytics YOLO producing VisDrone-space boxes.

    ``detect(frame)`` returns an ``(N, 6)`` float64 array of
    ``[x1, y1, x2, y2, score, cls]`` where ``cls`` is always a VisDrone class
    id (0-9), regardless of which underlying checkpoint produced it.
    """

    def __init__(self, cfg: "drone_config.DetectorConfig | None" = None) -> None:
        self.cfg = cfg or drone_config.DetectorConfig()
        self._model: Any = None
        self._model_source: str | None = None
        self.is_placeholder: bool = self.cfg.is_placeholder
        self.class_names: dict[int, str] = dict(VISDRONE_NAMES)

    # -- lazy load ----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "ultralytics is not installed. `pip install -r requirements.txt` "
                "in DRONE/ before running detection."
            ) from exc

        if self.cfg.weights_path is not None and self.cfg.weights_path.exists():
            source = str(self.cfg.weights_path)
            self.is_placeholder = False
            log.info("Loading fine-tuned drone detector: %s", source)
        else:
            if self.cfg.weights_path is not None:
                log.warning(
                    "detector.weights=%r does not exist on disk; falling back "
                    "to the generic placeholder model.",
                    str(self.cfg.weights_path),
                )
            source = PLACEHOLDER_MODEL_NAME
            self.is_placeholder = True
            _print_placeholder_banner(source)

        self._model = YOLO(source)
        self._model_source = source

    # -- inference ------------------------------------------------------
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection on one BGR frame. Returns (N, 6) [x1,y1,x2,y2,score,cls]."""
        if frame is None:
            raise ValueError("frame is None")
        self._ensure_loaded()

        device = None if (self.cfg.device or "auto") == "auto" else self.cfg.device

        if self.is_placeholder:
            predict_classes = sorted(COCO_TO_VISDRONE.keys())
        else:
            predict_classes = list(self.cfg.classes) if self.cfg.classes else None

        predict_kwargs: dict[str, Any] = dict(
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
            max_det=int(self.cfg.max_det),
            device=device,
            classes=predict_classes,
            verbose=False,
        )
        if self.cfg.half:
            # Only pass `half` when actually requesting it: newer Ultralytics
            # versions emit a deprecation warning on *every single call* if
            # this kwarg is passed at all, even as False.
            predict_kwargs["half"] = True

        results = self._model.predict(frame, **predict_kwargs)

        if not results:
            return np.empty((0, 6), dtype=np.float64)

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float64)

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = boxes.conf.cpu().numpy().astype(np.float64)
        cls = boxes.cls.cpu().numpy().astype(int)

        if self.is_placeholder:
            out_rows = []
            for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cls):
                vd = COCO_TO_VISDRONE.get(int(c))
                if vd is None:
                    continue
                out_rows.append([x1, y1, x2, y2, sc, float(vd)])
            if not out_rows:
                return np.empty((0, 6), dtype=np.float64)
            return np.asarray(out_rows, dtype=np.float64)

        out = np.empty((xyxy.shape[0], 6), dtype=np.float64)
        out[:, :4] = xyxy
        out[:, 4] = conf
        out[:, 5] = cls.astype(np.float64)
        return out

    def status(self) -> dict[str, Any]:
        return {
            "is_placeholder": bool(self.is_placeholder),
            "detector_finetuned": not bool(self.is_placeholder),
            "model_source": self._model_source,
            "configured_weights": self.cfg.weights,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf,
            "class_names": self.class_names,
        }


def load_detector(cfg: "drone_config.DetectorConfig | None" = None) -> DroneDetector:
    return DroneDetector(cfg)


if __name__ == "__main__":   # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.INFO)
    dcfg = drone_config.load_config()
    det = load_detector(dcfg.detector)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    boxes = det.detect(frame)
    print("boxes shape:", boxes.shape)
    print("status:", det.status())
