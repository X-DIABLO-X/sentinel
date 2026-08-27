"""Road-user detection.

One continuously-running neural network. That is the whole AI budget of this
system; everything downstream is geometry, temporal statistics and state
machines. The design rule from the evidence review is:

    deep learning for perception, deterministic reasoning for events

Backends
--------
``torch``     PyTorch/CUDA. Fast on the RTX 4050, used for offline benchmark
              runs and evaluation sweeps.
``openvino``  Intel OpenVINO IR, optionally INT8. This is the *deployment*
              target: the civic argument for this project is that a city
              should not have to buy a GPU per camera, and OpenVINO on a
              commodity CPU is how that claim is cashed.

Both backends go through the same ``Detection`` contract, so swapping one for
the other cannot change event logic -- only latency.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("YOLO_VERBOSE", "False")
# Keep Ultralytics runtime metadata inside the project. On locked-down Windows
# demo machines its default roaming-profile path is not writable, which used to
# make an uploaded job fail before the detector even loaded.
_RUNTIME_CONFIG = Path(__file__).resolve().parents[1] / ".runtime" / "ultralytics"
_RUNTIME_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_RUNTIME_CONFIG))

# COCO ids for the road users we care about. Deliberately small: every extra
# class is another source of false tracks, and none of the events need to know
# about a potted plant.
ROAD_USER_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# The IISc AIM UVH-26 detector, fine-tuned on ~2,800 Bengaluru Safe City CCTV
# cameras, uses its own 14-class India-specific taxonomy instead of COCO's.
# Reported up to +31.5% mAP50:95 over COCO baselines on Indian traffic, which is
# exactly the domain gap the IDD paper documents. Note it has *no person class* --
# so pedestrian events need a COCO model alongside it.
UVH26_CLASSES: dict[int, str] = {
    0: "Hatchback", 1: "Sedan", 2: "SUV", 3: "MUV", 4: "Bus", 5: "Truck",
    6: "Three-wheeler", 7: "Two-wheeler", 8: "LCV", 9: "Mini-bus",
    10: "Tempo-traveller", 11: "Bicycle", 12: "Van", 13: "Others",
}
UVH26_MOTORISED = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13}
UVH26_VULNERABLE = {11}

# Which taxonomy the engines should reason in. Set when a Detector is built.
ACTIVE_TAXONOMY = "coco"


class _TaxonomyAwareSet:
    """A set whose members depend on the detector currently loaded.

    The event engines are written against ``MOTORISED_CLASSES`` and
    ``VULNERABLE_CLASSES`` and should not have to know whether the detector
    speaks COCO or the UVH-26 Indian taxonomy. Rebinding a module global would
    not work -- the engines did ``from ..detect import MOTORISED_CLASSES`` and
    captured the object. So the object itself resolves late.
    """

    __slots__ = ("_coco", "_uvh", "_name")

    def __init__(self, coco: set, uvh: set, name: str):
        self._coco, self._uvh, self._name = coco, uvh, name

    def _live(self) -> set:
        return self._uvh if ACTIVE_TAXONOMY == "uvh26" else self._coco

    def __contains__(self, x) -> bool:
        return x in self._live()

    def __iter__(self):
        return iter(self._live())

    def __len__(self) -> int:
        return len(self._live())

    def __or__(self, other):
        return self._live() | set(other)

    def __ror__(self, other):
        return set(other) | self._live()

    def __repr__(self) -> str:
        return f"<{self._name} {ACTIVE_TAXONOMY}: {sorted(self._live())}>"


VEHICLE_CLASSES = _TaxonomyAwareSet({1, 2, 3, 5, 7}, UVH26_MOTORISED | UVH26_VULNERABLE, "VEHICLE")
MOTORISED_CLASSES = _TaxonomyAwareSet({2, 3, 5, 7}, UVH26_MOTORISED, "MOTORISED")
VULNERABLE_CLASSES = _TaxonomyAwareSet({0, 1}, UVH26_VULNERABLE, "VULNERABLE")


def class_name(cls: int) -> str:
    table = UVH26_CLASSES if ACTIVE_TAXONOMY == "uvh26" else ROAD_USER_CLASSES
    return table.get(int(cls), str(cls))


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    cls: int

    @property
    def name(self) -> str:
        return ROAD_USER_CLASSES.get(self.cls, str(self.cls))


class Detector:
    """Thin, backend-agnostic wrapper over an Ultralytics YOLO model.

    Notes on model choice
    ---------------------
    The evidence review recommends YOLO26n (NMS-free head, ~43% faster CPU ONNX
    than YOLO11n at higher mAP). Whether that checkpoint is available depends on
    the installed ultralytics version, so ``preferred`` is a *list* and the
    first one that resolves wins. The system records which one it actually used
    in the model-run row, so results stay reproducible.
    """

    PREFERRED = ["yolo26n.pt", "yolo12n.pt", "yolo11n.pt", "yolov8n.pt"]

    @staticmethod
    def is_uvh26(weights: str | None) -> bool:
        return bool(weights) and "UVH-26" in str(weights)

    def __init__(self,
                 weights: str | None = None,
                 backend: str = "auto",
                 device: str = "auto",
                 imgsz: int = 640,
                 aux_imgsz: int | None = 512,
                 conf: float = 0.10,
                 iou: float = 0.55,
                 classes: Sequence[int] | None = None,
                 half: bool = False) -> None:
        from ultralytics import YOLO  # imported lazily: keeps CLI help fast

        self.imgsz = int(imgsz)
        # Second, deliberately small pass for vehicles close to the
        # camera, which the primary pass loses as resolution rises.
        # See Detector.detect for the measurement behind this.
        self.aux_imgsz = int(aux_imgsz) if aux_imgsz else None
        self.conf = float(conf)
        self.iou = float(iou)
        if classes is not None:
            self.classes = sorted(set(classes))
        elif self.is_uvh26(weights):
            self.classes = sorted(UVH26_CLASSES)
        else:
            self.classes = sorted(ROAD_USER_CLASSES)
        self.half = bool(half)

        self.weights = self._resolve_weights(weights)
        self.backend = self._resolve_backend(backend)
        self.device = self._resolve_device(device)

        if self.backend == "openvino":
            ov_dir = self._ensure_openvino(self.weights)
            self.model = YOLO(str(ov_dir), task="detect")
            self.device = "cpu"
        else:
            self.model = YOLO(self.weights)

        self.names = getattr(self.model, "names", ROAD_USER_CLASSES)
        self.taxonomy = "uvh26" if self.is_uvh26(self.weights) else "coco"
        global ACTIVE_TAXONOMY
        ACTIVE_TAXONOMY = self.taxonomy
        self._warmed = False
        self.last_latency_ms = 0.0
        self._latencies: list[float] = []

    # -- setup helpers -----------------------------------------------------
    def _resolve_weights(self, weights: str | None) -> str:
        from ultralytics import YOLO
        if weights:
            return weights
        for name in self.PREFERRED:
            try:
                YOLO(name)          # triggers download / cache lookup
                return name
            except Exception:
                continue
        raise RuntimeError(
            "no YOLO checkpoint could be resolved; pass --weights explicitly"
        )

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend != "auto":
            return backend
        return "torch"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_openvino(self, weights: str) -> Path:
        """Export to OpenVINO IR once and cache it next to the weights."""
        from ultralytics import YOLO
        stem = Path(weights).stem
        out = Path("models") / f"{stem}_openvino_model"
        if out.exists() and any(out.glob("*.xml")):
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(weights)
        exported = model.export(format="openvino", imgsz=self.imgsz, half=False)
        src = Path(exported)
        if src != out:
            import shutil
            if out.exists():
                shutil.rmtree(out)
            shutil.move(str(src), str(out))
        return out

    # -- inference ---------------------------------------------------------
    def warmup(self, shape: tuple[int, int] = (640, 640)) -> None:
        if self._warmed:
            return
        dummy = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        self.detect(dummy)
        self._warmed = True
        self._latencies.clear()

    def _predict_at(self, frame: np.ndarray, imgsz: int) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for b, s, c in zip(xyxy, conf, cls):
            out.append(Detection(box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                                 score=float(s), cls=int(c)))
        return out

    @staticmethod
    def _merge(primary: list[Detection], extra: list[Detection],
               iou_thresh: float = 0.55) -> list[Detection]:
        """Add detections from the second scale that the first one missed."""
        if not extra:
            return primary
        kept = list(primary)
        for d in extra:
            duplicate = False
            for k in kept:
                if k.cls != d.cls:
                    continue
                ax1, ay1, ax2, ay2 = k.box
                bx1, by1, bx2, by2 = d.box
                ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                if inter <= 0:
                    continue
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                ua = area_a + area_b - inter
                # IoU alone is not enough across two scales. The same vehicle
                # seen at 1920 and at 512 gets boxes of noticeably different
                # extent, so IoU can fall below the threshold while one box sits
                # almost entirely inside the other. Containment catches that,
                # and without it the second scale spawns a duplicate track on
                # the same car -- two ids whose paths then cross each other and
                # look exactly like a collision.
                contain = inter / max(min(area_a, area_b), 1e-9)
                if inter / max(ua, 1e-9) >= iou_thresh or contain >= 0.70:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(d)
        return kept

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect at one or two scales.

        Two scales, because resolution is not a single trade-off. Measured on a
        tunnel frame where a lorry fills 38% of the image, the *same* model and
        the *same* frame give:

            imgsz  416 -> 0.479 (truck)      1280 -> 0.041
            imgsz  640 -> 0.142              1600 -> 0.030
            imgsz  960 -> 0.132 (as "train") 1920 -> not detected at all

        A monotonic collapse. High input resolution is what finds small distant
        vehicles -- an earlier measurement here showed roughly 5x more confident
        detections at 1280 than at 640 on dense traffic -- but for a vehicle
        close to the camera it does the opposite, because the object grows past
        the scale the network's largest stride was trained to handle.

        Optimising for one end of the range silently sacrifices the other, and a
        vehicle near the camera is the one most likely to be in the incident.
        So the frame is seen twice: once at full resolution for the distance,
        once small for what is close. The auxiliary pass costs little -- 416
        square is about a twentieth of the pixels of 1920 square -- and only
        contributes boxes the primary pass missed.
        """
        t0 = time.perf_counter()
        out = self._predict_at(frame, self.imgsz)
        if self.aux_imgsz and abs(self.aux_imgsz - self.imgsz) > 64:
            out = self._merge(out, self._predict_at(frame, self.aux_imgsz))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies.append(self.last_latency_ms)
        return out

    def detect_array(self, frame: np.ndarray) -> np.ndarray:
        """Detections as the (N, 6) array the tracker consumes."""
        dets = self.detect(frame)
        if not dets:
            return np.empty((0, 6), dtype=np.float64)
        return np.array(
            [[d.box[0], d.box[1], d.box[2], d.box[3], d.score, d.cls] for d in dets],
            dtype=np.float64,
        )

    # -- telemetry ---------------------------------------------------------
    def latency_stats(self) -> dict:
        """Latency summary. p95 is reported because a mean hides the stalls
        that actually break a live pipeline."""
        if not self._latencies:
            return {"n": 0}
        a = np.asarray(self._latencies)
        return {
            "n": int(a.size),
            "mean_ms": round(float(a.mean()), 2),
            "median_ms": round(float(np.median(a)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "max_ms": round(float(a.max()), 2),
            "implied_fps": round(1000.0 / max(float(a.mean()), 1e-6), 2),
        }

    def describe(self) -> dict:
        return {
            "weights": str(self.weights),
            "backend": self.backend,
            "device": str(self.device),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "classes": self.classes,
        }
