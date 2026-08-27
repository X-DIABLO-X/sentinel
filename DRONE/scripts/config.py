"""Configuration loader for the DRONE subsystem.

Every path in this project is stored in YAML as a **relative** path and is
resolved against :data:`PROJECT_ROOT`, which is derived from this file's own
location. There are no absolute paths anywhere in the DRONE tree, so the whole
folder can be moved or cloned onto a judge's machine and still run.

Usage::

    from config import load_config
    cfg = load_config()
    cfg.gmc.feature_type          # 'orb'
    cfg.detector.weights          # None  -> placeholder mode
    cfg.resolve('results')        # absolute Path under DRONE/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Anchors. PROJECT_ROOT is DRONE/. Everything relative resolves against it.
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "drone_config.yaml"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DETECTOR_DIR: Path = MODELS_DIR / "detector"
VISDRONE_CONFIG_PATH: Path = DETECTOR_DIR / "visdrone_config.yaml"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# The sibling CCTV subsystem, if it is present. Used only as an *optional*
# source for the ByteTrack implementation (see track_drone.py). Resolved
# relatively so nothing breaks when the repo moves.
CCTV_ROOT: Path = PROJECT_ROOT.parent / "CCTV"


def resolve_path(value: str | os.PathLike | None) -> Path | None:
    """Resolve a config path against :data:`PROJECT_ROOT`.

    Absolute inputs are honoured (a user may point at footage on another
    drive), but nothing we ship ever stores one.
    """
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


# --------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------

@dataclass
class GMCConfig:
    """Global Motion Compensation — background-homography ego-motion removal."""
    enabled: bool = True
    feature_type: str = "orb"
    max_features: int = 2000
    ransac_reproj_thresh: float = 3.0
    mask_dynamic_boxes: bool = True
    box_dilate_px: int = 12
    min_matches: int = 12
    min_inliers: int = 8
    ratio_test: float = 0.75
    max_scale_change: float = 1.25
    max_translation_frac: float = 0.25
    downscale: float = 1.0


@dataclass
class DetectorConfig:
    weights: str | None = None
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    device: str = "auto"
    half: bool = False
    classes: list[int] = field(default_factory=lambda: list(range(10)))
    vehicle_classes: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9])

    @property
    def is_placeholder(self) -> bool:
        """True when no fine-tuned checkpoint has been supplied.

        This is the single source of truth for the ``detector_finetuned: false``
        flag that appears in the API health payload and in every results file.
        """
        return not self.weights

    @property
    def weights_path(self) -> Path | None:
        return resolve_path(self.weights)


@dataclass
class TrackerConfig:
    backend: str = "auto"           # auto | netra | local
    high_thresh: float = 0.35
    low_thresh: float = 0.10
    match_thresh: float = 0.80
    max_time_lost: int = 30
    min_hits: int = 3


@dataclass
class TelemetryConfig:
    """Direct-georeferencing inputs. NOT AVAILABLE — ``enabled`` is false."""
    enabled: bool = False
    path: str | None = None
    format: str = "auto"
    gps_hz: float = 10.0
    imu_hz: float = 200.0

    @property
    def resolved_path(self) -> Path | None:
        return resolve_path(self.path)


@dataclass
class RoadPlaneConfig:
    """Reference-frame pixels -> metric ground plane.

    ``homography`` is None until someone calibrates against real footage with
    known ground distances. While it is None, all metric outputs are reported
    as null with an explicit reason rather than being fabricated from a guessed
    scale.
    """
    homography: list[list[float]] | None = None
    units: str = "metres"
    altitude_m: float | None = None
    focal_px: float | None = None

    @property
    def available(self) -> bool:
        return self.homography is not None

    def matrix(self):
        """Return the 3x3 homography as a numpy array, or None."""
        if self.homography is None:
            return None
        import numpy as np
        H = np.asarray(self.homography, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(
                f"road_plane.homography must be 3x3, got {H.shape}"
            )
        return H


@dataclass
class QualityConfig:
    enabled: bool = True
    min_blur_var: float = 25.0
    min_brightness: float = 20.0
    max_brightness: float = 240.0
    max_gmc_failure_streak: int = 15


@dataclass
class KinematicsConfig:
    speed_window_s: float = 1.0
    min_track_seconds: float = 0.5
    stationary_speed_px_s: float = 2.0


@dataclass
class ProcessingConfig:
    frame_stride: int = 1
    max_frames: int | None = None
    results_dir: str = "results"
    save_annotated: bool = False

    @property
    def results_path(self) -> Path:
        p = resolve_path(self.results_dir)
        assert p is not None
        return p


@dataclass
class ApiConfig:
    # 8011, deliberately not 8000. CCTV owns 8000; the two backends are
    # separate processes and must never contend for a port.
    host: str = "127.0.0.1"
    port: int = 8011


@dataclass
class DroneConfig:
    mode: str = "hover"
    gmc: GMCConfig = field(default_factory=GMCConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    road_plane: RoadPlaneConfig = field(default_factory=RoadPlaneConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    source_path: str | None = None      # which YAML this came from (for reporting)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def resolve(rel: str | os.PathLike | None) -> Path | None:
        return resolve_path(rel)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def provenance(self) -> dict[str, Any]:
        """A compact, honest description of what this run is actually using.

        Embedded verbatim in every results file so a reader never has to guess
        whether a number came from a real model or from the placeholder.
        """
        return {
            "mode": self.mode,
            "config_file": self.source_path,
            "detector_weights": self.detector.weights,
            "detector_finetuned": not self.detector.is_placeholder,
            "placeholder_detector": self.detector.is_placeholder,
            "gmc_enabled": self.gmc.enabled,
            "gmc_feature_type": self.gmc.feature_type,
            "telemetry_enabled": self.telemetry.enabled,
            "telemetry_available": False,   # hard false: no ingest implemented yet
            "road_plane_calibrated": self.road_plane.available,
            "metric_units_available": self.road_plane.available,
        }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _section(raw: dict, key: str, cls):
    """Build a dataclass from ``raw[key]``, ignoring unknown YAML keys.

    Unknown keys are dropped rather than raising, so a newer config file does
    not hard-fail an older checkout; known keys always win over defaults.
    """
    data = raw.get(key) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config section '{key}' must be a mapping, got {type(data).__name__}")
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | os.PathLike | None = None) -> DroneConfig:
    """Load ``config/drone_config.yaml`` (or an explicit path) into a dataclass.

    Missing file or missing sections fall back to the dataclass defaults, which
    are kept in sync with the shipped YAML.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} did not parse to a mapping")

    cfg = DroneConfig(
        mode=str(raw.get("mode", "hover")),
        gmc=_section(raw, "gmc", GMCConfig),
        detector=_section(raw, "detector", DetectorConfig),
        tracker=_section(raw, "tracker", TrackerConfig),
        telemetry=_section(raw, "telemetry", TelemetryConfig),
        road_plane=_section(raw, "road_plane", RoadPlaneConfig),
        quality=_section(raw, "quality", QualityConfig),
        kinematics=_section(raw, "kinematics", KinematicsConfig),
        processing=_section(raw, "processing", ProcessingConfig),
        api=_section(raw, "api", ApiConfig),
        # stored relative to PROJECT_ROOT so results files stay machine-agnostic
        source_path=_relative_to_root(cfg_path),
    )
    return cfg


def load_visdrone_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the VisDrone class map / fine-tune plan. Returns {} if absent."""
    p = Path(path) if path else VISDRONE_CONFIG_PATH
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _relative_to_root(p: Path) -> str:
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


if __name__ == "__main__":   # pragma: no cover - manual smoke check
    import json
    c = load_config()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(json.dumps(c.provenance(), indent=2))
