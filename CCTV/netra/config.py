"""Configuration loading.

Every threshold in NETRA lives in one YAML file. That is a deliberate design
constraint, not a convenience: the brief is judged partly on reproducibility,
and a system whose behaviour is scattered across dozens of hard-coded constants
cannot be reproduced or audited. The config is hashed into every model-run row,
so any number in a report can be traced back to the exact settings that
produced it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "detector": {
        # left blank so Detector picks the newest available checkpoint and
        # records which one it actually used
        "weights": None,
        "backend": "auto",          # auto | torch | openvino
        "device": "auto",           # auto | cpu | cuda:0
        "imgsz": 640,
        "aux_imgsz": 512,            # 0 disables the close-range second pass
        "conf": 0.10,               # low: ByteTrack wants the weak boxes too
        "iou": 0.55,
    },
    "tracker": {
        "high_thresh": 0.35,
        "low_thresh": 0.10,
        "match_thresh": 0.80,
        "second_match_thresh": 0.50,
        "max_time_lost": 30,
        "min_hits": 3,
    },
    "pipeline": {
        "analysis_fps": 8.0,        # detector cadence, not video fps
        "resize_long_side": 960,
        "warmup_seconds": 3.0,      # let baselines populate before judging
        "max_seconds": None,
    },
    "signals": {
        "bg_short_s": 20.0,         # Aboah: ideal-conditions window
        "bg_long_s": 120.0,         # Aboah: intersection/night window
        "bg_sample_hz": 2.0,        # denser: short clips must build a background fast
        "bg_rebuild_s": 1.5,
        "bg_scale": 0.5,
        "cp_window": 5,             # ACCIDENT modular baseline: smoothing w
        "cp_z": 1.5,                # ACCIDENT modular baseline: z threshold
        "cp_flow_scale": 0.25,
        "shift_tolerance_px": 12.0,
    },
    "wrong_way": {
        "alignment_threshold": 0.60,
        "min_displacement_px": 18.0,
        "direction_window_s": 1.5,
        "min_speed_px": 3.0,
        "min_persistence_s": 1.5,
        "cusum_beta": 0.25,
        "cusum_h": 1.2,
    },
    "lane_violation": {"enabled": True, "cusum_h": 0.6},
    "queue": {
        "min_vehicles": 4,
        "slow_ratio": 0.35,
        "slow_abs_px": 6.0,
        "stopped_ratio": 0.45,
        "min_occupancy": 0.10,
        "stop_speed_px": 3.0,
        "cusum_beta": 0.25,
        "cusum_h": 3.0,
        "update_interval_s": 5.0,
    },
    "blockage": {
        "stop_speed_px": 2.5,
        "stop_displacement_px": 12.0,
        "min_stationary_s": 12.0,
        "flow_drop_threshold": 0.35,
        "require_long_background": True,
        "cusum_beta": 0.3,
        "cusum_h": 2.5,
    },
    "pedestrian": {"enabled": True, "min_dwell_s": 10.0},
    "collision": {
        "fixed_camera_min_channels": 2,
        "candidate_score_model": None,
        "proximity_px": 90.0,
        "min_approach_angle_deg": 20.0,
        "decel_px_s2": -25.0,
        "min_approach_speed_px": 8.0,
        "cusum_beta": 0.25,
        "cusum_h": 2.2,
        "changepoint_weight": 0.5,
        "cooldown_s": 45.0,
        "peak_onset_lead_s": 0.6,
        "stationary_gate": 0.42,
        "max_pairs": 400,
        "confirm_window_s": 6.0,
        "stop_hold_s": 2.0,
        "min_convergence_ratio": 3.0,
        "min_start_separation": 2.5,
        "stop_ratio": 0.30,
        "deflect_deg": 45.0,
        "proximity_scale": 1.0,
        "max_speed_frac": 0.45,
        "min_confirm_delay_s": 0.4,
        # Rotation-gated pair scorer (netra/events/rotation_gate.py), ported
        # from COMBINED's inference2. Not yet wired into the per-frame loop
        # above; this mirrors config/config.yaml so
        # RotationGateConfig.from_config(config["collision"]) has a real
        # section to read even when no YAML override is present. Values are
        # the tuned COMBINED defaults, verbatim -- see the dataclass and its
        # module docstring for what each one means and the honesty caveat.
        "rotation_gate": {
            "min_track_frames": 8,
            "min_travel_diagonals": 1.0,
            "min_max_speed_kmh": 5.0,
            "contact_window_s": 1.0,
            "heading_lookback_s": 1.0,
            "min_speed_for_heading": 6.0,
            "stable_heading_std_deg": 25.0,
            "stable_aspect_rel_std": 0.30,
            "ref_yaw_deg": 45.0,
            "ref_aspect": 0.60,
            "ref_decel": 30.0,
            "ref_speed_drop_kmh": 20.0,
            "ref_momentum": 12000.0,
            "rotation_floor": 0.20,
            "break_implies_rotation": 0.70,
            "break_floor_score": 0.55,
            "track_break_window_s": 0.4,
            "track_break_min_life_s": 1.0,
            "pair_ref_gap": 0.5,
            "pair_max_gap": 1.5,
            "following_max_deg": 45.0,
            "oncoming_min_deg": 160.0,
            "prior_crossing": 1.00,
            "prior_following": 0.55,
            "prior_oncoming": 0.50,
            "prior_unknown": 0.60,
            "weights": {
                "rotation": 0.45,
                "decel": 0.15,
                "speed_drop": 0.15,
                "momentum": 0.10,
                "appearance": 0.35,
                "track_break": 0.25,
            },
            "max_pairs": 400,
        },
    },
    "stationary": {
        # background-image detection of stopped/crashed vehicles -- the method
        # shared by every winning AI City anomaly entry
        "interval_s": 1.5,
        "conf": 0.20,
        "min_dwell_s": 2.0,
    },
    "crash_classifier": {
        # yolo11n-cls fine-tuned on ACCIDENT crops. Threshold set for precision
        # (~0.95 P / ~0.85 R measured) because over-tagging was the dominant
        # complaint: a wrong box accuses an uninvolved vehicle.
        "weights": None,
        "threshold": 0.718,
    },
    "onset": {"max_backtrack_s": 15.0},
    "evidence": {
        "buffer_seconds": 30.0,
        "scale": 0.6,
        "pre_roll_s": 6.0,
        "post_roll_s": 6.0,
    },
    "paths": {
        "cameras": "config/cameras",
        "database": "netra.db",
        "evidence": "evidence",
        "reports": "reports",
        "road_graph": "config/road_graph.json",
    },
    "api": {"host": "127.0.0.1", "port": 8000},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = "config/config.yaml") -> dict:
    """Defaults merged with the YAML file, if it exists."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        return cfg
    p = Path(path)
    if p.exists():
        user = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = _deep_merge(cfg, user)
    return cfg


def save_default_config(path: str | Path = "config/config.yaml") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(DEFAULTS, sort_keys=False), encoding="utf-8")
    return p
