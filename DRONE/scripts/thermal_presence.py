"""Thermal vehicle-PRESENCE detection — total darkness only.

=============================================================================
STATUS: STUB. ``detect_presence_thermal()`` returns an empty list, always.
No thermal sensor, no thermal footage, no algorithm implemented.
=============================================================================

Scope, and it is deliberately narrow
------------------------------------
Thermal has exactly **one** sanctioned role in this system:

    In total darkness, answer the single binary question
    "is there a warm object on the roadway at all?"

That is the entire remit. Specifically, thermal output in this system is
**never** used for:

* **Classification.** It cannot tell you what a vehicle is. An 8-14 um
  microbolometer image of a car from 100 m AGL is a warm blob whose shape is
  dominated by the engine bay, exhaust and tyres, not by the vehicle's
  geometry. A hatchback with a hot engine and a van with a cold one can present
  identically. Car vs van vs truck vs autorickshaw is not recoverable, and the
  class taxonomy is what every downstream incident rule depends on.
* **Motion or speed.** Low native resolution (typically 640x512 or less, often
  320x256 on a drone payload), heavy fixed-pattern noise, and non-uniformity
  correction (NUC) shutter events that blank or shift the whole frame make
  frame-to-frame displacement unreliable. Worse for this project specifically:
  GMC needs sharp static background features — road markings, kerb edges, lane
  paint. Those are *thermally invisible*. White paint and the asphalt beside it
  sit at the same temperature. Ego-motion compensation, which everything in
  ``gmc.py`` depends on, has nothing to lock onto in a thermal frame.
* **Severity.** Severity in this system is a function of vehicle count, class
  mix, queue extent and stationarity duration. Every one of those inputs is
  either unavailable or unreliable from thermal, so a severity number derived
  from it would be a fabricated number wearing a measurement's clothes.

Why the narrow role is still worth having
-----------------------------------------
In genuine darkness with no street lighting, the RGB detector's recall goes to
approximately zero, and the system cannot distinguish "empty road" from
"blocked road, cameras blind". Those two states demand opposite responses.
A thermal presence trigger resolves that ambiguity: it says *something warm is
on the carriageway, escalate to a human operator / task the drone's lights*,
without pretending to say what.

So: thermal is a **trigger**, not a **sensor input to the incident model**.
It escalates. It never classifies, never measures, never scores.

Consistency with the settled sensor decision
--------------------------------------------
Core detection is RGB optical only. mmWave radar, LiDAR, stereo, SWIR, NIR,
event cameras and hyperspectral were all ruled out on physics grounds for this
altitude, range and task. Thermal's single narrow presence-trigger role above
is the one carve-out, and this module is the only place it is allowed to live.

Implementation notes for whoever writes the real version
--------------------------------------------------------
Not written here, but the shape is not mysterious, and none of it should be
mistaken for existing code:

* Radiometric input if available (16-bit), not the 8-bit colourised preview.
  The colour palette is a display artefact and destroys the actual signal.
* Handle NUC shutter events explicitly. Frames during a shutter are invalid,
  not merely noisy, and must be dropped rather than thresholded.
* Adaptive thresholding against a rolling background temperature estimate, not
  a fixed threshold. Asphalt at midnight in May and asphalt at midnight in
  December differ by tens of degrees; a fixed threshold is a seasonal bug.
* Morphological open/close then connected components, with a minimum blob area
  gated on the ground sampling distance at the current altitude.
* Output boxes only. No class, no score that implies classification confidence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("drone.thermal")

__all__ = [
    "THERMAL_IMPLEMENTED",
    "ALLOWED_USES",
    "FORBIDDEN_USES",
    "detect_presence_thermal",
    "thermal_status",
]

# Single source of truth for the honesty flag. Do not flip without a working
# implementation validated on real thermal footage.
THERMAL_IMPLEMENTED: bool = False

ALLOWED_USES: tuple[str, ...] = (
    "total_darkness_vehicle_presence_trigger",
)

FORBIDDEN_USES: tuple[str, ...] = (
    "classification",
    "vehicle_type",
    "motion_estimation",
    "speed_estimation",
    "tracking",
    "severity_scoring",
    "queue_length",
    "ego_motion_compensation",
)

_WARNED = False


def detect_presence_thermal(frame: np.ndarray | None,
                            *,
                            min_blob_area_px: int = 24) -> list[list[float]]:
    """Detect warm-object PRESENCE in a thermal frame.

    **STUB — returns ``[]`` unconditionally. Nothing is implemented.**

    Returns
    -------
    list of ``[x1, y1, x2, y2]``
        Presence boxes in pixels. Deliberately carries **no class and no
        score**: this function cannot classify and must not return a field that
        could be mistaken for a classification confidence. Empty today.

    Contract for callers
    --------------------
    The returned boxes mean exactly one thing: *something warm is present at
    this image location*. They may be used to raise a
    ``low_light_presence_detected`` escalation flag. They must not be fed to
    the tracker, the kinematics stage, the classifier or the severity model.
    Passing them anywhere else is a correctness bug, not a degradation.

    This is enforced by convention rather than by types, so it is written here
    in the docstring and again in ``FORBIDDEN_USES`` above.
    """
    global _WARNED
    if not _WARNED:
        log.warning(
            "THERMAL PRESENCE DETECTION IS A STUB - returning no detections. "
            "No thermal sensor, no thermal footage, no algorithm. "
            "Scope when implemented is presence-only in total darkness: "
            "never classification, never motion/speed, never severity."
        )
        _WARNED = True

    if frame is not None and not isinstance(frame, np.ndarray):
        raise TypeError(f"thermal frame must be ndarray or None, got {type(frame).__name__}")

    return []


def thermal_status() -> dict[str, Any]:
    """Machine-readable status, embedded in API responses and results files."""
    return {
        "implemented": THERMAL_IMPLEMENTED,
        "available": False,
        "role": "total-darkness vehicle-PRESENCE trigger only",
        "allowed_uses": list(ALLOWED_USES),
        "forbidden_uses": list(FORBIDDEN_USES),
        "note": (
            "Thermal cannot classify (a car is an engine-bay-dominated warm "
            "blob, not a vehicle silhouette), cannot support GMC (lane paint "
            "and asphalt are thermally identical, so there are no static "
            "background features to match), and therefore cannot feed "
            "severity. It escalates; it does not measure."
        ),
    }


if __name__ == "__main__":   # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.INFO)
    print("detect_presence_thermal(None) ->", detect_presence_thermal(None))
    print("status ->", thermal_status())
