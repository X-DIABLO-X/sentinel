"""Drone operating modes.

Only one mode is implemented, and that is a deliberate architectural decision
rather than an unfinished feature. This module exists to make the decision
explicit in code, so that anyone who reaches for patrol mode gets the reasoning
in the traceback instead of silently building on an unsupported assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperatingMode(str, Enum):
    """How the drone is being flown for a given mission.

    HOVER
        Implemented. The drone is dispatched to an incident that CCTV has
        **already confirmed**, flies to it, holds station above it, and
        provides the overhead view for verification and extent measurement.

    PATROL
        Not implemented. Raises NotImplementedError. See
        :func:`assert_supported` for why.
    """

    HOVER = "hover"
    PATROL = "patrol"

    @property
    def implemented(self) -> bool:
        return self is OperatingMode.HOVER


@dataclass(frozen=True)
class ModeProfile:
    """What a mode implies for the vision stack."""
    mode: OperatingMode
    implemented: bool
    ego_motion: str
    gmc_role: str
    endurance_note: str


HOVER_PROFILE = ModeProfile(
    mode=OperatingMode.HOVER,
    implemented=True,
    ego_motion="small bounded drift (wind, GPS hold error, gimbal micro-correction)",
    gmc_role="small-drift correction against a near-static reference frame",
    endurance_note="single incident, 20-40 min airframe endurance is sufficient",
)


def assert_supported(mode: "OperatingMode | str") -> OperatingMode:
    """Validate a requested operating mode.

    Returns the :class:`OperatingMode` for HOVER. Raises
    ``NotImplementedError`` for PATROL, with the full reasoning below, and
    ``ValueError`` for anything unrecognised.

    Why PATROL is not implemented
    -----------------------------
    Three independent reasons, any one of which is sufficient on its own.

    **1. Regulatory — Indian DGCA does not permit BVLOS for traffic
    monitoring.** Beyond-visual-line-of-sight operation has been approved only
    for narrow specific corridors and for other use cases; traffic surveillance
    is not among them. Standard operations require the remote pilot to maintain
    visual line of sight with the aircraft, under a 120 m AGL altitude cap.
    A patrol route that covers a meaningful stretch of a city road network
    necessarily leaves VLOS. Building a patrol mode would mean building a
    capability that cannot legally be flown, and demonstrating one would mean
    demonstrating a violation. The architecture is shaped to what can actually
    be operated.

    **2. Endurance — a 20-40 minute airframe cannot provide persistent
    coverage.** Continuous patrol implies a duty cycle the platform does not
    have. Covering a corridor persistently would need a fleet plus battery-swap
    logistics, and even then the revisit interval would be far worse than the
    CCTV network already achieves at zero marginal cost. The drone earns its
    place by seeing what a pole-mounted camera *cannot* — overhead extent,
    queue tail beyond the CCTV field of view, blockage geometry — not by
    duplicating coverage CCTV already has. Hence: CCTV is the persistent
    backbone, the drone is escalation and verification.

    **3. Vision — unbounded ego-motion drift.** This is the reason that lives
    in this codebase rather than in the ops manual. In hover, the frame-to-
    frame background homography is a *small correction* about a near-static
    reference: the scene barely changes, features persist across many frames,
    and the accumulated chain error stays bounded because the reference frame
    stays in view. Under patrol the scene translates continuously, the
    reference frame leaves the field of view within seconds, and the homography
    chain must be re-referenced constantly. Chain error then accumulates
    without bound, and every re-reference injects a discontinuity. Speed
    estimates derived from that chain degrade from "measurement" to "guess"
    with no visible symptom in the output — the failure mode this project's
    honesty rules exist to prevent.

    Patrol becomes tractable if, and only if, per-frame telemetry-assisted
    direct georeferencing is available (fused GPS + IMU + gimbal pose), because
    that provides an absolute reference that does not accumulate. See
    ``telemetry_ingest.py`` — not implemented, no hardware, signature frozen.
    """
    if isinstance(mode, str):
        try:
            mode = OperatingMode(mode.strip().lower())
        except ValueError:
            valid = ", ".join(m.value for m in OperatingMode)
            raise ValueError(
                f"unknown operating mode {mode!r}; expected one of: {valid}"
            ) from None

    if mode is OperatingMode.HOVER:
        return mode

    if mode is OperatingMode.PATROL:
        raise NotImplementedError(
            "PATROL mode is not implemented, and this is a deliberate "
            "architectural decision, not a missing feature.\n"
            "  (1) REGULATORY: Indian DGCA does not approve BVLOS for traffic "
            "monitoring (only narrow corridors for other uses). Standard ops "
            "require visual line of sight and a 120 m AGL cap; a useful patrol "
            "route cannot stay inside that envelope.\n"
            "  (2) ENDURANCE: 20-40 min airframe endurance cannot deliver "
            "persistent coverage. CCTV is the persistent backbone; the drone "
            "is dispatched escalation/verification for an already-confirmed "
            "incident.\n"
            "  (3) VISION: continuous translation makes background-homography "
            "ego-motion compensation an unbounded-drift problem instead of a "
            "small bounded correction. The reference frame leaves the field of "
            "view, the chain must be re-referenced constantly, and speed "
            "estimates silently degrade.\n"
            "See hover_mode.assert_supported.__doc__ for the full reasoning."
        )

    raise ValueError(f"unhandled operating mode: {mode!r}")


def profile_for(mode: "OperatingMode | str") -> ModeProfile:
    """Return the vision-stack profile for a supported mode."""
    assert_supported(mode)
    return HOVER_PROFILE


def describe() -> dict:
    """Machine-readable mode support, for /api/status and results files."""
    return {
        "modes": {
            OperatingMode.HOVER.value: {
                "implemented": True,
                "ego_motion": HOVER_PROFILE.ego_motion,
                "gmc_role": HOVER_PROFILE.gmc_role,
            },
            OperatingMode.PATROL.value: {
                "implemented": False,
                "reason": (
                    "DGCA does not approve BVLOS for traffic monitoring; "
                    "20-40 min endurance cannot give persistent coverage; "
                    "continuous translation makes ego-motion drift unbounded"
                ),
            },
        },
        "active_default": OperatingMode.HOVER.value,
    }


if __name__ == "__main__":   # pragma: no cover - manual smoke check
    print("hover ->", assert_supported("hover"))
    try:
        assert_supported("patrol")
    except NotImplementedError as exc:
        print("patrol -> NotImplementedError:\n" + str(exc))
