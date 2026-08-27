"""Traffic-impact severity.

The organiser's brief asks for severity. The honest response is to be precise
about *which* severity, because there are two and only one is observable.

**Not** injury severity. Nothing in an RGB traffic camera supports a claim about
whether anyone was hurt, who was at fault, or what medical response is needed.
NETRA never estimates that and says so on the dashboard.

**Yes** traffic-impact severity: how much of the road's function this event has
taken away, for how long, and how exposed other road users are to it. That is
observable, and it is what a traffic control room actually acts on. It also
matches the factors US FHWA incident-management guidance uses to grade
incidents -- duration, lane closures, traffic impact and complexity -- so the
choice of variables is grounded even though the weights are ours.

The model
---------
Five normalised components, all in [0, 1]:

    F  flow loss        (v_baseline - v_event) / v_baseline
    O  obstruction      affected corridors / configured corridors
    E  extent           affected vehicles / reference count
    D  duration         event duration / reference duration
    R  risk proxy       event-specific exposure term

combined as

    I = 0.40 F + 0.35 O + 0.25 E          (immediate impact)
    S = 0.60 I + 0.25 D + 0.15 R          (overall severity)

    S < 0.35  Low     0.35 <= S < 0.65  Medium     S >= 0.65  High

These weights are **our engineering model, calibrated on public footage. They
are not an ELCIA or FHWA standard**, and every report NETRA produces says so.
What is defensible is the choice to build severity out of measured impact
rather than out of model confidence.

Confidence is kept strictly out of this calculation. A blocked carriageway is
severe whether or not the detector is sure about it; folding uncertainty in
would rank a certain-but-trivial event above an uncertain-but-critical one.
"""

from __future__ import annotations

import numpy as np

from .events.base import (
    BLOCKAGE,
    COLLISION,
    LANE_VIOLATION,
    PEDESTRIAN,
    QUEUE,
    STOPPED,
    WRONG_WAY,
)

# component weights -- our proposed model, stated as such everywhere it appears
W_IMPACT = {"flow": 0.40, "obstruction": 0.35, "extent": 0.25}
W_TOTAL = {"impact": 0.60, "duration": 0.25, "risk": 0.15}

BANDS = ((0.35, "Low"), (0.65, "Medium"), (1.01, "High"))

# normalisation references. Camera-dependent in principle; these are the
# defaults used when a camera has not been individually tuned.
REF_VEHICLES = 12.0
REF_DURATION_S = 180.0
REF_OPPOSING = 8.0


def _band(score: float) -> str:
    for limit, label in BANDS:
        if score < limit:
            return label
    return "High"


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def compute(event, scene, ctx=None) -> tuple[float, str, dict]:
    """Score one event. Returns ``(severity, label, components)``."""
    t = event.triggers or {}
    n_corridors = max(1, len(scene.corridors))

    flow = _flow_component(event, t)
    obstruction = _obstruction_component(event, t, n_corridors)
    extent = _extent_component(event, t)
    duration = _clip01(event.duration / REF_DURATION_S)
    risk = _risk_component(event, t, scene)

    impact = (W_IMPACT["flow"] * flow
              + W_IMPACT["obstruction"] * obstruction
              + W_IMPACT["extent"] * extent)
    severity = (W_TOTAL["impact"] * impact
                + W_TOTAL["duration"] * duration
                + W_TOTAL["risk"] * risk)
    severity = _clip01(severity)

    components = {
        "flow_loss": round(flow, 4),
        "obstruction": round(obstruction, 4),
        "extent": round(extent, 4),
        "duration": round(duration, 4),
        "risk": round(risk, 4),
        "impact_subscore": round(impact, 4),
        "severity": round(severity, 4),
    }
    return severity, _band(severity), components


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------

def _flow_component(event, t) -> float:
    """How much speed the corridor lost, relative to its own free-flow norm."""
    if "speed_drop_pct" in t and t["speed_drop_pct"] is not None:
        return _clip01(float(t["speed_drop_pct"]) / 100.0)
    if "flow_drop_pct" in t and t["flow_drop_pct"] is not None:
        return _clip01(float(t["flow_drop_pct"]) / 100.0)
    if event.type in (WRONG_WAY, LANE_VIOLATION):
        # a counter-flow vehicle does not itself slow the corridor; its harm is
        # in the risk term, not the flow term
        return 0.25
    if event.type == COLLISION:
        return 0.55
    return 0.0


def _obstruction_component(event, t, n_corridors) -> float:
    """Share of the carriageway rendered unusable."""
    if "corridor_obstruction" in t and t["corridor_obstruction"] is not None:
        direct = _clip01(float(t["corridor_obstruction"]) * 2.5)
        share = 1.0 / n_corridors
        return _clip01(max(direct, share))
    if "occupancy" in t and t["occupancy"] is not None:
        return _clip01(float(t["occupancy"]) * 1.5)
    if event.type == COLLISION:
        return _clip01(1.5 / n_corridors)
    if event.type in (WRONG_WAY, LANE_VIOLATION):
        return _clip01(0.6 / n_corridors)
    return 0.0


def _extent_component(event, t) -> float:
    """How many road users are caught up in it."""
    for key in ("vehicles_peak", "vehicles", "opposing_traffic"):
        if key in t and t[key] is not None:
            return _clip01(float(t[key]) / REF_VEHICLES)
    if event.track_ids:
        return _clip01(len(event.track_ids) / REF_VEHICLES)
    return 0.0


def _risk_component(event, t, scene) -> float:
    """Event-specific exposure of other road users to harm.

    This is the term that stops the model from treating a wrong-way vehicle on a
    busy road as a minor event just because it has not yet slowed anyone down.
    """
    if event.type == WRONG_WAY:
        opposing = float(t.get("opposing_traffic", 0) or 0)
        speed = float(t.get("speed_px_s", 0) or 0)
        exposure = _clip01(opposing / REF_OPPOSING)
        velocity = _clip01(speed / 40.0)
        return _clip01(0.55 + 0.30 * exposure + 0.15 * velocity)

    if event.type == LANE_VIOLATION:
        return 0.45

    if event.type == COLLISION:
        base = 0.75
        if t.get("involves_vulnerable"):
            base = 0.95            # pedestrian or cyclist involved
        if t.get("either_stopped"):
            base = min(1.0, base + 0.05)
        return base

    if event.type == PEDESTRIAN:
        return 0.85

    if event.type == BLOCKAGE:
        base = 0.45
        if t.get("in_no_stop_zone"):
            base += 0.15
        if float(t.get("corridor_obstruction", 0) or 0) > 0.3:
            base += 0.20
        return _clip01(base)

    if event.type == STOPPED:
        return 0.35

    if event.type == QUEUE:
        growth = float(t.get("growth_veh_per_min", 0) or 0)
        return _clip01(0.20 + 0.05 * max(0.0, growth))

    return 0.3


# --------------------------------------------------------------------------
# response policy
# --------------------------------------------------------------------------

RECOMMENDATIONS = {
    WRONG_WAY: {
        "High": "Immediate traffic-control escalation; warn opposing corridor",
        "Medium": "Escalate to traffic control; monitor for repeat",
        "Low": "Log and monitor",
    },
    COLLISION: {
        "High": "Operator verification, then emergency and traffic response",
        "Medium": "Operator verification required before dispatch",
        "Low": "Operator review",
    },
    BLOCKAGE: {
        "High": "Dispatch road-response team; protect affected lane; consider diversion",
        "Medium": "Dispatch road-response team",
        "Low": "Monitor; verify whether the stop is legitimate",
    },
    QUEUE: {
        "High": "Traffic-control review; consider signal retiming or diversion",
        "Medium": "Traffic-control review",
        "Low": "Monitor",
    },
    LANE_VIOLATION: {
        "High": "Enforcement review", "Medium": "Enforcement review", "Low": "Log",
    },
    PEDESTRIAN: {
        "High": "Immediate safety escalation", "Medium": "Safety escalation",
        "Low": "Monitor",
    },
    STOPPED: {
        "High": "Dispatch enforcement", "Medium": "Enforcement review", "Low": "Log",
    },
}


def recommend(event) -> str:
    context = (getattr(event, "operational_context", None) or {}).get("classification")
    if context == "suspected_accident_related_congestion":
        return ("Verify collision evidence; if confirmed, dispatch incident response "
                "and consider diversion for the affected corridor")
    if context == "obstruction_related_congestion":
        return ("Dispatch road-response review; protect the affected lane and "
                "consider diversion if the obstruction persists")
    table = RECOMMENDATIONS.get(event.type)
    if not table:
        return "Operator review"
    return table.get(event.severity_label, "Operator review")


DISCLAIMER = (
    "Severity is traffic-impact severity computed from observable variables "
    "(flow loss, obstruction, extent, duration, exposure). It is NOT injury or "
    "casualty severity, which cannot be inferred from RGB video. Weights are a "
    "proposed engineering model calibrated on public footage, not a published "
    "standard."
)
