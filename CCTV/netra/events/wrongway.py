"""Wrong-side movement and wrong lane crossing.

Both are *relations between a trajectory and the road*, not properties of a
vehicle's appearance. No photograph of a car can tell you it is driving
illegally; you need to know which way that lane is supposed to flow. That fact
lives in the scene model, put there once by a human, and the decision is then a
dot product.

The rule
--------
For a track in corridor L with legal direction ``d_L`` and observed direction
``d_j``:

    alignment  c = (d_j . d_L) / (|d_j| |d_L|)
    violation  c < -tau            sustained for T seconds
               and displacement > minimum

``tau`` defaults to 0.60 (i.e. more than ~127 degrees off the legal heading), so
a lane change or a drift does not qualify -- only genuine counter-flow.

Three guards keep this honest, and each one exists because of a specific,
documented failure mode:

* **Exclusion zones.** Inside a junction box every legal turn briefly looks like
  counter-flow. Direction logic is suspended there.
* **Displacement floor.** A stationary vehicle's direction is pure noise; it
  must actually travel before we will judge it.
* **CUSUM rather than a frame counter.** The 2026 YOLOv9 wrong-way study reports
  false positives from legal U-turns near the frame edge; requiring accumulated
  evidence rather than N consecutive frames absorbs those.

Lane violation reuses the same machinery: a corridor-to-corridor transition is
a violation only when the scene model marks that boundary as a solid line.
"""

from __future__ import annotations

import numpy as np

from ..geometry import heading_degrees
from ..signals import Cusum
from .base import LANE_VIOLATION, WRONG_WAY, Event, EventEngine


class WrongWayEngine(EventEngine):
    name = "wrong_way"
    cooldown_s = 45.0

    def __init__(self, scene, config: dict):
        super().__init__(scene, config)
        c = config.get("wrong_way", {})
        self.tau = float(c.get("alignment_threshold", 0.60))
        self.min_displacement = float(c.get("min_displacement_px", 18.0))
        self.direction_window = float(c.get("direction_window_s", 1.5))
        self.cusum_beta = float(c.get("cusum_beta", 0.25))
        self.cusum_h = float(c.get("cusum_h", 1.2))
        self.min_speed_px = float(c.get("min_speed_px", 3.0))
        # A CUSUM alone can trip in two frames. Counter-flow is only meaningful
        # if it is sustained, so an explicit wall-clock floor is required as
        # well -- this is the guard against tracking jitter and brief swerves.
        self.min_persistence_s = float(c.get("min_persistence_s", 1.5))

        lc = config.get("lane_violation", {})
        self.lane_enabled = bool(lc.get("enabled", True))
        self.lane_cusum_h = float(lc.get("cusum_h", 0.6))

        self._cusum: dict[int, Cusum] = {}
        self._last_corridor: dict[int, str] = {}
        self._lane_events: dict[int, float] = {}

    # ------------------------------------------------------------------
    def update(self, ctx) -> list[Event]:
        raised: list[Event] = []
        if not ctx.geometry_valid:
            # camera has moved: our polygons no longer describe this road, so
            # refuse to emit geometric events rather than emit wrong ones
            return raised

        live_ids = set()
        for tr in ctx.tracks:
            live_ids.add(tr.track_id)
            gp = tr.ground_point

            if self.scene.in_exclusion(gp):
                self._decay(tr.track_id, ctx.t)
                self._last_corridor.pop(tr.track_id, None)
                continue

            corridor = self.scene.corridor_at(gp)
            if corridor is None:
                self._decay(tr.track_id, ctx.t)
                continue
            tr.corridor_id = corridor.id

            ev = self._check_lane_violation(tr, corridor, ctx)
            if ev is not None:
                raised.append(ev)

            direction = tr.direction(self.direction_window, min_span=6.0)
            if direction is None or tr.speed_px(1.0) < self.min_speed_px:
                self._decay(tr.track_id, ctx.t)
                continue

            alignment = corridor.alignment(direction)
            evidence = max(0.0, (-alignment - self.tau) / max(1.0 - self.tau, 1e-6))

            cus = self._cusum.setdefault(
                tr.track_id, Cusum(beta=self.cusum_beta, h=self.cusum_h, decay=0.08)
            )
            fired = cus.update(evidence, ctx.t)
            tr.wrongway_evidence = cus.s

            if not fired:
                continue

            onset = cus.first_evidence_t if cus.first_evidence_t is not None else ctx.t
            persistence = ctx.t - onset
            if persistence < self.min_persistence_s:
                # keep accumulating rather than resetting: the violation may
                # well be real, it simply has not lasted long enough yet
                cus.alarmed = False
                continue

            displacement = tr.displacement(self.direction_window * 2)
            if displacement < self.min_displacement:
                cus.reset()
                continue

            key = f"ww:{tr.track_id}"
            if not self.can_fire(key, ctx.t):
                cus.reset()
                continue
            ev = Event(
                type=WRONG_WAY,
                camera_id=self.scene.camera_id,
                started_t=onset,
                detected_t=ctx.t,
                confidence=self._confidence(tr, alignment, cus, ctx),
                corridor_id=corridor.id,
                track_ids=[tr.track_id],
                triggers={
                    "alignment": round(float(alignment), 4),
                    "alignment_threshold": -self.tau,
                    "observed_heading_deg": round(heading_degrees(direction), 1),
                    "legal_heading_deg": round(corridor.heading, 1),
                    "legal_direction_reviewed": self.scene.legal_direction_reviewed,
                    "direction_source": ("human-reviewed legal direction"
                                         if self.scene.legal_direction_reviewed
                                         else "observed majority flow; legal direction unreviewed"),
                    "persistence_s": round(ctx.t - onset, 2),
                    "displacement_px": round(displacement, 1),
                    "speed_px_s": round(tr.speed_px(1.0), 2),
                    "cusum_s": round(cus.s, 3),
                    "cusum_h": cus.h,
                    "detector_confidence": round(float(np.mean(tr.score_history)), 3),
                    "opposing_traffic": self._opposing_count(corridor.id, ctx),
                },
            )
            self.register(key, ev, ctx.t)
            raised.append(ev)
            cus.reset()

        for tid in list(self._cusum):
            if tid not in live_ids:
                self._cusum.pop(tid, None)
                self._last_corridor.pop(tid, None)
                self.close(f"ww:{tid}", ctx.t)
        return raised

    # ------------------------------------------------------------------
    def _decay(self, track_id: int, t: float) -> None:
        cus = self._cusum.get(track_id)
        if cus is not None:
            cus.update(0.0, t)

    def _confidence(self, tr, alignment, cus, ctx) -> float:
        """Evidence-weighted confidence, not a calibrated probability.

        Four independent things have to agree: how far from legal the heading
        is, how long it persisted, how sure the detector was, and how mature the
        track is. Reported as an evidence score, and the writeup says so.
        """
        a = min(1.0, (-alignment - self.tau) / max(1.0 - self.tau, 1e-6))
        p = min(1.0, cus.s / max(2.0 * cus.h, 1e-6))
        d = float(np.mean(tr.score_history)) if tr.score_history else 0.5
        m = min(1.0, tr.hits / 15.0)
        return float(np.clip(0.35 * a + 0.30 * p + 0.20 * d + 0.15 * m, 0.0, 0.99))

    def _opposing_count(self, corridor_id: str, ctx) -> int:
        """How many vehicles are exposed to this violation right now."""
        n = 0
        for tr in ctx.tracks:
            cid = tr.corridor_id
            if cid and cid != corridor_id and self.scene.is_opposing(corridor_id, cid):
                n += 1
        return n

    def _check_lane_violation(self, tr, corridor, ctx) -> Event | None:
        """Solid-line crossing = violation; dashed-line crossing = lane change."""
        if not self.lane_enabled:
            return None
        prev = self._last_corridor.get(tr.track_id)
        self._last_corridor[tr.track_id] = corridor.id
        if prev is None or prev == corridor.id:
            return None
        if not self.scene.boundary_is_solid(prev, corridor.id):
            return None
        key = f"lv:{tr.track_id}"
        last = self._lane_events.get(tr.track_id, -1e9)
        if ctx.t - last < 15.0:              # one alert per crossing, not per frame
            return None
        self._lane_events[tr.track_id] = ctx.t

        return Event(
            type=LANE_VIOLATION,
            camera_id=self.scene.camera_id,
            started_t=max(0.0, ctx.t - 0.5),
            detected_t=ctx.t,
            confidence=float(np.clip(0.55 + 0.4 * min(1.0, tr.hits / 20.0), 0, 0.95)),
            corridor_id=corridor.id,
            track_ids=[tr.track_id],
            triggers={
                "from_corridor": prev,
                "to_corridor": corridor.id,
                "boundary": "solid",
                "speed_px_s": round(tr.speed_px(1.0), 2),
                "detector_confidence": round(float(np.mean(tr.score_history)), 3),
            },
        )
