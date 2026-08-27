"""Blockage, abnormal stopping, and pedestrians on the carriageway.

A stationary vehicle is not an incident. Vehicles stop legally all the time --
at signals, in traffic, at the kerb. Reporting every stationary vehicle would
bury an operator, which is the failure mode that makes these systems get
switched off.

A blockage is a stationary vehicle **plus evidence that traffic is impaired**:

    stationary(j)   displacement < eps for T seconds
    inside a travel corridor, not a legal stop zone
    AND ( upstream flow has dropped  OR  the corridor is substantially blocked )
    AND persistent in the LONG background

That last condition is the one that does the real work, and it comes from Aboah
et al.'s dual-window background. A queue at a red light appears in a 30-second
median but dissolves in a 5-minute one; a broken-down lorry appears in both. It
is the cheapest available discriminator between "waiting" and "stuck", and it
costs a median over a handful of cached frames.

The engine also raises two lighter events from the same machinery:
* ``abnormal_stop`` -- stationary inside an explicitly-marked no-stop zone
* ``pedestrian_on_carriageway`` -- a person track dwelling in a travel corridor
"""

from __future__ import annotations

import numpy as np

from ..detect import MOTORISED_CLASSES, VULNERABLE_CLASSES
from ..geometry import boxes_occupancy
from ..signals import Cusum
from .base import BLOCKAGE, PEDESTRIAN, STOPPED, Event, EventEngine


class BlockageEngine(EventEngine):
    name = "blockage"
    cooldown_s = 120.0        # a stalled vehicle is one incident, not many

    def __init__(self, scene, config: dict):
        super().__init__(scene, config)
        c = config.get("blockage", {})
        self.stop_speed_px = float(c.get("stop_speed_px", 2.5))
        self.stop_displacement_px = float(c.get("stop_displacement_px", 12.0))
        self.min_stationary_s = float(c.get("min_stationary_s", 12.0))
        self.flow_drop_threshold = float(c.get("flow_drop_threshold", 0.35))
        self.cusum_beta = float(c.get("cusum_beta", 0.3))
        self.cusum_h = float(c.get("cusum_h", 2.5))
        self.require_long_background = bool(c.get("require_long_background", True))

        p = config.get("pedestrian", {})
        self.ped_enabled = bool(p.get("enabled", True))
        self.ped_dwell_s = float(p.get("min_dwell_s", 5.0))

        self._cusum: dict[int, Cusum] = {}
        self._flow_ref: dict[str, list[float]] = {}
        self._ped_cusum: dict[int, Cusum] = {}

    # ------------------------------------------------------------------
    def update(self, ctx) -> list[Event]:
        raised: list[Event] = []
        if not ctx.geometry_valid:
            return raised

        self._update_flow_reference(ctx)
        live = set()

        for tr in ctx.tracks:
            live.add(tr.track_id)
            gp = tr.ground_point
            corridor = self.scene.corridor_at(gp)

            if tr.cls in VULNERABLE_CLASSES:
                ev = self._check_pedestrian(tr, corridor, ctx)
                if ev is not None:
                    raised.append(ev)
                continue

            if tr.cls not in MOTORISED_CLASSES:
                continue

            moving = (tr.speed_px(1.0) >= self.stop_speed_px
                      or tr.displacement(3.0) >= self.stop_displacement_px)
            if moving:
                tr.stationary_since = None
                self._decay(tr.track_id, ctx.t)
                continue

            if tr.stationary_since is None:
                tr.stationary_since = ctx.t
            stationary_s = ctx.t - tr.stationary_since

            in_no_stop = self.scene.in_no_stop(gp)
            if corridor is None and not in_no_stop:
                self._decay(tr.track_id, ctx.t)
                continue

            if stationary_s < self.min_stationary_s:
                continue

            if in_no_stop and corridor is None:
                ev = self._stopped_event(tr, ctx, stationary_s)
                if ev is not None:
                    raised.append(ev)
                continue

            flow_drop = self._flow_drop(corridor.id, ctx)
            obstruction = self._obstruction(corridor, tr, ctx)
            persistent = True
            if self.require_long_background and ctx.background is not None:
                persistent = ctx.background.is_persistent(tr.box)

            impaired = (flow_drop >= self.flow_drop_threshold
                        or obstruction >= 0.25
                        or in_no_stop)

            evidence = 0.0
            if impaired and persistent:
                e_t = min(1.0, stationary_s / (self.min_stationary_s * 3.0))
                e_f = min(1.0, flow_drop / max(self.flow_drop_threshold, 1e-6))
                e_o = min(1.0, obstruction / 0.5)
                evidence = float(np.clip(0.4 * e_t + 0.35 * e_f + 0.25 * e_o, 0, 1))

            cus = self._cusum.setdefault(
                tr.track_id, Cusum(beta=self.cusum_beta, h=self.cusum_h, decay=0.05)
            )
            if not cus.update(evidence, ctx.t):
                continue

            key = f"bl:{tr.track_id}"
            if not self.can_fire(key, ctx.t):
                cus.reset()
                continue

            onset = tr.stationary_since
            ev = Event(
                type=BLOCKAGE,
                camera_id=self.scene.camera_id,
                started_t=onset,
                detected_t=ctx.t,
                confidence=self._confidence(tr, cus, flow_drop, obstruction, persistent),
                corridor_id=corridor.id,
                track_ids=[tr.track_id],
                triggers={
                    "stationary_s": round(stationary_s, 1),
                    "speed_px_s": round(tr.speed_px(1.0), 2),
                    "displacement_px_3s": round(tr.displacement(3.0), 1),
                    "flow_drop_pct": round(flow_drop * 100.0, 1),
                    "flow_drop_threshold_pct": round(self.flow_drop_threshold * 100, 1),
                    "corridor_obstruction": round(obstruction, 4),
                    "in_long_background": bool(persistent),
                    "in_no_stop_zone": bool(in_no_stop),
                    "object_class": int(tr.cls),
                    "cusum_s": round(cus.s, 3),
                    "detector_confidence": round(float(np.mean(tr.score_history)), 3),
                },
            )
            self.register(key, ev, ctx.t)
            raised.append(ev)
            cus.reset()

        for tid in list(self._cusum):
            if tid not in live:
                self._cusum.pop(tid, None)
                self.close(f"bl:{tid}", ctx.t)
        for tid in list(self._ped_cusum):
            if tid not in live:
                self._ped_cusum.pop(tid, None)
        return raised

    # ------------------------------------------------------------------
    def _decay(self, track_id: int, t: float) -> None:
        cus = self._cusum.get(track_id)
        if cus is not None:
            cus.update(0.0, t)

    def _update_flow_reference(self, ctx) -> None:
        """Maintain a rolling free-flow throughput reference per corridor."""
        for corridor in self.scene.corridors:
            members = [tr for tr in ctx.tracks
                       if tr.corridor_id == corridor.id and tr.cls in MOTORISED_CLASSES]
            speeds = [tr.speed_px(1.0) for tr in members]
            buf = self._flow_ref.setdefault(corridor.id, [])
            if len(members) >= 2:
                buf.append(float(np.median(speeds)))
                if len(buf) > 600:
                    del buf[: len(buf) - 600]

    def _flow_drop(self, corridor_id: str, ctx) -> float:
        """Relative loss of median corridor speed against its own recent norm."""
        buf = self._flow_ref.get(corridor_id, [])
        if len(buf) < 30:
            return 0.0
        reference = float(np.percentile(buf[:-10] if len(buf) > 40 else buf, 80))
        recent = float(np.median(buf[-10:]))
        if reference <= 1e-6:
            return 0.0
        return float(np.clip((reference - recent) / reference, 0.0, 1.0))

    def _obstruction(self, corridor, tr, ctx) -> float:
        """Fraction of the corridor this stationary object covers."""
        return boxes_occupancy([tr.box], corridor.polygon, ctx.frame_shape)

    @staticmethod
    def _confidence(tr, cus, flow_drop, obstruction, persistent) -> float:
        p = min(1.0, cus.s / max(2 * cus.h, 1e-6))
        d = float(np.mean(tr.score_history)) if tr.score_history else 0.5
        bg = 1.0 if persistent else 0.4
        impact = min(1.0, max(flow_drop, obstruction * 2.0))
        return float(np.clip(0.30 * p + 0.25 * d + 0.25 * bg + 0.20 * impact, 0, 0.97))

    def _stopped_event(self, tr, ctx, stationary_s) -> Event | None:
        cus = self._cusum.setdefault(
            tr.track_id, Cusum(beta=self.cusum_beta, h=self.cusum_h, decay=0.05)
        )
        if not cus.update(min(1.0, stationary_s / (self.min_stationary_s * 2)), ctx.t):
            return None
        ev = Event(
            type=STOPPED,
            camera_id=self.scene.camera_id,
            started_t=tr.stationary_since or ctx.t,
            detected_t=ctx.t,
            confidence=float(np.clip(0.5 + 0.4 * min(1.0, cus.s / max(2 * cus.h, 1e-6)), 0, 0.95)),
            corridor_id=None,
            track_ids=[tr.track_id],
            triggers={
                "stationary_s": round(stationary_s, 1),
                "zone": "no_stop",
                "detector_confidence": round(float(np.mean(tr.score_history)), 3),
            },
        )
        cus.reset()
        return ev

    def _check_pedestrian(self, tr, corridor, ctx) -> Event | None:
        """A person inside a travel corridor is a safety event in its own right."""
        if not self.ped_enabled or corridor is None:
            return None
        if self.scene.in_exclusion(tr.ground_point):
            return None
        cus = self._ped_cusum.setdefault(tr.track_id, Cusum(beta=0.2, h=1.0, decay=0.15))
        if not cus.update(1.0, ctx.t):
            return None
        dwell = ctx.t - (cus.first_evidence_t or ctx.t)
        if dwell < self.ped_dwell_s:
            cus.alarmed = False
            return None
        key = f"pd:{tr.track_id}"
        if not self.can_fire(key, ctx.t):
            cus.reset()
            return None
        self.register(key, None, ctx.t)
        ev = Event(
            type=PEDESTRIAN,
            camera_id=self.scene.camera_id,
            started_t=cus.first_evidence_t or ctx.t,
            detected_t=ctx.t,
            confidence=float(np.clip(0.45 + 0.5 * min(1.0, tr.hits / 20.0), 0, 0.95)),
            corridor_id=corridor.id,
            track_ids=[tr.track_id],
            triggers={
                "dwell_s": round(dwell, 2),
                "object_class": int(tr.cls),
                "detector_confidence": round(float(np.mean(tr.score_history)), 3),
            },
        )
        self.active[key] = ev
        cus.reset()
        return ev
