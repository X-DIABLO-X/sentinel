"""Queue and congestion detection.

A queue is not "many vehicles in one frame". A car park is full of vehicles and
carries no congestion; one car stopped at a red light is not a queue. The
definition that survives contact with real footage needs three things at once,
sustained over time:

    density   N >= N_min                      enough vehicles present
    slowness  median speed < v_slow           and they are not moving
    coverage  occupancy > O_min               and they fill the carriageway

    ... all three holding for T seconds

Occupancy is computed by rasterising the union of the vehicle boxes against the
corridor polygon rather than summing box areas. In dense traffic boxes overlap
heavily, and summing would exceed 1.0 and silently corrupt the severity score.

Speed is measured in pixels/second and compared against a *learned per-corridor
baseline*, never against an absolute threshold. Perspective means 30 px/s near
the camera and 30 px/s at the vanishing point are wildly different real speeds,
so only the ratio to that corridor's own free-flow speed is meaningful. Without
a homography NETRA never converts this to km/h -- see LIMITATIONS.md.

Queue *growth* is tracked as well as queue existence. "Queue length rose from 6
to 14 vehicles in 40 s" is traffic intelligence; "there are 14 vehicles" is not.
"""

from __future__ import annotations

import numpy as np

from ..detect import MOTORISED_CLASSES
from ..geometry import boxes_occupancy
from ..signals import Cusum
from .base import QUEUE, Event, EventEngine


class QueueEngine(EventEngine):
    name = "queue"

    def __init__(self, scene, config: dict):
        super().__init__(scene, config)
        c = config.get("queue", {})
        self.min_vehicles = int(c.get("min_vehicles", 4))
        self.slow_ratio = float(c.get("slow_ratio", 0.35))       # of baseline
        self.slow_abs_px = float(c.get("slow_abs_px", 6.0))      # fallback
        self.stopped_ratio = float(c.get("stopped_ratio", 0.45))
        self.min_occupancy = float(c.get("min_occupancy", 0.10))
        self.cusum_beta = float(c.get("cusum_beta", 0.25))
        self.cusum_h = float(c.get("cusum_h", 3.0))
        self.stop_speed_px = float(c.get("stop_speed_px", 3.0))
        self.report_every = float(c.get("update_interval_s", 5.0))

        self._cusum: dict[str, Cusum] = {}
        self._extent: dict[str, list[tuple[float, int]]] = {}
        self._last_update: dict[str, float] = {}
        self._baseline_samples: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    def update(self, ctx) -> list[Event]:
        raised: list[Event] = []
        shape = ctx.frame_shape

        for corridor in self.scene.corridors:
            members = [tr for tr in ctx.tracks
                       if tr.corridor_id == corridor.id and tr.cls in MOTORISED_CLASSES]
            n = len(members)

            speeds = [tr.speed_px(1.0) for tr in members]
            median_speed = float(np.median(speeds)) if speeds else 0.0
            stopped = sum(1 for s in speeds if s < self.stop_speed_px)
            stopped_frac = stopped / n if n else 0.0
            occupancy = boxes_occupancy([tr.box for tr in members],
                                        corridor.polygon, shape) if n else 0.0

            self._collect_baseline(corridor, members, speeds, occupancy)

            baseline = corridor.baseline_speed_px
            slow_thresh = (baseline * self.slow_ratio) if baseline else self.slow_abs_px
            is_slow = median_speed < slow_thresh

            candidate = (n >= self.min_vehicles and is_slow
                         and stopped_frac >= self.stopped_ratio
                         and occupancy >= self.min_occupancy)

            evidence = 0.0
            if candidate:
                # scale evidence by how far past each threshold we are, so a
                # severe jam accumulates faster than a marginal one
                e_n = min(1.0, n / max(self.min_vehicles * 2.0, 1.0))
                e_s = min(1.0, (slow_thresh - median_speed) / max(slow_thresh, 1e-6))
                e_o = min(1.0, occupancy / max(self.min_occupancy * 2.0, 1e-6))
                evidence = float(np.clip(0.4 * e_n + 0.3 * e_s + 0.3 * e_o, 0.0, 1.0))

            cus = self._cusum.setdefault(
                corridor.id, Cusum(beta=self.cusum_beta, h=self.cusum_h, decay=0.05)
            )
            fired = cus.update(evidence, ctx.t)

            hist = self._extent.setdefault(corridor.id, [])
            if candidate:
                hist.append((ctx.t, n))
                if len(hist) > 400:
                    del hist[: len(hist) - 400]
            elif cus.s <= 0:
                hist.clear()

            active = self.active.get(corridor.id)

            if fired and active is None:
                onset = cus.first_evidence_t if cus.first_evidence_t is not None else ctx.t
                ev = self._make_event(corridor, ctx, onset, n, median_speed, baseline,
                                      stopped_frac, occupancy, cus, hist)
                self.active[corridor.id] = ev
                self._last_update[corridor.id] = ctx.t
                raised.append(ev)

            elif active is not None:
                if candidate:
                    # keep the live event's numbers current so the dashboard
                    # shows a growing queue rather than a stale snapshot
                    if ctx.t - self._last_update.get(corridor.id, 0) >= self.report_every:
                        self._refresh(active, ctx, n, median_speed, baseline,
                                      stopped_frac, occupancy, hist)
                        self._last_update[corridor.id] = ctx.t
                elif cus.s <= 0:
                    active.ended_t = ctx.t
                    self.active.pop(corridor.id, None)

        return raised

    # ------------------------------------------------------------------
    def _collect_baseline(self, corridor, members, speeds, occupancy) -> None:
        """Learn free-flow speed from genuinely free-flowing samples only."""
        if corridor.baseline_speed_px is not None:
            return
        if len(members) < 2 or occupancy > 0.35:
            return
        buf = self._baseline_samples.setdefault(corridor.id, [])
        buf.extend(s for s in speeds if s > 1.0)
        if len(buf) >= 60:
            self.scene.learn_baseline(corridor.id, buf)

    @staticmethod
    def _growth(hist) -> tuple[float, int, int]:
        """Queue growth rate in vehicles/minute, plus first and peak counts."""
        if len(hist) < 2:
            return 0.0, (hist[0][1] if hist else 0), (hist[0][1] if hist else 0)
        t0, n0 = hist[0]
        t1, n1 = hist[-1]
        peak = max(n for _, n in hist)
        dt = max(t1 - t0, 1e-6)
        return float((n1 - n0) / dt * 60.0), n0, peak

    def _triggers(self, ctx, n, median_speed, baseline, stopped_frac, occupancy, hist,
                  onset) -> dict:
        growth, first_n, peak = self._growth(hist)
        drop = 0.0
        if baseline and baseline > 1e-6:
            drop = float(np.clip((baseline - median_speed) / baseline, 0.0, 1.0)) * 100.0
        return {
            "vehicles": int(n),
            "vehicles_at_onset": int(first_n),
            "vehicles_peak": int(peak),
            "growth_veh_per_min": round(growth, 2),
            "median_speed_px": round(median_speed, 2),
            "baseline_speed_px": round(baseline, 2) if baseline else None,
            "speed_drop_pct": round(drop, 1),
            "stopped_fraction": round(stopped_frac, 3),
            "occupancy": round(occupancy, 4),
            "persistence_s": round(max(0.0, ctx.t - onset), 1),
            "metric_units_available": self.scene.has_metric_scale,
            "length_note": ("queue reported in vehicles; no homography, so no metres"
                            if not self.scene.has_metric_scale else "homography available"),
        }

    def _make_event(self, corridor, ctx, onset, n, median_speed, baseline,
                    stopped_frac, occupancy, cus, hist) -> Event:
        trig = self._triggers(ctx, n, median_speed, baseline, stopped_frac,
                              occupancy, hist, onset)
        trig["cusum_s"] = round(cus.s, 3)
        trig["cusum_h"] = cus.h
        conf = float(np.clip(0.45 + 0.35 * min(1.0, cus.s / max(2 * cus.h, 1e-6))
                             + 0.20 * min(1.0, n / (self.min_vehicles * 2.0)), 0, 0.97))
        return Event(
            type=QUEUE,
            camera_id=self.scene.camera_id,
            started_t=onset,
            detected_t=ctx.t,
            confidence=conf,
            corridor_id=corridor.id,
            track_ids=[tr.track_id for tr in ctx.tracks if tr.corridor_id == corridor.id][:20],
            triggers=trig,
        )

    def _refresh(self, ev, ctx, n, median_speed, baseline, stopped_frac,
                 occupancy, hist) -> None:
        ev.triggers.update(self._triggers(ctx, n, median_speed, baseline,
                                          stopped_frac, occupancy, hist, ev.started_t))
        ev.ended_t = None
