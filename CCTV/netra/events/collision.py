"""Suspected collision-related disruption.

Read the name of this module's event carefully, because the name is the point.
NETRA does **not** claim to detect accidents. It raises a *suspected
collision-related disruption* and routes it to a human.

That is not modesty, it is what the current evidence supports. On the 2026
ACCIDENT benchmark -- 2,027 real CCTV accident clips -- the best published
system scores 0.571 on the unified metric against a human inter-annotator
ceiling of roughly 0.96, and it needs about thirteen GPU-hours on an RTX PRO
6000 to do it. Frozen 7B vision-language models score 0.115 on collision type,
*below* the majority-class floor of 0.335. Anyone claiming reliable automatic
accident detection on a laptop is claiming something the field has not achieved.

What can be done reliably, cheaply, is to find the *moments worth looking at*.
Two independent detectors run in parallel, and the event fires when either is
convincing or both agree:

**A. Pairwise trajectory conflict** (Ghahremannezhad et al., IEEE IST 2022)

    conflict = proximity  AND  approach angle  AND  mutual deceleration

Not "the boxes overlap". Under perspective, vehicles in adjacent lanes overlap
constantly; what they do not do is converge at an angle and then both stop. All
three conditions are required, and the angle is computed between trajectory
slopes rather than between box positions.

**B. Global motion change-point** (the OF + OSD baselines from ACCIDENT)

Z-scored mean optical-flow magnitude and summed box area. This fires *without
needing both vehicles to be tracked through the impact*, which matters because
that is precisely when tracking breaks. It is the safety net under detector A.

Whichever fires, the onset is then walked backwards with sparse optical flow
(Chen et al.) because the vehicle comes to rest seconds after the impact and
the resting time is the wrong number to report.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..detect import MOTORISED_CLASSES, VULNERABLE_CLASSES
from ..footprint import Footprint, separation as fp_separation
from ..pathconflict import PathConflictDetector
from ..geometry import angle_between, box_ground_point, iou
from ..signals import Cusum
from ..attribution import Candidate, ParticipantSelector
from ..stationary import StationaryDetector
from .base import COLLISION, Event, EventEngine


class CollisionEngine(EventEngine):
    name = "collision"

    def __init__(self, scene, config: dict):
        super().__init__(scene, config)
        c = config.get("collision", {})
        self.proximity_px = float(c.get("proximity_px", 90.0))
        self.min_approach_angle = float(c.get("min_approach_angle_deg", 20.0))
        self.decel_threshold = float(c.get("decel_px_s2", -25.0))
        self.min_approach_speed = float(c.get("min_approach_speed_px", 8.0))
        self.post_stop_s = float(c.get("post_stop_s", 2.0))
        self.cusum_beta = float(c.get("cusum_beta", 0.25))
        self.cusum_h = float(c.get("cusum_h", 1.6))
        self.changepoint_weight = float(c.get("changepoint_weight", 0.5))
        self.cooldown_s = float(c.get("cooldown_s", 20.0))
        self.max_pairs = int(c.get("max_pairs", 400))
        # The implementation is regression-tested, but enabling this channel
        # changed the selected event on several clips and reduced temporal T
        # from 0.454 to 0.390. Keep it behind an explicit experiment flag until
        # it is validated on crash-free traffic, rather than shipping a logic
        # repair as an unmeasured model improvement.
        self.pairwise_enabled = bool(c.get("pairwise_enabled", False))
        # a peak fires on the impact frame; the disruption began just before it
        self.peak_onset_lead_s = float(c.get("peak_onset_lead_s", 0.6))
        self.stationary_gate = float(c.get("stationary_gate", 0.42))
        # deferred confirmation: a close approach is a candidate, not an event
        self.confirm_window_s = float(c.get("confirm_window_s", 6.0))
        self.stop_ratio = float(c.get("stop_ratio", 0.30))     # speed collapse
        self.deflect_deg = float(c.get("deflect_deg", 45.0))   # violent heading swing
        self.proximity_scale = float(c.get("proximity_scale", 1.25))  # x vehicle size
        # Kinematic plausibility gate. An ID switch makes a track appear to
        # teleport, which reads as a huge speed followed by a huge deceleration
        # -- indistinguishable from an impact unless you reject it outright.
        # Measured on clean traffic: a switch produced 1164 px/s and
        # -1452 px/s^2, and the aftermath test duly "confirmed" a collision.
        # Expressed as a fraction of frame diagonal per second so it is
        # resolution-independent.
        self.max_speed_frac = float(c.get("max_speed_frac", 0.45))
        self.min_confirm_delay_s = float(c.get("min_confirm_delay_s", 0.4))
        # How long the stop must HOLD before a conflict is believed.
        #
        # Measured side by side, a clean-traffic near-pass and a real collision
        # were indistinguishable at the moment of contact: separation 0.85 vs
        # 0.91 vehicle-lengths, conflict score 0.82 vs 0.795, both showing a
        # hard speed collapse. Trajectory features simply do not separate
        # "braked hard" from "was hit".
        #
        # What separates them is the next few seconds: a braking vehicle
        # resumes, a crashed one does not. So a momentary collapse only arms
        # the candidate; it must still be slow after this hold before firing.
        self.stop_hold_s = float(c.get("stop_hold_s", 2.0))
        # convergence: how much closer the pair must have got, and how far
        # apart they must have started, both in vehicle-lengths
        self.min_convergence_ratio = float(c.get("min_convergence_ratio", 3.0))
        self.min_start_separation = float(c.get("min_start_separation", 2.5))
        # Pair history has to begin *before* contact.  The old implementation
        # recorded max separation only after ``proximity_scale`` passed, so a
        # configured start separation of 2.5 could never coexist with a contact
        # gate of 1.0.  Keep a bounded history radius large enough to observe
        # convergence without forming every pair in the frame.
        self.pair_history_scale = float(c.get(
            "pair_history_scale", max(4.0, 1.25 * self.min_start_separation)))
        self.pair_history_scale = max(self.pair_history_scale,
                                      self.proximity_scale,
                                      self.min_start_separation)
        # These defaults SHADOW the selector's own, so they have to move with
        # it. The gap stayed at 1.6 vehicle-lengths after the geometry switched
        # to footprint units, and the solo bar stayed at 0.62 after it was
        # lowered -- in both cases the change landed in the dataclass and was
        # silently overwritten here.
        self.selector = ParticipantSelector(
            max_gap_lengths=float(c.get("participant_gap_lengths", 1.25)),
            coincidence_s=float(c.get("participant_coincidence_s", 1.6)),
            pair_min_score=float(c.get("participant_pair_min", 0.28)),
            single_min_score=float(c.get("participant_single_min", 0.40)),
        )

        # A physical bound rather than a fitted constant: both vehicles past
        # driver control authority, coincident, and cancelling in momentum.
        self.impulse_gate = float(c.get("impulse_gate", 0.55))
        # Measured anti-correlated in-domain; see _build_candidates.
        self.use_crash_classifier = bool(c.get("use_crash_classifier", False))
        # Continuous, calibrated cameras have dense junction traffic, queues
        # and long-lived tracks.  In the supplied crash-free Traffic set every
        # false collision was a single-channel path finding (16/16 clips),
        # whereas the short accident clips are intentionally uncalibrated and
        # need the higher-recall path.  Treat camera geometry as an operating
        # regime, not a folder-name shortcut: fixed-camera collision promotion
        # requires independent corroboration; uncalibrated clips are unchanged.
        self.fixed_camera_min_channels = int(
            c.get("fixed_camera_min_channels", 2))
        self.is_calibrated_camera = bool(getattr(scene, "corridors", None))

        # Paths that cross, at the same moment, at an angle, followed by at
        # least one vehicle being knocked off its heading. Unlike every
        # state-based cue in this file, a queue cannot satisfy it: vehicles
        # nose to tail travel along the SAME line, and collinear paths do not
        # intersect.
        self.path_conflict = PathConflictDetector(
            history_s=float(c.get("path_history_s", 2.0)),
            min_angle_deg=float(c.get("path_min_angle_deg", 25.0)),
            max_time_gap_s=float(c.get("path_max_time_gap_s", 0.60)),
            min_deviation_deg=float(c.get("path_min_deviation_deg", 12.0)),
            min_score=float(c.get("path_min_score", 0.55)),
            score_model=c.get("candidate_score_model"),
        )

        # Run the conflict test on EVERY analysed frame.
        #
        # It was throttled to 5 Hz as a performance measure, and that silently
        # destroyed it: on a held-out clip the same footage yields one confirmed
        # collision when the test runs every frame and zero when it runs every
        # 0.2 s. Detection is staged -- a converging course is registered, then
        # confirmed when the vehicles arrive -- and both stages need to see the
        # frames in between, because the track set changes across an impact and
        # the confirmation window is narrow. The throttle was also no longer
        # buying much: removing the travelled-path intersection in the rewrite
        # took the cost to about 30 ms per frame at 70 simultaneous tracks.
        self.path_check_period_s = float(c.get("path_check_period_s", 0.0))
        self._last_path_check = -1e9
        self._path_cache: list = []

        self._cusum = Cusum(beta=self.cusum_beta, h=self.cusum_h, decay=0.06)
        self._last_event_t = -1e9
        self._pair_state: dict[tuple[int, int], dict] = {}
        self._pending: dict[tuple[int, int], dict] = {}

    # ------------------------------------------------------------------
    def update(self, ctx) -> list[Event]:
        raised: list[Event] = []

        # register close approaches as pending, then act only on confirmed ones
        if self.pairwise_enabled:
            self._pairwise_conflict(ctx)
            conflict, detail = self._confirm_pending(ctx)
        else:
            conflict, detail = 0.0, {}
        cp_z = ctx.changepoint.last_score if ctx.changepoint is not None else 0.0
        cp_evidence = 0.0
        if ctx.changepoint is not None and ctx.changepoint.z_threshold > 0:
            cp_evidence = float(np.clip(cp_z / (2.0 * ctx.changepoint.z_threshold), 0, 1))

        # Two detectors, two temporal characters, so two decision rules.
        #
        # A trajectory conflict *develops*: vehicles converge, decelerate, stop.
        # Evidence accumulates over ~a second, which is exactly what CUSUM is
        # for, and accumulating suppresses the jitter of dense traffic.
        #
        # An impact is *instantaneous*. Its optical-flow signature is one or two
        # frames of violent motion and then it is gone. Feeding that into an
        # accumulator is a category error -- the spike decays before the sum
        # crosses threshold, which is why the first run missed most collisions.
        # The ACCIDENT benchmark's own baselines use peak detection here, so
        # this channel gets a peak trigger.
        # A *confirmed* conflict already carries its own temporal evidence: it
        # required a close approach, then a speed collapse or hard deflection
        # within the confirmation window. Running it through a CUSUM as well
        # would double-count the wait and, worse, would arrive ~2.5 s after the
        # change-point has decayed -- so the two could never agree and nothing
        # would ever fire. Confirmation is the accumulator here.
        conflict_fired = conflict > 0.0
        self._cusum.update(conflict, ctx.t)
        peak_fired = (ctx.changepoint is not None and cp_z >= ctx.changepoint.z_threshold)

        # Channel C -- the one every AI City winner relies on, and the one whose
        # absence made our first attempt box the wrong car. A crashed vehicle
        # stops, so it settles into the background image, where it is crisp and
        # unoccluded and trivially detectable. Corroborate with the post-crash
        # signatures a camera can genuinely see: occupants getting out, a second
        # vehicle at rest alongside, debris on the road.
        static_fired, static_obj, static_detail = self._stationary_evidence(ctx)

        # The only channel that measures the impact itself. Two vehicles
        # past driver control authority, at the same instant, with their
        # momentum changes cancelling -- which braking in a queue cannot
        # produce, because braking together adds rather than cancels.
        impulse_fired, impulse_pair, impulse_detail = self._impulse_evidence(ctx)

        # Did one vehicle cut across another's path and disturb it?
        #
        # Evaluated a few times a second rather than every frame. The
        # test looks back over a two-second window, so a crossing stays
        # visible to it for two seconds; running it at frame rate repeats
        # the same answer thirty times and costs as much as the detector.
        if ctx.t - self._last_path_check >= self.path_check_period_s:
            self._last_path_check = ctx.t
            mon = getattr(ctx, "predictor", None)
            fc = (mon.forecasts(ctx.tracks)
                  if mon is not None and getattr(mon, "available", False)
                  else None)
            self._path_cache = self.path_conflict.find(
                ctx.tracks, ctx.t, forecasts=fc,
                frame_shape=getattr(ctx, "frame_shape", None))
        conflicts = self._path_cache
        path_fired = bool(conflicts)
        path_hit = conflicts[0] if conflicts else None
        path_detail = {"path_conflict": "none"} if not path_fired else {
            "path_conflict": "paths crossed and a vehicle was deflected",
            "track_ids": list(path_hit.track_ids),
            "crossing_angle_deg": round(path_hit.angle_deg, 1),
            "time_gap_s": round(path_hit.time_gap_s, 2),
            "heading_change_deg": [round(v, 1) for v in path_hit.deviation_deg],
            "speed_drop": [round(v, 2) for v in path_hit.speed_drop],
            "onset_t": round(path_hit.t_cross, 2),
            "score": round(path_hit.score, 3),
            "gates": path_hit.gates,
            "why": ("two paths intersected within "
                    f"{path_hit.time_gap_s:.2f}s of each other at "
                    f"{path_hit.angle_deg:.0f} degrees, and a vehicle left its "
                    "heading afterwards; vehicles queueing travel along the same "
                    "line and never intersect"),
        }

        # A global motion change-point on its own is NOT sufficient evidence,
        # and letting it fire alone was measurably wrong: on ordinary traffic
        # footage it flagged 13 of 16 clips, while catching only 8 of 15 real
        # collisions. It fires on camera pans, on a bus passing close to the
        # lens, on a drone repositioning. The ACCIDENT benchmark says the same
        # thing quantitatively -- their optical-flow baseline scores 0.287.
        #
        # So the peak is demoted to a *corroborator*. An event needs either a
        # specific object-level finding (a vehicle at rest with crash cues), or
        # two independent channels agreeing.
        # ONE channel may fire on its own, and it is not the obvious one.
        #
        # Trajectory conflict was tested against ground truth and failed to
        # separate the classes. Measured on a clean clip versus a real crash:
        # separation 0.85 vs 0.91 vehicle-lengths, conflict score 0.82 vs 0.795,
        # both with a hard speed collapse that held for 2 s. Requiring
        # persistence did not help either -- in ordinary traffic vehicles stop
        # close to one another and stay stopped, because that is what a junction
        # or a queue looks like. Trajectory geometry alone cannot tell "gave way"
        # from "was hit".
        #
        # This is exactly why every winning AI City entry is built around the
        # background image rather than around trajectories, and our own data
        # reproduces their reasoning independently.
        #
        # So the specific channel is: a vehicle at rest ON the carriageway with
        # post-crash corroboration -- occupants out, a companion vehicle stopped
        # at the same moment, debris. Conflict and change-point remain as
        # confidence corroborators; neither may raise an incident alone.
        # Momentum exchange stands alone for the same reason background-
        # stationary detection does, and on stronger grounds: a queue can
        # imitate every state-based cue in this file, but it cannot imitate
        # Newton's third law.
        # Background-stationary no longer raises an incident on its own.
        #
        # It was the channel every AI City winner built on, and on their
        # highway data it is excellent. On Indian urban footage it measured
        # *anti-correlated*: trigger score median 0.529 on crash-free clips
        # against 0.488 on collision clips, maximum 0.992 against 0.548. There
        # is no threshold that helps -- at 0.55 it loses all fifteen collision
        # clips and still keeps ten false alarms -- and it produced every one
        # of the 155.8 false collision alarms per crash-free hour.
        #
        # Worse, firing first, it consumed the incident and the cooldown then
        # blocked the path-crossing channel from reporting the real collision
        # seconds later. A cue that cannot discriminate should not be able to
        # pre-empt one that can.
        #
        # It remains as corroboration, and as the mechanism that finds stopped
        # vehicles for the blockage engine, where "stopped" is the finding
        # rather than a proxy for something else.
        specific = impulse_fired or path_fired or (
            static_fired and (conflict_fired or peak_fired))
        channels_agreeing = (int(conflict_fired) + int(peak_fired)
                             + int(static_fired) + int(impulse_fired)
                             + int(path_fired))
        if not self._promotion_allowed(specific, channels_agreeing,
                                       impulse_confirmed=impulse_fired):
            # Keep collecting raw candidates for audit/training, but do not
            # promote one noisy measurement to an operator-visible collision.
            specific = False
        if not specific:
            return raised
        if ctx.t - self._last_event_t < self.cooldown_s:
            return raised

        # a change-point with no moving traffic is a camera artefact, not a crash
        movers = [tr for tr in ctx.tracks
                  if tr.cls in MOTORISED_CLASSES | VULNERABLE_CLASSES]
        if len(movers) < 2 and not static_fired:
            return raised

        self._last_event_t = ctx.t
        path_tracks = (list(dict.fromkeys(path_hit.track_ids))
                       if path_fired and path_hit is not None else [])
        # A path can intersect its own earlier segment during a turn, spin or
        # tracker loop. That is not a two-vehicle attribution. Also, in the
        # first 1.5 s there is too little trajectory history to accuse a pair;
        # retain the event for review but withhold boxes.
        path_attributable = self._path_attributable(path_tracks, ctx.t)
        if path_fired and path_hit is not None and path_attributable:
            # The instant the paths met, which is the collision.
            onset = float(path_hit.t_cross)
        elif impulse_fired and impulse_pair is not None:
            # Measured, not inferred: the residual spike is the moment the
            # velocities changed, which is the collision.
            onset = float(impulse_pair["onset_t"])
        elif static_fired and static_obj is not None:
            # the vehicle came to rest here; onset recovery walks it back further
            onset = static_obj.first_seen_t
        elif conflict_fired and self._cusum.first_evidence_t is not None:
            onset = self._cusum.first_evidence_t
        else:
            onset = max(0.0, ctx.t - self.peak_onset_lead_s)

        # -- WHERE did it happen, and WHO was involved --------------------
        # Impact-point estimation removed. The flow-weighted centroid was
        # consistently wrong in review -- it follows the largest or nearest
        # moving object rather than the collision -- and a confidently drawn
        # wrong marker is worse than no marker. Vehicle identity, which the
        # classifier and footprint geometry supply, is the reliable answer to
        # "where".
        impact = None

        track_ids = list(detail.get("track_ids", []))
        involved_detail: dict = {}
        if static_detail:
            involved_detail.update(static_detail)
            for tid in static_detail.get("stationary_track_ids", []):
                if tid not in track_ids:
                    track_ids.insert(0, tid)
        if impact is not None:
            ix, iy, iconf = impact
            involved_detail["impact_point"] = [round(ix, 1), round(iy, 1)]
            involved_detail["impact_confidence"] = round(iconf, 3)

        # Pair-centric attribution. A ranked list of "suspicious vehicles" was
        # what produced the over-tagging -- the right pair plus two bystanders,
        # and in the worst case the two cars that had queued behind the crash.
        # Collisions are pairwise, participants touch, and participants stop
        # together while bystanders stop afterwards; the selector encodes all
        # three and returns nothing rather than guess.
        cands = self._build_candidates(ctx, static_obj, detail)
        involved_detail["candidates"] = [
            {"id": c.track_id, "score": round(c.score, 3),
             "p_crashed": c.detail.get("p_crashed"),
             "stop_t": None if c.stop_t is None else round(c.stop_t, 2),
             "rollover": round(c.rollover, 3), "off_road": c.off_road}
            for c in cands[:8]
        ]
        if path_fired and path_hit is not None:
            # The two vehicles whose paths crossed ARE the finding. No ranking
            # step can confuse a participant with a bystander here, because a
            # bystander's path did not intersect anything.
            chosen = []
            why = {
                "mode": "path-crossing pair",
                "reason": (f"paths intersected at {path_hit.angle_deg:.0f} degrees "
                           f"within {path_hit.time_gap_s:.2f}s of each other, and a "
                           f"vehicle's heading changed by "
                           f"{max(path_hit.deviation_deg):.0f} degrees afterwards"),
                "crossing_angle_deg": round(path_hit.angle_deg, 1),
                "heading_change_deg": [round(v, 1) for v in path_hit.deviation_deg],
            }
            involved_detail["participant_selection"] = why
            involved_detail["participant_boxes"] = [
                [round(float(v), 1) for v in b] for b in path_hit.boxes]
            involved_detail["participant_p_crashed"] = [None, None]
            track_ids = list(path_hit.track_ids)
        elif path_fired and path_hit is not None:
            chosen = []
            why = {
                "mode": "unattributed path conflict",
                "reason": ("path evidence did not identify two independently "
                           "observed vehicles; event retained but participant "
                           "boxes withheld"),
                "distinct_track_ids": path_tracks,
                "observation_time_s": round(ctx.t, 2),
            }
            involved_detail["participant_selection"] = why
            involved_detail["participant_boxes"] = []
            involved_detail["participant_p_crashed"] = []
            track_ids = []
        elif impulse_fired and impulse_pair is not None:
            # No selection step, because there is nothing to select between:
            # the measurement names both vehicles. This is the one path where
            # attribution is as strong as detection.
            chosen = []
            why = {
                "mode": "momentum-exchange pair",
                "reason": ("both vehicles changed velocity beyond driver control "
                           "authority, within "
                           f"{impulse_pair['coincidence_s']}s of each other, and "
                           "their momentum changes cancelled "
                           f"({impulse_pair['momentum_exchange']:.2f} of the total); "
                           "vehicles braking together produce changes that add"),
                "momentum_exchange": impulse_pair["momentum_exchange"],
                "impulse": impulse_pair["impulse"],
            }
            involved_detail["participant_selection"] = why
            involved_detail["participant_boxes"] = [
                [round(float(v), 1) for v in b] for b in impulse_pair["boxes"]]
            involved_detail["participant_p_crashed"] = [None, None]
            track_ids = list(impulse_pair["track_ids"])
        else:
            chosen, why = self.selector.select(cands)
            involved_detail["participant_selection"] = why
            involved_detail["participant_boxes"] = [
                [round(float(v), 1) for v in c.box] for c in chosen]
            involved_detail["participant_p_crashed"] = [
                c.detail.get("p_crashed") for c in chosen]
            track_ids = [c.track_id for c in chosen if c.track_id is not None]

        triggers = {
            "conflict_score": round(conflict, 3),
            "changepoint_z": round(cp_z, 3),
            "changepoint_threshold": (ctx.changepoint.z_threshold
                                      if ctx.changepoint is not None else None),
            "changepoint_evidence": round(cp_evidence, 3),
            "cusum_s": round(self._cusum.s, 3),
            "cusum_h": self._cusum.h,
            "detector": ("path-crossing" if path_fired
                         else "momentum-exchange" if impulse_fired
                         else "background-stationary" if static_fired
                         else "pairwise-conflict" if conflict_fired
                         else "motion-changepoint"),
            "trigger_mode": ("path-crossing" if path_fired
                             else "impulse" if impulse_fired
                             else "stationary" if static_fired
                             else "confirmed-conflict" if conflict_fired else "peak"),
            "channels_agreeing": channels_agreeing,
            "impulse_channel": impulse_detail,
            "path_conflict_channel": path_detail,
        }
        triggers.update(detail)
        triggers.update(involved_detail)
        if ctx.changepoint is not None:
            triggers["changepoint_detail"] = ctx.changepoint.snapshot()

        # Attribution quality is tracked separately from detection confidence.
        # Boxing the wrong vehicle is worse than boxing none: it actively
        # misleads an operator and destroys trust in every other finding. So a
        # vehicle is only named when the evidence actually points at it -- a
        # stationary object matched to a live track, or a pairwise conflict that
        # identified both parties. Kinematic ranking alone is a weak hint and is
        # recorded in the triggers but never promoted to a red box.
        if path_fired and path_attributable and track_ids:
            attribution = "path-crossing"
        elif impulse_fired and track_ids:
            attribution = "momentum-exchange"
        elif static_detail.get("stationary_track_ids"):
            attribution = "stationary-object-track"
        elif detail.get("track_ids"):
            attribution = "pairwise-conflict"
        else:
            attribution = "unattributed"
            track_ids = []

        triggers["attribution"] = attribution
        triggers["attribution_note"] = {
            "path-crossing": ("both vehicles identified by the crossing of their own "
                              "paths; a bystander's path did not intersect anything"),
            "momentum-exchange": ("both vehicles identified by the momentum they exchanged; "
                                  "a bystander cannot take part in an exchange"),
            "stationary-object-track": "vehicle located on the background image and matched to a live track",
            "pairwise-conflict": "both parties identified by converging trajectories",
            "unattributed": ("incident detected but no vehicle could be identified with "
                             "confidence; no box drawn rather than risk marking an "
                             "uninvolved vehicle"),
        }[attribution]

        ev = Event(
            type=COLLISION,
            camera_id=self.scene.camera_id,
            started_t=onset,
            detected_t=ctx.t,
            confidence=self._confidence(
                conflict, cp_evidence, detail,
                n_channels=int(conflict_fired) + int(peak_fired) + int(static_fired),
                localised=impact is not None or bool(static_detail),
                static_score=static_detail.get("crash_score", 0.0) if static_detail else 0.0),
            corridor_id=detail.get("corridor_id"),
            track_ids=track_ids[:6],
            triggers=triggers,
        )
        raised.append(ev)
        self._cusum.reset()
        return raised

    def _promotion_allowed(self, specific: bool, channels_agreeing: int,
                           impulse_confirmed: bool = False) -> bool:
        """Whether collision evidence may become an operator-visible event."""
        if not specific:
            return False
        if not self.is_calibrated_camera:
            return True
        # Continuous fixed-camera traffic produces many path crossings, hard
        # stops and background changes. None proves impact. Require the one
        # channel that measures a coincident, cancelling two-body momentum
        # change, plus independent corroboration. Other anomaly heads continue
        # to report queues, wrong-way motion, abnormal stops and blockages.
        return (impulse_confirmed and
                channels_agreeing >= self.fixed_camera_min_channels)

    @staticmethod
    def _path_attributable(track_ids, observation_t: float) -> bool:
        """Require two identities and enough history before drawing boxes."""
        return len(set(track_ids)) >= 2 and observation_t >= 1.5

    def _build_candidates(self, ctx, static_obj, conflict_detail) -> list[Candidate]:
        """Assemble every vehicle that might have been in the collision.

        Evidence is pooled from three places: the stationary object that
        triggered the event, the pair identified by trajectory conflict, and any
        track whose own kinematics broke. Each contributes a score; the selector
        then decides which of them actually go in the box.
        """
        by_id: dict[int, Candidate] = {}

        def add(box, score, track_id=None, stop_t=None, **kw):
            if box is None:
                return
            box = np.asarray(box, dtype=float)
            key = track_id if track_id is not None else f"b{len(by_id)}"
            for k, ex in by_id.items():          # merge duplicate evidence
                if iou(ex.box, box) >= 0.5:
                    key = k
                    break
            c = by_id.get(key)
            if c is None:
                c = Candidate(track_id=track_id, box=box, score=0.0, stop_t=stop_t)
                by_id[key] = c
            if c.track_id is None and track_id is not None:
                c.track_id = track_id
            c.score = max(c.score, float(score))
            if stop_t is not None and (c.stop_t is None or stop_t < c.stop_t):
                c.stop_t = stop_t
            c.rollover = max(c.rollover, float(kw.get("rollover", 0.0)))
            c.off_road = c.off_road or bool(kw.get("off_road", False))
            # Sticky: if any source says this vehicle never moved, it never
            # moved. Merged evidence must not launder a parked car.
            c.parked = c.parked or bool(kw.get("parked", False))
            if "arrived_moving" in kw:
                c.arrived_moving = c.arrived_moving and bool(kw["arrived_moving"])
            c.queue_member = c.queue_member or bool(kw.get("queue", False))
            c.stop_decel = max(c.stop_decel, float(kw.get("stop_decel", 0.0)))

        # 1. the stationary object that triggered this event
        #
        # Scored by the same rules as everything else rather than handed a
        # flat 0.9. Being the object that opened the event is not evidence
        # about the object; it is evidence about the trigger.
        if static_obj is not None:
            sc0, _ = StationaryDetector.crash_score(static_obj)
            add(static_obj.box, max(0.55, sc0), static_obj.track_id,
                static_obj.first_seen_t,
                rollover=getattr(static_obj, "aspect_shift", 0.0),
                off_road=(getattr(static_obj, "road_coverage", 1.0) < 0.3),
                arrived_moving=getattr(static_obj, "arrived_moving", True),
                parked=getattr(static_obj, "is_parked", False),
                queue=getattr(static_obj, "queue_member", False),
                stop_decel=getattr(static_obj, "stop_decel", 0.0))

        # 2. every stationary object with its own crash evidence
        for o in (ctx.stationary or []):
            sc, _ = StationaryDetector.crash_score(o)
            add(o.box, sc, o.track_id, o.first_seen_t,
                rollover=getattr(o, "aspect_shift", 0.0),
                off_road=(getattr(o, "road_coverage", 1.0) < 0.3),
                arrived_moving=getattr(o, "arrived_moving", True),
                parked=getattr(o, "is_parked", False),
                queue=getattr(o, "queue_member", False),
                stop_decel=getattr(o, "stop_decel", 0.0))

        # 3. the pair the conflict test named
        for tid in (conflict_detail.get("track_ids") or []):
            tr = self._track_by_id(ctx, tid)
            if tr is not None:
                add(tr.box, 0.7, tid, getattr(tr, "stationary_since", None))

        # 4. tracks whose own kinematics broke
        for h in self._tracks_by_flow(ctx, max_n=5):
            tr = self._track_by_id(ctx, h["id"])
            if tr is not None:
                add(tr.box, min(0.85, h["score"]), h["id"],
                    getattr(tr, "stationary_since", None),
                    rollover=tr.aspect_shift(2.5))

        out = list(by_id.values())

        # Ask the classifier what each candidate actually LOOKS like. This is
        # the evidence no geometric cue could supply: whether the vehicle is
        # crumpled and skewed, or merely stopped. It is the term that separates
        # a participant from a bystander who happened to stop nearby, which was
        # the single largest source of over-tagging in review.
        # ---- appearance classifier: OFF by default, and here is why -----
        #
        # A fine-tuned yolo11n-cls scores 0.993 AUC on held-out crops and is
        # still not used, because both of its high scores were measured against
        # negatives that shared no camera with the positives. Inside the
        # accident clips, where a parked car and a wreck appear in the same
        # frame, it inverts: on clip 1 it rated six of eight vehicles above 0.8,
        # gave an untracked bystander 0.995, and gave the vehicle that had
        # visibly rolled -- 265 px/s, a 41.7 px/s^2 stop, aspect swing 0.74 --
        # just 0.695. It had learned the scene, not the damage.
        #
        # It also suppressed pair attribution, because a partner scoring below
        # min_p_crashed is refused, and it refused the real ones.
        #
        # The code and the training scripts stay so the result is reproducible
        # and the negative finding is auditable. Enable with
        # events.collision.use_crash_classifier if a model trained on
        # same-camera negatives ever becomes available.
        cc = getattr(ctx, "crash_cls", None)
        if (self.use_crash_classifier and cc is not None
                and getattr(cc, "available", False) and out):
            probs = cc.score_boxes(ctx.frame, [c.box for c in out])
            for c, p_crash in zip(out, probs):
                p_crash = float(p_crash)
                c.detail["p_crashed"] = round(p_crash, 3)
                prior = c.score
                c.score = float(min(1.0, 0.30 * p_crash + 0.70 * prior))
                c.detail["cue_prior"] = round(prior, 3)

        out = self._apply_vetoes(out)
        out.sort(key=lambda c: -c.score)
        return out

    @staticmethod
    def _apply_vetoes(cands):
        """Enforce the hard rules *after* the classifier has spoken.

        Applied here rather than inside the cue prior because the prior is
        only a quarter of the fused score: a gate expressed as a zero in a
        weighted sum still leaves 0.75 * p_crashed standing, which is how a
        parked truck survived at 0.995. A veto has to be a veto.
        """
        kept = []
        for c in cands:
            if c.parked:
                # Never once seen moving. Whatever it looks like, we did not
                # witness it crash, and stationary vehicles beside roads are
                # the most common object in traffic footage.
                c.detail["veto"] = "parked: present from first frame, never moved"
                continue
            if c.queue_member:
                # Demoted, not dropped: the back of a queue is precisely
                # where rear-end collisions happen.
                c.score *= 0.35
                c.detail["queue_member"] = True
            if 0.0 < c.stop_decel < 25.0:
                # Braked smoothly. Queues do; impacts do not.
                c.score *= 0.55
                c.detail["gentle_stop"] = True
            if not c.has_motion_history:
                # Never watched moving, stopping or deforming. It may still
                # join a pair with a vehicle we did watch, but it cannot be
                # the one we accuse on its own.
                c.score = min(c.score, 0.34)
                c.detail["no_motion_history"] = True
            kept.append(c)
        return kept

    def _impulse_evidence(self, ctx):
        """Two vehicles that exchanged momentum: ``(fired, pair, detail)``.

        The strongest single piece of evidence this system can gather, because
        it is the only one that measures the collision rather than its
        aftermath, and because momentum conservation has no benign analogue in
        traffic. Reported unfired but described whenever the monitor is
        unavailable, so a low frame-rate clip is visibly a limitation of the
        footage rather than a silent absence of evidence.
        """
        mon = getattr(ctx, "predictor", None)
        if mon is None:
            return False, None, {"impulse": "monitor absent"}
        if not getattr(mon, "available", False):
            return False, None, {
                "impulse": "unavailable",
                "reason": (f"analysis rate {getattr(mon, 'fps', 0):.1f} fps is below "
                           f"{getattr(mon, 'min_fps', 15.0):.0f}; an impact would occupy "
                           "a single frame and be indistinguishable from a dropped "
                           "detection"),
            }
        pairs = mon.impulse_pairs(ctx.tracks, ctx.t)
        if not pairs:
            return False, None, {"impulse": "no coincident momentum exchange"}
        best = pairs[0]
        detail = {
            "impulse": "momentum exchange",
            "track_ids": best["track_ids"],
            "impulse_scores": best["impulse"],
            "momentum_exchange": best["momentum_exchange"],
            "coincidence_s": best["coincidence_s"],
            "footprint_separation": best["footprint_separation"],
            "onset_t": best["onset_t"],
            "score": round(best["score"], 3),
            "why": ("velocity changes exceeded driver control authority, coincided "
                    "in time, and cancelled in momentum -- vehicles braking "
                    "together produce changes that add, not cancel"),
        }
        return best["score"] >= self.impulse_gate, best, detail

    def _stationary_evidence(self, ctx):
        """Has a vehicle come to rest in a way that looks like a crash?

        Returns ``(fired, object, detail)``. The gate is on the corroboration
        score, not on stopping alone -- a parked car stops too. What separates a
        crash is people getting out, another vehicle at rest beside it, and
        debris; all three are counted by :class:`StationaryDetector`.
        """
        best, best_score, best_detail = None, 0.0, None
        for o in ctx.stationary_new or []:
            score, detail = StationaryDetector.crash_score(o)
            if score > best_score:
                best, best_score, best_detail = o, score, detail

        if best is None or best_score < self.stationary_gate:
            return False, None, {}

        best.reported = True
        ids = [best.track_id] if best.track_id is not None else []
        detail = {
            "crash_score": round(best_score, 3),
            "crash_score_gate": self.stationary_gate,
            "stationary_object": best.to_dict(),
            "stationary_track_ids": ids,
            "post_crash_cues": best_detail,
        }
        return True, best, detail

    def _tracks_by_flow(self, ctx, max_n: int = 3, gate: float = 0.28) -> list[dict]:
        """Rank road users by how anomalously *they themselves* moved.

        The obvious approach -- rank by optical-flow energy inside each box --
        was tried and is wrong, measurably so. Flow magnitude scales with an
        object's size, speed and nearness to the camera, so a lorry driving
        normally across the foreground outranks a motorcycle actually crashing
        in the distance. On the first night-time clip it locked onto a passing
        car and ignored the collision.

        What distinguishes a crashed vehicle is not that it moved a lot, but
        that *its own kinematics broke*: it decelerated hard, swung heading, or
        stopped dead after travelling. Those are Ghahremannezhad's conflict
        features applied per-track instead of per-pair, which means they work
        even when only one participant was ever tracked.

        Flow energy is kept, but demoted to a tie-breaker.
        """
        cand = [tr for tr in ctx.tracks
                if tr.cls in MOTORISED_CLASSES | VULNERABLE_CLASSES and tr.hits >= 3]
        if not cand:
            return []

        energies = [0.0] * len(cand)
        if ctx.changepoint is not None:
            energies = ctx.changepoint.flow_energy_in_boxes(
                [tr.box for tr in cand], ctx.frame_shape)
        emax = max(energies) if energies else 0.0

        ranked = []
        for tr, e in zip(cand, energies):
            decel = -min(0.0, tr.acceleration_px(1.5))
            head = tr.heading_change(1.5)
            was_moving = tr.speed_px(3.0) > self.min_approach_speed
            now_slow = tr.speed_px(0.8) < 3.0

            s_dec = min(1.0, decel / max(abs(self.decel_threshold) * 2.0, 1e-6))
            s_head = min(1.0, head / 70.0)
            s_stop = 1.0 if (was_moving and now_slow) else 0.0
            s_flow = (e / emax) if emax > 1e-9 else 0.0

            score = 0.40 * s_dec + 0.28 * s_head + 0.22 * s_stop + 0.10 * s_flow
            if score >= gate:
                ranked.append({
                    "id": tr.track_id, "score": round(score, 3),
                    "decel_px_s2": round(-decel, 1),
                    "heading_change_deg": round(head, 1),
                    "stopped_after_moving": bool(s_stop),
                    "flow_share": round(s_flow, 3),
                })
        ranked.sort(key=lambda r: -r["score"])
        return ranked[:max_n]

    def _tracks_near(self, ctx, x: float, y: float, max_n: int = 3) -> list[dict]:
        """Which tracked vehicles sit at the impact point.

        A raw coordinate is a weak output -- it drifts, and a viewer cannot tell
        whether it means anything. A *track identity* is strong: it survives the
        vehicle moving, it lets the annotated video draw a box that follows the
        right car, and it makes the finding checkable frame by frame.

        A box containing the point wins outright; otherwise the nearest boxes
        within a radius are taken, scaled to the frame so it works on any
        resolution.
        """
        h, w = ctx.frame_shape[:2]
        radius = 0.12 * float(np.hypot(w, h))
        inside, near = [], []
        for tr in ctx.tracks:
            if tr.cls not in MOTORISED_CLASSES | VULNERABLE_CLASSES:
                continue
            x1, y1, x2, y2 = tr.box
            if x1 <= x <= x2 and y1 <= y <= y2:
                inside.append({"id": tr.track_id, "dist": 0.0})
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = float(np.hypot(cx - x, cy - y))
            if d <= radius:
                near.append({"id": tr.track_id, "dist": round(d, 1)})
        if inside:
            return inside[:max_n]
        near.sort(key=lambda t: t["dist"])
        return near[:max_n]

    # ------------------------------------------------------------------
    def _plausible(self, ctx, tr) -> bool:
        """Reject tracks whose kinematics are physically impossible.

        A road user cannot cross a large fraction of the frame in one second.
        When one appears to, the identity has been reassigned, and every
        derived quantity -- speed, acceleration, heading -- is meaningless.
        Using them anyway is how a tracker bug becomes a reported collision.
        """
        h, w = ctx.frame_shape[:2]
        ceiling = self.max_speed_frac * float(np.hypot(w, h))
        return tr.speed_px(1.0) <= ceiling

    def _confirm_pending(self, ctx) -> tuple[float, dict]:
        """Deferred confirmation: a collision has an aftermath, a near pass does not.

        This exists because of a measured failure. The three-condition test
        (proximity, approach angle, mutual deceleration) fired on clean traffic
        with an approach angle of 176 degrees -- which is not a crash, it is two
        vehicles passing in opposite lanes. Oncoming traffic has head-on geometry
        by definition and passes within a vehicle-width every few seconds.

        Angle and proximity cannot separate those cases, because a near pass and
        a collision look identical *up to the moment of contact*. What separates
        them is what happens next: vehicles that collide stop or are violently
        deflected; vehicles that pass carry on.

        So a close approach only opens a **pending** conflict. It is confirmed
        only if, within ``confirm_window_s``, a participant's speed collapses
        relative to its own approach speed, or its heading swings hard. If the
        window expires with both vehicles still travelling, the candidate is
        discarded silently.

        This costs up to ``confirm_window_s`` of latency in *detection*, but not
        in *reporting*: the onset recorded is the moment of closest approach,
        which is already in the pending record.
        """
        best, best_detail = 0.0, {}
        expired = []

        for key, st in self._pending.items():
            age = ctx.t - st["t_closest"]
            if age > self.confirm_window_s:
                expired.append(key)
                continue
            # an "aftermath" on the same frame as the approach is not an
            # aftermath, it is noise
            if age < self.min_confirm_delay_s:
                continue

            a = self._track_by_id(ctx, key[0])
            b = self._track_by_id(ctx, key[1])
            if a is None and b is None:
                # Both tracks vanished after a close approach. Tempting to treat
                # as evidence -- tracking does break at impact -- but in busy
                # traffic tracks vanish constantly through occlusion and frame
                # exit, and this branch was measurably a false-positive source.
                #
                # It is also redundant. If tracking broke because of a real
                # impact, the vehicles came to rest, and a vehicle at rest is
                # exactly what the background-stationary channel is built to
                # find. Drop the candidate and let that channel speak.
                expired.append(key)
                continue

            stopped, detail = False, {}
            for tr, pre in ((a, st["speed_a"]), (b, st["speed_b"])):
                if tr is None or pre < 1e-6:
                    continue
                if not self._plausible(ctx, tr):
                    continue
                now = tr.speed_px(0.8)
                ratio = now / max(pre, 1e-6)
                swing = tr.heading_change(1.5)
                if ratio <= self.stop_ratio or swing >= self.deflect_deg:
                    stopped = True
                    detail = {
                        "aftermath_track": tr.track_id,
                        "speed_before_px_s": round(pre, 1),
                        "speed_after_px_s": round(now, 1),
                        "speed_ratio": round(ratio, 3),
                        "heading_swing_deg": round(swing, 1),
                    }
                    break

            if stopped and st.get("stopped_at") is None:
                # arm, do not fire: wait to see whether the stop holds
                st["stopped_at"] = ctx.t
                st["stop_detail"] = detail
                continue

            armed_at = st.get("stopped_at")
            if armed_at is not None:
                held = ctx.t - armed_at
                if not stopped:
                    # it resumed -- this was braking, not a collision
                    expired.append(key)
                    continue
                if held >= self.stop_hold_s:
                    score = min(1.0, 0.55 + 0.45 * st["approach"])
                    if score > best:
                        best, best_detail = score, {
                            **st["detail"], **st.get("stop_detail", {}), **detail,
                            "aftermath": "participant stopped and stayed stopped",
                            "stop_held_s": round(held, 2),
                            "confirm_delay_s": round(age, 2),
                        }
                    expired.append(key)

        for k in expired:
            self._pending.pop(k, None)
        return best, best_detail

    @staticmethod
    def _track_by_id(ctx, tid: int):
        for tr in ctx.tracks:
            if tr.track_id == tid:
                return tr
        return None

    def _pairwise_conflict(self, ctx) -> tuple[float, dict]:
        """Ghahremannezhad's three-condition test over every plausible pair."""
        movers = [tr for tr in ctx.tracks
                  if tr.cls in MOTORISED_CLASSES | VULNERABLE_CLASSES and tr.hits >= 4]
        if len(movers) < 2:
            return 0.0, {}

        # Plausibility is per-vehicle, so evaluating it inside the pair loop
        # recomputed it once per partner: 104,800 calls for 131 frames in a
        # profile. Compute it once per track, then form pairs only from the
        # survivors.
        plausible = [tr for tr in movers if self._plausible(ctx, tr)]
        if len(plausible) < 2:
            return 0.0, {}

        # Cheap vectorised distance prefilter. Vehicles at opposite ends of the
        # frame cannot be in contact, and rejecting them by array arithmetic is
        # far cheaper than the full three-condition test.
        gp = np.array([box_ground_point(tr.box) for tr in plausible], dtype=float)
        diag = np.array([float(np.hypot(tr.box[2] - tr.box[0],
                                        tr.box[3] - tr.box[1]))
                         for tr in plausible], dtype=float)

        best_score, best_detail = 0.0, {}
        pairs = 0
        for i, j in itertools.combinations(range(len(plausible)), 2):
            reach = self.pair_history_scale * max(
                0.5 * (diag[i] + diag[j]), 1e-6) * 2.0
            if abs(gp[i, 0] - gp[j, 0]) > reach or abs(gp[i, 1] - gp[j, 1]) > reach:
                continue
            pairs += 1
            if pairs > self.max_pairs:
                break
            a, b = plausible[i], plausible[j]

            pa, pb = box_ground_point(a.box), box_ground_point(b.box)
            sep = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))

            # Scale-relative, not an absolute pixel count. A vehicle's apparent
            # size encodes its depth, so "within one vehicle-length" means the
            # same thing near the camera and at the vanishing point -- and the
            # same thing at 720p and 4K. A fixed 90 px gate does not.
            # Footprint separation: are these two vehicles sharing road
            # surface? Box distance conflates height with ground position
            # and made background/foreground pairs look adjacent.
            sep_norm = fp_separation(Footprint.from_box(a.box),
                                     Footprint.from_box(b.box))
            if sep_norm > self.pair_history_scale:
                continue

            key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
            st = self._pair_state.setdefault(
                key, {"max_speed": 0.0, "min_sep": 1e9, "max_sep": 0.0})
            sp_a, sp_b = a.speed_px(1.5), b.speed_px(1.5)
            st["max_sep"] = max(st["max_sep"], sep_norm)
            st["max_speed"] = max(st["max_speed"], sp_a, sp_b)
            st["min_sep"] = min(st["min_sep"], sep)

            # History is collected in the wider radius, but a conflict can be
            # armed only at physical contact/near-contact.
            if sep_norm > self.proximity_scale:
                continue

            da, db = a.direction(1.5, min_span=3.0), b.direction(1.5, min_span=3.0)
            if da is None or db is None:
                continue
            approach_angle = angle_between(da, db)
            # near-parallel motion at close range is ordinary lane-following,
            # not a conflict -- this is the guard that stops dense traffic
            # producing a continuous stream of false collisions
            if approach_angle < self.min_approach_angle:
                continue

            if st["max_speed"] < self.min_approach_speed:
                continue

            # CONVERGENCE. The pair must have STARTED far apart and closed the
            # distance. This is what separates a collision from a queue: queued
            # vehicles are permanently close to one another and never converge,
            # whereas a striking vehicle was several lengths away moments before
            # impact and then arrived. Measured in vehicle-lengths so it holds
            # at any resolution or depth in the frame.
            if st["max_sep"] < self.min_convergence_ratio * max(sep_norm, 1e-6):
                continue
            if st["max_sep"] < self.min_start_separation:
                continue

            acc_a, acc_b = a.acceleration_px(1.5), b.acceleration_px(1.5)
            decel = min(acc_a, acc_b)
            if decel > self.decel_threshold:
                continue

            heading_change = max(a.heading_change(1.5), b.heading_change(1.5))
            now_stopped = (sp_a < 3.0) or (sp_b < 3.0)

            e_prox = 1.0 - min(1.0, sep / max(self.proximity_px, 1e-6))
            e_ang = min(1.0, approach_angle / 90.0)
            e_dec = min(1.0, abs(decel) / max(abs(self.decel_threshold) * 2.0, 1e-6))
            e_head = min(1.0, heading_change / 60.0)
            score = float(np.clip(
                0.30 * e_prox + 0.25 * e_ang + 0.30 * e_dec + 0.15 * e_head, 0, 1
            ))
            if now_stopped:
                score = min(1.0, score * 1.2)

            key2 = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
            pend = self._pending.get(key2)
            if pend is None or sep < pend.get("sep", 1e9):
                self._pending[key2] = {
                    "t_closest": ctx.t, "sep": sep, "approach": score,
                    "speed_a": max(sp_a, st["max_speed"]),
                    "speed_b": max(sp_b, st["max_speed"]),
                    "detail": {
                        "track_ids": [a.track_id, b.track_id],
                        "corridor_id": a.corridor_id or b.corridor_id,
                        "proximity_px": round(sep, 1),
                        "footprint_separation": round(sep_norm, 2),
                        "max_footprint_separation": round(st["max_sep"], 2),
                        "convergence_ratio": round(st["max_sep"] / max(sep_norm, 1e-6), 2),
                        "aspect_shift_a": round(a.aspect_shift(2.5), 3),
                        "aspect_shift_b": round(b.aspect_shift(2.5), 3),
                        "approach_angle_deg": round(approach_angle, 1),
                        "decel_px_s2": round(decel, 1),
                        "heading_change_deg": round(heading_change, 1),
                        "speed_a_px_s": round(sp_a, 2),
                        "speed_b_px_s": round(sp_b, 2),
                        "class_a": int(a.cls), "class_b": int(b.cls),
                        "involves_vulnerable": bool(
                            a.cls in VULNERABLE_CLASSES or b.cls in VULNERABLE_CLASSES),
                    },
                }

            if score > best_score:
                best_score = score
                cid = a.corridor_id or b.corridor_id
                best_detail = {
                    "track_ids": [a.track_id, b.track_id],
                    "corridor_id": cid,
                    "proximity_px": round(sep, 1),
                    "min_separation_px": round(float(st["min_sep"]), 1),
                    "approach_angle_deg": round(approach_angle, 1),
                    "decel_px_s2": round(decel, 1),
                    "heading_change_deg": round(heading_change, 1),
                    "speed_a_px_s": round(sp_a, 2),
                    "speed_b_px_s": round(sp_b, 2),
                    "class_a": int(a.cls),
                    "class_b": int(b.cls),
                    "either_stopped": bool(now_stopped),
                    "involves_vulnerable": bool(
                        a.cls in VULNERABLE_CLASSES or b.cls in VULNERABLE_CLASSES
                    ),
                }

        live = {tr.track_id for tr in ctx.tracks}
        for k in list(self._pair_state):
            if k[0] not in live and k[1] not in live:
                self._pair_state.pop(k, None)

        return best_score, best_detail

    def _confidence(self, conflict: float, cp: float, detail: dict,
                    n_channels: int = 1, localised: bool = False,
                    static_score: float = 0.0) -> float:
        """Deliberately capped below 0.9.

        This event type always requires human verification, so presenting it as
        near-certain would be misleading regardless of how much evidence
        accumulated. Agreement between the two independent detectors -- one
        object-centric, one pixel-centric -- is what raises it most, because
        they fail in uncorrelated ways.
        """
        base = 0.26 + 0.26 * conflict + 0.14 * cp + 0.22 * static_score
        base += 0.10 * max(0, n_channels - 1)
        if localised:
            base += 0.05
        if detail.get("either_stopped"):
            base += 0.05
        if detail.get("involves_vulnerable"):
            base += 0.03
        return float(np.clip(base, 0.0, 0.88))
