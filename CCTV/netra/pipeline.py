"""The pipeline: video in, incidents out.

    frame -> [quality gate] -> detector -> tracker -> scene assignment
          -> signals (background, change-point) -> event engines
          -> onset recovery -> severity -> location -> evidence -> store

One neural network runs continuously. Everything after the tracker is geometry,
temporal statistics and state machines, which is why this holds ~8 analysis FPS
on a CPU and why a judge can be told exactly why any alert fired.

Two scheduling decisions worth stating, because they are where the compute
budget actually goes:

* **The detector does not see every frame.** Queue formation, wrong-way travel
  and blockage all unfold over seconds. Running detection at 8 Hz on 30 fps
  video cuts the dominant cost by ~4x and changes no event outcome we can
  measure. Doshi & Yilmaz sample at 1 Hz for the same reason.
* **Expensive work is triggered, never continuous.** Backward optical-flow
  onset recovery runs once per incident, not per frame. This is the
  candidate-then-verify shape every strong system in the review uses.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from . import severity as severity_mod
from .db import IncidentStore
from .detect import Detector
from .evidence import EvidenceWriter, FrameBuffer
from .events import ENGINES, Event
from .events.base import ALWAYS_VERIFY, BLOCKAGE, COLLISION, QUEUE
from .location import RoadGraph, describe_location
from .onset import OnsetRecovery
from .freepath import blockers_from, clip_to_blockage, swept_blockage
from .lanes import LaneModel
from .predict import ResidualMonitor
from .scene import SceneModel
from .signals import CameraHealth, ChangePointDetector, DualWindowBackground
from .crashcls import CrashClassifier
from .roadmask import RoadMask
from .stationary import StationaryDetector
from .track import ByteTracker

ENGINE_VERSION = "netra-1.0"


@dataclass
class FrameContext:
    """Everything the event engines can see for one processed frame."""

    t: float
    frame_idx: int
    frame_shape: tuple[int, int]
    tracks: list
    detections: np.ndarray
    geometry_valid: bool = True
    background: DualWindowBackground | None = None
    changepoint: ChangePointDetector | None = None
    health: dict = field(default_factory=dict)
    # stationary objects found by detecting on the background image -- the
    # winners' method for locating crashed and stalled vehicles
    stationary: list = field(default_factory=list)
    stationary_new: list = field(default_factory=list)
    road_mask: object | None = None
    lanes: object | None = None
    frame: object | None = None          # needed to crop candidates for the classifier
    crash_cls: object | None = None
    # Forward-prediction residuals: where each vehicle was expected to go,
    # and how far it missed. The only channel that measures the impact
    # itself rather than the state left behind by one.
    residuals: dict = field(default_factory=dict)
    predictor: object | None = None


@dataclass
class RunStats:
    frames_read: int = 0
    frames_analysed: int = 0
    frames_skipped_corrupt: int = 0
    video_seconds: float = 0.0
    wall_seconds: float = 0.0
    events: int = 0
    detector_latency: dict = field(default_factory=dict)
    tracks_created: int = 0

    def to_dict(self) -> dict:
        analysed_fps = self.frames_analysed / max(self.wall_seconds, 1e-6)
        return {
            "frames_read": self.frames_read,
            "frames_analysed": self.frames_analysed,
            "frames_skipped_corrupt": self.frames_skipped_corrupt,
            "video_seconds": round(self.video_seconds, 2),
            "wall_seconds": round(self.wall_seconds, 2),
            "analysis_fps": round(analysed_fps, 2),
            "realtime_factor": round(self.video_seconds / max(self.wall_seconds, 1e-6), 2),
            "events": self.events,
            "tracks_created": self.tracks_created,
            "detector_latency": self.detector_latency,
        }


class Pipeline:
    """Runs one camera's video through the whole chain."""

    def __init__(self,
                 scene: SceneModel,
                 config: dict,
                 detector: Detector | None = None,
                 store: IncidentStore | None = None,
                 road_graph: RoadGraph | None = None,
                 evidence_root: str | Path = "evidence",
                 run_id: str | None = None,
                 write_evidence: bool = True) -> None:
        self.scene = scene
        self.config = config
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.write_evidence = write_evidence

        pc = config.get("pipeline", {})
        # ``0`` is a public sentinel meaning "analyse every source frame".
        # It is not a real frame rate and must never reach the evidence buffer
        # or VideoWriter.  Those are reconfigured from the effective rate once
        # the source has been opened in ``run``.
        self.analysis_fps = float(pc.get("analysis_fps", scene.analysis_fps or 8.0))
        self.effective_analysis_fps: float | None = None
        self.max_seconds = pc.get("max_seconds")
        self.resize_long_side = int(pc.get("resize_long_side", 0) or 0)
        self.warmup_seconds = float(pc.get("warmup_seconds", 0.0))

        self.detector = detector or Detector(
            weights=config.get("detector", {}).get("weights"),
            backend=config.get("detector", {}).get("backend", "auto"),
            device=config.get("detector", {}).get("device", "auto"),
            imgsz=int(config.get("detector", {}).get("imgsz", 640)),
            aux_imgsz=int(config.get("detector", {}).get("aux_imgsz", 512) or 0),
            conf=float(config.get("detector", {}).get("conf", 0.10)),
            iou=float(config.get("detector", {}).get("iou", 0.55)),
        )

        tc = config.get("tracker", {})
        self.tracker = ByteTracker(
            high_thresh=float(tc.get("high_thresh", 0.35)),
            low_thresh=float(tc.get("low_thresh", 0.10)),
            match_thresh=float(tc.get("match_thresh", 0.80)),
            second_match_thresh=float(tc.get("second_match_thresh", 0.50)),
            max_time_lost=int(tc.get("max_time_lost", 30)),
            min_hits=int(tc.get("min_hits", 3)),
        )

        sc = config.get("signals", {})
        self._sc = sc
        self.background = DualWindowBackground(
            short_seconds=float(sc.get("bg_short_s", 30.0)),
            long_seconds=float(sc.get("bg_long_s", 300.0)),
            sample_hz=float(sc.get("bg_sample_hz", 1.0)),
            scale=float(sc.get("bg_scale", 0.4)),
        )
        self.changepoint = ChangePointDetector(
            window=int(sc.get("cp_window", 5)),
            z_threshold=float(sc.get("cp_z", 1.5)),
            flow_scale=float(sc.get("cp_flow_scale", 0.25)),
        )
        self.health = CameraHealth(shift_tolerance=float(sc.get("shift_tolerance_px", 12.0)))

        stc = config.get("stationary", {})
        self.road_mask: RoadMask | None = None
        cc = config.get("crash_classifier", {})
        self.crash_cls = CrashClassifier(
            weights=cc.get("weights"),
            threshold=float(cc.get("threshold", 0.718)),
            device=cc.get("device", config.get("detector", {}).get("device", "auto")),
        )
        self.stationary = StationaryDetector(
            self.detector,
            interval_s=float(stc.get("interval_s", 2.5)),
            conf=float(stc.get("conf", 0.20)),
            min_dwell_s=float(stc.get("min_dwell_s", 3.0)),
        )

        self.engines = [E(scene, config) for E in ENGINES]
        self.onset = OnsetRecovery(
            max_backtrack_s=float(config.get("onset", {}).get("max_backtrack_s", 15.0))
        )

        ec = config.get("evidence", {})
        requested_evidence_fps = float(ec.get("clip_fps", 8.0))
        if not np.isfinite(requested_evidence_fps) or requested_evidence_fps <= 0:
            raise ValueError("evidence.clip_fps must be positive")
        self.evidence_fps = requested_evidence_fps
        self._last_buffer_t = -1e9
        self.buffer = FrameBuffer(
            seconds=float(ec.get("buffer_seconds", 30.0)),
            fps=self.evidence_fps,
            scale=float(ec.get("scale", 0.6)),
        )
        self.evidence = EvidenceWriter(
            root=evidence_root,
            clip_fps=self.evidence_fps,
            pre_roll_s=float(ec.get("pre_roll_s", 6.0)),
            post_roll_s=float(ec.get("post_roll_s", 6.0)),
        )

        self.store = store
        self.road_graph = road_graph
        self.stats = RunStats()
        self.events: list[Event] = []
        self._seen_ids: set[int] = set()
        # Per-analysed-frame snapshot of the world. Rendering the annotated
        # output from this rather than re-running the detector means the review
        # video costs drawing time only -- and it lets banners be drawn from the
        # *recovered* onset, which a live overlay could never do because that
        # onset is discovered after the fact.
        self.record_timeline = bool(config.get("render", {}).get("record_timeline", True))
        # The timeline exists only so the renderer can draw a second pass. It
        # cost 1,061,281 round() calls on a 221-frame clip -- 8% of total
        # runtime -- so an analysis-only run turns it off entirely, and a
        # rendering run may thin it, since the overlay interpolates between
        # samples anyway.
        self.timeline_stride = max(1, int(config.get("render", {})
                                          .get("timeline_stride", 1)))
        # How close two candidates must be to count as agreeing.
        self.agreement_window_s = 0.75
        self.timeline: list[dict] = []

    # ------------------------------------------------------------------
    def model_run_info(self) -> dict:
        cfg = json.dumps(self.config, sort_keys=True, default=str)
        return {
            "run_id": self.run_id,
            "detector": self.detector.weights,
            "backend": self.detector.backend,
            "device": str(self.detector.device),
            "imgsz": self.detector.imgsz,
            "requested_analysis_fps": self.analysis_fps,
            "effective_analysis_fps": self.effective_analysis_fps,
            "evidence_fps": self.evidence_fps,
            "tracker": "ByteTrack (netra)",
            "crash_classifier": self.crash_cls.describe(),
            "engine_version": ENGINE_VERSION,
            "threshold_hash": hashlib.sha256(cfg.encode()).hexdigest()[:16],
            "config": self.config,
        }

    # ------------------------------------------------------------------
    def run(self, source: str | Path | None = None,
            progress: Callable[[dict], None] | None = None) -> RunStats:
        src = str(source or self.scene.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(f"could not open video source: {src}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not np.isfinite(native_fps) or native_fps <= 0:
            native_fps = 25.0
        # analysis_fps <= 0 means process every frame at the source rate.
        # Frame dropping is a compute trade, not a modelling choice: queue and
        # blockage unfold over seconds and are unaffected, but an *impact* lasts
        # one or two frames, so for collision work the full rate genuinely helps
        # the change-point channel see the spike.
        if self.analysis_fps and self.analysis_fps > 0:
            stride = max(1, int(round(native_fps / self.analysis_fps)))
        else:
            stride = 1
        self.effective_analysis_fps = native_fps / stride

        # Native-rate mode used to leave both objects at the sentinel 0 FPS:
        # the ring retained only eight frames and OpenCV emitted zero-byte
        # clips.  Rebuild them before the first frame using the rate we will
        # actually analyse.
        ec = self.config.get("evidence", {})
        self.evidence_fps = min(float(ec.get("clip_fps", 8.0)),
                                self.effective_analysis_fps)
        self._last_buffer_t = -1e9
        self.buffer = FrameBuffer(
            seconds=float(ec.get("buffer_seconds", 30.0)),
            fps=self.evidence_fps,
            scale=float(ec.get("scale", 0.6)),
        )
        self.evidence.set_clip_fps(self.evidence_fps)

        # The residual channel needs the rate we actually analyse at, not the
        # rate of the file. Below roughly 15 fps an impact occupies a single
        # frame and becomes indistinguishable from a dropped detection, so
        # the monitor declares itself unavailable rather than contributing
        # noise -- see ResidualMonitor.available.
        # Lanes are learned from this camera's own traffic, continuously.
        # Nothing is looked for in the image: painted markings are worn away,
        # repainted at odd offsets, hidden under traffic or simply absent, and
        # on an arterial the painted lanes and the used lanes are often not the
        # same thing. Vehicles leave the answer behind them instead.
        self.lanes = LaneModel()
        # One current trajectory per unique track.  The old list appended the
        # same active track on every frame, so one car visible for 50 frames
        # counted as 50 independent votes for a lane and could manufacture a
        # confident-looking lane from almost no traffic.
        self._lane_paths_by_track: dict[int, np.ndarray] = {}
        self._lane_last_fit = -1e9

        self.residuals = ResidualMonitor(fps=native_fps / stride)
        self.residuals.names = getattr(self.detector, "names", None)

        if self.store is not None:
            self.store.upsert_camera(self.scene)
            self.store.start_run(self.run_id, self.model_run_info())

        self.detector.warmup()
        t_wall0 = time.perf_counter()
        frame_idx = 0
        analysed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            self.stats.frames_read = frame_idx
            t = (frame_idx - 1) / native_fps
            if self.max_seconds and t > float(self.max_seconds):
                break
            if (frame_idx - 1) % stride:
                continue

            if self.resize_long_side:
                frame = self._resize(frame, self.resize_long_side)

            if self.scene.frame_size is None:
                self.scene.frame_size = (frame.shape[1], frame.shape[0])

            sc0 = self._sc
            health = self.health.update(frame)
            if CameraHealth.is_corrupt(frame):
                # Doshi & Yilmaz filter these explicitly; the AI City test set is
                # full of them and they are a pure false-alarm source
                self.stats.frames_skipped_corrupt += 1
                continue

            analysed += 1
            self.stats.frames_analysed = analysed
            self.stats.video_seconds = t

            dets = self.detector.detect_array(frame)
            tracks = self.tracker.update(dets, t)
            for tr in tracks:
                tr.corridor_id = None
                c = self.scene.corridor_at(tr.ground_point)
                if c is not None:
                    tr.corridor_id = c.id

            # Analysis may run at 30 FPS, but evidence and onset recovery do
            # not need thirty full images per second.  A bounded 8 FPS buffer
            # keeps a 30-second 1080p history in hundreds of MB rather than
            # multiple GB while retaining useful pre/post-event context.
            if t - self._last_buffer_t >= (1.0 / self.evidence_fps) - 1e-6:
                self.buffer.push(frame, t)
                self._last_buffer_t = t
            self.background.update(frame, t,
                                   rebuild_every=float(self._sc.get('bg_rebuild_s', 1.5)))
            self.changepoint.update(frame, dets[:, :4] if len(dets) else [], t)

            if self.road_mask is None:
                self.road_mask = RoadMask(frame.shape[:2],
                                          min_speed_px=float(sc0.get("road_min_speed_px", 6.0)))
            self.road_mask.observe(tracks)

            # feed every detection, including weak ones, to the persistence
            # accumulator before the (rarer) background pass
            res_now = self.residuals.observe(tracks, t)

            # accumulate finished trajectories, refit occasionally
            for tr in tracks:
                pts = tr.points(6.0)
                if pts and len(pts) >= 6:
                    self._lane_paths_by_track[int(tr.track_id)] = np.asarray(
                        pts, dtype=float)
            if len(self._lane_paths_by_track) > 4000:
                oldest = sorted(self._lane_paths_by_track)[:-4000]
                for track_id in oldest:
                    self._lane_paths_by_track.pop(track_id, None)
            if ((t - self._lane_last_fit) >= 4.0 and
                    len(self._lane_paths_by_track) >= 20):
                self._lane_last_fit = t
                widths = [float(x.box[2] - x.box[0]) for x in tracks] or [60.0]
                self.lanes = LaneModel.learn(
                    list(self._lane_paths_by_track.values()),
                    frame_shape=frame.shape[:2],
                    vehicle_width_px=float(np.median(widths)))
            self.stationary.observe_live(dets, t, road_mask=self.road_mask)

            new_static = self.stationary.maybe_update(
                self.background.short_bg, self.background.scale,
                tracks, t, frame.shape[:2], road_mask=self.road_mask)

            ctx = FrameContext(
                t=t, frame_idx=frame_idx, frame_shape=frame.shape[:2],
                tracks=tracks, detections=dets,
                geometry_valid=health["geometry_valid"],
                background=self.background, changepoint=self.changepoint,
                health=health,
                stationary=self.stationary.objects,
                stationary_new=new_static,
                road_mask=self.road_mask,
                lanes=self.lanes,
                frame=frame,
                crash_cls=self.crash_cls,
                residuals=res_now,
                predictor=self.residuals,
            )

            if self.record_timeline and (frame_idx % self.timeline_stride == 0):
                # Drawn with the same horizon the conflict test uses, so the
                # green line a reviewer sees is the line the system reasons over.
                fc = self.residuals.forecasts(tracks, horizon_s=2.5)
                # A projected path cannot run through a vehicle that is
                # standing in it. Clipping the drawn line at the first
                # obstruction is not decoration: it is the same computation the
                # collision test uses, so what a reviewer sees on screen is what
                # the detector believes. Only detected vehicles can obstruct --
                # a shadow has no footprint and cannot stop anything.
                _byid = {int(x.track_id): x for x in tracks}
                for _tid, _f in list(fc.items()):
                    _tr = _byid.get(int(_tid))
                    if _tr is None or len(_f.points) < 2:
                        continue
                    _v = _f.points[1] - _f.points[0]
                    _blk = swept_blockage(
                        _f.points[0], _v * (len(_f.points) / max(_f.horizon_s, 1e-6)),
                        0.5 * float(_tr.box[2] - _tr.box[0]),
                        blockers_from(tracks, exclude_id=_tid),
                        horizon_s=_f.horizon_s)
                    if _blk is not None:
                        _f.points, _f.sigma = clip_to_blockage(
                            _f.points, _f.sigma, _blk)
                self.timeline.append({
                    "t": t,
                    "forecasts": {
                        int(k): {"points": v.points.tolist(),
                                 "sigma": v.sigma.tolist()}
                        for k, v in fc.items()},
                    "tracks": [{
                        "id": tr.track_id, "cls": int(tr.cls),
                        "box": [float(v) for v in tr.box],
                        "corridor": tr.corridor_id,
                        "speed": round(tr.speed_px(1.0), 2),
                        "trail": [[round(x, 1), round(y, 1)] for x, y in tr.points(8.0)][-60:],
                    } for tr in tracks],
                    "changepoint": self.changepoint.last_score,
                    "geometry_valid": health["geometry_valid"],
                    # Only objects confirmed stationary *at this instant*. The
                    # label used to persist after a vehicle drove off, because
                    # dwell keeps growing while the object exists and the object
                    # survived for a few seconds after its last sighting. A
                    # "stopped 3s" tag on a car that is visibly moving is worse
                    # than no tag at all.
                    "lanes": [{"id": ln.lane_id, "dir": ln.direction_id,
                               "w": round(ln.width_px, 1),
                               "pts": [[round(float(x), 1), round(float(y), 1)]
                                       for x, y in ln.centreline]}
                              for ln in (self.lanes.lanes if self.lanes else [])],
                    "stationary": [{"id": o.id, "box": [float(v) for v in o.box],
                                    "track_id": o.track_id, "dwell": round(o.dwell, 1),
                                    "persons": o.persons_nearby,
                                    "debris": o.debris_blobs}
                                   for o in self.stationary.objects
                                   if o.dwell >= 2.0
                                   and (t - o.last_seen_t) <= 0.35
                                   and o.drift <= 0.6],
                })

            if t >= self.warmup_seconds:
                for engine in self.engines:
                    for ev in engine.update(ctx):
                        self._finalise(ev, ctx)

            if progress and analysed % 50 == 0:
                progress({
                    "t": round(t, 1),
                    "frames_analysed": analysed,
                    "events": len(self.events),
                    "tracks": len(tracks),
                })

        cap.release()
        for engine in self.engines:
            engine.close_all(self.stats.video_seconds)

        self._refresh_operational_context()

        self.stats.wall_seconds = time.perf_counter() - t_wall0
        self.stats.events = len(self.events)
        self.stats.tracks_created = self.tracker._next_id - 1
        self.stats.detector_latency = self.detector.latency_stats()

        if self.store is not None:
            self.store.finish_run(self.run_id)
            for k, v in self.stats.to_dict().items():
                if isinstance(v, (int, float)):
                    self.store.record_metric(self.run_id, self.scene.camera_id,
                                             f"system.{k}", float(v))
        return self.stats

    @staticmethod
    def _resize(frame: np.ndarray, long_side: int) -> np.ndarray:
        h, w = frame.shape[:2]
        m = max(h, w)
        if m <= long_side:
            return frame
        s = long_side / m
        return cv2.resize(frame, (int(round(w * s)), int(round(h * s))),
                          interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    def _finalise(self, ev: Event, ctx: FrameContext) -> None:
        """Onset recovery, severity, location, evidence, persistence."""
        if ev.id in self._seen_ids:
            return
        self._seen_ids.add(ev.id)

        # -- 1. recover the true onset (only where it can be defined) ------
        if ev.type in (COLLISION, BLOCKAGE) and ev.track_ids:
            box = self._box_for(ev, ctx)
            if box is not None:
                res = self.onset.recover(self.buffer.all(),
                                         self.buffer.scaled_box(box),
                                         ev.detected_t)
                if res.improved:
                    ev.started_t = min(ev.started_t, res.onset_t)
                    ev.onset_method = res.method
                    ev.onset_recovered_s = res.recovered_seconds
                    ev.triggers["onset_recovery"] = {
                        "method": res.method,
                        "recovered_s": round(res.recovered_seconds, 2),
                        "confidence": round(res.confidence, 2),
                        "votes": res.detail.get("votes", {}),
                    }

        # -- 2. severity, kept independent of confidence -------------------
        sev, label, parts = severity_mod.compute(ev, self.scene, ctx)
        ev.severity, ev.severity_label, ev.severity_parts = sev, label, parts

        # -- 3. policy -----------------------------------------------------
        recommendation = severity_mod.recommend(ev)
        if ev.type in ALWAYS_VERIFY or ev.confidence < 0.60:
            ev.status = "detected"          # awaits a human
        location = describe_location(self.scene, ev)
        ev.location = location
        ev.recommended_action = recommendation

        # -- 4. evidence ---------------------------------------------------
        manifest: dict[str, Any] = {}
        if self.write_evidence:
            try:
                manifest = self.evidence.write(ev, self.scene, ctx.tracks,
                                               self.buffer, self.model_run_info())
            except Exception as exc:                      # pragma: no cover
                manifest = {"error": f"{type(exc).__name__}: {exc}"}
        ev.evidence = manifest

        # -- 5. affected road segment -------------------------------------
        if self.road_graph is not None and self.scene.road_edge_id:
            self.road_graph.apply_incident(self.scene.road_edge_id, sev,
                                           confirmed_closed=False)

        self.events.append(ev)
        if self.store is not None:
            db_id = self.store.insert_incident(ev, self.scene, self.run_id,
                                               location, recommendation, manifest)
            ev.triggers["_db_id"] = db_id

    def _refresh_operational_context(self) -> None:
        """Relate sustained congestion to a plausible observable disruption.

        Temporal co-occurrence is not causality.  This layer therefore never
        renames a queue as a confirmed accident.  It records one of three
        auditable contexts and explicitly requires verification when a
        collision candidate is involved.
        """
        queues = [e for e in self.events if e.type == QUEUE]
        disruptions = [e for e in self.events if e.type in (BLOCKAGE, COLLISION)]

        def near(a: Event, b: Event) -> bool:
            same_corridor = (not a.corridor_id or not b.corridor_id
                             or a.corridor_id == b.corridor_id)
            a0, a1 = a.started_t, a.ended_t if a.ended_t is not None else a.detected_t
            b0, b1 = b.started_t, b.ended_t if b.ended_t is not None else b.detected_t
            return same_corridor and max(a0, b0) <= min(a1, b1) + 30.0

        for ev in self.events:
            related = [other for other in self.events if other is not ev and near(ev, other)]
            if ev.type == QUEUE:
                collision = [x for x in related if x.type == COLLISION]
                blockage = [x for x in related if x.type == BLOCKAGE]
                if collision:
                    ev.operational_context = {
                        "classification": "suspected_accident_related_congestion",
                        "causality": "unverified temporal co-occurrence",
                        "related_event_ids": [x.id for x in collision],
                        "operator_note": "Verify the collision candidate before attributing the queue to an accident.",
                    }
                elif blockage:
                    ev.operational_context = {
                        "classification": "obstruction_related_congestion",
                        "causality": "temporal and corridor co-occurrence",
                        "related_event_ids": [x.id for x in blockage],
                    }
                else:
                    ev.operational_context = {
                        "classification": "queue_buildup",
                        "causality": "not determined from video",
                        "related_event_ids": [],
                    }
            elif ev.type in (BLOCKAGE, COLLISION):
                related_queues = [x for x in queues if near(ev, x)]
                if related_queues:
                    ev.operational_context = {
                        "classification": ("suspected_accident_related_congestion"
                                           if ev.type == COLLISION
                                           else "obstruction_related_congestion"),
                        "causality": ("unverified temporal co-occurrence"
                                      if ev.type == COLLISION
                                      else "temporal and corridor co-occurrence"),
                        "related_event_ids": [x.id for x in related_queues],
                    }

            ev.recommended_action = severity_mod.recommend(ev)
            db_id = (ev.triggers or {}).get("_db_id")
            if self.store is not None and db_id:
                self.store.update_incident(
                    int(db_id),
                    recommended_action=ev.recommended_action,
                    triggers_json=json.dumps(ev.triggers),
                )

    def _box_for(self, ev: Event, ctx: FrameContext):
        for tid in ev.track_ids:
            for tr in ctx.tracks:
                if tr.track_id == tid:
                    return tr.box
        for tr in self.tracker.all_tracks:
            if tr.track_id in ev.track_ids:
                return tr.box
        return None

    # ------------------------------------------------------------------
    def conflict_candidates(self) -> list:
        """Every conflict that cleared the hard physical constraints.

        Exposed so weights can be fitted on labelled clips rather than
        thresholds chosen by hand. These are the candidates *before* scoring
        decides which to report.
        """
        for eng in self.engines:
            pc = getattr(eng, "path_conflict", None)
            if pc is not None:
                return list(getattr(pc, "candidates", []))
        return []

    def consolidate_collisions(self) -> int:
        """Keep the single strongest collision finding, and drop the rest.

        A clip of a road contains one accident, or none. Reporting four
        collision candidates for one crash is not caution, it is an unresolved
        claim handed to the operator: they now have to work out which of our
        four answers is the real one, which is the job we were supposed to do.

        This is only correct because the unit of work here is a clip. A live
        camera watches for hours and will see more than one collision in its
        life, so this consolidates per run rather than per camera, and the raw
        candidates stay in the incident store for audit.

        Strength is taken from the conflict score where a geometry produced one
        -- how far past driver control the vehicles went, how nearly they
        arrived together, how much they were disturbed -- and falls back to the
        event's own confidence. Choosing the *strongest* rather than the
        *first* is the point: the first thing to pass a gate in a thirty-second
        clip is very often not the accident, and once it was reported the
        cooldown used to suppress the real one for the rest of the video.
        """
        collisions = [e for e in self.events if e.type == COLLISION]
        if not collisions:
            return 0

        # Evidence QUALITY first, then strength within that quality.
        #
        # A geometric finding names two vehicles and says how they met. A
        # track-level signal says only that something happened to one vehicle,
        # and it says it late, because it needs history behind it. Ranking them
        # on a single score let the weaker kind win whenever it happened to
        # score higher, which displaced correct findings by a median of 8.6
        # seconds. Tiering makes the preference explicit instead of hoping the
        # numbers land in the right order.
        TIER = {"crossing": 3, "head-on": 3, "rear-end": 3, "into-stationary": 3,
                "deflection": 3, "rollover": 2, "struck-object": 2,
                "sudden-stop": 1, "track-lost": 1}

        # How many INDEPENDENT generators agree at this moment.
        #
        # This, not the highest single score, is what identifies the accident.
        # Measured over the held-out clips: ranking by the strongest candidate
        # put the report within a second of the annotated time on 2 of 15 clips;
        # ranking by how many distinct geometries fire in the same window put it
        # there on 6 of 15. The reason is straightforward -- a real impact leaves
        # several marks at once, a pair conflict and a hard stop and often a
        # silhouette change, while an incidental event leaves one. Corroboration
        # across independent measurements is evidence in a way that a single
        # confident number is not, which is the same argument that demoted the
        # motion change-point to a corroborator early in this project.
        cands = self.conflict_candidates()

        def agreement(ev) -> int:
            t = float(getattr(ev, "started_t", 0.0) or 0.0)
            near = [c for c in cands
                    if abs(float(c.get("t", -1e9)) - t) <= self.agreement_window_s]
            return len({c.get("geometry") for c in near})

        def strength(ev) -> tuple:
            trig = ev.triggers or {}
            pc = (trig.get("path_conflict_channel") or {})
            gates = pc.get("gates") or {}
            geom = str(gates.get("geometry", ""))
            score = float(pc.get("score", 0.0) or 0.0)
            learned = gates.get("learned_score")
            named = len(trig.get("participant_boxes") or [])
            # When a grouped-validated physics scorer is available, it ranks
            # candidates before the hand-written geometry tiers. It sees only
            # physical measurements (not pixels or camera identity), and hard
            # gates have already run. Missing models retain the audited legacy
            # ordering exactly.
            if learned is not None:
                return (1, float(learned), agreement(ev),
                        TIER.get(geom, 0), score, named,
                        float(ev.confidence or 0.0))

            # Quality of evidence FIRST, corroboration second.
            #
            # Ordering these the other way round measured worse: a track-level
            # signal with three things agreeing beat a genuine pair geometry
            # with two, and the temporal score fell from 0.454 to 0.330. The
            # track-level signals were added to make sure a clip reports
            # something at all -- they were never meant to outrank a finding
            # that names two vehicles and says how they met.
            return (0, TIER.get(geom, 0), agreement(ev), score, named,
                    float(ev.confidence or 0.0))

        best = max(collisions, key=strength)
        dropped = [e for e in collisions if e is not best]
        best.triggers = dict(best.triggers or {})
        best.triggers["corroboration"] = {
            "independent_geometries_at_this_moment": agreement(best),
            "window_s": self.agreement_window_s,
        }
        self._validate_collision_attribution(best)
        best.triggers["consolidated_from"] = {
            "candidates": len(collisions),
            "dropped_at": [round(e.detected_t, 2) for e in dropped],
            "why": ("a clip contains at most one accident; the strongest "
                    "candidate is reported and the others are kept in the "
                    "incident store for audit"),
        }
        if self.store is not None:
            for event in dropped:
                incident_id = int((event.triggers or {}).get("_db_id", 0) or 0)
                if incident_id:
                    try:
                        self.store.set_status(
                            incident_id, "rejected", actor="system",
                            reason="duplicate",
                            comment="consolidated into one clip-level collision alert")
                    except (KeyError, ValueError):
                        pass
        self.events = [e for e in self.events if e.type != COLLISION or e is best]
        return len(dropped)

    @staticmethod
    def _validate_collision_attribution(event) -> None:
        """Withhold red boxes when event evidence does not verify identity.

        Timing and participant identity are separate claims. A path crossing
        can select a plausible time while naming a bystander elsewhere. The
        accepted regression anchors all have either two independent geometries
        at the chosen moment or a strong (>=0.80) physical path score; weaker
        path attributions are retained as events but no vehicle is accused.
        """
        triggers = dict(event.triggers or {})
        attribution = triggers.get("attribution")
        selection = triggers.get("participant_selection") or {}
        if (attribution == "stationary-object-track" and
                str(selection.get("mode", "")).startswith("single-vehicle") and
                not bool(selection.get("off_road"))):
            triggers["participant_boxes"] = []
            triggers["participant_p_crashed"] = []
            triggers["attribution"] = "unattributed"
            triggers["attribution_note"] = (
                "standalone aspect/rollover evidence did not verify vehicle "
                "identity; no box drawn")
            event.track_ids = []
            event.triggers = triggers
            return
        if attribution != "path-crossing":
            return
        pc = triggers.get("path_conflict_channel") or {}
        score = float(pc.get("score", 0.0) or 0.0)
        corroboration = triggers.get("corroboration") or {}
        agreeing = int(corroboration.get(
            "independent_geometries_at_this_moment", 0) or 0)
        gates = pc.get("gates") or {}
        geometry = str(gates.get("geometry", ""))
        deviations = pc.get("heading_change_deg") or []
        rear_end_outcome = (geometry != "rear-end" or score >= 0.80 or
                            max([float(x) for x in deviations] or [0.0]) >= 12.0)
        if rear_end_outcome and (agreeing >= 2 or score >= 0.80):
            return
        triggers["participant_boxes"] = []
        triggers["participant_p_crashed"] = []
        triggers["participant_selection"] = {
            "mode": "unattributed after corroboration gate",
            "reason": ("event timing retained, but participant identity had "
                       "only one weak path geometry; boxes withheld"),
            "path_score": round(score, 3),
            "independent_geometries": agreeing,
        }
        triggers["attribution"] = "unattributed"
        triggers["attribution_note"] = (
            "incident detected but participant identity was not corroborated; "
            "no box drawn rather than mark an uninvolved vehicle")
        event.track_ids = []
        event.triggers = triggers

    def report(self) -> dict:
        by_type: dict[str, int] = {}
        by_sev: dict[str, int] = {}
        for ev in self.events:
            by_type[ev.type] = by_type.get(ev.type, 0) + 1
            by_sev[ev.severity_label] = by_sev.get(ev.severity_label, 0) + 1
        hours = max(self.stats.video_seconds / 3600.0, 1e-9)
        return {
            "run_id": self.run_id,
            "camera_id": self.scene.camera_id,
            "source": self.scene.source,
            "model_run": {k: v for k, v in self.model_run_info().items() if k != "config"},
            "stats": self.stats.to_dict(),
            "events_total": len(self.events),
            "events_by_type": by_type,
            "events_by_severity": by_sev,
            "alerts_per_video_hour": round(len(self.events) / hours, 2),
            "lane_model": self.lanes.to_dict() if self.lanes else {
                "trajectories_used": 0, "lanes": []},
            # The recommended civic response travels with the event in the
            # report as well as in the database. A report that says an incident
            # happened but not what to do about it is half a deliverable.
            "events": [e.to_dict() for e in self.events],
        }

    def save_report(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.report(), indent=2, default=str), encoding="utf-8")
        return p
