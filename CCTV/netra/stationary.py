"""Finding crashed and stalled vehicles on the background image.

This module implements the single most important idea shared by every winning
AI City anomaly-detection entry, and its absence was the reason our first
attempt at clip 14 failed so badly.

The problem with live frames
----------------------------
A crashed vehicle at night is the *hardest* thing in the frame to detect. It is
motion-blurred at impact, often partly occluded by the vehicle that hit it, at
an unusual orientation, and surrounded by other traffic. Measured on our own
footage, ``yolo12n`` scored the two colliding vehicles at 0.08 and 0.05 -- below
any usable threshold -- while confidently detecting the undamaged cars parked
further up the road. The system then, quite logically, put its box around a
vehicle that was perfectly fine.

The winners' insight
--------------------
Run background modelling first. Moving traffic melts away; anything that stays
is stationary. A crashed vehicle *is* stationary, and in the background image it
appears crisp, isolated, unoccluded and without motion blur -- the easiest
possible input for a detector. Zhao et al. (1st, 2021), Chen et al., Doshi &
Yilmaz and Aboah et al. all detect on the background; Li et al. (1st, 2020,
S4 0.9695) combine box-level and pixel-level branches over it.

So the detector is run a second time, rarely, on the background image, and what
it finds there is treated as a stationary-object hypothesis. Cost is one extra
inference every few seconds -- roughly 3% of the detection budget.

Three corroborating cues
------------------------
A stationary vehicle alone is not a crash; it could be parked. Three
post-crash signatures raise it, all of which a camera can genuinely see:

* **Occupants get out.** A person appearing beside a newly-stopped vehicle, on
  the carriageway, is one of the strongest and earliest post-crash signals.
* **A second vehicle stops beside the first.** Collisions involve at least two
  parties, and they come to rest together.
* **Debris.** Impact scatters fragments across the road; they show up as new,
  small, static foreground blobs near the vehicle that were not there before.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import MOTORISED_CLASSES, VULNERABLE_CLASSES
from .geometry import iou, iou_matrix


@dataclass
class StationaryObject:
    """A vehicle-shaped thing that has stopped and stayed stopped."""

    id: int
    box: np.ndarray
    cls: int
    score: float
    first_seen_t: float
    last_seen_t: float
    hits: int = 1
    track_id: int | None = None
    persons_nearby: int = 0
    companions: int = 0
    debris_blobs: int = 0
    road_coverage: float = 0.0      # how much of it sits on the drivable surface
    present_from_start: bool = False
    arrived_moving: bool = False    # we saw it drive here, then stop
    ever_moved: bool = False        # moved at all, at any point
    track_seen: bool = False        # a track was ever associated with it
    stop_decel: float = 0.0         # how hard it stopped (px/s^2, positive)
    queue_member: bool = False      # part of a stationary chain
    anchor: np.ndarray | None = None   # centre when first seen
    peak_speed: float = 0.0
    aspect_shift: float = 0.0       # silhouette change -- rollover / spin
    reported: bool = False

    @property
    def dwell(self) -> float:
        return max(0.0, self.last_seen_t - self.first_seen_t)

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2.0,
                         (self.box[1] + self.box[3]) / 2.0], dtype=float)

    @property
    def drift(self) -> float:
        """How far this object has moved from where it was first seen.

        Expressed in box widths, so it is scale-free. A genuinely stationary
        vehicle stays near zero. Measured on one held-out clip, objects were
        reaching 21.4 box widths -- the "stationary" object was following a
        moving car across the entire scene, which is how a moving vehicle ended
        up labelled "stopped" and how the wrong car was accused of a collision.
        """
        if self.anchor is None:
            return 0.0
        w = max(1e-6, float(self.box[2] - self.box[0]))
        return float(np.linalg.norm(self.centre - self.anchor) / w)

    @property
    def is_parked(self) -> bool:
        """Stationary since the first frame and never moved => parked.

        Stated as a hard property rather than a score contribution: no amount
        of other evidence should promote a vehicle we never once saw move.
        """
        return self.present_from_start and not self.ever_moved

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "box": [round(float(v), 1) for v in self.box],
            "cls": int(self.cls),
            "score": round(float(self.score), 3),
            "first_seen_t": round(self.first_seen_t, 2),
            "dwell_s": round(self.dwell, 2),
            "track_id": self.track_id,
            "persons_nearby": self.persons_nearby,
            "companions": self.companions,
            "debris_blobs": self.debris_blobs,
        }


class StationaryDetector:
    """Detects stopped vehicles on the background image and corroborates them."""

    def __init__(self,
                 detector,
                 interval_s: float = 2.5,
                 conf: float = 0.20,
                 min_dwell_s: float = 3.0,
                 match_iou: float = 0.35,
                 person_radius_factor: float = 1.6,
                 debris_min_area: int = 40,
                 debris_max_area: int = 2500,
                 min_hits: int = 8) -> None:
        self.detector = detector
        self.interval_s = interval_s
        self.conf = conf
        self.min_dwell_s = min_dwell_s
        self.match_iou = match_iou
        self.person_radius_factor = person_radius_factor
        self.debris_min_area = debris_min_area
        self.debris_max_area = debris_max_area
        self.min_hits = min_hits

        self.objects: list[StationaryObject] = []
        self._next_id = 1
        # Coarse spatial index over the working set, rebuilt each frame.
        self._grid: dict[tuple[int, int], list] = {}
        self._last_live_t: float | None = None
        # Upper bound on accumulated candidates. Persistence is the
        # signal, so when this is hit the least-confirmed are dropped.
        self.max_working_set = 400
        self._last_run_t = -1e9
        self.last_background_dets: np.ndarray = np.empty((0, 6))
        self._baseline_bg: np.ndarray | None = None
        self.road_mask = None
        self.rejected_off_road = 0
        self._first_pass_t: float | None = None
        self._bg_ring: deque = deque(maxlen=24)
        self.debris_lookback_s = 4.0
        self.person_min_hits = 8

    # ------------------------------------------------------------------
    def _rebuild_grid(self, cell: float) -> None:
        """Bucket the working set by coarse image cell for O(1) lookup."""
        grid: dict[tuple[int, int], list] = {}
        for o in self.objects:
            cx = (o.box[0] + o.box[2]) / 2.0
            cy = (o.box[1] + o.box[3]) / 2.0
            grid.setdefault((int(cx // cell), int(cy // cell)), []).append(o)
        self._grid = grid

    def observe_live(self, detections, t: float, road_mask=None,
                     min_conf: float = 0.08, jitter_px: float = 14.0,
                     max_drift_widths: float = 0.6,
                     max_speed_px_s: float = 45.0) -> None:
        """Accumulate weak-but-persistent live detections into stationary objects.

        This is the path that actually works, and finding out why is the most
        useful thing measured in this project.

        The background image looks ideal for the job -- a jackknifed tanker sat
        in it perfectly crisp, with all moving traffic dissolved away. But the
        detector found *nothing* there, while finding the same truck at 0.13
        confidence on the raw frame. A median over samples is a denoiser: it
        removes exactly the high-frequency texture a CNN keys on, which is fatal
        for an object that was already marginal.

        Worse, that 0.13 detection never became a track either, because
        ByteTrack only *initiates* tracks from detections above its high
        threshold (0.35). So a genuinely visible, genuinely stationary crashed
        vehicle was invisible to every downstream stage.

        The fix is to let **temporal consistency stand in for per-frame
        confidence**. A weak detection that keeps reappearing in the same place,
        frame after frame, is real -- noise does not hold still. This costs no
        extra inference: the detections are already computed.
        """
        if detections is None or len(detections) == 0:
            return
        if self._first_pass_t is None:
            self._first_pass_t = t

        for d in detections:
            if float(d[4]) < min_conf or int(d[5]) not in MOTORISED_CLASSES:
                continue
            box = np.asarray(d[:4], dtype=np.float64)
            if road_mask is not None and not road_mask.on_road(box):
                continue

            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

            # Candidates are looked up in a coarse spatial hash rather than by
            # scanning every accumulated object. Dense 4K Indian traffic yields
            # a couple of hundred detections a frame against a working set of
            # thousands, and the quadratic version of this loop stalled the
            # pipeline hard enough to look like a hang -- GPU idle at 0% while
            # Python compared boxes.
            # Matching tolerance is a SPEED, not a per-frame pixel budget.
            # A fixed 14 px between frames is 210 px/s at 15 fps -- a quarter of
            # a 960-wide frame every second, which is a moving car, not a
            # stationary one. Scaling by the frame interval makes the threshold
            # mean what it says.
            dt = max(1e-3, t - self._last_live_t) if self._last_live_t is not None else 1.0
            jitter_px = min(jitter_px, max(2.0, max_speed_px_s * dt))
            cell = max(16.0, jitter_px * 2.0)
            key = (int(cx // cell), int(cy // cell))
            matched = None
            for dk in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
                for o in self._grid.get((key[0] + dk[0], key[1] + dk[1]), ()):
                    if o.last_seen_t < 0:            # already dropped
                        continue
                    ox = (o.box[0] + o.box[2]) / 2.0
                    oy = (o.box[1] + o.box[3]) / 2.0
                    if np.hypot(ox - cx, oy - cy) <= jitter_px or iou(o.box, box) >= 0.4:
                        matched = o
                        break
                if matched is not None:
                    break

            if matched is None:
                o = StationaryObject(id=self._next_id, box=box, cls=int(d[5]),
                                     score=float(d[4]), first_seen_t=t, last_seen_t=t)
                o.anchor = np.array([cx, cy], dtype=float)
                o.present_from_start = (t - (self._first_pass_t or t)) < 1.0
                self.objects.append(o)
                self._next_id += 1
            else:
                # exponential smoothing keeps the box steady without letting a
                # slow-moving vehicle masquerade as a stationary one
                matched.box = 0.7 * matched.box + 0.3 * box
                matched.score = max(matched.score, float(d[4]))
                matched.last_seen_t = t
                matched.hits += 1

        self._last_live_t = t

        # anything not re-seen recently was never really stationary
        self.objects = [o for o in self.objects if (t - o.last_seen_t) < 3.0]

        # ...and anything that has wandered from where it was first seen was
        # never stationary either: it is a moving vehicle the matcher dragged
        # along. Dropping it here is what stops a moving car being labelled
        # "stopped" and being offered to the collision engine as a candidate.
        self.objects = [o for o in self.objects if o.drift <= max_drift_widths]

        # A stopped vehicle is re-seen every frame, so a large working set is
        # made of transient boxes from moving traffic, not of candidates. Keep
        # the most persistent ones: persistence is the whole signal here.
        if len(self.objects) > self.max_working_set:
            self.objects.sort(key=lambda o: (-o.hits, -o.last_seen_t))
            del self.objects[self.max_working_set:]

        self._rebuild_grid(max(16.0, jitter_px * 2.0))

    def maybe_update(self, background: np.ndarray | None, bg_scale: float,
                     tracks, t: float, frame_shape,
                     road_mask=None) -> list[StationaryObject]:
        """Run background detection if due; returns objects confirmed this call.

        ``road_mask`` is what makes this usable off a motorway. Without it, a
        car park in frame yields dozens of "stationary vehicles" that all
        corroborate each other. With it, only vehicles resting on the drivable
        surface are considered at all.
        """
        if background is None or (t - self._last_run_t) < self.interval_s:
            return []
        self._last_run_t = t
        self.road_mask = road_mask
        if self._first_pass_t is None:
            self._first_pass_t = t

        dets = self.detector.detect_array(background)
        if len(dets):
            # background is held at reduced scale; lift boxes back to frame coords
            dets = dets.copy()
            dets[:, :4] /= max(bg_scale, 1e-6)
        self.last_background_dets = dets

        veh = [d for d in dets if int(d[5]) in MOTORISED_CLASSES]
        if road_mask is not None:
            before = len(veh)
            # Only vehicles resting on the learned drivable surface can be
            # incidents. This one line is what stops a car park in frame from
            # producing two dozen mutually-corroborating "crashes".
            veh = [d for d in veh if road_mask.on_road(d[:4])]
            self.rejected_off_road = before - len(veh)
        vehicles = np.array(veh) if veh else np.empty((0, 6))

        persons = [d for d in dets if int(d[5]) in VULNERABLE_CLASSES]
        persons = np.array(persons) if persons else np.empty((0, 6))

        # Background detections are a *bonus* confirmation, not the primary
        # source -- see observe_live for why. Merge rather than replace.
        if len(vehicles):
            self._associate(vehicles, t)
        self._road_obstruction(road_mask, t)
        self._motion_history(tracks)
        self._attach_tracks(tracks)
        self._count_persons(persons, tracks)
        self._count_companions()
        self._mark_queues()
        self._debris(background, bg_scale, frame_shape, t)

        return [o for o in self.objects
                if o.dwell >= self.min_dwell_s
                and o.hits >= self.min_hits
                and not o.reported]

    # ------------------------------------------------------------------
    def _associate(self, vehicles: np.ndarray, t: float) -> None:
        """Match this pass's background detections to known stationary objects."""
        if len(vehicles) == 0:
            # nothing stationary any more: let existing objects age out
            self.objects = [o for o in self.objects if (t - o.last_seen_t) < 8.0]
            return

        if self.objects:
            existing = np.stack([o.box for o in self.objects])
            M = iou_matrix(existing, vehicles[:, :4])
        else:
            M = np.zeros((0, len(vehicles)))

        claimed = set()
        for i, o in enumerate(self.objects):
            if M.shape[0] == 0:
                break
            j = int(np.argmax(M[i])) if M.shape[1] else -1
            if j >= 0 and M[i, j] >= self.match_iou and j not in claimed:
                claimed.add(j)
                o.box = vehicles[j, :4].copy()
                o.score = float(vehicles[j, 4])
                o.cls = int(vehicles[j, 5])
                o.last_seen_t = t
                o.hits += 1

        for j in range(len(vehicles)):
            if j in claimed:
                continue
            self.objects.append(StationaryObject(
                id=self._next_id, box=vehicles[j, :4].copy(),
                cls=int(vehicles[j, 5]), score=float(vehicles[j, 4]),
                first_seen_t=t, last_seen_t=t,
            ))
            self._next_id += 1

        self.objects = [o for o in self.objects if (t - o.last_seen_t) < 8.0]

    def _motion_history(self, tracks, move_thresh: float = 12.0) -> None:
        """Did this thing ever drive, and did its silhouette change?

        Two of the most-requested discriminators, and the cheapest available.

        *Was it ever moving?* A parked car and a crashed car are both stationary
        vehicles on a road, and no amount of threshold tuning separates them
        from a single frame. But a parked car was never seen driving, whereas a
        crashed one arrived under its own power and stopped. Requiring evidence
        of prior motion removes the entire parked-car class -- which was, by
        observation, our largest false-positive source.

        *Did its shape change?* A vehicle that rolls, flips or is spun broadside
        changes silhouette violently, and even an axis-aligned box registers that
        as a large aspect-ratio swing.
        """
        for o in self.objects:
            for tr in tracks:
                if iou(o.box, tr.box) < 0.25:
                    continue
                o.track_seen = True
                pk = tr.peak_speed()
                o.peak_speed = max(o.peak_speed, pk)
                o.aspect_shift = max(o.aspect_shift, tr.aspect_shift(2.5))
                if pk >= 4.0:
                    o.ever_moved = True
                if pk >= move_thresh:
                    o.arrived_moving = True
                # how hard it stopped: a queue decelerates gently, an impact
                # does not. This is the discriminator geometry cannot supply,
                # because a rear-end crash and a queue have the same shape.
                o.stop_decel = max(o.stop_decel, -min(0.0, tr.acceleration_px(1.5)))
                break

    def _attach_tracks(self, tracks) -> None:
        """Bind each stationary object to a live track where one overlaps.

        This is what turns "there is a stopped car at these pixels" into "vehicle
        #7 is the one involved", which is what the annotated video needs in order
        to keep a box on the right car as the scene continues.
        """
        for o in self.objects:
            best, best_iou = None, 0.0
            for tr in tracks:
                v = iou(o.box, tr.box)
                if v > best_iou:
                    best, best_iou = tr, v
            if best is not None and best_iou >= 0.25:
                o.track_id = best.track_id

    def _count_persons(self, persons: np.ndarray, tracks) -> None:
        """Occupants who got OUT of the vehicle -- not pedestrians walking past.

        The naive version, "count people near the stopped vehicle", was
        measurably wrong: on a clean traffic clip it counted three pedestrians
        strolling past a parked car and scored it 0.688 as a probable crash.
        In any urban street, people near stopped vehicles is the normal state of
        the world, not evidence of anything.

        What is actually diagnostic is *where the person came from*. Someone who
        climbed out of a car has a track that BEGINS at the car. Someone walking
        past has a track that began somewhere else and moved through. So the
        test is on the person track's first observed position, not its current
        one -- which is cheap, because the tracker already stores it.
        """
        for o in self.objects:
            x1, y1, x2, y2 = o.box
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # emergence radius is tight: an occupant appears at the vehicle,
            # not a car-length away from it
            r_emerge = 0.9 * max(w, h)

            emerged = 0
            for tr in tracks:
                if tr.cls not in VULNERABLE_CLASSES:
                    continue
                if not tr.history:
                    continue
                # A fragmented pedestrian track restarts wherever the detector
                # reacquired it, which frequently happens to be beside a parked
                # car -- and that reads as a false "emergence". Requiring a
                # mature track discards those fragments: a real occupant is
                # tracked continuously as they get out and move around.
                if tr.hits < self.person_min_hits:
                    continue
                # must have APPEARED here, not walked in from elsewhere
                _, fx, fy = tr.history[0]
                if np.hypot(fx - cx, fy - cy) > r_emerge:
                    continue
                # must have appeared after the vehicle came to rest
                if tr.first_t < o.first_seen_t - 1.0:
                    continue
                # and must have stayed with the vehicle, not passed through
                px, py = tr.ground_point
                if np.hypot(px - cx, py - cy) > 2.0 * r_emerge:
                    continue
                emerged += 1

            o.persons_nearby = max(o.persons_nearby, emerged)

    def _road_obstruction(self, road_mask, t: float) -> None:
        """How much of a stopped vehicle sits on the active carriageway.

        This exists because of a clip the system could not possibly have caught
        as designed: a jackknifed tanker, stationary and skewed across the road
        from the very first frame to the last. There is no moving-to-stopped
        transition to detect, so every cue built around *the moment of impact*
        is unavailable -- and the social cues (occupants out, companion vehicle,
        debris) had not appeared within the clip either.

        What remains observable is simply that a large vehicle is sitting still
        on a surface that traffic is actively using. That is an obstruction
        whether or not we ever witnessed the crash that caused it, and it is
        arguably the more useful thing to report to a control room.
        """
        for o in self.objects:
            o.road_coverage = (road_mask.coverage(o.box)
                               if road_mask is not None else 0.0)
            if o.first_seen_t <= t - o.dwell and o.dwell > 0:
                pass
            o.present_from_start = o.first_seen_t <= self._first_pass_t + 1e-6

    def _mark_queues(self, min_chain: int = 3, near: float = 2.2,
                     max_considered: int = 60) -> None:
        """Flag stationary vehicles that form a chain -- a queue, not a pile-up.

        A collision involves two vehicles, occasionally three. When four or
        five stationary vehicles sit in a connected chain of adjacent
        footprints, that is traffic waiting, and treating it as a multi-
        vehicle collision is how a red light becomes an incident.
        """
        from .footprint import Footprint, separation
        # Only persistent objects can form a queue, and the pass is quadratic,
        # so it runs over the most-confirmed few rather than the whole set.
        objs = sorted(self.objects, key=lambda o: -o.hits)[:max_considered]
        if len(objs) < min_chain:
            for o in objs:
                o.queue_member = False
            return
        fps = [Footprint.from_box(o.box) for o in objs]
        adj = {i: set() for i in range(len(objs))}
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if separation(fps[i], fps[j]) <= near:
                    adj[i].add(j)
                    adj[j].add(i)
        seen = set()
        for i in range(len(objs)):
            if i in seen:
                continue
            stack, comp = [i], []
            while stack:
                k = stack.pop()
                if k in seen:
                    continue
                seen.add(k)
                comp.append(k)
                stack.extend(adj[k] - seen)
            is_queue = len(comp) >= min_chain
            for k in comp:
                objs[k].queue_member = is_queue

    def _count_companions(self, max_dwell_gap: float = 6.0) -> None:
        """Another vehicle that stopped *at about the same time*, right alongside.

        The naive version -- count nearby stopped vehicles -- is actively
        harmful: in a car park every vehicle corroborates every other and the
        whole lot scores as a pile-up. Two extra conditions fix it. The
        neighbour must have come to rest at a similar moment (a collision stops
        both parties together, a car park fills over hours), and the count is
        capped at 2 because a genuine collision has few participants.
        """
        for o in self.objects:
            x1, y1, x2, y2 = o.box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            r = 1.8 * max(x2 - x1, y2 - y1)
            n = 0
            for other in self.objects:
                if other.id == o.id:
                    continue
                if abs(other.first_seen_t - o.first_seen_t) > max_dwell_gap:
                    continue
                ox = (other.box[0] + other.box[2]) / 2.0
                oy = (other.box[1] + other.box[3]) / 2.0
                if np.hypot(ox - cx, oy - cy) <= r:
                    n += 1
            o.companions = min(n, 2)

    def _debris(self, background: np.ndarray, bg_scale: float, frame_shape,
                t: float = 0.0) -> None:
        """New small static blobs around a stopped vehicle.

        Impact scatters fragments, and unlike the vehicles they never move
        again, so they enter the background and stay. Differencing the current
        background against the first one recorded isolates *what has appeared*,
        and small connected components near a stopped vehicle are debris
        candidates.

        Deliberately reported as a corroborating count, never as a standalone
        detection: shadows, litter and wet patches produce blobs too.
        """
        # Compare against a background from a few seconds ago, not from the
        # first frame. Debris is defined by having APPEARED -- differencing
        # against t=0 also lights up every parked car that arrived, every shadow
        # that moved and every exposure drift since the clip began.
        self._bg_ring.append((t, background.copy()))
        recent = [(bt, b) for bt, b in self._bg_ring if t - bt >= self.debris_lookback_s]
        if not recent:
            return
        self._baseline_bg = recent[-1][1]
        if self._baseline_bg.shape != background.shape:
            return

        a = cv2.cvtColor(self._baseline_bg, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(a, b)
        _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if self.debris_min_area <= area <= self.debris_max_area:
                blobs.append((cents[i][0] / max(bg_scale, 1e-6),
                              cents[i][1] / max(bg_scale, 1e-6)))

        for o in self.objects:
            x1, y1, x2, y2 = o.box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            r = 2.0 * max(x2 - x1, y2 - y1)
            o.debris_blobs = sum(1 for bx, by in blobs
                                 if np.hypot(bx - cx, by - cy) <= r)

    # ------------------------------------------------------------------
    @staticmethod
    def crash_score(o: StationaryObject) -> tuple[float, dict]:
        """How much this stationary object looks like a crash rather than a park.

        Every term is something a camera can actually observe, and each is
        reported alongside the score so an operator can disagree with a specific
        one rather than with the number.
        """
        s_person = min(1.0, o.persons_nearby / 2.0)
        s_comp = min(1.0, o.companions / 2.0)
        s_debris = min(1.0, o.debris_blobs / 3.0)
        # dwell saturates fast and is capped low: sitting still for a long time
        # is what parked cars do, so it must never carry a finding on its own
        s_dwell = min(1.0, o.dwell / 15.0)
        # obstruction: parked cars sit beside the road, obstructions sit on it
        s_road = min(1.0, o.road_coverage / 0.55)

        # The companion term has now been a false-positive source three times:
        # a car park (parked cars corroborate each other), queued traffic at a
        # junction (queued cars corroborate each other), and adjacent stopped
        # vehicles generally. Proximity between stopped vehicles is the NORMAL
        # state of traffic, not evidence of a collision. It is kept only as a
        # weak tie-breaker.
        #
        # The load is shifted onto the two cues that genuinely distinguish a
        # crash from a stop: people out of their vehicles, and debris.
        s_flip = min(1.0, o.aspect_shift / 0.55)

        # Rebalanced around measured reliability rather than intuition.
        #
        # The appearance classifier scores AUC 0.954 on held-out data. Every
        # hand-built cue here was, individually, close to useless: on a clean
        # clip versus a real crash, companions and nearby pedestrians both
        # fired on the clean one. Debris and rollover are genuinely
        # informative but sparse -- absent from most real crashes at CCTV
        # resolution. So these cues are now a modest prior that the
        # classifier's verdict dominates downstream, and the two that proved
        # actively misleading are nearly zeroed.
        score = (0.30 * s_debris        # real signal, sparse
                 + 0.28 * s_flip        # rollover / spin, real but sparse
                 + 0.24 * s_person      # occupants OUT, after the emergence fix
                 + 0.14 * s_road        # on the carriageway, not beside it
                 + 0.03 * s_dwell
                 + 0.01 * s_comp)       # near-zero: fired on car parks and queues

        # ---- absolute gates -------------------------------------------
        # Parked: stationary since the first frame, never observed moving.
        # Zero, not reduced -- we did not witness a crash and almost certainly
        # there was not one.
        if o.is_parked:
            return 0.0, {"gate": "parked: stationary from first frame, never moved",
                         "present_from_start": True, "ever_moved": False}

        # Never seen driving -- but only counted when a track actually
        # watched it and saw no movement. With no track associated at all,
        # nothing is known either way, and scoring the unknown as innocent
        # suppressed a real incident by a factor of six.
        if o.track_seen and not o.arrived_moving:
            score *= 0.55

        # Member of a stationary chain: that is a queue. A collision does not
        # come in fives.
        if o.queue_member:
            score *= 0.20

        # Stopped gently rather than abruptly. Queues decelerate smoothly;
        # impacts do not. This is the only temporal signal that separates a
        # rear-end collision from the queue it geometrically resembles.
        if 0.0 < o.stop_decel < 25.0:
            score *= 0.45
        return float(np.clip(score, 0.0, 1.0)), {
            "road_coverage": round(o.road_coverage, 3),
            "road_term": round(s_road, 3),
            "arrived_moving": o.arrived_moving,
            "ever_moved": o.ever_moved,
            "track_seen": o.track_seen,
            "parked": o.is_parked,
            "queue_member": o.queue_member,
            "stop_decel_px_s2": round(o.stop_decel, 1),
            "peak_speed_px_s": round(o.peak_speed, 1),
            "aspect_shift": round(o.aspect_shift, 3),
            "rollover_term": round(s_flip, 3),
            "present_from_start": o.present_from_start,
            "persons_nearby": o.persons_nearby,
            "companion_stopped_vehicles": o.companions,
            "debris_blobs": o.debris_blobs,
            "dwell_s": round(o.dwell, 1),
            "person_term": round(s_person, 3),
            "companion_term": round(s_comp, 3),
            "debris_term": round(s_debris, 3),
            "dwell_term": round(s_dwell, 3),
        }
