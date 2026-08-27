"""Deciding *which* vehicles were in the collision.

Detecting that a collision happened and deciding who was in it are different
problems, and reviewing fifteen annotated clips made clear that we had been
treating them as one. Seven of fifteen clips detected the event correctly and
then drew boxes around the wrong set of vehicles -- typically the right pair
plus two bystanders. Over-tagging is not a cosmetic flaw: a box that says
COLLISION on an uninvolved car is a false accusation, and it teaches an operator
to distrust every other box on the screen.

Three domain facts drive the selection rule.

**Collisions are pairwise.** Roughly nine times in ten a collision involves two
or more vehicles, and overwhelmingly it is exactly two. A ranked list of
"suspicious vehicles" ignores this; what we should be looking for is the best
*pair*. Single-vehicle events are real but rare, so they are allowed only on
much stronger evidence -- a rollover, or a vehicle come to rest off the
carriageway.

**Participants touch.** The two vehicles in a collision end up within about a
vehicle-length of each other, usually in contact. A bystander that merely
stopped nearby does not.

**Participants stop together; bystanders stop afterwards.** This is the
observation that explains the worst over-tagging case, where the two cars that
queued *behind* a crash were tagged as participants. They did stop, on the road,
near the incident -- but they stopped later, and further back. Requiring
temporal coincidence separates cause from consequence.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .footprint import Footprint, contact_score, separation
from .geometry import iou


@dataclass
class Candidate:
    """A vehicle that might have been in the collision."""

    track_id: int | None
    box: np.ndarray
    score: float                     # crash evidence, 0..1
    stop_t: float | None = None      # when it came to rest
    rollover: float = 0.0            # aspect-ratio swing
    off_road: bool = False
    arrived_moving: bool = True
    parked: bool = False             # never observed moving, present from frame 0
    queue_member: bool = False       # part of a stationary chain
    stop_decel: float = 0.0          # how hard it stopped, px/s^2
    detail: dict = field(default_factory=dict)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def diag(self) -> float:
        return float(np.hypot(self.box[2] - self.box[0], self.box[3] - self.box[1]))

    @property
    def has_motion_history(self) -> bool:
        """Did we actually watch this vehicle move, stop, or deform?

        A collision is an event, and an event has to be witnessed. A box that
        merely appeared in the background image carries no such witness: we
        never saw it drive in, never saw it stop, never saw its shape change.
        Convicting it means trusting appearance alone, and appearance has been
        measured scoring 0.995 on parked vehicles in a scene that contains a
        crash somewhere else.
        """
        if self.parked:
            return False
        if self.rollover >= 0.25:      # we saw it deform or spin
            return True
        if self.stop_decel >= 25.0:    # we saw it stop, and stop hard
            return True
        # otherwise only a real track counts as having watched it
        return self.track_id is not None and self.arrived_moving


def _gap_lengths(a: Candidate, b: Candidate, depth_ratio: float = 0.35) -> float:
    """Separation between two vehicles' ROAD FOOTPRINTS.

    Not between their bounding boxes. A box spans a vehicle's height, and a
    collision happens on the road surface -- under perspective a background
    bus and a foreground car can have near-touching boxes while standing
    metres apart on the ground. Comparing the patches of road the vehicles
    actually stand on removes that entire class of false contact.

    Returns normalised footprint separation: below 1.0 the vehicles are
    sharing road surface.
    """
    fa = Footprint.from_box(a.box, depth_ratio)
    fb = Footprint.from_box(b.box, depth_ratio)
    return separation(fa, fb)


class ParticipantSelector:
    """Chooses the vehicles to mark, and refuses to guess when unsure."""

    def __init__(self,
                 max_gap_lengths: float = 1.25,
                 coincidence_s: float = 1.6,
                 pair_min_score: float = 0.28,
                 single_min_score: float = 0.40,
                 third_gap_lengths: float = 1.1,
                 third_min_score: float = 0.55,
                 min_p_crashed: float = 0.35,
                 depth_ratio: float = 0.35,
                 max_participants: int = 3) -> None:
        self.max_gap_lengths = max_gap_lengths
        self.coincidence_s = coincidence_s
        self.pair_min_score = pair_min_score
        self.single_min_score = single_min_score
        self.third_gap_lengths = third_gap_lengths
        self.third_min_score = third_min_score
        self.min_p_crashed = min_p_crashed
        self.depth_ratio = depth_ratio
        self.max_participants = max_participants

    # ------------------------------------------------------------------
    def select(self, candidates: list[Candidate]) -> tuple[list[Candidate], dict]:
        """Return ``(track_ids, explanation)``.

        An empty list is a legitimate answer. Marking nothing is strictly better
        than marking the wrong vehicle, because the incident is still raised and
        the operator still gets the clip -- they simply are not told a falsehood
        about who was in it.
        """
        # A candidate is a BOX; a track id is optional metadata. The vehicles
        # that matter most are often ones the tracker never locked onto,
        # because a crashed vehicle is low-confidence, oddly shaped and often
        # occluded -- precisely what fails to start a track.
        cands = [c for c in candidates if c.box is not None]
        if not cands:
            return [], {"mode": "none", "reason": "no candidate vehicles"}

        cands.sort(key=lambda c: -c.score)

        pair, pair_detail = self._best_pair(cands)
        if pair:
            chosen = list(pair)
            third = self._maybe_third(cands, pair)
            if third is not None:
                chosen.append(third)
                pair_detail["third_added"] = third.track_id
            return chosen[: self.max_participants], pair_detail

        conf = self._confident_single(cands)
        if conf is not None:
            return [conf], {
                "mode": "single-vehicle (classifier-confident)",
                "reason": ("no partner looked damaged; marking the one vehicle the classifier is confident about rather than inventing a second"),
                "p_crashed": conf.detail.get("p_crashed"),
            }

        single = self._best_single(cands)
        if single is not None:
            return [single], {
                "mode": "single-vehicle",
                "reason": ("no valid pair; accepted on strong single-vehicle evidence "
                           "(rollover or off-carriageway rest)"),
                "score": round(single.score, 3),
                "rollover": round(single.rollover, 3),
                "off_road": single.off_road,
            }

        return [], {
            "mode": "unattributed",
            "reason": ("no vehicle pair satisfied contact and stop-time coincidence, "
                       "and no single vehicle met the higher solo bar; marking none "
                       "rather than risk accusing an uninvolved vehicle"),
            "candidates_considered": len(cands),
            "best_score": round(cands[0].score, 3),
        }

    # ------------------------------------------------------------------
    def _looks_crashed(self, c: Candidate, touching: bool = False) -> bool:
        """Would a human call this vehicle damaged?

        The pairing prior says collisions involve two vehicles, but applied
        naively it *manufactures* a second participant: on one clip the
        partner scored 0.196 from the classifier -- visibly intact -- and was
        boxed anyway purely to complete a pair. A prior should shape a
        decision, not override direct evidence against it.

        So a candidate the classifier judges intact is admitted only when the
        boxes physically overlap, where contact is itself strong evidence and
        appearance is often occluded by the other vehicle.
        """
        p = c.detail.get("p_crashed")
        if p is None:
            return True          # no classifier available: fall back to geometry
        return bool(p >= self.min_p_crashed or touching)

    def _best_pair(self, cands: list[Candidate]):
        """Highest-scoring pair that is in contact and stopped together."""
        best, best_val, best_detail = None, -1.0, {}
        for a, b in itertools.combinations(cands[:8], 2):
            if a.score < self.pair_min_score or b.score < self.pair_min_score:
                continue

            gap = _gap_lengths(a, b, self.depth_ratio)
            # sharing road surface, not merely overlapping in image space
            touching = gap <= 1.0
            if gap > self.max_gap_lengths and not touching:
                continue
            if not (self._looks_crashed(a, touching) and self._looks_crashed(b, touching)):
                continue

            # temporal coincidence -- the guard against secondary stoppers
            dt = None
            if a.stop_t is not None and b.stop_t is not None:
                dt = abs(a.stop_t - b.stop_t)
                if dt > self.coincidence_s:
                    continue

            # prefer pairs that are close, in contact, and stopped together
            val = (a.score + b.score) + (0.35 if touching else 0.0) \
                + 0.30 * max(0.0, 1.0 - gap / max(self.max_gap_lengths, 1e-6))
            if dt is not None:
                val += 0.25 * max(0.0, 1.0 - dt / max(self.coincidence_s, 1e-6))

            if val > best_val:
                best, best_val = (a, b), val
                best_detail = {
                    "mode": "pair",
                    "reason": "two vehicles in contact that came to rest together",
                    "footprint_separation": round(gap, 2),
                    "footprints_overlap": bool(gap <= 1.0),
                    "boxes_touching": bool(touching),
                    "stop_time_gap_s": None if dt is None else round(dt, 2),
                    "scores": [round(a.score, 3), round(b.score, 3)],
                }
        return best, best_detail

    def _maybe_third(self, cands: list[Candidate], pair) -> Candidate | None:
        """Admit a third vehicle only for a genuine pile-up.

        It must be in contact with one of the pair and carry strong evidence of
        its own. Without both conditions this is how a queue behind the crash
        gets swept in.
        """
        a, b = pair
        chosen_ids = {id(a), id(b)}
        for c in cands:
            if id(c) in chosen_ids or c.score < self.third_min_score:
                continue
            if min(_gap_lengths(c, a, self.depth_ratio),
                   _gap_lengths(c, b, self.depth_ratio)) > self.third_gap_lengths:
                continue
            if c.stop_t is not None:
                ref = [t for t in (a.stop_t, b.stop_t) if t is not None]
                if ref and abs(c.stop_t - min(ref)) > self.coincidence_s:
                    continue
            return c
        return None

    def _confident_single(self, cands: list[Candidate], gate: float = 0.88):
        """A vehicle the classifier is sure about AND whose motion we watched.

        The appearance requirement alone used to be enough, and it picked the
        wrong vehicle in review: in clip 1 it took an untracked bystander at
        0.995 over the vehicle that had actually rolled and stopped hard at
        0.695. Requiring motion history makes appearance a confirming vote on
        a vehicle we have independent reason to suspect, rather than the sole
        basis for accusing one.
        """
        for c in cands:
            p = c.detail.get("p_crashed")
            if p is not None and p >= gate and c.has_motion_history:
                return c
        return None

    def _best_single(self, cands: list[Candidate],
                     rollover_gate: float = 0.28) -> Candidate | None:
        """A lone vehicle convicts on evidence a stopped vehicle cannot fake.

        Single-vehicle collisions are the minority, so stopping is never
        enough -- parked cars, queues and red lights all produce it. What is
        enough is a shape change: an aspect ratio that swings by a third means
        the tracker watched the vehicle spin, roll or crumple, and no ordinary
        traffic behaviour does that. Coming to rest off the carriageway is the
        same kind of evidence by position rather than by shape.

        Both routes require motion history by construction: a candidate we
        never tracked has no measurable shape change and is capped below the
        score bar, so this cannot readmit the bystanders the vetoes removed.
        """
        for c in cands:
            if c.score < self.single_min_score:
                continue
            if not c.has_motion_history:
                continue
            if c.rollover >= rollover_gate or c.off_road:
                return c
        return None
