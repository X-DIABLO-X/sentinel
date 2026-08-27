"""Regression tests for the veto rules that keep parked and queued cars unaccused.

The single most persistent failure in this project was ordinary stationary
traffic being reported as collisions. Each rule below was added in response to a
specific reviewed clip, and each is tested here as a rule rather than through the
detector, so a future change that quietly disables one fails immediately instead
of at the next full run.
"""

from __future__ import annotations

import numpy as np
import pytest

from netra.attribution import Candidate, ParticipantSelector
from netra.stationary import StationaryDetector, StationaryObject


def make_object(**kw) -> StationaryObject:
    """A stationary vehicle on the carriageway that drove in and stopped hard."""
    base = dict(
        id=1, box=np.array([100.0, 100.0, 220.0, 190.0]), cls=2, score=0.8,
        first_seen_t=4.0, last_seen_t=12.0, hits=20, track_id=7,
        persons_nearby=1, companions=1, debris_blobs=2, road_coverage=1.0,
        present_from_start=False, arrived_moving=True, ever_moved=True,
        track_seen=True, stop_decel=60.0, queue_member=False,
        peak_speed=240.0, aspect_shift=0.6,
    )
    base.update(kw)
    return StationaryObject(**base)


# --------------------------------------------------------------------------
# The reviewer's rule: never moved, from the first frame, means parked
# --------------------------------------------------------------------------

def test_parked_vehicle_is_vetoed_outright():
    """Present from frame zero and never observed moving is parked, not crashed.

    Enforced as a hard zero rather than a penalty: no combination of debris,
    bystanders or appearance should convict a vehicle we never saw move, because
    stationary vehicles beside roads are the most common object in traffic
    footage.
    """
    parked = make_object(present_from_start=True, ever_moved=False,
                         arrived_moving=False)
    assert parked.is_parked is True
    score, detail = StationaryDetector.crash_score(parked)
    assert score == 0.0
    assert "parked" in detail.get("gate", "").lower()


def test_a_vehicle_that_moved_is_not_parked():
    o = make_object(present_from_start=True, ever_moved=True)
    assert o.is_parked is False
    assert StationaryDetector.crash_score(o)[0] > 0.0


def test_queue_membership_demotes_but_does_not_veto():
    """A chain of stationary vehicles is a queue; a collision does not come in fives.

    Demoted rather than vetoed, because the back of a queue is exactly where
    rear-end collisions happen, so this must stay overturnable.
    """
    alone = StationaryDetector.crash_score(make_object())[0]
    queued = StationaryDetector.crash_score(make_object(queue_member=True))[0]
    assert 0.0 < queued < alone


# --------------------------------------------------------------------------
# Unmeasured is not innocent
# --------------------------------------------------------------------------

def test_unmeasured_deceleration_is_not_treated_as_a_gentle_stop():
    """stop_decel == 0 means "never measured", not "stopped smoothly".

    Reading it as a gentle stop cost a factor of 0.45 on clips where the
    deceleration was simply never observed.
    """
    unmeasured = StationaryDetector.crash_score(make_object(stop_decel=0.0))[0]
    gentle = StationaryDetector.crash_score(make_object(stop_decel=5.0))[0]
    abrupt = StationaryDetector.crash_score(make_object(stop_decel=60.0))[0]
    assert gentle < abrupt
    assert unmeasured == pytest.approx(abrupt), "no measurement, no penalty"


def test_no_track_association_is_not_treated_as_stillness():
    """arrived_moving == False only counts when a track actually watched it.

    With no track associated, nothing is known either way. Scoring the unknown
    as innocent suppressed a real incident by a factor of six.
    """
    never_watched = StationaryDetector.crash_score(
        make_object(track_seen=False, arrived_moving=False))[0]
    watched_still = StationaryDetector.crash_score(
        make_object(track_seen=True, arrived_moving=False))[0]
    assert watched_still < never_watched


# --------------------------------------------------------------------------
# Attribution: a vehicle whose motion was never observed cannot be accused
# --------------------------------------------------------------------------

def candidate(**kw) -> Candidate:
    base = dict(track_id=3, box=np.array([100.0, 100.0, 200.0, 180.0]),
                score=0.7, stop_t=5.0, rollover=0.5, off_road=False,
                arrived_moving=True, parked=False, queue_member=False,
                stop_decel=50.0)
    base.update(kw)
    return Candidate(**base)


def test_motion_history_requires_something_actually_observed():
    assert candidate().has_motion_history is True
    assert candidate(parked=True).has_motion_history is False
    # untracked, no deformation, no measured stop: nothing was witnessed
    assert candidate(track_id=None, rollover=0.0,
                     stop_decel=0.0).has_motion_history is False
    # untracked but visibly deformed: that IS an observation
    assert candidate(track_id=None, rollover=0.6,
                     stop_decel=0.0).has_motion_history is True


def test_witnessless_candidate_is_never_accused_alone():
    """The clip-1 failure: an untracked bystander boxed over the real vehicle.

    The appearance model rated the bystander 0.995 and the vehicle that had
    actually rolled 0.695. Selection followed the score. It must not.
    """
    ghost = candidate(track_id=None, rollover=0.0, stop_decel=0.0, score=0.34)
    ghost.detail["p_crashed"] = 0.995
    chosen, why = ParticipantSelector().select([ghost])
    assert chosen == [], why


def test_deformed_tracked_vehicle_is_accepted_alone():
    """Aspect swing has no benign generator; stopping does."""
    rolled = candidate(rollover=0.5, score=0.55)
    chosen, why = ParticipantSelector().select([rolled])
    assert len(chosen) == 1
    assert chosen[0] is rolled


def test_selector_returns_nothing_rather_than_guessing():
    """Marking nothing beats marking the wrong vehicle.

    The incident is still raised and the operator still gets the clip; they are
    simply not told a falsehood about who was in it.
    """
    weak = candidate(score=0.10, rollover=0.0, stop_decel=0.0, track_id=None)
    chosen, why = ParticipantSelector().select([weak])
    assert chosen == []
    assert why["mode"] == "unattributed"


def test_selector_handles_an_empty_candidate_set():
    chosen, why = ParticipantSelector().select([])
    assert chosen == []
    assert why["mode"] == "none"
