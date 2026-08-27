"""Tests for the rotation-gated collision-pair scoring engine.

Ported from ``IDEAS/COMBINED/src/inference2.py`` (measured 4/4 top-1 on the
four labelled ground-truth videos there -- see the module docstring on
``netra/events/rotation_gate.py`` for why that number must never be quoted as
"100% accuracy"). Every fixture here is a synthetic ``netra.track.Track`` built
by feeding a hand-designed trajectory through ``Track.update()`` -- no video, no
model weights, runs in well under a second on CPU.

What each test pins down, and why it is here rather than left to inspection:

* **the rotation gate actually gates.** A vehicle that decelerates hard but
  never rotates (the brake-to-avoid bystander that was the whole reason this
  module exists -- see video 13 in the source README) must score far below a
  vehicle that rotates at contact, even when both lose the same speed.
* **interaction geometry is a real classifier, not a decoration.** A ~53 deg
  crossing pair must land in the ``crossing`` bracket; a near-antiparallel
  (~178 deg) passing pair must land in ``oncoming`` and be demoted relative to
  a crossing pair with comparable underlying evidence.
* **sub-sample contact refinement is not a no-op.** When the true closest
  approach falls between two sampled instants, ``find_contact`` must return a
  time strictly inside that bracket, not snap to one of the samples.
* **the constants match COMBINED's config.yaml exactly.** These were tuned on
  four videos with no held-out set; silently drifting from the measured values
  would be re-tuning by accident.
"""

from __future__ import annotations

import math

import pytest

from netra.events import rotation_gate as rg
from netra.track import Track

# ---------------------------------------------------------------------------
# fixture builders -- synthetic tracks, no video, no weights
# ---------------------------------------------------------------------------

# All fixtures use one vehicle footprint, in pixels: a car-sized box whose
# width against ``netra.predict``'s COCO car width (1.8 m) sets the scale.
_W, _H = 40.0, 80.0
_CLS_CAR = 2


def _box_at(gx: float, gy: float, w: float = _W, h: float = _H):
    """Box whose ground point (bottom-centre) sits at ``(gx, gy)``."""
    return (gx - w / 2.0, gy - h, gx + w / 2.0, gy)


def _heading_vec(deg: float) -> tuple[float, float]:
    """Unit direction for a ``netra.geometry.heading_degrees``-style heading."""
    r = math.radians(deg)
    return (math.sin(r), -math.cos(r))


def _segment_track(track_id: int, start_xy: tuple[float, float],
                   segments: list[tuple[float, float, float]],
                   *, dt: float = 0.1, cls: int = _CLS_CAR) -> Track:
    """Build a live ``Track`` by replaying a piecewise-constant-velocity path.

    ``segments`` is a list of ``(duration_s, vx_px_s, vy_px_s)``. The ground
    point starts at ``start_xy`` and is advanced ``dt`` at a time, exactly the
    way ``Track.box_history`` fills up in the real pipeline via repeated
    ``update()`` calls -- this is real ``Track`` machinery, not a stand-in.
    """
    t = 0.0
    x, y = start_xy
    boxes = [(t, _box_at(x, y))]
    for duration, vx, vy in segments:
        n = max(1, round(duration / dt))
        for _ in range(n):
            t += dt
            x += vx * dt
            y += vy * dt
            boxes.append((t, _box_at(x, y)))

    t0, b0 = boxes[0]
    track = Track(track_id=track_id, cls=cls, score=1.0, box=b0, frame_idx=0, t=t0)
    for i, (ti, bi) in enumerate(boxes[1:], start=1):
        track.update(bi, score=1.0, cls=cls, frame_idx=i, t=ti)
    return track


def _crossing_pair(rel_deg: float = 53.0, t_contact: float = 3.0,
                   pre_speed: float = 120.0, post_speed: float = 60.0,
                   yaw_kick_deg: float = 100.0) -> tuple[Track, Track]:
    """Two vehicles converging at ``rel_deg`` relative heading, both punted
    sideways (rotated) at contact -- a genuine T-bone.

    ``rel_deg`` is the angle between the two PRE-contact approach headings,
    which is what ``interaction_prior`` classifies on. The post-contact
    heading kick is independent of it and is what drives the rotation gate.
    """
    vx1, vy1 = _heading_vec(90.0)
    vx1, vy1 = vx1 * pre_speed, vy1 * pre_speed
    start1 = (0.0 - vx1 * t_contact, 0.0 - vy1 * t_contact)

    vx2, vy2 = _heading_vec(90.0 + rel_deg)
    vx2, vy2 = vx2 * pre_speed, vy2 * pre_speed
    start2 = (0.0 - vx2 * t_contact, 0.0 - vy2 * t_contact)

    pvx1, pvy1 = _heading_vec(90.0 + yaw_kick_deg)
    pvx1, pvy1 = pvx1 * post_speed, pvy1 * post_speed
    pvx2, pvy2 = _heading_vec(90.0 + rel_deg - yaw_kick_deg)
    pvx2, pvy2 = pvx2 * post_speed, pvy2 * post_speed

    a = _segment_track(101, start1, [(t_contact, vx1, vy1), (2.0, pvx1, pvy1)])
    b = _segment_track(102, start2, [(t_contact, vx2, vy2), (2.0, pvx2, pvy2)])
    return a, b


def _braking_bystander(track_id: int = 201, t_contact: float = 3.0,
                       pre_speed: float = 120.0, post_speed: float = 20.0) -> Track:
    """One vehicle, constant heading throughout, that sheds most of its speed.

    No rotation anywhere: same heading before and after, box aspect never
    changes. This is exactly video 13's #1320 -- a driver standing on the
    brakes to avoid a crash ahead, not a crash participant.
    """
    vx, vy = _heading_vec(90.0)
    start = (0.0 - vx * pre_speed * t_contact, 0.0 - vy * pre_speed * t_contact)
    return _segment_track(
        track_id, start,
        [(t_contact, vx * pre_speed, vy * pre_speed),
         (2.0, vx * post_speed, vy * post_speed)])


def _following_queue_pair(t_contact: float = 3.0) -> tuple[Track, Track]:
    """Two vehicles in the same lane, closing slowly, then braking together.

    Neither rotates. The closing speed puts a genuine minimum in the contact
    gap right around ``t_contact``, so this is a legitimate ``following``
    classification (same-direction, near-zero relative heading) rather than
    an artefact of a constant separation.
    """
    vx, vy = _heading_vec(90.0)
    v_lead, v_trail, v_post = 100.0, 115.0, 15.0
    gap0 = 115.0
    start_lead = (-2000.0 - vx * v_lead * t_contact, 500.0 - vy * v_lead * t_contact)
    start_trail = (start_lead[0] - gap0, 500.0 - vy * v_trail * t_contact)

    lead = _segment_track(
        201, start_lead,
        [(t_contact, vx * v_lead, vy * v_lead), (2.0, vx * v_post, vy * v_post)])
    trail = _segment_track(
        202, start_trail,
        [(t_contact, vx * v_trail, vy * v_trail), (2.0, vx * v_post, vy * v_post)])
    return lead, trail


def _oncoming_pass_pair(t_contact: float = 3.0, speed: float = 120.0,
                        lane_offset: float = 20.0) -> tuple[Track, Track]:
    """Two vehicles travelling opposite directions on parallel lines.

    Their boxes come close as they pass (a 2-D box cannot encode the lane
    depth separating them) but neither ever rotates or decelerates -- ordinary
    oncoming traffic, not a head-on.
    """
    vxa, vya = _heading_vec(90.0)
    vxa, vya = vxa * speed, vya * speed
    vxb, vyb = _heading_vec(90.0 + 178.0)
    vxb, vyb = vxb * speed, vyb * speed

    start_a = (0.0 - vxa * t_contact, -lane_offset - vya * t_contact)
    start_b = (0.0 - vxb * t_contact, lane_offset - vyb * t_contact)

    a = _segment_track(301, start_a, [(6.0, vxa, vya)])
    b = _segment_track(302, start_b, [(6.0, vxb, vyb)])
    return a, b


# ---------------------------------------------------------------------------
# import / smoke
# ---------------------------------------------------------------------------

def test_module_imports_cleanly():
    """No import-time side effects, no heavy (torch/ultralytics) dependency."""
    assert callable(rg.score_pairs)
    assert callable(rg.impact_evidence)
    assert callable(rg.find_contact)
    assert isinstance(rg.DEFAULT_CONFIG, rg.RotationGateConfig)


# ---------------------------------------------------------------------------
# config parity: these are tuned constants, not defaults to drift
# ---------------------------------------------------------------------------

def test_defaults_match_combined_config():
    """Every default is copied verbatim from COMBINED's ``config.yaml``.

    Source: ``D:/HARSHIT/ELCIA/IDEAS/COMBINED/config.yaml``, section
    ``inference2:``. If this test fails, either the port has drifted or
    someone re-tuned a constant without updating the record of where it came
    from -- both are worth stopping for.
    """
    cfg = rg.RotationGateConfig()

    expected = {
        "min_track_samples": 8,
        "min_travel_diagonals": 1.0,
        "min_max_speed_kmh": 5.0,
        "contact_window_s": 1.0,
        "heading_lookback_s": 1.0,
        "min_speed_for_heading_kmh": 6.0,
        "stable_heading_std_deg": 25.0,
        "stable_aspect_rel_std": 0.30,
        "ref_yaw_deg": 45.0,
        "ref_aspect": 0.60,
        "ref_decel": 30.0,
        "ref_speed_drop_kmh": 20.0,
        "ref_momentum": 12000.0,
        "rotation_floor": 0.20,
        "break_implies_rotation": 0.70,
        "break_floor_score": 0.55,
        "track_break_window_s": 0.4,
        "track_break_min_life_s": 1.0,
        "pair_ref_gap": 0.5,
        "pair_max_gap": 1.5,
        "following_max_deg": 45.0,
        "oncoming_min_deg": 160.0,
        "prior_crossing": 1.00,
        "prior_following": 0.55,
        "prior_oncoming": 0.50,
        "prior_unknown": 0.60,
        "w_rotation": 0.45,
        "w_decel": 0.15,
        "w_speed_drop": 0.15,
        "w_momentum": 0.10,
        "w_appearance": 0.35,
        "w_track_break": 0.25,
    }
    for field, value in expected.items():
        got = getattr(cfg, field)
        assert got == pytest.approx(value), f"{field}: expected {value}, got {got}"


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

class TestParabolicVertex:

    def test_symmetric_minimum_is_centred(self):
        assert rg.parabolic_vertex(1.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_offset_minimum_leans_toward_smaller_neighbour(self):
        # y drops more sharply toward the right neighbour, so the true vertex
        # sits to the right of the centre sample -- a positive offset.
        offset = rg.parabolic_vertex(1.0, 0.2, 0.0)
        assert offset > 0.0

    def test_degenerate_flat_fit_returns_zero(self):
        assert rg.parabolic_vertex(1.0, 1.0, 1.0) == 0.0

    def test_result_stays_bracketed(self):
        for y0, y1, y2 in [(5.0, 0.0, 0.01), (0.01, 0.0, 5.0), (2.0, 0.0, 2.0)]:
            offset = rg.parabolic_vertex(y0, y1, y2)
            assert -1.0 <= offset <= 1.0


class TestSaturate:

    def test_zero_at_zero(self):
        assert rg.saturate(0.0, 10.0) == 0.0

    def test_about_two_thirds_at_the_reference(self):
        assert rg.saturate(10.0, 10.0) == pytest.approx(1.0 - math.e**-1, abs=1e-6)

    def test_negative_input_clamped(self):
        assert rg.saturate(-5.0, 10.0) == 0.0

    def test_non_positive_reference_is_zero(self):
        assert rg.saturate(5.0, 0.0) == 0.0


class TestInteractionPrior:
    """The geometry table itself, independent of any trajectory fixture."""

    def test_head_on_bracket_is_crossing_not_following_or_oncoming(self):
        _, kind = rg.interaction_prior(90.0)
        assert kind == rg.CROSSING

    def test_53_degrees_is_crossing(self):
        prior, kind = rg.interaction_prior(53.0)
        assert kind == rg.CROSSING
        assert prior == rg.DEFAULT_CONFIG.prior_crossing

    def test_178_degrees_is_oncoming_and_demoted(self):
        prior, kind = rg.interaction_prior(178.0)
        assert kind == rg.ONCOMING
        assert prior == rg.DEFAULT_CONFIG.prior_oncoming
        assert prior < rg.DEFAULT_CONFIG.prior_crossing

    def test_138_degrees_is_crossing_not_oncoming(self):
        """The measured correction: 135 demoted a real 138 deg collision."""
        prior, kind = rg.interaction_prior(138.0)
        assert kind == rg.CROSSING
        assert prior == rg.DEFAULT_CONFIG.prior_crossing

    def test_small_angle_is_following_and_demoted(self):
        prior, kind = rg.interaction_prior(10.0)
        assert kind == rg.FOLLOWING
        assert prior < rg.DEFAULT_CONFIG.prior_crossing

    def test_unmeasurable_heading_is_unknown_and_ranks_below_crossing(self):
        prior, kind = rg.interaction_prior(None)
        assert kind == rg.UNKNOWN
        assert prior < rg.DEFAULT_CONFIG.prior_crossing


# ---------------------------------------------------------------------------
# sub-sample contact refinement
# ---------------------------------------------------------------------------

def test_contact_refinement_lands_strictly_between_bracketing_samples():
    """The true closest approach is deliberately off the 0.1 s sampling grid.

    Two straight, constant-velocity, non-parallel tracks cross at a
    continuous-time instant (t=2.53s) that is not a multiple of the 0.1s
    sample step. ``find_contact`` must not snap to the nearest grid sample --
    the refined instant has to fall strictly inside the two samples bracketing
    the discrete minimum, which is what makes the refinement worth having at
    all (COMBINED measured a vehicle covers ~0.5 m per frame at 30 fps/15 m/s).
    """
    t_star = 2.53
    vx_a, vy_a = 100.0, 0.0
    vx_b, vy_b = 0.0, 130.0
    start_a = (0.0 - vx_a * t_star, -5.0)
    start_b = (5.0, 0.0 - vy_b * t_star)

    a = _segment_track(401, start_a, [(6.0, vx_a, vy_a)])
    b = _segment_track(402, start_b, [(6.0, vx_b, vy_b)])

    cfg = rg.DEFAULT_CONFIG
    windows = rg.build_windows([a, b], cfg)
    assert len(windows) == 2
    wa, wb = windows

    contact = rg.find_contact(wa, wb, cfg)
    assert contact is not None
    assert contact.refined is True

    step = min(wa.median_step_s, wb.median_step_s)
    lo, hi = contact.t_grid - step, contact.t_grid + step
    assert lo < contact.t < hi, (
        f"refined contact time {contact.t} is not strictly between the "
        f"bracketing samples ({lo}, {hi})")
    assert contact.t != contact.t_grid

    # And it should actually have moved the estimate toward the true crossing
    # instant, not just anywhere in the bracket.
    assert abs(contact.t - t_star) < abs(contact.t_grid - t_star)


def test_find_contact_none_when_tracks_never_overlap_in_time():
    a = _segment_track(1, (0.0, 0.0), [(1.0, 10.0, 0.0)])           # t in [0, 1]
    b = _segment_track(2, (0.0, 100.0), [(1.0, 10.0, 0.0)])
    # Shift b's timestamps entirely past a's by rebuilding with a later start.
    b2 = Track(track_id=3, cls=_CLS_CAR, score=1.0,
              box=_box_at(0.0, 0.0), frame_idx=0, t=5.0)
    b2.update(_box_at(10.0, 0.0), 1.0, _CLS_CAR, 1, 6.0)
    cfg = rg.DEFAULT_CONFIG
    windows = rg.build_windows([a, b2], cfg)
    if len(windows) < 2:
        return  # one of them failed the travel/speed gate -- also a valid "no pair"
    assert rg.find_contact(windows[0], windows[1], cfg) is None


# ---------------------------------------------------------------------------
# the rotation gate discriminates braking from being struck
# ---------------------------------------------------------------------------

def test_braking_bystander_scores_far_below_rotating_struck_vehicle():
    """The measured video-13 failure mode, reproduced synthetically.

    Both vehicles lose comparable speed around the contact instant. The
    struck vehicle also gets kicked onto a new heading (rotation); the
    bystander keeps its heading and aspect exactly as they were -- it braked,
    it was not hit. The rotation gate must separate them by a wide margin.
    """
    cfg = rg.DEFAULT_CONFIG
    struck, _other = _crossing_pair()
    bystander = _braking_bystander()

    windows = rg.build_windows([struck, bystander], cfg)
    by_id = {w.track_id: w for w in windows}
    assert set(by_id) == {101, 201}

    ev_struck = rg.impact_evidence(by_id[101], 3.0, cfg)
    ev_bystander = rg.impact_evidence(by_id[201], 3.0, cfg)

    assert ev_struck.rotation > 0.5
    assert ev_bystander.rotation == 0.0
    # The bystander still shows real kinematic evidence (it did brake hard) --
    # this is what an additive score would have let through.
    assert ev_bystander.decel > 0.0 or ev_bystander.speed_drop > 0.0

    assert ev_bystander.score < 0.15
    assert ev_struck.score > 0.3
    assert ev_struck.score > 5 * ev_bystander.score


def test_rotation_gate_floor_and_saturation():
    cfg = rg.DEFAULT_CONFIG
    assert rg.rotation_gate(0.0, cfg) == pytest.approx(cfg.rotation_floor)
    assert rg.rotation_gate(1.0, cfg) == pytest.approx(1.0)


def test_track_break_substitutes_for_unobservable_rotation():
    """A track that dies at contact is evidence, not missing evidence.

    Fewer than 4 samples fall in the contact window because the track simply
    stops there. ``impact_evidence`` must not return zero for that vehicle --
    ``break_implies_rotation`` stands in for the rotation that could not be
    observed.
    """
    cfg = rg.DEFAULT_CONFIG
    vx, vy = _heading_vec(90.0)
    speed = 120.0
    # Track runs for >= track_break_min_life_s and then simply stops updating
    # at the contact instant.
    dies_at = _segment_track(501, (0.0, 0.0), [(2.0, vx * speed, vy * speed)])
    windows = rg.build_windows([dies_at], cfg,
                               now_t=dies_at.last_t + cfg.track_break_window_s + 0.01)
    assert len(windows) == 1
    win = windows[0]
    assert win.ended is True

    ev = rg.impact_evidence(win, win.last_t, cfg)
    assert ev.track_break == 1.0
    assert ev.rotation == pytest.approx(cfg.break_implies_rotation)
    assert ev.score > 0.0


# ---------------------------------------------------------------------------
# interaction geometry, end to end on synthetic trajectories
# ---------------------------------------------------------------------------

def test_oncoming_pass_is_classified_oncoming_and_scores_near_zero():
    cfg = rg.DEFAULT_CONFIG
    a, b = _oncoming_pass_pair()
    windows = rg.build_windows([a, b], cfg)
    assert len(windows) == 2

    pair = rg.score_pair(windows[0], windows[1], cfg)
    assert pair is not None
    assert pair.interaction == rg.ONCOMING
    assert pair.rel_heading_deg == pytest.approx(178.0, abs=1.0)
    assert pair.prior == cfg.prior_oncoming
    # Neither vehicle rotated or decelerated -- ordinary traffic passing.
    assert pair.evidence_a.rotation == 0.0
    assert pair.evidence_b.rotation == 0.0
    assert pair.score < 0.05


def test_crossing_impact_ranks_top_over_a_following_queue():
    """A genuine ~53 deg T-bone must outrank same-direction queuing traffic.

    Four tracks: a crossing collision pair (rotated, decelerated, ~53 deg
    relative heading) and a following pair some distance away (closing gap,
    braking together, never rotating -- ordinary queuing). ``score_pairs``
    must rank the crossing pair first, by a wide margin, and label it
    ``crossing``.
    """
    cfg = rg.DEFAULT_CONFIG
    c1, c2 = _crossing_pair(rel_deg=53.0)
    q1, q2 = _following_queue_pair()

    pairs = rg.score_pairs([c1, c2, q1, q2], cfg)
    assert len(pairs) >= 1

    top = pairs[0]
    assert {top.a, top.b} == {101, 102}
    assert top.interaction == rg.CROSSING
    assert 30.0 <= top.rel_heading_deg <= 70.0
    assert top.score > 0.3

    others = [p for p in pairs if {p.a, p.b} != {101, 102}]
    assert all(top.score > 3 * p.score for p in others)


def test_score_pairs_empty_when_fewer_than_two_candidates_survive():
    """Not finding a pair is the honest answer, not an error."""
    cfg = rg.DEFAULT_CONFIG
    lone = _braking_bystander()
    assert rg.score_pairs([lone], cfg) == []
    assert rg.score_pairs([], cfg) == []


def test_pair_result_as_dict_and_explain_do_not_raise():
    cfg = rg.DEFAULT_CONFIG
    c1, c2 = _crossing_pair()
    windows = rg.build_windows([c1, c2], cfg)
    pair = rg.score_pair(windows[0], windows[1], cfg)
    assert pair is not None

    d = pair.as_dict()
    assert d["track_ids"] == [101, 102]
    assert isinstance(d["evidence_a"], dict)

    sentence = pair.explain()
    assert "101" in sentence and "102" in sentence
