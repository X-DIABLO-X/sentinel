"""Regression tests for the collision reasoning.

Every test here corresponds to a defect that actually shipped into a run and was
caught by measurement. They are written as assertions about *physics and stated
rules* rather than about detector outputs, so they run in under a second on CPU
with no model weights and no video.

The bugs they pin down:

* a momentum test that never executed, because it asked about the present;
* a per-frame impulse measure that saturated on ordinary driving;
* a parked-vehicle veto that was outvoted by a weighted sum;
* footprint geometry that never took effect because a caller shadowed it;
* penalties that treated an unmeasured value as an innocent one.
"""

from __future__ import annotations

import numpy as np
import pytest

from netra.footprint import Footprint, separation
from netra.predict import (
    MotionPredictor,
    ResidualMonitor,
    momentum_exchange,
    pixels_per_metre,
    vehicle_mass_kg,
)


# --------------------------------------------------------------------------
# Momentum exchange: the test that separates a crash from a queue
# --------------------------------------------------------------------------

def _inelastic(v_a, m_a, v_b, m_b) -> float:
    """Exchange value for a perfectly inelastic collision (momentum conserved)."""
    v_a, v_b = np.array(v_a, float), np.array(v_b, float)
    v_common = (m_a * v_a + m_b * v_b) / (m_a + m_b)
    return momentum_exchange(v_common - v_a, m_a, v_common - v_b, m_b)


CAR, TRUCK, BUS, BIKE = 1400.0, 9000.0, 12000.0, 150.0


@pytest.mark.parametrize("v_a, m_a, v_b, m_b, label", [
    ([20, 0], CAR,  [8, 0],  CAR,   "rear-end into a slower car"),
    ([14, 0], CAR,  [0, 12], CAR,   "T-bone at a junction"),
    ([18, 0], CAR,  [5, 0],  TRUCK, "car into a truck, 6:1 mass"),
    ([12, 0], BIKE, [0, 9],  BUS,   "two-wheeler into a bus, 80:1 mass"),
])
def test_collisions_exchange_momentum(v_a, m_a, v_b, m_b, label):
    """Any momentum-conserving impact cancels almost completely."""
    assert _inelastic(v_a, m_a, v_b, m_b) > 0.95, label


@pytest.mark.parametrize("dv_a, dv_b, label", [
    ([-5, 0],   [-4, 0],   "both braking for a signal"),
    ([3, 0],    [2.5, 0],  "both accelerating away"),
    ([-6, 0],   [0, 0],    "one brakes, the other holds"),
    ([-9, 0],   [-9, 0],   "an entire queue stopping together"),
])
def test_queues_do_not_exchange_momentum(dv_a, dv_b, label):
    """Common-mode braking adds rather than cancels; this is the discriminator.

    A rear-end collision and a queue are geometrically identical -- vehicles
    nose to tail with touching footprints -- so this is the only signal that
    separates them.
    """
    assert momentum_exchange(dv_a, CAR, dv_b, CAR) < 0.05, label


def test_exchange_is_symmetric_and_bounded():
    a, b = [-7.0, 1.0], [6.0, -1.0]
    assert momentum_exchange(a, CAR, b, TRUCK) == pytest.approx(
        momentum_exchange(b, TRUCK, a, CAR), abs=1e-9)
    assert 0.0 <= momentum_exchange(a, CAR, b, TRUCK) <= 1.0


def test_exchange_of_nothing_is_zero():
    assert momentum_exchange(None, CAR, [1, 1], CAR) == 0.0
    assert momentum_exchange([0, 0], CAR, [0, 0], CAR) == 0.0


# --------------------------------------------------------------------------
# The lag bug: velocity change needs observations on BOTH sides of the moment
# --------------------------------------------------------------------------

def _drive(pred: MotionPredictor, frames: int, dx: float, *, t0: float = 0.0,
           fps: float = 30.0, x0: float = 100.0):
    """Feed a vehicle moving at constant image speed."""
    for i in range(frames):
        t = t0 + i / fps
        x = x0 + i * dx
        pred.observe(t, [x, 100.0, x + 80.0, 160.0], 2, None, dt_nominal=1 / fps)
    return t0 + (frames - 1) / fps


def test_impulse_at_present_instant_is_unanswerable():
    """The defect that silently disabled the whole channel.

    Asking about "now" can never succeed, because the far side of the window has
    not been observed yet. The first implementation did exactly this and
    returned None on every call, so no pair was ever formed across 299,499
    residuals -- a dead channel that looked like a clean one.
    """
    p = MotionPredictor()
    last = _drive(p, 40, 4.0)
    assert p.impulse_at(last) is None
    assert p.impulse_at(last - 0.4) is not None


def test_constant_velocity_produces_no_impulse():
    """Ordinary driving must not saturate the measure.

    Scored per frame, clean traffic reached a 99.9th percentile of 0.986 --
    differentiating a filtered estimate over a 33 ms step measures the filter,
    not the vehicle.
    """
    p = MotionPredictor()
    last = _drive(p, 60, 4.0)
    imp = p.impulse_at(last - 0.4)
    assert imp is not None
    assert imp.score < 0.05
    assert imp.accel_m_s2 < 0.5


def test_abrupt_stop_produces_an_impulse():
    p = MotionPredictor()
    fps = 30.0
    for i in range(30):                      # travelling
        t = i / fps
        x = 100.0 + i * 12.0
        p.observe(t, [x, 100, x + 80, 160], 2, None, dt_nominal=1 / fps)
    x_stop = 100.0 + 29 * 12.0
    for i in range(30, 60):                  # stopped dead
        t = i / fps
        p.observe(t, [x_stop, 100, x_stop + 80, 160], 2, None, dt_nominal=1 / fps)
    imp = p.impulse_at(29 / fps)
    assert imp is not None
    assert imp.score > 0.3, "an instant stop must exceed driver control authority"


def test_box_at_recovers_the_past_not_the_present():
    p = MotionPredictor()
    last = _drive(p, 40, 4.0)
    past = p.box_at(last - 0.4)
    now = p.box_at(last)
    assert past is not None and now is not None
    assert past[0] < now[0], "the earlier box must be further back along travel"
    assert p.box_at(last - 99.0) is None


# --------------------------------------------------------------------------
# Frame-rate floor: below it, an impact and a dropped detection are identical
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fps, available", [(30.0, True), (25.0, True),
                                            (15.0, True), (14.954, True),
                                            (10.0, True), (8.0, True),
                                            (4.0, False)])
def test_availability_follows_samples_not_a_frame_rate(fps, available):
    """Real CCTV runs at whatever rate it runs at, and 15 fps is not a law.

    A held-out clip reporting 14.954 fps was once refused by 0.046 fps, which
    switched prediction off for every vehicle in it. What a velocity estimate
    needs is a few samples either side of the moment, so the window widens on
    slow footage instead of the channel disabling itself. Only genuinely
    degenerate rates, where a vehicle teleports between observations, are
    refused.
    """
    assert ResidualMonitor(fps=fps).available is available


def test_window_widens_to_keep_enough_samples():
    fast, slow = ResidualMonitor(fps=30.0), ResidualMonitor(fps=8.0)
    assert slow.window_s > fast.window_s
    for mon in (fast, slow):
        assert mon.window_s * mon.fps >= mon.min_samples - 1e-6
        assert mon.lag_s > mon.window_s, "the far side of the window must be observable"


def test_unavailable_monitor_returns_no_pairs():
    mon = ResidualMonitor(fps=4.0)          # genuinely degenerate, not merely slow
    assert mon.impulse_pairs([], 1.0) == []
    assert mon.observe([], 1.0) == {}


# --------------------------------------------------------------------------
# Scale: the detection box is the ruler
# --------------------------------------------------------------------------

def test_pixels_per_metre_shrinks_with_distance():
    """One physical threshold must hold near the camera and at the horizon."""
    near = pixels_per_metre([0, 0, 180, 120], 2)     # 180 px wide car
    far = pixels_per_metre([0, 0, 45, 30], 2)        # same car, further away
    assert near == pytest.approx(100.0, rel=0.01)
    assert far == pytest.approx(25.0, rel=0.01)


def test_mass_lookup_distinguishes_indian_classes():
    names = {0: "two-wheeler", 1: "auto-rickshaw", 2: "car", 3: "bus"}
    assert vehicle_mass_kg(0, names) < vehicle_mass_kg(1, names)
    assert vehicle_mass_kg(1, names) < vehicle_mass_kg(2, names)
    assert vehicle_mass_kg(2, names) < vehicle_mass_kg(3, names)


# --------------------------------------------------------------------------
# Footprints: proximity on the road surface, not in image space
# --------------------------------------------------------------------------

def test_touching_vehicles_share_road_surface():
    a = Footprint.from_box([100, 100, 200, 180])
    b = Footprint.from_box([200, 100, 300, 180])
    assert separation(a, b) <= 1.0


def test_background_and_foreground_vehicles_are_far_apart():
    """The failure footprints were introduced to fix.

    A distant bus and a near car can have almost touching bounding boxes while
    standing many metres apart on the ground.
    """
    far = Footprint.from_box([300, 100, 340, 130])      # small: distant
    near = Footprint.from_box([280, 300, 480, 460])     # large: close
    assert separation(far, near) > 1.5


def test_separation_is_symmetric():
    a = Footprint.from_box([0, 0, 100, 80])
    b = Footprint.from_box([160, 40, 260, 120])
    assert separation(a, b) == pytest.approx(separation(b, a))


# --------------------------------------------------------------------------
# Path conflict: courses that converge, then confirm
# --------------------------------------------------------------------------

from netra.pathconflict import (                              # noqa: E402
    PathConflict,
    PathConflictDetector,
    ray_conflict,
)

FPS = 15.0


class _Track:
    """A track with a distinct box, because two vehicles occupy two places."""

    def __init__(self, tid, history, box):
        self.track_id = tid
        self.history = history
        self.box = np.array(box, dtype=float)


def _box_following(hist, w=50.0, h=30.0):
    """A box that sits under the vehicle's current position."""
    x, y = hist[-1][1], hist[-1][2]
    return [x - w / 2, y - h, x + w / 2, y]


def _play(hist_a, hist_b, box_a, box_b):
    """Drive the detector frame by frame, as the pipeline does.

    Detection is staged -- a converging course is registered first and only
    confirmed once the vehicles arrive -- so it cannot be exercised with a
    single call.
    """
    d = PathConflictDetector()
    n = max(len(hist_a), len(hist_b))
    for k in range(2, n + 1):
        t = hist_a[min(k, len(hist_a)) - 1][0]
        ba = box_a if box_a is not None else _box_following(hist_a[:k])
        bb = box_b if box_b is not None else _box_following(hist_b[:k])
        hits = d.find([_Track(1, hist_a[:k], ba),
                       _Track(2, hist_b[:k], bb)], t)
        if hits:
            return hits[0]
    return None


def _converging():
    """A east, B north, meeting at (200, 300); both stop dead on contact."""
    a = [(i / FPS, 100 + i * 10, 300) for i in range(11)]
    a += [(a[-1][0] + i / FPS, 200 + i * 1.0, 300 + i * 4.0) for i in range(1, 12)]
    b = [(i / FPS, 200, 400 - i * 10) for i in range(11)]
    b += [(b[-1][0] + i / FPS, 200 + i * 2.0, 300 + i * 0.5) for i in range(1, 12)]
    return a, b


def test_converging_pair_that_is_disturbed_is_detected():
    """The pattern review identified: courses cross, then the vehicles arrive.

    Which geometry reports it is not fixed. A vehicle struck at a junction is
    also knocked off its heading, and a corner beyond tyre grip is stronger
    evidence than a predicted crossing -- it is a measurement of a force that
    has already acted rather than a forecast that two paths will meet. So this
    asserts that the collision is found and both vehicles are named, not which
    of the two descriptions won.
    """
    a, b = _converging()
    hit = _play(a, b, [160, 270, 200, 300], [200, 270, 240, 300])
    assert hit is not None
    assert set(hit.track_ids) == {1, 2}
    assert hit.mode in ("crossing", "deflection"), hit.mode
    assert hit.score > 0.6
    if hit.mode == "crossing":
        assert hit.time_gap_s < 0.3, "they arrive together"
        assert hit.angle_deg > 60


def test_same_courses_passing_cleanly_are_not_a_collision():
    """Being on a converging course is not a collision; most of them resolve."""
    a = [(i / FPS, 100 + i * 10, 300) for i in range(22)]
    b = [(i / FPS, 200, 400 - i * 10) for i in range(22)]
    assert _play(a, b, [160, 270, 200, 300], [400, 270, 440, 300]) is None


def test_a_queue_has_parallel_courses_and_never_conflicts():
    """Parallel courses do not intersect, so no threshold can readmit queues."""
    a = [(i / FPS, 100 + i * 6, 300) for i in range(22)]
    b = [(i / FPS, 220 + i * 6, 300) for i in range(22)]
    assert _play(a, b, [100, 270, 140, 300], [220, 270, 260, 300]) is None


def test_fitted_candidate_score_is_recorded_but_does_not_bypass_gates(tmp_path):
    """The fitted model ranks physical candidates; it cannot invent one."""
    import json

    model = tmp_path / "candidate_score.json"
    model.write_text(json.dumps({
        "features": ["score_rule"],
        "mean": [0.0], "std": [1.0], "weights": [2.0], "bias": 0.0,
    }), encoding="utf-8")
    detector = PathConflictDetector(score_model=model)
    pc = PathConflict(
        track_ids=(4, 9), point=np.array([100.0, 200.0]), t_cross=3.0,
        angle_deg=60.0, time_gap_s=0.1, deviation_deg=(20.0, 5.0),
        speed_drop=(0.5, 0.1), boxes=[np.array([10, 20, 50, 60])],
        gates={"geometry": "crossing"}, mode="crossing",
    )
    row = detector._record_candidate(pc, {})
    assert row["boxes"] == [[10.0, 20.0, 50.0, 60.0]]
    assert 0.5 < row["learned_score"] < 1.0
    assert pc.gates["learned_score"] == row["learned_score"]
    assert len(detector.candidates) == 1


def test_duplicate_tracks_on_one_vehicle_cannot_collide():
    """Two ids on one car must not read as two cars colliding.

    The claim under test is about PAIR attribution. A solo finding on either id
    -- the vehicle did stop dead -- is a different and legitimate claim, so what
    must not happen is the two ids being reported as two vehicles in contact
    with each other.
    """
    a, b = _converging()
    same = [160, 270, 200, 300]
    hit = _play(a, b, same, same)
    if hit is not None:
        ka, kb = hit.track_ids
        assert ka == kb, f"reported as a pair: {hit.track_ids} {hit.gates}"


def test_post_encroachment_time_geometry():
    """PET is how far apart in time the two would reach the conflict point."""
    hit = ray_conflict([0, 300], [100, 0], [200, 500], [0, -100], 3.0)
    assert hit is not None
    point, t_a, t_b = hit
    assert point[0] == pytest.approx(200) and point[1] == pytest.approx(300)
    assert abs(t_a - t_b) == pytest.approx(0.0, abs=1e-6)

    # parallel courses never conflict, whatever the spacing
    assert ray_conflict([0, 300], [100, 0], [0, 340], [100, 0], 3.0) is None
    # diverging courses have already passed their crossing
    assert ray_conflict([0, 300], [-100, 0], [200, 500], [0, 100], 3.0) is None
    # a meeting beyond the horizon is not yet anybody's problem
    assert ray_conflict([0, 300], [10, 0], [200, 500], [0, -10], 3.0) is None


def test_deflection_needs_more_than_tyres_can_deliver():
    """A corner sharper than grip allows was not steered -- something pushed it.

    Momentum is m*v along the heading; turning it needs a lateral force, and
    the only one a driver commands is grip, about 0.9 g.

    Both vehicles are disturbed here, as Newton's third law requires. An
    earlier version of this fixture had the striking vehicle continue
    undisturbed, which is not a collision at all -- see the test below for why
    that distinction matters.
    """
    d = PathConflictDetector()
    # the other vehicle is deflected too, in the opposite sense
    other = [(i / FPS, 205 + i * 12, 445) for i in range(12)]
    other += [(12 / FPS + i / FPS, 349 + i * 13, 445 - i * 4) for i in range(1, 14)]

    struck = [(i / FPS, 200 + i * 12, 400) for i in range(12)]
    struck += [(12 / FPS + i / FPS, 344 + i * 7, 400 + i * 9) for i in range(1, 14)]
    hit = _play(struck, other, None, None)
    assert hit is not None and hit.mode == "deflection"
    assert hit.gates["lateral_g"] > 1.3

    undisturbed = [(i / FPS, 205 + i * 12, 445) for i in range(26)]
    lane_change = [(i / FPS, 200 + i * 12, 400) for i in range(12)]
    lane_change += [(12 / FPS + i / FPS, 344 + i * 11.5, 400 + i * 1.6)
                    for i in range(1, 14)]
    assert _play(lane_change, undisturbed, None, None) is None


def test_partner_disturbance_is_recorded_for_fitting():
    """Whether the other vehicle felt it is measured, not used as a veto.

    As a hard gate this removed more than half the recall on the held-out set,
    because mass ratios are large and a struck lorry barely moves. It is real
    evidence all the same, so it is recorded as a feature and its weight is
    fitted from labelled clips rather than assumed.
    """
    d = PathConflictDetector()
    swerver = [(i / FPS, 200 + i * 12, 400) for i in range(12)]
    swerver += [(12 / FPS + i / FPS, 344 + i * 7, 400 + i * 9) for i in range(1, 14)]
    unmoved = [(i / FPS, 205 + i * 12, 445) for i in range(26)]
    for k in range(2, len(swerver) + 1):
        d.find([_Track(1, swerver[:k], _box_following(swerver[:k])),
                _Track(2, unmoved[:k], _box_following(unmoved[:k]))],
               swerver[k - 1][0])
    assert d.candidates, "candidates must be recorded even when not reported"
    assert all(set(PathConflictDetector.FEATURES) <= set(c) for c in d.candidates)


# --------------------------------------------------------------------------
# Lanes inferred from traffic rather than from paint
# --------------------------------------------------------------------------

from netra.lanes import LaneModel   # noqa: E402
from netra.scene import Corridor, SceneModel   # noqa: E402


def _road(rng):
    """Two-way road: three lanes one way, two the other, no markings involved."""
    trajs = []
    for y, n, vx in [(300, 9, 8), (360, 11, 8), (420, 7, 8),
                     (520, 10, -8), (580, 8, -8)]:
        for _ in range(n):
            y0 = y + rng.normal(0, 6)
            x0 = rng.uniform(100, 200) if vx > 0 else rng.uniform(800, 900)
            trajs.append(np.array([[x0 + vx * i * 6, y0 + rng.normal(0, 2)]
                                   for i in range(20)]))
    return trajs


def test_lanes_are_recovered_from_vehicle_paths():
    """Vehicles concentrate into bands; the gaps between them declare the lanes.

    No cluster count is supplied -- the number of lanes falls out of where
    nobody drives, which is why this works on roads whose markings are worn
    away, repainted at an offset, or absent.
    """
    m = LaneModel.learn(_road(np.random.default_rng(0)),
                        frame_shape=(720, 1280), vehicle_width_px=60.0)
    assert len(m.lanes) == 5
    assert len({ln.direction_id for ln in m.lanes}) == 2, "a two-way road"
    for ln in m.lanes:
        assert ln.support >= 4


def test_the_gap_between_carriageways_is_not_a_lane():
    m = LaneModel.learn(_road(np.random.default_rng(1)), vehicle_width_px=60.0)
    assert m.lane_at((500, 480)) is None


def test_lane_departure_is_reported():
    m = LaneModel.learn(_road(np.random.default_rng(2)), vehicle_width_px=60.0)
    moved = m.departure((500, 362), (500, 300))
    assert moved is not None and moved[0].lane_id != moved[1].lane_id


def test_no_lanes_are_invented_from_too_little_traffic():
    """One wandering vehicle is not a lane."""
    lone = [np.array([[100 + 8 * i, 300] for i in range(20)])]
    assert LaneModel.learn(lone, vehicle_width_px=60.0).lanes == []


def test_saved_scene_geometry_scales_with_analysis_resolution():
    scene = SceneModel(
        camera_id="scaled", frame_size=(1280, 720),
        corridors=[Corridor(id="c1", polygon=[(100, 100), (300, 100),
                                               (300, 300), (100, 300)],
                            direction=np.array([1.0, 0.0]),
                            baseline_speed_px=20.0)])
    scaled = scene.scaled_to((1920, 1080))
    assert scaled.frame_size == (1920, 1080)
    assert scaled.corridors[0].polygon[0] == (150.0, 150.0)
    assert scaled.corridors[0].baseline_speed_px == pytest.approx(30.0)
    assert np.allclose(scaled.corridors[0].direction, [1.0, 0.0])


# --------------------------------------------------------------------------
# Rollover: the silhouette turns over and stays over
# --------------------------------------------------------------------------

class _BoxTrack:
    def __init__(self, tid, history, boxes):
        self.track_id = tid
        self.history = history
        self.box_history = boxes
        self.box = np.array(boxes[-1][1], dtype=float)


def _drive_boxes(n, upright=True, flip_at=None):
    hist = [(i / FPS, 200 + i * 12, 400) for i in range(n)]
    boxes = []
    for i in range(n):
        x = 200 + i * 12
        if flip_at is not None and i >= flip_at:
            boxes.append((i / FPS, [x - 45, 368, x + 45, 400]))     # wide, flat
        else:
            boxes.append((i / FPS, [x - 25, 340, x + 25, 400]))     # upright
    return hist, boxes


def _first_hit(hist, boxes):
    d = PathConflictDetector()
    for k in range(4, len(hist) + 1):
        hits = d.find([_BoxTrack(1, hist[:k], boxes[:k])], hist[k - 1][0])
        if hits:
            return hits[0]
    return None


def test_a_rollover_is_detected_from_the_silhouette_alone():
    """No partner, no crossing, no prediction -- the vehicle is on its roof.

    Single-vehicle accidents were 0 for 4 on the held-out set while five of the
    sixteen clips are annotated as rollovers, and this is the most direct
    evidence a camera can have of one.
    """
    hist, boxes = _drive_boxes(34, flip_at=13)
    hit = _first_hit(hist, boxes)
    assert hit is not None
    assert hit.gates["geometry"] == "rollover"
    assert hit.gates["aspect_ratio_change"] > 1.75


def test_a_turn_is_not_a_rollover():
    """Presenting a different face to the camera is reversible; a rollover isn't."""
    hist, _ = _drive_boxes(34)
    boxes = []
    for i in range(34):
        x = 200 + i * 12
        wide = 9 <= i < 17
        boxes.append((i / FPS, [x - (45 if wide else 25), 368 if wide else 340,
                                x + (45 if wide else 25), 400]))
    assert _first_hit(hist, boxes) is None


def test_ordinary_driving_never_looks_like_a_rollover():
    hist, boxes = _drive_boxes(34)
    assert _first_hit(hist, boxes) is None
