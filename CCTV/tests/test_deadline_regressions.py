"""Regression tests for failures that invalidate a completed batch."""

from types import SimpleNamespace

import numpy as np
import pytest

from netra.evidence import EvidenceWriter, FrameBuffer
from netra.db import IncidentStore
from netra.events.collision import CollisionEngine
from scripts.run_problems import saved_calibration_for
from netra.pipeline import Pipeline


def test_native_rate_sentinel_cannot_reach_evidence_objects(tmp_path):
    with pytest.raises(ValueError):
        FrameBuffer(seconds=30.0, fps=0.0)
    with pytest.raises(ValueError):
        EvidenceWriter(root=tmp_path, clip_fps=0.0)


def test_frame_buffer_capacity_uses_effective_rate():
    buf = FrameBuffer(seconds=2.0, fps=30.0, scale=1.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(100):
        buf.push(frame, i / 30.0)
    assert buf.fps == 30.0
    assert len(buf.buf) == 68


def test_evidence_paths_are_namespaced_by_run(tmp_path):
    writer = EvidenceWriter(root=tmp_path, clip_fps=8.0)
    first = writer.incident_dir("CAM_1", 1, run_id="run-a")
    second = writer.incident_dir("CAM_1", 1, run_id="run-b")
    assert first != second
    assert first == tmp_path / "CAM_1" / "run-a" / "INC-00001"


class _PairTrack:
    def __init__(self, tid, box, direction):
        self.track_id = tid
        self.box = np.asarray(box, dtype=float)
        self._direction = np.asarray(direction, dtype=float)
        self.cls = 2
        self.hits = 10
        self.corridor_id = None

    def speed_px(self, _window):
        return 40.0

    def direction(self, _window, min_span=0.0):
        return self._direction

    def acceleration_px(self, _window):
        return -80.0

    def heading_change(self, _window):
        return 50.0

    def aspect_shift(self, _window):
        return 0.0


def _ctx(a, b, t):
    return SimpleNamespace(tracks=[a, b], frame_shape=(720, 1280), t=t)


def test_pair_convergence_is_observed_before_contact():
    engine = CollisionEngine(SimpleNamespace(camera_id="test"), {
        "collision": {
            "proximity_scale": 1.0,
            "min_start_separation": 2.5,
            "pair_history_scale": 4.0,
            "min_convergence_ratio": 3.0,
        }
    })

    # First observe the two vehicles several footprints apart.
    a = _PairTrack(1, [0, 100, 100, 180], [1, 0])
    b = _PairTrack(2, [360, 100, 460, 180], [0, 1])
    engine._pairwise_conflict(_ctx(a, b, 0.0))
    state = engine._pair_state[(1, 2)]
    assert state["max_sep"] >= engine.min_start_separation
    assert not engine._pending

    # Then move into contact. The remembered separation makes convergence
    # reachable and arms the deferred aftermath check.
    b.box = np.asarray([100, 100, 200, 180], dtype=float)
    engine._pairwise_conflict(_ctx(a, b, 1.0))
    assert (1, 2) in engine._pending
    detail = engine._pending[(1, 2)]["detail"]
    assert detail["max_footprint_separation"] >= 2.5
    assert detail["convergence_ratio"] >= 3.0


def test_unvalidated_pair_channel_is_opt_in():
    engine = CollisionEngine(SimpleNamespace(camera_id="test"), {})
    assert engine.pairwise_enabled is False


def test_calibrated_camera_requires_independent_collision_channels():
    scene = SimpleNamespace(camera_id="fixed", corridors=[object()])
    engine = CollisionEngine(scene, {})
    assert engine.is_calibrated_camera is True
    assert engine.fixed_camera_min_channels == 2
    assert engine._promotion_allowed(True, 1) is False
    assert engine._promotion_allowed(True, 2) is False
    assert engine._promotion_allowed(True, 2, impulse_confirmed=True) is True


def test_uncalibrated_accident_clip_keeps_high_recall_mode():
    scene = SimpleNamespace(camera_id="clip", corridors=[])
    engine = CollisionEngine(scene, {})
    assert engine.is_calibrated_camera is False
    assert engine._promotion_allowed(True, 1) is True


def test_path_attribution_rejects_self_intersection_and_immature_pair():
    assert not CollisionEngine._path_attributable([5, 5], 10.0)
    assert not CollisionEngine._path_attributable([5, 7], 0.9)
    assert CollisionEngine._path_attributable([5, 7], 2.0)


def test_calibration_reuse_follows_video_across_group_rename(tmp_path):
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    old = camera_dir / "TRAFFIC_same_clip.json"
    old.write_text("{}", encoding="utf-8")

    wanted = camera_dir / "ELCIADATASET_same_clip.json"
    assert saved_calibration_for(wanted, "same_clip") == old


def test_dashboard_scope_keeps_only_latest_run_per_camera(tmp_path):
    store = IncidentStore(tmp_path / "incidents.db")
    for run_id, created in (("old", 1.0), ("new", 2.0)):
        store.conn.execute(
            "INSERT INTO incidents (run_id,camera_id,event_type,status,"
            "needs_verification,created_at) VALUES (?,?,?,?,?,?)",
            (run_id, "CAM", "queue", "detected", 0, created),
        )
    store.conn.commit()

    assert len(store.incidents(limit=10)) == 2
    assert len(store.incidents(limit=10, latest_only=True)) == 1
    assert store.summary(latest_only=True)["total_incidents"] == 1
    store.close()


def test_clean_latest_run_hides_older_false_alarm(tmp_path):
    """A zero-incident rerun is still a run and must clear the live feed."""
    store = IncidentStore(tmp_path / "incidents.db")
    store.conn.execute(
        "INSERT INTO incidents (run_id,camera_id,event_type,status,created_at) "
        "VALUES (?,?,?,?,?)", ("old", "CAM", "collision_candidate", "detected", 1.0))
    store.conn.commit()
    store.record_metric("clean", "CAM", "system.video_seconds", 30.0)
    assert store.incidents(camera_id="CAM", latest_only=True) == []
    store.close()


def test_dashboard_consolidates_raw_collision_candidates_without_deleting_them(tmp_path):
    store = IncidentStore(tmp_path / "collisions.db")
    for priority in (0.2, 0.8, 0.4):
        store.conn.execute(
            "INSERT INTO incidents (run_id,camera_id,event_type,status,priority,"
            "created_at) VALUES (?,?,?,?,?,?)",
            ("run", "CAM", "collision_candidate", "detected", priority, 1.0),
        )
    store.conn.commit()
    assert len(store.incidents(limit=10)) == 3
    rows = store.incidents(limit=10, consolidated_only=True)
    assert len(rows) == 1
    assert rows[0]["priority"] == pytest.approx(0.8)
    store.close()


def test_weak_single_geometry_keeps_event_but_withholds_participants():
    event = SimpleNamespace(
        track_ids=[1, 2],
        triggers={
            "attribution": "path-crossing",
            "participant_boxes": [[0, 0, 10, 10], [10, 0, 20, 10]],
            "path_conflict_channel": {"score": 0.59},
            "corroboration": {"independent_geometries_at_this_moment": 1},
        },
    )
    Pipeline._validate_collision_attribution(event)
    assert event.track_ids == []
    assert event.triggers["participant_boxes"] == []
    assert event.triggers["attribution"] == "unattributed"


def test_strong_or_corroborated_attribution_remains_visible():
    for score, agreeing in ((0.81, 1), (0.60, 2)):
        event = SimpleNamespace(
            track_ids=[1, 2],
            triggers={
                "attribution": "path-crossing",
                "participant_boxes": [[0, 0, 10, 10]],
                "path_conflict_channel": {"score": score},
                "corroboration": {
                    "independent_geometries_at_this_moment": agreeing},
            },
        )
        Pipeline._validate_collision_attribution(event)
        assert event.track_ids == [1, 2]
        assert event.triggers["participant_boxes"]


def test_weak_rear_end_and_aspect_only_single_are_unattributed():
    rear = SimpleNamespace(
        track_ids=[1, 2],
        triggers={
            "attribution": "path-crossing",
            "participant_boxes": [[0, 0, 10, 10]],
            "path_conflict_channel": {
                "score": 0.72,
                "heading_change_deg": [5.6, 5.6],
                "gates": {"geometry": "rear-end"},
            },
            "corroboration": {"independent_geometries_at_this_moment": 2},
        },
    )
    Pipeline._validate_collision_attribution(rear)
    assert rear.triggers["attribution"] == "unattributed"

    aspect_only = SimpleNamespace(
        track_ids=[153],
        triggers={
            "attribution": "stationary-object-track",
            "participant_boxes": [[0, 0, 10, 10]],
            "participant_selection": {
                "mode": "single-vehicle", "off_road": False},
        },
    )
    Pipeline._validate_collision_attribution(aspect_only)
    assert aspect_only.triggers["attribution"] == "unattributed"
