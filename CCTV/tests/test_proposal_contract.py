"""Safety and output contracts for the ELCIA proposal deliverables."""

from netra.events.base import Event, QUEUE, WRONG_WAY
from netra.api import _without_deprecated_aliases
from netra.pipeline import Pipeline
from netra.scene import Corridor, SceneModel
from netra.severity import recommend


def make_event(kind, started=10.0, detected=20.0, corridor="c1"):
    return Event(
        type=kind, camera_id="CAM", started_t=started, detected_t=detected,
        confidence=0.9, corridor_id=corridor,
    )


def test_report_event_preserves_location_action_and_context():
    ev = make_event(QUEUE)
    ev.location = {"zone": "ELCIA", "precision": "zone-only"}
    ev.recommended_action = "Traffic-control review"
    ev.operational_context = {"classification": "queue_buildup"}
    data = ev.to_dict()
    assert data["location"]["zone"] == "ELCIA"
    assert data["recommended_action"] == "Traffic-control review"
    assert data["operational_context"]["classification"] == "queue_buildup"


def test_draft_direction_wrong_way_requires_verification():
    ev = make_event(WRONG_WAY)
    ev.triggers["legal_direction_reviewed"] = False
    assert ev.needs_verification is True


def test_scene_draft_is_not_a_legal_direction():
    corridor = Corridor("c1", [(0, 0), (10, 0), (10, 10)], (1, 0), name="lane")
    draft = SceneModel("CAM", corridors=[corridor], notes="DRAFT auto-calibration")
    reviewed = SceneModel("CAM", corridors=[corridor], notes="Reviewed by operator")
    assert draft.legal_direction_reviewed is False
    assert reviewed.legal_direction_reviewed is True


def test_collision_queue_fusion_is_cautious_not_confirmed():
    from netra.events.base import COLLISION

    queue = make_event(QUEUE, started=10.0, detected=20.0)
    collision = make_event(COLLISION, started=15.0, detected=18.0)
    pipe = object.__new__(Pipeline)
    pipe.events = [queue, collision]
    pipe.store = None
    Pipeline._refresh_operational_context(pipe)

    assert queue.operational_context["classification"] == "suspected_accident_related_congestion"
    assert "unverified" in queue.operational_context["causality"]
    assert "Verify collision" in recommend(queue)


def test_canonical_elcia_camera_hides_legacy_traffic_alias():
    canonical = SceneModel("ELCIADATASET_123")
    legacy = SceneModel("TRAFFIC_123")
    other = SceneModel("TRAFFIC_456")
    out = _without_deprecated_aliases({
        canonical.camera_id: canonical, legacy.camera_id: legacy,
        other.camera_id: other,
    })
    assert "ELCIADATASET_123" in out
    assert "TRAFFIC_123" not in out
    assert "TRAFFIC_456" in out
