"""Event engines: deterministic reasoning over tracks and scene geometry."""

from .base import (
    ALWAYS_VERIFY,
    AUTO_CONFIRMABLE,
    BLOCKAGE,
    COLLISION,
    EVENT_LABELS,
    LANE_VIOLATION,
    PEDESTRIAN,
    QUEUE,
    STOPPED,
    WRONG_WAY,
    Event,
    EventEngine,
)
from .blockage import BlockageEngine
from .collision import CollisionEngine
from .queueing import QueueEngine
from .wrongway import WrongWayEngine

ENGINES = [WrongWayEngine, QueueEngine, BlockageEngine, CollisionEngine]

__all__ = [
    "Event", "EventEngine", "ENGINES",
    "WrongWayEngine", "QueueEngine", "BlockageEngine", "CollisionEngine",
    "WRONG_WAY", "LANE_VIOLATION", "QUEUE", "BLOCKAGE", "COLLISION",
    "STOPPED", "PEDESTRIAN", "EVENT_LABELS", "AUTO_CONFIRMABLE", "ALWAYS_VERIFY",
]
