"""NETRA -- Network for Event Tracking, Response and Analysis.

Road incident intelligence for fixed traffic cameras.

Design thesis, in one line:
    the model recognises objects; the system recognises incidents.

One neural network (detection) runs continuously. Tracking, road geometry,
temporal statistics and deterministic state machines do everything else, which
is what makes the system explainable, cheap enough for commodity CPUs, and
honest about what it cannot know.
"""

__version__ = "1.0.0"

from .config import load_config          # noqa: F401
from .scene import SceneModel            # noqa: F401

__all__ = ["load_config", "SceneModel", "__version__"]
