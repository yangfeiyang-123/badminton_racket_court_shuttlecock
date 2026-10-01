"""Skip tests that rebuild full-body scenes when MuscleMimic is not installed."""
from __future__ import annotations

import importlib.util

_NEEDS_MUSCLEMIMIC = [
    "environment/double_play/tests/test_double_play_scene.py",
    "environment/double_play/tests/test_rally_physics.py",
    "tests/unit/test_right_hand_racket_grip.py",
]


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


collect_ignore = [] if _has("musclemimic") and _has("musclemimic_models") else _NEEDS_MUSCLEMIMIC
