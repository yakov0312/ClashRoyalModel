"""Crown tower stats (``data/cards/Towers.json``, level 11).

Units: ``hitSpeed`` seconds, ``range``/``collisionRadius`` tiles,
``projectileSpeed`` tiles per second, ``size`` = square footprint in tiles.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict

from .path import TOWERS_FILE

PRINCESS_TOWER = "PrincessTower"
KING_TOWER = "KingTower"
CROWN_TOWER_NAMES = frozenset({PRINCESS_TOWER, KING_TOWER})


@lru_cache(maxsize=None)
def load_tower_stats(path: str = str(TOWERS_FILE)) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)


def tower_footprint_size(name: str) -> int:
    return int(load_tower_stats()[name]["size"])
