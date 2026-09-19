"""Observation conventions shared by the simulator (``clasher/rl/obs_cv.py``)
and the real-game perception (``perception/bot/tensorBuilder.py``).

Both builders must produce identical tensors for the same game state, so any
normalisation lives here and is imported by both.

Frame: the player's screen view, 18 x 32 tiles, x left->right, y top->bottom,
enemy side at the top, own side at the bottom.
"""

BOARD_WIDTH, BOARD_HEIGHT = 18, 32

# Regulation length in seconds; the time feature is regulation time *remaining*
# (what the in-game clock shows), divided by this and clipped to [0, 1].
MATCH_DURATION = 180.0

# Tower HP feature = hp / TOWER_HP_NORM, clipped to [0, 1]; 0 once destroyed.
# Chosen above every crown tower's maximum HP at any level, so the feature
# never saturates while a tower is still healthy.
TOWER_HP_NORM = 8000.0


def towerHpFeature(hp: float) -> float:
    if hp <= 0:
        return 0.0
    return min(1.0, float(hp) / TOWER_HP_NORM)


def timeFeature(secondsRemaining: float) -> float:
    return min(1.0, max(0.0, float(secondsRemaining) / MATCH_DURATION))


# ---------------------------------------------------------------------------
# Static arena layout in the view frame (enemy rows 0-14, own rows 17-31).
# Matches the simulator's TileGrid (checked by tests/test_rl_obs_cv.py) and
# the real arena (bridges span the Princess Towers' three columns).
# ---------------------------------------------------------------------------
RIVER_ROWS = (15, 16)
BRIDGE_COLUMNS = ((2, 3, 4), (13, 14, 15))
WALL_SIDE_WIDTH = 6                      # fenced tiles each side of rows 0 and 31
RIVERSIDE_BLOCKED = ((0, 14), (17, 14), (0, 17), (17, 17))  # (x, y)
KING_TOWER_SIZE = 4
PRINCESS_TOWER_SIZE = 3

# Top-left tile (x, y) of each tower footprint.
TOWER_POSITIONS = {
    "enemy": {"left": (2, 5), "right": (13, 5), "king": (7, 1)},
    "own": {"left": (2, 24), "right": (13, 24), "king": (7, 27)},
}


def towerSize(tower: str) -> int:
    return KING_TOWER_SIZE if tower == "king" else PRINCESS_TOWER_SIZE


def staticUnwalkableMask(towersAlive=None):
    """(32, 18) float mask of unwalkable tiles. ``towersAlive`` maps
    (side, tower) -> bool; missing entries count as alive."""
    import numpy as np

    mask = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for row in (0, BOARD_HEIGHT - 1):
        mask[row, :WALL_SIDE_WIDTH] = 1.0
        mask[row, BOARD_WIDTH - WALL_SIDE_WIDTH:] = 1.0
    for row in RIVER_ROWS:
        mask[row, :] = 1.0
        for columns in BRIDGE_COLUMNS:
            mask[row, list(columns)] = 0.0
    for x, y in RIVERSIDE_BLOCKED:
        mask[y, x] = 1.0
    for side, towers in TOWER_POSITIONS.items():
        for tower, (x, y) in towers.items():
            if towersAlive is not None and not towersAlive.get((side, tower), True):
                continue
            size = towerSize(tower)
            mask[y:y + size, x:x + size] = 1.0
    return mask
