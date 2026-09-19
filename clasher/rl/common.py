from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOARD_WIDTH = 18
BOARD_HEIGHT = 32
NUM_TILES = BOARD_WIDTH * BOARD_HEIGHT
NUM_HAND_SLOTS = 4
N_MAX = 40


# ---------------------------------------------------------------------------
# Player view frame
# ---------------------------------------------------------------------------
# Observations and actions use the frame the real game (and the perception
# TensorBuilder) shows a player: x runs left->right on screen, y runs top->
# bottom, the ENEMY is at the top (rows 0-14) and YOUR side at the bottom
# (rows 17-31, own King on rows 27-30). Blue (player 0) sits at low world-y
# and red (player 1) at high world-y, so blue's view flips y and red's view
# flips x; both players see a board that looks identical from their seat.

def world_to_view(x: float, y: float, player_id: int) -> tuple[float, float]:
    """Continuous world position -> continuous view position."""
    if player_id == 0:
        return x, BOARD_HEIGHT - y
    return BOARD_WIDTH - x, y


def view_to_world(x: float, y: float, player_id: int) -> tuple[float, float]:
    return world_to_view(x, y, player_id)  # the mapping is its own inverse


def world_tile_to_view(x: int, y: int, player_id: int) -> tuple[int, int]:
    """World tile -> view tile (integer tiles, same mapping as positions)."""
    if player_id == 0:
        return x, BOARD_HEIGHT - 1 - y
    return BOARD_WIDTH - 1 - x, y


def view_tile_to_world(x: int, y: int, player_id: int) -> tuple[int, int]:
    return world_tile_to_view(x, y, player_id)


@dataclass(frozen=True)
class CvObservation:
    board: np.ndarray
    entities: np.ndarray
    entity_mask: np.ndarray
    global_features: np.ndarray
