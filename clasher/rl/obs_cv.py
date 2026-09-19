"""Simulator observation, tensor-for-tensor compatible with the real-game
perception (``perception/bot/tensorBuilder.py``).

Everything is expressed in the acting player's *view frame* (see
``common.world_to_view``): enemy at the top, own side at the bottom, exactly
what that player sees on screen. Both players therefore get identical
observations for mirrored situations, and actions use the same frame.

board (3, 32, 18):
    0  static unwalkable tiles: fence blocks, river (minus bridges), living
       crown-tower footprints
    1  tiles occupied by own units
    2  tiles occupied by enemy units
entities (N_MAX, 6): card id, owner (0 own / 1 enemy), hp fraction,
    tile x / 18, tile y / 32, present flag. Units only (troops and
    non-crown buildings), like the perception detector; crown towers are in
    the terrain channel and the global features.
global (18): regulation time remaining, double/triple elixir, overtime,
    own elixir, own crowns, enemy crowns, own King/left/right tower HP,
    enemy King/left/right tower HP ("left"/"right" as seen on screen),
    4 hand card ids, next card id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from clasher.arena import Position
from clasher.battle import BattleState
from clasher.entities import Building, Troop
from clasher.towers import CROWN_TOWER_NAMES
from shared.cardRegistry import CardDatabase
from shared.observationSpec import staticUnwalkableMask, timeFeature, towerHpFeature

from .common import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CvObservation,
    N_MAX,
    NUM_HAND_SLOTS,
    world_to_view,
)


SPEC_VERSION = 2
BOARD_CHANNELS = 3
ENTITY_FEATURES = 6
GLOBAL_FEATURE_COUNT = 13 + NUM_HAND_SLOTS + 1

ENTITY_CLASS = 0
ENTITY_OWNER = 1
ENTITY_HP = 2
ENTITY_X = 3
ENTITY_Y = 4
ENTITY_PRESENT = 5

GLOBAL_TIME = 0
GLOBAL_DOUBLE_ELIXIR = 1
GLOBAL_TRIPLE_ELIXIR = 2
GLOBAL_OVERTIME = 3
GLOBAL_OWN_ELIXIR = 4
GLOBAL_OWN_CROWNS = 5
GLOBAL_ENEMY_CROWNS = 6
GLOBAL_OWN_KING_HP = 7
GLOBAL_OWN_LEFT_HP = 8
GLOBAL_OWN_RIGHT_HP = 9
GLOBAL_ENEMY_KING_HP = 10
GLOBAL_ENEMY_LEFT_HP = 11
GLOBAL_ENEMY_RIGHT_HP = 12
GLOBAL_HAND_START = 13
GLOBAL_NEXT_CARD = 17

REGULATION_SECONDS = 180.0


@dataclass(frozen=True)
class ObservationSpec:
    board_channels: int
    entity_count: int
    entity_features: int
    global_features: int
    card_vocab: Sequence[str]


class ObservationBuilder:
    def __init__(self, cards: CardDatabase | None = None, canonical_perspective: bool = True, card_vocab: Sequence[str] | None = None):
        self.cards = cards or CardDatabase()
        if card_vocab is None:
            card_vocab = [
                self.cards.name(card_id)
                for card_id in range(1, self.cards.vocabSize)
                if self.cards.name(card_id) is not None
            ]
        self.card_vocab = list(card_vocab)
        # Kept for API compatibility; observations are always in the player's view frame.
        self.canonical_perspective = canonical_perspective
        self.spec = ObservationSpec(
            board_channels=BOARD_CHANNELS,
            entity_count=N_MAX,
            entity_features=ENTITY_FEATURES,
            global_features=GLOBAL_FEATURE_COUNT,
            card_vocab=self.card_vocab,
        )

    # ------------------------------------------------------------------
    def build(self, battle: BattleState, player_id: int) -> CvObservation:
        units = [e for e in battle.entities.values() if self._is_unit(e)]
        entities = self._build_entities(units, player_id)
        return CvObservation(
            board=self._build_board(battle, units, player_id),
            entities=entities,
            entity_mask=entities[:, ENTITY_PRESENT] > 0.0,
            global_features=self._build_global_features(battle, player_id),
        )

    @staticmethod
    def _is_unit(entity) -> bool:
        if not getattr(entity, "is_alive", False) or not isinstance(entity, (Troop, Building)):
            return False
        name = getattr(getattr(entity, "card_stats", None), "name", None)
        return name is not None and name not in CROWN_TOWER_NAMES

    @staticmethod
    def _view_tile(position: Position, player_id: int):
        vx, vy = world_to_view(position.x, position.y, player_id)
        x, y = int(math.floor(vx)), int(math.floor(vy))
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
            return x, y
        return None

    # ------------------------------------------------------------------
    def _terrain(self, battle: BattleState, player_id: int) -> np.ndarray:
        """Shared static layout (``shared.observationSpec``) with destroyed
        crown towers cleared, exactly as the perception builder does."""
        alive = {}
        arena = battle.arena
        for side, owner in (("own", player_id), ("enemy", 1 - player_id)):
            king_pos = arena.BLUE_KING_TOWER if owner == 0 else arena.RED_KING_TOWER
            alive[(side, "king")] = arena._is_tower_alive(king_pos, owner, battle)
            _, left_hp, right_hp = self._tower_hps(battle, owner, player_id)
            alive[(side, "left")] = left_hp > 0
            alive[(side, "right")] = right_hp > 0
        return staticUnwalkableMask(alive)

    def _build_board(self, battle: BattleState, units, player_id: int) -> np.ndarray:
        board = np.zeros((BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        board[0] = self._terrain(battle, player_id)
        for entity in units:
            tile = self._view_tile(entity.position, player_id)
            if tile is not None:
                x, y = tile
                board[1 if entity.player_id == player_id else 2, y, x] = 1.0
        return board

    def _build_entities(self, units, player_id: int) -> np.ndarray:
        entities = np.zeros((N_MAX, ENTITY_FEATURES), dtype=np.float32)
        rows = []
        for entity in units:
            tile = self._view_tile(entity.position, player_id)
            if tile is None:
                continue
            x, y = tile
            max_hp = float(getattr(entity, "max_hitpoints", 0) or 0)
            hp_fraction = float(entity.hitpoints) / max_hp if max_hp > 0 else 0.0
            owner = 0.0 if entity.player_id == player_id else 1.0
            rows.append((
                float(self.cards.id(entity.card_stats.name)),
                owner,
                float(np.clip(hp_fraction, 0.0, 1.0)),
                x / BOARD_WIDTH,
                y / BOARD_HEIGHT,
                1.0,
            ))
        rows.sort(key=lambda r: (r[ENTITY_OWNER], r[ENTITY_Y], r[ENTITY_X], r[ENTITY_CLASS]))
        for index, row in enumerate(rows[:N_MAX]):
            entities[index] = row
        return entities

    # ------------------------------------------------------------------
    def _build_global_features(self, battle: BattleState, player_id: int) -> np.ndarray:
        features = np.zeros(GLOBAL_FEATURE_COUNT, dtype=np.float32)
        own = battle.players[player_id]
        enemy_id = 1 - player_id

        features[GLOBAL_TIME] = timeFeature(REGULATION_SECONDS - battle.time)
        features[GLOBAL_DOUBLE_ELIXIR] = float(battle.double_elixir)
        features[GLOBAL_TRIPLE_ELIXIR] = float(battle.triple_elixir)
        features[GLOBAL_OVERTIME] = float(battle.overtime)
        features[GLOBAL_OWN_ELIXIR] = float(own.elixir) / 10.0
        features[GLOBAL_OWN_CROWNS] = battle.crowns(player_id) / 3.0
        features[GLOBAL_ENEMY_CROWNS] = battle.crowns(enemy_id) / 3.0

        for base, owner in ((GLOBAL_OWN_KING_HP, player_id), (GLOBAL_ENEMY_KING_HP, enemy_id)):
            king, screen_left, screen_right = self._tower_hps(battle, owner, player_id)
            features[base] = towerHpFeature(king)
            features[base + 1] = towerHpFeature(screen_left)
            features[base + 2] = towerHpFeature(screen_right)

        for slot, card_name in enumerate(own.hand[:NUM_HAND_SLOTS]):
            features[GLOBAL_HAND_START + slot] = self.cards.id(card_name)
        next_card = own.get_next_card()
        if next_card is not None:
            features[GLOBAL_NEXT_CARD] = self.cards.id(next_card)
        return features

    @staticmethod
    def _tower_hps(battle: BattleState, owner: int, viewer: int) -> tuple[float, float, float]:
        """(King, screen-left Princess, screen-right Princess) HP of ``owner``'s
        towers as ``viewer`` sees them (the sim's left/right are world-x based)."""
        arena = battle.arena
        player = battle.players[owner]
        left_pos = arena.BLUE_LEFT_TOWER if owner == 0 else arena.RED_LEFT_TOWER
        right_pos = arena.BLUE_RIGHT_TOWER if owner == 0 else arena.RED_RIGHT_TOWER
        world_left_on_screen_left = world_to_view(left_pos.x, left_pos.y, viewer)[0] < world_to_view(right_pos.x, right_pos.y, viewer)[0]
        if world_left_on_screen_left:
            return player.king_tower_hp, player.left_tower_hp, player.right_tower_hp
        return player.king_tower_hp, player.right_tower_hp, player.left_tower_hp
